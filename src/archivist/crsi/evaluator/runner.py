"""FROZEN evaluator runner supporting in-process harnesses and external scorer commands."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from archivist.crsi.evaluator.splits import TaskDataset
from archivist.crsi.evaluator.types import (
    EvaluationReport,
    ScorerProtocol,
    SplitName,
    TaskScore,
    TaskSpec,
)
from archivist.crsi.guards.heldout_firewall import HeldOutLeakError
from archivist.crsi.guards.secrets import SecretRedactor, sanitize_env
from archivist.crsi.harness.contract import HarnessProtocol, HarnessResult
from archivist.crsi.llm.budget import BudgetExhaustedError, PaidApiNotEnabledError

_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _sanitize_fs_token(raw: str, *, max_len: int = 40) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", raw).strip("._-")
    return (cleaned or "task")[:max_len]


def sanitize_trace(
    result: HarnessResult,
    *,
    redactor: SecretRedactor | None = None,
    max_turns: int = 12,
    max_chars_per_turn: int = 400,
) -> list[str]:
    """Produce a bounded, secret-scrubbed trace excerpt for evolve feedback ``F_t``."""
    scrub = redactor if redactor is not None else SecretRedactor()
    excerpt: list[str] = []
    for rec in result.trace[:max_turns]:
        snippet = scrub.redact_text(rec.output).replace("\n", " ")[:max_chars_per_turn]
        err_part = f" | err={scrub.redact_text(rec.error)}" if rec.error else ""
        excerpt.append(f"step={rec.step} action={rec.action}: {snippet}{err_part}")
    return excerpt


def _aggregate_report(
    split: SplitName,
    task_scores: list[TaskScore],
    task_order: Sequence[str],
) -> EvaluationReport:
    scores_by_task: dict[str, list[float]] = {tid: [] for tid in task_order}
    passed_by_task: dict[str, list[bool]] = {tid: [] for tid in task_order}
    all_errors: list[str] = []

    for ts in task_scores:
        scores_by_task.setdefault(ts.task_id, []).append(ts.score)
        passed_by_task.setdefault(ts.task_id, []).append(ts.passed)
        all_errors.extend(ts.errors)

    by_task_score: dict[str, float] = {}
    by_task_passed: dict[str, bool] = {}
    for tid in task_order:
        vals = scores_by_task.get(tid) or [0.0]
        pvals = passed_by_task.get(tid) or [False]
        by_task_score[tid] = round(sum(vals) / len(vals), 6)
        by_task_passed[tid] = all(pvals)

    total = len(task_order)
    passed = sum(1 for ok in by_task_passed.values() if ok)
    mean_score = round(sum(by_task_score.values()) / total, 6) if total > 0 else 0.0
    return EvaluationReport(
        split=split,
        mean_score=max(0.0, min(1.0, mean_score)),
        passed=passed,
        total=total,
        task_scores=task_scores,
        by_task_passed=by_task_passed,
        by_task_score=by_task_score,
        errors=all_errors,
    )


class FrozenEvaluator:
    """Isolated evaluator running either in-process harnesses or external scorer commands.

    Invariants:
    - Hands the harness ONLY ``TaskSpec.to_task_input()`` (never expected answers).
    - Bounds concurrency via ``parallelism`` and per-task duration via ``task_timeout_s``.
    - Non-zero exits, timeouts, or invalid external ``result.json`` score ``0.0``.
    - Held-out evaluation writes strictly to private human-facing report files.
    """

    def __init__(
        self,
        *,
        harness_factory: Callable[[], HarnessProtocol] | None = None,
        scorer: ScorerProtocol | None = None,
        run_task_argv: Sequence[str] | None = None,
        parallelism: int = 2,
        task_timeout_s: float = 60.0,
        redactor: SecretRedactor | None = None,
    ) -> None:
        has_inproc = harness_factory is not None and scorer is not None
        has_cmd = bool(run_task_argv)
        if not has_inproc and not has_cmd:
            raise ValueError(
                "FrozenEvaluator requires either (harness_factory + scorer) or run_task_argv."
            )
        self._harness_factory = harness_factory
        self._scorer = scorer
        self._run_task_argv: list[str] = list(run_task_argv) if run_task_argv else []
        self.parallelism = max(1, parallelism)
        self.task_timeout_s = max(0.05, task_timeout_s)
        self.redactor = redactor if redactor is not None else SecretRedactor()

    async def _run_single_inproc(
        self,
        task: TaskSpec,
        rollout: int,
        semaphore: asyncio.Semaphore,
    ) -> TaskScore:
        assert self._harness_factory is not None
        assert self._scorer is not None
        async with semaphore:
            harness = self._harness_factory()
            sanitized_input = task.to_task_input()
            try:
                result = await asyncio.wait_for(
                    harness.run(sanitized_input),
                    timeout=self.task_timeout_s,
                )
            except TimeoutError:
                return TaskScore(
                    task_id=task.id,
                    rollout=rollout,
                    score=0.0,
                    passed=False,
                    summary=f"Task timed out after {self.task_timeout_s:.2f}s",
                    errors=[f"timeout:{task.id}:rollout={rollout}"],
                )
            except (BudgetExhaustedError, PaidApiNotEnabledError, HeldOutLeakError):
                raise
            except Exception as exc:
                safe_err = self.redactor.redact_text(str(exc))
                return TaskScore(
                    task_id=task.id,
                    rollout=rollout,
                    score=0.0,
                    passed=False,
                    summary=f"Harness raised exception: {safe_err}",
                    errors=[safe_err],
                )

            scored = self._scorer.score(task, result)
            return scored.model_copy(
                update={
                    "rollout": rollout,
                    "summary": self.redactor.redact_text(scored.summary),
                    "errors": [self.redactor.redact_text(e) for e in scored.errors],
                    "trace_excerpt": (
                        [self.redactor.redact_text(line) for line in scored.trace_excerpt]
                        if scored.trace_excerpt
                        else sanitize_trace(result, redactor=self.redactor)
                    ),
                }
            )

    def _run_single_command(
        self,
        task: TaskSpec,
        rollout: int,
        checkout_dir: Path,
    ) -> TaskScore:
        safe_task_prefix = _sanitize_fs_token(task.id)
        with tempfile.TemporaryDirectory(prefix=f"crsi_eval_{safe_task_prefix}_") as tmp_str:
            tmp_dir = Path(tmp_str)
            task_file = tmp_dir / "task.json"
            out_dir = tmp_dir / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            task_file.write_text(
                json.dumps(task.model_dump(mode="json"), sort_keys=True),
                encoding="utf-8",
            )

            mapping = {
                "task_file": str(task_file),
                "task_id": task.id,
                "checkout": str(checkout_dir.resolve()),
                "out": str(out_dir),
                "rollout": str(rollout),
            }
            rendered_argv = [arg.format_map(mapping) for arg in self._run_task_argv]
            exe_candidate = Path(rendered_argv[0])
            resolved_exe = (
                str(exe_candidate)
                if exe_candidate.is_absolute() and exe_candidate.is_file()
                else shutil.which(rendered_argv[0])
            )
            if resolved_exe is None:
                return TaskScore(
                    task_id=task.id,
                    rollout=rollout,
                    score=0.0,
                    passed=False,
                    summary=f"Scorer executable not found: {rendered_argv[0]}",
                    errors=[f"executable_not_found:{rendered_argv[0]}"],
                )

            cmd = [resolved_exe, *rendered_argv[1:]]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(checkout_dir.resolve()),
                    env=sanitize_env(),
                    capture_output=True,
                    text=True,
                    timeout=self.task_timeout_s,
                    check=False,
                    shell=False,
                )
            except subprocess.TimeoutExpired:
                return TaskScore(
                    task_id=task.id,
                    rollout=rollout,
                    score=0.0,
                    passed=False,
                    summary=f"External scorer timed out after {self.task_timeout_s:.2f}s",
                    errors=[f"timeout:{task.id}:rollout={rollout}"],
                )

            if proc.returncode != 0:
                safe_stderr = self.redactor.redact_text((proc.stderr or "")[:400])
                return TaskScore(
                    task_id=task.id,
                    rollout=rollout,
                    score=0.0,
                    passed=False,
                    summary=f"External scorer exited {proc.returncode}: {safe_stderr}",
                    errors=[f"nonzero_exit:{proc.returncode}"],
                )

            result_path = out_dir / "result.json"
            if not result_path.is_file():
                return TaskScore(
                    task_id=task.id,
                    rollout=rollout,
                    score=0.0,
                    passed=False,
                    summary="External scorer did not write result.json",
                    errors=["missing_result_json"],
                )

            try:
                raw: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
                score_val = float(raw["score"])
                if not (0.0 <= score_val <= 1.0):
                    raise ValueError(f"score out of [0,1]: {score_val}")
                passed_val = bool(raw["passed"])
                summary_val = self.redactor.redact_text(str(raw.get("summary", "")))
                metrics_raw = raw.get("metrics") or {}
                metrics_val = {str(k): float(v) for k, v in metrics_raw.items()}
                errors_raw = raw.get("errors") or []
                errors_val = [self.redactor.redact_text(str(e)) for e in errors_raw]
                trace_raw = raw.get("trace_excerpt") or []
                trace_val = [self.redactor.redact_text(str(t)) for t in trace_raw]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return TaskScore(
                    task_id=task.id,
                    rollout=rollout,
                    score=0.0,
                    passed=False,
                    summary=f"Invalid result.json: {exc}",
                    errors=["invalid_result_json"],
                )

            return TaskScore(
                task_id=task.id,
                rollout=rollout,
                score=score_val,
                passed=passed_val,
                summary=summary_val,
                errors=errors_val,
                trace_excerpt=trace_val,
                metrics=metrics_val,
            )

    async def _evaluate_tasks(
        self,
        *,
        split: SplitName,
        tasks: Sequence[TaskSpec],
        rollouts: int = 1,
        checkout_dir: Path | None = None,
    ) -> EvaluationReport:
        rollouts_count = max(1, rollouts)
        work_items = [(task, r) for task in tasks for r in range(rollouts_count)]
        task_order = [t.id for t in tasks]

        if self._run_task_argv:
            target_checkout = checkout_dir if checkout_dir is not None else Path.cwd()
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=self.parallelism) as pool:
                futures = [
                    loop.run_in_executor(
                        pool,
                        self._run_single_command,
                        task,
                        r,
                        target_checkout,
                    )
                    for task, r in work_items
                ]
                scores = list(await asyncio.gather(*futures))
        else:
            sem = asyncio.Semaphore(self.parallelism)
            scores = list(
                await asyncio.gather(
                    *(self._run_single_inproc(task, r, sem) for task, r in work_items)
                )
            )

        return _aggregate_report(split=split, task_scores=scores, task_order=task_order)

    async def evaluate_evolve(
        self,
        dataset: TaskDataset,
        *,
        rollouts: int = 1,
        checkout_dir: Path | None = None,
    ) -> EvaluationReport:
        """Evaluate strictly on the ``evolve`` split for in-loop CRSI feedback ``F_t``."""
        return await self._evaluate_tasks(
            split="evolve",
            tasks=dataset.evolve_tasks(),
            rollouts=rollouts,
            checkout_dir=checkout_dir,
        )

    async def evaluate_heldout(
        self,
        dataset: TaskDataset,
        *,
        report_dir: Path,
        run_id: str,
        rollouts: int = 1,
        checkout_dir: Path | None = None,
    ) -> tuple[EvaluationReport, Path]:
        """Run out-of-loop held-out evaluation and write atomic human-only report JSON.

        Never returns evolve feedback and never mutates ``F_t``, ``D_t``, or SIM.
        """
        if not _SAFE_RUN_ID_RE.match(run_id) or ".." in run_id:
            raise ValueError(
                f"Invalid run_id {run_id!r}: must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$ "
                "and contain no path traversal."
            )
        report = await self._evaluate_tasks(
            split="heldout",
            tasks=dataset.heldout_tasks(),
            rollouts=rollouts,
            checkout_dir=checkout_dir,
        )
        resolved_dir = report_dir.resolve()
        resolved_dir.mkdir(parents=True, exist_ok=True)
        out_path = (resolved_dir / f"heldout_{run_id}.json").resolve()
        if not out_path.is_relative_to(resolved_dir):
            raise ValueError(f"Held-out report path escapes report_dir: {run_id!r}")
        payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        fd, tmp_name = tempfile.mkstemp(dir=str(report_dir), prefix=".heldout_", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, out_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
        return report, out_path
