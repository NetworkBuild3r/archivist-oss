"""FROZEN deterministic scorer for the toy CRSI domain (zero LLM calls)."""

from __future__ import annotations

import re

from archivist.crsi.evaluator.types import TaskScore, TaskSpec
from archivist.crsi.harness.contract import HarnessResult

_SCORE_RE = re.compile(r"score\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


class ToyScorer:
    """Frozen deterministic scorer enforcing execution proof, exactness, and optimality."""

    def score(self, task: TaskSpec, result: HarnessResult) -> TaskScore:
        kind = str(task.verifier_params.get("kind", "verification"))
        commands_executed = int(result.metrics.get("commands_executed", 0.0))
        executed_clean = any(
            rec.executed_command and rec.command_exit_code == 0 for rec in result.trace
        )

        if not result.completed:
            return TaskScore(
                task_id=task.id,
                score=0.0,
                passed=False,
                summary="Task did not complete within step budget.",
                errors=["incomplete"],
            )

        if commands_executed < 1 or not executed_clean:
            return TaskScore(
                task_id=task.id,
                score=0.0,
                passed=False,
                summary=(
                    "premature_completion: task marked complete without error-free command execution"
                ),
                errors=["premature_completion_no_execution"],
            )

        if kind == "optimization":
            min_score = float(task.verifier_params.get("min_score", 0.90))
            combined = "\n".join([result.final_answer, *(r.output for r in result.trace)])
            matches = _SCORE_RE.findall(combined)
            best_val = max((float(m) for m in matches), default=0.0)
            if best_val >= min_score:
                return TaskScore(
                    task_id=task.id,
                    score=1.0,
                    passed=True,
                    summary=f"Optimal feasible solution verified (score={best_val:.2f} >= {min_score:.2f}).",
                    metrics={"reported_score": best_val},
                )
            if best_val >= 0.50:
                return TaskScore(
                    task_id=task.id,
                    score=0.5,
                    passed=False,
                    summary=(
                        f"feasible_suboptimal: constraints passed with score={best_val:.2f} "
                        f"< target {min_score:.2f}"
                    ),
                    errors=["feasible_suboptimal"],
                    metrics={"reported_score": best_val},
                )
            return TaskScore(
                task_id=task.id,
                score=0.0,
                passed=False,
                summary=f"Optimization solution failed threshold (best score={best_val:.2f}).",
                errors=["optimization_failed"],
            )

        if kind == "v2_normalized":
            expected = str(task.expected).strip()
            actual = result.final_answer.strip()
            verified_in_trace = any(
                expected in rec.output for rec in result.trace if rec.executed_command
            )
            if actual == expected and verified_in_trace:
                return TaskScore(
                    task_id=task.id,
                    score=1.0,
                    passed=True,
                    summary=f"Normalized output verified: {actual}",
                )
            return TaskScore(
                task_id=task.id,
                score=0.0,
                passed=False,
                summary=f"v2_unnormalized: expected normalized output, got {actual!r}",
                errors=["v2_unnormalized"],
            )

        # Default: verification task.
        expected = str(task.expected).strip()
        actual = result.final_answer.strip()
        verified_in_trace = any(
            expected in rec.output for rec in result.trace if rec.executed_command
        )
        if actual == expected and verified_in_trace:
            return TaskScore(
                task_id=task.id,
                score=1.0,
                passed=True,
                summary=f"Verified exact answer {actual}",
            )
        return TaskScore(
            task_id=task.id,
            score=0.0,
            passed=False,
            summary=f"Answer mismatch or unverified execution: got {actual!r}",
            errors=["answer_mismatch"],
        )
