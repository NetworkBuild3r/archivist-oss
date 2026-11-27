"""FROZEN held-out isolation firewall preventing held-out leakage to Proposer/Selector.

Scans every Proposer and Selector prompt payload, context path list, and feedback
directory before any LLM call, raising ``HeldOutLeakError`` if any held-out task ID,
prompt, answer token, score marker, or private held-out path is detected.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from archivist.crsi.evaluator.splits import TaskDataset
    from archivist.crsi.evaluator.types import EvaluationReport
    from archivist.crsi.llm.base import LLMRequest, Message

_DEFAULT_HELDOUT_PATH_MARKERS: tuple[str, ...] = (
    "tasks_heldout.jsonl",
    "heldout_reports",
    "tasks.jsonl",
    "split.json",
)


class HeldOutLeakError(RuntimeError):
    """Raised when held-out task data, answers, scores, or paths enter P/S context."""


def _extract_answer_tokens(expected: Any, *, min_len: int = 4) -> list[str]:
    """Extract distinctive string tokens from a task's ``expected`` answer payload."""
    tokens: list[str] = []
    if expected is None:
        return tokens
    if isinstance(expected, str):
        stripped = expected.strip()
        if len(stripped) >= min_len:
            tokens.append(stripped)
        return tokens
    if isinstance(expected, dict):
        for val in expected.values():
            tokens.extend(_extract_answer_tokens(val, min_len=min_len))
        return tokens
    if isinstance(expected, (list, tuple)):
        for item in expected:
            tokens.extend(_extract_answer_tokens(item, min_len=min_len))
        return tokens
    serialized = json.dumps(expected, sort_keys=True)
    if len(serialized) >= min_len:
        tokens.append(serialized)
    return tokens


@dataclass(frozen=True)
class HeldOutFingerprint:
    """Immutable signature set representing held-out tasks, answers, and storage paths."""

    heldout_task_ids: frozenset[str]
    heldout_prompts: frozenset[str]
    heldout_answer_tokens: frozenset[str]
    heldout_paths: frozenset[str]
    extra_forbidden_tokens: frozenset[str] = frozenset()

    @classmethod
    def from_dataset(
        cls,
        dataset: TaskDataset,
        *,
        private_dir: Path | None = None,
        extra_forbidden_tokens: Sequence[str] = (),
    ) -> HeldOutFingerprint:
        """Build a firewall fingerprint from ``dataset.heldout_tasks()``."""
        evolve_corpus_parts: list[str] = []
        evolve_expected_tokens: set[str] = set()
        for evo_task in dataset.evolve_tasks():
            evolve_corpus_parts.append(evo_task.prompt)
            for tok in _extract_answer_tokens(evo_task.expected):
                evolve_expected_tokens.add(tok)
        evolve_corpus = "\n".join(evolve_corpus_parts)

        heldout_ids: set[str] = set()
        prompts: set[str] = set()
        answer_tokens: set[str] = set()
        for task in dataset.heldout_tasks():
            heldout_ids.add(task.id)
            stripped_prompt = task.prompt.strip()
            if len(stripped_prompt) >= 8:
                prompts.add(stripped_prompt)
            for tok in _extract_answer_tokens(task.expected):
                # Exclude shared domain literals that legitimately appear in evolve tasks.
                if tok not in evolve_expected_tokens and tok not in evolve_corpus:
                    answer_tokens.add(tok)

        paths: set[str] = set(_DEFAULT_HELDOUT_PATH_MARKERS)
        if private_dir is not None:
            resolved = private_dir.resolve()
            paths.add(str(resolved))
            paths.add(str(resolved / "tasks_heldout.jsonl"))
            paths.add(str(resolved / "heldout_reports"))
            paths.add(str(resolved / "tasks.jsonl"))
            paths.add(str(resolved / "split.json"))

        extras = frozenset(tok.strip() for tok in extra_forbidden_tokens if tok.strip())
        return cls(
            heldout_task_ids=frozenset(heldout_ids),
            heldout_prompts=frozenset(prompts),
            heldout_answer_tokens=frozenset(answer_tokens),
            heldout_paths=frozenset(paths),
            extra_forbidden_tokens=extras,
        )


def _contains_token_boundary(haystack: str, needle: str) -> bool:
    pattern = re.compile(rf"(?<![A-Za-z0-9_\-]){re.escape(needle)}(?![A-Za-z0-9_\-])")
    return pattern.search(haystack) is not None


class HeldOutIsolationGuard:
    """Firewall guaranteeing zero held-out leakage into Proposer/Selector/SIM."""

    def __init__(
        self,
        fingerprint: HeldOutFingerprint,
        *,
        evolve_ids: frozenset[str] = frozenset(),
        private_dir: Path | None = None,
    ) -> None:
        overlap = sorted(evolve_ids & fingerprint.heldout_task_ids)
        if overlap:
            raise ValueError(
                f"Cannot construct HeldOutIsolationGuard: evolve/heldout overlap {overlap}"
            )
        self.fingerprint = fingerprint
        self.evolve_ids = evolve_ids
        self.private_dir = private_dir.resolve() if private_dir is not None else None

    @classmethod
    def from_dataset(
        cls,
        dataset: TaskDataset,
        *,
        private_dir: Path | None = None,
        extra_forbidden_tokens: Sequence[str] = (),
    ) -> HeldOutIsolationGuard:
        """Create a guard directly from a validated ``TaskDataset``."""
        fp = HeldOutFingerprint.from_dataset(
            dataset,
            private_dir=private_dir,
            extra_forbidden_tokens=extra_forbidden_tokens,
        )
        return cls(fp, evolve_ids=dataset.evolve_ids(), private_dir=private_dir)

    def assert_evolve_only_report(self, report: EvaluationReport) -> None:
        """Verify an ``EvaluationReport`` contains strictly evolve-split results."""
        if report.split != "evolve":
            raise HeldOutLeakError(
                f"Held-out evaluation report (split={report.split!r}) cannot enter CRSI loop."
            )
        for ts in report.task_scores:
            if ts.task_id in self.fingerprint.heldout_task_ids:
                raise HeldOutLeakError(
                    "Held-out task result detected inside evolve EvaluationReport."
                )

    def _scan_text(self, text: str) -> list[str]:
        """Return leak categories detected in ``text`` (never returns raw held-out answers)."""
        findings: list[str] = []
        if not text:
            return findings

        for tid in self.fingerprint.heldout_task_ids:
            if _contains_token_boundary(text, tid):
                findings.append("heldout_task_id")
                break

        for prompt in self.fingerprint.heldout_prompts:
            if prompt in text:
                findings.append("heldout_prompt")
                break

        for ans in self.fingerprint.heldout_answer_tokens:
            if ans in text:
                findings.append("heldout_expected_answer")
                break

        for path_marker in self.fingerprint.heldout_paths:
            if path_marker in text:
                findings.append("heldout_path_reference")
                break

        for extra in self.fingerprint.extra_forbidden_tokens:
            if extra in text:
                findings.append("heldout_score_or_marker")
                break

        return findings

    def _scan_paths(self, context_paths: Sequence[str | Path]) -> list[str]:
        findings: list[str] = []
        for raw in context_paths:
            raw_str = str(raw)
            norm_posix = raw_str.replace("\\", "/")
            for marker in self.fingerprint.heldout_paths:
                if marker in norm_posix:
                    findings.append("heldout_context_path")
                    return findings
            if self.private_dir is not None:
                try:
                    resolved = Path(raw).resolve()
                except OSError:
                    resolved = None
                if resolved is not None and resolved.is_relative_to(self.private_dir):
                    findings.append("private_dir_context_path")
                    return findings
        return findings

    def assert_no_leak(
        self,
        *,
        prompt_text: str = "",
        messages: Sequence[Message] = (),
        context_paths: Sequence[str | Path] = (),
        feedback_dir: Path | None = None,
        role: str = "proposer/selector",
    ) -> None:
        """Scan prompts, messages, context paths, and feedback files; raise on any leak."""
        findings: list[str] = []

        combined_parts: list[str] = []
        if prompt_text:
            combined_parts.append(prompt_text)
        for msg in messages:
            combined_parts.append(msg.content)
        combined_text = "\n".join(combined_parts)
        findings.extend(self._scan_text(combined_text))
        findings.extend(self._scan_paths(context_paths))

        if feedback_dir is not None and feedback_dir.exists():
            findings.extend(self._scan_paths([feedback_dir]))
            for child in sorted(feedback_dir.rglob("*")):
                findings.extend(self._scan_paths([child]))
                if child.is_file():
                    try:
                        text = child.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    findings.extend(self._scan_text(text))

        if findings:
            unique = ", ".join(sorted(set(findings)))
            raise HeldOutLeakError(
                f"Blocked {role} call: held-out isolation firewall detected leakage ({unique})."
            )

    def assert_request_clean(
        self,
        request: LLMRequest,
        *,
        context_paths: Sequence[str | Path] = (),
        feedback_dir: Path | None = None,
    ) -> LLMRequest:
        """Validate a Proposer or Selector ``LLMRequest`` before dispatching to an LLM."""
        self.assert_no_leak(
            messages=request.messages,
            context_paths=context_paths,
            feedback_dir=feedback_dir,
            role=request.role,
        )
        return request
