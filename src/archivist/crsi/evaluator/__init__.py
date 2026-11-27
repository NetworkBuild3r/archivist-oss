"""Frozen evaluator types, disjoint split manager, and isolated rollout runner."""

from __future__ import annotations

from archivist.crsi.evaluator.runner import FrozenEvaluator, sanitize_trace
from archivist.crsi.evaluator.splits import (
    SplitManifest,
    SplitViolationError,
    TaskDataset,
    build_deterministic_split,
    load_private_dataset,
    load_tasks_jsonl,
    save_tasks_jsonl,
)
from archivist.crsi.evaluator.types import (
    EvaluationReport,
    ScorerProtocol,
    SplitName,
    TaskScore,
    TaskSpec,
)

__all__ = [
    "EvaluationReport",
    "FrozenEvaluator",
    "ScorerProtocol",
    "SplitManifest",
    "SplitName",
    "SplitViolationError",
    "TaskDataset",
    "TaskScore",
    "TaskSpec",
    "build_deterministic_split",
    "load_private_dataset",
    "load_tasks_jsonl",
    "sanitize_trace",
    "save_tasks_jsonl",
]
