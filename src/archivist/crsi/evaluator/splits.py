"""FROZEN private task loader and deterministic disjoint evolve/held-out split manager."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from archivist.crsi.evaluator.types import TaskSpec


class SplitViolationError(ValueError):
    """Raised when evolve and held-out splits overlap, are empty, or fail validation."""


class SplitManifest(BaseModel):
    """Canonical ``split.json`` partitioning task IDs into strictly disjoint sets."""

    model_config = ConfigDict(extra="forbid")

    evolve: list[str] = Field(min_length=1)
    heldout: list[str] = Field(default_factory=list)
    seed: int | None = None

    @model_validator(mode="after")
    def _validate_disjoint(self) -> SplitManifest:
        if len(set(self.evolve)) != len(self.evolve):
            raise ValueError("Duplicate task IDs detected in evolve split.")
        if len(set(self.heldout)) != len(self.heldout):
            raise ValueError("Duplicate task IDs detected in heldout split.")
        overlap = sorted(set(self.evolve) & set(self.heldout))
        if overlap:
            raise ValueError(
                f"Evolve and heldout splits MUST be strictly disjoint; overlap: {overlap}"
            )
        return self


@dataclass(frozen=True)
class TaskDataset:
    """In-memory private dataset holding validated tasks and their disjoint split."""

    tasks: dict[str, TaskSpec]
    split: SplitManifest

    def evolve_tasks(self) -> list[TaskSpec]:
        return [self.tasks[tid] for tid in self.split.evolve]

    def heldout_tasks(self) -> list[TaskSpec]:
        return [self.tasks[tid] for tid in self.split.heldout]

    def evolve_ids(self) -> frozenset[str]:
        return frozenset(self.split.evolve)

    def heldout_ids(self) -> frozenset[str]:
        return frozenset(self.split.heldout)


def load_tasks_jsonl(path: Path) -> list[TaskSpec]:
    """Load private ``TaskSpec`` records from a JSONL file, rejecting duplicate IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Task JSONL file not found: {path}")
    tasks: list[TaskSpec] = []
    seen: set[str] = set()
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SplitViolationError(f"Invalid JSON at {path}:{lineno}: {exc}") from None
        spec = TaskSpec.model_validate(data)
        if spec.id in seen:
            raise SplitViolationError(f"Duplicate task id {spec.id!r} in {path}:{lineno}")
        seen.add(spec.id)
        tasks.append(spec)
    return tasks


def save_tasks_jsonl(path: Path, tasks: Sequence[TaskSpec]) -> Path:
    """Persist ``TaskSpec`` records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(t.model_dump(mode="json"), sort_keys=True) for t in tasks]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def build_deterministic_split(
    tasks: Sequence[TaskSpec],
    *,
    heldout_fraction: float = 0.3,
    seed: int = 42,
) -> SplitManifest:
    """Partition ``tasks`` deterministically into strictly disjoint evolve/heldout splits."""
    if not tasks:
        raise SplitViolationError("Cannot build a split from an empty task list.")
    if not (0.0 <= heldout_fraction < 1.0):
        raise SplitViolationError("heldout_fraction must be in [0.0, 1.0).")

    ordered = sorted(
        tasks,
        key=lambda t: hashlib.sha256(f"{seed}:{t.id}".encode()).hexdigest(),
    )
    if heldout_fraction == 0.0 or len(ordered) == 1:
        heldout_count = 0
    else:
        heldout_count = max(1, min(len(ordered) - 1, round(len(ordered) * heldout_fraction)))

    heldout = [t.id for t in ordered[:heldout_count]]
    evolve = [t.id for t in ordered[heldout_count:]]
    return SplitManifest(evolve=evolve, heldout=heldout, seed=seed)


def load_private_dataset(private_dir: Path) -> TaskDataset:
    """Load and validate evolve + held-out tasks from a private directory outside git.

    Supports either:
    1. ``tasks_evolve.jsonl`` + optional ``tasks_heldout.jsonl`` (with optional ``split.json``)
    2. ``tasks.jsonl`` + ``split.json``
    """
    evolve_file = private_dir / "tasks_evolve.jsonl"
    heldout_file = private_dir / "tasks_heldout.jsonl"
    combined_file = private_dir / "tasks.jsonl"
    split_file = private_dir / "split.json"

    if evolve_file.is_file():
        evolve_list = load_tasks_jsonl(evolve_file)
        heldout_list = load_tasks_jsonl(heldout_file) if heldout_file.is_file() else []

        evolve_ids = [t.id for t in evolve_list]
        heldout_ids = [t.id for t in heldout_list]
        overlap = sorted(set(evolve_ids) & set(heldout_ids))
        if overlap:
            raise SplitViolationError(
                f"Overlapping task IDs between tasks_evolve.jsonl and tasks_heldout.jsonl: "
                f"{overlap}"
            )
        if not evolve_ids:
            raise SplitViolationError(f"No evolve tasks found in {evolve_file}")

        tasks_map = {t.id: t for t in (*evolve_list, *heldout_list)}
        if split_file.is_file():
            manifest = SplitManifest.model_validate_json(split_file.read_text(encoding="utf-8"))
            if set(manifest.evolve) != set(evolve_ids) or set(manifest.heldout) != set(heldout_ids):
                raise SplitViolationError(
                    "split.json disagrees with tasks_evolve.jsonl / tasks_heldout.jsonl"
                )
        else:
            manifest = SplitManifest(evolve=evolve_ids, heldout=heldout_ids)
        return TaskDataset(tasks=tasks_map, split=manifest)

    if combined_file.is_file() and split_file.is_file():
        all_tasks = load_tasks_jsonl(combined_file)
        tasks_map = {t.id: t for t in all_tasks}
        try:
            manifest = SplitManifest.model_validate_json(split_file.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise SplitViolationError(f"Invalid split.json: {exc}") from None

        missing = [tid for tid in (*manifest.evolve, *manifest.heldout) if tid not in tasks_map]
        if missing:
            raise SplitViolationError(
                f"split.json references task IDs not present in tasks.jsonl: {missing}"
            )
        return TaskDataset(tasks=tasks_map, split=manifest)

    raise SplitViolationError(
        f"Private task directory {private_dir} must contain either tasks_evolve.jsonl "
        "(+ optional tasks_heldout.jsonl) or tasks.jsonl + split.json."
    )
