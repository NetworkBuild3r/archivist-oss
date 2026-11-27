"""Pydantic v2 data models for Self-Improvement Memory (SIM) — Yu et al., Appendix D.3."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROBLEM_ID_RE = re.compile(r"^P([1-9][0-9]*)$")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_problem_number(problem_id: str) -> int:
    """Parse numeric suffix of a canonical problem ID such as ``'P5' -> 5``."""
    match = PROBLEM_ID_RE.match(problem_id.strip())
    if not match:
        raise ValueError(
            f"Invalid SIM problem id {problem_id!r}: expected format 'P<positive_int>'"
        )
    return int(match.group(1))


class CarriedFrom(BaseModel):
    """Provenance metadata stamped when SIM is carried across models or task domains."""

    model_config = ConfigDict(extra="forbid")

    arm: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    model: str = ""
    into: str = Field(min_length=1)
    carried_at: str = Field(default_factory=_utc_now_iso)


class ObservedOutcome(BaseModel):
    """Post-evaluation observed task outcomes compared against pre-eval predictions."""

    model_config = ConfigDict(extra="forbid")

    fixed: list[str] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)
    regressed: list[str] = Field(default_factory=list)
    passed: int = Field(ge=0)
    total: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_counts(self) -> ObservedOutcome:
        if self.passed > self.total:
            raise ValueError(f"observed.passed ({self.passed}) cannot exceed total ({self.total})")
        return self

    @classmethod
    def from_task_results(
        cls,
        *,
        predicted: Sequence[str],
        before_passed: Mapping[str, bool],
        after_passed: Mapping[str, bool],
        after_scores: Mapping[str, float] | None = None,
    ) -> ObservedOutcome:
        """Compute predicted-vs-observed outcome deltas from before/after evolve evaluations."""
        all_ids = sorted(set(before_passed.keys()) | set(after_passed.keys()))
        fixed = [
            tid
            for tid in all_ids
            if not before_passed.get(tid, False) and after_passed.get(tid, False)
        ]
        regressed = [
            tid
            for tid in all_ids
            if before_passed.get(tid, False) and not after_passed.get(tid, False)
        ]
        missed: list[str] = []
        seen_missed: set[str] = set()
        for tid in predicted:
            if not after_passed.get(tid, False) and tid not in seen_missed:
                missed.append(tid)
                seen_missed.add(tid)

        total = len(after_passed)
        passed = sum(1 for ok in after_passed.values() if ok)
        if after_scores:
            score = sum(after_scores.values()) / len(after_scores)
        else:
            score = (passed / total) if total > 0 else 0.0
        score = max(0.0, min(1.0, round(score, 6)))
        return cls(
            fixed=fixed,
            missed=missed,
            regressed=regressed,
            passed=passed,
            total=total,
            score=score,
        )


class AttemptRecord(BaseModel):
    """Single optimization attempt recorded under a root-cause SIM problem entry."""

    model_config = ConfigDict(extra="forbid")

    round: int = Field(ge=0)
    sample: str = Field(min_length=1)
    accepted: bool
    backfilled: bool = False
    what: str = Field(min_length=1)
    files: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    predicted: list[str] = Field(default_factory=list)
    observed: ObservedOutcome
    model: str = Field(min_length=1)
    domain: str = Field(min_length=1)


class ProblemEntry(BaseModel):
    """Root-cause problem bucket grouping chronological improvement attempts."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = Field(min_length=1)
    opened_round: int = Field(ge=0)
    attempts: list[AttemptRecord] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        parse_problem_number(value)
        return value.strip()

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("problem name must be non-empty")
        return cleaned


class SimDocument(BaseModel):
    """Canonical top-level JSON schema for ``.crsi/sim.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    next_id: int = Field(default=1, ge=1)
    carried_from: CarriedFrom | None = None
    aliases: dict[str, str] = Field(default_factory=dict)
    problems: list[ProblemEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_invariants(self) -> SimDocument:
        seen_ids: set[str] = set()
        max_num = 0
        for entry in self.problems:
            if entry.id in seen_ids:
                raise ValueError(f"Duplicate SIM problem id: {entry.id!r}")
            seen_ids.add(entry.id)
            max_num = max(max_num, parse_problem_number(entry.id))

        for src, dst in self.aliases.items():
            max_num = max(max_num, parse_problem_number(src), parse_problem_number(dst))
            if src == dst:
                raise ValueError(f"Self-referential SIM alias is invalid: {src!r} -> {dst!r}")

        if self.next_id <= max_num:
            raise ValueError(
                f"next_id ({self.next_id}) must exceed highest problem/alias number ({max_num})"
            )

        # Verify no cycles exist in alias mappings.
        for start in self.aliases:
            visited: set[str] = {start}
            cur = self.aliases[start]
            while cur in self.aliases:
                if cur in visited:
                    raise ValueError(f"Cycle detected in SIM aliases starting at {start!r}")
                visited.add(cur)
                cur = self.aliases[cur]

        return self


# Friendly short aliases matching requirements brief terminology.
Attempt = AttemptRecord
Entry = ProblemEntry
