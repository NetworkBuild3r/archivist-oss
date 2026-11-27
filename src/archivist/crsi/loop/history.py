"""Run history D_t persisting accepted and rejected candidates with drop reasons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from archivist.crsi.guards.secrets import SecretRedactor

DropReason = Literal[
    "illegal_path",
    "invalid_manifest",
    "secret_leak",
    "check_failed",
    "interlock_guard",
    "screen_dropped",
    "below_tolerance",
    "not_selected",
    "apply_failed",
]


class HistoryRecord(BaseModel):
    """Single candidate outcome stored in run history ``D_t``."""

    model_config = ConfigDict(extra="forbid")

    round: int = Field(ge=0)
    candidate_id: str = Field(min_length=1)
    status: Literal["accepted", "rejected"]
    drop_reason: DropReason | None = None
    detail: str = ""
    problem_ref: str = ""
    what: str = ""
    files: list[str] = Field(default_factory=list)
    predicted: list[str] = Field(default_factory=list)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_score: float | None = Field(default=None, ge=0.0, le=1.0)
    commit_sha: str | None = None


class RunHistory:
    """Append-only JSONL history ``D_t`` retaining both accepted and rejected candidates."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        records: list[HistoryRecord] | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self.path = path
        self.redactor = redactor if redactor is not None else SecretRedactor()
        self.records: list[HistoryRecord] = list(records) if records is not None else []

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        *,
        redactor: SecretRedactor | None = None,
    ) -> RunHistory:
        """Load existing ``history.jsonl`` if present, else start an empty history."""
        records: list[HistoryRecord] = []
        if path.is_file():
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                records.append(HistoryRecord.model_validate_json(line))
        return cls(path=path, records=records, redactor=redactor)

    def append(self, record: HistoryRecord) -> HistoryRecord:
        """Scrub secrets, append ``record`` in memory, and persist to JSONL if configured."""
        cleaned = record.model_copy(
            update={
                "detail": self.redactor.redact_text(record.detail),
                "what": self.redactor.redact_text(record.what),
                "problem_ref": self.redactor.redact_text(record.problem_ref),
            }
        )
        self.records.append(cleaned)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(cleaned.model_dump(mode="json"), sort_keys=True) + "\n"
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        return cleaned

    def recent(self, limit: int = 20) -> list[HistoryRecord]:
        """Return up to ``limit`` most recent records."""
        if limit <= 0:
            return []
        return self.records[-limit:]

    def format_for_prompt(self, *, limit: int = 20) -> str:
        """Render bounded recent ``D_t`` showing accepted and rejected candidates."""
        items = self.recent(limit)
        if not items:
            return "## Run History (D_t)\n(no prior candidates recorded)"
        lines: list[str] = ["## Run History (D_t)"]
        for rec in items:
            score_str = f"score={rec.score:.4f}" if rec.score is not None else "score=N/A"
            reason_str = f" reason={rec.drop_reason}" if rec.drop_reason else ""
            files_str = ", ".join(rec.files) if rec.files else "(none)"
            lines.append(
                f"- round={rec.round} cand={rec.candidate_id} status={rec.status}{reason_str} "
                f"{score_str} problem={rec.problem_ref or '(unspecified)'} files=[{files_str}] "
                f"what={rec.what!r} detail={rec.detail!r}"
            )
        return "\n".join(lines)
