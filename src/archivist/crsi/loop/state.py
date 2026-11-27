"""Atomic checkpoint state (run_state.json) for resumable CRSI runs."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from archivist.crsi.config import MemoryMode
from archivist.crsi.evaluator.types import EvaluationReport

RoundPhase = Literal[
    "baseline",
    "propose",
    "filter",
    "screen",
    "evaluate",
    "select",
    "record",
    "complete",
]

RunStatus = Literal["running", "completed", "stopped_budget", "failed"]


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class RunState(BaseModel):
    """Serializable checkpoint written atomically after every CRSI phase."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    run_branch: str
    base_sha: str
    current_sha: str
    current_round: int = Field(default=0, ge=0)
    phase: RoundPhase = "baseline"
    memory_mode: MemoryMode = "full"
    domain: str = "default"
    policy_model: str = "fake-policy-v1"
    baseline_score: float | None = None
    current_score: float | None = None
    baseline_report: EvaluationReport | None = None
    round_proposals: list[dict[str, Any]] = Field(default_factory=list)
    round_legal_ids: list[str] = Field(default_factory=list)
    round_diffs: dict[str, str] = Field(default_factory=dict)
    round_candidate_shas: dict[str, str] = Field(default_factory=dict)
    round_screened_ids: list[str] = Field(default_factory=list)
    round_eval_reports: dict[str, EvaluationReport] = Field(default_factory=dict)
    round_selected_id: str | None = None
    round_select_reason: str = ""
    round_rejected_by_tolerance: bool = False
    accepted_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    spend_snapshot: dict[str, Any] = Field(default_factory=dict)
    started_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    elapsed_wall_s: float = Field(default=0.0, ge=0.0)
    status: RunStatus = "running"


def save_run_state(path: Path, state: RunState) -> Path:
    """Atomically write ``run_state.json`` via tempfile + fsync + os.replace."""
    state.updated_at = utc_now_iso()
    payload = json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".run_state_", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return path


def load_run_state(path: Path) -> RunState:
    """Load and validate an existing ``run_state.json`` checkpoint."""
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))
