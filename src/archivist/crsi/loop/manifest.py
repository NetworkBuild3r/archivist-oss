"""Pre-evaluation ChangeManifest and CandidateProposal schemas for CRSI."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from archivist.crsi.sim.models import parse_problem_number


class ChangeManifest(BaseModel):
    """Structured pre-evaluation diagnosis and prediction required on every candidate."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str | None = None
    new_problem_name: str | None = None
    what: str = Field(min_length=1)
    files: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    predicted: list[str] = Field(min_length=1)
    interlock_removal_sim_evidence: str | None = None

    @field_validator("problem_id", "interlock_removal_sim_evidence")
    @classmethod
    def _validate_optional_problem_id(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        parse_problem_number(value)
        return value.strip()

    @field_validator("new_problem_name")
    @classmethod
    def _normalize_optional_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned if cleaned else None

    @field_validator("files", "predicted")
    @classmethod
    def _validate_nonempty_items(cls, items: list[str]) -> list[str]:
        cleaned = [item.strip() for item in items if item and item.strip()]
        if not cleaned:
            raise ValueError("list must contain at least one non-empty item")
        return cleaned

    @model_validator(mode="after")
    def _validate_single_problem_ref(self) -> ChangeManifest:
        has_id = self.problem_id is not None
        has_name = self.new_problem_name is not None
        if has_id == has_name:
            raise ValueError(
                "ChangeManifest must specify exactly one of 'problem_id' or 'new_problem_name'."
            )
        return self

    def problem_ref_label(self) -> str:
        """Return cited problem ID or new problem name for logging in run history."""
        return self.problem_id if self.problem_id is not None else (self.new_problem_name or "")


class CandidateProposal(BaseModel):
    """Single candidate edit emitted by the Proposer prior to legal filtering."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    manifest: ChangeManifest
    file_edits: dict[str, str] = Field(default_factory=dict)
    delete_files: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_has_changes(self) -> CandidateProposal:
        if not self.file_edits and not self.delete_files:
            raise ValueError("CandidateProposal must include at least one file edit or deletion.")
        return self
