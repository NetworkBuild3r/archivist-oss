"""FROZEN evaluator data types (TaskSpec, TaskScore, EvaluationReport, ScorerProtocol)."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from archivist.crsi.harness.contract import HarnessResult, TaskInput

SplitName = Literal["evolve", "heldout"]


class TaskSpec(BaseModel):
    """Evaluator-side private task specification (contains expected answers & verifier params).

    NEVER passed directly to the harness; use ``to_task_input()`` to strip answers.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    input_files: dict[str, str] = Field(default_factory=dict)
    expected: Any = None
    verifier_params: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=12, ge=1, le=200)

    def to_task_input(self) -> TaskInput:
        """Project into a sanitized ``TaskInput`` strictly excluding answers & verifiers."""
        return TaskInput(
            task_id=self.id,
            prompt=self.prompt,
            workspace_seed_files=dict(self.input_files),
            max_steps=self.max_steps,
            metadata={"tags": list(self.tags)},
        )


class TaskScore(BaseModel):
    """Frozen scorer result for a single task rollout."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    rollout: int = Field(default=0, ge=0)
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    summary: str = ""
    errors: list[str] = Field(default_factory=list)
    trace_excerpt: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Aggregated evaluation report for either the evolve or held-out split."""

    model_config = ConfigDict(extra="forbid")

    split: SplitName
    mean_score: float = Field(ge=0.0, le=1.0)
    passed: int = Field(ge=0)
    total: int = Field(ge=0)
    task_scores: list[TaskScore] = Field(default_factory=list)
    by_task_passed: dict[str, bool] = Field(default_factory=dict)
    by_task_score: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_passed_total(self) -> EvaluationReport:
        if self.passed > self.total:
            raise ValueError(f"passed ({self.passed}) cannot exceed total ({self.total})")
        return self

    def format_evolve_feedback(self) -> str:
        """Format sanitized evolve feedback ``F_t`` for Proposer/Selector prompts.

        Refuses to format held-out reports as defense-in-depth against leakage.
        """
        if self.split != "evolve":
            raise ValueError(
                "Held-out EvaluationReport must never be formatted as proposer/selector feedback."
            )
        lines: list[str] = [
            (
                f"Evolve Evaluation Summary: mean_score={self.mean_score:.4f} "
                f"({self.passed}/{self.total} tasks passed)"
            )
        ]
        for ts in self.task_scores:
            status = "PASS" if ts.passed else "FAIL"
            lines.append(
                f"- [{status}] {ts.task_id} (rollout={ts.rollout}, score={ts.score:.3f}): "
                f"{ts.summary}"
            )
            for err in ts.errors:
                lines.append(f"    error: {err}")
            for trace_line in ts.trace_excerpt:
                lines.append(f"    trace: {trace_line}")
        return "\n".join(lines)


@runtime_checkable
class ScorerProtocol(Protocol):
    """Frozen in-process task scorer interface."""

    def score(self, task: TaskSpec, result: HarnessResult) -> TaskScore: ...
