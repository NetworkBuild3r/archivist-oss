"""FROZEN stable I/O contract for the CRSI seed harness (TaskInput -> HarnessResult).

This contract stays fixed across policy-model and domain swaps so SIM rationales
and harness middleware remain portable.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "expected",
        "expected_answer",
        "ground_truth",
        "verifier_params",
        "verifier",
        "split",
        "heldout",
    }
)


class TaskInput(BaseModel):
    """Sanitized task payload passed to the harness (strictly excludes answers & verifiers)."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    workspace_seed_files: dict[str, str] = Field(default_factory=dict)
    max_steps: int = Field(default=12, ge=1, le=200)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _reject_forbidden_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        leaked = FORBIDDEN_METADATA_KEYS.intersection(k.lower() for k in value)
        if leaked:
            raise ValueError(
                f"TaskInput.metadata must not contain evaluator/answer keys: {sorted(leaked)}"
            )
        return value


class CommandResult(BaseModel):
    """Structured result of running a sandboxed command (shell=False)."""

    model_config = ConfigDict(extra="forbid")

    argv: list[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: float = Field(default=0.0, ge=0.0)


class TurnRecord(BaseModel):
    """One policy/tool turn captured in the harness execution trace."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1)
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    output: str = ""
    error: str | None = None
    executed_command: bool = False
    command_exit_code: int | None = None
    discarded: bool = False


class InterlockEvent(BaseModel):
    """Record of an ExecutionGatingInterlock evaluation on a completion attempt."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1)
    allowed: bool
    case: str
    reason: str
    advisor_notes: list[str] = Field(default_factory=list)


class HarnessResult(BaseModel):
    """Structured output returned by a harness run to the frozen evaluator."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    completed: bool
    final_answer: str = ""
    workspace_artifacts: dict[str, str] = Field(default_factory=dict)
    trace: list[TurnRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    interlock_events: list[InterlockEvent] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


@runtime_checkable
class HarnessProtocol(Protocol):
    """Stable execution protocol implemented by H_0 and evolved harnesses."""

    async def run(self, task: TaskInput) -> HarnessResult: ...
