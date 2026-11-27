"""EDITABLE ExecutionGatingInterlock middleware and additive quality/feasibility hooks.

Directly implements Yu et al. (arXiv:2611.18856), Appendix F.2.1–F.2.2:
- Core interlock blocks premature/hallucinated completion across Cases 0, A, B, and C
  with bounded consecutive rejections to prevent deadlock.
- Additive ``CompletionAdvisor`` hooks (e.g. ``QualityFeasibilityCheck``) run beside
  the core interlock without replacing or weakening it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from archivist.crsi.harness.contract import CommandResult

ERROR_SIGNATURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("python_traceback", re.compile(r"Traceback \(most recent call last\):")),
    (
        "unhandled_exception",
        re.compile(
            r"\b(?:AssertionError|SyntaxError|NameError|TypeError|ValueError|"
            r"RuntimeError|KeyError|IndexError|ZeroDivisionError|ModuleNotFoundError|"
            r"ImportError|FileNotFoundError)\b"
        ),
    ),
    ("test_failure", re.compile(r"(?m)^\s*(?:FAILED|ERROR)\b|assertions? failed", re.IGNORECASE)),
    ("http_error", re.compile(r"\bHTTP/?\d?\.?\d?\s+[45]\d{2}\b")),
    ("infeasible", re.compile(r"\b(?:INFEASIBLE|CONSTRAINT(?:S)?\s+FAILED)\b", re.IGNORECASE)),
)

_FEASIBLE_SCORE_RE = re.compile(
    r"(?:CONSTRAINTS?\s+PASSED|FEASIBLE)\b.*?\b(?:score|objective|cost|metric)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)


class AdvisorFeedback(BaseModel):
    """Advisory or one-shot verification feedback from an additive CompletionAdvisor."""

    model_config = ConfigDict(extra="forbid")

    name: str
    block_once: bool = False
    message: str


class InterlockState(BaseModel):
    """Cross-turn execution proof state tracked by ``ExecutionGatingInterlock``."""

    model_config = ConfigDict(extra="forbid")

    total_executed: int = Field(default=0, ge=0)
    last_turn_had_error: bool = False
    last_turn_discarded: bool = False
    consecutive_rejections: int = Field(default=0, ge=0)
    pending_confirmation: bool = False
    last_command_exit_code: int | None = None
    last_command_output: str = ""
    recent_outputs: list[str] = Field(default_factory=list)
    triggered_advisors: list[str] = Field(default_factory=list)


class InterlockDecision(BaseModel):
    """Outcome of checking a completion attempt against the interlock."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    case: str
    feedback: str
    advisor_notes: list[str] = Field(default_factory=list)


@runtime_checkable
class CompletionAdvisor(Protocol):
    """Additive check executed beside the core execution interlock (Appendix F.2.2)."""

    name: str

    def advise(
        self,
        *,
        answer: str,
        state: InterlockState,
        recent_outputs: Sequence[str],
    ) -> AdvisorFeedback | None: ...


class QualityFeasibilityCheck:
    """Additive check detecting feasible-but-potentially-suboptimal outputs (Appendix F.2.2).

    Runs beside ``ExecutionGatingInterlock`` without replacing execution verification.
    """

    def __init__(self, *, block_once: bool = True, name: str = "quality_feasibility") -> None:
        self.name = name
        self.block_once = block_once

    def advise(
        self,
        *,
        answer: str,
        state: InterlockState,
        recent_outputs: Sequence[str],
    ) -> AdvisorFeedback | None:
        combined = "\n".join([*recent_outputs[-3:], answer])
        match = _FEASIBLE_SCORE_RE.search(combined)
        if not match:
            return None
        metric_val = match.group(1)
        return AdvisorFeedback(
            name=self.name,
            block_once=self.block_once,
            message=(
                f"Quality/feasibility check: constraints passed with reported metric={metric_val}. "
                "Verify whether solution quality/optimality can be improved before finalizing."
            ),
        )


def detect_error_signature(text: str) -> str | None:
    """Return the first matched failure/error signature label in ``text``, or None."""
    if not text:
        return None
    for label, pattern in ERROR_SIGNATURE_PATTERNS:
        if pattern.search(text):
            return label
    return None


class ExecutionGatingInterlock:
    """Cross-turn execution-verification middleware blocking unverified completion."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_required_executions: int = 1,
        max_rejections: int = 3,
        advisors: Sequence[CompletionAdvisor] = (),
    ) -> None:
        self.enabled = enabled
        self.min_required_executions = max(1, min_required_executions)
        self.max_rejections = max(1, max_rejections)
        self.advisors: list[CompletionAdvisor] = list(advisors)
        self.state = InterlockState()

    def add_advisor(self, advisor: CompletionAdvisor) -> None:
        """Attach an additive ``CompletionAdvisor`` beside the core interlock."""
        self.advisors.append(advisor)

    def record_turn(
        self,
        *,
        executed_command: bool = False,
        command_result: CommandResult | None = None,
        tool_error: str | None = None,
        discarded: bool = False,
    ) -> None:
        """Update cross-turn execution state after a policy step."""
        self.state.last_turn_discarded = discarded

        if executed_command and command_result is not None:
            self.state.total_executed += 1
            self.state.last_command_exit_code = command_result.exit_code
            combined = "\n".join(
                part for part in (command_result.stdout, command_result.stderr) if part
            )
            self.state.last_command_output = combined
            if combined:
                self.state.recent_outputs.append(combined)
            has_sig = detect_error_signature(combined) is not None
            self.state.last_turn_had_error = (
                command_result.exit_code != 0
                or command_result.timed_out
                or has_sig
                or bool(tool_error)
            )
            if not self.state.last_turn_had_error and not discarded:
                self.state.pending_confirmation = False
            return

        if tool_error:
            self.state.last_turn_had_error = True
            self.state.recent_outputs.append(tool_error)
        elif discarded:
            self.state.last_turn_had_error = True

    def check_completion_attempt(
        self,
        *,
        answer: str,
        step: int,
        ran_command_this_turn: bool = False,
    ) -> InterlockDecision:
        """Evaluate whether a completion claim has valid error-free execution proof."""
        _ = step
        if not self.enabled:
            return InterlockDecision(
                allowed=True,
                case="disabled",
                feedback="ExecutionGatingInterlock is disabled.",
            )

        # Bounded override prevents deadlock when the policy insists on finishing.
        if self.state.consecutive_rejections >= self.max_rejections:
            self.state.pending_confirmation = False
            return InterlockDecision(
                allowed=True,
                case="max_rejections_override",
                feedback=(
                    f"Allowing completion after {self.state.consecutive_rejections} consecutive "
                    "interlock rejections (deadlock bound reached)."
                ),
            )

        # Case 0: Active error / non-zero exit / failure signature in recent execution output.
        sig = detect_error_signature(self.state.last_command_output)
        if (
            self.state.last_command_exit_code is not None and self.state.last_command_exit_code != 0
        ) or sig is not None:
            self.state.consecutive_rejections += 1
            self.state.pending_confirmation = True
            detail = (
                f"exit_code={self.state.last_command_exit_code}"
                if self.state.last_command_exit_code not in (None, 0)
                else f"error signature '{sig}'"
            )
            return InterlockDecision(
                allowed=False,
                case="case_0_active_error",
                feedback=(
                    f"Completion rejected (Case 0): recent execution contains active failure "
                    f"({detail}). Fix the issue and run a clean verification command first."
                ),
            )

        # Case A: Prior turn had an error and no recovery command ran this turn.
        if self.state.last_turn_had_error and not ran_command_this_turn:
            self.state.consecutive_rejections += 1
            self.state.pending_confirmation = True
            return InterlockDecision(
                allowed=False,
                case="case_a_prior_error",
                feedback=(
                    "Completion rejected (Case A): the previous turn had an error with no "
                    "subsequent successful command execution."
                ),
            )

        # Case B: Prior turn was discarded/truncated/malformed with no recovery run.
        if self.state.last_turn_discarded and not ran_command_this_turn:
            self.state.consecutive_rejections += 1
            self.state.pending_confirmation = True
            return InterlockDecision(
                allowed=False,
                case="case_b_discarded_turn",
                feedback=(
                    "Completion rejected (Case B): the previous turn's command block was "
                    "discarded or truncated. Re-run and verify before completing."
                ),
            )

        # Case C: Zero (or insufficient) verified command executions — hallucinated success.
        if self.state.total_executed < self.min_required_executions and not ran_command_this_turn:
            self.state.consecutive_rejections += 1
            self.state.pending_confirmation = True
            return InterlockDecision(
                allowed=False,
                case="case_c_no_execution",
                feedback=(
                    f"Completion rejected (Case C): {self.state.total_executed} command(s) "
                    f"executed (minimum required: {self.min_required_executions}). Execute a "
                    "verification command to prove your solution works before finishing."
                ),
            )

        # Core execution proof satisfied; now run additive CompletionAdvisors beside it.
        advisor_notes: list[str] = []
        for advisor in self.advisors:
            fb = advisor.advise(
                answer=answer,
                state=self.state,
                recent_outputs=self.state.recent_outputs,
            )
            if fb is None:
                continue
            advisor_notes.append(fb.message)
            if fb.block_once and fb.name not in self.state.triggered_advisors:
                self.state.triggered_advisors.append(fb.name)
                self.state.consecutive_rejections += 1
                self.state.pending_confirmation = True
                return InterlockDecision(
                    allowed=False,
                    case=f"advisor_{fb.name}",
                    feedback=fb.message,
                    advisor_notes=advisor_notes,
                )

        self.state.consecutive_rejections = 0
        self.state.pending_confirmation = False
        return InterlockDecision(
            allowed=True,
            case="verified_pass",
            feedback="Completion approved with error-free execution proof.",
            advisor_notes=advisor_notes,
        )
