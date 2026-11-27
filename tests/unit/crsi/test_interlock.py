"""Unit tests for ExecutionGatingInterlock and additive QualityFeasibilityCheck (SPEC-004)."""

from __future__ import annotations

import json
import sys

import pytest

from archivist.crsi.config import CrsiConfig
from archivist.crsi.harness.contract import CommandResult, TaskInput
from archivist.crsi.harness.interlock import (
    ExecutionGatingInterlock,
    QualityFeasibilityCheck,
)
from archivist.crsi.harness.runner import SeedHarness
from archivist.crsi.llm.base import RoutedLLMClient
from archivist.crsi.llm.fake import FakeScriptedAdapter

pytestmark = [pytest.mark.crsi, pytest.mark.unit]


def test_interlock_case_c_blocks_completion_with_zero_executions():
    interlock = ExecutionGatingInterlock(enabled=True, min_required_executions=1, max_rejections=3)
    decision = interlock.check_completion_attempt(answer="done", step=1)
    assert decision.allowed is False
    assert decision.case == "case_c_no_execution"
    assert interlock.state.consecutive_rejections == 1


def test_interlock_case_0_blocks_nonzero_exit_and_active_error_signature():
    interlock = ExecutionGatingInterlock(enabled=True)
    interlock.record_turn(
        executed_command=True,
        command_result=CommandResult(
            argv=["python3", "test.py"],
            exit_code=1,
            stdout="",
            stderr="AssertionError: expected 42",
        ),
    )
    d1 = interlock.check_completion_attempt(answer="done", step=2)
    assert d1.allowed is False
    assert d1.case == "case_0_active_error"

    # Even with exit_code=0, an unhandled traceback in stdout triggers Case 0.
    interlock2 = ExecutionGatingInterlock(enabled=True)
    interlock2.record_turn(
        executed_command=True,
        command_result=CommandResult(
            argv=["python3", "run.py"],
            exit_code=0,
            stdout="Traceback (most recent call last):\nValueError: bad input",
            stderr="",
        ),
    )
    d2 = interlock2.check_completion_attempt(answer="done", step=2)
    assert d2.allowed is False
    assert d2.case == "case_0_active_error"


def test_interlock_case_a_blocks_prior_turn_tool_error_without_recovery():
    interlock = ExecutionGatingInterlock(enabled=True)
    # First run a clean command so total_executed == 1.
    interlock.record_turn(
        executed_command=True,
        command_result=CommandResult(
            argv=["python3", "-c", "print('ok')"], exit_code=0, stdout="ok"
        ),
    )
    # Next turn fails with a tool error (e.g., missing file / permission denied).
    interlock.record_turn(executed_command=False, tool_error="FileNotFoundError: missing.py")

    decision = interlock.check_completion_attempt(answer="done", step=3)
    assert decision.allowed is False
    assert decision.case == "case_a_prior_error"

    # Running a clean verification command clears the prior error and allows completion.
    interlock.record_turn(
        executed_command=True,
        command_result=CommandResult(
            argv=["python3", "-c", "print('clean')"], exit_code=0, stdout="clean"
        ),
    )
    decision_ok = interlock.check_completion_attempt(answer="done", step=4)
    assert decision_ok.allowed is True
    assert decision_ok.case == "verified_pass"


def test_interlock_case_b_blocks_discarded_turn_without_recovery():
    interlock = ExecutionGatingInterlock(enabled=True)
    interlock.record_turn(
        executed_command=True,
        command_result=CommandResult(
            argv=["python3", "-c", "print('ok')"], exit_code=0, stdout="ok"
        ),
    )
    # Simulate a malformed / truncated turn with no command execution.
    interlock.state.last_turn_had_error = False
    interlock.state.last_turn_discarded = True

    decision = interlock.check_completion_attempt(answer="done", step=3)
    assert decision.allowed is False
    assert decision.case == "case_b_discarded_turn"


def test_interlock_bounds_consecutive_rejections_to_prevent_deadlock():
    interlock = ExecutionGatingInterlock(enabled=True, max_rejections=2)
    d1 = interlock.check_completion_attempt(answer="done", step=1)
    d2 = interlock.check_completion_attempt(answer="done", step=2)
    d3 = interlock.check_completion_attempt(answer="done", step=3)
    assert d1.allowed is False
    assert d2.allowed is False
    assert d3.allowed is True
    assert d3.case == "max_rejections_override"


def test_additive_quality_feasibility_advisor_runs_beside_core_interlock():
    interlock = ExecutionGatingInterlock(
        enabled=True,
        advisors=[QualityFeasibilityCheck(block_once=True)],
    )
    # Core Case C still triggers first when zero commands ran.
    d0 = interlock.check_completion_attempt(answer="CONSTRAINTS PASSED score=0.5", step=1)
    assert d0.allowed is False
    assert d0.case == "case_c_no_execution"

    # Now execute a feasible-but-suboptimal run.
    interlock.record_turn(
        executed_command=True,
        command_result=CommandResult(
            argv=["python3", "eval_plan.py"],
            exit_code=0,
            stdout="CONSTRAINTS PASSED score=0.60",
        ),
    )
    # Advisor triggers a one-shot quality verification rejection.
    d1 = interlock.check_completion_attempt(answer="score=0.60", step=2)
    assert d1.allowed is False
    assert d1.case == "advisor_quality_feasibility"
    assert any("0.60" in note for note in d1.advisor_notes)

    # After policy improves and re-runs, completion is allowed (advisor only blocks once).
    interlock.record_turn(
        executed_command=True,
        command_result=CommandResult(
            argv=["python3", "eval_plan.py"],
            exit_code=0,
            stdout="CONSTRAINTS PASSED score=0.95",
        ),
    )
    d2 = interlock.check_completion_attempt(answer="score=0.95", step=3)
    assert d2.allowed is True
    assert d2.case == "verified_pass"


async def test_seed_harness_interlock_forces_real_execution_on_premature_finish():
    """Scripted policy tries to finish immediately; interlock rejects and forces execution."""
    fake = FakeScriptedAdapter(
        responses={
            "policy": [
                json.dumps({"action": "finish", "answer": "hallucinated"}),
                json.dumps(
                    {
                        "action": "run_command",
                        "argv": [sys.executable, "-c", "print('verified-42')"],
                    }
                ),
                json.dumps({"action": "finish", "answer": "verified-42"}),
            ]
        }
    )
    client = RoutedLLMClient(CrsiConfig(), adapters={"fake": fake})
    harness = SeedHarness(client)
    result = await harness.run(TaskInput(task_id="hallucinate-1", prompt="Compute and verify."))

    assert result.completed is True
    assert result.final_answer == "verified-42"
    assert len(result.interlock_events) == 2
    assert result.interlock_events[0].allowed is False
    assert result.interlock_events[0].case == "case_c_no_execution"
    assert result.interlock_events[1].allowed is True
    assert result.interlock_events[1].case == "verified_pass"
    assert result.metrics["commands_executed"] == 1.0
    assert result.metrics["interlock_rejections"] == 1.0
