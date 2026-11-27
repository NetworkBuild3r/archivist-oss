"""Unit tests for SeedHarness H_0, WorkspaceManager, ToolRuntime, and PermissionGate (SPEC-004)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from archivist.crsi.config import CrsiConfig
from archivist.crsi.harness.contract import TaskInput
from archivist.crsi.harness.permissions import (
    ActionRisk,
    PermissionDeniedError,
    PermissionGate,
)
from archivist.crsi.harness.runner import SeedHarness
from archivist.crsi.harness.tools import ToolRuntime
from archivist.crsi.harness.workspace import WorkspaceManager, WorkspaceSecurityError
from archivist.crsi.llm.base import RoutedLLMClient
from archivist.crsi.llm.fake import FakeScriptedAdapter

pytestmark = [pytest.mark.crsi, pytest.mark.unit]


def test_task_input_rejects_forbidden_evaluator_metadata():
    with pytest.raises(ValidationError, match="expected"):
        TaskInput(task_id="t1", prompt="solve", metadata={"expected": "42"})
    with pytest.raises(ValidationError, match="heldout"):
        TaskInput(task_id="t1", prompt="solve", metadata={"heldout": True})


def test_workspace_manager_blocks_traversal_absolute_and_symlink_escapes(tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("SECRET_OUTSIDE", encoding="utf-8")

    ws_root = tmp_path / "ws"
    with WorkspaceManager(root=ws_root, auto_cleanup=False) as ws:
        ws.write_file("hello.txt", "world")
        assert ws.read_file("hello.txt") == "world"
        assert ws.list_files() == ["hello.txt"]

        for bad in ("../outside.txt", "sub/../../outside.txt", str(outside), "a\x00b"):
            with pytest.raises(WorkspaceSecurityError):
                ws.read_file(bad)
            with pytest.raises(WorkspaceSecurityError):
                ws.write_file(bad, "pwn")

        # Symlink inside workspace pointing outside must be blocked on read & write.
        escape_link = ws_root / "escape.txt"
        escape_link.symlink_to(outside)
        with pytest.raises(WorkspaceSecurityError):
            ws.read_file("escape.txt")
        with pytest.raises(WorkspaceSecurityError):
            ws.write_file("escape.txt", "overwrite")
        assert "escape.txt" not in ws.list_files()


def test_tool_runtime_strips_secret_env_and_captures_output_on_timeout(tmp_path: Path):
    with WorkspaceManager(root=tmp_path / "ws", auto_cleanup=False) as ws:
        gate = PermissionGate()
        tools = ToolRuntime(
            workspace=ws,
            permission_gate=gate,
            command_timeout_s=5.0,
            base_environ={
                "PATH": "/usr/bin:/bin",
                "OPENAI_API_KEY": "sk-live-do-not-expose-1234567890",
                "VAULT_TOKEN": "vault-super-secret-token",
                "SAFE_MARKER": "visible",
            },
        )

        # Verify secret env vars are stripped in child process.
        res = tools.run_command(
            [
                sys.executable,
                "-c",
                "import os; print(os.getenv('OPENAI_API_KEY'), os.getenv('VAULT_TOKEN'), os.getenv('SAFE_MARKER'))",
            ]
        )
        assert res.exit_code == 0
        assert res.timed_out is False
        assert res.stdout.strip() == "None None visible"

        # Verify timeout retains partial stdout and sets timed_out=True.
        timeout_res = tools.run_command(
            [
                sys.executable,
                "-u",
                "-c",
                "import sys, time; print('partial-before-timeout', flush=True); time.sleep(5)",
            ],
            timeout_s=0.3,
        )
        assert timeout_res.timed_out is True
        assert timeout_res.exit_code == -1
        assert "partial-before-timeout" in timeout_res.stdout
        assert "timed out" in timeout_res.stderr.lower()


def test_permission_gate_is_fail_closed_by_default_and_honors_explicit_approval(tmp_path: Path):
    gate = PermissionGate()
    assert gate.classify_command(["python3", "script.py"]) == ActionRisk.SAFE
    assert gate.classify_command(["curl", "https://example.com"]) == ActionRisk.RISKY
    assert gate.classify_command(["rm", "-rf", "/"]) == ActionRisk.DESTRUCTIVE
    assert gate.classify_command(["bash", "-c", "echo hi"]) == ActionRisk.DESTRUCTIVE
    assert gate.classify_command(["git", "push", "origin", "main"]) == ActionRisk.DESTRUCTIVE

    with WorkspaceManager(root=tmp_path / "ws", auto_cleanup=False) as ws:
        tools = ToolRuntime(workspace=ws, permission_gate=gate)
        for blocked_argv in (
            ["curl", "https://example.com"],
            ["rm", "-rf", "anything"],
            ["bash", "-c", "echo nope"],
            ["git", "push", "origin", "main"],
        ):
            with pytest.raises(PermissionDeniedError):
                tools.run_command(blocked_argv)

    assert len(gate.audit_log) == 4
    assert all(not d.allowed for d in gate.audit_log)

    # Explicit operator approval callback allows a risky action.
    approved_calls: list[tuple[str, ActionRisk]] = []

    def approve_risky(summary: str, risk: ActionRisk) -> bool:
        approved_calls.append((summary, risk))
        return risk == ActionRisk.RISKY

    explicit_gate = PermissionGate(approval_callback=approve_risky)
    decision = explicit_gate.check_action(
        action="run_command",
        risk=ActionRisk.RISKY,
        summary="curl https://example.com",
    )
    assert decision.allowed is True
    assert approved_calls == [("curl https://example.com", ActionRisk.RISKY)]

    destructive_decision = explicit_gate.check_action(
        action="run_command",
        risk=ActionRisk.DESTRUCTIVE,
        summary="rm -rf /",
    )
    assert destructive_decision.allowed is False


async def test_seed_harness_runs_write_verify_and_finish_end_to_end():
    fake = FakeScriptedAdapter(
        responses={
            "policy": [
                json.dumps(
                    {
                        "action": "write_file",
                        "path": "solve.py",
                        "content": "print(21 * 2)\n",
                    }
                ),
                json.dumps(
                    {
                        "action": "run_command",
                        "argv": [sys.executable, "solve.py"],
                    }
                ),
                json.dumps({"action": "finish", "answer": "42"}),
            ]
        }
    )
    client = RoutedLLMClient(CrsiConfig(), adapters={"fake": fake})
    harness = SeedHarness(client)
    task = TaskInput(task_id="math-1", prompt="Write solve.py printing 42 and verify it.")

    result = await harness.run(task)
    assert result.completed is True
    assert result.final_answer == "42"
    assert result.errors == []
    assert result.workspace_artifacts["solve.py"] == "print(21 * 2)\n"
    assert result.metrics["commands_executed"] == 1.0
    assert result.metrics["interlock_rejections"] == 0.0
    assert len(result.interlock_events) == 1
    assert result.interlock_events[0].allowed is True
    assert result.interlock_events[0].case == "verified_pass"


def test_maj1_workspace_preserves_dotfile_names(tmp_path: Path):
    ws_root = tmp_path / "ws_dot"
    with WorkspaceManager(root=ws_root, auto_cleanup=False) as ws:
        ws.write_file(".env", "FOO=bar")
        assert (ws_root / ".env").is_file()
        assert not (ws_root / "env").exists()
        assert ws.read_file(".env") == "FOO=bar"
        assert ws.read_file("./.env") == "FOO=bar"


def test_maj2_permission_gate_blocks_traversal_dotfiles_and_git_global_flag_bypasses():
    gate = PermissionGate()

    # File actions targeting traversal, absolute paths, .git/, or .crsi/ MUST be DESTRUCTIVE.
    for bad_path in (
        "../etc/passwd",
        "/etc/passwd",
        ".git/config",
        "./.git/HEAD",
        ".crsi/sim.json",
        "./.crsi/config.toml",
    ):
        assert gate.classify_file_action("write_file", bad_path) == ActionRisk.DESTRUCTIVE, bad_path

    # Git global flag bypasses (-C, -c, --git-dir, --config-env) and mutating subcommands blocked.
    for bad_git in (
        ["git", "-C", ".", "push", "origin", "main"],
        ["git", "-c", "a=b", "reset", "--hard"],
        ["git", "-c", "core.sshCommand=sh", "status"],
        ["git", "--config-env=core.sshCommand=ENV", "status"],
        ["git", "-C", "/etc", "status"],
        ["git", "-C", "..", "status"],
        ["git", "--git-dir=/tmp/.git", "status"],
        ["git", "checkout", "main"],
        ["git", "commit", "-m", "hi"],
    ):
        assert gate.classify_command(bad_git) == ActionRisk.DESTRUCTIVE, bad_git

    # Read-only allowlisted git subcommands inside workspace remain SAFE.
    assert gate.classify_command(["git", "status"]) == ActionRisk.SAFE
    assert gate.classify_command(["git", "-C", ".", "diff"]) == ActionRisk.SAFE
    assert gate.classify_command(["git", "log", "-n", "1"]) == ActionRisk.SAFE


async def test_min1_seed_harness_run_command_does_not_block_event_loop():
    import asyncio

    fake = FakeScriptedAdapter(
        responses={
            "policy": [
                json.dumps(
                    {
                        "action": "run_command",
                        "argv": [sys.executable, "-c", "import time; time.sleep(2.0)"],
                    }
                ),
            ]
        }
    )
    client = RoutedLLMClient(CrsiConfig(), adapters={"fake": fake})
    harness = SeedHarness(client, command_timeout_s=5.0)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            harness.run(TaskInput(task_id="slow-1", prompt="sleep")),
            timeout=0.35,
        )
