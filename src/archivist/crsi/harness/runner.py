"""EDITABLE seed harness turn loop wiring tools, context, permission gates, and interlock."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from archivist.crsi.guards.secrets import SecretRedactor
from archivist.crsi.harness.context import SessionContext
from archivist.crsi.harness.contract import (
    HarnessResult,
    InterlockEvent,
    TaskInput,
    TurnRecord,
)
from archivist.crsi.harness.interlock import ExecutionGatingInterlock
from archivist.crsi.harness.permissions import PermissionDeniedError, PermissionGate
from archivist.crsi.harness.tools import ToolRuntime
from archivist.crsi.harness.workspace import WorkspaceManager, WorkspaceSecurityError
from archivist.crsi.llm.base import LLMClient, LLMRequest


def _parse_policy_action(raw_content: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a JSON action object from policy output; return ``(payload, error)``."""
    stripped = raw_content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"Malformed/truncated JSON action block: {exc}"
    if not isinstance(parsed, dict):
        return None, "Policy response must be a JSON object."
    if parsed.get("task_complete") is True and "action" not in parsed:
        return {"action": "finish", "answer": str(parsed.get("answer", ""))}, None
    if not isinstance(parsed.get("action"), str) or not parsed["action"].strip():
        return None, "Policy JSON missing string 'action' field."
    return parsed, None


class SeedHarness:
    """Runnable seed harness ``H_0`` implementing ``HarnessProtocol``."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        interlock_factory: Callable[[], ExecutionGatingInterlock] | None = None,
        permission_gate_factory: Callable[[], PermissionGate] | None = None,
        context: SessionContext | None = None,
        command_timeout_s: float = 30.0,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self.llm_client = llm_client
        self._interlock_factory = (
            interlock_factory
            if interlock_factory is not None
            else lambda: ExecutionGatingInterlock(enabled=True)
        )
        self._permission_gate_factory = (
            permission_gate_factory if permission_gate_factory is not None else PermissionGate
        )
        self.context = context if context is not None else SessionContext()
        self.command_timeout_s = command_timeout_s
        self.redactor = redactor if redactor is not None else SecretRedactor()

    async def run(self, task: TaskInput) -> HarnessResult:
        """Execute ``task`` inside an isolated workspace sandbox with interlock enforcement."""
        interlock = self._interlock_factory()
        permission_gate = self._permission_gate_factory()
        trace: list[TurnRecord] = []
        errors: list[str] = []
        interlock_events: list[InterlockEvent] = []
        last_interlock_feedback: str | None = None
        completed = False
        final_answer = ""

        with WorkspaceManager() as workspace:
            workspace.seed_files(task.workspace_seed_files)
            tools = ToolRuntime(
                workspace=workspace,
                permission_gate=permission_gate,
                command_timeout_s=self.command_timeout_s,
                redactor=self.redactor,
            )

            for step in range(1, task.max_steps + 1):
                messages = self.context.build_messages(
                    task=task,
                    step=step,
                    workspace_files=workspace.list_files(),
                    trace=trace,
                    interlock_feedback=last_interlock_feedback,
                )
                last_interlock_feedback = None

                response = await self.llm_client.complete(
                    LLMRequest(
                        role="policy",
                        messages=messages,
                        json_mode=True,
                        metadata={"task_id": task.task_id, "step": step},
                    )
                )

                payload, parse_err = _parse_policy_action(response.content)
                if parse_err is not None or payload is None:
                    err_msg = parse_err or "Invalid policy action"
                    errors.append(err_msg)
                    trace.append(
                        TurnRecord(
                            step=step,
                            action="invalid",
                            error=err_msg,
                            discarded=True,
                        )
                    )
                    interlock.record_turn(
                        executed_command=False,
                        tool_error=err_msg,
                        discarded=True,
                    )
                    continue

                action = str(payload["action"]).strip()
                args = {k: v for k, v in payload.items() if k != "action"}

                if action == "finish":
                    candidate_answer = str(payload.get("answer", ""))
                    decision = interlock.check_completion_attempt(
                        answer=candidate_answer,
                        step=step,
                        ran_command_this_turn=False,
                    )
                    interlock_events.append(
                        InterlockEvent(
                            step=step,
                            allowed=decision.allowed,
                            case=decision.case,
                            reason=decision.feedback,
                            advisor_notes=list(decision.advisor_notes),
                        )
                    )
                    trace.append(
                        TurnRecord(
                            step=step,
                            action="finish",
                            args={"answer": candidate_answer},
                            output=decision.feedback,
                            error=None if decision.allowed else decision.feedback,
                        )
                    )
                    if decision.allowed:
                        completed = True
                        final_answer = candidate_answer
                        break
                    last_interlock_feedback = decision.feedback
                    continue

                try:
                    if action == "list_files":
                        listing = tools.list_files()
                        trace.append(
                            TurnRecord(
                                step=step,
                                action=action,
                                args=args,
                                output=json.dumps(listing),
                            )
                        )
                        interlock.record_turn(executed_command=False)
                    elif action == "read_file":
                        content = tools.read_file(str(payload.get("path", "")))
                        trace.append(
                            TurnRecord(
                                step=step,
                                action=action,
                                args=args,
                                output=content,
                            )
                        )
                        interlock.record_turn(executed_command=False)
                    elif action == "write_file":
                        msg = tools.write_file(
                            str(payload.get("path", "")),
                            str(payload.get("content", "")),
                        )
                        trace.append(
                            TurnRecord(
                                step=step,
                                action=action,
                                args={"path": str(payload.get("path", ""))},
                                output=msg,
                            )
                        )
                        interlock.record_turn(executed_command=False)
                    elif action == "run_command":
                        raw_argv = payload.get("argv")
                        if not isinstance(raw_argv, list) or not raw_argv:
                            raise ValueError("run_command requires a non-empty 'argv' list.")
                        cmd_res = await asyncio.to_thread(
                            tools.run_command, [str(x) for x in raw_argv]
                        )
                        combined_out = "\n".join(
                            part for part in (cmd_res.stdout, cmd_res.stderr) if part
                        )
                        cmd_err = (
                            f"Command exited with code {cmd_res.exit_code}"
                            if cmd_res.exit_code != 0 or cmd_res.timed_out
                            else None
                        )
                        if cmd_err:
                            errors.append(cmd_err)
                        trace.append(
                            TurnRecord(
                                step=step,
                                action=action,
                                args={"argv": cmd_res.argv},
                                output=combined_out,
                                error=cmd_err,
                                executed_command=True,
                                command_exit_code=cmd_res.exit_code,
                            )
                        )
                        interlock.record_turn(
                            executed_command=True,
                            command_result=cmd_res,
                            tool_error=cmd_err,
                        )
                    else:
                        unknown_err = f"Unknown harness action: {action!r}"
                        errors.append(unknown_err)
                        trace.append(
                            TurnRecord(
                                step=step,
                                action=action,
                                args=args,
                                error=unknown_err,
                                discarded=True,
                            )
                        )
                        interlock.record_turn(
                            executed_command=False,
                            tool_error=unknown_err,
                            discarded=True,
                        )
                except (
                    PermissionDeniedError,
                    WorkspaceSecurityError,
                    FileNotFoundError,
                    ValueError,
                ) as exc:
                    err_text = self.redactor.redact_text(str(exc))
                    errors.append(err_text)
                    trace.append(
                        TurnRecord(
                            step=step,
                            action=action,
                            args=args,
                            error=err_text,
                        )
                    )
                    interlock.record_turn(executed_command=False, tool_error=err_text)

            artifacts = workspace.snapshot_artifacts()

        return HarnessResult(
            task_id=task.task_id,
            completed=completed,
            final_answer=final_answer,
            workspace_artifacts=artifacts,
            trace=trace,
            errors=errors,
            interlock_events=interlock_events,
            metrics={
                "steps": float(len(trace)),
                "commands_executed": float(interlock.state.total_executed),
                "interlock_rejections": float(sum(1 for ev in interlock_events if not ev.allowed)),
            },
        )
