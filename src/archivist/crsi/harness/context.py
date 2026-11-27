"""EDITABLE session context assembly and step-budget reminders for the seed harness."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from archivist.crsi.harness.contract import TaskInput, TurnRecord
from archivist.crsi.llm.base import Message

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_FALLBACK_SYSTEM_PROMPT = (
    "You are a coding agent in an isolated workspace. Respond with JSON actions: "
    "list_files, read_file, write_file, run_command, or finish."
)
_FALLBACK_CHECKLIST = (
    "Verify your solution with an error-free command execution before returning finish."
)


def _load_prompt_file(filename: str, fallback: str) -> str:
    path = _PROMPTS_DIR / filename
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return fallback


class SessionContext:
    """Assembles bounded policy messages from templates, workspace state, and trace."""

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        completion_checklist: str | None = None,
        max_history_turns: int = 8,
        max_output_chars: int = 4000,
    ) -> None:
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else _load_prompt_file("system.md", _FALLBACK_SYSTEM_PROMPT)
        )
        self.completion_checklist = (
            completion_checklist
            if completion_checklist is not None
            else _load_prompt_file("completion_checklist.md", _FALLBACK_CHECKLIST)
        )
        self.max_history_turns = max(1, max_history_turns)
        self.max_output_chars = max(200, max_output_chars)

    def build_messages(
        self,
        *,
        task: TaskInput,
        step: int,
        workspace_files: Sequence[str],
        trace: Sequence[TurnRecord],
        interlock_feedback: str | None = None,
    ) -> list[Message]:
        """Construct the bounded message list for policy step ``step``."""
        remaining = max(0, task.max_steps - step + 1)
        system_block = f"{self.system_prompt}\n\n{self.completion_checklist}"
        files_listing = ", ".join(workspace_files) if workspace_files else "(empty)"

        messages: list[Message] = [
            Message(role="system", content=system_block),
            Message(
                role="user",
                content=(
                    f"Task ID: {task.task_id}\n"
                    f"Workspace files: {files_listing}\n"
                    f"Prompt:\n{task.prompt}"
                ),
            ),
        ]

        recent = trace[-self.max_history_turns :]
        for rec in recent:
            assistant_summary = f"Action step {rec.step}: {rec.action} {rec.args}"
            messages.append(Message(role="assistant", content=assistant_summary))
            obs_parts: list[str] = []
            if rec.output:
                obs_parts.append(rec.output[: self.max_output_chars])
            if rec.error:
                obs_parts.append(f"ERROR: {rec.error}")
            obs_text = "\n".join(obs_parts) if obs_parts else "(no output)"
            messages.append(
                Message(role="user", content=f"Observation (step {rec.step}):\n{obs_text}")
            )

        reminder = (
            f"Step {step}/{task.max_steps} (remaining turns including this one: {remaining})."
        )
        if interlock_feedback:
            reminder = f"{interlock_feedback}\n\n{reminder}"
        messages.append(Message(role="user", content=reminder))
        return messages
