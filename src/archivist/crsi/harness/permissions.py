"""FROZEN permission gate requiring explicit human/operator approval for risky actions.

Defaults to fail-closed (``False``) so unattended/CI runs never execute destructive
or risky operations without explicit approval.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ActionRisk(str, Enum):
    """Risk classification for harness tool operations."""

    SAFE = "safe"
    RISKY = "risky"
    DESTRUCTIVE = "destructive"


ApprovalCallback = Callable[[str, ActionRisk], bool]


class PermissionDeniedError(PermissionError):
    """Raised when a RISKY or DESTRUCTIVE action is denied by ``PermissionGate``."""


@dataclass(frozen=True)
class PermissionDecision:
    """Immutable audit record of a permission gate check."""

    allowed: bool
    risk: ActionRisk
    action: str
    summary: str
    reason: str


def fail_closed_approval(summary: str, risk: ActionRisk) -> bool:
    """Default non-interactive approval callback: always denies risky/destructive actions."""
    _ = (summary, risk)
    return False


_SHELL_META_BINS: frozenset[str] = frozenset(
    {"sh", "bash", "zsh", "dash", "ksh", "fish", "cmd", "powershell", "pwsh"}
)
_DESTRUCTIVE_BINS: frozenset[str] = frozenset(
    {
        "rm",
        "rmdir",
        "mkfs",
        "dd",
        "chmod",
        "chown",
        "kill",
        "pkill",
        "killall",
        "shutdown",
        "reboot",
        "sudo",
        "su",
    }
)
_RISKY_NETWORK_BINS: frozenset[str] = frozenset(
    {"curl", "wget", "nc", "netcat", "ncat", "ssh", "scp", "sftp", "rsync", "ftp", "telnet"}
)
_DESTRUCTIVE_GIT_SUBCOMMANDS: frozenset[str] = frozenset(
    {"push", "merge", "rebase", "reset", "clean", "checkout", "switch", "commit", "tag"}
)


class PermissionGate:
    """Frozen action-risk classifier and fail-closed approval enforcer."""

    def __init__(self, approval_callback: ApprovalCallback | None = None) -> None:
        self._approval_callback: ApprovalCallback = (
            approval_callback if approval_callback is not None else fail_closed_approval
        )
        self.audit_log: list[PermissionDecision] = []

    @staticmethod
    def classify_file_action(op: str, rel_path: str) -> ActionRisk:
        """Classify a workspace file operation by risk level."""
        cleaned = rel_path.replace("\\", "/").lstrip("./")
        if ".." in cleaned.split("/") or cleaned.startswith("/"):
            return ActionRisk.DESTRUCTIVE
        if cleaned.startswith((".git/", ".crsi/")) or cleaned in {".git", ".crsi"}:
            return ActionRisk.DESTRUCTIVE
        if op in {"read_file", "list_files", "write_file"}:
            return ActionRisk.SAFE
        if op in {"delete_file", "remove_file"}:
            return ActionRisk.RISKY
        return ActionRisk.RISKY

    @staticmethod
    def classify_command(argv: Sequence[str]) -> ActionRisk:
        """Classify a command ``argv`` into SAFE, RISKY, or DESTRUCTIVE."""
        if not argv or not argv[0].strip():
            return ActionRisk.DESTRUCTIVE

        binary = Path(argv[0].strip()).name.lower()
        args_lower = [a.lower() for a in argv[1:]]

        if binary in _SHELL_META_BINS:
            return ActionRisk.DESTRUCTIVE
        if binary in _DESTRUCTIVE_BINS:
            return ActionRisk.DESTRUCTIVE
        if binary in _RISKY_NETWORK_BINS:
            return ActionRisk.RISKY
        if binary == "git":
            subcmd = next((a for a in args_lower if not a.startswith("-")), "")
            if subcmd in _DESTRUCTIVE_GIT_SUBCOMMANDS or "--force" in args_lower:
                return ActionRisk.DESTRUCTIVE
            return ActionRisk.SAFE

        if "--force" in args_lower or "-rf" in args_lower:
            return ActionRisk.DESTRUCTIVE
        return ActionRisk.SAFE

    def check_action(
        self,
        *,
        action: str,
        risk: ActionRisk,
        summary: str,
    ) -> PermissionDecision:
        """Evaluate and audit an action without raising."""
        if risk == ActionRisk.SAFE:
            decision = PermissionDecision(
                allowed=True,
                risk=risk,
                action=action,
                summary=summary,
                reason="auto-approved SAFE action",
            )
            self.audit_log.append(decision)
            return decision

        approved = bool(self._approval_callback(summary, risk))
        decision = PermissionDecision(
            allowed=approved,
            risk=risk,
            action=action,
            summary=summary,
            reason=(
                f"operator approved {risk.value} action"
                if approved
                else f"denied {risk.value} action (fail-closed approval gate)"
            ),
        )
        self.audit_log.append(decision)
        return decision

    def require_approval(
        self,
        *,
        action: str,
        risk: ActionRisk,
        summary: str,
    ) -> PermissionDecision:
        """Enforce permission gate; raise ``PermissionDeniedError`` when denied."""
        decision = self.check_action(action=action, risk=risk, summary=summary)
        if not decision.allowed:
            raise PermissionDeniedError(f"Permission denied for {action}: {decision.reason}")
        return decision
