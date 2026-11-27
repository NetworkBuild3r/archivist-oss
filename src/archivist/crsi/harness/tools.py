"""EDITABLE file I/O and safe command runner tools for the seed harness."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from archivist.crsi.guards.secrets import SecretRedactor, sanitize_env
from archivist.crsi.harness.contract import CommandResult
from archivist.crsi.harness.permissions import PermissionGate
from archivist.crsi.harness.workspace import WorkspaceManager


def _decode_stream(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


class ToolRuntime:
    """Sandboxed file I/O and ``shell=False`` command runner backed by frozen guards."""

    def __init__(
        self,
        workspace: WorkspaceManager,
        permission_gate: PermissionGate,
        *,
        command_timeout_s: float = 30.0,
        redactor: SecretRedactor | None = None,
        base_environ: Mapping[str, str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.permission_gate = permission_gate
        self.command_timeout_s = command_timeout_s
        self.redactor = redactor if redactor is not None else SecretRedactor()
        self._base_environ = base_environ

    def read_file(self, path: str) -> str:
        """Read a UTF-8 file inside the workspace after permission & containment checks."""
        risk = self.permission_gate.classify_file_action("read_file", path)
        self.permission_gate.require_approval(
            action="read_file",
            risk=risk,
            summary=f"read_file({path})",
        )
        content = self.workspace.read_file(path)
        return self.redactor.redact_text(content)

    def write_file(self, path: str, content: str) -> str:
        """Write a UTF-8 file inside the workspace after permission & containment checks."""
        risk = self.permission_gate.classify_file_action("write_file", path)
        self.permission_gate.require_approval(
            action="write_file",
            risk=risk,
            summary=f"write_file({path}, {len(content)} bytes)",
        )
        self.workspace.write_file(path, content)
        return f"Wrote {len(content)} chars to {path}"

    def list_files(self) -> list[str]:
        """List files contained inside the workspace."""
        risk = self.permission_gate.classify_file_action("list_files", ".")
        self.permission_gate.require_approval(
            action="list_files",
            risk=risk,
            summary="list_files()",
        )
        return self.workspace.list_files()

    def _resolve_executable(self, exe: str) -> str | None:
        candidate = Path(exe)
        if candidate.is_absolute():
            return str(candidate) if candidate.is_file() else None
        return shutil.which(exe)

    def run_command(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float | None = None,
    ) -> CommandResult:
        """Execute ``argv`` with ``shell=False``, resolved executable, and sanitized env."""
        argv_list = [str(part) for part in argv]
        risk = self.permission_gate.classify_command(argv_list)
        self.permission_gate.require_approval(
            action="run_command",
            risk=risk,
            summary=f"run_command({' '.join(argv_list)})",
        )

        resolved_exe = self._resolve_executable(argv_list[0])
        if resolved_exe is None:
            return CommandResult(
                argv=argv_list,
                exit_code=127,
                stdout="",
                stderr=f"Executable not found: {argv_list[0]}",
                timed_out=False,
                duration_ms=0.0,
            )

        effective_timeout = timeout_s if timeout_s is not None else self.command_timeout_s
        child_env = sanitize_env(self._base_environ)
        child_env["PWD"] = str(self.workspace.root)
        cmd = [resolved_exe, *argv_list[1:]]

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.workspace.root),
                env=child_env,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
                shell=False,
            )
            duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return CommandResult(
                argv=argv_list,
                exit_code=int(proc.returncode),
                stdout=self.redactor.redact_text(proc.stdout or ""),
                stderr=self.redactor.redact_text(proc.stderr or ""),
                timed_out=False,
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            stdout = self.redactor.redact_text(_decode_stream(exc.stdout))
            stderr = self.redactor.redact_text(_decode_stream(exc.stderr))
            timeout_note = f"Command timed out after {effective_timeout:.2f}s"
            combined_stderr = f"{stderr}\n{timeout_note}".strip() if stderr else timeout_note
            return CommandResult(
                argv=argv_list,
                exit_code=-1,
                stdout=stdout,
                stderr=combined_stderr,
                timed_out=True,
                duration_ms=duration_ms,
            )
