"""FROZEN sandboxed workspace manager with strict path containment."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import TracebackType


class WorkspaceSecurityError(PermissionError):
    """Raised when a file operation attempts to escape the task workspace sandbox."""


class WorkspaceManager:
    """Per-task temporary sandbox enforcing strict workspace containment."""

    def __init__(self, root: Path | None = None, *, auto_cleanup: bool = True) -> None:
        self._owns_root = root is None
        self._auto_cleanup = auto_cleanup
        if root is None:
            self.root = Path(tempfile.mkdtemp(prefix="crsi_ws_")).resolve()
        else:
            root.mkdir(parents=True, exist_ok=True)
            self.root = root.resolve()

    def __enter__(self) -> WorkspaceManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._owns_root and self._auto_cleanup:
            self.cleanup()

    def cleanup(self) -> None:
        """Delete the temporary workspace directory if owned by this manager."""
        if self._owns_root and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def resolve_path(self, rel_path: str | Path) -> Path:
        """Resolve ``rel_path`` inside ``self.root``, rejecting traversal and symlink escapes."""
        raw = str(rel_path)
        if not raw or not raw.strip():
            raise WorkspaceSecurityError("Empty workspace path is forbidden.")
        if "\x00" in raw:
            raise WorkspaceSecurityError("NUL byte in workspace path is forbidden.")

        posix = raw.replace("\\", "/")
        candidate = Path(raw)
        if candidate.is_absolute():
            raise WorkspaceSecurityError(f"Absolute path is forbidden in workspace: {raw!r}")

        pure = PurePosixPath(posix)
        if pure.is_absolute() or ".." in pure.parts:
            raise WorkspaceSecurityError(f"Path traversal is forbidden in workspace: {raw!r}")

        norm_rel = pure.as_posix().lstrip("./")
        if not norm_rel or norm_rel == ".":
            raise WorkspaceSecurityError(f"Invalid relative workspace path: {raw!r}")

        current = self.root
        for part in PurePosixPath(norm_rel).parts:
            current = current / part
            if current.is_symlink():
                resolved_link = current.resolve()
                if not resolved_link.is_relative_to(self.root):
                    raise WorkspaceSecurityError(f"Symlink escapes workspace sandbox: {raw!r}")

        target = (self.root / norm_rel).resolve()
        if not target.is_relative_to(self.root):
            raise WorkspaceSecurityError(f"Resolved path escapes workspace sandbox: {raw!r}")
        return target

    def seed_files(self, files: Mapping[str, str]) -> None:
        """Populate initial task input files inside the workspace."""
        for rel_path, content in files.items():
            self.write_file(rel_path, content)

    def read_file(self, rel_path: str) -> str:
        """Read a UTF-8 text file from inside the workspace."""
        target = self.resolve_path(rel_path)
        if not target.is_file():
            raise FileNotFoundError(f"Workspace file not found: {rel_path}")
        return target.read_text(encoding="utf-8")

    def write_file(self, rel_path: str, content: str) -> Path:
        """Write a UTF-8 text file inside the workspace, creating parent dirs as needed."""
        target = self.resolve_path(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def list_files(self) -> list[str]:
        """Return sorted workspace-relative POSIX paths of contained regular files."""
        results: list[str] = []
        for entry in sorted(self.root.rglob("*")):
            if entry.is_symlink():
                if not entry.resolve().is_relative_to(self.root):
                    continue
            if entry.is_file():
                results.append(entry.relative_to(self.root).as_posix())
        return results

    def snapshot_artifacts(self, *, max_file_bytes: int = 65_536) -> dict[str, str]:
        """Return UTF-8 text contents of workspace files up to ``max_file_bytes`` each."""
        artifacts: dict[str, str] = {}
        for rel in self.list_files():
            path = self.root / rel
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if len(raw) > max_file_bytes:
                artifacts[rel] = raw[:max_file_bytes].decode("utf-8", errors="replace")
            else:
                artifacts[rel] = raw.decode("utf-8", errors="replace")
        return artifacts
