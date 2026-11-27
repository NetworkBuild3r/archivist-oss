"""Isolated git worktree sandbox manager enforcing dedicated run branches (never main)."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from archivist.crsi.guards.secrets import sanitize_env

_PROTECTED_BRANCHES: frozenset[str] = frozenset({"main", "master"})


class GitWorktreeError(RuntimeError):
    """Raised when a git worktree operation fails or violates branch-isolation rules."""


class GitWorktreeManager:
    """Manages isolated git worktrees and commits accepted candidates to ``crsi/<run_id>``."""

    def __init__(
        self,
        repo_root: Path,
        *,
        worktree_root: Path,
        base_branch: str = "main",
        branch_prefix: str = "crsi/",
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.worktree_root = worktree_root.resolve()
        self.base_branch = base_branch.strip()
        self.branch_prefix = branch_prefix.strip()
        self._git_exe = shutil.which("git")
        if self._git_exe is None:
            raise GitWorktreeError("git executable not found on PATH.")
        self._active_worktrees: list[Path] = []

    def _run_git(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        assert self._git_exe is not None
        cmd = [self._git_exe, *args]
        proc = subprocess.run(
            cmd,
            cwd=str((cwd if cwd is not None else self.repo_root).resolve()),
            env=sanitize_env(),
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
        if check and proc.returncode != 0:
            raise GitWorktreeError(
                f"git {' '.join(args)} failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc

    def assert_safe_run_branch(self, run_branch: str) -> str:
        """Enforce hard governance rule: never target ``main``, ``master``, or ``base_branch``."""
        cleaned = run_branch.strip()
        if not cleaned:
            raise GitWorktreeError("Run branch name must be non-empty.")
        if cleaned in _PROTECTED_BRANCHES or cleaned == self.base_branch:
            raise GitWorktreeError(
                f"Refusing to mutate protected/base branch {cleaned!r}; "
                f"CRSI runs must use a dedicated '{self.branch_prefix}*' branch."
            )
        if not cleaned.startswith(self.branch_prefix):
            raise GitWorktreeError(
                f"Run branch {cleaned!r} must start with configured prefix {self.branch_prefix!r}."
            )
        return cleaned

    def rev_parse(self, ref: str = "HEAD") -> str:
        """Resolve ``ref`` to a full 40-character commit SHA."""
        proc = self._run_git(["rev-parse", "--verify", ref])
        return proc.stdout.strip()

    def ensure_run_branch(self, run_branch: str, *, base_ref: str = "HEAD") -> str:
        """Ensure ``refs/heads/<run_branch>`` exists (creating from ``base_ref`` if needed)."""
        safe_branch = self.assert_safe_run_branch(run_branch)
        check = self._run_git(
            ["rev-parse", "--verify", f"refs/heads/{safe_branch}"],
            check=False,
        )
        if check.returncode == 0:
            return check.stdout.strip()
        base_sha = self.rev_parse(base_ref)
        self._run_git(["branch", safe_branch, base_sha])
        return base_sha

    def checkout_detached_worktree(self, name: str, commit_sha: str) -> Path:
        """Create a clean detached git worktree at ``commit_sha``."""
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        wt_path = (self.worktree_root / safe_name).resolve()
        if wt_path.exists():
            self.remove_worktree(wt_path)
        self._run_git(["worktree", "add", "--detach", str(wt_path), commit_sha])
        self._active_worktrees.append(wt_path)
        return wt_path

    def reset_and_clean(self, wt_path: Path, commit_sha: str) -> None:
        """Deterministically reset worktree to ``commit_sha`` and remove all untracked files."""
        self._run_git(["reset", "--hard", commit_sha], cwd=wt_path)
        self._run_git(["clean", "-fdx"], cwd=wt_path)

    def apply_file_edits(
        self,
        wt_path: Path,
        file_edits: Mapping[str, str],
        delete_files: Sequence[str] = (),
    ) -> list[str]:
        """Apply file writes/deletions inside ``wt_path`` and return touched repo-relative paths."""
        resolved_wt = wt_path.resolve()
        touched_explicit: list[str] = []

        for rel_path, content in file_edits.items():
            target = self._resolve_worktree_target(resolved_wt, rel_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            touched_explicit.append(target.relative_to(resolved_wt).as_posix())

        for rel_path in delete_files:
            target = self._resolve_worktree_target(resolved_wt, rel_path)
            if target.exists() and target.is_file():
                target.unlink()
                touched_explicit.append(target.relative_to(resolved_wt).as_posix())

        return self.list_changed_files(resolved_wt, fallback_paths=touched_explicit)

    def _resolve_worktree_target(self, resolved_wt: Path, rel_path: str) -> Path:
        raw = rel_path.replace("\\", "/")
        if not raw.strip() or "\x00" in raw:
            raise GitWorktreeError(f"Invalid relative edit path: {rel_path!r}")
        pure = PurePosixPath(raw)
        if pure.is_absolute() or ".." in pure.parts:
            raise GitWorktreeError(f"Illegal path traversal in edit path: {rel_path!r}")
        norm = pure.as_posix().lstrip("./")
        target = (resolved_wt / norm).resolve()
        if not target.is_relative_to(resolved_wt):
            raise GitWorktreeError(f"Edit path escapes worktree root: {rel_path!r}")
        return target

    def list_changed_files(
        self,
        wt_path: Path,
        *,
        fallback_paths: Sequence[str] = (),
    ) -> list[str]:
        """Return sorted repo-relative paths modified, added, or deleted in ``wt_path``."""
        proc = self._run_git(["status", "--porcelain", "-uall"], cwd=wt_path)
        changed: list[str] = []
        for line in proc.stdout.splitlines():
            if len(line) < 4:
                continue
            path_part = line[3:].strip()
            if " -> " in path_part:
                path_part = path_part.split(" -> ", 1)[1].strip()
            if path_part.startswith('"') and path_part.endswith('"'):
                path_part = path_part[1:-1]
            if path_part:
                changed.append(path_part)
        if not changed and fallback_paths:
            return sorted(set(fallback_paths))
        return sorted(set(changed))

    def diff_against_head(self, wt_path: Path) -> str:
        """Return unified diff of tracked changes plus headers for untracked files."""
        proc = self._run_git(["diff", "HEAD"], cwd=wt_path)
        parts: list[str] = [proc.stdout] if proc.stdout else []

        untracked = self._run_git(
            ["ls-files", "--others", "--exclude-standard"],
            cwd=wt_path,
        )
        for rel in untracked.stdout.splitlines():
            rel = rel.strip()
            if not rel:
                continue
            full = wt_path / rel
            if full.is_file():
                body = full.read_text(encoding="utf-8", errors="replace")
                parts.append(f"+++ new file: {rel}\n{body}\n")
        return "\n".join(parts).strip()

    def commit_worktree_changes(self, wt_path: Path, *, message: str) -> str:
        """Stage all changes in ``wt_path``, commit to detached HEAD, and return SHA."""
        self._run_git(["add", "-A"], cwd=wt_path)
        status = self._run_git(["status", "--porcelain"], cwd=wt_path)
        if not status.stdout.strip():
            return self.rev_parse_in(wt_path, "HEAD")
        self._run_git(
            [
                "-c",
                "user.name=CRSI Loop",
                "-c",
                "user.email=crsi@local.invalid",
                "commit",
                "-m",
                message,
            ],
            cwd=wt_path,
        )
        return self.rev_parse_in(wt_path, "HEAD")

    def rev_parse_in(self, wt_path: Path, ref: str = "HEAD") -> str:
        proc = self._run_git(["rev-parse", "--verify", ref], cwd=wt_path)
        return proc.stdout.strip()

    def advance_run_branch(self, run_branch: str, target_sha: str) -> None:
        """Advance ONLY ``refs/heads/<run_branch>`` to ``target_sha`` (never base/main)."""
        safe_branch = self.assert_safe_run_branch(run_branch)
        self._run_git(["update-ref", f"refs/heads/{safe_branch}", target_sha])

    def remove_worktree(self, wt_path: Path) -> None:
        """Remove a single managed worktree cleanly."""
        resolved = wt_path.resolve()
        self._run_git(["worktree", "remove", "--force", str(resolved)], check=False)
        if resolved.exists():
            shutil.rmtree(resolved, ignore_errors=True)
        self._active_worktrees = [p for p in self._active_worktrees if p != resolved]

    def cleanup_all(self) -> None:
        """Remove all worktrees created by this manager and prune git metadata."""
        for wt_path in list(self._active_worktrees):
            self.remove_worktree(wt_path)
        self._run_git(["worktree", "prune"], check=False)
