"""Frozen path-boundary enforcement for CRSI harness evolution.

Canonicalizes paths, blocks directory traversal and symlink escapes, enforces the
non-overridable system frozen deny-list plus dynamic frozen paths, and verifies
that candidate edits touch ONLY explicitly configured editable paths.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath

# Non-overridable system deny-list (Section 2.3 of INIT-001 plan).
SYSTEM_FROZEN_PATTERNS: tuple[str, ...] = (
    # Core Archivist packages (ADR-006 / GR-LAYER-001)
    "src/archivist/app/**",
    "src/archivist/core/**",
    "src/archivist/storage/**",
    "src/archivist/retrieval/**",
    "src/archivist/lifecycle/**",
    "src/archivist/write/**",
    "src/archivist/features/**",
    "src/archivist/utils/**",
    "src/archivist/__init__.py",
    # CRSI control plane, guards, SIM, LLM client, evaluator, and outer loop
    "src/archivist/crsi/__init__.py",
    "src/archivist/crsi/__main__.py",
    "src/archivist/crsi/cli.py",
    "src/archivist/crsi/config.py",
    "src/archivist/crsi/guards/**",
    "src/archivist/crsi/sim/**",
    "src/archivist/crsi/llm/**",
    "src/archivist/crsi/evaluator/**",
    "src/archivist/crsi/loop/**",
    # Frozen harness core (contract, permissions, sandbox)
    "src/archivist/crsi/harness/__init__.py",
    "src/archivist/crsi/harness/contract.py",
    "src/archivist/crsi/harness/permissions.py",
    "src/archivist/crsi/harness/workspace.py",
    # Frozen toy evaluator & tasks
    "src/archivist/crsi/toy/__init__.py",
    "src/archivist/crsi/toy/tasks.py",
    "src/archivist/crsi/toy/scorer.py",
    "src/archivist/crsi/toy/scripted_agents.py",
    # Repo metadata, CI, tests, scripts, and CRSI runtime state
    ".crsi/**",
    ".git/**",
    ".github/**",
    "pyproject.toml",
    "requirements*.txt",
    "Dockerfile*",
    ".pre-commit-config.yaml",
    "conftest.py",
    "tests/**",
    "scripts/**",
)


class PathPolicyViolation(ValueError):
    """Raised when one or more paths violate the CRSI editable/frozen boundary."""

    def __init__(self, message: str, violations: Sequence[PathCheckResult] = ()) -> None:
        super().__init__(message)
        self.violations: tuple[PathCheckResult, ...] = tuple(violations)


@dataclass(frozen=True)
class PathCheckResult:
    """Outcome of checking a single path against ``PathPolicyGuard``."""

    raw_path: str
    normalized_path: str
    allowed: bool
    reason: str | None = None


def _strip_curdir_prefix(posix_path: str) -> str:
    """Strip only repeated leading ``'./'`` segments (preserving dotfiles and leading '/')."""
    cleaned = posix_path
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


@lru_cache(maxsize=512)
def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a POSIX glob pattern supporting ``*``, ``?``, and ``**`` segments."""
    norm = _strip_curdir_prefix(pattern.strip().replace("\\", "/"))
    if not norm:
        return re.compile(r"^$")

    parts: list[str] = ["^"]
    i = 0
    n = len(norm)
    while i < n:
        ch = norm[i]
        if ch == "*":
            if i + 1 < n and norm[i + 1] == "*":
                if i + 2 < n and norm[i + 2] == "/":
                    # ``**/`` matches zero or more path segments.
                    parts.append("(?:.+/)?")
                    i += 3
                else:
                    # Trailing ``**`` matches any remainder including nested slashes.
                    parts.append(".*")
                    i += 2
            else:
                parts.append("[^/]*")
                i += 1
        elif ch == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(ch))
            i += 1
    parts.append("$")
    return re.compile("".join(parts))


def glob_matches(posix_path: str, pattern: str) -> bool:
    """Return True if ``posix_path`` matches the POSIX glob ``pattern``."""
    cleaned_path = _strip_curdir_prefix(posix_path.strip().replace("\\", "/"))
    cleaned_pat = _strip_curdir_prefix(pattern.strip().replace("\\", "/"))
    if not cleaned_path or not cleaned_pat:
        return False
    if _compile_glob(cleaned_pat).match(cleaned_path):
        return True
    # Treat directory prefix patterns like ``dir/**`` as also denying the directory itself.
    return cleaned_pat.endswith("/**") and cleaned_path == cleaned_pat[:-3]


def _has_glob_meta(pattern: str) -> bool:
    return any(ch in pattern for ch in ("*", "?", "["))


class PathPolicyGuard:
    """Enforces editable-path allowlist and non-overridable frozen deny-list."""

    def __init__(
        self,
        repo_root: Path,
        editable_patterns: Sequence[str],
        *,
        extra_frozen_patterns: Sequence[str] = (),
        dynamic_frozen_paths: Sequence[str | Path] = (),
        private_dir: Path | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self._raw_editable_patterns: tuple[str, ...] = tuple(
            p.strip().replace("\\", "/") for p in editable_patterns if p.strip()
        )
        self.editable_patterns: tuple[str, ...] = tuple(
            _strip_curdir_prefix(p) for p in self._raw_editable_patterns
        )
        self.private_dir = private_dir.resolve() if private_dir is not None else None

        frozen: list[str] = list(SYSTEM_FROZEN_PATTERNS)
        for pat in extra_frozen_patterns:
            cleaned = _strip_curdir_prefix(pat.strip().replace("\\", "/"))
            if cleaned:
                frozen.append(cleaned)
        for dyn in dynamic_frozen_paths:
            rel = self._try_relativize(dyn)
            if rel is not None:
                frozen.append(rel)
        self.frozen_patterns: tuple[str, ...] = tuple(dict.fromkeys(frozen))

    def _try_relativize(self, raw: str | Path) -> str | None:
        candidate = Path(raw)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(self.repo_root).as_posix()
            except ValueError:
                return None
        posix = str(raw).strip().replace("\\", "/")
        if not posix or posix.startswith("/"):
            return None
        pure = PurePosixPath(posix)
        if ".." in pure.parts:
            return None
        norm = _strip_curdir_prefix(pure.as_posix())
        return None if not norm or norm == "." else norm

    def _normalize_and_check_containment(
        self, raw_path: str | Path
    ) -> tuple[str | None, str | None]:
        """Return ``(normalized_posix_rel_path, error_reason)``."""
        raw_str = str(raw_path)
        if not raw_str or not raw_str.strip():
            return None, "empty path"
        if "\x00" in raw_str:
            return None, "NUL byte in path"

        posix_str = raw_str.replace("\\", "/")
        candidate = Path(raw_str)

        if candidate.is_absolute():
            resolved = candidate.resolve()
            if self.private_dir is not None and resolved.is_relative_to(self.private_dir):
                return None, f"path falls inside frozen private_dir ({self.private_dir})"
            try:
                rel = resolved.relative_to(self.repo_root)
            except ValueError:
                return None, f"absolute path escapes repo root: {raw_str}"
            rel_posix = rel.as_posix()
        else:
            pure = PurePosixPath(posix_str)
            if ".." in pure.parts:
                return None, f"path traversal ('..') is forbidden: {raw_str}"
            if pure.is_absolute():
                return None, f"absolute path is forbidden: {raw_str}"
            rel_posix = _strip_curdir_prefix(pure.as_posix())
            if not rel_posix or rel_posix == ".":
                return None, f"invalid repository-relative path: {raw_str}"

        # Check every existing ancestor and the target itself for symlink escapes.
        target_on_disk = self.repo_root / rel_posix
        current = self.repo_root
        for part in PurePosixPath(rel_posix).parts:
            current = current / part
            if current.is_symlink():
                resolved_link = current.resolve()
                if self.private_dir is not None and resolved_link.is_relative_to(self.private_dir):
                    return None, f"symlink escapes into private_dir: {rel_posix}"
                if not resolved_link.is_relative_to(self.repo_root):
                    return None, f"symlink escapes repo root: {rel_posix}"
        resolved_target = target_on_disk.resolve()
        if self.private_dir is not None and resolved_target.is_relative_to(self.private_dir):
            return None, f"resolved path falls inside private_dir: {rel_posix}"
        if not resolved_target.is_relative_to(self.repo_root):
            return None, f"resolved path escapes repo root: {rel_posix}"

        # Also check canonical relative path (in case an internal symlink points to a frozen file).
        canonical_rel = resolved_target.relative_to(self.repo_root).as_posix()
        if canonical_rel != rel_posix:
            if self._matches_frozen(canonical_rel):
                return (
                    canonical_rel,
                    f"symlink target {canonical_rel!r} points to a frozen path",
                )
            return canonical_rel, None

        return rel_posix, None

    def _matches_frozen(self, posix_path: str) -> str | None:
        for pat in self.frozen_patterns:
            if glob_matches(posix_path, pat):
                return pat
        return None

    def _matches_editable(self, posix_path: str) -> str | None:
        for pat in self.editable_patterns:
            if glob_matches(posix_path, pat):
                return pat
        return None

    def is_frozen(self, raw_path: str | Path) -> bool:
        """Return True if ``raw_path`` is frozen or escapes the repository."""
        norm, err = self._normalize_and_check_containment(raw_path)
        if err is not None or norm is None:
            return True
        return self._matches_frozen(norm) is not None

    def check_path(self, raw_path: str | Path) -> PathCheckResult:
        """Evaluate whether ``raw_path`` is legal for a proposer/candidate to edit."""
        raw_str = str(raw_path)
        norm, err = self._normalize_and_check_containment(raw_path)
        if err is not None or norm is None:
            return PathCheckResult(
                raw_path=raw_str,
                normalized_path=norm or raw_str,
                allowed=False,
                reason=err or "path failed containment check",
            )

        frozen_pat = self._matches_frozen(norm)
        if frozen_pat is not None:
            return PathCheckResult(
                raw_path=raw_str,
                normalized_path=norm,
                allowed=False,
                reason=f"path matches frozen deny-list pattern {frozen_pat!r}",
            )

        editable_pat = self._matches_editable(norm)
        if editable_pat is None:
            return PathCheckResult(
                raw_path=raw_str,
                normalized_path=norm,
                allowed=False,
                reason="path is outside configured editable allowlist",
            )

        return PathCheckResult(
            raw_path=raw_str,
            normalized_path=norm,
            allowed=True,
            reason=None,
        )

    def is_editable(self, raw_path: str | Path) -> bool:
        """Return True if and only if ``raw_path`` is allowed to be edited."""
        return self.check_path(raw_path).allowed

    def assert_all_editable(self, paths: Sequence[str | Path]) -> list[str]:
        """Verify every path in ``paths`` is editable; return normalized paths or raise."""
        normalized: list[str] = []
        violations: list[PathCheckResult] = []
        for path in paths:
            result = self.check_path(path)
            if result.allowed:
                normalized.append(result.normalized_path)
            else:
                violations.append(result)
        if violations:
            detail = "; ".join(f"{v.raw_path} ({v.reason})" for v in violations)
            raise PathPolicyViolation(
                f"Illegal edit to non-editable or frozen path(s): {detail}",
                violations=violations,
            )
        return normalized

    def validate_configuration(self, command_argvs: Sequence[Sequence[str]] = ()) -> list[str]:
        """Return a list of configuration errors (empty means valid).

        Checks that:
        1. ``editable_patterns`` is non-empty and contains no traversal/absolute patterns.
        2. No literal editable path directly targets a frozen pattern.
        3. No file referenced by ``commands.run_task`` / ``commands.check`` falls inside
           ``editable_patterns`` (prevents scorer self-modification).
        """
        errors: list[str] = []
        if not self._raw_editable_patterns:
            errors.append("harness.editable must specify at least one editable path or glob.")

        for raw_pat in self._raw_editable_patterns:
            if raw_pat.startswith("/") or ".." in PurePosixPath(raw_pat).parts:
                errors.append(f"Invalid editable pattern {raw_pat!r}: must be repo-relative.")
                continue
            pat = _strip_curdir_prefix(raw_pat)
            if not pat or pat == ".":
                errors.append(f"Invalid editable pattern {raw_pat!r}: must not be repo root.")
                continue
            if not _has_glob_meta(pat):
                frozen_hit = self._matches_frozen(pat)
                if frozen_hit is not None:
                    errors.append(
                        f"Editable path {pat!r} conflicts with frozen pattern {frozen_hit!r}."
                    )

        for argv in command_argvs:
            for token in argv:
                if not token or token.startswith("-") or "{" in token:
                    continue
                if "/" not in token and "." not in token:
                    continue
                candidate = Path(token)
                rel = self._try_relativize(candidate)
                if rel is None:
                    continue
                # Only treat as a target file if it looks like a script/file in the repo
                # (exists on disk or matches editable directly).
                if (
                    self._matches_editable(rel) is not None
                    and self._matches_frozen(rel) is None
                    and ((self.repo_root / rel).exists() or os.path.splitext(rel)[1] != "")
                ):
                    errors.append(
                        f"Frozen command target {rel!r} falls inside harness.editable; "
                        "scorers and checks must stay in frozen paths."
                    )

        return errors
