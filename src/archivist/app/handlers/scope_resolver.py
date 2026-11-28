"""MCP ``roots/list`` + canonical git-remote auto-scoping (INIT-024/SPEC-008).

Derives canonical repository scopes of the form ``repo:<host>/<owner>/<name>``
from MCP ``roots/list`` URIs or ``cwd`` via ``git remote get-url origin`` so
coding-agent tool calls require zero manual ``agent_id`` or ``namespace`` args.
"""

from __future__ import annotations

import contextvars
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote, urlparse

from archivist.security.trust_tiers import RequestContext

GLOBAL_SCOPE = "user:global"

_SCP_REMOTE_RE = re.compile(r"^(?:[^@\s]+@)?([^:/\s]+):(.+)$")

# Per-task session context isolated across concurrent daemon connections.
_CURRENT_REQUEST_CONTEXT: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
    "archivist_request_context", default=None
)
_CURRENT_ROOTS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "archivist_session_roots", default=()
)


def set_request_context(ctx: RequestContext | None) -> contextvars.Token[RequestContext | None]:
    """Bind *ctx* for the current asyncio task."""
    return _CURRENT_REQUEST_CONTEXT.set(ctx)


def get_request_context() -> RequestContext:
    """Return active :class:`RequestContext`, defaulting to local OS UID context."""
    ctx = _CURRENT_REQUEST_CONTEXT.get()
    if ctx is not None:
        return ctx
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return RequestContext.for_local_uid(uid, source_client="local")


def set_session_roots(roots: Sequence[str]) -> contextvars.Token[tuple[str, ...]]:
    """Bind MCP ``roots/list`` URI strings for the current session."""
    return _CURRENT_ROOTS.set(tuple(str(r) for r in roots if str(r).strip()))


def get_session_roots() -> tuple[str, ...]:
    """Return bound MCP ``roots/list`` URI strings for the current session."""
    return _CURRENT_ROOTS.get()


def canonicalize_git_remote(remote_url: str) -> str | None:
    """Normalize SSH/HTTPS/SCP git remote URLs to ``repo:<host>/<owner>/<name>``.

    Strips embedded credentials (``user:pass@``), port numbers, trailing ``/``,
    and ``.git`` suffixes, and lowercases host and path segments so SSH and
    HTTPS remotes for the same repository produce identical scope IDs.
    """
    raw = (remote_url or "").strip()
    if not raw:
        return None

    host = ""
    path = ""

    if "://" in raw:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").strip().lower()
        path = (parsed.path or "").strip()
    else:
        match = _SCP_REMOTE_RE.match(raw)
        if not match:
            return None
        host = match.group(1).strip().lower()
        path = match.group(2).strip()

    if not host:
        return None

    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    path = path.strip("/").lower()
    if not path:
        return None

    return f"repo:{host}/{path}"


def _uri_to_local_path(uri_or_path: str) -> Path | None:
    raw = (uri_or_path or "").strip()
    if not raw:
        return None
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        path_str = unquote(parsed.path)
        if (
            os.name == "nt"
            and path_str.startswith("/")
            and len(path_str) > 2
            and path_str[2] == ":"
        ):
            path_str = path_str[1:]
        return Path(path_str)
    return Path(raw)


def git_remote_origin_for_path(path: Path) -> str | None:
    """Run ``git -C <path> remote get-url origin`` safely with a short timeout."""
    target = path if path.is_dir() else path.parent
    if not target.exists():
        return None
    git_bin = shutil.which("git")
    if not git_bin:
        return None
    try:
        proc = subprocess.run(
            [git_bin, "-C", str(target), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def resolve_write_scope(
    explicit_scope: str | None = None,
    *,
    roots: Sequence[str] | None = None,
    cwd: Path | str | None = None,
) -> str:
    """Resolve the single canonical target scope for a write operation."""
    if explicit_scope and explicit_scope.strip():
        return explicit_scope.strip()

    candidate_paths: list[Path] = []
    active_roots = roots if roots is not None else get_session_roots()
    for root_uri in active_roots:
        p = _uri_to_local_path(root_uri)
        if p is not None:
            candidate_paths.append(p)

    if cwd is not None:
        candidate_paths.append(Path(cwd))
    elif not candidate_paths:
        candidate_paths.append(Path.cwd())

    for path in candidate_paths:
        remote = git_remote_origin_for_path(path)
        if remote:
            canonical = canonicalize_git_remote(remote)
            if canonical:
                return canonical

    if candidate_paths:
        name = candidate_paths[0].resolve().name
        if name:
            safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower()
            if safe_name:
                return f"dir:{safe_name}"

    return GLOBAL_SCOPE


def resolve_read_scopes(
    explicit_scope: str | None = None,
    *,
    roots: Sequence[str] | None = None,
    cwd: Path | str | None = None,
    include_global: bool = True,
) -> list[str]:
    """Resolve read scopes, automatically blending canonical repo scope with ``user:global``."""
    if explicit_scope and explicit_scope.strip() == "*":
        return ["*"]

    primary = resolve_write_scope(explicit_scope, roots=roots, cwd=cwd)
    scopes = [primary]
    if include_global and primary != GLOBAL_SCOPE:
        scopes.append(GLOBAL_SCOPE)
    return scopes
