"""Idempotent ``archivist init`` IDE installer & ``archivist doctor`` (INIT-024/SPEC-009).

Supports idempotent MCP + passive hook configuration for:
- Cursor (``~/.cursor/mcp.json`` + ``.cursor/hooks.json``)
- Claude Code (``~/.claude.json`` + ``~/.claude/settings.json``)
- Codex (``~/.codex/config.toml``)
- OpenClaw (``~/.openclaw/mcp.json`` + ``~/.openclaw/hooks.json``)
- ``auto`` (detects installed IDE config directories; defaults to Cursor + Claude Code)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from archivist import __version__
from archivist.app.handlers.tools_docs import read_bundled_doc
from archivist.app.socket_security import default_socket_path
from archivist.features.local_onnx import get_model_cache_dir, verify_cached_models
from archivist.storage.local_store import default_local_db_path
from archivist.storage.metadata import read_embedding_metadata_sync

SUPPORTED_IDES = ("cursor", "claude", "codex", "openclaw")


def _default_mcp_entry(command: str = "uvx", args: list[str] | None = None) -> dict[str, Any]:
    return {
        "command": command,
        "args": list(args) if args is not None else ["archivist-mcp"],
    }


def _hook_command_prefix(command: str) -> str:
    """Resolve the hook CLI prefix aligned with the configured MCP launcher."""
    cmd = (command or "uvx").strip()
    base = Path(cmd).name.lower()
    if base in ("uvx", "uvx.exe"):
        return f"{cmd} --from archivist-mcp archivist hook"
    if base in ("archivist-mcp", "archivist-mcp.exe"):
        sibling = (
            str(Path(cmd).with_name("archivist")) if Path(cmd).parent != Path(".") else "archivist"
        )
        return f"{sibling} hook"
    if base in ("archivist", "archivist.exe"):
        return f"{cmd} hook"
    return "archivist hook"


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text via temp file + fsync + os.replace, refusing symlinks."""
    existing_mode: int | None = None
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        st = None

    if st is not None:
        if stat.S_ISLNK(st.st_mode):
            raise PermissionError(f"Refusing to overwrite symlink config file: {path}")
        if not stat.S_ISREG(st.st_mode):
            raise PermissionError(f"Refusing to overwrite non-regular config path: {path}")
        if os.name == "posix":
            existing_mode = stat.S_IMODE(st.st_mode)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        if existing_mode is not None and os.name == "posix":
            os.chmod(tmp_path, existing_mode)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _read_json_obj(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(parsed).__name__}")
    return parsed


def _write_json_idempotent(path: Path, data: dict[str, Any], *, dry_run: bool) -> bool:
    """Write canonical JSON atomically only if content actually changes. Returns True if changed."""
    serialized = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    if path.exists() and not path.is_symlink() and path.read_text(encoding="utf-8") == serialized:
        return False
    if not dry_run:
        _atomic_write_text(path, serialized)
    return True


def _upsert_json_mcp_server(
    path: Path,
    *,
    mcp_entry: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    doc = _read_json_obj(path)
    if "mcpServers" not in doc:
        servers: dict[str, Any] = {}
        doc["mcpServers"] = servers
    else:
        raw_servers = doc["mcpServers"]
        if not isinstance(raw_servers, dict):
            raise ValueError(
                f"Refusing to clobber non-object 'mcpServers' in {path} "
                f"(got {type(raw_servers).__name__})"
            )
        servers = raw_servers
    servers["archivist"] = mcp_entry
    changed = _write_json_idempotent(path, doc, dry_run=dry_run)
    return {"path": str(path), "changed": changed, "action": "upsert_mcp_server"}


def _upsert_hooks_json(
    path: Path,
    *,
    hook_style: str,
    hook_prefix: str = "archivist hook",
    dry_run: bool,
) -> dict[str, Any]:
    """Idempotently install read-only SessionStart + zero-LLM PreCompact/Stop hooks."""
    doc = _read_json_obj(path)
    if "hooks" not in doc:
        hooks: dict[str, Any] = {}
        doc["hooks"] = hooks
    else:
        raw_hooks = doc["hooks"]
        if not isinstance(raw_hooks, dict):
            raise ValueError(
                f"Refusing to clobber non-object 'hooks' in {path} (got {type(raw_hooks).__name__})"
            )
        hooks = raw_hooks

    if hook_style == "claude":
        desired: dict[str, list[dict[str, Any]]] = {
            "SessionStart": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f"{hook_prefix} session-start"}],
                }
            ],
            "PreCompact": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f"{hook_prefix} pre-compact"}],
                }
            ],
            "Stop": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f"{hook_prefix} stop"}],
                }
            ],
        }
        for event, entries in desired.items():
            if event not in hooks:
                existing: list[Any] = []
            else:
                raw_existing = hooks[event]
                if not isinstance(raw_existing, list):
                    raise ValueError(
                        f"Refusing to clobber non-list hooks[{event!r}] in {path} "
                        f"(got {type(raw_existing).__name__})"
                    )
                existing = raw_existing
            # Remove prior archivist hook entry if present, preserving sibling hooks.
            filtered = [
                item
                for item in existing
                if not (
                    isinstance(item, dict)
                    and any(
                        isinstance(h, dict) and "archivist hook" in str(h.get("command", ""))
                        for h in (item.get("hooks") or [])
                    )
                )
            ]
            filtered.extend(entries)
            hooks[event] = filtered
    else:
        # Cursor / OpenClaw hooks.json schema
        doc.setdefault("version", 1)
        desired_flat: dict[str, list[dict[str, str]]] = {
            "sessionStart": [{"command": f"{hook_prefix} session-start"}],
            "preCompact": [{"command": f"{hook_prefix} pre-compact"}],
            "stop": [{"command": f"{hook_prefix} stop"}],
        }
        for event, entries_flat in desired_flat.items():
            if event not in hooks:
                existing_flat: list[Any] = []
            else:
                raw_existing_flat = hooks[event]
                if not isinstance(raw_existing_flat, list):
                    raise ValueError(
                        f"Refusing to clobber non-list hooks[{event!r}] in {path} "
                        f"(got {type(raw_existing_flat).__name__})"
                    )
                existing_flat = raw_existing_flat
            filtered_flat = [
                item
                for item in existing_flat
                if not (isinstance(item, dict) and "archivist hook" in str(item.get("command", "")))
            ]
            filtered_flat.extend(entries_flat)
            hooks[event] = filtered_flat

    changed = _write_json_idempotent(path, doc, dry_run=dry_run)
    return {"path": str(path), "changed": changed, "action": "upsert_passive_hooks"}


def _upsert_codex_toml(
    path: Path,
    *,
    mcp_entry: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    """Idempotently upsert ``[mcp_servers.archivist]`` in Codex ``config.toml``."""
    existing_text = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing_text.strip():
        # Validate existing TOML before modifying so we never corrupt malformed user config.
        tomllib.loads(existing_text)

    args_toml = ", ".join(json.dumps(str(a)) for a in mcp_entry["args"])
    block = (
        "[mcp_servers.archivist]\n"
        f"command = {json.dumps(str(mcp_entry['command']))}\n"
        f"args = [{args_toml}]\n"
    )

    section_re = re.compile(r"(?ms)^\[mcp_servers\.archivist\]\s*\n.*?(?=^\[|\Z)")
    if section_re.search(existing_text):
        # Use callable replacement so backslash escapes in Windows paths are never mangled.
        new_text = section_re.sub(lambda _m: block, existing_text, count=1)
    else:
        sep = "\n" if existing_text and not existing_text.endswith("\n\n") else ""
        if existing_text and not existing_text.endswith("\n"):
            sep = "\n\n"
        new_text = existing_text + sep + block

    # Validate output TOML syntax before persisting.
    tomllib.loads(new_text)

    changed = new_text != existing_text
    if changed and not dry_run:
        _atomic_write_text(path, new_text)
    return {"path": str(path), "changed": changed, "action": "upsert_codex_toml"}


def init_ide(
    target: str = "auto",
    *,
    home_dir: Path | None = None,
    project_dir: Path | None = None,
    command: str = "uvx",
    args: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Configure Archivist MCP server & zero-LLM passive hooks for *target* IDE(s)."""
    home = Path(home_dir) if home_dir is not None else Path.home()
    proj = Path(project_dir) if project_dir is not None else Path.cwd()
    norm = (target or "auto").strip().lower()

    if norm == "auto":
        detected: list[str] = []
        if (home / ".cursor").exists():
            detected.append("cursor")
        if (home / ".claude").exists() or (home / ".claude.json").exists():
            detected.append("claude")
        if (home / ".codex").exists():
            detected.append("codex")
        if (home / ".openclaw").exists():
            detected.append("openclaw")
        targets = detected or ["cursor", "claude"]
    elif norm in SUPPORTED_IDES:
        targets = [norm]
    else:
        raise ValueError(
            f"Unsupported IDE target {target!r}; expected one of: {', '.join(SUPPORTED_IDES)}, auto"
        )

    mcp_entry = _default_mcp_entry(command=command, args=args)
    hook_prefix = _hook_command_prefix(command)
    changes: list[dict[str, Any]] = []

    for ide in targets:
        if ide == "cursor":
            changes.append(
                _upsert_json_mcp_server(
                    home / ".cursor" / "mcp.json",
                    mcp_entry=mcp_entry,
                    dry_run=dry_run,
                )
            )
            changes.append(
                _upsert_hooks_json(
                    proj / ".cursor" / "hooks.json",
                    hook_style="cursor",
                    hook_prefix=hook_prefix,
                    dry_run=dry_run,
                )
            )
        elif ide == "claude":
            changes.append(
                _upsert_json_mcp_server(
                    home / ".claude.json",
                    mcp_entry=mcp_entry,
                    dry_run=dry_run,
                )
            )
            changes.append(
                _upsert_hooks_json(
                    home / ".claude" / "settings.json",
                    hook_style="claude",
                    hook_prefix=hook_prefix,
                    dry_run=dry_run,
                )
            )
        elif ide == "codex":
            changes.append(
                _upsert_codex_toml(
                    home / ".codex" / "config.toml",
                    mcp_entry=mcp_entry,
                    dry_run=dry_run,
                )
            )
        elif ide == "openclaw":
            changes.append(
                _upsert_json_mcp_server(
                    home / ".openclaw" / "mcp.json",
                    mcp_entry=mcp_entry,
                    dry_run=dry_run,
                )
            )
            changes.append(
                _upsert_hooks_json(
                    home / ".openclaw" / "hooks.json",
                    hook_style="openclaw",
                    hook_prefix=hook_prefix,
                    dry_run=dry_run,
                )
            )

    return {
        "targets": targets,
        "dry_run": dry_run,
        "changes": changes,
        "any_changed": any(c["changed"] for c in changes),
    }


def run_doctor(
    *,
    db_path: Path | None = None,
    socket_path: Path | None = None,
    home_dir: Path | None = None,
    cache_dir: Path | None = None,
    expected_embed_sha256: str = "",
    expected_rerank_sha256: str = "",
) -> dict[str, Any]:
    """Run local health, permission, model cache SHA-256, DB integrity, and IDE config checks."""
    checks: list[dict[str, Any]] = []
    home = Path(home_dir) if home_dir is not None else Path.home()
    db = Path(db_path) if db_path is not None else default_local_db_path()
    sock = Path(socket_path) if socket_path is not None else default_socket_path()

    # 1. Python & bundled docs
    py_ok = sys.version_info >= (3, 12)
    checks.append(
        {
            "name": "python_runtime",
            "ok": py_ok,
            "detail": f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}, archivist {__version__}",
        }
    )

    docs_ok = read_bundled_doc("skill") is not None
    checks.append(
        {
            "name": "bundled_docs",
            "ok": docs_ok,
            "detail": "archivist.app.docs_data accessible via importlib.resources"
            if docs_ok
            else "Missing bundled reference docs",
        }
    )

    # 2. Core local runtime libraries (sqlite-vec, google-re2)
    for mod_name in ("sqlite_vec", "re2"):
        try:
            __import__(mod_name)
            checks.append({"name": f"dep:{mod_name}", "ok": True, "detail": "importable"})
        except Exception as exc:
            checks.append({"name": f"dep:{mod_name}", "ok": False, "detail": str(exc)})

    # 3. Model cache directory + ONNX SHA-256 integrity (MIN-4)
    model_cache = cache_dir if cache_dir is not None else get_model_cache_dir()
    embed_sha = expected_embed_sha256 or os.environ.get("ARCHIVIST_EMBED_SHA256", "")
    rerank_sha = expected_rerank_sha256 or os.environ.get("ARCHIVIST_RERANK_SHA256", "")
    model_report = verify_cached_models(
        model_cache,
        expected_embed_sha256=embed_sha,
        expected_rerank_sha256=rerank_sha,
    )
    cache_ok = model_cache.exists() and os.access(model_cache, os.W_OK) and bool(model_report["ok"])
    cache_detail = (
        "; ".join(model_report["errors"])
        if model_report["errors"]
        else (f"{model_cache} (embed_sha256={model_report['embed_sha256'] or 'not-cached'})")
    )
    checks.append(
        {
            "name": "onnx_model_cache",
            "ok": cache_ok,
            "detail": cache_detail,
        }
    )

    # 4. Unix socket type & 0600 permissions via lstat (MIN-4)
    try:
        sock_st = os.lstat(sock)
    except FileNotFoundError:
        sock_st = None

    if sock_st is not None:
        is_symlink = stat.S_ISLNK(sock_st.st_mode)
        is_sock = stat.S_ISSOCK(sock_st.st_mode)
        mode = stat.S_IMODE(sock_st.st_mode) if os.name == "posix" else 0o600
        sock_ok = (not is_symlink) and is_sock and (mode == 0o600)
        if is_symlink:
            detail = f"{sock} is a symlink (unsafe)"
        elif not is_sock:
            detail = f"{sock} is not a Unix domain socket"
        else:
            detail = f"{sock} mode={oct(mode)}" + ("" if sock_ok else " (expected 0o600)")
        checks.append({"name": "archivistd_socket", "ok": sock_ok, "detail": detail})
    else:
        checks.append(
            {
                "name": "archivistd_socket",
                "ok": True,
                "detail": f"{sock} (not currently running; auto-spawns on demand)",
            }
        )

    # 5. Local DB integrity, 0600 permissions & archivist_meta
    if db.exists():
        mode = stat.S_IMODE(os.stat(db).st_mode) if os.name == "posix" else 0o600
        perm_ok = mode == 0o600
        checks.append(
            {
                "name": "memory_db_permissions",
                "ok": perm_ok,
                "detail": f"{db} mode={oct(mode)}" + ("" if perm_ok else " (expected 0o600)"),
            }
        )
        try:
            conn = sqlite3.connect(str(db))
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            meta = read_embedding_metadata_sync(conn)
            conn.close()
            checks.append(
                {
                    "name": "memory_db_integrity",
                    "ok": integrity == "ok",
                    "detail": f"integrity={integrity}, embed_model={meta.embed_model if meta else 'unset'}, dim={meta.vector_dim if meta else 'unset'}",
                }
            )
        except Exception as exc:
            checks.append({"name": "memory_db_integrity", "ok": False, "detail": str(exc)})
    else:
        checks.append(
            {
                "name": "memory_db_integrity",
                "ok": True,
                "detail": f"{db} (will be created on first memory write)",
            }
        )

    # 6. IDE configs detected
    ide_configs = {
        "cursor": home / ".cursor" / "mcp.json",
        "claude": home / ".claude.json",
        "codex": home / ".codex" / "config.toml",
        "openclaw": home / ".openclaw" / "mcp.json",
    }
    configured = [name for name, p in ide_configs.items() if p.exists()]
    checks.append(
        {
            "name": "ide_configs",
            "ok": True,
            "detail": f"detected: {', '.join(configured) if configured else 'none (run archivist init)'}",
        }
    )

    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
    }
