"""Unified Archivist v3 CLI entrypoint (INIT-024/SPEC-005, SPEC-007, SPEC-009).

Keeps top-level imports strictly to Python stdlib + ``archivist.__version__``
so ``archivist mcp`` / ``archivist-mcp`` and ``archivist --version`` never
import ONNX, numpy, or MCP server internals on the hot ``stdio`` shim path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from archivist import __version__


def mcp_entrypoint(argv: list[str] | None = None) -> int:
    """Console script entrypoint ``archivist-mcp`` — ultra-fast ``stdio`` shim."""
    from archivist.app.stdio_shim import main as shim_main

    return shim_main(argv)


def _open_store(db_path: Path | None, *, offline_embed: bool = False) -> Any:
    from archivist.storage.local_store import LocalMemoryStore

    backend = os.environ.get("ARCHIVIST_LOCAL_VECTOR_BACKEND", "sqlite-vec").strip()
    offline = offline_embed or os.environ.get(
        "ARCHIVIST_LOCAL_EMBED_OFFLINE", ""
    ).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    return LocalMemoryStore(
        db_path=db_path,
        vector_backend=backend,
        offline_deterministic=offline,
    )


def _cmd_mcp(args: argparse.Namespace) -> int:
    from archivist.app.stdio_shim import run_stdio_bridge

    return run_stdio_bridge(
        socket_path=args.socket,
        db_path=args.db,
        embed_url=args.embed_url or "",
        offline_embed=bool(args.offline_embed),
    )


def _cmd_daemon(args: argparse.Namespace) -> int:
    from archivist.app.daemon import main as daemon_main

    forwarded: list[str] = []
    if args.socket:
        forwarded.extend(["--socket", str(args.socket)])
    if args.db:
        forwarded.extend(["--db", str(args.db)])
    if args.embed_url:
        forwarded.extend(["--embed-url", str(args.embed_url)])
    if args.vector_backend:
        forwarded.extend(["--vector-backend", str(args.vector_backend)])
    if args.offline_embed:
        forwarded.append("--offline-embed")
    return daemon_main(forwarded)


def _cmd_init(args: argparse.Namespace) -> int:
    from archivist.cli.init_cmd import init_ide

    res = init_ide(
        target=args.ide,
        home_dir=args.home,
        project_dir=args.project_dir,
        command=args.command,
        dry_run=bool(args.dry_run),
    )
    sys.stdout.write(json.dumps(res, indent=2) + "\n")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from archivist.cli.init_cmd import run_doctor

    res = run_doctor(db_path=args.db, socket_path=args.socket, home_dir=args.home)
    if args.json:
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
    else:
        status = "OK" if res["ok"] else "ISSUES DETECTED"
        sys.stdout.write(f"Archivist Doctor ({res['version']}): {status}\n")
        for chk in res["checks"]:
            mark = "PASS" if chk["ok"] else "FAIL"
            sys.stdout.write(f"  [{mark}] {chk['name']}: {chk['detail']}\n")
    return 0 if res["ok"] else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    from archivist.cli.inspect_cmd import export_shared_jsonl, render_memories_markdown

    store = _open_store(args.db, offline_embed=bool(args.offline_embed))
    try:
        if args.export:
            res = export_shared_jsonl(store, Path(args.export), scope=args.scope)
            sys.stdout.write(json.dumps(res, indent=2) + "\n")
            return 0
        inv = store.inspect_memories(
            scope=args.scope,
            memory_id=args.memory_id,
            limit=args.limit,
        )
        if args.format == "json":
            sys.stdout.write(json.dumps(inv, indent=2, ensure_ascii=False) + "\n")
        else:
            sys.stdout.write(render_memories_markdown(inv))
        return 0
    finally:
        store.close()


def _cmd_import(args: argparse.Namespace) -> int:
    from archivist.cli.import_cmd import import_source

    store = _open_store(args.db, offline_embed=bool(args.offline_embed))
    try:
        res = import_source(
            store,
            Path(args.source),
            fmt=args.format,
            scope=args.scope,
        )
        sys.stdout.write(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
        return 0
    finally:
        store.close()


def _cmd_hook(args: argparse.Namespace) -> int:
    from archivist.cli.hooks import run_passive_capture_hook, run_session_start_hook
    from archivist.security.trust_tiers import RequestContext

    store = _open_store(args.db, offline_embed=bool(args.offline_embed))
    try:
        if args.event == "session-start":
            res = run_session_start_hook(store, scope=args.scope, max_tokens=args.max_tokens)
        else:
            payload = args.input_text
            if payload is None and not sys.stdin.isatty():
                payload = sys.stdin.read()
            uid = os.getuid() if hasattr(os, "getuid") else 0
            ctx = RequestContext.for_local_uid(uid, source_client=f"hook-{args.event}")
            hook_label = "PreCompact" if args.event == "pre-compact" else "Stop"
            res = run_passive_capture_hook(
                store,
                hook_name=hook_label,
                payload_text=payload or "",
                scope=args.scope,
                ctx=ctx,
            )
        sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
        return 0
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archivist",
        description="Archivist v3 — single-file local & fleet memory server for AI coding agents",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"archivist {__version__}",
    )
    sub = parser.add_subparsers(dest="subcommand")

    # archivist mcp
    p_mcp = sub.add_parser("mcp", help="Run lightweight stdio MCP shim connecting to archivistd")
    p_mcp.add_argument("--socket", type=Path, default=None)
    p_mcp.add_argument("--db", type=Path, default=None)
    p_mcp.add_argument("--embed-url", default="")
    p_mcp.add_argument("--offline-embed", action="store_true")
    p_mcp.set_defaults(func=_cmd_mcp)

    # archivist daemon
    p_daemon = sub.add_parser("daemon", help="Run local archivistd Unix socket daemon")
    p_daemon.add_argument("--socket", type=Path, default=None)
    p_daemon.add_argument("--db", type=Path, default=None)
    p_daemon.add_argument("--embed-url", default="")
    p_daemon.add_argument(
        "--vector-backend", choices=["sqlite-vec", "usearch"], default="sqlite-vec"
    )
    p_daemon.add_argument("--offline-embed", action="store_true")
    p_daemon.set_defaults(func=_cmd_daemon)

    # archivist init
    p_init = sub.add_parser("init", help="Install Archivist MCP config and passive hooks for IDEs")
    p_init.add_argument(
        "ide", nargs="?", default="auto", choices=["cursor", "claude", "codex", "openclaw", "auto"]
    )
    p_init.add_argument("--home", type=Path, default=None)
    p_init.add_argument("--project-dir", type=Path, default=None)
    p_init.add_argument("--command", default="uvx")
    p_init.add_argument("--dry-run", action="store_true")
    p_init.set_defaults(func=_cmd_init)

    # archivist doctor
    p_doc = sub.add_parser(
        "doctor", help="Check Python, ONNX cache, socket 0600, DB, and IDE configs"
    )
    p_doc.add_argument("--db", type=Path, default=None)
    p_doc.add_argument("--socket", type=Path, default=None)
    p_doc.add_argument("--home", type=Path, default=None)
    p_doc.add_argument("--json", action="store_true")
    p_doc.set_defaults(func=_cmd_doctor)

    # archivist inspect
    p_ins = sub.add_parser(
        "inspect", help="Inspect stored memories as Markdown/JSON or export JSONL"
    )
    p_ins.add_argument("--scope", default=None)
    p_ins.add_argument("--memory-id", default=None)
    p_ins.add_argument("--format", choices=["md", "json"], default="md")
    p_ins.add_argument("--export", default=None, help="Export sanitized memories to JSONL path")
    p_ins.add_argument("--limit", type=int, default=50)
    p_ins.add_argument("--db", type=Path, default=None)
    p_ins.add_argument("--offline-embed", action="store_true")
    p_ins.set_defaults(func=_cmd_inspect)

    # archivist import
    p_imp = sub.add_parser(
        "import", help="Import Mem0 JSON, CLAUDE.md, AGENTS.md, or .cursor/rules/*.mdc"
    )
    p_imp.add_argument("source", help="File or directory to import")
    p_imp.add_argument("--format", choices=["auto", "markdown", "mdc", "mem0-json"], default="auto")
    p_imp.add_argument("--scope", default=None)
    p_imp.add_argument("--db", type=Path, default=None)
    p_imp.add_argument("--offline-embed", action="store_true")
    p_imp.set_defaults(func=_cmd_import)

    # archivist hook
    p_hook = sub.add_parser("hook", help="Run zero-LLM deterministic IDE lifecycle hooks")
    p_hook.add_argument("event", choices=["session-start", "pre-compact", "stop"])
    p_hook.add_argument("--scope", default=None)
    p_hook.add_argument("--max-tokens", type=int, default=300)
    p_hook.add_argument("--input-text", default=None)
    p_hook.add_argument("--db", type=Path, default=None)
    p_hook.add_argument("--offline-embed", action="store_true")
    p_hook.set_defaults(func=_cmd_hook)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint for ``archivist``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
