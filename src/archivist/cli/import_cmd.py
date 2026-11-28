"""Importers for Mem0 JSON, ``CLAUDE.md``, ``AGENTS.md``, and ``.cursor/rules/*.mdc`` (INIT-024/SPEC-009).

Every imported entry is tagged with ``channel="import"`` (attested ``T2_EXTERNAL``
trust tier unless flagged as ``T3_QUARANTINED``) and passes through the ``<5ms``
``re2`` Memory Firewall BEFORE embedding or persisting — guaranteeing zero secret
leakage during migration.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from archivist.app.handlers.scope_resolver import resolve_write_scope
from archivist.security.trust_tiers import RequestContext
from archivist.storage.local_store import LocalMemoryStore

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_markdown_sections(text: str, *, default_tag: str) -> list[tuple[str, str, list[str]]]:
    """Split ``CLAUDE.md`` / ``AGENTS.md`` into ``(kind, content, tags)`` tuples."""
    entries: list[tuple[str, str, list[str]]] = []
    heading = ""
    buf: list[str] = []

    def _flush() -> None:
        body = "\n".join(buf).strip()
        if len(body) >= 12:
            full = f"{heading}: {body}" if heading else body
            entries.append(("fact", full, ["imported", default_tag]))

    for line in text.splitlines():
        if line.startswith(("## ", "### ")):
            _flush()
            heading = line.lstrip("#").strip()
            buf = []
        else:
            buf.append(line)
    _flush()

    if not entries and text.strip():
        entries.append(("fact", text.strip(), ["imported", default_tag]))
    return entries


def _parse_cursor_mdc(text: str, *, filename: str) -> list[tuple[str, str, list[str]]]:
    """Parse a Cursor ``.mdc`` rule file with optional YAML frontmatter."""
    tags = ["imported", "cursor-rule", Path(filename).stem]
    body = text.strip()
    match = _FRONTMATTER_RE.match(text)
    if match:
        fm_raw, rest = match.group(1), match.group(2)
        try:
            fm = yaml.safe_load(fm_raw) or {}
            if isinstance(fm, dict):
                desc = str(fm.get("description") or "").strip()
                globs = fm.get("globs")
                if isinstance(globs, str) and globs.strip():
                    tags.append(f"globs:{globs.strip()}")
                body = f"{desc}\n\n{rest.strip()}" if desc else rest.strip()
        except yaml.YAMLError:
            body = rest.strip()

    if not body:
        return []
    return [("preference", body, tags)]


def _parse_mem0_json(text: str) -> list[tuple[str, str, list[str]]]:
    """Parse Mem0 JSON exports (list, ``{"results": [...]}``, or ``{"memories": [...]}``)."""
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        items = parsed.get("results") or parsed.get("memories") or parsed.get("data") or []
    elif isinstance(parsed, list):
        items = parsed
    else:
        raise ValueError("Unsupported Mem0 JSON root structure")

    entries: list[tuple[str, str, list[str]]] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            entries.append(("fact", item.strip(), ["imported", "mem0"]))
            continue
        if not isinstance(item, dict):
            continue
        content = str(item.get("memory") or item.get("text") or item.get("content") or "").strip()
        if not content:
            continue
        meta = item.get("metadata")
        tags = ["imported", "mem0"]
        kind = "fact"
        if isinstance(meta, dict):
            if meta.get("kind"):
                kind = str(meta["kind"])
            if isinstance(meta.get("tags"), list):
                tags.extend(str(t) for t in meta["tags"] if str(t).strip())
            if meta.get("category"):
                tags.append(str(meta["category"]))
        entries.append((kind, content, tags))
    return entries


def _detect_format(path: Path, explicit_fmt: str) -> str:
    norm = (explicit_fmt or "auto").strip().lower()
    if norm != "auto":
        return norm
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "mem0-json"
    if suffix == ".mdc":
        return "mdc"
    return "markdown"


def import_source(
    store: LocalMemoryStore,
    source_path: Path,
    *,
    fmt: str = "auto",
    scope: str | None = None,
) -> dict[str, Any]:
    """Import memories from a file or ``.cursor/rules`` directory through Memory Firewall."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Import source not found: {path}")

    resolved_scope = resolve_write_scope(scope, cwd=path if path.is_dir() else path.parent)
    uid = os.getuid() if hasattr(os, "getuid") else 0
    ctx = RequestContext.for_local_uid(
        uid,
        source_client="archivist-import",
        channel="import",
        actor_type="verified_local",
    )

    files: list[Path]
    if path.is_dir():
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in (".mdc", ".md", ".json")
        )
    else:
        files = [path]

    imported_ids: list[str] = []
    dedup_count = 0
    redacted_total = 0
    quarantined_count = 0

    for fp in files:
        detected = _detect_format(fp, fmt)
        raw = fp.read_text(encoding="utf-8")
        if detected == "mem0-json":
            entries = _parse_mem0_json(raw)
        elif detected == "mdc":
            entries = _parse_cursor_mdc(raw, filename=fp.name)
        elif detected == "markdown":
            entries = _parse_markdown_sections(raw, default_tag=fp.name.lower())
        else:
            raise ValueError(f"Unsupported import format: {detected!r}")

        for kind, content, tags in entries:
            saved = store.save_memory(
                content,
                scope=resolved_scope,
                kind=kind,
                tags=tags,
                ctx=ctx,
                deduplicate=True,
            )
            if saved.deduplicated:
                dedup_count += 1
            else:
                imported_ids.append(saved.memory_id)
            redacted_total += saved.redacted_count
            if saved.quarantined:
                quarantined_count += 1

    return {
        "source": str(path),
        "scope": resolved_scope,
        "files_processed": len(files),
        "imported_count": len(imported_ids),
        "imported_ids": imported_ids,
        "deduplicated_count": dedup_count,
        "redacted_count": redacted_total,
        "quarantined_count": quarantined_count,
    }
