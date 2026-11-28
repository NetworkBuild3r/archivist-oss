"""Human-readable Markdown inspection & JSONL export (INIT-024/SPEC-009).

Provides ``archivist inspect [--format md|json] [--export .archivist/shared.jsonl]``
so developers can inspect, audit, and version-control memories as cleanly as
``CLAUDE.md``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from archivist.security.firewall import default_firewall
from archivist.security.trust_tiers import TrustTier
from archivist.storage.local_store import LocalMemoryStore


def render_memories_markdown(inspection: dict[str, Any]) -> str:
    """Render ``LocalMemoryStore.inspect_memories()`` output as human-readable Markdown."""
    memories = inspection.get("memories") or []
    scope_label = inspection.get("scope", "*")
    lines: list[str] = [
        f"# Archivist Memory Inspection (`{scope_label}`)",
        "",
        f"- **Embed Model:** `{inspection.get('embed_model', 'unknown')}` "
        f"(`dim={inspection.get('vector_dim', 0)}`, backend=`{inspection.get('vector_backend', 'sqlite-vec')}`)",
        f"- **Shown:** `{inspection.get('total_shown', len(memories))}` "
        f"| **Quarantined (`T3_QUARANTINED`):** `{inspection.get('quarantined_count', 0)}`",
        "",
    ]
    if not memories:
        lines.append("_No memories stored for this scope._")
        return "\n".join(lines) + "\n"

    # Group by scope for readable Markdown hierarchy.
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for item in memories:
        by_scope.setdefault(str(item.get("scope", "user:global")), []).append(item)

    for sc, items in sorted(by_scope.items()):
        lines.append(f"## Scope `{sc}`")
        lines.append("")
        for item in items:
            mid = item.get("id", "")
            kind = item.get("kind", "fact")
            tier = item.get("trust_tier", TrustTier.T1_AGENT.value)
            actor = item.get("actor", "local")
            created = item.get("created_at", "")
            tags = item.get("tags") or []
            tag_str = f" | tags: {', '.join(tags)}" if tags else ""
            lines.append(f"### `{mid}` [{kind}] (`{tier}`)")
            lines.append(f"_Actor: `{actor}` | Created: `{created}`{tag_str}_")
            if item.get("quarantine_reasons"):
                reasons = ", ".join(str(r) for r in item["quarantine_reasons"])
                lines.append(f"**Quarantined:** {reasons}")
            lines.append("")
            lines.append(str(item.get("text", "")).strip())
            lines.append("")

    return "\n".join(lines)


def export_shared_jsonl(
    store: LocalMemoryStore,
    export_path: Path,
    *,
    scope: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Export non-quarantined, firewall-sanitized memories to deterministic JSONL."""
    inv = store.inspect_memories(scope=scope, limit=limit, include_suppressed=False)
    memories = inv.get("memories") or []
    fw = default_firewall()

    export_rows: list[dict[str, Any]] = []
    for item in sorted(memories, key=lambda m: str(m.get("id", ""))):
        if str(item.get("trust_tier", "")) == TrustTier.T3_QUARANTINED.value:
            continue
        # Defensive re-scan of content, kind, and tags before writing to a git-tracked file.
        res = fw.scan(str(item.get("text", "")))
        if res.quarantined:
            continue
        kind_res = fw.scan(str(item.get("kind", "fact")))
        if kind_res.quarantined:
            continue
        safe_tags: list[str] = []
        tag_quarantined = False
        for raw_tag in item.get("tags") or []:
            tag_res = fw.scan(str(raw_tag))
            if tag_res.quarantined:
                tag_quarantined = True
                break
            cleaned = tag_res.text.strip()
            if cleaned:
                safe_tags.append(cleaned)
        if tag_quarantined:
            continue
        export_rows.append(
            {
                "id": item["id"],
                "scope": item["scope"],
                "kind": kind_res.text.strip() or "fact",
                "tags": safe_tags,
                "trust_tier": item["trust_tier"],
                "created_at": item["created_at"],
                "content": res.text,
            }
        )

    export_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(export_path.parent),
            prefix=f".{export_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            for row in export_rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, export_path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return {
        "export_path": str(export_path),
        "exported_count": len(export_rows),
        "scope": scope or "*",
    }
