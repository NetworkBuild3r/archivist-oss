"""Compressed index generator — builds a per-namespace semantic TOC (zer0dex-inspired).

The compressed index is a short text (~500-800 tokens) listing what categories
of knowledge exist in a namespace, so agents can bridge cross-domain queries
without relying solely on vector similarity.
"""

import logging
from collections import defaultdict

from graph import get_db

logger = logging.getLogger("archivist.compressed_index")


def build_namespace_index(namespace: str, agent_ids: list[str] | None = None) -> str:
    """Build a compressed index for a namespace from graph entities and Qdrant metadata.

    Returns a compact text string suitable for injection into agent context.
    """
    conn = get_db()

    try:
        if agent_ids:
            placeholders = ",".join("?" for _ in agent_ids)
            cur = conn.execute(
                f"""SELECT DISTINCT e.name, e.entity_type, e.mention_count
                    FROM entities e
                    JOIN facts f ON f.entity_id = e.id AND f.is_active = 1
                    WHERE f.agent_id IN ({placeholders})
                    ORDER BY e.mention_count DESC
                    LIMIT 100""",
                agent_ids,
            )
        else:
            cur = conn.execute(
                """SELECT DISTINCT e.name, e.entity_type, e.mention_count
                   FROM entities e
                   WHERE e.mention_count >= 2
                   ORDER BY e.mention_count DESC
                   LIMIT 100""",
            )
        entities = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    if not entities:
        return f"[Namespace: {namespace}] No indexed knowledge yet."

    by_type: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        by_type[e["entity_type"]].append(e["name"])

    lines = [f"# Memory Index — {namespace}"]
    for etype, names in sorted(by_type.items()):
        label = etype.replace("_", " ").title() if etype != "unknown" else "General"
        lines.append(f"- **{label}**: {', '.join(names[:15])}")

    topic_line = ", ".join(e["name"] for e in entities[:20])
    lines.append(f"\nTop topics: {topic_line}")

    return "\n".join(lines)


def build_fleet_index(accessible_namespaces: list[str]) -> str:
    """Build a compressed index spanning multiple namespaces."""
    parts = []
    for ns in accessible_namespaces[:5]:
        idx = build_namespace_index(ns)
        if idx:
            parts.append(idx)
    return "\n---\n".join(parts) if parts else "No indexed knowledge across accessible namespaces."
