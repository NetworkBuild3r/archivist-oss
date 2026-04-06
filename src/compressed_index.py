"""Compressed index generator — builds a per-namespace semantic TOC (zer0dex-inspired).

The compressed index is a short text (~500-800 tokens) listing what categories
of knowledge exist in a namespace, so agents can bridge cross-domain queries
without relying solely on vector similarity.
"""

import logging
from collections import defaultdict

from graph import get_db

logger = logging.getLogger("archivist.compressed_index")


def _query_entities(conn, agent_ids: list[str] | None, limit: int = 100) -> list[dict]:
    """Fetch top entities, optionally scoped by agent_ids."""
    if agent_ids:
        placeholders = ",".join("?" for _ in agent_ids)
        cur = conn.execute(
            f"""SELECT DISTINCT e.id, e.name, e.entity_type, e.mention_count,
                       e.retention_class, e.last_seen
                FROM entities e
                JOIN facts f ON f.entity_id = e.id AND f.is_active = 1
                WHERE f.agent_id IN ({placeholders})
                ORDER BY e.mention_count DESC
                LIMIT ?""",
            agent_ids + [limit],
        )
    else:
        cur = conn.execute(
            """SELECT id, name, entity_type, mention_count, retention_class, last_seen
               FROM entities
               WHERE mention_count >= 2
               ORDER BY mention_count DESC
               LIMIT ?""",
            (limit,),
        )
    return [dict(r) for r in cur.fetchall()]


def _query_key_facts(conn, entity_ids: list[int], per_entity: int = 2) -> dict[int, list[str]]:
    """Fetch the most recent active facts for a set of entities."""
    if not entity_ids:
        return {}
    placeholders = ",".join("?" for _ in entity_ids)
    cur = conn.execute(
        f"""SELECT entity_id, fact_text, retention_class
            FROM facts
            WHERE entity_id IN ({placeholders}) AND is_active = 1 AND superseded_by IS NULL
            ORDER BY
                CASE retention_class WHEN 'permanent' THEN 0 WHEN 'durable' THEN 1 ELSE 2 END,
                created_at DESC""",
        entity_ids,
    )
    result: dict[int, list[str]] = defaultdict(list)
    for row in cur.fetchall():
        eid = row["entity_id"]
        if len(result[eid]) < per_entity:
            prefix = "[pinned] " if row["retention_class"] == "permanent" else ""
            result[eid].append(f"{prefix}{row['fact_text']}")
    return dict(result)


def build_namespace_index(namespace: str, agent_ids: list[str] | None = None) -> str:
    """Build a compressed index for a namespace from graph entities and Qdrant metadata.

    Returns a compact text string suitable for injection into agent context.
    Includes entity categories, key facts for top entities, pinned items,
    and recent changes.
    """
    conn = get_db()

    try:
        entities = _query_entities(conn, agent_ids)

        if not entities:
            return f"[Namespace: {namespace}] No indexed knowledge yet."

        top_ids = [e["id"] for e in entities[:15]]
        key_facts = _query_key_facts(conn, top_ids, per_entity=2)

        pinned = [e for e in entities if e.get("retention_class") in ("durable", "permanent")]
        recent = sorted(
            [e for e in entities if e.get("last_seen")],
            key=lambda x: x.get("last_seen", ""),
            reverse=True,
        )[:5]
    finally:
        conn.close()

    by_type: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        by_type[e["entity_type"]].append(e["name"])

    lines = [f"# Memory Index — {namespace}"]

    for etype, names in sorted(by_type.items()):
        label = etype.replace("_", " ").title() if etype != "unknown" else "General"
        lines.append(f"- **{label}**: {', '.join(names[:15])}")

    if pinned:
        lines.append(f"\n**Pinned/Durable** ({len(pinned)}): {', '.join(e['name'] for e in pinned[:10])}")

    if key_facts:
        lines.append("\n**Key Facts:**")
        eid_to_name = {e["id"]: e["name"] for e in entities}
        for eid, facts in list(key_facts.items())[:8]:
            name = eid_to_name.get(eid, "?")
            for f in facts:
                lines.append(f"  - {name}: {f[:120]}")

    if recent:
        lines.append(f"\n**Recently active**: {', '.join(e['name'] for e in recent)}")

    topic_line = ", ".join(e["name"] for e in entities[:20])
    lines.append(f"\nTop topics: {topic_line}")

    return "\n".join(lines)


