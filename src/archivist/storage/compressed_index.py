"""Compressed index generator — builds a per-namespace semantic TOC.

The compressed index is a short text (~500-800 tokens) listing what categories
of knowledge exist in a namespace, so agents can bridge cross-domain queries
without relying solely on vector similarity.

Also produces compact wake-up payloads (~200 tokens) for session start,
compiling identity, critical facts, and namespace overview into a single payload.
"""

import json
import logging
from collections import defaultdict

from archivist.storage.graph import get_curator_state, get_db, set_curator_state

logger = logging.getLogger("archivist.compressed_index")

_WAKE_UP_PRIMARY_TOOLS = (
    "archivist_search, archivist_store, archivist_wake_up, archivist_recall,"
    " archivist_timeline, archivist_namespaces"
)


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
        lines.append(
            f"\n**Pinned/Durable** ({len(pinned)}): {', '.join(e['name'] for e in pinned[:10])}"
        )

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


# ---------------------------------------------------------------------------
# Wake-up context — compact session bootstrap payload
# ---------------------------------------------------------------------------

_WAKE_UP_CACHE_PREFIX = "wake_up:"


def build_wake_up_context(namespace: str, agent_id: str = "") -> dict:
    """Build a compact wake-up payload for session start.

    Pulls identity from permanent/durable entities (L0), critical facts from
    permanent + most-recent active facts (L1), and the existing namespace TOC.

    Target: L0+L1 combined under ~200 tokens.
    """
    conn = get_db()
    try:
        agent_ids = [agent_id] if agent_id else None

        # L0: permanent/durable entities for identity
        if agent_ids:
            placeholders = ",".join("?" for _ in agent_ids)
            cur = conn.execute(
                f"""SELECT DISTINCT e.name, e.entity_type, e.retention_class
                    FROM entities e
                    JOIN facts f ON f.entity_id = e.id AND f.is_active = 1
                    WHERE f.agent_id IN ({placeholders})
                      AND e.retention_class IN ('permanent', 'durable')
                    ORDER BY e.mention_count DESC
                    LIMIT 10""",
                agent_ids,
            )
        else:
            cur = conn.execute(
                """SELECT name, entity_type, retention_class
                   FROM entities
                   WHERE retention_class IN ('permanent', 'durable')
                   ORDER BY mention_count DESC
                   LIMIT 10"""
            )
        identity_entities = [dict(r) for r in cur.fetchall()]

        l0_parts = []
        if namespace:
            l0_parts.append(f"Namespace: {namespace}")
        if agent_id:
            l0_parts.append(f"Agent: {agent_id}")
        if identity_entities:
            names = ", ".join(e["name"] for e in identity_entities[:6])
            l0_parts.append(f"Core entities: {names}")
        l0_identity = "; ".join(l0_parts) if l0_parts else "No identity data yet."

        # L1: pinned/permanent facts + most recent active facts
        entity_ids = [e["name"] for e in identity_entities]
        pinned_facts: list[str] = []
        if entity_ids:
            name_placeholders = ",".join("?" for _ in entity_ids)
            cur = conn.execute(
                f"""SELECT e.name, f.fact_text
                    FROM facts f
                    JOIN entities e ON f.entity_id = e.id
                    WHERE e.name IN ({name_placeholders})
                      AND f.is_active = 1 AND f.superseded_by IS NULL
                      AND f.retention_class = 'permanent'
                    ORDER BY f.created_at DESC
                    LIMIT 5""",
                entity_ids,
            )
            for row in cur.fetchall():
                pinned_facts.append(f"[{row['name']}] {row['fact_text'][:100]}")

        recent_facts: list[str] = []
        if agent_ids:
            placeholders = ",".join("?" for _ in agent_ids)
            cur = conn.execute(
                f"""SELECT e.name, f.fact_text
                    FROM facts f
                    JOIN entities e ON f.entity_id = e.id
                    WHERE f.agent_id IN ({placeholders})
                      AND f.is_active = 1 AND f.superseded_by IS NULL
                    ORDER BY f.created_at DESC
                    LIMIT 5""",
                agent_ids,
            )
        else:
            cur = conn.execute(
                """SELECT e.name, f.fact_text
                   FROM facts f
                   JOIN entities e ON f.entity_id = e.id
                   WHERE f.is_active = 1 AND f.superseded_by IS NULL
                   ORDER BY f.created_at DESC
                   LIMIT 5"""
            )
        for row in cur.fetchall():
            line = f"[{row['name']}] {row['fact_text'][:100]}"
            if line not in pinned_facts:
                recent_facts.append(line)

        l1_lines = pinned_facts + recent_facts[: max(0, 5 - len(pinned_facts))]
        l1_critical = "\n".join(l1_lines) if l1_lines else "No facts recorded yet."

        # Counts
        if namespace:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM memory_chunks WHERE namespace = ?",
                (namespace,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS c FROM memory_chunks").fetchone()
        total_memories = row["c"] if row else 0

        # Last activity
        if agent_ids:
            placeholders = ",".join("?" for _ in agent_ids)
            row = conn.execute(
                f"SELECT MAX(last_seen) AS ls FROM entities "
                f"WHERE name IN (SELECT DISTINCT e.name FROM entities e "
                f"JOIN facts f ON f.entity_id = e.id WHERE f.agent_id IN ({placeholders}))",
                agent_ids,
            ).fetchone()
        else:
            row = conn.execute("SELECT MAX(last_seen) AS ls FROM entities").fetchone()
        last_activity = (row["ls"] or "")[:10] if row else ""

        top_entities = [e["name"] for e in identity_entities[:10]]
    finally:
        conn.close()

    namespace_toc = build_namespace_index(namespace, agent_ids=agent_ids)

    fleet_tips: list[str] = []
    try:
        _tips_conn = get_db()
        _tips_cur = _tips_conn.execute(
            "SELECT tip_text FROM tips WHERE agent_id = 'fleet' AND archived = 0 "
            "ORDER BY usage_count DESC LIMIT 3"
        )
        fleet_tips = [r["tip_text"][:150] for r in _tips_cur.fetchall()]
        _tips_conn.close()
    except Exception:
        pass

    return {
        "l0_identity": l0_identity,
        "l1_critical": l1_critical,
        "namespace_toc": namespace_toc,
        "fleet_tips": fleet_tips,
        "total_memories": total_memories,
        "last_activity": last_activity,
        "top_entities": top_entities,
    }


def get_cached_wake_up(namespace: str, agent_id: str = "") -> dict | None:
    """Return pre-computed wake-up context from curator_state, or None."""
    key = f"{_WAKE_UP_CACHE_PREFIX}{namespace}:{agent_id}"
    raw = get_curator_state(key)
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def cache_wake_up(namespace: str, agent_id: str = "") -> dict:
    """Build wake-up context and persist it in curator_state for fast retrieval."""
    ctx = build_wake_up_context(namespace, agent_id=agent_id)
    key = f"{_WAKE_UP_CACHE_PREFIX}{namespace}:{agent_id}"
    set_curator_state(key, json.dumps(ctx))
    return ctx


def format_wake_up_text(ctx: dict, agent_id: str = "") -> str:
    """Render a wake-up context dict as a compact text block for agent consumption.

    Args:
        ctx: Wake-up context produced by :func:`build_wake_up_context`.
        agent_id: The requesting agent. When provided, a one-line access summary
            is prepended listing the agent's namespace and access rights so the
            agent does not need a separate ``archivist_namespaces`` call.
    """
    lines: list[str] = []

    # Compact one-line summary — prepended when agent_id is known.
    if agent_id:
        from archivist.core.rbac import (
            list_accessible_namespaces,  # local to avoid circular import at module load
        )

        namespace = ctx.get("l0_identity", "")
        # Extract namespace from the l0_identity string ("Namespace: X; Agent: Y; ...")
        ns_display = ""
        for part in namespace.split(";"):
            part = part.strip()
            if part.startswith("Namespace:"):
                ns_display = part.split(":", 1)[1].strip()
                break

        accessible = list_accessible_namespaces(agent_id)
        _MAX_SUMMARY_NS = 8
        access_parts: list[str] = []
        for entry in accessible[:_MAX_SUMMARY_NS]:
            perm = ""
            if entry["can_read"] and entry["can_write"]:
                perm = "rw"
            elif entry["can_read"]:
                perm = "r"
            else:
                perm = "w"
            access_parts.append(f"{entry['namespace']}({perm})")
        if len(accessible) > _MAX_SUMMARY_NS:
            access_parts.append("...")
        access_str = ", ".join(access_parts) if access_parts else "(none)"

        summary_parts = [f"Namespace: {ns_display}" if ns_display else f"Agent: {agent_id}"]
        summary_parts.append(f"Access: {access_str}")
        summary_parts.append(f"Tools: {_WAKE_UP_PRIMARY_TOOLS}")
        lines.append(" | ".join(summary_parts))

    lines += [
        "## Wake-Up Context",
        f"**Identity:** {ctx.get('l0_identity', 'unknown')}",
        f"**Memories:** {ctx.get('total_memories', 0)} | **Last active:** {ctx.get('last_activity', 'n/a')}",
    ]
    l1 = ctx.get("l1_critical", "")
    if l1 and l1 != "No facts recorded yet.":
        lines.append(f"\n**Critical facts:**\n{l1}")
    fleet_tips = ctx.get("fleet_tips", [])
    if fleet_tips:
        lines.append("\n**Fleet tips:**")
        for tip in fleet_tips:
            lines.append(f"  - {tip}")
    toc = ctx.get("namespace_toc", "")
    if toc and "No indexed knowledge" not in toc:
        lines.append(f"\n{toc}")
    return "\n".join(lines)
