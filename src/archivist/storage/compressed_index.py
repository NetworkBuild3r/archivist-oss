"""Compressed index generator — builds a per-namespace progressive-disclosure map.

The compressed index is a short navigational map (~500 tokens) of entity/type
pointers and search hints — not citable evidence (ADR-004 GR-CE-001). Agents
should use search / get_context for provenance-bearing facts.

Also produces compact wake-up payloads (~200 tokens) for session start,
compiling identity, critical facts, and namespace overview into a single payload.

INIT-004/SPEC-005: map-only builder — suppress/supersede-aware Graph SQL,
namespace-scoped reads, token estimate helper; no synchronous LLM (GR-CE-002).
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any

from archivist.core import metrics as m
from archivist.lifecycle.visibility import recall_visible_sql_facts
from archivist.storage.graph import get_curator_state, set_curator_state
from archivist.utils.tokenizer import count_tokens

logger = logging.getLogger("archivist.compressed_index")

_WAKE_UP_PRIMARY_TOOLS = (
    "archivist_search, archivist_store, archivist_wake_up, archivist_recall,"
    " archivist_timeline, archivist_namespaces"
)

# Shared SELECT list for entity map rows (cheap Graph path).
_ENTITY_SELECT = "e.id, e.name, e.entity_type, e.mention_count, e.retention_class, e.last_seen"


async def _query_entities(
    conn,
    namespace: str = "",
    agent_ids: list[str] | None = None,
    limit: int = 100,
) -> list[dict]:
    """Fetch top entities with at least one recall-visible fact.

    Omits entities whose only facts are suppressed, superseded, or inactive
    (INIT-003 visibility via ``recall_visible_sql_facts``). When *namespace*
    is set, reads are hard-scoped to that tenant.
    """
    visible = recall_visible_sql_facts("f")
    params: list[Any] = []
    where: list[str] = [visible]

    if namespace:
        where.append("e.namespace = ?")
        params.append(namespace)

    if agent_ids:
        placeholders = ",".join("?" for _ in agent_ids)
        where.append(f"f.agent_id IN ({placeholders})")
        params.extend(agent_ids)
    elif not namespace:
        # Fleet / unscoped fallback: keep prior mention threshold so empty
        # agent+namespace callers still get a bounded map.
        where.append("e.mention_count >= 2")

    sql = (
        f"SELECT DISTINCT {_ENTITY_SELECT} "
        f"FROM entities e "
        f"JOIN facts f ON f.entity_id = e.id "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY e.mention_count DESC "
        f"LIMIT ?"
    )
    params.append(limit)
    rows = await conn.fetchall(sql, params)
    return [dict(r) for r in rows]


def estimate_index_tokens(
    *,
    markdown: str = "",
    index_map: dict | None = None,
) -> dict[str, int]:
    """Cheap token estimate for map markdown / structured map (no LLM).

    Uses :func:`archivist.utils.tokenizer.count_tokens` (tiktoken or chars//4).
    """
    md_tokens = count_tokens(markdown) if markdown else 0
    if index_map is None:
        map_tokens = 0
    else:
        map_tokens = count_tokens(json.dumps(index_map, separators=(",", ":"), sort_keys=True))
    return {
        "markdown_tokens": md_tokens,
        "map_tokens": map_tokens,
        "total_tokens": md_tokens + map_tokens,
    }


def _render_index_markdown(namespace: str, index_map: dict) -> str:
    """Render progressive-disclosure map markdown (no key-fact prose)."""
    if index_map.get("empty"):
        return f"[Namespace: {namespace}] No indexed knowledge yet."

    lines = [
        f"# Memory Index — {namespace}",
        "",
        "Navigational map only — not evidence. Cite facts from "
        "`archivist_search` / `archivist_get_context` memories[], not this TOC.",
    ]

    entity_types: dict[str, list[str]] = index_map.get("entity_types") or {}
    for etype, names in sorted(entity_types.items()):
        label = etype.replace("_", " ").title() if etype != "unknown" else "General"
        lines.append(f"- **{label}**: {', '.join(names[:15])}")

    pinned = index_map.get("pinned") or []
    if pinned:
        lines.append(f"\n**Pinned/Durable** ({len(pinned)}): {', '.join(pinned[:10])}")

    recent = index_map.get("recently_active") or []
    if recent:
        lines.append(f"\n**Recently active**: {', '.join(recent)}")

    hints = index_map.get("search_hints") or []
    if hints:
        lines.append("\n**Search hints:**")
        for hint in hints:
            lines.append(f"- {hint}")

    topics = index_map.get("top_topics") or []
    if topics:
        lines.append(f"\nTop topics: {', '.join(topics)}")

    return "\n".join(lines)


def _build_index_map(namespace: str, entities: list[dict]) -> dict:
    """Build structured navigational map from entity rows (no fact prose)."""
    if not entities:
        return {
            "namespace": namespace,
            "empty": True,
            "entity_types": {},
            "entities": [],
            "pinned": [],
            "recently_active": [],
            "top_topics": [],
            "search_hints": [
                "archivist_search for a topic once memories exist",
                "archivist_get_context for budgeted recall",
            ],
        }

    by_type: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        by_type[e["entity_type"]].append(e["name"])

    pinned = [e["name"] for e in entities if e.get("retention_class") in ("durable", "permanent")]
    recent = sorted(
        [e for e in entities if e.get("last_seen")],
        key=lambda x: x.get("last_seen", ""),
        reverse=True,
    )[:5]
    recent_names = [e["name"] for e in recent]
    top_topics = [e["name"] for e in entities[:20]]
    hint_seeds = top_topics[:5] or pinned[:3]
    search_hints = [f'archivist_search query="{name}"' for name in hint_seeds] + [
        "archivist_get_context for budgeted, provenance-bearing recall"
    ]

    return {
        "namespace": namespace,
        "empty": False,
        "entity_types": {k: v[:15] for k, v in sorted(by_type.items())},
        "entities": [{"name": e["name"], "type": e["entity_type"]} for e in entities[:40]],
        "pinned": pinned[:10],
        "recently_active": recent_names,
        "top_topics": top_topics,
        "search_hints": search_hints,
    }


async def build_namespace_index_payload(namespace: str, agent_ids: list[str] | None = None) -> dict:
    """Build dual-shape index payload: ``{markdown, map}`` (INIT-004/SPEC-005).

    Map-only progressive disclosure — no key-fact prose (ADR-004 GR-CE-001).
    Active map entries omit suppressed / superseded losers (INIT-003 visibility).
    No synchronous LLM on this path (GR-CE-002). Namespace-scoped when set.
    """
    from archivist.storage.sqlite_pool import pool as _pool

    t0 = time.monotonic()
    entity_count = 0
    try:
        async with _pool.read() as conn:
            entities = await _query_entities(conn, namespace=namespace, agent_ids=agent_ids)
            entity_count = len(entities)

        index_map = _build_index_map(namespace, entities)
        markdown = _render_index_markdown(namespace, index_map)
        token_estimate = estimate_index_tokens(markdown=markdown, index_map=index_map)
        return {
            "markdown": markdown,
            "map": index_map,
            "token_estimate": token_estimate,
        }
    finally:
        rebuild_ms = round((time.monotonic() - t0) * 1000, 1)
        m.observe(m.INDEX_DURATION_MS, rebuild_ms)
        logger.info(
            "compressed_index.rebuild_complete",
            extra={
                "namespace": namespace,
                "rebuild_ms": rebuild_ms,
                "entity_count": entity_count,
                "agent_scoped": bool(agent_ids),
            },
        )


async def build_namespace_index(namespace: str, agent_ids: list[str] | None = None) -> str:
    """Build compressed map markdown for a namespace (compat wrapper).

    Prefer :func:`build_namespace_index_payload` for the dual ``{markdown, map}``
    shape used by ``archivist_index``. Emits timing hooks via the payload builder
    (INIT-004/SPEC-001).
    """
    payload = await build_namespace_index_payload(namespace, agent_ids=agent_ids)
    return payload["markdown"]


# ---------------------------------------------------------------------------
# Wake-up context — compact session bootstrap payload
# ---------------------------------------------------------------------------

_WAKE_UP_CACHE_PREFIX = "wake_up:"


async def build_wake_up_context(namespace: str, agent_id: str = "") -> dict:
    """Build a compact wake-up payload for session start.

    Pulls identity from permanent/durable entities (L0), critical facts from
    permanent + most-recent *recall-visible* facts (L1), and the map-only
    namespace TOC (no Key Facts prose section — INIT-004/SPEC-005).

    Target: L0+L1 combined under ~200 tokens.
    """
    agent_ids = [agent_id] if agent_id else None
    visible = recall_visible_sql_facts("f")

    from archivist.storage.sqlite_pool import pool as _pool

    async with _pool.read() as conn:
        # L0: permanent/durable entities for identity (visible facts only)
        if agent_ids:
            placeholders = ",".join("?" for _ in agent_ids)
            identity_sql = (
                f"SELECT DISTINCT e.name, e.entity_type, e.retention_class "
                f"FROM entities e "
                f"JOIN facts f ON f.entity_id = e.id "
                f"WHERE {visible} "
                f"AND f.agent_id IN ({placeholders}) "
                f"AND e.retention_class IN ('permanent', 'durable')"
            )
            params: list[Any] = list(agent_ids)
            if namespace:
                identity_sql += " AND e.namespace = ?"
                params.append(namespace)
            identity_sql += " ORDER BY e.mention_count DESC LIMIT 10"
            identity_rows = await conn.fetchall(identity_sql, params)
        else:
            identity_sql = (
                f"SELECT DISTINCT e.name, e.entity_type, e.retention_class "
                f"FROM entities e "
                f"JOIN facts f ON f.entity_id = e.id "
                f"WHERE {visible} "
                f"AND e.retention_class IN ('permanent', 'durable')"
            )
            params = []
            if namespace:
                identity_sql += " AND e.namespace = ?"
                params.append(namespace)
            identity_sql += " ORDER BY e.mention_count DESC LIMIT 10"
            identity_rows = await conn.fetchall(identity_sql, params)
        identity_entities = [dict(r) for r in identity_rows]

        l0_parts = []
        if namespace:
            l0_parts.append(f"Namespace: {namespace}")
        if agent_id:
            l0_parts.append(f"Agent: {agent_id}")
        if identity_entities:
            names = ", ".join(e["name"] for e in identity_entities[:6])
            l0_parts.append(f"Core entities: {names}")
        l0_identity = "; ".join(l0_parts) if l0_parts else "No identity data yet."

        # L1: pinned/permanent facts + most recent active (visible) facts
        # INIT-004/SPEC-007 M3 — namespace-scope pinned/recent facts when set
        # (entity name alone is not unique across tenants).
        entity_names = [e["name"] for e in identity_entities]
        pinned_facts: list[str] = []
        if entity_names:
            name_placeholders = ",".join("?" for _ in entity_names)
            pf_sql = (
                f"SELECT e.name, f.fact_text "
                f"FROM facts f "
                f"JOIN entities e ON f.entity_id = e.id "
                f"WHERE e.name IN ({name_placeholders}) "
                f"AND {visible} "
                f"AND f.retention_class = 'permanent'"
            )
            pf_params: list[Any] = list(entity_names)
            if namespace:
                pf_sql += " AND e.namespace = ? AND f.namespace = ?"
                pf_params.extend([namespace, namespace])
            pf_sql += " ORDER BY f.created_at DESC LIMIT 5"
            pf_rows = await conn.fetchall(pf_sql, pf_params)
            for row in pf_rows:
                pinned_facts.append(f"[{row['name']}] {row['fact_text'][:100]}")

        recent_facts: list[str] = []
        if agent_ids:
            placeholders = ",".join("?" for _ in agent_ids)
            rf_sql = (
                f"SELECT e.name, f.fact_text "
                f"FROM facts f "
                f"JOIN entities e ON f.entity_id = e.id "
                f"WHERE f.agent_id IN ({placeholders}) "
                f"AND {visible}"
            )
            rf_params: list[Any] = list(agent_ids)
            if namespace:
                rf_sql += " AND e.namespace = ? AND f.namespace = ?"
                rf_params.extend([namespace, namespace])
            rf_sql += " ORDER BY f.created_at DESC LIMIT 5"
            rf_rows = await conn.fetchall(rf_sql, rf_params)
        else:
            rf_sql = (
                f"SELECT e.name, f.fact_text "
                f"FROM facts f "
                f"JOIN entities e ON f.entity_id = e.id "
                f"WHERE {visible}"
            )
            rf_params: list[Any] = []
            if namespace:
                rf_sql += " AND e.namespace = ? AND f.namespace = ?"
                rf_params.extend([namespace, namespace])
            rf_sql += " ORDER BY f.created_at DESC LIMIT 5"
            rf_rows = await conn.fetchall(rf_sql, rf_params)
        for row in rf_rows:
            line = f"[{row['name']}] {row['fact_text'][:100]}"
            if line not in pinned_facts:
                recent_facts.append(line)

        l1_lines = pinned_facts + recent_facts[: max(0, 5 - len(pinned_facts))]
        l1_critical = "\n".join(l1_lines) if l1_lines else "No facts recorded yet."

        # Memory count (namespace-scoped when set)
        if namespace:
            cnt_row = await conn.fetchone(
                "SELECT COUNT(*) AS c FROM memory_chunks WHERE namespace = ?",
                (namespace,),
            )
        else:
            cnt_row = await conn.fetchone("SELECT COUNT(*) AS c FROM memory_chunks")
        total_memories = cnt_row["c"] if cnt_row else 0

        # Last activity
        if agent_ids:
            placeholders = ",".join("?" for _ in agent_ids)
            act_sql = (
                "SELECT MAX(last_seen) AS ls FROM entities "
                "WHERE name IN (SELECT DISTINCT e.name FROM entities e "
                f"JOIN facts f ON f.entity_id = e.id WHERE f.agent_id IN ({placeholders})"
            )
            act_params: list[Any] = list(agent_ids)
            if namespace:
                act_sql += " AND e.namespace = ?"
                act_params.append(namespace)
            act_sql += ")"
            act_row = await conn.fetchone(act_sql, act_params)
        elif namespace:
            act_row = await conn.fetchone(
                "SELECT MAX(last_seen) AS ls FROM entities WHERE namespace = ?",
                (namespace,),
            )
        else:
            act_row = await conn.fetchone("SELECT MAX(last_seen) AS ls FROM entities")
        last_activity = (act_row["ls"] or "")[:10] if act_row else ""

        # Fleet tips (no secrets; tip_text only)
        fleet_tips: list[str] = []
        try:
            tip_rows = await conn.fetchall(
                "SELECT tip_text FROM tips WHERE agent_id = 'fleet' AND archived = 0 "
                "ORDER BY usage_count DESC LIMIT 3"
            )
            fleet_tips = [r["tip_text"][:150] for r in tip_rows]
        except Exception:
            pass

        top_entities = [e["name"] for e in identity_entities[:10]]

    # Map-only TOC — shared helper; must not reintroduce Key Facts prose
    namespace_toc = await build_namespace_index(namespace, agent_ids=agent_ids)

    return {
        "l0_identity": l0_identity,
        "l1_critical": l1_critical,
        "namespace_toc": namespace_toc,
        "fleet_tips": fleet_tips,
        "total_memories": total_memories,
        "last_activity": last_activity,
        "top_entities": top_entities,
    }


async def get_cached_wake_up(namespace: str, agent_id: str = "") -> dict | None:
    """Return pre-computed wake-up context from curator_state, or None."""
    key = f"{_WAKE_UP_CACHE_PREFIX}{namespace}:{agent_id}"
    raw = await get_curator_state(key)
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


async def cache_wake_up(namespace: str, agent_id: str = "") -> dict:
    """Build wake-up context and persist it in curator_state for fast retrieval."""
    ctx = await build_wake_up_context(namespace, agent_id=agent_id)
    key = f"{_WAKE_UP_CACHE_PREFIX}{namespace}:{agent_id}"
    await set_curator_state(key, json.dumps(ctx))
    return ctx


def format_wake_up_text(ctx: dict, agent_id: str = "") -> str:
    """Render a wake-up context dict as a compact text block for agent consumption.

    Args:
        ctx: Wake-up context produced by :func:`build_wake_up_context`.
        agent_id: The requesting agent. When provided, a one-line access summary
            is prepended listing the agent's namespace and access rights so the
            agent does not need a separate ``archivist_namespaces`` call.

    The embedded ``namespace_toc`` is the map-only index (no Key Facts section).
    Ops L1 critical facts remain a separate wake_up field (not index TOC).
    """
    lines: list[str] = []

    if agent_id:
        from archivist.core.rbac import (
            list_accessible_namespaces,
        )

        namespace = ctx.get("l0_identity", "")
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
        # Guard: never render a Key Facts prose block from the TOC
        if "Key Facts" not in toc:
            lines.append(f"\n{toc}")
        else:
            logger.warning(
                "compressed_index.wake_up_toc_skipped_key_facts",
                extra={"reason": "map-only contract violated"},
            )
    return "\n".join(lines)
