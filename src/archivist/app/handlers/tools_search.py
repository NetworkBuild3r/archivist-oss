"""MCP tool handlers — core search and retrieval."""

from datetime import UTC

from mcp.types import TextContent, Tool

from archivist.core.archivist_uri import memory_uri
from archivist.core.rbac import (
    can_read_agent_memory,
    filter_agents_for_read,
    get_namespace_for_agent,
    is_permissive_mode,
)
from archivist.retrieval.rlm_retriever import recursive_retrieve, search_vectors
from archivist.storage.compressed_index import (
    build_namespace_index,
    cache_wake_up,
    format_wake_up_text,
    get_cached_wake_up,
)
from archivist.storage.graph import (
    add_entity_alias,
    get_entity_facts,
    get_entity_relationships,
    search_entities,
)
from archivist.storage.graph_retrieval import detect_contradictions, get_entity_brief

from ._common import _rbac_gate, error_response, require_caller, resolve_caller, success_response

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="archivist_search",
        description=(
            "Semantic search across agent memories with RLM recursive retrieval. "
            "Supports fleet-wide search or a list of agent_ids (multi-agent memory). "
            "Set caller_agent_id when reading other agents' memories so RBAC can apply. "
            "Returns synthesized answer with source citations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "agent_id": {
                    "type": "string",
                    "description": "Filter to one agent's memories (optional)",
                    "default": "",
                },
                "agent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Search only these agents' memories (OR). Omit for fleet-wide or use agent_id for a single agent.",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Identity of the invoking agent — used for RBAC when reading others' namespaces. Defaults to agent_id if set.",
                },
                "namespace": {
                    "type": "string",
                    "description": "Memory namespace to search (optional, auto-detect from agent_id)",
                    "default": "",
                },
                "team": {
                    "type": "string",
                    "description": "Filter by team (optional)",
                    "default": "",
                },
                "refine": {
                    "type": "boolean",
                    "description": "Use LLM refinement for higher quality (slower). Default false.",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max chunks to refine/synthesize after retrieval",
                    "default": 20,
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum vector similarity (0–1). Overrides RETRIEVAL_THRESHOLD for this call; omit to use env default. Use 0 to disable filtering.",
                },
                "tier": {
                    "type": "string",
                    "enum": ["l0", "l1", "l2"],
                    "description": "Context tier: l0 (abstract ~100 tokens), l1 (overview ~500 tokens), l2 (full detail). Default l2.",
                    "default": "l2",
                },
                "date_from": {
                    "type": "string",
                    "description": "ISO date lower bound (inclusive), e.g. 2026-01-01",
                    "default": "",
                },
                "date_to": {
                    "type": "string",
                    "description": "ISO date upper bound (inclusive)",
                    "default": "",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Approximate token budget for context returned (caps chunk count). Omit for unlimited.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["experience", "skill", "general", ""],
                    "description": "Filter by memory type: experience, skill, or general. Omit for all types.",
                    "default": "",
                },
                "actor_type": {
                    "type": "string",
                    "enum": ["agent", "human", "system", "tool", ""],
                    "description": "Filter by actor type. Omit for all types.",
                    "default": "",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="archivist_recall",
        description=(
            "Graph-based multi-hop recall. Finds entities and their relationships/facts. "
            "Use for questions like 'What do we know about X?' or 'How does X relate to Y?'. "
            "Use as_of to see facts valid at a specific date."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity name to look up"},
                "related_to": {
                    "type": "string",
                    "description": "Optional second entity to find connections",
                    "default": "",
                },
                "as_of": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD) — only return facts valid at this date. Omit for current facts.",
                    "default": "",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Calling agent for RBAC (optional)",
                    "default": "",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Identity for read access checks (defaults to agent_id). Required when RBAC namespaces are configured.",
                },
                "namespace": {
                    "type": "string",
                    "description": "Memory namespace scope (optional)",
                    "default": "",
                },
            },
            "required": ["entity"],
        },
    ),
    Tool(
        name="archivist_timeline",
        description=(
            "Get a chronological timeline of memories about a topic. "
            "Use for questions like 'What happened with X over the last 2 weeks?'. "
            "Use as_of to anchor the timeline to a specific date."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic to build timeline for"},
                "agent_id": {
                    "type": "string",
                    "description": "Filter to specific agent (optional)",
                    "default": "",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Invoker identity for RBAC (defaults to agent_id when set).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Memory namespace to search (optional)",
                    "default": "",
                },
                "days": {
                    "type": "integer",
                    "description": "How many days back to search",
                    "default": 14,
                },
                "as_of": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD) — anchor the timeline to this date for entity facts. Omit for current.",
                    "default": "",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="archivist_insights",
        description=(
            "Get curated cross-agent insights for a topic. "
            "Searches across all accessible namespaces to find shared knowledge and patterns."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to get insights on"},
                "agent_id": {
                    "type": "string",
                    "description": "Calling agent for RBAC (optional)",
                    "default": "",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Invoker identity for RBAC when reading cross-agent insights (defaults to agent_id).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace scope (optional)",
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max insights to return",
                    "default": 10,
                },
            },
            "required": ["topic"],
        },
    ),
    Tool(
        name="archivist_deref",
        description=(
            "Dereference a memory point by ID. Returns the full L2 text and metadata for a specific memory. "
            "Use after an L0/L1 search to get full detail on a relevant hit."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Qdrant point ID to retrieve"},
                "agent_id": {
                    "type": "string",
                    "description": "Calling agent for RBAC",
                    "default": "",
                },
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="archivist_index",
        description=(
            "Get a compressed navigational index of what knowledge exists in a namespace. "
            "Returns a short text (~500 tokens) listing entity categories and top topics — "
            "useful for cross-domain bridging and deciding what to search for."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Calling agent for RBAC and default namespace resolution",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace to index (default: auto-detect)",
                    "default": "",
                },
            },
            "required": ["agent_id"],
        },
    ),
    Tool(
        name="archivist_contradictions",
        description=(
            "Surface contradicting facts about an entity from different agents. "
            "Uses heuristic keyword detection on the knowledge graph."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity name to check for contradictions",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Calling agent for RBAC",
                    "default": "",
                },
            },
            "required": ["entity"],
        },
    ),
    Tool(
        name="archivist_entity_brief",
        description=(
            "Get a structured knowledge card for an entity: all known facts, relationships, "
            "retention class, mention count, timeline. Use for questions like "
            "'What do we know about prod-db-01?' or 'Show me everything about Alice'. "
            "Returns structured data — better than grep through markdown files. "
            "Use as_of to see facts valid at a specific date (e.g. 'What was true on 2025-01-15?')."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity name to look up"},
                "add_alias": {
                    "type": "string",
                    "description": "Optional alias to register for this entity (e.g. 'k8s' for 'Kubernetes')",
                    "default": "",
                },
                "as_of": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD) — only return facts valid at this date. Omit for current facts.",
                    "default": "",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Calling agent for RBAC",
                    "default": "",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Invoker identity for RBAC (defaults to agent_id).",
                },
            },
            "required": ["entity"],
        },
    ),
    Tool(
        name="archivist_wake_up",
        description=(
            "Load session context for an agent — identity, critical facts, "
            "and namespace overview in ~200 tokens. Call once at session start "
            "to bootstrap the agent's working memory."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Agent identity for namespace resolution and fact scoping",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace to load (default: auto-detect from agent_id)",
                    "default": "",
                },
            },
            "required": ["agent_id"],
        },
    ),
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _filter_facts_rels_for_caller(caller: str, facts: list[dict], rels: list[dict]):
    """Drop graph rows the caller is not allowed to read (fact/rel agent_id)."""
    ff = [f for f in facts if can_read_agent_memory(caller, (f.get("agent_id") or "").strip())]
    rr = [r for r in rels if can_read_agent_memory(caller, (r.get("agent_id") or "").strip())]
    return ff, rr


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_search(arguments: dict) -> list[TextContent]:
    agent_id = arguments.get("agent_id", "")
    namespace = arguments.get("namespace", "")

    raw_ids = arguments.get("agent_ids")
    agent_ids: list[str] | None = None
    if raw_ids is not None:
        if isinstance(raw_ids, str):
            agent_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
        else:
            agent_ids = [str(x).strip() for x in raw_ids if str(x).strip()]

    caller = resolve_caller(arguments)

    if namespace and caller:
        denied = _rbac_gate(caller, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    if agent_ids:
        allowed, denied_list = filter_agents_for_read(caller, agent_ids)
        if not allowed:
            return error_response(
                {
                    "error": "access_denied",
                    "reason": "Caller cannot read any of the requested agents' namespaces",
                    "hint": (
                        f"Use archivist_namespaces(agent_id='{caller}') to see your"
                        " accessible namespaces."
                    ),
                    "denied_agents": denied_list,
                    "caller_agent_id": caller,
                }
            )
        agent_ids = allowed
    elif agent_id:
        allowed, denied_list = filter_agents_for_read(caller, [agent_id])
        if not allowed:
            resolved_ns = get_namespace_for_agent(agent_id)
            return error_response(
                {
                    "error": "access_denied",
                    "reason": (
                        f"Cannot read memories for agent '{agent_id}' — resolved to"
                        f" namespace '{resolved_ns}', caller lacks read access"
                    ),
                    "hint": (
                        f"To search your own namespace, pass namespace='{resolved_ns}'"
                        f" directly instead of agent_id='{agent_id}', or omit agent_id"
                        " entirely."
                    ),
                    "denied_agents": denied_list,
                    "caller_agent_id": caller,
                }
            )

    min_score = arguments.get("min_score")
    threshold = float(min_score) if min_score is not None else None

    result = await recursive_retrieve(
        query=arguments["query"],
        agent_id="" if agent_ids else agent_id,
        agent_ids=agent_ids,
        team=arguments.get("team", ""),
        namespace=namespace,
        limit=arguments.get("limit", 20),
        refine=arguments.get("refine", False),
        threshold=threshold,
        tier=arguments.get("tier", "l2"),
        date_from=arguments.get("date_from", ""),
        date_to=arguments.get("date_to", ""),
        max_tokens=arguments.get("max_tokens"),
        memory_type=arguments.get("memory_type", ""),
        actor_type=arguments.get("actor_type", ""),
    )
    return success_response(result)


async def _handle_recall(arguments: dict) -> list[TextContent]:
    entity_name = arguments["entity"]
    related_name = arguments.get("related_to", "")
    namespace = arguments.get("namespace", "")
    as_of = (arguments.get("as_of") or "").strip()

    caller = resolve_caller(arguments)
    if err := require_caller(caller):
        return err

    if namespace and caller:
        denied = _rbac_gate(caller, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    entities = await search_entities(entity_name, namespace=namespace)
    if not entities:
        return error_response({"error": "entity_not_found", "entity": entity_name})

    eid = entities[0]["id"]
    facts = await get_entity_facts(eid, as_of=as_of)
    rels = await get_entity_relationships(eid)

    facts, rels = _filter_facts_rels_for_caller(caller, facts, rels)

    result = {
        "entity": entities[0],
        "facts": facts[:20],
        "relationships": rels[:20],
    }

    if related_name:
        rel_entities = await search_entities(related_name, namespace=namespace)
        if rel_entities:
            rel_eid = rel_entities[0]["id"]
            rel_facts = await get_entity_facts(rel_eid, as_of=as_of)
            rel_facts, _ = _filter_facts_rels_for_caller(caller, rel_facts, [])
            result["related_entity"] = rel_entities[0]
            result["related_facts"] = rel_facts[:10]
            shared = [
                r
                for r in rels
                if r["target_entity_id"] == rel_eid or r["source_entity_id"] == rel_eid
            ]
            result["shared_relationships"] = shared

    return success_response(result, default=str)


async def _handle_timeline(arguments: dict) -> list[TextContent]:
    from datetime import datetime, timedelta

    query = arguments["query"]
    agent_id = arguments.get("agent_id", "")
    namespace = arguments.get("namespace", "")
    days = int(arguments.get("days", 14))
    as_of = (arguments.get("as_of") or "").strip()
    caller = resolve_caller(arguments)
    if err := require_caller(caller):
        return err

    if namespace and caller:
        denied = _rbac_gate(caller, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    if agent_id:
        allowed, denied_list = filter_agents_for_read(caller, [agent_id])
        if not allowed:
            resolved_ns = get_namespace_for_agent(agent_id)
            return error_response(
                {
                    "error": "access_denied",
                    "reason": (
                        f"Cannot read timeline for agent '{agent_id}' — resolved to"
                        f" namespace '{resolved_ns}', caller lacks read access"
                    ),
                    "hint": (
                        f"Use archivist_namespaces(agent_id='{caller}') to see your"
                        " accessible namespaces."
                    ),
                    "denied_agents": denied_list,
                }
            )

    if as_of:
        anchor = datetime.strptime(as_of, "%Y-%m-%d").replace(tzinfo=UTC)
        date_to = as_of
        date_from = (anchor - timedelta(days=days)).strftime("%Y-%m-%d") if days > 0 else ""
    else:
        now = datetime.now(UTC)
        date_to = ""
        date_from = (now - timedelta(days=days)).strftime("%Y-%m-%d") if days > 0 else ""

    results = await search_vectors(
        query,
        agent_id=agent_id,
        namespace=namespace,
        date_from=date_from,
        date_to=date_to,
        limit=50,
    )
    results = [r for r in results if can_read_agent_memory(caller, r.get("agent_id", ""))]
    results.sort(key=lambda x: x.get("date", ""))

    timeline = [
        {
            "date": r["date"],
            "agent": r["agent_id"],
            "source": r["file_path"],
            "namespace": r.get("namespace", ""),
            "text": r["text"][:500],
            "score": r["score"],
        }
        for r in results
    ]

    return success_response({"query": query, "timeline": timeline[:30]})


async def _handle_insights(arguments: dict) -> list[TextContent]:
    topic = arguments["topic"]
    limit = arguments.get("limit", 10)
    namespace = arguments.get("namespace", "")
    caller = resolve_caller(arguments)
    if err := require_caller(caller):
        return err

    if namespace and caller:
        denied = _rbac_gate(caller, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    results = await search_vectors(topic, namespace=namespace, limit=limit * 3)
    results = [r for r in results if can_read_agent_memory(caller, r.get("agent_id", ""))]

    agents_seen: set[str] = set()
    insights = []
    teams_seen: set[str] = set()
    for r in results:
        key = f"{r['agent_id']}:{r['text'][:100]}"
        if key not in agents_seen:
            agents_seen.add(key)
            teams_seen.add(r["team"])
            insights.append(
                {
                    "agent": r["agent_id"],
                    "team": r["team"],
                    "date": r["date"],
                    "namespace": r.get("namespace", ""),
                    "text": r["text"][:500],
                    "source": r["file_path"],
                    "score": r["score"],
                }
            )
        if len(insights) >= limit:
            break

    return success_response(
        {
            "topic": topic,
            "teams_represented": list(teams_seen),
            "insights": insights,
        }
    )


async def _handle_deref(arguments: dict) -> list[TextContent]:
    """Dereference a single memory point by ID — returns full L2 text + metadata."""
    from archivist.core.config import QDRANT_COLLECTION
    from archivist.storage.qdrant import qdrant_client

    memory_id = arguments["memory_id"]
    agent_id = arguments.get("agent_id", "")

    client = qdrant_client()
    try:
        points = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[memory_id],
            with_payload=True,
        )
    except Exception as e:
        return error_response({"error": str(e)})

    if not points:
        return error_response({"error": "not_found", "memory_id": memory_id})

    p = points[0]
    payload = p.payload or {}

    if agent_id and not is_permissive_mode():
        ns = payload.get("namespace", "")
        if ns:
            denied = _rbac_gate(agent_id, "read", ns)
            if denied:
                return [TextContent(type="text", text=denied)]

    return success_response(
        {
            "memory_id": str(p.id),
            "uri": memory_uri(payload.get("namespace", ""), str(p.id)),
            "text": payload.get("text", ""),
            "l0": payload.get("l0", ""),
            "l1": payload.get("l1", ""),
            "agent_id": payload.get("agent_id", ""),
            "namespace": payload.get("namespace", ""),
            "file_path": payload.get("file_path", ""),
            "file_type": payload.get("file_type", ""),
            "date": payload.get("date", ""),
            "parent_id": payload.get("parent_id"),
            "is_parent": payload.get("is_parent", False),
            "importance_score": payload.get("importance_score", 0.5),
            "retention_class": payload.get("retention_class", "standard"),
            "version": payload.get("version", 1),
        }
    )


async def _handle_index(arguments: dict) -> list[TextContent]:
    """Return a compressed navigational index for a namespace."""
    agent_id = arguments.get("agent_id", "")
    namespace = arguments.get("namespace", "")

    if not namespace and agent_id:
        namespace = get_namespace_for_agent(agent_id)

    if namespace and agent_id and not is_permissive_mode():
        denied = _rbac_gate(agent_id, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    index_text = await build_namespace_index(namespace, agent_ids=[agent_id] if agent_id else None)
    return [TextContent(type="text", text=index_text)]


async def _handle_contradictions(arguments: dict) -> list[TextContent]:
    """Surface contradicting facts about an entity from different agents."""
    entity_name = arguments["entity"]
    namespace = arguments.get("namespace", "")

    entities = await search_entities(entity_name, limit=1, namespace=namespace)
    if not entities:
        return success_response(
            {
                "entity": entity_name,
                "contradictions": [],
                "note": "Entity not found in knowledge graph",
            }
        )

    eid = entities[0]["id"]
    contradictions = await detect_contradictions(eid)

    return success_response(
        {
            "entity": entity_name,
            "entity_id": eid,
            "contradictions": contradictions,
            "count": len(contradictions),
        }
    )


async def _handle_entity_brief(arguments: dict) -> list[TextContent]:
    """Structured knowledge card for a single entity."""
    entity_name = arguments["entity"]
    alias = (arguments.get("add_alias") or "").strip()
    as_of = (arguments.get("as_of") or "").strip()
    namespace = arguments.get("namespace", "")

    caller = resolve_caller(arguments)

    entities = await search_entities(entity_name, limit=1, namespace=namespace)
    if not entities:
        return error_response(
            {
                "error": f"Entity '{entity_name}' not found in knowledge graph",
                "hint": "Use archivist_recall or archivist_search first to discover entities.",
            }
        )

    eid = entities[0]["id"]

    if alias:
        await add_entity_alias(eid, alias)

    brief = await get_entity_brief(eid, as_of=as_of)
    if not brief:
        return error_response({"error": "Failed to build entity brief"})

    if caller and not is_permissive_mode():
        brief["facts"] = [
            f for f in brief["facts"] if can_read_agent_memory(caller, f.get("agent_id", ""))
        ]
        brief["fact_count"] = len(brief["facts"])

    return success_response(brief)


async def _handle_wake_up(arguments: dict) -> list[TextContent]:
    """Session bootstrap — return compact identity + critical facts + TOC."""
    agent_id = arguments.get("agent_id", "")
    namespace = arguments.get("namespace", "")

    if not namespace and agent_id:
        namespace = get_namespace_for_agent(agent_id)

    if namespace and agent_id and not is_permissive_mode():
        denied = _rbac_gate(agent_id, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    ctx = await get_cached_wake_up(namespace, agent_id=agent_id)
    if not ctx:
        ctx = await cache_wake_up(namespace, agent_id=agent_id)

    return [TextContent(type="text", text=format_wake_up_text(ctx, agent_id=agent_id))]


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, object] = {
    "archivist_search": _handle_search,
    "archivist_recall": _handle_recall,
    "archivist_timeline": _handle_timeline,
    "archivist_insights": _handle_insights,
    "archivist_deref": _handle_deref,
    "archivist_index": _handle_index,
    "archivist_contradictions": _handle_contradictions,
    "archivist_entity_brief": _handle_entity_brief,
    "archivist_wake_up": _handle_wake_up,
}
