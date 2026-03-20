"""Archivist MCP server — exposes memory tools via HTTP SSE (Model Context Protocol)."""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timezone

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rlm_retriever import recursive_retrieve, search_vectors
from graph import (
    search_entities, get_entity_facts, get_entity_relationships,
    upsert_entity, add_fact, init_schema, get_db,
)
from embeddings import embed_text
from config import QDRANT_URL, QDRANT_COLLECTION, TEAM_MAP
from rbac import (
    check_access, get_namespace_for_agent, list_accessible_namespaces,
    get_namespace_config,
)

logger = logging.getLogger("archivist.mcp")

server = Server("archivist")


def _rbac_gate(agent_id: str, action: str, namespace: str) -> str | None:
    """Return error JSON string if access denied, None if allowed."""
    policy = check_access(agent_id, action, namespace)
    if not policy.allowed:
        return json.dumps({"error": "access_denied", "reason": policy.reason})
    return None


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
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
                    "agent_id": {"type": "string", "description": "Filter to one agent's memories (optional)", "default": ""},
                    "agent_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search only these agents' memories (OR). Omit for fleet-wide or use agent_id for a single agent.",
                    },
                    "caller_agent_id": {
                        "type": "string",
                        "description": "Identity of the invoking agent — used for RBAC when reading others' namespaces. Defaults to agent_id if set.",
                    },
                    "namespace": {"type": "string", "description": "Memory namespace to search (optional, auto-detect from agent_id)", "default": ""},
                    "team": {"type": "string", "description": "Filter by team (optional)", "default": ""},
                    "refine": {"type": "boolean", "description": "Use LLM refinement for higher quality (slower). Default true.", "default": True},
                    "limit": {"type": "integer", "description": "Max chunks to refine/synthesize after retrieval", "default": 20},
                    "min_score": {
                        "type": "number",
                        "description": "Minimum vector similarity (0–1). Overrides RETRIEVAL_THRESHOLD for this call; omit to use env default. Use 0 to disable filtering.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="archivist_recall",
            description=(
                "Graph-based multi-hop recall. Finds entities and their relationships/facts. "
                "Use for questions like 'What do we know about X?' or 'How does X relate to Y?'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity name to look up"},
                    "related_to": {"type": "string", "description": "Optional second entity to find connections", "default": ""},
                    "agent_id": {"type": "string", "description": "Calling agent for RBAC (optional)", "default": ""},
                    "namespace": {"type": "string", "description": "Memory namespace scope (optional)", "default": ""},
                },
                "required": ["entity"],
            },
        ),
        Tool(
            name="archivist_store",
            description=(
                "Explicitly store a memory/fact with entity extraction. "
                "Use when you want to ensure something specific is remembered."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The memory or fact to store"},
                    "agent_id": {"type": "string", "description": "Which agent is storing this"},
                    "namespace": {"type": "string", "description": "Target namespace (default: auto-detect from agent_id)", "default": ""},
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity names mentioned (optional, will auto-extract if empty)",
                        "default": [],
                    },
                    "importance_score": {"type": "number", "description": "0.0-1.0 importance score (higher = longer retention)", "default": 0.5},
                },
                "required": ["text", "agent_id"],
            },
        ),
        Tool(
            name="archivist_timeline",
            description=(
                "Get a chronological timeline of memories about a topic. "
                "Use for questions like 'What happened with X over the last 2 weeks?'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to build timeline for"},
                    "agent_id": {"type": "string", "description": "Filter to specific agent (optional)", "default": ""},
                    "namespace": {"type": "string", "description": "Memory namespace to search (optional)", "default": ""},
                    "days": {"type": "integer", "description": "How many days back to search", "default": 14},
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
                    "agent_id": {"type": "string", "description": "Calling agent for RBAC (optional)", "default": ""},
                    "namespace": {"type": "string", "description": "Namespace scope (optional)", "default": ""},
                    "limit": {"type": "integer", "description": "Max insights to return", "default": 10},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="archivist_namespaces",
            description="List memory namespaces accessible to the calling agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "The calling agent's ID"},
                },
                "required": ["agent_id"],
            },
        ),
        Tool(
            name="archivist_audit_trail",
            description="View audit log for a specific memory or agent activity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Calling agent"},
                    "memory_id": {"type": "string", "description": "Specific memory ID to audit (optional)", "default": ""},
                    "target_agent": {"type": "string", "description": "Agent whose activity to view (optional)", "default": ""},
                    "limit": {"type": "integer", "description": "Max entries to return", "default": 50},
                },
                "required": ["agent_id"],
            },
        ),
        Tool(
            name="archivist_merge",
            description=(
                "Merge conflicting memory entries using a specified strategy. "
                "Strategies: latest (keep newest), concat (join all), semantic (LLM synthesis), manual (flag for review)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Calling agent"},
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Memory point IDs to merge",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["latest", "concat", "semantic", "manual"],
                        "description": "Merge strategy",
                    },
                    "namespace": {"type": "string", "description": "Namespace for the merged result", "default": ""},
                },
                "required": ["agent_id", "memory_ids", "strategy"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "archivist_search":
            return await _handle_search(arguments)
        elif name == "archivist_recall":
            return await _handle_recall(arguments)
        elif name == "archivist_store":
            return await _handle_store(arguments)
        elif name == "archivist_timeline":
            return await _handle_timeline(arguments)
        elif name == "archivist_insights":
            return await _handle_insights(arguments)
        elif name == "archivist_namespaces":
            return await _handle_namespaces(arguments)
        elif name == "archivist_audit_trail":
            return await _handle_audit_trail(arguments)
        elif name == "archivist_merge":
            return await _handle_merge(arguments)
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e, exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


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

    caller = (arguments.get("caller_agent_id") or "").strip() or agent_id

    if namespace and caller:
        denied = _rbac_gate(caller, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    from rbac import filter_agents_for_read

    if agent_ids:
        allowed, denied_list = filter_agents_for_read(caller, agent_ids)
        if not allowed:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "access_denied",
                    "reason": "Caller cannot read any of the requested agents' namespaces",
                    "denied_agents": denied_list,
                    "caller_agent_id": caller,
                }),
            )]
        agent_ids = allowed
    elif agent_id:
        allowed, denied_list = filter_agents_for_read(caller, [agent_id])
        if not allowed:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "access_denied",
                    "reason": f"Cannot read memories for agent '{agent_id}'",
                    "denied_agents": denied_list,
                    "caller_agent_id": caller,
                }),
            )]

    min_score = arguments.get("min_score")
    threshold = float(min_score) if min_score is not None else None

    result = await recursive_retrieve(
        query=arguments["query"],
        agent_id="" if agent_ids else agent_id,
        agent_ids=agent_ids,
        team=arguments.get("team", ""),
        namespace=namespace,
        limit=arguments.get("limit", 20),
        refine=arguments.get("refine", True),
        threshold=threshold,
    )
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_recall(arguments: dict) -> list[TextContent]:
    entity_name = arguments["entity"]
    related_name = arguments.get("related_to", "")

    entities = search_entities(entity_name)
    if not entities:
        return [TextContent(type="text", text=json.dumps({"error": f"Entity '{entity_name}' not found in knowledge graph"}))]

    eid = entities[0]["id"]
    facts = get_entity_facts(eid)
    rels = get_entity_relationships(eid)

    result = {
        "entity": entities[0],
        "facts": facts[:20],
        "relationships": rels[:20],
    }

    if related_name:
        rel_entities = search_entities(related_name)
        if rel_entities:
            rel_eid = rel_entities[0]["id"]
            rel_facts = get_entity_facts(rel_eid)
            result["related_entity"] = rel_entities[0]
            result["related_facts"] = rel_facts[:10]
            shared = [
                r for r in rels
                if r["target_entity_id"] == rel_eid or r["source_entity_id"] == rel_eid
            ]
            result["shared_relationships"] = shared

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _handle_store(arguments: dict) -> list[TextContent]:
    text = arguments["text"]
    agent_id = arguments["agent_id"]
    namespace = arguments.get("namespace", "") or get_namespace_for_agent(agent_id)
    entity_names = arguments.get("entities", [])
    importance = arguments.get("importance_score", 0.5)

    denied = _rbac_gate(agent_id, "write", namespace)
    if denied:
        return [TextContent(type="text", text=denied)]

    ns_config = get_namespace_config(namespace)
    consistency = ns_config.consistency if ns_config else "eventual"

    for ename in entity_names:
        eid = upsert_entity(ename.strip())
        add_fact(eid, text, f"explicit/{agent_id}", agent_id)

    if not entity_names:
        eid = upsert_entity(agent_id, "agent")
        add_fact(eid, text, f"explicit/{agent_id}", agent_id)

    vec = await embed_text(text)
    client = QdrantClient(url=QDRANT_URL, timeout=30)
    pid = hashlib.md5(f"explicit:{agent_id}:{text[:100]}".encode()).hexdigest()
    now = datetime.now(timezone.utc)
    checksum = hashlib.sha256(f"{text}:{agent_id}:{namespace}".encode()).hexdigest()

    ttl_expires_at = None
    if ns_config and ns_config.ttl_days is not None:
        if importance < 0.9:
            from datetime import timedelta
            ttl_expires_at = int((now + timedelta(days=ns_config.ttl_days)).timestamp())

    payload = {
        "agent_id": agent_id,
        "text": text,
        "file_path": f"explicit/{agent_id}",
        "file_type": "explicit",
        "date": now.strftime("%Y-%m-%d"),
        "team": TEAM_MAP.get(agent_id, "unknown"),
        "chunk_index": 0,
        "namespace": namespace,
        "version": 1,
        "consistency_level": consistency,
        "checksum": checksum,
        "importance_score": importance,
    }
    if ttl_expires_at is not None:
        payload["ttl_expires_at"] = ttl_expires_at

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[PointStruct(id=pid, vector=vec, payload=payload)],
    )

    from audit import log_memory_event
    await log_memory_event(
        agent_id=agent_id,
        action="create",
        memory_id=pid,
        namespace=namespace,
        text_hash=checksum,
        version=1,
        metadata={"trigger": "api", "importance_score": importance},
    )

    return [TextContent(type="text", text=json.dumps({
        "stored": True,
        "memory_id": pid,
        "namespace": namespace,
        "entities": entity_names or [agent_id],
        "version": 1,
    }))]


async def _handle_timeline(arguments: dict) -> list[TextContent]:
    query = arguments["query"]
    agent_id = arguments.get("agent_id", "")
    namespace = arguments.get("namespace", "")
    results = await search_vectors(query, agent_id=agent_id, namespace=namespace, limit=50)

    results.sort(key=lambda x: x.get("date", ""))

    timeline = []
    for r in results:
        timeline.append({
            "date": r["date"],
            "agent": r["agent_id"],
            "source": r["file_path"],
            "namespace": r.get("namespace", ""),
            "text": r["text"][:500],
            "score": r["score"],
        })

    return [TextContent(type="text", text=json.dumps({"query": query, "timeline": timeline[:30]}, indent=2))]


async def _handle_insights(arguments: dict) -> list[TextContent]:
    topic = arguments["topic"]
    limit = arguments.get("limit", 10)
    namespace = arguments.get("namespace", "")

    results = await search_vectors(topic, namespace=namespace, limit=limit * 3)

    agents_seen = set()
    insights = []
    teams_seen = set()
    for r in results:
        key = f"{r['agent_id']}:{r['text'][:100]}"
        if key not in agents_seen:
            agents_seen.add(key)
            teams_seen.add(r["team"])
            insights.append({
                "agent": r["agent_id"],
                "team": r["team"],
                "date": r["date"],
                "namespace": r.get("namespace", ""),
                "text": r["text"][:500],
                "source": r["file_path"],
                "score": r["score"],
            })
        if len(insights) >= limit:
            break

    return [TextContent(type="text", text=json.dumps({
        "topic": topic,
        "teams_represented": list(teams_seen),
        "insights": insights,
    }, indent=2))]


async def _handle_namespaces(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]
    namespaces = list_accessible_namespaces(agent_id)
    return [TextContent(type="text", text=json.dumps({
        "agent_id": agent_id,
        "default_namespace": get_namespace_for_agent(agent_id),
        "accessible_namespaces": namespaces,
    }, indent=2))]


async def _handle_audit_trail(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]

    from audit import get_audit_trail, get_agent_activity
    memory_id = arguments.get("memory_id", "")
    target_agent = arguments.get("target_agent", "")
    limit = arguments.get("limit", 50)

    if memory_id:
        entries = get_audit_trail(memory_id, limit=limit)
    elif target_agent:
        entries = get_agent_activity(target_agent, limit=limit)
    else:
        entries = get_agent_activity("", limit=limit)

    return [TextContent(type="text", text=json.dumps({"entries": entries}, indent=2, default=str))]


async def _handle_merge(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]
    memory_ids = arguments["memory_ids"]
    strategy = arguments["strategy"]
    namespace = arguments.get("namespace", "")

    from merge import merge_memories
    result = await merge_memories(memory_ids, strategy, agent_id, namespace)
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
