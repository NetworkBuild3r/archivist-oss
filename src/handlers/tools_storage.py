"""MCP tool handlers — memory storage, merge, and compression."""

import json
import hashlib
import uuid
import logging
from datetime import datetime, timezone

from mcp.types import Tool, TextContent
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from embeddings import embed_text
from graph import upsert_entity, add_fact
from config import (
    QDRANT_URL, QDRANT_COLLECTION, TEAM_MAP,
    CONFLICT_CHECK_ON_STORE, CONFLICT_BLOCK_ON_STORE,
)
from conflict_detection import check_for_conflicts, llm_adjudicated_dedup
from rbac import get_namespace_for_agent, get_namespace_config
from archivist_uri import memory_uri
import hot_cache
import journal
import metrics as m
import webhooks
import curator_queue

from ._common import _rbac_gate, error_response, success_response

logger = logging.getLogger("archivist.mcp")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
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
                "memory_type": {
                    "type": "string",
                    "enum": ["experience", "skill", "general"],
                    "description": "Tag this memory as an experience (I did X), skill (how to do X), or general. Default general.",
                    "default": "general",
                },
                "force_skip_conflict_check": {
                    "type": "boolean",
                    "description": "If true, skip vector similarity conflict check against other agents' memories (use sparingly).",
                    "default": False,
                },
            },
            "required": ["text", "agent_id"],
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
    Tool(
        name="archivist_compress",
        description=(
            "Archive memory blocks and return a compact summary. "
            "Agents call this mid-session to manage context budget. "
            "Originals are archived (kept but excluded from default search)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent requesting compression"},
                "namespace": {"type": "string", "description": "Target namespace"},
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory point IDs to compress",
                },
                "summary": {
                    "type": "string",
                    "description": "Optional agent-provided summary. If omitted, LLM generates one.",
                    "default": "",
                },
                "format": {
                    "type": "string",
                    "enum": ["flat", "structured"],
                    "description": "Output format: 'flat' (default, single paragraph) or 'structured' (Goal/Progress/Decisions/Next Steps).",
                    "default": "flat",
                },
                "previous_summary": {
                    "type": "string",
                    "description": "Optional prior structured summary JSON to merge with (for incremental compaction).",
                    "default": "",
                },
            },
            "required": ["agent_id", "namespace", "memory_ids"],
        },
    ),
]

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_store(arguments: dict) -> list[TextContent]:
    text = arguments["text"]
    agent_id = arguments["agent_id"]
    namespace = arguments.get("namespace", "") or get_namespace_for_agent(agent_id)
    entity_names = arguments.get("entities", [])
    importance = arguments.get("importance_score", 0.5)
    force_skip = bool(arguments.get("force_skip_conflict_check", False))

    denied = _rbac_gate(agent_id, "write", namespace)
    if denied:
        return [TextContent(type="text", text=denied)]

    if CONFLICT_CHECK_ON_STORE and not force_skip:
        cr = await check_for_conflicts(text, namespace, agent_id)
        if cr.has_conflict and CONFLICT_BLOCK_ON_STORE:
            m.inc(m.STORE_CONFLICT, {"namespace": namespace})
            webhooks.fire_background("memory_conflict", {
                "agent_id": agent_id, "namespace": namespace,
                "max_similarity": cr.max_similarity,
                "conflicting_ids": cr.conflicting_ids,
            })
            return error_response({
                "stored": False,
                "conflict": True,
                "max_similarity": cr.max_similarity,
                "conflicting_ids": cr.conflicting_ids,
                "recommendation": cr.recommendation,
                "hint": "Set force_skip_conflict_check true to store anyway, or merge with conflicting memories.",
            })

    if not force_skip:
        dedup = await llm_adjudicated_dedup(text, namespace, agent_id)
        if dedup and dedup.action == "skip":
            return error_response({
                "stored": False,
                "dedup_action": "skip",
                "reason": "LLM determined this memory is a duplicate",
                "existing_ids": dedup.existing_ids,
                "decisions": dedup.decisions,
            })
        if dedup and dedup.action == "merge":
            curator_queue.enqueue("merge_memory", {
                "new_text": text, "agent_id": agent_id, "namespace": namespace,
                "existing_ids": dedup.existing_ids, "decisions": dedup.decisions,
            })
        if dedup and dedup.action == "delete_old":
            for d in dedup.decisions:
                if d.get("decision") == "delete":
                    curator_queue.enqueue("archive_memory", {
                        "memory_ids": [d.get("existing_id", "")],
                        "reason": "superseded",
                    })

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
    pid = str(uuid.uuid4())
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
        "memory_type": arguments.get("memory_type", "general"),
    }
    if ttl_expires_at is not None:
        payload["ttl_expires_at"] = ttl_expires_at

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[PointStruct(id=pid, vector=vec, payload=payload)],
    )

    from config import BM25_ENABLED
    if BM25_ENABLED:
        from graph import upsert_fts_chunk
        upsert_fts_chunk(
            qdrant_id=pid,
            text=text,
            file_path=payload["file_path"],
            chunk_index=0,
            agent_id=agent_id,
            namespace=namespace,
            date=payload["date"],
            memory_type=arguments.get("memory_type", "general"),
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

    hot_cache.invalidate_namespace(namespace)

    m.inc(m.STORE_TOTAL, {"namespace": namespace})
    webhooks.fire_background("memory_store", {
        "memory_id": pid, "agent_id": agent_id, "namespace": namespace,
    })

    journal.append_entry(
        memory_id=pid,
        agent_id=agent_id,
        namespace=namespace,
        text=text,
        memory_type=arguments.get("memory_type", "general"),
        importance=importance,
    )

    return success_response({
        "stored": True,
        "memory_id": pid,
        "uri": memory_uri(namespace, pid),
        "namespace": namespace,
        "entities": entity_names or [agent_id],
        "version": 1,
    })


async def _handle_merge(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]
    memory_ids = arguments["memory_ids"]
    strategy = arguments["strategy"]
    namespace = arguments.get("namespace", "")

    from merge import merge_memories
    result = await merge_memories(memory_ids, strategy, agent_id, namespace)
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _handle_compress(arguments: dict) -> list[TextContent]:
    """Archive memory blocks and return a compact summary.

    Supports format="flat" (default, single paragraph) and
    format="structured" (Goal/Progress/Decisions/Next Steps JSON).
    """
    from compaction import compact_structured, compact_flat, format_structured_summary

    agent_id = arguments["agent_id"]
    namespace = arguments["namespace"]
    memory_ids = arguments["memory_ids"]
    user_summary = arguments.get("summary", "")
    fmt = arguments.get("format", "flat")
    previous_summary = arguments.get("previous_summary", "")

    denied = _rbac_gate(agent_id, "write", namespace)
    if denied:
        return [TextContent(type="text", text=denied)]

    if not memory_ids:
        return error_response({"error": "memory_ids required"})

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    texts: list[tuple[str, str]] = []
    for mid in memory_ids:
        try:
            points = client.retrieve(
                collection_name=QDRANT_COLLECTION, ids=[mid], with_payload=True,
            )
            if points:
                texts.append((str(points[0].id), (points[0].payload or {}).get("text", "")))
        except Exception as e:
            logger.warning("Compress: failed to retrieve %s: %s", mid, e)

    if not texts:
        return error_response({"error": "no memories found for given IDs"})

    if user_summary:
        summary_text = user_summary
        structured_data = None
    elif fmt == "structured":
        structured_data = await compact_structured(texts, previous_summary=previous_summary)
        summary_text = format_structured_summary(structured_data)
    else:
        summary_text = await compact_flat(texts)
        structured_data = None

    store_result = await _handle_store({
        "text": f"[Compressed summary]\n{summary_text}",
        "agent_id": agent_id,
        "namespace": namespace,
        "importance_score": 0.8,
        "memory_type": "general",
        "force_skip_conflict_check": True,
    })

    stored_data = {}
    try:
        stored_data = json.loads(store_result[0].text)
    except Exception:
        pass

    if not stored_data.get("stored"):
        return error_response({
            "compressed": False,
            "error": "Failed to store compressed summary",
            "store_result": stored_data,
        })

    curator_queue.enqueue("archive_memory", {
        "memory_ids": memory_ids,
        "reason": "compressed",
        "compressed_into": stored_data.get("memory_id", ""),
    })

    hot_cache.invalidate_namespace(namespace)

    result = {
        "compressed": True,
        "compressed_memory_id": stored_data.get("memory_id"),
        "uri": stored_data.get("uri"),
        "format": fmt,
        "summary_l0": summary_text[:200],
        "archived_count": len(memory_ids),
        "archived_ids": memory_ids,
    }
    if structured_data:
        result["structured_summary"] = structured_data

    return success_response(result)


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, object] = {
    "archivist_store": _handle_store,
    "archivist_merge": _handle_merge,
    "archivist_compress": _handle_compress,
}
