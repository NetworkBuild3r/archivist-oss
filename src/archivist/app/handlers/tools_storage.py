"""MCP tool handlers — memory storage, merge, and compression."""

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime

from mcp.types import TextContent, Tool
from qdrant_client.models import PointStruct

import archivist.core.journal as journal
import archivist.core.metrics as m
import archivist.features.webhooks as webhooks
import archivist.lifecycle.curator_queue as curator_queue
import archivist.retrieval.hot_cache as hot_cache
from archivist.core.archivist_uri import memory_uri
from archivist.core.config import (
    CONFLICT_BLOCK_ON_STORE,
    CONFLICT_CHECK_ON_STORE,
    TEAM_MAP,
)
from archivist.core.rbac import get_namespace_config, get_namespace_for_agent
from archivist.features.embeddings import embed_batch, embed_text
from archivist.storage.collection_router import (
    collection_for,
    collections_for_query,
    ensure_collection,
)
from archivist.storage.compressed_index import invalidate_index_cache
from archivist.storage.graph import (
    add_fact,
    register_memory_points_batch,
    resolve_entity_id,
    upsert_entity,
)
from archivist.storage.qdrant import qdrant_client
from archivist.utils.chunking import _extract_needle_micro_chunks
from archivist.utils.text_utils import compute_memory_checksum
from archivist.write.conflict_detection import (
    _query_similar,
    check_for_conflicts,
    llm_adjudicated_dedup,
)
from archivist.write.contextual_augment import augment_chunk
from archivist.write.indexer import compute_ttl
from archivist.write.pre_extractor import extract_needle_entities, pre_extract

from ._common import (
    _rbac_gate,
    error_response,
    resolve_actor,
    success_response,
)

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
                "namespace": {
                    "type": "string",
                    "description": "Target namespace (default: auto-detect from agent_id)",
                    "default": "",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity names mentioned (optional, will auto-extract if empty)",
                    "default": [],
                },
                "importance_score": {
                    "type": "number",
                    "description": "0.0-1.0 importance score (higher = longer retention and retrieval boost)",
                    "default": 0.5,
                },
                "retention_class": {
                    "type": "string",
                    "enum": ["ephemeral", "standard", "durable", "permanent"],
                    "description": "How long to retain: ephemeral (auto-expire), standard (default decay), durable (no TTL, reduced decay), permanent (never decay, retrieval boost). Use permanent for critical facts like host IPs, person names, org structure.",
                    "default": "standard",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["experience", "skill", "general"],
                    "description": "Tag this memory as an experience (I did X), skill (how to do X), or general. Default general.",
                    "default": "general",
                },
                "thought_type": {
                    "type": "string",
                    "enum": [
                        "decision",
                        "lesson",
                        "constraint",
                        "insight",
                        "preference",
                        "milestone",
                        "correction",
                        "general",
                    ],
                    "description": "Semantic thought type for precise filtering. Auto-detected if omitted.",
                    "default": "",
                },
                "force_skip_conflict_check": {
                    "type": "boolean",
                    "description": "If true, skip vector similarity conflict check against other agents' memories (use sparingly).",
                    "default": False,
                },
                "actor_id": {
                    "type": "string",
                    "description": "Who produced this content (defaults to agent_id). Can be a human username, tool name, or system process.",
                    "default": "",
                },
                "actor_type": {
                    "type": "string",
                    "enum": ["agent", "human", "system", "tool"],
                    "description": "Type of actor storing this memory.",
                    "default": "agent",
                },
                "confidence": {
                    "type": "number",
                    "description": "0.0-1.0 confidence in this memory's accuracy (default based on actor_type).",
                    "default": -1,
                },
                "source_trace": {
                    "type": "object",
                    "description": "Structured origin context: {tool, session_id, upstream_source, parent_memory_id, extra}.",
                    "default": {},
                },
            },
            "required": ["text", "agent_id"],
        },
    ),
    Tool(
        name="archivist_delete",
        description=(
            "Soft-delete a memory by ID. Immediately hides it from all search paths "
            "(vector, BM25, needle registry) by marking it deleted in Qdrant and FTS, "
            "then enqueues a background hard-cascade. Returns in ~5 ms."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "Qdrant point ID of the memory to delete",
                },
                "agent_id": {"type": "string", "description": "Agent requesting the deletion"},
                "namespace": {
                    "type": "string",
                    "description": "Namespace (default: auto-detect from agent_id)",
                    "default": "",
                },
            },
            "required": ["memory_id", "agent_id"],
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
                "namespace": {
                    "type": "string",
                    "description": "Namespace for the merged result",
                    "default": "",
                },
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
    Tool(
        name="archivist_pin",
        description=(
            "Pin a memory or entity so it is never forgotten. "
            "Sets retention_class to 'permanent' and importance_score to 1.0. "
            "Use for critical facts: host IPs, person names, credentials, org structure, "
            "service ownership — anything the agent must never lose."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Calling agent"},
                "memory_id": {
                    "type": "string",
                    "description": "Qdrant point ID to pin (optional if entity_name given)",
                    "default": "",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Entity name to pin (optional if memory_id given)",
                    "default": "",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this is being pinned (stored in audit log)",
                    "default": "",
                },
                "namespace": {"type": "string", "description": "Namespace context", "default": ""},
            },
            "required": ["agent_id"],
        },
    ),
    Tool(
        name="archivist_unpin",
        description=(
            "Remove permanent retention from a memory or entity, "
            "returning it to 'standard' retention class."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Calling agent"},
                "memory_id": {
                    "type": "string",
                    "description": "Qdrant point ID to unpin (optional if entity_name given)",
                    "default": "",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Entity name to unpin (optional if memory_id given)",
                    "default": "",
                },
                "namespace": {"type": "string", "description": "Namespace context", "default": ""},
            },
            "required": ["agent_id"],
        },
    ),
]

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_store(arguments: dict) -> list[TextContent]:
    _t_store = time.monotonic()
    text = arguments["text"]
    agent_id = arguments["agent_id"]
    namespace = arguments.get("namespace", "") or get_namespace_for_agent(agent_id)
    entity_names = arguments.get("entities", [])
    importance = arguments.get("importance_score", 0.5)
    retention = arguments.get("retention_class", "standard")
    force_skip = bool(arguments.get("force_skip_conflict_check", False))

    actor_id, actor_type = resolve_actor(arguments)
    from archivist.core.provenance import SourceTrace, default_confidence

    raw_confidence = arguments.get("confidence", -1)
    confidence = (
        raw_confidence
        if isinstance(raw_confidence, int | float) and raw_confidence >= 0
        else default_confidence(actor_type)
    )
    _raw_trace = arguments.get("source_trace") or {}
    source_trace = (
        SourceTrace.from_dict(_raw_trace) if isinstance(_raw_trace, dict) else SourceTrace()
    )
    if not source_trace.tool:
        source_trace.tool = "archivist_store"

    if retention == "permanent":
        importance = max(importance, 1.0)

    denied = _rbac_gate(agent_id, "write", namespace)
    if denied:
        return [TextContent(type="text", text=denied)]

    if CONFLICT_CHECK_ON_STORE and not force_skip:
        _shared_vec, _shared_results = await _query_similar(text, namespace)
        cr = await check_for_conflicts(
            text, namespace, agent_id, _shared_vec=_shared_vec, _shared_results=_shared_results
        )
        if cr.has_conflict and CONFLICT_BLOCK_ON_STORE:
            m.inc(m.STORE_CONFLICT, {"namespace": namespace})
            webhooks.fire_background(
                "memory_conflict",
                {
                    "agent_id": agent_id,
                    "namespace": namespace,
                    "max_similarity": cr.max_similarity,
                    "conflicting_ids": cr.conflicting_ids,
                },
            )
            return error_response(
                {
                    "stored": False,
                    "conflict": True,
                    "max_similarity": cr.max_similarity,
                    "conflicting_ids": cr.conflicting_ids,
                    "recommendation": cr.recommendation,
                    "hint": "Set force_skip_conflict_check true to store anyway, or merge with conflicting memories.",
                }
            )
    else:
        _shared_results = None

    if not force_skip:
        dedup = await llm_adjudicated_dedup(
            text, namespace, agent_id, _shared_results=_shared_results
        )
        if dedup and dedup.action == "skip":
            return error_response(
                {
                    "stored": False,
                    "dedup_action": "skip",
                    "reason": "LLM determined this memory is a duplicate",
                    "existing_ids": dedup.existing_ids,
                    "decisions": dedup.decisions,
                }
            )
        if dedup and dedup.action == "merge":
            curator_queue.enqueue(
                "merge_memory",
                {
                    "new_text": text,
                    "agent_id": agent_id,
                    "namespace": namespace,
                    "existing_ids": dedup.existing_ids,
                    "decisions": dedup.decisions,
                },
            )
        if dedup and dedup.action == "delete_old":
            for d in dedup.decisions:
                if d.get("decision") == "delete":
                    curator_queue.enqueue(
                        "archive_memory",
                        {
                            "memory_ids": [d.get("existing_id", "")],
                            "reason": "superseded",
                        },
                    )

    ns_config = get_namespace_config(namespace)
    consistency = ns_config.consistency if ns_config else "eventual"

    pid = str(uuid.uuid4())

    _fact_kw = dict(
        retention_class=retention,
        namespace=namespace or "global",
        memory_id=pid,
        confidence=confidence,
        provenance=source_trace.tool or "explicit",
        actor_id=actor_id,
    )

    for ename in entity_names:
        try:
            eid = await upsert_entity(
                ename.strip(),
                retention_class=retention,
                namespace=namespace or "global",
                actor_id=actor_id,
                actor_type=actor_type,
            )
        except Exception as exc:
            # Regression guard: upsert_entity uses ON CONFLICT DO UPDATE and
            # should never raise a unique-constraint error in normal operation.
            # If it does (e.g. concurrent DDL migration, Postgres asyncpg
            # UniqueViolationError, or a misconfigured schema), we MUST NOT
            # return here — doing so aborts the entire store pipeline and
            # skips embedding/Qdrant/FTS/audit for all remaining entities.
            # Instead: resolve the existing row's ID and continue the loop.
            _err_str = str(exc).lower()
            if "unique" not in _err_str and "integrity" not in _err_str:
                raise
            m.inc(m.ENTITY_UPSERT_CONFLICTS, {"namespace": namespace or "global"})
            logger.warning(
                "entity_upsert.integrity_fallback",
                extra={
                    "entity_name": ename,
                    "namespace": namespace,
                    "error": str(exc),
                    "note": "ON CONFLICT was bypassed; resolving via SELECT",
                },
            )
            eid = await resolve_entity_id(ename.strip(), namespace or "global")
            if not eid:
                continue
        await add_fact(eid, text[:200], f"explicit/{agent_id}", agent_id, **_fact_kw)

    if not entity_names:
        try:
            eid = await upsert_entity(
                agent_id,
                "agent",
                retention_class=retention,
                namespace=namespace or "global",
                actor_id=actor_id,
                actor_type=actor_type,
            )
        except Exception as exc:
            _err_str = str(exc).lower()
            if "unique" not in _err_str and "integrity" not in _err_str:
                raise
            m.inc(m.ENTITY_UPSERT_CONFLICTS, {"namespace": namespace or "global"})
            logger.warning(
                "entity_upsert.integrity_fallback",
                extra={
                    "entity_name": agent_id,
                    "namespace": namespace,
                    "error": str(exc),
                    "note": "agent self-entity; resolving via SELECT",
                },
            )
            eid = await resolve_entity_id(agent_id, namespace or "global")
        if eid:
            await add_fact(eid, text[:200], f"explicit/{agent_id}", agent_id, **_fact_kw)

        _auto_hints = pre_extract(text)
        _auto_entities = _auto_hints.get("entities", [])
        _needle_entities = extract_needle_entities(text)
        from archivist.core.config import DEFAULT_CONFIDENCE_BY_ACTOR_TYPE

        _extracted_conf = DEFAULT_CONFIDENCE_BY_ACTOR_TYPE.get("extracted", 0.5)
        _extracted_fact_kw = dict(_fact_kw, confidence=_extracted_conf, provenance="deterministic")
        for ent in _auto_entities + _needle_entities:
            ename = ent["name"].strip()
            if ename and ename != agent_id:
                etype = ent.get("type", "unknown")
                try:
                    _eid = await upsert_entity(
                        ename,
                        etype,
                        retention_class=retention,
                        namespace=namespace or "global",
                        actor_id=actor_id,
                        actor_type=actor_type,
                    )
                except Exception as exc:
                    _err_str = str(exc).lower()
                    if "unique" not in _err_str and "integrity" not in _err_str:
                        raise
                    m.inc(m.ENTITY_UPSERT_CONFLICTS, {"namespace": namespace or "global"})
                    logger.warning(
                        "entity_upsert.integrity_fallback",
                        extra={
                            "entity_name": ename,
                            "namespace": namespace,
                            "error": str(exc),
                            "note": "auto-extract entity; resolving via SELECT",
                        },
                    )
                    _eid = await resolve_entity_id(ename, namespace or "global")
                    if not _eid:
                        continue
                await add_fact(
                    _eid, text[:200], f"explicit/{agent_id}", agent_id, **_extracted_fact_kw
                )
    else:
        _auto_hints = pre_extract(text)

    thought_type = (arguments.get("thought_type") or "").strip()
    if not thought_type:
        thought_type = _auto_hints.get("thought_type", "general")

    from archivist.core.config import TOPIC_ROUTING_ENABLED
    from archivist.retrieval.topic_detector import detect_topics

    _detected_topic = ""
    if TOPIC_ROUTING_ENABLED:
        _topics = detect_topics(text)
        _detected_topic = _topics[0] if _topics else ""

    embed_input = text
    from archivist.core.config import CONTEXTUAL_AUGMENTATION_ENABLED

    if CONTEXTUAL_AUGMENTATION_ENABLED:
        embed_input = augment_chunk(
            text,
            agent_id=agent_id,
            file_path=f"explicit/{agent_id}",
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            thought_type=thought_type,
            topic=_detected_topic,
            actor_id=actor_id,
            actor_type=actor_type,
        )
    vec = await embed_text(embed_input)
    client = qdrant_client()
    now = datetime.now(UTC)
    checksum = compute_memory_checksum(text, agent_id, namespace)

    ttl_expires_at = compute_ttl(namespace, importance=importance)

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
        "retention_class": retention,
        "memory_type": arguments.get("memory_type", "general"),
        "thought_type": thought_type,
        "representation_type": "chunk",
        "actor_id": actor_id,
        "actor_type": actor_type,
        "confidence": confidence,
        "source_trace": source_trace.to_dict(),
    }
    if retention in ("durable", "permanent"):
        ttl_expires_at = None
    if ttl_expires_at is not None:
        payload["ttl_expires_at"] = ttl_expires_at

    _coll = ensure_collection(namespace)

    from archivist.core.config import OUTBOX_ENABLED
    from archivist.storage.transaction import MemoryTransaction

    _primary_point = PointStruct(id=pid, vector=vec, payload=payload)

    from archivist.core.config import BM25_ENABLED

    # Generate micro-chunks for high-specificity tokens (IPs, crons, UUIDs, etc.)
    # Embedding must happen before the transaction (async LLM/embed call).
    _micro_chunks = _extract_needle_micro_chunks(text)
    _micro_points = []
    if _micro_chunks:
        from archivist.core.config import MAX_MICRO_CHUNKS_PER_MEMORY

        _micro_chunks = _micro_chunks[:MAX_MICRO_CHUNKS_PER_MEMORY]
        _micro_embed_inputs = _micro_chunks
        if CONTEXTUAL_AUGMENTATION_ENABLED:
            _micro_embed_inputs = [
                augment_chunk(
                    mc,
                    agent_id=agent_id,
                    file_path=f"explicit/{agent_id}",
                    date=now.strftime("%Y-%m-%d"),
                )
                for mc in _micro_chunks
            ]
        _micro_vecs = await embed_batch(_micro_embed_inputs)
        for mi, (mc, mv) in enumerate(zip(_micro_chunks, _micro_vecs)):
            _mc_id = str(uuid.uuid4())
            _mc_payload = {
                "agent_id": agent_id,
                "text": mc,
                "file_path": f"explicit/{agent_id}",
                "file_type": "explicit",
                "date": now.strftime("%Y-%m-%d"),
                "team": TEAM_MAP.get(agent_id, "unknown"),
                "chunk_index": mi + 1,
                "namespace": namespace,
                "version": 1,
                "consistency_level": consistency,
                "importance_score": importance,
                "retention_class": retention,
                "memory_type": arguments.get("memory_type", "general"),
                "thought_type": thought_type,
                "parent_id": pid,
                "is_parent": False,
                "actor_id": actor_id,
                "actor_type": actor_type,
                "confidence": confidence,
                "source_trace": source_trace.to_dict(),
            }
            if retention in ("durable", "permanent"):
                pass
            elif ttl_expires_at is not None:
                _mc_payload["ttl_expires_at"] = ttl_expires_at
            _micro_points.append(PointStruct(id=_mc_id, vector=mv, payload=_mc_payload))

    # Single atomic transaction: FTS5, needle registry, memory_points, and outbox
    # all commit together.  A crash at any point leaves nothing half-written.
    _now_iso = datetime.now(UTC).isoformat()
    async with MemoryTransaction() as txn:
        if BM25_ENABLED:
            await txn.upsert_fts_chunk(
                qdrant_id=pid,
                text=text,
                file_path=payload["file_path"],
                chunk_index=0,
                agent_id=agent_id,
                namespace=namespace,
                date=payload["date"],
                memory_type=arguments.get("memory_type", "general"),
                actor_id=actor_id,
                actor_type=actor_type,
            )
        await txn.register_needle_tokens(
            pid,
            text,
            namespace=namespace,
            agent_id=agent_id,
            actor_id=actor_id,
            actor_type=actor_type,
        )
        for mp in _micro_points:
            mc_text = mp.payload.get("text", "")
            mc_id = str(mp.id)
            if BM25_ENABLED:
                await txn.upsert_fts_chunk(
                    qdrant_id=mc_id,
                    text=mc_text,
                    file_path=f"explicit/{agent_id}",
                    chunk_index=mp.payload.get("chunk_index", 0),
                    agent_id=agent_id,
                    namespace=namespace,
                    date=now.strftime("%Y-%m-%d"),
                    memory_type=arguments.get("memory_type", "general"),
                    actor_id=actor_id,
                    actor_type=actor_type,
                )
            await txn.register_needle_tokens(
                mc_id,
                mc_text,
                namespace=namespace,
                agent_id=agent_id,
                actor_id=actor_id,
                actor_type=actor_type,
            )
        await txn.execute(
            """INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)
               VALUES (?, ?, 'primary', ?)
               ON CONFLICT (memory_id, qdrant_id) DO NOTHING""",
            (pid, pid, _now_iso),
        )
        if _micro_points:
            await txn.executemany(
                """INSERT INTO memory_points
                       (memory_id, qdrant_id, point_type, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT (memory_id, qdrant_id) DO NOTHING""",
                [(pid, str(mp.id), "micro_chunk", _now_iso) for mp in _micro_points],
            )
        txn.enqueue_qdrant_upsert(_coll, [_primary_point], memory_id=pid)
        if _micro_points:
            txn.enqueue_qdrant_upsert(_coll, _micro_points, memory_id=pid)

    # When the outbox is disabled, apply Qdrant writes inline (legacy behaviour).
    if not OUTBOX_ENABLED:
        client.upsert(
            collection_name=_coll,
            points=[_primary_point],
        )
        if _micro_points:
            client.upsert(collection_name=_coll, points=_micro_points)

    # Reverse HyDE: fire-and-forget — generate hypothetical questions in background
    from archivist.core.config import REVERSE_HYDE_ENABLED

    if REVERSE_HYDE_ENABLED:

        async def _reverse_hyde_background():
            from archivist.write.hyde import generate_reverse_hyde_questions

            _rh_questions = await generate_reverse_hyde_questions(text)
            if not _rh_questions:
                return
            _rh_vecs = await embed_batch(_rh_questions)
            _rh_points = []
            for qi, (q, qv) in enumerate(zip(_rh_questions, _rh_vecs)):
                _q_id = str(uuid.uuid4())
                _rh_trace = source_trace.with_parent(pid)
                _rh_points.append(
                    PointStruct(
                        id=_q_id,
                        vector=qv,
                        payload={
                            "agent_id": agent_id,
                            "text": text,
                            "file_path": f"explicit/{agent_id}",
                            "file_type": "reverse_hyde",
                            "date": now.strftime("%Y-%m-%d"),
                            "team": TEAM_MAP.get(agent_id, "unknown"),
                            "chunk_index": 0,
                            "namespace": namespace,
                            "version": 1,
                            "importance_score": importance,
                            "retention_class": retention,
                            "memory_type": arguments.get("memory_type", "general"),
                            "thought_type": thought_type,
                            "source_memory_id": pid,
                            "is_reverse_hyde": True,
                            "reverse_hyde_question": q,
                            "actor_id": actor_id,
                            "actor_type": actor_type,
                            "confidence": confidence,
                            "source_trace": _rh_trace.to_dict(),
                        },
                    )
                )
            if _rh_points:
                from archivist.core.config import OUTBOX_ENABLED
                from archivist.storage.transaction import MemoryTransaction

                _rh_mp_records = [
                    {"memory_id": pid, "qdrant_id": str(rp.id), "point_type": "reverse_hyde"}
                    for rp in _rh_points
                ]
                if OUTBOX_ENABLED:
                    from datetime import UTC, datetime

                    async with MemoryTransaction() as txn:
                        await txn.executemany(
                            """INSERT INTO memory_points
                                   (memory_id, qdrant_id, point_type, created_at)
                               VALUES (?, ?, ?, ?)
                               ON CONFLICT (memory_id, qdrant_id) DO NOTHING""",
                            [
                                (
                                    r["memory_id"],
                                    r["qdrant_id"],
                                    r["point_type"],
                                    datetime.now(UTC).isoformat(),
                                )
                                for r in _rh_mp_records
                            ],
                        )
                        txn.enqueue_qdrant_upsert(_coll, _rh_points, memory_id=pid)
                else:
                    client.upsert(collection_name=_coll, points=_rh_points)
                    await register_memory_points_batch(_rh_mp_records)
            logger.info(
                "reverse_hyde.background_complete",
                extra={"memory_id": pid, "question_count": len(_rh_questions)},
            )

        def _rh_done(task: asyncio.Task):
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.warning("Reverse HyDE background task failed for %s: %s", pid, exc)

        _rh_task = asyncio.create_task(_reverse_hyde_background(), name=f"reverse_hyde_{pid}")
        _rh_task.add_done_callback(_rh_done)

    # Synthetic question generation (background, non-blocking)
    from archivist.core.config import SYNTHETIC_QUESTIONS_ENABLED as _SQ_ENABLED

    if _SQ_ENABLED:

        async def _synthetic_questions_background():
            from archivist.write.synthetic_questions import generate_and_embed_synthetic_points

            _sq_trace = source_trace.with_parent(pid)
            base_payload = {
                "agent_id": agent_id,
                "text": text,
                "file_path": f"explicit/{agent_id}",
                "file_type": "explicit",
                "date": now.strftime("%Y-%m-%d"),
                "team": TEAM_MAP.get(agent_id, "unknown"),
                "chunk_index": 0,
                "namespace": namespace,
                "version": 1,
                "importance_score": importance,
                "retention_class": retention,
                "memory_type": arguments.get("memory_type", "general"),
                "thought_type": thought_type,
                "actor_id": actor_id,
                "actor_type": actor_type,
                "confidence": confidence,
                "source_trace": _sq_trace.to_dict(),
            }
            sq_points = await generate_and_embed_synthetic_points(
                chunk_point_id=pid,
                chunk_text=text,
                base_payload=base_payload,
            )
            if sq_points:
                from archivist.core.config import OUTBOX_ENABLED
                from archivist.storage.transaction import MemoryTransaction

                _sq_mp_records = [
                    {
                        "memory_id": pid,
                        "qdrant_id": str(sp.id),
                        "point_type": "synthetic_question",
                    }
                    for sp in sq_points
                ]
                if OUTBOX_ENABLED:
                    from datetime import UTC, datetime

                    async with MemoryTransaction() as txn:
                        await txn.executemany(
                            """INSERT INTO memory_points
                                   (memory_id, qdrant_id, point_type, created_at)
                               VALUES (?, ?, ?, ?)
                               ON CONFLICT (memory_id, qdrant_id) DO NOTHING""",
                            [
                                (
                                    r["memory_id"],
                                    r["qdrant_id"],
                                    r["point_type"],
                                    datetime.now(UTC).isoformat(),
                                )
                                for r in _sq_mp_records
                            ],
                        )
                        txn.enqueue_qdrant_upsert(_coll, sq_points, memory_id=pid)
                else:
                    client.upsert(collection_name=_coll, points=sq_points)
                    await register_memory_points_batch(_sq_mp_records)
            logger.info(
                "synthetic_questions.background_complete",
                extra={"memory_id": pid, "question_count": len(sq_points)},
            )

        def _sq_done(task: asyncio.Task):
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.warning("Synthetic questions background task failed for %s: %s", pid, exc)

        _sq_task = asyncio.create_task(_synthetic_questions_background(), name=f"synthetic_q_{pid}")
        _sq_task.add_done_callback(_sq_done)

    from archivist.core.audit import log_memory_event

    await log_memory_event(
        agent_id=agent_id,
        action="create",
        memory_id=pid,
        namespace=namespace,
        text_hash=checksum,
        version=1,
        metadata={
            "trigger": "api",
            "importance_score": importance,
            "retention_class": retention,
            "actor_id": actor_id,
            "actor_type": actor_type,
            "confidence": confidence,
            "source_trace": source_trace.to_dict(),
        },
    )

    hot_cache.invalidate_namespace(namespace)
    invalidate_index_cache(namespace)

    m.inc(m.STORE_TOTAL, {"namespace": namespace})
    webhooks.fire_background(
        "memory_store",
        {
            "memory_id": pid,
            "agent_id": agent_id,
            "namespace": namespace,
        },
    )

    journal.append_entry(
        memory_id=pid,
        agent_id=agent_id,
        namespace=namespace,
        text=text,
        memory_type=arguments.get("memory_type", "general"),
        importance=importance,
    )

    logger.info(
        "store_pipeline.complete",
        extra={
            "memory_id": pid,
            "namespace": namespace,
            "agent_id": agent_id,
            "chunk_count": 1,
            "micro_chunk_count": len(_micro_chunks),
            "entity_count": len(entity_names) if entity_names else 1,
            "reverse_hyde_queued": REVERSE_HYDE_ENABLED,
            "duration_ms": int((time.monotonic() - _t_store) * 1000),
        },
    )

    return success_response(
        {
            "stored": True,
            "memory_id": pid,
            "uri": memory_uri(namespace, pid),
            "namespace": namespace,
            "entities": entity_names or [agent_id],
            "version": 1,
        }
    )


async def _handle_merge(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]
    memory_ids = arguments["memory_ids"]
    strategy = arguments["strategy"]
    namespace = arguments.get("namespace", "")

    from archivist.lifecycle.merge import merge_memories

    result = await merge_memories(memory_ids, strategy, agent_id, namespace)

    hot_cache.invalidate_namespace(namespace)

    invalidate_index_cache(namespace)

    return success_response(result, default=str)


async def _handle_compress(arguments: dict) -> list[TextContent]:
    """Archive memory blocks and return a compact summary.

    Supports format="flat" (default, single paragraph) and
    format="structured" (Goal/Progress/Decisions/Next Steps JSON).
    """
    from archivist.write.compaction import (
        compact_flat,
        compact_structured,
        format_structured_summary,
    )

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

    client = qdrant_client()
    texts: list[tuple[str, str]] = []
    source_agent_ids: list[str] = []
    _colls = collections_for_query("")
    for mid in memory_ids:
        try:
            points = client.retrieve(
                collection_name=_colls[0],
                ids=[mid],
                with_payload=True,
            )
            if points:
                pl = points[0].payload or {}
                texts.append((str(points[0].id), pl.get("text", "")))
                aid = pl.get("agent_id") or ""
                if aid:
                    source_agent_ids.append(str(aid))
        except Exception as e:
            logger.warning("Compress: failed to retrieve %s: %s", mid, e)

    if not texts:
        return error_response({"error": "no memories found for given IDs"})

    multi_agent = len(set(source_agent_ids)) > 1

    if user_summary:
        summary_text = user_summary
        structured_data = None
    elif fmt == "structured":
        structured_data = await compact_structured(
            texts, previous_summary=previous_summary, multi_agent=multi_agent
        )
        summary_text = format_structured_summary(structured_data)
    else:
        summary_text = await compact_flat(texts, multi_agent=multi_agent)
        structured_data = None

    store_result = await _handle_store(
        {
            "text": f"[Compressed summary]\n{summary_text}",
            "agent_id": agent_id,
            "namespace": namespace,
            "importance_score": 0.8,
            "memory_type": "general",
            "force_skip_conflict_check": True,
        }
    )

    stored_data = {}
    try:
        stored_data = json.loads(store_result[0].text)
    except Exception:
        pass

    if not stored_data.get("stored"):
        return error_response(
            {
                "compressed": False,
                "error": "Failed to store compressed summary",
                "store_result": stored_data,
            }
        )

    curator_queue.enqueue(
        "archive_memory",
        {
            "memory_ids": memory_ids,
            "reason": "compressed",
            "compressed_into": stored_data.get("memory_id", ""),
        },
    )

    hot_cache.invalidate_namespace(namespace)

    invalidate_index_cache(namespace)

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


async def _handle_pin(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]
    memory_id = arguments.get("memory_id", "").strip()
    entity_name = arguments.get("entity_name", "").strip()
    reason = arguments.get("reason", "")
    namespace = arguments.get("namespace", "") or get_namespace_for_agent(agent_id)

    if not memory_id and not entity_name:
        return error_response({"error": "Provide memory_id or entity_name (or both)"})

    denied = _rbac_gate(agent_id, "write", namespace)
    if denied:
        return [TextContent(type="text", text=denied)]

    pinned = []

    if memory_id:
        _pin_coll = collection_for(namespace)
        client = qdrant_client()
        try:
            points = client.retrieve(collection_name=_pin_coll, ids=[memory_id], with_payload=True)
            if points:
                client.set_payload(
                    collection_name=_pin_coll,
                    payload={"retention_class": "permanent", "importance_score": 1.0},
                    points=[memory_id],
                )
                pinned.append({"type": "memory", "id": memory_id})
            else:
                return error_response({"error": f"Memory {memory_id} not found"})
        except Exception as e:
            return error_response({"error": f"Failed to pin memory: {e}"})

    if entity_name:
        from archivist.storage.sqlite_pool import pool

        async with pool.write() as conn:
            cur = await conn.execute(
                "SELECT id FROM entities WHERE name = ? AND namespace = ?",
                (entity_name, namespace or "global"),
            )
            row = await cur.fetchone()
            if row:
                await conn.execute(
                    "UPDATE entities SET retention_class='permanent' WHERE id=?", (row["id"],)
                )
                await conn.execute(
                    "UPDATE facts SET retention_class='permanent' WHERE entity_id=? AND is_active=1",
                    (row["id"],),
                )
                pinned.append({"type": "entity", "name": entity_name, "id": row["id"]})
            else:
                eid = await upsert_entity(
                    entity_name, retention_class="permanent", namespace=namespace or "global"
                )
                pinned.append({"type": "entity", "name": entity_name, "id": eid, "created": True})

    from archivist.core.audit import log_memory_event

    await log_memory_event(
        agent_id=agent_id,
        action="pin",
        memory_id=memory_id or entity_name,
        namespace=namespace,
        text_hash="",
        version=0,
        metadata={"reason": reason, "pinned": pinned},
    )

    hot_cache.invalidate_namespace(namespace)

    invalidate_index_cache(namespace)

    return success_response(
        {
            "pinned": True,
            "items": pinned,
            "retention_class": "permanent",
            "reason": reason,
        }
    )


async def _handle_unpin(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]
    memory_id = arguments.get("memory_id", "").strip()
    entity_name = arguments.get("entity_name", "").strip()
    namespace = arguments.get("namespace", "") or get_namespace_for_agent(agent_id)

    if not memory_id and not entity_name:
        return error_response({"error": "Provide memory_id or entity_name (or both)"})

    denied = _rbac_gate(agent_id, "write", namespace)
    if denied:
        return [TextContent(type="text", text=denied)]

    unpinned = []

    if memory_id:
        _unpin_coll = collection_for(namespace)
        client = qdrant_client()
        try:
            client.set_payload(
                collection_name=_unpin_coll,
                payload={"retention_class": "standard", "importance_score": 0.5},
                points=[memory_id],
            )
            unpinned.append({"type": "memory", "id": memory_id})
        except Exception as e:
            return error_response({"error": f"Failed to unpin memory: {e}"})

    if entity_name:
        from archivist.storage.sqlite_pool import pool

        async with pool.write() as conn:
            cur = await conn.execute("SELECT id FROM entities WHERE name = ?", (entity_name,))
            row = await cur.fetchone()
            if row:
                await conn.execute(
                    "UPDATE entities SET retention_class='standard' WHERE id=?", (row["id"],)
                )
                await conn.execute(
                    "UPDATE facts SET retention_class='standard' WHERE entity_id=? AND retention_class='permanent'",
                    (row["id"],),
                )
                unpinned.append({"type": "entity", "name": entity_name, "id": row["id"]})

    hot_cache.invalidate_namespace(namespace)

    invalidate_index_cache(namespace)

    return success_response(
        {
            "unpinned": True,
            "items": unpinned,
            "retention_class": "standard",
        }
    )


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------


async def _handle_delete(arguments: dict) -> list[TextContent]:
    """Soft-delete a memory by ID.

    Sets ``deleted=True`` on the Qdrant point immediately (so it vanishes from
    all search paths), marks the FTS entry as excluded, then enqueues a
    background hard-cascade via ``curator_queue``.  Returns in ~5 ms.
    """
    from archivist.core.rbac import get_namespace_for_agent
    from archivist.lifecycle.memory_lifecycle import soft_delete_memory

    memory_id = arguments.get("memory_id", "").strip()
    agent_id = arguments.get("agent_id", "").strip()
    namespace = arguments.get("namespace", "").strip()

    if not memory_id:
        return error_response({"error": "memory_id is required"})
    if not agent_id:
        return error_response({"error": "agent_id is required"})

    ns_err = _rbac_gate(agent_id, "write", namespace)
    if ns_err:
        return ns_err
    if not namespace:
        namespace = get_namespace_for_agent(agent_id)

    try:
        result = await soft_delete_memory(memory_id, namespace)
    except Exception as e:
        logger.error("archivist_delete failed for %s: %s", memory_id, e)
        return error_response({"error": str(e)})

    hot_cache.invalidate_namespace(namespace)

    invalidate_index_cache(namespace)

    return success_response(
        {
            "deleted": True,
            "memory_id": memory_id,
            "namespace": namespace,
            **result,
        }
    )


HANDLERS: dict[str, object] = {
    "archivist_store": _handle_store,
    "archivist_merge": _handle_merge,
    "archivist_compress": _handle_compress,
    "archivist_pin": _handle_pin,
    "archivist_unpin": _handle_unpin,
    "archivist_delete": _handle_delete,
}
