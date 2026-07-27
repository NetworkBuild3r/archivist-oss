"""MCP tool handlers — memory storage, merge, and compression."""

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

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
    DEFAULT_CONFIDENCE_BY_ACTOR_TYPE,
    TEAM_MAP,
)
from archivist.core.provenance import SourceTrace, default_confidence
from archivist.core.rbac import get_namespace_config, get_namespace_for_agent
from archivist.features.embeddings import embed_batch, embed_text
from archivist.retrieval.topic_detector import detect_topics
from archivist.storage.collection_router import (
    collection_for,
    collections_for_query,
    ensure_collection,
)
from archivist.storage.graph import (
    _is_postgres,
    add_fact,
    register_memory_points_batch,
    upsert_entity,
)
from archivist.storage.qdrant import qdrant_client
from archivist.storage.transaction import MemoryTransaction
from archivist.utils.chunking import _extract_needle_micro_chunks
from archivist.utils.text_utils import compute_memory_checksum
from archivist.write.conflict_detection import (
    _query_similar,
    check_for_conflicts,
    conflict_vec_for_primary_embed,
    llm_adjudicated_dedup,
)
from archivist.write.contextual_augment import augment_chunk
from archivist.write.indexer import compute_ttl
from archivist.write.pre_extractor import extract_needle_entities, pre_extract

from ._common import error_response, require_rbac, resolve_actor, success_response

# NOTE (INIT-022/SPEC-008, ac-5): `SourceTrace`/`default_confidence`, `detect_topics`,
# `DEFAULT_CONFIDENCE_BY_ACTOR_TYPE`, and `MemoryTransaction` are hoisted here because no
# test rebinds them at call time. The feature-flag constants below (`OUTBOX_ENABLED`,
# `BM25_ENABLED`, `REVERSE_HYDE_ENABLED`, `SYNTHETIC_QUESTIONS_ENABLED`,
# `CONTEXTUAL_AUGMENTATION_ENABLED`, `TOPIC_ROUTING_ENABLED`, `MAX_MICRO_CHUNKS_PER_MEMORY`)
# and `log_memory_event` are deliberately NOT hoisted and stay as function-local imports in
# `_handle_store` — the existing test suite (e.g. `tests/system/conftest.py`'s `_enable_outbox`
# autouse fixture, `tests/integration/mcp/test_provenance_handlers.py`) monkeypatches these on
# their *source* modules (`archivist.core.config`, `archivist.core.audit`) expecting the
# re-import inside the handler to observe the patched value at call time; a top-of-file import
# would bind the pre-patch value once at module-import time and silently stop honoring those
# per-test overrides.

logger = logging.getLogger("archivist.mcp")

# INIT-003/SPEC-006 — coach provenance envelope validation (size/enum).
_SENSITIVITY_VALUES = frozenset({"standard", "sensitive", "secret", "health", "public"})
_STATEMENT_KINDS = frozenset({"user", "inferred"})
_MAX_SUBJECT_LEN = 128
_MAX_PURPOSE_LEN = 128
_MAX_SOURCE_LEN = 256
_MAX_CORRECTION_OF_LEN = 64

# INIT-005/SPEC-006 — store success lag hint when ARCHIVIST_EMBED_DEFER path used.
# Clients must not assume hybrid/vector rank is ready at ack (ADR-005 GR-LAG-001).
_SEARCHABLE_LAG_HINT_DEFERRED = "vector_rank_may_lag_until_outbox_drain"


def _validate_store_provenance(arguments: dict) -> dict | list[TextContent]:
    """Parse additive provenance args; return dict or an error response list.

    INIT-003/SPEC-006 — validates size/enum so free-form fields cannot become
    unbounded secret exfil channels. Empty/omitted fields keep defaults so
    older clients keep working.
    """
    source = str(arguments.get("source") or "").strip()
    subject = str(arguments.get("subject") or "").strip()
    purpose = str(arguments.get("purpose") or "").strip()
    sensitivity = str(arguments.get("sensitivity") or "standard").strip().lower() or "standard"
    statement_kind = str(arguments.get("statement_kind") or "user").strip().lower() or "user"
    correction_of = str(arguments.get("correction_of") or "").strip()

    if len(source) > _MAX_SOURCE_LEN:
        return error_response(
            {
                "error": "invalid_provenance",
                "field": "source",
                "reason": f"max {_MAX_SOURCE_LEN} chars",
            }
        )
    if len(subject) > _MAX_SUBJECT_LEN:
        return error_response(
            {
                "error": "invalid_provenance",
                "field": "subject",
                "reason": f"max {_MAX_SUBJECT_LEN} chars",
            }
        )
    if len(purpose) > _MAX_PURPOSE_LEN:
        return error_response(
            {
                "error": "invalid_provenance",
                "field": "purpose",
                "reason": f"max {_MAX_PURPOSE_LEN} chars",
            }
        )
    if sensitivity not in _SENSITIVITY_VALUES:
        return error_response(
            {
                "error": "invalid_provenance",
                "field": "sensitivity",
                "reason": f"must be one of {sorted(_SENSITIVITY_VALUES)}",
            }
        )
    if statement_kind not in _STATEMENT_KINDS:
        return error_response(
            {
                "error": "invalid_provenance",
                "field": "statement_kind",
                "reason": "must be 'user' or 'inferred'",
            }
        )
    if len(correction_of) > _MAX_CORRECTION_OF_LEN:
        return error_response(
            {
                "error": "invalid_provenance",
                "field": "correction_of",
                "reason": f"max {_MAX_CORRECTION_OF_LEN} chars",
            }
        )

    raw_confidence = arguments.get("confidence", -1)
    if isinstance(raw_confidence, int | float) and raw_confidence >= 0:
        if raw_confidence > 1.0:
            return error_response(
                {
                    "error": "invalid_provenance",
                    "field": "confidence",
                    "reason": "must be between 0.0 and 1.0",
                }
            )

    return {
        "source": source,
        "subject": subject,
        "purpose": purpose,
        "sensitivity": sensitivity,
        "statement_kind": statement_kind,
        "correction_of": correction_of,
    }


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="archivist_store",
        description=(
            "Explicitly store a memory/fact with entity extraction and optional "
            "coach provenance (source, subject, sensitivity, purpose, statement_kind). "
            "Ack after durable graph + outbox commit (not Qdrant sync). Success JSON "
            "includes duration_ms, stage_timings, and embed_deferred; when "
            "ARCHIVIST_EMBED_DEFER deferred the primary embed, also searchable_lag_hint "
            "and searchable_lag_metric (vector rank may lag until outbox drain — "
            "FTS/graph remain usable; empty vector hit is cite-or-refuse OK). "
            "Never returns embedding vectors. Call archivist_index afterward for a "
            "fresh navigational index (live rebuild; store also busts search hot cache)."
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
                "source": {
                    "type": "string",
                    "description": "Provenance source label (e.g. session, harness, import). Max 256 chars.",
                    "default": "",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject/topic key for pre-rank filters (max 128 chars).",
                    "default": "",
                },
                "sensitivity": {
                    "type": "string",
                    "enum": ["standard", "sensitive", "secret", "health", "public"],
                    "description": "Sensitivity class for pre-rank filters.",
                    "default": "standard",
                },
                "purpose": {
                    "type": "string",
                    "description": "Purpose/use tag for pre-rank filters (max 128 chars).",
                    "default": "",
                },
                "statement_kind": {
                    "type": "string",
                    "enum": ["user", "inferred"],
                    "description": "Whether the statement is user-stated or inferred.",
                    "default": "user",
                },
                "correction_of": {
                    "type": "string",
                    "description": (
                        "Prior memory_id this store corrects. After durable write, "
                        "links the new memory as superseding the prior id (SPEC-007)."
                    ),
                    "default": "",
                },
            },
            "required": ["text", "agent_id"],
        },
    ),
    Tool(
        name="archivist_delete",
        description=(
            "Forget path (ADR archivist_forget): mode=delete soft-deletes with "
            "background hard-cascade; mode=suppress hides from default recall "
            "without erase. Namespace write RBAC required. Alias name: forget → delete."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "Qdrant point ID of the memory to forget",
                },
                "agent_id": {"type": "string", "description": "Agent requesting the mutation"},
                "namespace": {
                    "type": "string",
                    "description": "Namespace (default: auto-detect from agent_id)",
                    "default": "",
                },
                "mode": {
                    "type": "string",
                    "enum": ["delete", "suppress"],
                    "description": (
                        "delete (default): soft-delete/tombstone. "
                        "suppress: hide from default search/recall; record remains."
                    ),
                    "default": "delete",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional audit reason (not stored as memory text).",
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


async def _persist_background_points(
    coll: str,
    points: list[PointStruct],
    mp_records: list[dict],
    memory_id: str,
) -> None:
    """Persist background-task points (reverse-HyDE / synthetic-questions) plus
    their ``memory_points`` rows.

    Extracted from ``_reverse_hyde_background`` and
    ``_synthetic_questions_background``, which each duplicated this ~30-line
    branch inline (INIT-022/SPEC-008, M8). Modeled on the equivalent
    ``_persist_points`` helper in ``archivist.write.indexer``
    (INIT-022/SPEC-001, M13) — same ``OUTBOX_ENABLED`` branching, but without
    that helper's ``force_transaction``/``extra_txn_work`` options, which
    neither background task needs.
    """
    from archivist.core.config import OUTBOX_ENABLED

    if OUTBOX_ENABLED:
        async with MemoryTransaction() as txn:
            await txn.executemany(
                """INSERT OR IGNORE INTO memory_points
                       (memory_id, qdrant_id, point_type, created_at)
                   VALUES (?, ?, ?, ?)""",
                [
                    (
                        r["memory_id"],
                        r["qdrant_id"],
                        r["point_type"],
                        datetime.now(UTC).isoformat(),
                    )
                    for r in mp_records
                ],
            )
            txn.enqueue_qdrant_upsert(coll, points, memory_id=memory_id)
    else:
        from archivist.storage.outbox import warn_legacy_inline_qdrant_once

        warn_legacy_inline_qdrant_once()
        qdrant_client().upsert(collection_name=coll, points=points)
        await register_memory_points_batch(mp_records)


async def _extract_and_store_entities(
    text: str,
    agent_id: str,
    *,
    namespace: str,
    retention: str,
    actor_id: str,
    actor_type: str,
    fact_kw: dict,
) -> dict:
    """Auto-extract entities from ``text`` and store each as an entity/fact pair.

    Combines ``pre_extract`` hints with ``extract_needle_entities`` output,
    skips the agent's own name, and returns the ``pre_extract`` hints dict
    (used by the caller for thought-type detection).

    Extracted from ``_handle_store``'s inline auto-entity-extraction block
    (INIT-022/SPEC-008, M7) — pure extraction, no behavior change.
    """
    _auto_hints = pre_extract(text)
    _auto_entities = _auto_hints.get("entities", [])
    _needle_entities = extract_needle_entities(text)

    _extracted_conf = DEFAULT_CONFIDENCE_BY_ACTOR_TYPE.get("extracted", 0.5)
    _extracted_fact_kw = dict(fact_kw, confidence=_extracted_conf, provenance="deterministic")
    for ent in _auto_entities + _needle_entities:
        ename = ent["name"].strip()
        if ename and ename != agent_id:
            etype = ent.get("type", "unknown")
            _eid = await upsert_entity(
                ename,
                etype,
                retention_class=retention,
                namespace=namespace or "global",
                actor_id=actor_id,
                actor_type=actor_type,
            )
            await add_fact(_eid, text[:200], f"explicit/{agent_id}", agent_id, **_extracted_fact_kw)
    return _auto_hints


async def _reverse_hyde_background(
    *,
    text: str,
    pid: str,
    agent_id: str,
    namespace: str,
    now: datetime,
    importance: float,
    retention: str,
    memory_type: str,
    thought_type: str,
    actor_id: str,
    actor_type: str,
    confidence: float,
    source_trace: SourceTrace,
    coll: str,
) -> None:
    """Fire-and-forget: generate hypothetical questions for ``text`` and persist them.

    Promoted from a nested closure inside ``_handle_store`` to a module-level
    function taking explicit parameters instead of capturing locals via
    closure (INIT-022/SPEC-008, M7) — pure extraction, no behavior change.
    """
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
                    "memory_type": memory_type,
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
        _rh_mp_records = [
            {"memory_id": pid, "qdrant_id": str(rp.id), "point_type": "reverse_hyde"}
            for rp in _rh_points
        ]
        await _persist_background_points(coll, _rh_points, _rh_mp_records, pid)
    logger.info(
        "reverse_hyde.background_complete",
        extra={"memory_id": pid, "question_count": len(_rh_questions)},
    )


async def _synthetic_questions_background(
    *,
    text: str,
    pid: str,
    agent_id: str,
    namespace: str,
    now: datetime,
    importance: float,
    retention: str,
    memory_type: str,
    thought_type: str,
    actor_id: str,
    actor_type: str,
    confidence: float,
    source_trace: SourceTrace,
    coll: str,
) -> None:
    """Fire-and-forget: generate synthetic questions for ``text`` and persist them.

    Promoted from a nested closure inside ``_handle_store`` to a module-level
    function taking explicit parameters instead of capturing locals via
    closure (INIT-022/SPEC-008, M7) — pure extraction, no behavior change.
    """
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
        "memory_type": memory_type,
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
        _sq_mp_records = [
            {
                "memory_id": pid,
                "qdrant_id": str(sp.id),
                "point_type": "synthetic_question",
            }
            for sp in sq_points
        ]
        await _persist_background_points(coll, sq_points, _sq_mp_records, pid)
    logger.info(
        "synthetic_questions.background_complete",
        extra={"memory_id": pid, "question_count": len(sq_points)},
    )


def _is_graph_pool_uninitialized(exc: BaseException) -> bool:
    """True when SQLite/Postgres graph pool was never started (INIT-003/SPEC-004)."""
    msg = str(exc).lower()
    return "not initialized" in msg and (
        "pool" in msg or "sqlite" in msg or "asyncpg" in msg or "graph" in msg
    )


# INIT-005/SPEC-003 — optional pre-ack gates hard-skip when remaining ack budget
# falls below this floor (or is fully expired). Sized to leave headroom for
# durable graph commit + outbox enqueue (GR-DUR-001); never applies to outbox.
_ACK_OPTIONAL_GATE_MIN_REMAINING_MS = 250

# Counter name used via m.inc without owning metrics.py (SPEC-002 owns metrics).
_STORE_ACK_HARD_SKIP_TOTAL = "archivist_store_ack_hard_skip_total"


def _should_hard_skip_optional_gates(ack_budget) -> bool:
    """True when optional write gates must hard-skip under STORE_ACK_BUDGET_MS.

    INIT-005/SPEC-003 / ADR-005: conflict, LLM dedup, and optional entity extract
    may hard-skip when the ack budget is expired or remaining ms is below
    ``_ACK_OPTIONAL_GATE_MIN_REMAINING_MS``. Durable graph + outbox are never
    skipped by this predicate.
    """
    if ack_budget.is_expired():
        return True
    return ack_budget.remaining_ms() < _ACK_OPTIONAL_GATE_MIN_REMAINING_MS


def _emit_ack_hard_skip(gate: str, *, namespace: str, ack_budget) -> None:
    """Observable hard-skip (log + counter); never includes fact text (SM-002)."""
    m.inc(_STORE_ACK_HARD_SKIP_TOTAL, {"gate": gate, "namespace": namespace or "global"})
    logger.info(
        "store_pipeline.hard_skip",
        extra={
            "gate": gate,
            "namespace": namespace or "global",
            "budget": ack_budget.summary(),
            "min_remaining_ms": _ACK_OPTIONAL_GATE_MIN_REMAINING_MS,
        },
    )


async def _handle_store(arguments: dict) -> list[TextContent]:
    _t_store = time.monotonic()
    text = arguments["text"]
    agent_id = arguments["agent_id"]
    namespace = arguments.get("namespace", "") or get_namespace_for_agent(agent_id)
    entity_names = arguments.get("entities", [])
    importance = arguments.get("importance_score", 0.5)
    retention = arguments.get("retention_class", "standard")
    force_skip = bool(arguments.get("force_skip_conflict_check", False))

    # INIT-003/SPEC-006: validate additive provenance before RBAC/side effects.
    provenance = _validate_store_provenance(arguments)
    if isinstance(provenance, list):
        return provenance

    actor_id, actor_type = resolve_actor(arguments)

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

    denied = require_rbac(agent_id, "write", namespace)
    if denied:
        return denied

    try:
        return await _handle_store_inner(
            arguments=arguments,
            text=text,
            agent_id=agent_id,
            namespace=namespace,
            entity_names=entity_names,
            importance=importance,
            retention=retention,
            force_skip=force_skip,
            actor_id=actor_id,
            actor_type=actor_type,
            confidence=confidence,
            source_trace=source_trace,
            provenance=provenance,
            t_store=_t_store,
        )
    except RuntimeError as exc:
        # Fail-fast when the graph pool was never initialized — do not wait on Qdrant.
        if _is_graph_pool_uninitialized(exc):
            logger.error(
                "store_pipeline.graph_pool_unavailable",
                extra={
                    "namespace": namespace,
                    "agent_id": agent_id,
                    "error": str(exc),
                    "duration_ms": int((time.monotonic() - _t_store) * 1000),
                },
            )
            return error_response(
                {
                    "stored": False,
                    "error": "graph_pool_unavailable",
                    "reason": "Graph storage pool is not initialized",
                    # Namespace only — never echo other tenants' data.
                    "namespace": namespace,
                }
            )
        raise


async def _handle_store_inner(
    *,
    arguments: dict,
    text: str,
    agent_id: str,
    namespace: str,
    entity_names: list,
    importance: float,
    retention: str,
    force_skip: bool,
    actor_id: str,
    actor_type: str,
    confidence: float,
    source_trace: SourceTrace,
    provenance: dict,
    t_store: float,
) -> list[TextContent]:
    """Durable store body after RBAC (INIT-003/SPEC-004 ack path)."""
    from archivist.core.config import STORE_ACK_BUDGET_MS
    from archivist.core.latency_budget import LatencyBudget

    ack_budget = LatencyBudget(max_ms=STORE_ACK_BUDGET_MS)
    prov_source = provenance.get("source", "")
    prov_subject = provenance.get("subject", "")
    prov_purpose = provenance.get("purpose", "")
    prov_sensitivity = provenance.get("sensitivity", "standard")
    prov_statement_kind = provenance.get("statement_kind", "user")
    correction_of = provenance.get("correction_of", "")

    # INIT-005/SPEC-002: store stage timings (observability only — no hard-skip /
    # embed-reuse here; those belong to SPEC-003 / SPEC-004).
    stage_timings: dict[str, float] = {}

    # INIT-005/SPEC-003: hard-skip optional conflict/dedup under ack budget.
    # Outbox + graph commit below are never gated by this predicate (GR-DUR-001).
    # INIT-005/SPEC-004: _shared_vec stays None when conflict does not run (no reuse).
    _shared_vec = None
    _shared_results = None
    if CONFLICT_CHECK_ON_STORE and not force_skip:
        if _should_hard_skip_optional_gates(ack_budget):
            _emit_ack_hard_skip("conflict", namespace=namespace, ack_budget=ack_budget)
            stage_timings["conflict_ms"] = 0.0
        else:
            # Bound pre-txn Qdrant similarity; fail-open on timeout/dead backend.
            _t_conflict = time.monotonic()  # INIT-005/SPEC-002
            _shared_vec, _shared_results = await _query_similar(text, namespace)
            cr = await check_for_conflicts(
                text,
                namespace,
                agent_id,
                _shared_vec=_shared_vec,
                _shared_results=_shared_results,
            )
            stage_timings["conflict_ms"] = round((time.monotonic() - _t_conflict) * 1000, 1)
            m.observe(m.STORE_CONFLICT_MS, stage_timings["conflict_ms"])
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
                        "hint": (
                            "Set force_skip_conflict_check true to store anyway, "
                            "or merge with conflicting memories."
                        ),
                    }
                )

    if not force_skip:
        if _should_hard_skip_optional_gates(ack_budget):
            dedup = None
            _emit_ack_hard_skip("dedup", namespace=namespace, ack_budget=ack_budget)
        else:
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
            await curator_queue.enqueue(
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
                    await curator_queue.enqueue(
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
        eid = await upsert_entity(
            ename.strip(),
            retention_class=retention,
            namespace=namespace or "global",
            actor_id=actor_id,
            actor_type=actor_type,
        )
        await add_fact(eid, text[:200], f"explicit/{agent_id}", agent_id, **_fact_kw)

    if not entity_names:
        eid = await upsert_entity(
            agent_id,
            "agent",
            retention_class=retention,
            namespace=namespace or "global",
            actor_id=actor_id,
            actor_type=actor_type,
        )
        await add_fact(eid, text[:200], f"explicit/{agent_id}", agent_id, **_fact_kw)

        # INIT-005/SPEC-003: optional auto-extract is a quality gate — hard-skip
        # under ack budget; agent entity/fact above remains (durable graph).
        if _should_hard_skip_optional_gates(ack_budget):
            _emit_ack_hard_skip("extract", namespace=namespace, ack_budget=ack_budget)
            _auto_hints = pre_extract(text)
        else:
            _auto_hints = await _extract_and_store_entities(
                text,
                agent_id,
                namespace=namespace,
                retention=retention,
                actor_id=actor_id,
                actor_type=actor_type,
                fact_kw=_fact_kw,
            )
    else:
        _auto_hints = pre_extract(text)

    thought_type = (arguments.get("thought_type") or "").strip()
    if not thought_type:
        thought_type = _auto_hints.get("thought_type", "general")

    from archivist.core.config import TOPIC_ROUTING_ENABLED

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
    # INIT-005/SPEC-002 — primary embed stage timing.
    # INIT-005/SPEC-004 — reuse conflict _shared_vec when embed_input is
    # byte-identical to the conflict-query text (same store call / namespace).
    # INIT-005/SPEC-005 — ARCHIVIST_EMBED_DEFER: skip blocking primary embed when
    # outbox can fill vectors on drain (ADR-005; default false).
    from archivist.core.config import ARCHIVIST_EMBED_DEFER, OUTBOX_ENABLED

    _embed_defer = bool(ARCHIVIST_EMBED_DEFER and OUTBOX_ENABLED)
    _primary_embed_deferred = False
    _t_embed = time.monotonic()
    _reused_vec = conflict_vec_for_primary_embed(
        conflict_text=text,
        embed_input=embed_input,
        shared_vec=_shared_vec,
    )
    if _reused_vec is not None:
        vec = _reused_vec
        logger.info(
            "store_pipeline.embed_reuse_hit",
            extra={"namespace": namespace},
        )
    elif _embed_defer:
        # Placeholder — drain embeds before Qdrant upsert (GR-LAG-001).
        vec = []
        _primary_embed_deferred = True
        logger.info(
            "store_pipeline.embed_deferred",
            extra={"namespace": namespace, "role": "primary"},
        )
    else:
        vec = await embed_text(embed_input)
        if _shared_vec is not None:
            logger.info(
                "store_pipeline.embed_reuse_miss",
                extra={"namespace": namespace},
            )
    stage_timings["embed_ms"] = round((time.monotonic() - _t_embed) * 1000, 1)
    m.observe(m.STORE_EMBED_MS, stage_timings["embed_ms"])
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
        "tier_label": "l2",
        # INIT-003/SPEC-006 — coach provenance envelope (also on memory_chunks)
        "source": prov_source,
        "subject": prov_subject,
        "purpose": prov_purpose,
        "sensitivity": prov_sensitivity,
        "statement_kind": prov_statement_kind,
    }
    if retention in ("durable", "permanent"):
        ttl_expires_at = None
    if ttl_expires_at is not None:
        payload["ttl_expires_at"] = ttl_expires_at

    _coll = ensure_collection(namespace)

    if _primary_embed_deferred:
        # Dict form with empty vector — drain fills before PointStruct upsert.
        _primary_point: Any = {"id": pid, "vector": [], "payload": payload}
    else:
        _primary_point = PointStruct(id=pid, vector=vec, payload=payload)

    from archivist.core.config import BM25_ENABLED

    # Generate micro-chunks for high-specificity tokens (IPs, crons, UUIDs, etc.)
    # Embedding must happen before the transaction unless embed-defer is on
    # (INIT-005/SPEC-005 — micro embeds also block ack when sync).
    _micro_chunks = _extract_needle_micro_chunks(text)
    _micro_points: list[Any] = []
    _micro_embed_inputs_by_id: dict[str, str] = {}
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
        _micro_vecs: list[list[float]] | None = None
        if not _embed_defer:
            _micro_vecs = await embed_batch(_micro_embed_inputs)
        for mi, mc in enumerate(_micro_chunks):
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
            if _embed_defer:
                _micro_points.append({"id": _mc_id, "vector": [], "payload": _mc_payload})
                _micro_embed_inputs_by_id[_mc_id] = _micro_embed_inputs[mi]
            else:
                assert _micro_vecs is not None
                _micro_points.append(
                    PointStruct(id=_mc_id, vector=_micro_vecs[mi], payload=_mc_payload)
                )

    # Single atomic transaction: FTS5, needle registry, memory_points, and outbox
    # all commit together.  A crash at any point leaves nothing half-written.
    # Ack boundary (OUTBOX_ENABLED default-on): durable SQLite/Postgres + outbox
    # row — not Qdrant sync (INIT-003/SPEC-004).
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
                importance=importance,
                tier_label=payload.get("tier_label", "l2"),
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
            # PointStruct or deferred dict (INIT-005/SPEC-005).
            if isinstance(mp, dict):
                mc_payload = mp.get("payload") or {}
                mc_id = str(mp["id"])
            else:
                mc_payload = mp.payload or {}
                mc_id = str(mp.id)
            mc_text = mc_payload.get("text", "")
            if BM25_ENABLED:
                await txn.upsert_fts_chunk(
                    qdrant_id=mc_id,
                    text=mc_text,
                    file_path=f"explicit/{agent_id}",
                    chunk_index=mc_payload.get("chunk_index", 0),
                    agent_id=agent_id,
                    namespace=namespace,
                    date=now.strftime("%Y-%m-%d"),
                    memory_type=arguments.get("memory_type", "general"),
                    actor_id=actor_id,
                    actor_type=actor_type,
                    importance=float(mc_payload.get("importance_score", importance)),
                    tier_label=mc_payload.get("tier_label", "l2"),
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
            """INSERT OR IGNORE INTO memory_points (memory_id, qdrant_id, point_type, created_at)
               VALUES (?, ?, 'primary', ?)""",
            (pid, pid, _now_iso),
        )
        if _micro_points:
            await txn.executemany(
                """INSERT OR IGNORE INTO memory_points
                       (memory_id, qdrant_id, point_type, created_at)
                   VALUES (?, ?, ?, ?)""",
                [
                    (
                        pid,
                        str(mp["id"] if isinstance(mp, dict) else mp.id),
                        "micro_chunk",
                        _now_iso,
                    )
                    for mp in _micro_points
                ],
            )
        # INIT-003/SPEC-006: persist SPEC-002 provenance columns on primary chunk.
        # When BM25 upsert already created the row, ON CONFLICT patches envelope fields;
        # when BM25 is off, this INSERT is the durable chunk row.
        await txn.execute(
            """INSERT INTO memory_chunks (
                   qdrant_id, text, file_path, chunk_index, agent_id, namespace, date,
                   memory_type, actor_id, actor_type, importance, tier_label,
                   source, subject, confidence, sensitivity, purpose, statement_kind,
                   created_at, updated_at
               ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(qdrant_id) DO UPDATE SET
                   source=excluded.source,
                   subject=excluded.subject,
                   confidence=excluded.confidence,
                   sensitivity=excluded.sensitivity,
                   purpose=excluded.purpose,
                   statement_kind=excluded.statement_kind,
                   updated_at=excluded.updated_at,
                   actor_id=excluded.actor_id,
                   actor_type=excluded.actor_type""",
            (
                pid,
                text,
                payload["file_path"],
                agent_id,
                namespace,
                payload["date"],
                arguments.get("memory_type", "general"),
                actor_id,
                actor_type,
                importance,
                payload.get("tier_label", "l2"),
                prov_source,
                prov_subject,
                confidence,
                prov_sensitivity,
                prov_purpose,
                prov_statement_kind,
                _now_iso,
                _now_iso,
            ),
        )
        if _primary_embed_deferred:
            txn.enqueue_qdrant_upsert(
                _coll,
                [_primary_point],
                memory_id=pid,
                embed_deferred=True,
                embed_inputs={pid: embed_input},
            )
        else:
            txn.enqueue_qdrant_upsert(_coll, [_primary_point], memory_id=pid)
        if _micro_points:
            if _embed_defer:
                txn.enqueue_qdrant_upsert(
                    _coll,
                    _micro_points,
                    memory_id=pid,
                    embed_deferred=True,
                    embed_inputs=_micro_embed_inputs_by_id,
                )
            else:
                txn.enqueue_qdrant_upsert(_coll, _micro_points, memory_id=pid)

    # When the outbox is disabled, apply Qdrant writes inline (legacy behaviour).
    # Embed-defer requires outbox; this branch always has filled PointStructs.
    if not OUTBOX_ENABLED:
        from archivist.storage.outbox import warn_legacy_inline_qdrant_once

        warn_legacy_inline_qdrant_once()
        client.upsert(
            collection_name=_coll,
            points=[_primary_point],
        )
        if _micro_points:
            client.upsert(collection_name=_coll, points=_micro_points)

    # Reverse HyDE: fire-and-forget — generate hypothetical questions in background
    from archivist.core.config import REVERSE_HYDE_ENABLED

    if REVERSE_HYDE_ENABLED:

        def _rh_done(task: asyncio.Task):
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.warning("Reverse HyDE background task failed for %s: %s", pid, exc)

        _rh_task = asyncio.create_task(
            _reverse_hyde_background(
                text=text,
                pid=pid,
                agent_id=agent_id,
                namespace=namespace,
                now=now,
                importance=importance,
                retention=retention,
                memory_type=arguments.get("memory_type", "general"),
                thought_type=thought_type,
                actor_id=actor_id,
                actor_type=actor_type,
                confidence=confidence,
                source_trace=source_trace,
                coll=_coll,
            ),
            name=f"reverse_hyde_{pid}",
        )
        _rh_task.add_done_callback(_rh_done)

    # Synthetic question generation (background, non-blocking)
    from archivist.core.config import SYNTHETIC_QUESTIONS_ENABLED as _SQ_ENABLED

    if _SQ_ENABLED:

        def _sq_done(task: asyncio.Task):
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.warning("Synthetic questions background task failed for %s: %s", pid, exc)

        _sq_task = asyncio.create_task(
            _synthetic_questions_background(
                text=text,
                pid=pid,
                agent_id=agent_id,
                namespace=namespace,
                now=now,
                importance=importance,
                retention=retention,
                memory_type=arguments.get("memory_type", "general"),
                thought_type=thought_type,
                actor_id=actor_id,
                actor_type=actor_type,
                confidence=confidence,
                source_trace=source_trace,
                coll=_coll,
            ),
            name=f"synthetic_q_{pid}",
        )
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

    _duration_ms = int((time.monotonic() - t_store) * 1000)
    _budget_summary = ack_budget.summary()
    if _duration_ms > STORE_ACK_BUDGET_MS:
        logger.warning(
            "store_pipeline.ack_budget_exceeded",
            extra={
                "memory_id": pid,
                "namespace": namespace,
                "duration_ms": _duration_ms,
                "budget_ms": STORE_ACK_BUDGET_MS,
                "outbox_enabled": OUTBOX_ENABLED,
            },
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
            "duration_ms": _duration_ms,
            "ack_budget_ms": STORE_ACK_BUDGET_MS,
            "ack_budget": _budget_summary,
            "outbox_enabled": OUTBOX_ENABLED,
            # INIT-005/SPEC-002: numeric stage map only (never fact text / secrets).
            "stage_timings": stage_timings,
            # Searchable-lag SLO hook name (GR-LAG-001); value from gauges loop.
            "searchable_lag_metric": m.SEARCHABLE_LAG_SECONDS,
            # INIT-005/SPEC-005: defer flag only (no embed text / vectors).
            "embed_deferred": bool(_primary_embed_deferred or (_embed_defer and _micro_points)),
        },
    )

    # INIT-003/SPEC-006: optional correction link via SPEC-007 lifecycle.
    # INIT-003/SPEC-009 (SEC-001): prior chunk must belong to *namespace*.
    correction_status = None
    if correction_of:
        from archivist.lifecycle.correct import correct_memory
        from archivist.storage.chunk_lifecycle import get_chunk_lifecycle_row

        prior = await get_chunk_lifecycle_row(correction_of, namespace)
        if prior is None:
            logger.warning(
                "store_pipeline.correction_failed",
                extra={
                    "memory_id": pid,
                    "correction_of": correction_of,
                    "namespace": namespace,
                    "error": "prior memory not found in namespace",
                },
            )
            correction_status = {
                "status": "correction_failed",
                "error": "prior memory not found in namespace",
            }
        else:
            try:
                correction_status = await correct_memory(
                    correction_of,
                    pid,
                    namespace,
                    agent_id=agent_id,
                    reason="store.correction_of",
                )
            except Exception as exc:
                logger.warning(
                    "store_pipeline.correction_failed",
                    extra={
                        "memory_id": pid,
                        "correction_of": correction_of,
                        "namespace": namespace,
                        "error": str(exc),
                    },
                )
                correction_status = {"status": "correction_failed", "error": str(exc)}

    # INIT-005/SPEC-006: additive lag / defer contract (no vectors or secrets).
    _embed_deferred = bool(_primary_embed_deferred or (_embed_defer and _micro_points))
    response: dict = {
        "stored": True,
        "memory_id": pid,
        "uri": memory_uri(namespace, pid),
        "namespace": namespace,
        "entities": entity_names or [agent_id],
        "version": 1,
        # Coach-path store-ack wall clock (INIT-004/SPEC-001); also logged on
        # store_pipeline.complete as duration_ms.
        "duration_ms": _duration_ms,
        # INIT-005/SPEC-002: additive store stage map (at least embed_ms).
        "stage_timings": stage_timings,
        # INIT-005/SPEC-005 + SPEC-006: defer / searchable-lag signal (GR-LAG-001).
        "embed_deferred": _embed_deferred,
        # Metric name for ops/evals (alias SEARCHABLE_LAG_SECONDS → outbox lag).
        "searchable_lag_metric": m.SEARCHABLE_LAG_SECONDS,
        "provenance": {
            "source": prov_source,
            "subject": prov_subject,
            "purpose": prov_purpose,
            "sensitivity": prov_sensitivity,
            "statement_kind": prov_statement_kind,
            "actor_id": actor_id,
            "actor_type": actor_type,
            "confidence": confidence,
        },
    }
    if _embed_deferred:
        response["searchable_lag_hint"] = _SEARCHABLE_LAG_HINT_DEFERRED
    if correction_of:
        response["correction_of"] = correction_of
        response["correction"] = correction_status
    return success_response(response)


async def _handle_merge(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]
    memory_ids = arguments["memory_ids"]
    strategy = arguments["strategy"]
    namespace = arguments.get("namespace", "")

    from archivist.lifecycle.merge import merge_memories

    result = await merge_memories(memory_ids, strategy, agent_id, namespace)
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

    denied = require_rbac(agent_id, "write", namespace)
    if denied:
        return denied

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

    await curator_queue.enqueue(
        "archive_memory",
        {
            "memory_ids": memory_ids,
            "reason": "compressed",
            "compressed_into": stored_data.get("memory_id", ""),
        },
    )

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


async def _handle_pin(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]
    memory_id = arguments.get("memory_id", "").strip()
    entity_name = arguments.get("entity_name", "").strip()
    reason = arguments.get("reason", "")
    namespace = arguments.get("namespace", "") or get_namespace_for_agent(agent_id)

    if not memory_id and not entity_name:
        return error_response({"error": "Provide memory_id or entity_name (or both)"})

    denied = require_rbac(agent_id, "write", namespace)
    if denied:
        return denied

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
                # INIT-010/SPEC-006 (SEC-010-01): sync pin into SQLite so relevance
                # forget candidates (importance-gated) honor archivist_pin.
                from archivist.storage.sqlite_pool import pool as _pin_pool

                async with _pin_pool.write() as conn:
                    await conn.execute(
                        "UPDATE memory_chunks SET importance=1.0 WHERE qdrant_id=? AND namespace=?",
                        (memory_id, namespace or ""),
                    )
                pinned.append({"type": "memory", "id": memory_id})
            else:
                return error_response({"error": f"Memory {memory_id} not found"})
        except Exception as e:
            return error_response({"error": f"Failed to pin memory: {e}"})

    if entity_name:
        from archivist.storage.sqlite_pool import pool

        async with pool.write() as conn:
            _collate = "" if _is_postgres() else " COLLATE NOCASE"
            cur = await conn.execute(
                f"SELECT id FROM entities WHERE name = ?{_collate} AND namespace = ?",
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

    denied = require_rbac(agent_id, "write", namespace)
    if denied:
        return denied

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
            from archivist.storage.sqlite_pool import pool as _unpin_pool

            async with _unpin_pool.write() as conn:
                await conn.execute(
                    "UPDATE memory_chunks SET importance=0.5 WHERE qdrant_id=? AND namespace=?",
                    (memory_id, namespace or ""),
                )
            unpinned.append({"type": "memory", "id": memory_id})
        except Exception as e:
            return error_response({"error": f"Failed to unpin memory: {e}"})

    if entity_name:
        from archivist.storage.sqlite_pool import pool

        async with pool.write() as conn:
            _collate = "" if _is_postgres() else " COLLATE NOCASE"
            cur = await conn.execute(
                f"SELECT id FROM entities WHERE name = ?{_collate}", (entity_name,)
            )
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
    """Forget path: ``mode=delete`` (default) or ``mode=suppress``.

    INIT-003/SPEC-006 — ADR names this ``archivist_forget``; the core tool remains
    ``archivist_delete`` with a ``mode`` arg (GR-PROD-002). Namespace write RBAC
    is enforced after namespace resolution so cross-namespace calls cannot skip
    the check via an empty namespace default.
    """
    from archivist.lifecycle.correct import delete_memory, suppress_memory

    memory_id = arguments.get("memory_id", "").strip()
    agent_id = arguments.get("agent_id", "").strip()
    namespace = arguments.get("namespace", "").strip() or get_namespace_for_agent(agent_id)
    mode = str(arguments.get("mode") or "delete").strip().lower() or "delete"
    reason = str(arguments.get("reason") or "").strip()

    if not memory_id:
        return error_response({"error": "memory_id is required"})
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if mode not in ("delete", "suppress"):
        return error_response(
            {
                "error": "invalid_mode",
                "reason": "mode must be 'delete' or 'suppress'",
            }
        )

    denied = require_rbac(agent_id, "write", namespace)
    if denied:
        return denied

    try:
        if mode == "suppress":
            result = await suppress_memory(
                memory_id,
                namespace,
                agent_id=agent_id,
                reason=reason,
            )
            hot_cache.invalidate_namespace(namespace)
            return success_response(
                {
                    "forgotten": True,
                    "deleted": False,
                    "suppressed": True,
                    "mode": "suppress",
                    "memory_id": memory_id,
                    "namespace": namespace,
                    **result,
                }
            )

        result = await delete_memory(memory_id, namespace, agent_id=agent_id)
    except Exception as e:
        logger.error("archivist_delete failed for %s: %s", memory_id, e)
        return error_response({"error": str(e)})

    hot_cache.invalidate_namespace(namespace)

    return success_response(
        {
            "forgotten": True,
            "deleted": True,
            "suppressed": False,
            "mode": "delete",
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
