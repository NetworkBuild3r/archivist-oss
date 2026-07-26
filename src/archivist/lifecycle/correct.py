"""Supersede / suppress / correct / delete lifecycle service APIs.

INIT-003/SPEC-007 — public service layer for SPEC-006 handlers and SPEC-005
retrieval filters. INIT-003/SPEC-009 — Qdrant payload sync is fail-closed on
namespace-scoped durable row presence (SEC-001).

Semantics:
  - **suppress** — hide from default recall; record remains (not hard erase)
  - **supersede / correct** — winner links ``supersedes_id`` → loser; loser
    excluded from default recall helpers; winner remains visible
  - **delete** — soft-delete / tombstone via existing ``soft_delete_memory``;
    second delete is idempotent
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

import archivist.core.metrics as m
from archivist.core.audit import log_memory_event
from archivist.lifecycle.cascade import _qdrant_set_payload
from archivist.lifecycle.memory_lifecycle import soft_delete_memory
from archivist.storage.chunk_lifecycle import (
    get_chunk_lifecycle_row,
    is_chunk_soft_deleted,
    list_superseded_loser_ids,
    set_chunk_supersedes,
    set_chunk_suppressed,
)
from archivist.storage.collection_router import collection_for
from archivist.storage.qdrant import qdrant_client

logger = logging.getLogger("archivist.lifecycle.correct")


async def _best_effort_qdrant_payload(
    memory_id: str,
    namespace: str,
    payload: dict[str, Any],
    step_name: str,
) -> list[str]:
    """Set Qdrant payload on primary + children; failures are non-fatal."""
    failed: list[str] = []
    col = collection_for(namespace)
    client = qdrant_client()

    await asyncio.to_thread(
        _qdrant_set_payload,
        client,
        col,
        payload,
        [memory_id],
        f"{step_name}_primary",
        memory_id,
        failed,
    )
    await asyncio.to_thread(
        _qdrant_set_payload,
        client,
        col,
        payload,
        Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]),
        f"{step_name}_micro_chunks",
        memory_id,
        failed,
    )
    await asyncio.to_thread(
        _qdrant_set_payload,
        client,
        col,
        payload,
        Filter(must=[FieldCondition(key="source_memory_id", match=MatchValue(value=memory_id))]),
        f"{step_name}_reverse_hyde",
        memory_id,
        failed,
    )
    return failed


async def suppress_memory(
    memory_id: str,
    namespace: str,
    *,
    agent_id: str = "system",
    reason: str = "",
) -> dict[str, Any]:
    """Suppress a memory in *namespace* — hide from default recall, keep the record.

    Sets ``memory_chunks.is_suppressed = 1`` (namespace-scoped) and best-effort
    Qdrant ``is_suppressed=True``. Does **not** hard-delete or soft-delete.
    Idempotent: repeating suppress on an already-suppressed row succeeds.
    """
    if not memory_id or not namespace:
        raise ValueError("memory_id and namespace are required")

    rows = await set_chunk_suppressed(memory_id, namespace, suppressed=True)
    # INIT-003/SPEC-009 (SEC-001): fail closed — never mutate Qdrant unless the
    # durable namespace-scoped UPDATE matched (prevents cross-tenant recall denial).
    qdrant_failed: list[str] = []
    if rows > 0:
        qdrant_failed = await _best_effort_qdrant_payload(
            memory_id,
            namespace,
            {"is_suppressed": True},
            "suppress",
        )

    metadata = {
        "status": "suppressed",
        "rows_updated": rows,
        "reason": reason or "",
        "failed_steps": qdrant_failed,
    }
    await log_memory_event(
        agent_id=agent_id,
        action="suppress",
        memory_id=memory_id,
        namespace=namespace,
        text_hash="",
        metadata=metadata,
    )
    m.inc("archivist_memory_suppress_total", {"namespace": namespace})
    logger.info(
        "memory.suppressed",
        extra={
            "memory_id": memory_id,
            "namespace": namespace,
            "rows_updated": rows,
            "failed_steps": qdrant_failed,
        },
    )
    return {
        "status": "suppressed",
        "memory_id": memory_id,
        "namespace": namespace,
        "rows_updated": rows,
        "already_suppressed": rows == 0,
    }


async def unsuppress_memory(
    memory_id: str,
    namespace: str,
    *,
    agent_id: str = "system",
    reason: str = "",
) -> dict[str, Any]:
    """Explicit unsuppress — the only path that restores a suppressed memory to recall.

    Without this call, suppressed memories stay hidden from default recall
    (no silent resurrection).
    """
    if not memory_id or not namespace:
        raise ValueError("memory_id and namespace are required")

    rows = await set_chunk_suppressed(memory_id, namespace, suppressed=False)
    # INIT-003/SPEC-009 (SEC-001): fail closed — skip Qdrant when no durable row.
    qdrant_failed: list[str] = []
    if rows > 0:
        qdrant_failed = await _best_effort_qdrant_payload(
            memory_id,
            namespace,
            {"is_suppressed": False},
            "unsuppress",
        )
    await log_memory_event(
        agent_id=agent_id,
        action="unsuppress",
        memory_id=memory_id,
        namespace=namespace,
        text_hash="",
        metadata={
            "status": "unsuppressed",
            "rows_updated": rows,
            "reason": reason or "",
            "failed_steps": qdrant_failed,
        },
    )
    logger.info(
        "memory.unsuppressed",
        extra={"memory_id": memory_id, "namespace": namespace, "rows_updated": rows},
    )
    return {
        "status": "unsuppressed",
        "memory_id": memory_id,
        "namespace": namespace,
        "rows_updated": rows,
    }


async def supersede_memory(
    old_id: str,
    new_id: str,
    namespace: str,
    *,
    agent_id: str = "system",
    reason: str = "",
) -> dict[str, Any]:
    """Mark *new_id* as superseding *old_id* within *namespace*.

    Winner gets ``supersedes_id = old_id``. Loser is excluded from default
    recall helpers; winner remains present. Does not delete the loser.
    """
    if not old_id or not new_id or not namespace:
        raise ValueError("old_id, new_id, and namespace are required")
    if old_id == new_id:
        raise ValueError("old_id and new_id must differ")

    rows = await set_chunk_supersedes(new_id, old_id, namespace)

    # INIT-003/SPEC-009 (SEC-001): fail closed — Qdrant only when winner UPDATE
    # matched in *namespace*; loser payload only when loser also exists there.
    loser_failed: list[str] = []
    winner_failed: list[str] = []
    if rows > 0:
        loser_row = await get_chunk_lifecycle_row(old_id, namespace)
        if loser_row is not None:
            loser_failed = await _best_effort_qdrant_payload(
                old_id,
                namespace,
                {"is_superseded": True, "superseded_by": new_id},
                "supersede_loser",
            )
        winner_failed = await _best_effort_qdrant_payload(
            new_id,
            namespace,
            {"supersedes_id": old_id, "is_superseded": False},
            "supersede_winner",
        )
    failed = loser_failed + winner_failed

    metadata = {
        "status": "superseded",
        "old_id": old_id,
        "new_id": new_id,
        "rows_updated": rows,
        "reason": reason or "",
        "failed_steps": failed,
    }
    await log_memory_event(
        agent_id=agent_id,
        action="supersede",
        memory_id=new_id,
        namespace=namespace,
        text_hash="",
        metadata=metadata,
    )
    m.inc("archivist_memory_supersede_total", {"namespace": namespace})
    logger.info(
        "memory.superseded",
        extra={
            "old_id": old_id,
            "new_id": new_id,
            "namespace": namespace,
            "rows_updated": rows,
            "failed_steps": failed,
        },
    )
    return {
        "status": "superseded",
        "old_id": old_id,
        "new_id": new_id,
        "namespace": namespace,
        "rows_updated": rows,
        "supersedes_id": old_id,
    }


async def correct_memory(
    old_id: str,
    new_id: str,
    namespace: str,
    *,
    agent_id: str = "system",
    reason: str = "",
) -> dict[str, Any]:
    """Correction entrypoint — links *new_id* as the superseding winner of *old_id*.

    Called from store (when ``correction_of`` is set) or an explicit correct
    path. Equivalent to :func:`supersede_memory` with ``action=correct`` audit.
    """
    result = await supersede_memory(
        old_id,
        new_id,
        namespace,
        agent_id=agent_id,
        reason=reason or "correction",
    )
    # Additional audit with action=correct for store/correct callers.
    await log_memory_event(
        agent_id=agent_id,
        action="correct",
        memory_id=new_id,
        namespace=namespace,
        text_hash="",
        metadata={
            "status": "corrected",
            "old_id": old_id,
            "new_id": new_id,
            "reason": reason or "correction",
        },
    )
    result["status"] = "corrected"
    return result


async def delete_memory(
    memory_id: str,
    namespace: str,
    *,
    agent_id: str = "system",
) -> dict[str, Any]:
    """Governed delete — wraps ``soft_delete_memory`` with idempotent second delete.

    First call tombstones (soft-delete + background hard-cascade enqueue).
    Second call on an already-excluded chunk returns ``already_deleted``
    without re-enqueueing cascade work.
    """
    if not memory_id or not namespace:
        raise ValueError("memory_id and namespace are required")

    if await is_chunk_soft_deleted(memory_id, namespace):
        await log_memory_event(
            agent_id=agent_id,
            action="delete",
            memory_id=memory_id,
            namespace=namespace,
            text_hash="",
            metadata={"status": "already_deleted", "idempotent": True},
        )
        logger.info(
            "memory.delete_idempotent",
            extra={"memory_id": memory_id, "namespace": namespace},
        )
        return {
            "status": "already_deleted",
            "memory_id": memory_id,
            "namespace": namespace,
            "idempotent": True,
        }

    result = await soft_delete_memory(memory_id, namespace)
    return {
        "status": result.get("status", "soft_delete_initiated"),
        "memory_id": memory_id,
        "namespace": namespace,
        "op_id": result.get("op_id"),
        "idempotent": False,
    }


async def default_recall_rows(
    rows: list[dict[str, Any]],
    namespace: str,
) -> list[dict[str, Any]]:
    """Filter *rows* with namespace-aware supersede loser set (service helper for tests/callers)."""
    from archivist.lifecycle.visibility import filter_recall_visible

    losers = await list_superseded_loser_ids(namespace)
    return filter_recall_visible(rows, known_superseded_ids=losers)


async def get_lifecycle_state(memory_id: str, namespace: str) -> dict[str, Any] | None:
    """Return durable lifecycle flags for *memory_id* in *namespace* (ops/audit path)."""
    return await get_chunk_lifecycle_row(memory_id, namespace)
