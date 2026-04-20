"""Unified delete and archive functions for memories and all derived artifacts.

Every code path that removes or archives a memory MUST go through this module.
Adding a new artifact type to the write pipeline requires adding a corresponding
cleanup step here and registering it in ``cascade.py``'s orphan sweeper.
"""

import asyncio
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

if TYPE_CHECKING:
    pass

import archivist.core.metrics as m
import archivist.lifecycle.curator_queue as curator_queue
from archivist.core.audit import log_memory_event
from archivist.lifecycle.cascade import (
    PartialDeletionError,
    _qdrant_delete,
    _qdrant_set_payload,
    _scroll_all,
)
from archivist.storage.collection_router import collection_for
from archivist.storage.graph import (
    delete_fts_chunks_batch,
    delete_hotness,
    delete_memory_points,
    delete_needle_tokens_batch,
    lookup_memory_points,
    set_fts_excluded_batch,
)
from archivist.storage.qdrant import qdrant_client

logger = logging.getLogger("archivist.memory_lifecycle")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class DeleteResult:
    """Counts of artifacts removed per type."""

    memory_id: str = ""
    qdrant_primary: int = 0
    qdrant_reverse_hyde: int = 0
    qdrant_micro_chunks: int = 0
    fts_entries: int = 0
    registry_tokens: int = 0
    entity_facts: int = 0
    memory_hotness: int = 0
    relationship_rows: int = 0
    version_rows: int = 0
    failed_steps: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            self.qdrant_primary
            + self.qdrant_reverse_hyde
            + self.qdrant_micro_chunks
            + self.fts_entries
            + self.registry_tokens
            + self.entity_facts
            + self.memory_hotness
            + self.relationship_rows
            + self.version_rows
        )


@dataclass
class ArchiveResult:
    """Per-step success flags for archive operations."""

    memory_id: str = ""
    primary_archived: bool = False
    reverse_hyde_archived: bool = False
    micro_chunks_archived: bool = False
    failed_steps: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(
            [
                self.primary_archived,
                self.reverse_hyde_archived,
                self.micro_chunks_archived,
            ]
        )


# ---------------------------------------------------------------------------
# Delete helpers
# ---------------------------------------------------------------------------


async def _resolve_child_ids(
    memory_id: str,
    client,
    col: str,
    failed_steps: list[str],
) -> tuple[list[str], list[str]]:
    """Return (micro_ids, hyde_ids) for a given memory.

    Tries the SQLite ``memory_points`` table first (O(1), no Qdrant round-trip).
    Falls back to paginated Qdrant scroll for legacy memories that pre-date the
    ``memory_points`` write path.
    """
    mp_rows = await lookup_memory_points(memory_id)
    if mp_rows:
        micro_ids = [r["qdrant_id"] for r in mp_rows if r["point_type"] == "micro_chunk"]
        hyde_ids = [r["qdrant_id"] for r in mp_rows if r["point_type"] == "reverse_hyde"]
        logger.debug(
            "delete.child_lookup from memory_points: micro=%d hyde=%d",
            len(micro_ids),
            len(hyde_ids),
        )
        return micro_ids, hyde_ids

    micro_ids = await asyncio.to_thread(
        _scroll_all,
        client,
        col,
        Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]),
        "scroll_micro_chunks",
        memory_id,
        failed_steps,
    )
    hyde_ids = await asyncio.to_thread(
        _scroll_all,
        client,
        col,
        Filter(must=[FieldCondition(key="source_memory_id", match=MatchValue(value=memory_id))]),
        "scroll_reverse_hyde",
        memory_id,
        failed_steps,
    )
    return micro_ids, hyde_ids


async def _delete_qdrant_points(
    memory_id: str,
    micro_ids: list[str],
    hyde_ids: list[str],
    client,
    col: str,
    failed_steps: list[str],
    txn: Any = None,
) -> tuple[int, int, int]:
    """Delete primary + child Qdrant points.  Returns (primary, hyde, micro) counts.

    When ``OUTBOX_ENABLED=True``, the Qdrant deletes are enqueued in the outbox
    table atomically with the SQLite artefact cleanup in ``delete_memory_complete``.
    Pass an open ``MemoryTransaction`` via *txn* so both writes share one commit.

    When ``OUTBOX_ENABLED=False`` (default), behaviour is identical to Phase 2
    (synchronous delete with retry, failures appended to *failed_steps*).
    """
    from archivist.core.config import OUTBOX_ENABLED

    if OUTBOX_ENABLED:
        # Enqueue outbox events.  The actual Qdrant calls happen asynchronously
        # via OutboxProcessor.  Return approximate counts so result metrics are
        # still populated.
        all_ids = [memory_id] + hyde_ids + micro_ids
        if txn is not None:
            txn.enqueue_qdrant_delete(col, all_ids, memory_id=memory_id)
        else:
            from archivist.storage.transaction import MemoryTransaction

            async with MemoryTransaction() as _txn:
                _txn.enqueue_qdrant_delete(col, all_ids, memory_id=memory_id)
        return 1, len(hyde_ids), len(micro_ids)

    # Legacy synchronous path (OUTBOX_ENABLED=False).
    primary_count = await _qdrant_delete(
        client,
        col,
        [memory_id],
        "qdrant_primary",
        memory_id,
        failed_steps,
    )

    all_child_ids = hyde_ids + micro_ids
    if all_child_ids:
        await _qdrant_delete(
            client,
            col,
            all_child_ids,
            "qdrant_children",
            memory_id,
            failed_steps,
        )

    return primary_count, len(hyde_ids), len(micro_ids)


async def _delete_sqlite_artifacts(
    memory_id: str,
    all_ids: list[str],
    failed_steps: list[str],
    txn: Any = None,
) -> tuple[int, int, int]:
    """Delete FTS, needle registry, and entity facts rows.  Returns (fts, needle, facts) counts.

    Each step is independently try/excepted so a failure in one does not block
    the others.

    Args:
        txn: Optional open ``MemoryTransaction``.  When provided, the
            ``delete_fts_chunks_batch`` and ``delete_needle_tokens_batch``
            calls are joined to the same pool.write() connection so all
            SQLite cleanup and the outbox event land in one atomic commit.
            When ``None`` each helper acquires its own lock (legacy behaviour).
    """
    fts_count = 0
    try:
        fts_count = await delete_fts_chunks_batch(all_ids)
    except (sqlite3.Error, OSError) as e:
        logger.error("cascade.fts_batch failed for %s: %s", memory_id, e)
        failed_steps.append("fts_batch")

    needle_count = 0
    try:
        needle_count = await delete_needle_tokens_batch(all_ids)
    except (sqlite3.Error, OSError) as e:
        logger.error("cascade.needle_batch failed for %s: %s", memory_id, e)
        failed_steps.append("needle_batch")

    facts_count = 0
    try:
        facts_count = await _delete_entity_facts_for_memory(memory_id)
    except (sqlite3.Error, OSError) as e:
        logger.error("cascade.entity_facts failed for %s: %s", memory_id, e)
        failed_steps.append("entity_facts")

    return fts_count, needle_count, facts_count


async def _delete_best_effort_rows(memory_id: str) -> tuple[int, int, int]:
    """Delete memory_hotness, memory_points, and memory_versions rows.

    Also removes any entity relationships that have become fully orphaned
    (source or target entity has zero remaining active facts across all memories).

    All steps are best-effort — failures are logged at DEBUG since these rows
    do not affect Qdrant visibility.  Returns (hotness_count, relationship_rows, version_rows).
    """
    from archivist.storage.sqlite_pool import pool

    hotness_count = 0
    try:
        hotness_count = await delete_hotness(memory_id)
    except Exception as e:
        logger.debug("cascade.memory_hotness skipped for %s: %s", memory_id, e)

    try:
        await delete_memory_points(memory_id)
    except Exception as e:
        logger.debug("cascade.memory_points cleanup skipped for %s: %s", memory_id, e)

    version_rows = 0
    try:
        async with pool.write() as conn:
            cur = await conn.execute(
                "DELETE FROM memory_versions WHERE memory_id = ?",
                (memory_id,),
            )
            version_rows = cur.rowcount
    except Exception as e:
        logger.debug("cascade.memory_versions skipped for %s: %s", memory_id, e)

    # Remove relationship rows whose source or target entity now has no active
    # facts left (orphaned after deactivation of this memory's entity facts).
    relationship_rows = 0
    try:
        async with pool.write() as conn:
            cur = await conn.execute(
                """DELETE FROM relationships
                   WHERE source_entity_id NOT IN (
                       SELECT DISTINCT entity_id FROM facts WHERE is_active = 1
                   ) OR target_entity_id NOT IN (
                       SELECT DISTINCT entity_id FROM facts WHERE is_active = 1
                   )""",
            )
            relationship_rows = cur.rowcount
    except Exception as e:
        logger.debug("cascade.relationships skipped for %s: %s", memory_id, e)

    return hotness_count, relationship_rows, version_rows


async def _finalize_delete(
    result: DeleteResult,
    namespace: str,
    col: str,
) -> None:
    """Emit metrics, structured log, audit entry, and raise PartialDeletionError if needed.

    All observability and error-gate logic in one place.  Must be the last step
    in the delete orchestrator.
    """
    m.inc(m.DELETE_COMPLETE, {"namespace": namespace})

    logger.info(
        "memory.deleted_complete",
        extra={
            "memory_id": result.memory_id,
            "namespace": namespace,
            "collection": col,
            "qdrant_primary": result.qdrant_primary,
            "qdrant_reverse_hyde": result.qdrant_reverse_hyde,
            "qdrant_micro_chunks": result.qdrant_micro_chunks,
            "fts_entries": result.fts_entries,
            "registry_tokens": result.registry_tokens,
            "entity_facts": result.entity_facts,
            "memory_hotness": result.memory_hotness,
            "failed_steps": result.failed_steps,
            "total_artifacts": result.total,
        },
    )

    metadata = asdict(result)
    metadata["result_type"] = "delete"
    await log_memory_event(
        agent_id="system",
        action="delete",
        memory_id=result.memory_id,
        namespace=namespace,
        text_hash="",
        metadata=metadata,
    )

    _critical = {"qdrant_primary", "qdrant_children"}
    if _critical & set(result.failed_steps):
        raise PartialDeletionError(result)
    if result.failed_steps:
        logger.warning(
            "delete_cascade partial failure for %s: %s",
            result.memory_id,
            result.failed_steps,
        )


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------


async def _archive_qdrant_points(
    memory_id: str,
    client,
    col: str,
    failed_steps: list[str],
) -> ArchiveResult:
    """Set archived=True on primary + child Qdrant points.  Returns partial ArchiveResult.

    Failures are recorded into the caller-supplied *failed_steps* list; the
    returned ``ArchiveResult.failed_steps`` is the same list object.
    """
    result = ArchiveResult(memory_id=memory_id, failed_steps=failed_steps)
    payload = {"archived": True}

    result.primary_archived = await asyncio.to_thread(
        _qdrant_set_payload,
        client,
        col,
        payload,
        [memory_id],
        "archive_primary",
        memory_id,
        failed_steps,
    )

    result.reverse_hyde_archived = await asyncio.to_thread(
        _qdrant_set_payload,
        client,
        col,
        payload,
        Filter(must=[FieldCondition(key="source_memory_id", match=MatchValue(value=memory_id))]),
        "archive_reverse_hyde",
        memory_id,
        failed_steps,
    )

    result.micro_chunks_archived = await asyncio.to_thread(
        _qdrant_set_payload,
        client,
        col,
        payload,
        Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]),
        "archive_micro_chunks",
        memory_id,
        failed_steps,
    )

    return result


async def _archive_fts_rows(
    memory_id: str,
    micro_ids: list[str],
    hyde_ids: list[str],
) -> None:
    """Mark all FTS5 rows for a memory and its children as excluded (best-effort)."""
    all_ids = [memory_id] + micro_ids + hyde_ids
    try:
        await set_fts_excluded_batch(all_ids, 1)
    except Exception as e:
        logger.warning("archive.fts_excluded failed for %s: %s", memory_id, e)


async def _finalize_archive(
    result: ArchiveResult,
    namespace: str,
    col: str,
) -> None:
    """Emit metrics, structured log, and audit entry for an archive operation."""
    m.inc(m.ARCHIVE_COMPLETE, {"namespace": namespace})

    logger.info(
        "memory.archived_complete",
        extra={
            "memory_id": result.memory_id,
            "namespace": namespace,
            "collection": col,
            "primary": result.primary_archived,
            "reverse_hyde": result.reverse_hyde_archived,
            "micro_chunks": result.micro_chunks_archived,
            "failed_steps": result.failed_steps,
        },
    )

    metadata = asdict(result)
    metadata["result_type"] = "archive"
    await log_memory_event(
        agent_id="system",
        action="archive",
        memory_id=result.memory_id,
        namespace=namespace,
        text_hash="",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# SQLite entity-facts helper (synchronous, called via asyncio.to_thread)
# ---------------------------------------------------------------------------


async def _delete_entity_facts_for_memory(memory_id: str) -> int:
    """Soft-deactivate entity facts linked to *memory_id*.

    Primary path: exact match on the ``memory_id`` column (indexed, O(log n)).
    Fallback path: LIKE match on ``source_file`` for pre-migration rows where
    ``memory_id`` is still empty.  Once all rows are backfilled the fallback
    becomes a no-op.

    Returns count of soft-deactivated facts.
    """
    from archivist.storage.sqlite_pool import pool

    async with pool.write() as conn:
        try:
            cur = await conn.execute(
                "UPDATE facts SET is_active = 0 WHERE memory_id = ? AND is_active = 1",
                (memory_id,),
            )
            deactivated = cur.rowcount

            cur2 = await conn.execute(
                "UPDATE facts SET is_active = 0 "
                "WHERE source_file LIKE ? AND is_active = 1 AND memory_id = ''",
                (f"%{memory_id}%",),
            )
            deactivated += cur2.rowcount

            return deactivated
        except Exception as e:
            logger.warning(
                "delete_entity_facts failed for memory %s: %s",
                memory_id,
                e,
            )
            return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def delete_memory_complete(
    memory_id: str,
    namespace: str,
    *,
    collection: str | None = None,
) -> DeleteResult:
    """Delete a memory and ALL derived artifacts.

    This is the ONLY function that should delete memories.

    Args:
        memory_id: The Qdrant point ID of the primary memory.
        namespace: The namespace for collection routing.
        collection: Override collection name (if known). Defaults to
                    ``collection_for(namespace)``.

    Returns:
        DeleteResult with counts of deleted artifacts per type.

    Raises:
        PartialDeletionError: If the primary Qdrant point delete fails or
            more than two cascade steps fail.
    """
    from archivist.core.config import OUTBOX_ENABLED

    col = collection or collection_for(namespace)
    client = qdrant_client()
    result = DeleteResult(memory_id=memory_id)

    micro_ids, hyde_ids = await _resolve_child_ids(memory_id, client, col, result.failed_steps)
    all_ids = [memory_id] + hyde_ids + micro_ids

    if OUTBOX_ENABLED:
        # Atomically enqueue the Qdrant-delete event AND clean up all SQLite
        # artefacts (FTS, needle registry, memory_points) inside a single
        # pool.write() transaction.  A crash mid-way rolls everything back —
        # no orphaned outbox row without the corresponding SQLite cleanup, and
        # no SQLite cleanup without the outbox row.
        from archivist.storage.transaction import MemoryTransaction

        async with MemoryTransaction() as txn:
            txn.enqueue_qdrant_delete(col, all_ids, memory_id=memory_id)

            # SQLite cleanup — each step try/excepted inside helpers but all
            # sharing the same connection so they commit together.
            fts_count = 0
            try:
                fts_count = await delete_fts_chunks_batch(all_ids)
            except (sqlite3.Error, OSError) as e:
                logger.error("cascade.fts_batch failed for %s: %s", memory_id, e)
                result.failed_steps.append("fts_batch")

            needle_count = 0
            try:
                needle_count = await delete_needle_tokens_batch(all_ids)
            except (sqlite3.Error, OSError) as e:
                logger.error("cascade.needle_batch failed for %s: %s", memory_id, e)
                result.failed_steps.append("needle_batch")

            facts_count = 0
            try:
                facts_count = await _delete_entity_facts_for_memory(memory_id)
            except (sqlite3.Error, OSError) as e:
                logger.error("cascade.entity_facts failed for %s: %s", memory_id, e)
                result.failed_steps.append("entity_facts")

        result.qdrant_primary = 1
        result.qdrant_reverse_hyde = len(hyde_ids)
        result.qdrant_micro_chunks = len(micro_ids)
        result.fts_entries = fts_count
        result.registry_tokens = needle_count
        result.entity_facts = facts_count
    else:
        # Legacy path: synchronous Qdrant deletes then independent SQLite steps.
        (
            result.qdrant_primary,
            result.qdrant_reverse_hyde,
            result.qdrant_micro_chunks,
        ) = await _delete_qdrant_points(
            memory_id, micro_ids, hyde_ids, client, col, result.failed_steps
        )

        (
            result.fts_entries,
            result.registry_tokens,
            result.entity_facts,
        ) = await _delete_sqlite_artifacts(memory_id, all_ids, result.failed_steps)

    (
        result.memory_hotness,
        result.relationship_rows,
        result.version_rows,
    ) = await _delete_best_effort_rows(memory_id)

    await _finalize_delete(result, namespace, col)
    return result


async def archive_memory_complete(
    memory_id: str,
    namespace: str,
    *,
    collection: str | None = None,
) -> ArchiveResult:
    """Set archived=True on a memory and ALL derived Qdrant points.

    Archives: primary point + reverse HyDE + micro-chunks.
    Also marks all related FTS5 rows as excluded so archived memories no longer
    appear in BM25/FTS keyword search.

    Returns ArchiveResult with per-step success flags.
    """
    col = collection or collection_for(namespace)
    client = qdrant_client()
    failed_steps: list[str] = []

    micro_ids, hyde_ids = await _resolve_child_ids(memory_id, client, col, failed_steps)

    result = await _archive_qdrant_points(memory_id, client, col, failed_steps)
    await _archive_fts_rows(memory_id, micro_ids, hyde_ids)
    await _finalize_archive(result, namespace, col)
    return result


async def soft_delete_memory(memory_id: str, namespace: str) -> dict:
    """Mark a memory as deleted and enqueue a background hard-cascade.

    Hot path (~5 ms, non-blocking):
      1. Set ``deleted=True`` on the primary Qdrant point and all child points
         (micro-chunks, reverse HyDE) so they disappear from vector search
         immediately.
      2. Mark the primary FTS entry (and any discovered child entries) as
         excluded so the memory disappears from BM25 search immediately.
      3. Enqueue a ``delete_memory`` job in ``curator_queue`` for the full
         hard-cascade (run by the background drain loop).
      4. Log to ``audit_log`` with status ``"soft_delete_initiated"``.

    Returns a dict with ``{"status": "soft_delete_initiated", "op_id": ...}``.

    Raises:
        RuntimeError: If the primary Qdrant ``set_payload`` call fails (critical).
    """
    col = collection_for(namespace)
    client = qdrant_client()
    failed_steps: list[str] = []
    deleted_payload = {"deleted": True}

    # 1a. Mark primary point as deleted in Qdrant.
    await asyncio.to_thread(
        _qdrant_set_payload,
        client,
        col,
        deleted_payload,
        [memory_id],
        "soft_delete_primary",
        memory_id,
        failed_steps,
    )
    if "soft_delete_primary" in failed_steps:
        raise RuntimeError(f"soft_delete_memory: primary Qdrant set_payload failed for {memory_id}")

    # 1b. Mark child points (micro-chunks + reverse HyDE) as deleted.
    await asyncio.to_thread(
        _qdrant_set_payload,
        client,
        col,
        deleted_payload,
        Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]),
        "soft_delete_micro_chunks",
        memory_id,
        failed_steps,
    )
    await asyncio.to_thread(
        _qdrant_set_payload,
        client,
        col,
        deleted_payload,
        Filter(must=[FieldCondition(key="source_memory_id", match=MatchValue(value=memory_id))]),
        "soft_delete_reverse_hyde",
        memory_id,
        failed_steps,
    )

    # 2. Exclude the primary FTS entry; children cleaned by the hard-cascade.
    try:
        await set_fts_excluded_batch([memory_id], 1)
    except Exception as e:
        logger.warning("soft_delete.fts_excluded failed for %s: %s", memory_id, e)

    # 3. Enqueue the background hard-cascade.
    op_id = curator_queue.enqueue(
        "delete_memory",
        {"memory_ids": [memory_id], "namespace": namespace},
    )

    # 4. Audit log.
    await log_memory_event(
        agent_id="system",
        action="soft_delete",
        memory_id=memory_id,
        namespace=namespace,
        text_hash="",
        metadata={
            "op_id": op_id,
            "failed_steps": failed_steps,
            "status": "soft_delete_initiated",
        },
    )

    m.inc(m.SOFT_DELETE_INITIATED, {"namespace": namespace})

    logger.info(
        "memory.soft_deleted",
        extra={
            "memory_id": memory_id,
            "namespace": namespace,
            "op_id": op_id,
            "failed_steps": failed_steps,
        },
    )

    return {"status": "soft_delete_initiated", "op_id": op_id}
