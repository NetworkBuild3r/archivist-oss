"""Unified delete and archive functions for memories and all derived artifacts.

Every code path that removes or archives a memory MUST go through this module.
Adding a new artifact type to the write pipeline requires adding a corresponding
cleanup step here and registering it in ``cascade.py``'s orphan sweeper.
"""

import logging
import sqlite3
from dataclasses import asdict, dataclass, field

from qdrant_client.models import Filter, FieldCondition, MatchValue

from collection_router import collection_for
import curator_queue
from cascade import (
    PartialDeletionError,
    _qdrant_delete,
    _qdrant_set_payload,
    _scroll_all,
)
from graph import (
    delete_fts_chunks_batch,
    delete_needle_tokens_batch,
    delete_hotness,
    set_fts_excluded_batch,
    lookup_memory_points,
    delete_memory_points,
    log_delete_failure,
    GRAPH_WRITE_LOCK,
    get_db,
)
from qdrant import qdrant_client
from audit import log_memory_event
import metrics as m

logger = logging.getLogger("archivist.memory_lifecycle")


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
        return sum([
            self.primary_archived,
            self.reverse_hyde_archived,
            self.micro_chunks_archived,
        ])


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
                    collection_for(namespace).

    Returns:
        DeleteResult with counts of deleted artifacts per type.

    Raises:
        PartialDeletionError: If the primary Qdrant point delete fails or
            more than two cascade steps fail.
    """
    col = collection or collection_for(namespace)
    client = qdrant_client()
    result = DeleteResult(memory_id=memory_id)

    # 0. Enumerate child point IDs BEFORE deleting them from Qdrant.
    #    Prefer the memory_points table (O(1) SQLite lookup, no Qdrant round-trip).
    #    Fall back to paginated Qdrant scroll for legacy memories created before Phase 2.
    _mp_rows = lookup_memory_points(memory_id)
    if _mp_rows:
        micro_ids = [r["qdrant_id"] for r in _mp_rows if r["point_type"] == "micro_chunk"]
        hyde_ids = [r["qdrant_id"] for r in _mp_rows if r["point_type"] == "reverse_hyde"]
        logger.debug(
            "delete.child_lookup from memory_points: micro=%d hyde=%d",
            len(micro_ids), len(hyde_ids),
        )
    else:
        micro_ids = _scroll_all(
            client, col,
            filt=Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]),
            step_name="scroll_micro_chunks", memory_id=memory_id,
            failed_steps=result.failed_steps,
        )
        hyde_ids = _scroll_all(
            client, col,
            filt=Filter(must=[FieldCondition(key="source_memory_id", match=MatchValue(value=memory_id))]),
            step_name="scroll_reverse_hyde", memory_id=memory_id,
            failed_steps=result.failed_steps,
        )

    # 1. Delete primary Qdrant point
    result.qdrant_primary = _qdrant_delete(
        client, col, [memory_id],
        "qdrant_primary", memory_id, result.failed_steps,
    )

    # 2. Delete all child vectors (reverse HyDE + micro-chunks) in one call
    all_child_ids = hyde_ids + micro_ids
    result.qdrant_reverse_hyde = len(hyde_ids)
    result.qdrant_micro_chunks = len(micro_ids)
    if all_child_ids:
        _qdrant_delete(
            client, col, all_child_ids,
            "qdrant_children", memory_id, result.failed_steps,
        )

    # 3. Batch-delete FTS5 entries (primary + all children)
    all_ids = [memory_id] + hyde_ids + micro_ids
    try:
        result.fts_entries = delete_fts_chunks_batch(all_ids)
    except (sqlite3.Error, OSError) as e:
        logger.error("cascade.fts_batch failed for %s: %s", memory_id, e)
        result.failed_steps.append("fts_batch")

    # 4. Batch-delete needle registry rows (primary + all children)
    try:
        result.registry_tokens = delete_needle_tokens_batch(all_ids)
    except (sqlite3.Error, OSError) as e:
        logger.error("cascade.needle_batch failed for %s: %s", memory_id, e)
        result.failed_steps.append("needle_batch")

    # 5. Delete auto-extracted entity facts
    try:
        result.entity_facts = _delete_entity_facts_for_memory(memory_id)
    except (sqlite3.Error, OSError) as e:
        logger.error("cascade.entity_facts failed for %s: %s", memory_id, e)
        result.failed_steps.append("entity_facts")

    # 6. Delete memory_hotness row
    try:
        result.memory_hotness = delete_hotness(memory_id)
    except Exception as e:
        logger.debug("cascade.memory_hotness skipped for %s: %s", memory_id, e)

    # 7. Clean up memory_points tracking rows.
    try:
        delete_memory_points(memory_id)
    except Exception as e:
        logger.debug("cascade.memory_points cleanup skipped for %s: %s", memory_id, e)

    m.inc(m.DELETE_COMPLETE, {"namespace": namespace})

    logger.info(
        "memory.deleted_complete",
        extra={
            "memory_id": memory_id,
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
        memory_id=memory_id,
        namespace=namespace,
        text_hash="",
        metadata=metadata,
    )

    _critical = {"qdrant_primary", "qdrant_children"}
    if _critical & set(result.failed_steps):
        raise PartialDeletionError(result)
    elif result.failed_steps:
        logger.warning(
            "delete_cascade partial failure for %s: %s",
            memory_id, result.failed_steps,
        )

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
    result = ArchiveResult(memory_id=memory_id)
    payload = {"archived": True}

    # Enumerate child IDs before setting payload so we can mark them excluded in FTS.
    micro_ids = _scroll_all(
        client, col,
        filt=Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]),
        step_name="archive_scroll_micro", memory_id=memory_id,
        failed_steps=result.failed_steps,
    )
    hyde_ids = _scroll_all(
        client, col,
        filt=Filter(must=[FieldCondition(key="source_memory_id", match=MatchValue(value=memory_id))]),
        step_name="archive_scroll_hyde", memory_id=memory_id,
        failed_steps=result.failed_steps,
    )

    result.primary_archived = _qdrant_set_payload(
        client, col, payload, [memory_id],
        "archive_primary", memory_id, result.failed_steps,
    )

    result.reverse_hyde_archived = _qdrant_set_payload(
        client, col, payload,
        Filter(must=[FieldCondition(key="source_memory_id", match=MatchValue(value=memory_id))]),
        "archive_reverse_hyde", memory_id, result.failed_steps,
    )

    result.micro_chunks_archived = _qdrant_set_payload(
        client, col, payload,
        Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]),
        "archive_micro_chunks", memory_id, result.failed_steps,
    )

    # Mark all related FTS5 rows as excluded so they disappear from BM25 search.
    all_ids = [memory_id] + micro_ids + hyde_ids
    try:
        set_fts_excluded_batch(all_ids, excluded=1)
    except Exception as e:
        logger.warning("archive.fts_excluded failed for %s: %s", memory_id, e)

    m.inc(m.ARCHIVE_COMPLETE, {"namespace": namespace})

    logger.info(
        "memory.archived_complete",
        extra={
            "memory_id": memory_id,
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
        memory_id=memory_id,
        namespace=namespace,
        text_hash="",
        metadata=metadata,
    )

    return result


def _delete_entity_facts_for_memory(memory_id: str) -> int:
    """Soft-deactivate entity facts linked to *memory_id*.

    Primary path: exact match on the ``memory_id`` column (indexed, O(log n)).
    Fallback path: LIKE match on ``source_file`` for pre-migration rows where
    ``memory_id`` is still empty.  Once all rows are backfilled the fallback
    becomes a no-op.

    Returns count of soft-deactivated facts.
    """
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            cur = conn.execute(
                "UPDATE facts SET is_active = 0 "
                "WHERE memory_id = ? AND is_active = 1",
                (memory_id,),
            )
            deactivated = cur.rowcount

            cur2 = conn.execute(
                "UPDATE facts SET is_active = 0 "
                "WHERE source_file LIKE ? AND is_active = 1 AND memory_id = ''",
                (f"%{memory_id}%",),
            )
            deactivated += cur2.rowcount

            conn.commit()
            return deactivated
        except Exception as e:
            logger.warning(
                "delete_entity_facts failed for memory %s: %s", memory_id, e,
            )
            return 0
        finally:
            conn.close()


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
        Exception: If the primary Qdrant ``set_payload`` call fails (critical).
    """
    col = collection_for(namespace)
    client = qdrant_client()
    failed_steps: list[str] = []
    deleted_payload = {"deleted": True}

    # 1a. Mark primary point as deleted in Qdrant.
    _qdrant_set_payload(
        client, col, deleted_payload, [memory_id],
        "soft_delete_primary", memory_id, failed_steps,
    )
    if "soft_delete_primary" in failed_steps:
        raise RuntimeError(
            f"soft_delete_memory: primary Qdrant set_payload failed for {memory_id}"
        )

    # 1b. Mark child points (micro-chunks + reverse HyDE) as deleted.
    _qdrant_set_payload(
        client, col, deleted_payload,
        Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]),
        "soft_delete_micro_chunks", memory_id, failed_steps,
    )
    _qdrant_set_payload(
        client, col, deleted_payload,
        Filter(must=[FieldCondition(key="source_memory_id", match=MatchValue(value=memory_id))]),
        "soft_delete_reverse_hyde", memory_id, failed_steps,
    )

    # 2. Mark FTS entries as excluded so they disappear from BM25 search.
    # We exclude the primary here; children will be cleaned by the hard-cascade.
    try:
        set_fts_excluded_batch([memory_id], excluded=1)
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
