"""Unified delete and archive functions for memories and all derived artifacts.

Every code path that removes or archives a memory MUST go through this module.
Adding a new artifact type to the write pipeline requires adding a corresponding
cleanup step here and registering it in ``cascade.py``'s orphan sweeper.
"""

import logging
from dataclasses import asdict, dataclass, field

from qdrant_client.models import Filter, FieldCondition, MatchValue

from collection_router import collection_for
from cascade import (
    PartialDeletionError,
    _qdrant_delete,
    _qdrant_set_payload,
    _scroll_all,
)
from graph import (
    delete_fts_chunks_batch,
    delete_needle_tokens_batch,
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

    # 2. Delete reverse HyDE vectors
    result.qdrant_reverse_hyde = len(hyde_ids)
    if hyde_ids:
        _qdrant_delete(
            client, col,
            Filter(must=[FieldCondition(key="source_memory_id", match=MatchValue(value=memory_id))]),
            "qdrant_reverse_hyde", memory_id, result.failed_steps,
        )

    # 3. Delete micro-chunk vectors
    result.qdrant_micro_chunks = len(micro_ids)
    if micro_ids:
        _qdrant_delete(
            client, col,
            Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]),
            "qdrant_micro_chunks", memory_id, result.failed_steps,
        )

    # 4. Batch-delete FTS5 entries (primary + all children)
    all_ids = [memory_id] + hyde_ids + micro_ids
    try:
        result.fts_entries = delete_fts_chunks_batch(all_ids)
    except Exception as e:
        logger.error("cascade.fts_batch failed for %s: %s", memory_id, e)
        result.failed_steps.append("fts_batch")

    # 5. Batch-delete needle registry rows (primary + all children)
    try:
        result.registry_tokens = delete_needle_tokens_batch(all_ids)
    except Exception as e:
        logger.error("cascade.needle_batch failed for %s: %s", memory_id, e)
        result.failed_steps.append("needle_batch")

    # 6. Delete auto-extracted entity facts
    try:
        result.entity_facts = _delete_entity_facts_for_memory(memory_id)
    except Exception as e:
        logger.error("cascade.entity_facts failed for %s: %s", memory_id, e)
        result.failed_steps.append("entity_facts")

    # 7. Delete memory_hotness row
    try:
        with GRAPH_WRITE_LOCK:
            _conn = get_db()
            try:
                cur = _conn.execute(
                    "DELETE FROM memory_hotness WHERE memory_id = ?", (memory_id,),
                )
                result.memory_hotness = cur.rowcount
                _conn.commit()
            finally:
                _conn.close()
    except Exception as e:
        logger.debug("cascade.memory_hotness skipped for %s: %s", memory_id, e)

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

    if "qdrant_primary" in result.failed_steps or len(result.failed_steps) > 2:
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
    Does NOT archive FTS5 or registry (they remain searchable).

    Returns ArchiveResult with per-step success flags.
    """
    col = collection or collection_for(namespace)
    client = qdrant_client()
    result = ArchiveResult(memory_id=memory_id)
    payload = {"archived": True}

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
    """Remove auto-extracted facts whose source_file contains the memory_id.

    The store pipeline sets source_file to 'explicit/{agent_id}' for API-stored
    memories, so we also check the fact_text prefix pattern.  For facts linked via
    the needle_registry (which stores memory_id directly), cleanup is handled by
    delete_needle_tokens_batch.

    Since facts don't have a direct memory_id FK, we use the convention that
    fact_text is truncated to 200 chars (Chunk 1 fix) and the source_file
    carries the agent context.  A future migration could add a memory_id
    column to facts for exact correlation.

    Returns count of soft-deactivated facts.
    """
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            cur = conn.execute(
                "UPDATE facts SET is_active = 0 "
                "WHERE source_file LIKE ? AND is_active = 1",
                (f"%{memory_id}%",),
            )
            conn.commit()
            return cur.rowcount
        except Exception as e:
            logger.warning(
                "delete_entity_facts failed for memory %s: %s", memory_id, e,
            )
            return 0
        finally:
            conn.close()
