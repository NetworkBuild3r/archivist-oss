"""Unified delete and archive functions for memories and all derived artifacts.

Every code path that removes or archives a memory MUST go through this module.
Adding a new artifact type to the write pipeline requires adding a corresponding
cleanup step here.
"""

import logging
from dataclasses import dataclass, field

from qdrant_client.models import Filter, FieldCondition, MatchValue

from collection_router import collection_for
from graph import (
    delete_fts_chunks_by_qdrant_id,
    delete_needle_tokens_by_memory,
)
from qdrant import qdrant_client
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

    @property
    def total(self) -> int:
        return (
            self.qdrant_primary
            + self.qdrant_reverse_hyde
            + self.qdrant_micro_chunks
            + self.fts_entries
            + self.registry_tokens
            + self.entity_facts
        )


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

    Side effects:
        - Removes Qdrant points (primary + reverse HyDE + micro-chunks)
        - Removes FTS5 entries (memory_chunks + memory_fts + memory_fts_exact)
        - Removes needle_registry rows
        - Removes auto-extracted entity facts
        - Logs structured deletion summary
    """
    col = collection or collection_for(namespace)
    client = qdrant_client()
    result = DeleteResult(memory_id=memory_id)

    # 0. Enumerate child point IDs BEFORE deleting them from Qdrant.
    #    Micro-chunks and reverse HyDE vectors each have their own FTS5
    #    and needle registry rows keyed by their own qdrant_id, not by the
    #    parent memory_id.  We must collect these IDs first.
    _child_ids: list[str] = []
    for _filter_key in ("parent_id", "source_memory_id"):
        try:
            _pts, _ = client.scroll(
                collection_name=col,
                scroll_filter=Filter(
                    must=[FieldCondition(key=_filter_key, match=MatchValue(value=memory_id))]
                ),
                limit=500,
                with_payload=False,
            )
            _child_ids.extend(str(p.id) for p in _pts)
        except Exception as e:
            logger.warning("delete_cascade.enumerate_%s failed for %s: %s", _filter_key, memory_id, e)

    # 1. Delete primary Qdrant point
    try:
        client.delete(collection_name=col, points_selector=[memory_id])
        result.qdrant_primary = 1
    except Exception as e:
        logger.warning("delete_cascade.qdrant_primary failed for %s: %s", memory_id, e)

    # 2. Delete reverse HyDE vectors (source_memory_id == memory_id)
    try:
        resp = client.delete(
            collection_name=col,
            points_selector=Filter(
                must=[FieldCondition(key="source_memory_id", match=MatchValue(value=memory_id))]
            ),
        )
        result.qdrant_reverse_hyde = getattr(resp, "operation_id", 0) or 0
    except Exception as e:
        logger.warning("delete_cascade.qdrant_reverse_hyde failed for %s: %s", memory_id, e)

    # 3. Delete micro-chunk vectors (parent_id == memory_id)
    try:
        resp = client.delete(
            collection_name=col,
            points_selector=Filter(
                must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]
            ),
        )
        result.qdrant_micro_chunks = getattr(resp, "operation_id", 0) or 0
    except Exception as e:
        logger.warning("delete_cascade.qdrant_micro_chunks failed for %s: %s", memory_id, e)

    # 4. Delete FTS5 entries — primary + all enumerated children
    _fts_deleted = 0
    _all_fts_ids = [memory_id] + _child_ids
    for _fts_id in _all_fts_ids:
        try:
            _fts_deleted += delete_fts_chunks_by_qdrant_id(_fts_id)
        except Exception as e:
            logger.warning("delete_cascade.fts failed for %s: %s", _fts_id, e)
    result.fts_entries = _fts_deleted

    # 5. Delete needle registry rows — primary + all enumerated children
    _reg_deleted = 0
    for _reg_id in _all_fts_ids:
        try:
            _reg_deleted += delete_needle_tokens_by_memory(_reg_id)
        except Exception as e:
            logger.warning("delete_cascade.needle_registry failed for %s: %s", _reg_id, e)
    result.registry_tokens = _reg_deleted

    # 6. Delete auto-extracted entity facts linked to this memory
    try:
        result.entity_facts = _delete_entity_facts_for_memory(memory_id)
    except Exception as e:
        logger.warning("delete_cascade.entity_facts failed for %s: %s", memory_id, e)

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
            "total_artifacts": result.total,
        },
    )

    return result


async def archive_memory_complete(
    memory_id: str,
    namespace: str,
    *,
    collection: str | None = None,
) -> int:
    """Set archived=True on a memory and ALL derived Qdrant points.

    Archives: primary point + reverse HyDE + micro-chunks.
    Does NOT archive FTS5 or registry (they remain searchable).

    Returns count of Qdrant points archived.
    """
    col = collection or collection_for(namespace)
    client = qdrant_client()
    archived = 0

    # Archive primary point
    try:
        client.set_payload(
            collection_name=col,
            payload={"archived": True},
            points=[memory_id],
        )
        archived += 1
    except Exception as e:
        logger.warning("archive_cascade.primary failed for %s: %s", memory_id, e)

    # Archive reverse HyDE points
    try:
        client.set_payload(
            collection_name=col,
            payload={"archived": True},
            points=Filter(
                must=[FieldCondition(key="source_memory_id", match=MatchValue(value=memory_id))]
            ),
        )
    except Exception as e:
        logger.warning("archive_cascade.reverse_hyde failed for %s: %s", memory_id, e)

    # Archive micro-chunk points
    try:
        client.set_payload(
            collection_name=col,
            payload={"archived": True},
            points=Filter(
                must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]
            ),
        )
    except Exception as e:
        logger.warning("archive_cascade.micro_chunks failed for %s: %s", memory_id, e)

    m.inc(m.ARCHIVE_COMPLETE, {"namespace": namespace})

    logger.info(
        "memory.archived_complete",
        extra={"memory_id": memory_id, "namespace": namespace, "collection": col},
    )

    return archived


def _delete_entity_facts_for_memory(memory_id: str) -> int:
    """Remove auto-extracted facts whose source_file contains the memory_id.

    The store pipeline sets source_file to 'explicit/{agent_id}' for API-stored
    memories, so we also check the fact_text prefix pattern.  For facts linked via
    the needle_registry (which stores memory_id directly), cleanup is handled by
    delete_needle_tokens_by_memory.

    Since facts don't have a direct memory_id FK, we use the convention that
    fact_text is truncated to 200 chars (Chunk 1 fix) and the source_file
    carries the agent context.  A future migration could add a memory_id
    column to facts for exact correlation.

    Returns count of soft-deactivated facts.
    """
    from graph import GRAPH_WRITE_LOCK, get_db

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
