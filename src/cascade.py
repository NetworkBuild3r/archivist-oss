"""Cascade helpers for memory deletion and archival.

Provides private Qdrant wrappers, paginated scroll, and the orphan sweeper.
memory_lifecycle.py is the public API — external callers should never import
from this module directly.

Consistency Model
-----------------
There is no transactional atomicity across Qdrant + SQLite.  A crash midway
through a cascade will leave partial state.  The contract:

  1. **Detectable** — ``failed_steps`` in ``DeleteResult`` records which steps
     failed; ``PartialDeletionError`` is raised when critical steps fail.
  2. **Repairable** — ``sweep_orphans()`` periodically reconciles SQLite rows
     against Qdrant point existence and cleans up stragglers.
  3. **Auditable** — Every delete/archive is logged to ``audit_log`` with the
     full result including ``failed_steps``.
"""

import logging

from qdrant_client.models import Filter, FieldCondition, MatchValue

import metrics as m

logger = logging.getLogger("archivist.cascade")

_BATCH_CHUNK_SIZE = 500


class PartialDeletionError(Exception):
    """Raised when critical cascade steps fail during memory deletion."""

    def __init__(self, result):
        self.result = result
        super().__init__(
            f"Partial deletion for {result.memory_id}: "
            f"failed_steps={result.failed_steps}"
        )


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def _qdrant_delete(
    client,
    col: str,
    selector,
    step_name: str,
    memory_id: str,
    failed_steps: list[str],
) -> int:
    """Delete Qdrant points and return a count.

    For a list selector the count is ``len(selector)``.  For a Filter selector
    the count is obtained via ``client.count`` *before* deleting (the delete
    response does not include a deleted-point count).

    On failure the *step_name* is appended to *failed_steps* and 0 is returned.
    """
    try:
        count = 0
        if isinstance(selector, list):
            count = len(selector)
        elif isinstance(selector, Filter):
            resp = client.count(collection_name=col, count_filter=selector)
            count = getattr(resp, "count", 0)
        client.delete(collection_name=col, points_selector=selector)
        return count
    except Exception as e:
        logger.error(
            "cascade.%s failed for %s: %s", step_name, memory_id, e,
        )
        failed_steps.append(step_name)
        return 0


def _qdrant_set_payload(
    client,
    col: str,
    payload: dict,
    selector,
    step_name: str,
    memory_id: str,
    failed_steps: list[str],
) -> bool:
    """Set *payload* on Qdrant points matching *selector*.

    Returns ``True`` on success.  On failure the *step_name* is appended to
    *failed_steps* and ``False`` is returned.
    """
    try:
        client.set_payload(
            collection_name=col, payload=payload, points=selector,
        )
        return True
    except Exception as e:
        logger.error(
            "cascade.%s failed for %s: %s", step_name, memory_id, e,
        )
        failed_steps.append(step_name)
        return False


# ---------------------------------------------------------------------------
# Paginated scroll
# ---------------------------------------------------------------------------

def _scroll_all(
    client,
    col: str,
    filt: Filter,
    step_name: str,
    memory_id: str,
    failed_steps: list[str],
    batch: int = _BATCH_CHUNK_SIZE,
) -> list[str]:
    """Paginate ``client.scroll`` until ``next_page_offset`` is ``None``.

    Returns a list of all matching point IDs (as strings).
    """
    ids: list[str] = []
    offset = None
    try:
        while True:
            pts, next_offset = client.scroll(
                collection_name=col,
                scroll_filter=filt,
                offset=offset,
                limit=batch,
                with_payload=False,
            )
            ids.extend(str(p.id) for p in pts)
            if next_offset is None:
                break
            offset = next_offset
    except Exception as e:
        logger.error(
            "cascade.%s failed for %s: %s", step_name, memory_id, e,
        )
        failed_steps.append(step_name)
    return ids


# ---------------------------------------------------------------------------
# Orphan sweeper
# ---------------------------------------------------------------------------

def sweep_orphans() -> dict[str, int]:
    """Reconcile SQLite rows against Qdrant point existence.

    Scans ``memory_chunks`` and ``needle_registry`` for IDs that have no
    corresponding Qdrant point and removes the orphaned rows.

    Returns a dict with counts of cleaned rows per table.
    """
    from graph import get_db, GRAPH_WRITE_LOCK, _delete_fts_rows
    from qdrant import qdrant_client
    from collection_router import collections_for_query

    client = qdrant_client()
    collections = collections_for_query("")

    conn = get_db()
    try:
        all_qdrant_ids = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT qdrant_id FROM memory_chunks"
            ).fetchall()
        ]
    finally:
        conn.close()

    orphan_ids: list[str] = []
    for i in range(0, len(all_qdrant_ids), 100):
        batch = all_qdrant_ids[i : i + 100]
        found_ids: set[str] = set()
        for coll in collections:
            try:
                points = client.retrieve(
                    collection_name=coll, ids=batch, with_payload=False,
                )
                found_ids.update(str(p.id) for p in points)
            except Exception as e:
                logger.warning("sweep_orphans.retrieve failed for %s: %s", coll, e)
                found_ids.update(batch)
        orphan_ids.extend(bid for bid in batch if bid not in found_ids)

    fts_cleaned = 0
    needle_cleaned = 0

    if orphan_ids:
        from graph import delete_fts_chunks_batch, delete_needle_tokens_batch
        try:
            fts_cleaned = delete_fts_chunks_batch(orphan_ids)
        except Exception as e:
            logger.warning("sweep_orphans.fts_batch failed: %s", e)
        try:
            needle_cleaned = delete_needle_tokens_batch(orphan_ids)
        except Exception as e:
            logger.warning("sweep_orphans.needle_batch failed: %s", e)

    total = fts_cleaned + needle_cleaned
    if total:
        logger.info(
            "sweep_orphans cleaned %d FTS + %d needle orphans",
            fts_cleaned, needle_cleaned,
        )
    m.inc(m.ORPHAN_SWEEP, value=total)

    return {"fts_cleaned": fts_cleaned, "needle_cleaned": needle_cleaned}
