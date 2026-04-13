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
  4. **Retry-resilient** — Qdrant helpers retry once on transient errors
     (429, 503, connection timeout).  Permanent errors (404, 400) fail
     immediately.  SQLite batch functions retry once on OperationalError.
  5. **Sweeper coverage** — ``sweep_orphans()`` reconciles both
     ``memory_chunks`` (by ``qdrant_id``) and ``needle_registry`` (by
     ``memory_id``, which stores both primary and micro-chunk IDs).
     Aborts early if Qdrant is unreachable.
"""

import logging
import time

from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from qdrant_client.models import Filter, FieldCondition, MatchValue

from collection_router import collections_for_query
from graph import (
    get_db, GRAPH_WRITE_LOCK, _delete_fts_rows,
    delete_fts_chunks_batch, delete_needle_tokens_batch,
    _ensure_needle_registry,
)
from qdrant import qdrant_client
import metrics as m

logger = logging.getLogger("archivist.cascade")

_BATCH_CHUNK_SIZE = 500

_RETRYABLE_STATUS_CODES = frozenset({429, 503})


def _is_transient(exc: Exception) -> bool:
    """Return True if *exc* is a transient Qdrant error worth retrying."""
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, (ResponseHandlingException, TimeoutError, ConnectionError, OSError))


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
    retries: int = 1,
) -> int:
    """Delete Qdrant points and return a count.

    For a list selector the count is ``len(selector)``.  For a Filter selector
    the count is obtained via ``client.count`` *before* deleting (the delete
    response does not include a deleted-point count).

    Transient errors (429, 503, connection/timeout) are retried up to *retries*
    times with a 0.5 s back-off.  Permanent errors fail immediately.

    On failure the *step_name* is appended to *failed_steps* and the pre-count
    (not 0) is returned so audit logs remain truthful.
    """
    count = 0
    try:
        if isinstance(selector, list):
            count = len(selector)
        elif isinstance(selector, Filter):
            resp = client.count(collection_name=col, count_filter=selector)
            count = getattr(resp, "count", 0)
    except Exception as e:
        logger.error("cascade.%s count failed for %s: %s", step_name, memory_id, e)

    last_err = None
    for attempt in range(1 + retries):
        try:
            client.delete(collection_name=col, points_selector=selector)
            return count
        except Exception as e:
            last_err = e
            if not _is_transient(e) or attempt >= retries:
                break
            time.sleep(0.5)

    logger.error(
        "cascade.%s failed for %s after %d attempt(s): %s",
        step_name, memory_id, attempt + 1, last_err,
    )
    failed_steps.append(step_name)
    return count


def _qdrant_set_payload(
    client,
    col: str,
    payload: dict,
    selector,
    step_name: str,
    memory_id: str,
    failed_steps: list[str],
    retries: int = 1,
) -> bool:
    """Set *payload* on Qdrant points matching *selector*.

    Transient errors are retried up to *retries* times (0.5 s back-off).
    Returns ``True`` on success.  On failure the *step_name* is appended to
    *failed_steps* and ``False`` is returned.
    """
    last_err = None
    for attempt in range(1 + retries):
        try:
            client.set_payload(
                collection_name=col, payload=payload, points=selector,
            )
            return True
        except Exception as e:
            last_err = e
            if not _is_transient(e) or attempt >= retries:
                break
            time.sleep(0.5)

    logger.error(
        "cascade.%s failed for %s after %d attempt(s): %s",
        step_name, memory_id, attempt + 1, last_err,
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
    max_pages: int = 1000,
) -> list[str]:
    """Paginate ``client.scroll`` until ``next_page_offset`` is ``None``.

    Returns a list of all matching point IDs (as strings).

    *max_pages* limits pagination to ``max_pages * batch`` points (default
    500 000).  If the limit is hit a warning is logged and the step is added
    to *failed_steps* so callers know the result is incomplete.
    """
    ids: list[str] = []
    offset = None
    pages = 0
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
            pages += 1
            if next_offset is None:
                break
            if pages >= max_pages:
                logger.warning(
                    "cascade.%s hit max_pages=%d for %s — result is incomplete",
                    step_name, max_pages, memory_id,
                )
                failed_steps.append(step_name)
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

_SWEEP_PAGE_SIZE = 5000


def sweep_orphans() -> dict[str, int | str]:
    """Reconcile SQLite rows against Qdrant point existence.

    Scans ``memory_chunks`` (by ``qdrant_id``) and ``needle_registry``
    (by ``memory_id``) for IDs that have no corresponding Qdrant point and
    removes the orphaned rows.  Uses keyset pagination (``WHERE id > ?
    ORDER BY id``) for O(n) scanning regardless of table size.

    Aborts early with ``{"skipped": "qdrant_unavailable"}`` if Qdrant is
    unreachable at the start of the sweep.

    Returns a dict with counts of cleaned rows per table.
    """
    client = qdrant_client()

    try:
        client.get_collections()
    except Exception as e:
        logger.warning("sweep_orphans: Qdrant unavailable, skipping: %s", e)
        return {"fts_cleaned": 0, "needle_cleaned": 0, "skipped": "qdrant_unavailable"}

    collections = collections_for_query("")

    # --- Phase 1: memory_chunks orphan scan (keyset pagination) ---
    orphan_ids: list[str] = []
    last_id = ""
    while True:
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT DISTINCT qdrant_id FROM memory_chunks "
                "WHERE qdrant_id > ? ORDER BY qdrant_id LIMIT ?",
                (last_id, _SWEEP_PAGE_SIZE),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            break

        batch_ids = [r[0] for r in rows]
        last_id = batch_ids[-1]
        for i in range(0, len(batch_ids), 100):
            sub = batch_ids[i : i + 100]
            found_ids: set[str] = set()
            any_retrieve_failed = False
            for coll in collections:
                try:
                    points = client.retrieve(
                        collection_name=coll, ids=sub, with_payload=False,
                    )
                    found_ids.update(str(p.id) for p in points)
                except Exception:
                    any_retrieve_failed = True
                    break
            if any_retrieve_failed:
                continue
            orphan_ids.extend(bid for bid in sub if bid not in found_ids)

    fts_cleaned = 0
    needle_cleaned_from_fts = 0

    if orphan_ids:
        try:
            fts_cleaned = delete_fts_chunks_batch(orphan_ids)
        except Exception as e:
            logger.warning("sweep_orphans.fts_batch failed: %s", e)
        try:
            needle_cleaned_from_fts = delete_needle_tokens_batch(orphan_ids)
        except Exception as e:
            logger.warning("sweep_orphans.needle_batch (fts pass) failed: %s", e)

    # --- Phase 2: needle_registry orphan scan (keyset pagination) ---
    _ensure_needle_registry()
    needle_orphan_ids: list[str] = []
    last_id = ""
    while True:
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT DISTINCT memory_id FROM needle_registry "
                "WHERE memory_id > ? ORDER BY memory_id LIMIT ?",
                (last_id, _SWEEP_PAGE_SIZE),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            break

        batch_ids = [r[0] for r in rows]
        last_id = batch_ids[-1]
        for i in range(0, len(batch_ids), 100):
            sub = batch_ids[i : i + 100]
            found: set[str] = set()
            any_retrieve_failed = False
            for coll in collections:
                try:
                    points = client.retrieve(
                        collection_name=coll, ids=sub, with_payload=False,
                    )
                    found.update(str(p.id) for p in points)
                except Exception:
                    any_retrieve_failed = True
                    break
            if any_retrieve_failed:
                continue
            needle_orphan_ids.extend(bid for bid in sub if bid not in found)

    needle_cleaned_direct = 0
    if needle_orphan_ids:
        try:
            needle_cleaned_direct = delete_needle_tokens_batch(needle_orphan_ids)
        except Exception as e:
            logger.warning("sweep_orphans.needle_batch (direct) failed: %s", e)

    needle_cleaned = needle_cleaned_from_fts + needle_cleaned_direct
    total = fts_cleaned + needle_cleaned
    if total:
        logger.info(
            "sweep_orphans cleaned %d FTS + %d needle orphans",
            fts_cleaned, needle_cleaned,
        )
    m.inc(m.ORPHAN_SWEEP, value=total)

    return {"fts_cleaned": fts_cleaned, "needle_cleaned": needle_cleaned}
