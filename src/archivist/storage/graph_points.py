"""Hotness cleanup, Qdrant point tracking, and delete dead-letter logging.

Split from the former monolithic ``storage/graph.py`` (INIT-001/SPEC-003).
Provenance: INIT-001/SPEC-003.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from archivist.storage.graph_schema import _BATCH_CHUNK

logger = logging.getLogger("archivist.graph")


async def delete_hotness(memory_id: str) -> int:
    """Remove the ``memory_hotness`` row for *memory_id*.

    Returns the number of rows deleted (0 or 1).  Silently returns 0 if the
    ``memory_hotness`` table does not yet exist (it is lazily created by
    ``hotness.refresh_hotness``).  Any other database error is logged at
    ``warning`` for visibility — the return contract (always ``0`` on
    failure, since this runs as part of hard-delete cascade cleanup) is
    unchanged; only the missing error visibility is fixed.
    """
    from archivist.storage.sqlite_pool import pool

    try:
        async with pool.write() as conn:
            cur = await conn.execute(
                "DELETE FROM memory_hotness WHERE memory_id = ?",
                (memory_id,),
            )
            return cur.rowcount
    except Exception as e:
        if "no such table" in str(e).lower():
            return 0
        logger.warning("delete_hotness failed for %s: %s", memory_id, e)
        return 0


async def register_memory_points_batch(
    points: list[dict],
) -> int:
    """Insert rows into ``memory_points`` for a batch of Qdrant points.

    Each element of *points* must have::

        {
            "memory_id": str,   # primary memory Qdrant ID
            "qdrant_id": str,   # this point's Qdrant ID
            "point_type": str,  # "primary" | "micro_chunk" | "reverse_hyde"
        }

    Rows are inserted with ``INSERT OR IGNORE`` so re-running on the same IDs
    is idempotent.

    Returns the number of newly inserted rows.
    """
    if not points:
        return 0
    from archivist.storage.sqlite_pool import pool

    now = datetime.now(UTC).isoformat()
    total = 0
    try:
        async with pool.write() as conn:
            for i in range(0, len(points), _BATCH_CHUNK):
                chunk = points[i : i + _BATCH_CHUNK]
                await conn.executemany(
                    "INSERT OR IGNORE INTO memory_points "
                    "(memory_id, qdrant_id, point_type, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (p["memory_id"], p["qdrant_id"], p.get("point_type", "primary"), now)
                        for p in chunk
                    ],
                )
                total += len(chunk)
    except Exception as e:
        logging.getLogger("archivist.graph").warning(
            "register_memory_points_batch failed: %s",
            e,
        )
    return total


async def lookup_memory_points(memory_id: str) -> list[dict]:
    """Return all ``memory_points`` rows for *memory_id*.

    Result rows have keys: ``memory_id``, ``qdrant_id``, ``point_type``,
    ``created_at``.  Returns an empty list if no rows exist (legacy memory
    created before Phase 2).
    """
    from archivist.storage.sqlite_pool import pool

    try:
        async with pool.read() as conn:
            rows = await (
                await conn.execute(
                    "SELECT memory_id, qdrant_id, point_type, created_at "
                    "FROM memory_points WHERE memory_id = ?",
                    (memory_id,),
                )
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logging.getLogger("archivist.graph").warning(
            "lookup_memory_points failed for %s: %s",
            memory_id,
            e,
        )
        return []


async def delete_memory_points(memory_id: str) -> int:
    """Remove all ``memory_points`` rows for *memory_id*.

    Called by the hard-delete cascade after Qdrant points have been removed.
    Returns the number of rows deleted.
    """
    from archivist.storage.sqlite_pool import pool

    try:
        async with pool.write() as conn:
            cur = await conn.execute(
                "DELETE FROM memory_points WHERE memory_id = ?",
                (memory_id,),
            )
            return cur.rowcount
    except Exception as e:
        logging.getLogger("archivist.graph").warning(
            "delete_memory_points failed for %s: %s",
            memory_id,
            e,
        )
        return 0


async def log_delete_failure(memory_id: str, qdrant_ids: list[str], error: str) -> None:
    """Record a failed Qdrant delete to the ``delete_failures`` dead-letter table.

    Used by the hard-delete cascade when a Qdrant batch delete fails so that
    the orphaned IDs can be inspected and retried later.
    """
    import json as _json
    import uuid as _uuid

    from archivist.storage.sqlite_pool import pool

    now = datetime.now(UTC).isoformat()
    try:
        async with pool.write() as conn:
            await conn.execute(
                "INSERT INTO delete_failures (id, memory_id, qdrant_ids, error, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(_uuid.uuid4()), memory_id, _json.dumps(qdrant_ids), error, now),
            )
    except Exception as e:
        logging.getLogger("archivist.graph").warning(
            "log_delete_failure insert failed for %s: %s",
            memory_id,
            e,
        )
