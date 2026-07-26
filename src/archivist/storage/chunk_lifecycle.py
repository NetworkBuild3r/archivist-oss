"""Namespace-scoped memory_chunks flag helpers for suppress / supersede.

INIT-003/SPEC-007 — thin storage helpers used by lifecycle service APIs.
Does not hard-delete rows; suppress and supersede only flip durable flags.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("archivist.chunk_lifecycle")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def set_chunk_suppressed(
    memory_id: str,
    namespace: str,
    *,
    suppressed: bool = True,
) -> int:
    """Set ``is_suppressed`` on ``memory_chunks`` rows for *memory_id* in *namespace*.

    Returns the number of rows updated. Namespace-scoped — other namespaces
    are never touched.
    """
    from archivist.storage.sqlite_pool import pool

    flag = 1 if suppressed else 0
    now = _now_iso()
    async with pool.write() as conn:
        cur = await conn.execute(
            "UPDATE memory_chunks SET is_suppressed = ?, updated_at = ? "
            "WHERE qdrant_id = ? AND namespace = ?",
            (flag, now, memory_id, namespace),
        )
        return int(cur.rowcount or 0)


async def set_chunk_supersedes(
    winner_id: str,
    loser_id: str,
    namespace: str,
) -> int:
    """Link *winner_id* → *loser_id* via ``supersedes_id`` (namespace-scoped).

    Sets ``supersedes_id`` on the winner's ``memory_chunks`` row(s) in
    *namespace*. Returns rows updated.
    """
    from archivist.storage.sqlite_pool import pool

    now = _now_iso()
    async with pool.write() as conn:
        cur = await conn.execute(
            "UPDATE memory_chunks SET supersedes_id = ?, updated_at = ? "
            "WHERE qdrant_id = ? AND namespace = ?",
            (loser_id, now, winner_id, namespace),
        )
        return int(cur.rowcount or 0)


async def get_chunk_lifecycle_row(
    memory_id: str,
    namespace: str,
) -> dict[str, Any] | None:
    """Return lifecycle flags for a chunk in *namespace*, or ``None`` if absent."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        row = await conn.fetchone(
            "SELECT qdrant_id, namespace, is_suppressed, supersedes_id, is_excluded "
            "FROM memory_chunks WHERE qdrant_id = ? AND namespace = ? LIMIT 1",
            (memory_id, namespace),
        )
    return dict(row) if row is not None else None


async def list_superseded_loser_ids(namespace: str) -> set[str]:
    """Return qdrant_ids in *namespace* that are pointed to by a winner's ``supersedes_id``."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        rows = await conn.fetchall(
            "SELECT DISTINCT supersedes_id FROM memory_chunks "
            "WHERE namespace = ? AND supersedes_id != '' AND supersedes_id IS NOT NULL",
            (namespace,),
        )
    out: set[str] = set()
    for r in rows:
        value = r["supersedes_id"] if hasattr(r, "keys") else r[0]
        if value:
            out.add(str(value))
    return out


async def is_chunk_soft_deleted(memory_id: str, namespace: str) -> bool:
    """True when the chunk is already tombstoned (``is_excluded=1``) in *namespace*."""
    row = await get_chunk_lifecycle_row(memory_id, namespace)
    if row is None:
        return False
    return bool(row.get("is_excluded"))
