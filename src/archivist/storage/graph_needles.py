"""Deterministic needle registry — O(1) recall for structured tokens (v2.0).

Split from the former monolithic ``storage/graph.py`` (INIT-001/SPEC-003).
Provenance: INIT-001/SPEC-003.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

from archivist.storage.graph_schema import _BATCH_CHUNK, schema_guard
from archivist.utils.chunking import NEEDLE_PATTERNS

logger = logging.getLogger("archivist.graph")

_ensure_needle_registry = schema_guard("""
    CREATE TABLE IF NOT EXISTS needle_registry (
        token TEXT NOT NULL,
        memory_id TEXT NOT NULL,
        namespace TEXT NOT NULL DEFAULT '',
        agent_id TEXT NOT NULL DEFAULT '',
        actor_id TEXT NOT NULL DEFAULT '',
        actor_type TEXT NOT NULL DEFAULT '',
        chunk_text TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        PRIMARY KEY (token, memory_id)
    );
    CREATE INDEX IF NOT EXISTS idx_needle_token ON needle_registry(token);
    CREATE INDEX IF NOT EXISTS idx_needle_token_ns ON needle_registry(token, namespace);
""")


async def register_needle_tokens(
    memory_id: str,
    text: str,
    namespace: str = "",
    agent_id: str = "",
    actor_id: str = "",
    actor_type: str = "",
    conn: aiosqlite.Connection | None = None,
):
    """Extract and register high-specificity tokens from text for O(1) lookup.

    Args:
        conn: Optional open ``aiosqlite.Connection``.  When provided (e.g. from
            inside a ``MemoryTransaction``), writes join the caller's transaction
            instead of acquiring a new ``pool.write()`` lock.  When ``None``
            (default), a fresh write-lock is acquired from the pool.
    """
    import aiosqlite as _aiosqlite

    from archivist.storage.sqlite_pool import pool

    _ensure_needle_registry()
    tokens: set[str] = set()
    for pat in NEEDLE_PATTERNS:
        for mt in pat.finditer(text):
            tok = mt.group().strip()
            if tok and len(tok) >= 3:
                tokens.add(tok)
    if not tokens:
        return
    now = datetime.now(UTC).isoformat()
    snippet = text[:500]

    async def _run(c: _aiosqlite.Connection) -> None:
        for tok in tokens:
            await c.execute(
                "INSERT OR REPLACE INTO needle_registry "
                "(token, memory_id, namespace, agent_id, actor_id, actor_type, chunk_text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tok, memory_id, namespace, agent_id, actor_id, actor_type, snippet, now),
            )

    try:
        if conn is not None:
            await _run(conn)
        else:
            async with pool.write() as c:
                await _run(c)
    except Exception as e:
        logging.getLogger("archivist.graph").warning("Needle registry insert failed: %s", e)


async def lookup_needle_tokens(query: str, namespace: str = "", agent_id: str = "") -> list[dict]:
    """Find exact token matches in the needle registry. O(1) per token."""
    from archivist.storage.sqlite_pool import pool

    _ensure_needle_registry()
    tokens: set[str] = set()
    for pat in NEEDLE_PATTERNS:
        for mt in pat.finditer(query):
            tok = mt.group().strip()
            if tok and len(tok) >= 3:
                tokens.add(tok)
    if not tokens:
        return []
    try:
        async with pool.read() as conn:
            results: list[dict] = []
            seen_ids: set[str] = set()
            for tok in tokens:
                where = "WHERE token = ?"
                params: list = [tok]
                if namespace:
                    where += " AND namespace = ?"
                    params.append(namespace)
                if agent_id:
                    where += " AND agent_id = ?"
                    params.append(agent_id)
                cur = await conn.execute(f"SELECT * FROM needle_registry {where}", params)
                for row in await cur.fetchall():
                    r = dict(row)
                    if r["memory_id"] not in seen_ids:
                        seen_ids.add(r["memory_id"])
                        results.append(r)
            return results
    except Exception as e:
        logging.getLogger("archivist.graph").warning("Needle registry lookup failed: %s", e)
        return []


async def delete_needle_tokens_by_memory(memory_id: str) -> int:
    """Remove all registry entries for a given memory ID.

    Thin wrapper around :func:`delete_needle_tokens_batch` for single-ID callers.
    """
    return await delete_needle_tokens_batch([memory_id])


async def delete_needle_tokens_batch(
    memory_ids: list[str],
    conn: aiosqlite.Connection | None = None,
) -> int:
    """Remove needle_registry rows for multiple memory IDs.

    Internally chunks the ID list into groups of 500 to stay under the
    sqlite3 ~999-parameter limit.  When acquiring its own connection,
    retries once on ``OperationalError`` (e.g. "database is locked").

    Args:
        conn: Optional open ``aiosqlite.Connection`` (e.g. from a
            ``MemoryTransaction``).  When provided, the deletes join the
            caller's transaction instead of acquiring a new ``pool.write()``
            lock, and the locked-database retry is skipped (see
            ``delete_fts_chunks_batch`` for the rationale). When ``None``
            (default) a fresh write-lock is acquired from the pool
            (INIT-022/SPEC-004, `M4`).
    """
    if not memory_ids:
        return 0
    from archivist.storage.sqlite_pool import pool

    _ensure_needle_registry()

    async def _run(c: aiosqlite.Connection) -> int:
        total = 0
        for i in range(0, len(memory_ids), _BATCH_CHUNK):
            chunk = memory_ids[i : i + _BATCH_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cur = await c.execute(
                f"DELETE FROM needle_registry WHERE memory_id IN ({placeholders})",
                chunk,
            )
            total += cur.rowcount
        return total

    if conn is not None:
        return await _run(conn)

    for attempt in range(2):
        try:
            async with pool.write() as c:
                return await _run(c)
        except Exception as e:
            if attempt == 0 and "locked" in str(e).lower():
                import asyncio as _asyncio

                await _asyncio.sleep(0.2)
                continue
            raise
    return 0
