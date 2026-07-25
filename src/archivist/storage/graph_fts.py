"""FTS5 (SQLite) / tsvector (Postgres) chunk upsert, delete, and BM25 search.

Split from the former monolithic ``storage/graph.py`` (INIT-001/SPEC-003).
Provenance: INIT-001/SPEC-003.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

import archivist.core.metrics as m
from archivist.storage.graph_schema import _BATCH_CHUNK

logger = logging.getLogger("archivist.graph")


async def upsert_fts_chunk(
    qdrant_id: str,
    text: str,
    file_path: str,
    chunk_index: int,
    agent_id: str = "",
    namespace: str = "",
    date: str = "",
    memory_type: str = "general",
    actor_id: str = "",
    actor_type: str = "",
    importance: float = 0.5,
    tier_label: str = "l2",
    conn: aiosqlite.Connection | None = None,
):
    """Insert or replace a chunk in memory_chunks and sync to both FTS indexes.

    On SQLite this also maintains the FTS5 shadow-row tables (``memory_fts``
    and ``memory_fts_exact``).  On Postgres the ``fts_vector`` /
    ``fts_vector_simple`` columns are ``GENERATED ALWAYS AS ... STORED``
    so they update automatically when the ``text`` column changes — no
    shadow-row maintenance is needed.

    Args:
        importance: 0.0–1.0 priority signal for tier-aware retrieval packing.
        tier_label: One of 'l0' | 'l1' | 'l2' | 'ephemeral'.
        conn: Optional open ``aiosqlite.Connection``.  When provided (e.g. from
            inside a ``MemoryTransaction``), writes join the caller's transaction
            instead of acquiring a new ``pool.write()`` lock.  When ``None``
            (default), a fresh write-lock is acquired from the pool.
    """
    from archivist.core.config import GRAPH_BACKEND

    if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
        await _upsert_fts_chunk_postgres(
            qdrant_id=qdrant_id,
            text=text,
            file_path=file_path,
            chunk_index=chunk_index,
            agent_id=agent_id,
            namespace=namespace,
            date=date,
            memory_type=memory_type,
            actor_id=actor_id,
            actor_type=actor_type,
            importance=importance,
            tier_label=tier_label,
        )
        m.inc(m.FTS_UPSERT_TOTAL, {"backend": "postgres"})
        return

    await _upsert_fts_chunk_sqlite(
        qdrant_id=qdrant_id,
        text=text,
        file_path=file_path,
        chunk_index=chunk_index,
        agent_id=agent_id,
        namespace=namespace,
        date=date,
        memory_type=memory_type,
        actor_id=actor_id,
        actor_type=actor_type,
        importance=importance,
        tier_label=tier_label,
        conn=conn,
    )
    m.inc(m.FTS_UPSERT_TOTAL, {"backend": "sqlite"})


async def _upsert_fts_chunk_postgres(
    qdrant_id: str,
    text: str,
    file_path: str,
    chunk_index: int,
    agent_id: str = "",
    namespace: str = "",
    date: str = "",
    memory_type: str = "general",
    actor_id: str = "",
    actor_type: str = "",
    importance: float = 0.5,
    tier_label: str = "l2",
) -> None:
    """Postgres upsert: insert/replace memory_chunks row.

    ``fts_vector`` and ``fts_vector_simple`` are GENERATED ALWAYS AS STORED
    columns, so no shadow-row maintenance is required.
    """
    from archivist.storage.sqlite_pool import pool

    try:
        async with pool.write() as conn:
            await conn.execute(
                "INSERT INTO memory_chunks "
                "(qdrant_id, text, file_path, chunk_index, agent_id, namespace, date, "
                "memory_type, actor_id, actor_type, importance, tier_label) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (qdrant_id) DO UPDATE SET "
                "text=EXCLUDED.text, file_path=EXCLUDED.file_path, "
                "chunk_index=EXCLUDED.chunk_index, agent_id=EXCLUDED.agent_id, "
                "namespace=EXCLUDED.namespace, date=EXCLUDED.date, "
                "memory_type=EXCLUDED.memory_type, actor_id=EXCLUDED.actor_id, "
                "actor_type=EXCLUDED.actor_type, importance=EXCLUDED.importance, "
                "tier_label=EXCLUDED.tier_label",
                (
                    qdrant_id,
                    text,
                    file_path,
                    chunk_index,
                    agent_id,
                    namespace,
                    date,
                    memory_type,
                    actor_id,
                    actor_type,
                    importance,
                    tier_label,
                ),
            )
    except Exception as e:
        m.inc(m.FTS_UPSERT_ERRORS_TOTAL, {"backend": "postgres"})
        logging.getLogger("archivist.graph").warning(
            "Postgres FTS upsert failed for %s: %s", qdrant_id, e
        )


async def _upsert_fts_chunk_sqlite(
    qdrant_id: str,
    text: str,
    file_path: str,
    chunk_index: int,
    agent_id: str = "",
    namespace: str = "",
    date: str = "",
    memory_type: str = "general",
    actor_id: str = "",
    actor_type: str = "",
    importance: float = 0.5,
    tier_label: str = "l2",
    conn: aiosqlite.Connection | None = None,
) -> None:
    """SQLite upsert: insert/replace memory_chunks row and maintain FTS5 shadow rows."""
    import aiosqlite as _aiosqlite

    from archivist.storage.sqlite_pool import pool

    async def _run(c: _aiosqlite.Connection) -> None:
        old = await (
            await c.execute(
                "SELECT rowid, text FROM memory_chunks WHERE qdrant_id = ?", (qdrant_id,)
            )
        ).fetchone()
        if old:
            await c.execute(
                "INSERT INTO memory_fts(memory_fts, rowid, text) VALUES('delete', ?, ?)",
                (old["rowid"], old["text"]),
            )
            try:
                await c.execute(
                    "INSERT INTO memory_fts_exact(memory_fts_exact, rowid, text) VALUES('delete', ?, ?)",
                    (old["rowid"], old["text"]),
                )
            except Exception as _e:
                logger.debug(
                    "FTS shadow-row delete (exact) failed for rowid %s: %s", old["rowid"], _e
                )
            await c.execute("DELETE FROM memory_chunks WHERE qdrant_id = ?", (qdrant_id,))

        await c.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, agent_id, namespace, date, memory_type, actor_id, actor_type, importance, tier_label) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                qdrant_id,
                text,
                file_path,
                chunk_index,
                agent_id,
                namespace,
                date,
                memory_type,
                actor_id,
                actor_type,
                importance,
                tier_label,
            ),
        )
        rowid_row = await (
            await c.execute("SELECT rowid FROM memory_chunks WHERE qdrant_id = ?", (qdrant_id,))
        ).fetchone()
        rowid = rowid_row["rowid"]
        await c.execute(
            "INSERT INTO memory_fts (rowid, text) VALUES (?, ?)",
            (rowid, text),
        )
        try:
            await c.execute(
                "INSERT INTO memory_fts_exact (rowid, text) VALUES (?, ?)",
                (rowid, text),
            )
        except Exception as _e:
            logger.debug("FTS shadow-row insert (exact) failed for rowid %s: %s", rowid, _e)

    try:
        if conn is not None:
            await _run(conn)
        else:
            async with pool.write() as c:
                await _run(c)
    except Exception as e:
        logging.getLogger("archivist.graph").warning("FTS upsert failed for %s: %s", qdrant_id, e)


async def _delete_fts_rows_async(conn, rows):
    """Delete FTS5 shadow-table entries for the given memory_chunks rows (async).

    On Postgres this is a no-op — deleting the ``memory_chunks`` row
    automatically removes the corresponding ``GENERATED ALWAYS AS`` tsvector
    data.  On SQLite the FTS5 shadow rows must be explicitly removed.

    Best-effort — FTS5 extension may be unavailable on SQLite.
    """
    from archivist.core.config import GRAPH_BACKEND

    if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
        return

    for row in rows:
        try:
            await conn.execute(
                "INSERT INTO memory_fts (memory_fts, rowid, text) VALUES ('delete', ?, "
                "(SELECT text FROM memory_chunks WHERE rowid = ?))",
                (row["rowid"], row["rowid"]),
            )
        except Exception as _e:
            logger.debug(
                "FTS shadow-row delete (stemmed) failed for rowid %s: %s", row["rowid"], _e
            )
        try:
            await conn.execute(
                "INSERT INTO memory_fts_exact (memory_fts_exact, rowid, text) VALUES ('delete', ?, "
                "(SELECT text FROM memory_chunks WHERE rowid = ?))",
                (row["rowid"], row["rowid"]),
            )
        except Exception as _e:
            logger.debug("FTS shadow-row delete (exact) failed for rowid %s: %s", row["rowid"], _e)


async def delete_fts_chunks_by_file(file_path: str):
    """Remove all FTS5 entries and memory_chunks rows for a given file path.

    FTS5 index cleanup is best-effort — memory_chunks rows are always deleted
    even if the FTS5 extension is unavailable.
    """
    from archivist.storage.sqlite_pool import pool

    try:
        async with pool.write() as conn:
            rows = await (
                await conn.execute(
                    "SELECT rowid FROM memory_chunks WHERE file_path = ?", (file_path,)
                )
            ).fetchall()
            await _delete_fts_rows_async(conn, rows)
            await conn.execute("DELETE FROM memory_chunks WHERE file_path = ?", (file_path,))
    except Exception as e:
        logging.getLogger("archivist.graph").warning("FTS delete failed for %s: %s", file_path, e)


async def delete_fts_chunks_by_qdrant_id(qdrant_id: str) -> int:
    """Remove FTS5 entries and memory_chunks rows for a single Qdrant point ID.

    Thin wrapper around :func:`delete_fts_chunks_batch` for single-ID callers.
    """
    return await delete_fts_chunks_batch([qdrant_id])


async def search_fts(
    query: str,
    namespace: str = "",
    agent_id: str = "",
    memory_type: str = "",
    limit: int = 30,
    actor_type: str = "",
    raw_query: str = "",
    fts_mode: str = "or",
) -> list[dict]:
    """BM25 keyword search via FTS5 (SQLite) or tsvector (Postgres).

    Dispatches to the appropriate search implementation based on
    the active ``GRAPH_BACKEND`` setting.

    Args:
        query: Pre-built FTS5 query string for SQLite (e.g. ``"k8s" OR "deploy"``).
            Ignored when the backend is Postgres — ``raw_query`` is used instead.
        namespace: Filter by namespace (empty = all namespaces).
        agent_id: Filter by agent ID (empty = all agents).
        memory_type: Filter by memory type (empty = all types).
        limit: Maximum number of results to return.
        actor_type: Filter by actor type (empty = all types).
        raw_query: Original unformatted user query.  Used by the Postgres backend
            to build an appropriate ``tsquery`` expression.  Falls back to
            ``query`` when empty.
        fts_mode: Query mode for Postgres tsquery building.  One of ``"or"``
            (default, high recall), ``"and"`` (high precision), or ``"phrase"``
            (sequential token match).  Ignored for SQLite.

    Returns:
        List of result dicts with ``qdrant_id``, ``bm25_score``, and payload fields.
    """
    from archivist.core.config import GRAPH_BACKEND

    if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
        return await _search_fts_postgres(
            raw_query=raw_query or query,
            fts_mode=fts_mode,
            namespace=namespace,
            agent_id=agent_id,
            memory_type=memory_type,
            limit=limit,
            actor_type=actor_type,
        )
    return await _search_fts_sqlite(
        query=query,
        namespace=namespace,
        agent_id=agent_id,
        memory_type=memory_type,
        limit=limit,
        actor_type=actor_type,
    )


def _build_fts_where(
    namespace: str, agent_id: str, memory_type: str, actor_type: str
) -> tuple[list[str], list]:
    """Build the shared namespace/agent/type/actor filter clauses common to all FTS variants."""
    where_clauses = ["mc.is_excluded = 0"]
    params: list = []

    if namespace:
        where_clauses.append("mc.namespace = ?")
        params.append(namespace)
    if agent_id:
        where_clauses.append("mc.agent_id = ?")
        params.append(agent_id)
    if memory_type:
        where_clauses.append("mc.memory_type = ?")
        params.append(memory_type)
    if actor_type:
        where_clauses.append("mc.actor_type = ?")
        params.append(actor_type)

    return where_clauses, params


async def _run_fts_query(
    sql: str,
    params: list,
    *,
    backend: str,
    negate_score: bool,
    error_context: str,
) -> list[dict]:
    """Execute a built FTS query, shape results, record metrics, and log-and-swallow errors.

    Shared tail end of all four ``_search_fts_*`` variants: connection acquisition,
    execution, ``bm25_score`` normalization, timing/count metrics, and error handling.
    """
    from archivist.storage.sqlite_pool import pool

    _t0 = time.monotonic()
    try:
        async with pool.read() as conn:
            cur = await conn.execute(sql, params)
            results = []
            for row in await cur.fetchall():
                r = dict(row)
                rank = r.pop("bm25_rank", 0)
                r["bm25_score"] = -rank if negate_score else rank
                results.append(r)
        m.observe(m.FTS_SEARCH_DURATION_MS, (time.monotonic() - _t0) * 1000.0, {"backend": backend})
        m.inc(m.FTS_SEARCH_TOTAL, {"backend": backend})
        return results
    except Exception as e:
        logging.getLogger("archivist.graph").warning("%s: %s", error_context, e)
        return []


async def _search_fts_sqlite_family(
    *,
    fts_table: str,
    match_query: str,
    namespace: str,
    agent_id: str,
    memory_type: str,
    limit: int,
    actor_type: str,
    error_context: str,
) -> list[dict]:
    """Shared SQLite FTS5 BM25 search body for both the stemmed and exact-match tables."""
    where_clauses, filter_params = _build_fts_where(namespace, agent_id, memory_type, actor_type)
    where_sql = " AND " + " AND ".join(where_clauses)

    sql = (
        "SELECT mc.qdrant_id, mc.file_path, mc.chunk_index, mc.agent_id, "
        "mc.namespace, mc.date, mc.memory_type, mc.text, "
        "mc.actor_id, mc.actor_type, mc.importance, mc.tier_label, "
        "rank AS bm25_rank "
        f"FROM {fts_table} "
        f"JOIN memory_chunks mc ON {fts_table}.rowid = mc.rowid "
        "LEFT JOIN memory_hotness mh ON mh.memory_id = mc.qdrant_id "
        f"WHERE {fts_table} MATCH ? {where_sql} "
        "ORDER BY (rank * (1 + 0.3 * COALESCE(mh.importance_signal, 0.5))) "
        "LIMIT ?"
    )
    params = [match_query, *filter_params, limit]

    return await _run_fts_query(
        sql, params, backend="sqlite", negate_score=True, error_context=error_context
    )


async def _search_fts_sqlite(
    query: str,
    namespace: str = "",
    agent_id: str = "",
    memory_type: str = "",
    limit: int = 30,
    actor_type: str = "",
) -> list[dict]:
    """SQLite FTS5 BM25 search implementation (stemmed / ``memory_fts``)."""
    return await _search_fts_sqlite_family(
        fts_table="memory_fts",
        match_query=query,
        namespace=namespace,
        agent_id=agent_id,
        memory_type=memory_type,
        limit=limit,
        actor_type=actor_type,
        error_context="FTS search failed",
    )


async def _search_fts_postgres_family(
    *,
    tsquery_expr: str,
    fts_column: str,
    ts_config: str,
    namespace: str,
    agent_id: str,
    memory_type: str,
    limit: int,
    actor_type: str,
    error_context: str,
) -> list[dict]:
    """Shared Postgres tsvector FTS search body for both the stemmed and exact-match configs."""
    if not tsquery_expr:
        return []

    where_clauses, filter_params = _build_fts_where(namespace, agent_id, memory_type, actor_type)
    where_sql = " AND " + " AND ".join(where_clauses)

    # ts_rank_cd returns values in [0,1]; multiply by 32 to normalize
    # into the same ballpark as SQLite FTS5 BM25 scores (~0.5-30 range).
    sql = (
        "SELECT mc.qdrant_id, mc.file_path, mc.chunk_index, mc.agent_id, "
        "mc.namespace, mc.date, mc.memory_type, mc.text, "
        "mc.actor_id, mc.actor_type, mc.importance, mc.tier_label, "
        f"ts_rank_cd(mc.{fts_column}, to_tsquery('{ts_config}', ?)) * 32 AS bm25_rank "
        "FROM memory_chunks mc "
        "LEFT JOIN memory_hotness mh ON mh.memory_id = mc.qdrant_id "
        f"WHERE mc.{fts_column} @@ to_tsquery('{ts_config}', ?) {where_sql} "
        f"ORDER BY (ts_rank_cd(mc.{fts_column}, to_tsquery('{ts_config}', ?)) * 32 "
        "          * (1 + 0.3 * COALESCE(mh.importance_signal, 0.5))) DESC "
        "LIMIT ?"
    )
    # tsquery_expr used three times: SELECT ranking, WHERE filter, ORDER BY
    params = [tsquery_expr, tsquery_expr, *filter_params, tsquery_expr, limit]

    return await _run_fts_query(
        sql, params, backend="postgres", negate_score=False, error_context=error_context
    )


async def _search_fts_postgres(
    raw_query: str,
    fts_mode: str = "or",
    namespace: str = "",
    agent_id: str = "",
    memory_type: str = "",
    limit: int = 30,
    actor_type: str = "",
) -> list[dict]:
    """Postgres tsvector FTS search implementation (stemmed / ``fts_vector``)."""
    from archivist.storage.fts_search import _pg_tsquery_and, _pg_tsquery_or, _pg_tsquery_phrase

    builder = {
        "or": _pg_tsquery_or,
        "and": _pg_tsquery_and,
        "phrase": _pg_tsquery_phrase,
    }.get(fts_mode, _pg_tsquery_or)
    tsquery_expr = builder(raw_query)

    return await _search_fts_postgres_family(
        tsquery_expr=tsquery_expr,
        fts_column="fts_vector",
        ts_config="english",
        namespace=namespace,
        agent_id=agent_id,
        memory_type=memory_type,
        limit=limit,
        actor_type=actor_type,
        error_context="FTS Postgres search failed",
    )


async def search_fts_exact(
    query: str,
    namespace: str = "",
    agent_id: str = "",
    memory_type: str = "",
    limit: int = 30,
    actor_type: str = "",
    raw_query: str = "",
) -> list[dict]:
    """Non-stemmed keyword search for exact token matching (IPs, UUIDs, ticket IDs).

    Dispatches to FTS5 ``memory_fts_exact`` (SQLite) or ``fts_vector_simple``
    tsvector (Postgres) based on the active ``GRAPH_BACKEND``.

    Args:
        query: Pre-built FTS5 query string for SQLite.  Ignored on Postgres.
        namespace: Filter by namespace (empty = all namespaces).
        agent_id: Filter by agent ID (empty = all agents).
        memory_type: Filter by memory type (empty = all types).
        limit: Maximum number of results to return.
        actor_type: Filter by actor type (empty = all types).
        raw_query: Original unformatted user query.  Used by the Postgres backend
            to build the ``tsquery`` expression.  Falls back to ``query`` when empty.

    Returns:
        List of result dicts with ``qdrant_id``, ``bm25_score``, and payload fields.
    """
    from archivist.core.config import GRAPH_BACKEND

    if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
        return await _search_fts_exact_postgres(
            raw_query=raw_query or query,
            namespace=namespace,
            agent_id=agent_id,
            memory_type=memory_type,
            limit=limit,
            actor_type=actor_type,
        )
    return await _search_fts_exact_sqlite(
        query=query,
        namespace=namespace,
        agent_id=agent_id,
        memory_type=memory_type,
        limit=limit,
        actor_type=actor_type,
    )


async def _search_fts_exact_sqlite(
    query: str,
    namespace: str = "",
    agent_id: str = "",
    memory_type: str = "",
    limit: int = 30,
    actor_type: str = "",
) -> list[dict]:
    """SQLite FTS5 exact (non-stemmed) BM25 search via ``memory_fts_exact``."""
    return await _search_fts_sqlite_family(
        fts_table="memory_fts_exact",
        match_query=query,
        namespace=namespace,
        agent_id=agent_id,
        memory_type=memory_type,
        limit=limit,
        actor_type=actor_type,
        error_context="FTS exact search failed",
    )


async def _search_fts_exact_postgres(
    raw_query: str,
    namespace: str = "",
    agent_id: str = "",
    memory_type: str = "",
    limit: int = 30,
    actor_type: str = "",
) -> list[dict]:
    """Postgres exact (non-stemmed) FTS search via ``fts_vector_simple``.

    Uses the ``simple`` text-search configuration which skips stemming —
    equivalent to FTS5's ``unicode61`` tokenizer.
    """
    from archivist.storage.fts_search import _pg_tsquery_or

    tsquery_expr = _pg_tsquery_or(raw_query)

    return await _search_fts_postgres_family(
        tsquery_expr=tsquery_expr,
        fts_column="fts_vector_simple",
        ts_config="simple",
        namespace=namespace,
        agent_id=agent_id,
        memory_type=memory_type,
        limit=limit,
        actor_type=actor_type,
        error_context="FTS exact Postgres search failed",
    )


async def delete_fts_chunks_batch(
    qdrant_ids: list[str],
    conn: aiosqlite.Connection | None = None,
) -> int:
    """Remove FTS5 entries and memory_chunks rows for multiple Qdrant IDs.

    Internally chunks the ID list into groups of 500 to stay under the
    sqlite3 ~999-parameter limit.  When acquiring its own connection,
    retries once on ``OperationalError`` (e.g. "database is locked").

    Args:
        conn: Optional open ``aiosqlite.Connection`` (e.g. from a
            ``MemoryTransaction``).  When provided, the deletes join the
            caller's transaction instead of acquiring a new ``pool.write()``
            lock, and the locked-database retry is skipped — a retry here
            would need to re-run the caller's *whole* transaction, not just
            this call, so the caller owns that decision. When ``None``
            (default) a fresh write-lock is acquired from the pool
            (INIT-022/SPEC-004, `M4`).
    """
    if not qdrant_ids:
        return 0
    from archivist.storage.sqlite_pool import pool

    async def _run(c: aiosqlite.Connection) -> int:
        total = 0
        for i in range(0, len(qdrant_ids), _BATCH_CHUNK):
            chunk = qdrant_ids[i : i + _BATCH_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = await (
                await c.execute(
                    f"SELECT rowid FROM memory_chunks WHERE qdrant_id IN ({placeholders})",
                    chunk,
                )
            ).fetchall()
            await _delete_fts_rows_async(c, rows)
            cur = await c.execute(
                f"DELETE FROM memory_chunks WHERE qdrant_id IN ({placeholders})",
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


async def set_fts_excluded_batch(qdrant_ids: list[str], excluded: int = 1) -> int:
    """Mark memory_chunks rows as excluded (or restore them) by Qdrant ID.

    Sets ``is_excluded`` to *excluded* (1 = excluded from search, 0 = restored).
    Used by archive and soft-delete to hide memories from BM25/FTS5 search
    without physically removing the rows.

    Chunks the ID list into groups of 500 to stay under the sqlite3 ~999
    parameter limit.
    """
    if not qdrant_ids:
        return 0
    from archivist.storage.sqlite_pool import pool

    total = 0
    try:
        async with pool.write() as conn:
            for i in range(0, len(qdrant_ids), _BATCH_CHUNK):
                chunk = qdrant_ids[i : i + _BATCH_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                cur = await conn.execute(
                    f"UPDATE memory_chunks SET is_excluded = ? WHERE qdrant_id IN ({placeholders})",
                    [excluded] + chunk,
                )
                total += cur.rowcount
    except Exception as e:
        logging.getLogger("archivist.graph").warning(
            "set_fts_excluded_batch failed: %s",
            e,
        )
    return total
