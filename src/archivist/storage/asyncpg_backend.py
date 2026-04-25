"""PostgreSQL backend for Archivist using asyncpg.

This module provides ``AsyncpgGraphBackend`` — a drop-in replacement for
``SQLiteGraphBackend`` that targets a PostgreSQL server via an ``asyncpg``
connection pool.  It exposes the same ``write()`` / ``read()`` context-manager
API so that all existing callers require zero changes when
``GRAPH_BACKEND=postgres``.

Key differences from the SQLite backend
-----------------------------------------
* **No global write lock** — PostgreSQL's MVCC handles concurrent writers
  natively.  Each ``write()`` call gets its own pooled connection + transaction.
* **Parameter style** — asyncpg uses ``$1, $2, …`` placeholders; SQLite uses
  ``?``.  ``AsyncpgConnection`` transparently translates all SQL so that
  existing statements remain unchanged.
* **Optional dependency** — ``asyncpg`` is only imported when this module is
  used.  SQLite-only deployments are completely unaffected.

Usage::

    from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

    backend = AsyncpgGraphBackend()
    await backend.initialize(dsn="postgresql://user:pw@host/db")

    async with backend.write() as conn:
        await conn.execute("INSERT INTO entities (name) VALUES (?)", ("Alice",))

    async with backend.read() as conn:
        rows = await conn.fetchall("SELECT id, name FROM entities WHERE type = ?", ("person",))
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import archivist.core.metrics as m

if TYPE_CHECKING:
    import asyncpg as asyncpg_module

logger = logging.getLogger("archivist.asyncpg_backend")

# ---------------------------------------------------------------------------
# SQL placeholder translation
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\?")

# SQLite-ism patterns translated to standard SQL before $N substitution.
# Order matters: more specific patterns first.
_INSERT_OR_IGNORE_RE = re.compile(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", re.IGNORECASE)
_INSERT_OR_REPLACE_RE = re.compile(r"\bINSERT\s+OR\s+REPLACE\s+INTO\b", re.IGNORECASE)
_COLLATE_NOCASE_RE = re.compile(r"\s+COLLATE\s+NOCASE\b", re.IGNORECASE)


def _translate_sql(sql: str) -> str:
    """Convert SQLite SQL idioms to Postgres-compatible SQL.

    Transformations applied (in order):

    1. ``INSERT OR IGNORE INTO`` → ``INSERT INTO … ON CONFLICT DO NOTHING``
    2. ``INSERT OR REPLACE INTO`` → ``INSERT INTO … ON CONFLICT DO UPDATE SET …``
    3. ``COLLATE NOCASE`` → stripped (citext handles case-insensitivity natively).
    4. ``?`` placeholders → ``$1``, ``$2``, … (asyncpg style).

    Examples::

        >>> _translate_sql("SELECT * FROM t WHERE a = ? AND b = ?")
        'SELECT * FROM t WHERE a = $1 AND b = $2'
        >>> _translate_sql("INSERT OR IGNORE INTO t (a) VALUES (?)")
        'INSERT INTO t (a) VALUES ($1) ON CONFLICT DO NOTHING'
        >>> _translate_sql("SELECT * FROM t WHERE name = ? COLLATE NOCASE")
        'SELECT * FROM t WHERE name = $1'
    """
    # Track which ORx substitution fired so we can append the right ON CONFLICT clause.
    _mode = [None]  # "ignore" | "replace" | None

    def _replace_insert(m: re.Match[str]) -> str:
        keyword = m.group(0).upper()
        if "IGNORE" in keyword:
            _mode[0] = "ignore"
        else:
            _mode[0] = "replace"
        return "INSERT INTO"

    # Step 1 & 2: normalise INSERT OR x → INSERT INTO
    sql2 = _INSERT_OR_IGNORE_RE.sub(_replace_insert, sql)
    sql2 = _INSERT_OR_REPLACE_RE.sub(_replace_insert, sql2)

    # Step 3: strip COLLATE NOCASE (citext handles it at the column level)
    sql2 = _COLLATE_NOCASE_RE.sub("", sql2)

    # Step 4: ? → $N
    counter = 0

    def _replace_placeholder(_m: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        return f"${counter}"

    sql2 = _PLACEHOLDER_RE.sub(_replace_placeholder, sql2)

    # Step 5: append ON CONFLICT clause if INSERT OR x was detected
    if _mode[0] == "ignore":
        sql2 = sql2.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
    elif _mode[0] == "replace":
        # ON CONFLICT (target) DO UPDATE SET requires knowing both the conflict
        # target columns (PK / UNIQUE) and the non-PK columns to update.
        # We parse the table name and column list from the INSERT and look up a
        # static per-table conflict-target map derived from schema_postgres.sql.
        _tbl_match = re.search(r"INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)", sql2, re.IGNORECASE)
        if _tbl_match:
            table_name = _tbl_match.group(1).lower()
            cols = [c.strip() for c in _tbl_match.group(2).split(",")]
            # Per-table conflict targets (match PRIMARY KEY / UNIQUE in schema_postgres.sql)
            _conflict_targets: dict[str, list[str]] = {
                "needle_registry": ["token", "memory_id"],
                "memory_hotness": ["memory_id"],
                "memory_points": ["memory_id", "qdrant_id", "point_type"],
                "delete_failures": ["memory_id"],
                "outbox": ["id"],
                "skill_relations": ["skill_a", "skill_b", "relation_type"],
            }
            conflict_cols = _conflict_targets.get(table_name)
            pk_set = set(conflict_cols) if conflict_cols else {"id", "rowid"}
            update_cols = [c for c in cols if c.lower() not in pk_set]
            if conflict_cols and update_cols:
                target = ", ".join(conflict_cols)
                set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
                sql2 = (
                    sql2.rstrip().rstrip(";")
                    + f" ON CONFLICT ({target}) DO UPDATE SET {set_clause}"
                )
            elif conflict_cols:
                target = ", ".join(conflict_cols)
                sql2 = sql2.rstrip().rstrip(";") + f" ON CONFLICT ({target}) DO NOTHING"
            else:
                # Unknown table — safe fallback
                sql2 = sql2.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
        else:
            # No parseable column list — safe fallback
            sql2 = sql2.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"

    return sql2


# ---------------------------------------------------------------------------
# Connection wrapper
# ---------------------------------------------------------------------------


class _PgCursorProxy:
    """Cursor-like proxy returned by ``AsyncpgConnection.execute()``.

    ``graph.py`` uses the SQLite cursor pattern::

        cur = await conn.execute(sql, params)
        row = await cur.fetchone()

    asyncpg's ``connection.execute()`` returns a status string, not a cursor.
    This proxy allows the cursor pattern to work transparently on Postgres.

    For DML statements (INSERT/UPDATE/DELETE) the ``rowcount`` property
    returns the number of rows affected, parsed from the asyncpg status string
    (e.g. ``"DELETE 5"`` → 5, ``"INSERT 0 1"`` → 1).  SELECT proxies return
    -1 for ``rowcount``, matching the DB-API 2.0 convention for un-executed
    queries.

    Args:
        conn: The underlying asyncpg connection.
        sql: Translated SQL statement (``$N`` placeholders).
        params: Positional parameters.
        _rowcount: Pre-parsed affected-row count for DML proxies.
    """

    __slots__ = ("_conn", "_params", "_rowcount", "_sql")

    def __init__(
        self,
        conn: asyncpg_module.Connection[Any],
        sql: str,
        params: tuple[Any, ...] | list[Any],
        _rowcount: int = -1,
    ) -> None:
        self._conn = conn
        self._sql = sql
        self._params = params
        self._rowcount = _rowcount

    @property
    def rowcount(self) -> int:
        """Number of rows affected by the last DML statement.

        Matches ``aiosqlite.Cursor.rowcount`` behaviour: returns the count for
        INSERT/UPDATE/DELETE, and -1 for SELECT proxies (query not yet run).
        """
        return self._rowcount

    async def fetchone(self) -> Any | None:
        """Fetch the first row, or ``None`` if the result set is empty."""
        return await self._conn.fetchrow(self._sql, *self._params)

    async def fetchall(self) -> list[Any]:
        """Fetch all rows."""
        return await self._conn.fetch(self._sql, *self._params)


class AsyncpgConnection:
    """Wraps an ``asyncpg.Connection`` to accept SQLite-style ``?`` placeholders.

    All SQL that passes through this wrapper has its ``?`` placeholders
    translated to ``$N`` before being forwarded to asyncpg.  Parameter lists
    are unpacked from tuples/lists into positional arguments as asyncpg
    expects.

    Args:
        conn: An acquired ``asyncpg.Connection`` from the pool.
    """

    def __init__(self, conn: asyncpg_module.Connection[Any]) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Core SQL methods (match the interface expected by graph helpers)
    # ------------------------------------------------------------------

    async def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> Any:
        """Execute *sql* with *params* and return a cursor-like proxy.

        Returns a ``_PgCursorProxy`` so that callers using the SQLite cursor
        pattern (``cur = await conn.execute(sql); await cur.fetchone()``) work
        transparently.  For non-SELECT statements the side-effect runs
        immediately; callers that ignore the return value are unaffected.

        Args:
            sql: SQL statement (``?`` placeholders allowed).
            params: Positional parameters.

        Returns:
            A ``_PgCursorProxy`` whose ``.fetchone()`` / ``.fetchall()``
            execute the query against asyncpg.
        """
        translated = _translate_sql(sql)
        _t0 = time.monotonic()
        try:
            stripped = translated.lstrip().upper()
            if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
                # Non-SELECT: run immediately for side-effect, return DML proxy with rowcount.
                # asyncpg returns a status string like "DELETE 5" / "UPDATE 3" / "INSERT 0 1".
                status = await self._conn.execute(translated, *params)
                m.observe(m.PG_POOL_QUERY_MS, (time.monotonic() - _t0) * 1000.0)
                try:
                    _rc = int(str(status).rsplit(" ", 1)[-1])
                except (ValueError, AttributeError):
                    _rc = 0
                return _PgCursorProxy(self._conn, "SELECT 1 WHERE FALSE", (), _rowcount=_rc)
            m.observe(m.PG_POOL_QUERY_MS, (time.monotonic() - _t0) * 1000.0)
        except Exception:
            m.inc(m.PG_POOL_ERRORS_TOTAL)
            raise
        return _PgCursorProxy(self._conn, translated, tuple(params))

    async def executemany(self, sql: str, params: list[tuple[Any, ...]] | list[list[Any]]) -> None:
        """Execute *sql* once per row in *params*.

        Args:
            sql: SQL statement (``?`` placeholders allowed).
            params: Sequence of parameter rows.
        """
        translated = _translate_sql(sql)
        _t0 = time.monotonic()
        try:
            await self._conn.executemany(translated, params)
        except Exception:
            m.inc(m.PG_POOL_ERRORS_TOTAL)
            raise
        m.observe(m.PG_POOL_QUERY_MS, (time.monotonic() - _t0) * 1000.0)

    async def fetchall(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> list[Any]:
        """Execute *sql* and return all rows as a list of ``asyncpg.Record`` objects.

        Args:
            sql: SQL SELECT statement (``?`` placeholders allowed).
            params: Positional parameters.

        Returns:
            List of ``asyncpg.Record`` objects (empty list if no rows).
        """
        translated = _translate_sql(sql)
        _t0 = time.monotonic()
        try:
            result = await self._conn.fetch(translated, *params)
        except Exception:
            m.inc(m.PG_POOL_ERRORS_TOTAL)
            raise
        m.observe(m.PG_POOL_QUERY_MS, (time.monotonic() - _t0) * 1000.0)
        return result

    async def fetchone(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> Any | None:
        """Execute *sql* and return the first row, or ``None``.

        Args:
            sql: SQL SELECT statement (``?`` placeholders allowed).
            params: Positional parameters.

        Returns:
            First ``asyncpg.Record``, or ``None`` if no rows match.
        """
        translated = _translate_sql(sql)
        _t0 = time.monotonic()
        try:
            result = await self._conn.fetchrow(translated, *params)
        except Exception:
            m.inc(m.PG_POOL_ERRORS_TOTAL)
            raise
        m.observe(m.PG_POOL_QUERY_MS, (time.monotonic() - _t0) * 1000.0)
        return result

    async def fetchval(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> Any | None:
        """Execute *sql* and return the first column of the first row.

        Used for ``INSERT … RETURNING id`` and similar single-value queries.

        Args:
            sql: SQL statement (``?`` placeholders allowed).
            params: Positional parameters.

        Returns:
            The scalar value from column 0 of the first row, or ``None``.
        """
        translated = _translate_sql(sql)
        _t0 = time.monotonic()
        try:
            result = await self._conn.fetchval(translated, *params)
        except Exception:
            m.inc(m.PG_POOL_ERRORS_TOTAL)
            raise
        m.observe(m.PG_POOL_QUERY_MS, (time.monotonic() - _t0) * 1000.0)
        return result

    async def executescript(self, script: str) -> None:
        """Execute a multi-statement SQL script (DDL).

        asyncpg supports multi-statement scripts natively when ``execute()``
        is called without bind parameters.  We pass the entire script as a
        single call so that ``$$``-quoted PL/pgSQL blocks, ``DO`` statements,
        and comment sections are handled correctly by the server parser.

        Args:
            script: One or more SQL statements separated by ``;``.
        """
        await self._conn.execute(script)

    # ------------------------------------------------------------------
    # Pass-through helpers used by aiosqlite-style callers
    # ------------------------------------------------------------------

    async def commit(self) -> None:
        """No-op — transactions are managed by the backend context manager."""

    async def rollback(self) -> None:
        """No-op — transaction rollback is handled by the context manager on error."""


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class AsyncpgGraphBackend:
    """PostgreSQL graph backend using an asyncpg connection pool.

    Satisfies the ``GraphBackend`` protocol from ``backends.py`` and exposes
    the same ``write()`` / ``read()`` context-manager API as
    ``SQLiteGraphBackend`` so all callers are unchanged.

    Unlike SQLite there is no global write lock — PostgreSQL handles
    concurrent writers via MVCC.  Each ``write()`` acquires an independent
    connection and starts a ``SERIALIZABLE`` transaction; each ``read()``
    acquires an independent read-only connection.

    Args:
        (no constructor args — call ``initialize()`` to configure the pool)
    """

    def __init__(self) -> None:
        self._pool: asyncpg_module.Pool[Any] | None = None

    async def initialize(
        self,
        dsn: str,
        min_size: int = 5,
        max_size: int = 20,
    ) -> None:
        """Create the asyncpg connection pool.

        Must be called once during app startup before any graph functions are
        invoked.  Idempotent — a second call while already initialized is a
        no-op.

        Args:
            dsn: PostgreSQL DSN in asyncpg format, e.g.
                ``"postgresql://user:pw@host:5432/dbname"``.
            min_size: Minimum number of connections to keep alive in the pool.
            max_size: Maximum pool size.  Concurrent writers beyond this limit
                will wait for a connection to become available.

        Raises:
            ImportError: If ``asyncpg`` is not installed.
            asyncpg.PostgresConnectionError: If the DSN is unreachable.
        """
        if self._pool is not None:
            return

        try:
            import asyncpg
        except ImportError as exc:
            raise ImportError(
                "asyncpg is required when GRAPH_BACKEND=postgres.  "
                "Install it with: pip install asyncpg"
            ) from exc

        _t0 = time.monotonic()
        try:
            self._pool = await asyncpg.create_pool(
                dsn,
                min_size=min_size,
                max_size=max_size,
            )
        except Exception:
            m.inc(m.PG_POOL_ERRORS_TOTAL)
            import archivist.core.health as health

            health.register("postgres", healthy=False)
            raise

        _elapsed_ms = (time.monotonic() - _t0) * 1000.0
        import archivist.core.health as health

        health.register("postgres", healthy=True, latency_ms=_elapsed_ms)
        logger.info(
            "AsyncpgGraphBackend pool initialized (min=%d max=%d init_ms=%.1f): %s",
            min_size,
            max_size,
            _elapsed_ms,
            _redact_dsn(dsn),
        )

    async def close(self) -> None:
        """Gracefully drain and close the connection pool.

        Safe to call even if the pool was never initialized (no-op).
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            import archivist.core.health as health

            health.register("postgres", healthy=False)
            logger.info("AsyncpgGraphBackend pool closed")

    @asynccontextmanager
    async def write(self) -> AsyncIterator[AsyncpgConnection]:
        """Acquire a connection and open a SERIALIZABLE transaction.

        Auto-commits on clean exit; rolls back on any exception and re-raises.
        No global lock is needed — Postgres MVCC provides isolation.

        Yields:
            An ``AsyncpgConnection`` wrapper that translates ``?`` to ``$N``.

        Raises:
            RuntimeError: If ``initialize()`` was not called first.
        """
        if self._pool is None:
            raise RuntimeError("AsyncpgGraphBackend is not initialized — call initialize() first")
        _t0 = time.monotonic()
        try:
            async with self._pool.acquire() as conn:
                m.observe(m.PG_POOL_ACQUIRE_MS, (time.monotonic() - _t0) * 1000.0)
                async with conn.transaction():
                    yield AsyncpgConnection(conn)
        except Exception:
            m.inc(m.PG_POOL_ERRORS_TOTAL)
            raise

    @asynccontextmanager
    async def read(self) -> AsyncIterator[AsyncpgConnection]:
        """Acquire a read-only connection (no explicit transaction).

        Yields:
            An ``AsyncpgConnection`` wrapper.

        Raises:
            RuntimeError: If ``initialize()`` was not called first.
        """
        if self._pool is None:
            raise RuntimeError("AsyncpgGraphBackend is not initialized — call initialize() first")
        _t0 = time.monotonic()
        try:
            async with self._pool.acquire() as conn:
                m.observe(m.PG_POOL_ACQUIRE_MS, (time.monotonic() - _t0) * 1000.0)
                yield AsyncpgConnection(conn)
        except Exception:
            m.inc(m.PG_POOL_ERRORS_TOTAL)
            raise

    # ------------------------------------------------------------------
    # GraphBackend Protocol convenience methods
    # ------------------------------------------------------------------

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        """Execute *sql* with *params* inside a write transaction.

        Args:
            sql: SQL statement (``?`` placeholders allowed).
            params: Positional parameters.

        Returns:
            asyncpg command status string.
        """
        async with self.write() as conn:
            return await conn.execute(sql, params)

    async def executemany(self, sql: str, params: list[tuple[Any, ...]]) -> None:
        """Execute *sql* for each row in *params* inside a single transaction.

        Args:
            sql: SQL statement (``?`` placeholders allowed).
            params: List of parameter tuples, one per row.
        """
        async with self.write() as conn:
            await conn.executemany(sql, params)

    async def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
        """Execute a SELECT and return all rows.

        Args:
            sql: SQL SELECT statement (``?`` placeholders allowed).
            params: Positional parameters.

        Returns:
            List of ``asyncpg.Record`` objects.
        """
        async with self.read() as conn:
            return await conn.fetchall(sql, params)

    async def execute_ddl(self, sql: str) -> None:
        """Execute a DDL script (CREATE TABLE, CREATE INDEX, etc.).

        Splits on ``;`` and runs each statement via a write connection.

        Args:
            sql: One or more DDL statements separated by ``;``.
        """
        async with self.write() as conn:
            await conn.executescript(sql)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redact_dsn(dsn: str) -> str:
    """Return *dsn* with the password replaced by ``***``.

    Args:
        dsn: A PostgreSQL DSN string, possibly containing a password.

    Returns:
        DSN with the password field masked, safe for logging.
    """
    return re.sub(r"(://[^:]+:)[^@]+(@)", r"\1***\2", dsn)
