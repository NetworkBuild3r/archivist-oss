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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg as asyncpg_module

logger = logging.getLogger("archivist.asyncpg_backend")

# ---------------------------------------------------------------------------
# SQL placeholder translation
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\?")


def _translate_sql(sql: str) -> str:
    """Convert SQLite ``?`` placeholders to asyncpg ``$N`` style.

    Examples::

        >>> _translate_sql("SELECT * FROM t WHERE a = ? AND b = ?")
        'SELECT * FROM t WHERE a = $1 AND b = $2'
        >>> _translate_sql("INSERT INTO t VALUES (?, ?)")
        'INSERT INTO t VALUES ($1, $2)'
        >>> _translate_sql("SELECT 1")  # no placeholders — unchanged
        'SELECT 1'

    Args:
        sql: SQL string using ``?`` placeholders.

    Returns:
        SQL string with ``?`` replaced by ``$1``, ``$2``, … in order.
    """
    counter = 0

    def _replace(_m: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        return f"${counter}"

    return _PLACEHOLDER_RE.sub(_replace, sql)


# ---------------------------------------------------------------------------
# Connection wrapper
# ---------------------------------------------------------------------------


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
        """Execute *sql* with *params* and return the status string.

        Args:
            sql: SQL statement (``?`` placeholders allowed).
            params: Positional parameters.

        Returns:
            The asyncpg command status string (e.g. ``"INSERT 0 1"``).
        """
        translated = _translate_sql(sql)
        return await self._conn.execute(translated, *params)

    async def executemany(
        self, sql: str, params: list[tuple[Any, ...]] | list[list[Any]]
    ) -> None:
        """Execute *sql* once per row in *params*.

        Args:
            sql: SQL statement (``?`` placeholders allowed).
            params: Sequence of parameter rows.
        """
        translated = _translate_sql(sql)
        await self._conn.executemany(translated, params)

    async def fetchall(
        self, sql: str, params: tuple[Any, ...] | list[Any] = ()
    ) -> list[Any]:
        """Execute *sql* and return all rows as a list of ``asyncpg.Record`` objects.

        Args:
            sql: SQL SELECT statement (``?`` placeholders allowed).
            params: Positional parameters.

        Returns:
            List of ``asyncpg.Record`` objects (empty list if no rows).
        """
        translated = _translate_sql(sql)
        return await self._conn.fetch(translated, *params)

    async def fetchone(
        self, sql: str, params: tuple[Any, ...] | list[Any] = ()
    ) -> Any | None:
        """Execute *sql* and return the first row, or ``None``.

        Args:
            sql: SQL SELECT statement (``?`` placeholders allowed).
            params: Positional parameters.

        Returns:
            First ``asyncpg.Record``, or ``None`` if no rows match.
        """
        translated = _translate_sql(sql)
        return await self._conn.fetchrow(translated, *params)

    async def executescript(self, script: str) -> None:
        """Execute a multi-statement SQL script (DDL).

        asyncpg does not have a native ``executescript``.  We split on ``;``
        and execute each non-empty statement individually.

        Args:
            script: One or more SQL statements separated by ``;``.
        """
        statements = [s.strip() for s in script.split(";") if s.strip()]
        for stmt in statements:
            await self._conn.execute(stmt)

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

        self._pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
        )
        logger.info(
            "AsyncpgGraphBackend pool initialized (min=%d max=%d): %s",
            min_size,
            max_size,
            _redact_dsn(dsn),
        )

    async def close(self) -> None:
        """Gracefully drain and close the connection pool.

        Safe to call even if the pool was never initialized (no-op).
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
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
            raise RuntimeError(
                "AsyncpgGraphBackend is not initialized — call initialize() first"
            )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield AsyncpgConnection(conn)

    @asynccontextmanager
    async def read(self) -> AsyncIterator[AsyncpgConnection]:
        """Acquire a read-only connection (no explicit transaction).

        Yields:
            An ``AsyncpgConnection`` wrapper.

        Raises:
            RuntimeError: If ``initialize()`` was not called first.
        """
        if self._pool is None:
            raise RuntimeError(
                "AsyncpgGraphBackend is not initialized — call initialize() first"
            )
        async with self._pool.acquire() as conn:
            yield AsyncpgConnection(conn)

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
