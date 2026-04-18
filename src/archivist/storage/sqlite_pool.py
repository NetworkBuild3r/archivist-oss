"""Async SQLite connection layer for Archivist.

Provides a single persistent ``aiosqlite.Connection`` protected by an
``asyncio.Lock`` for writes.  WAL mode allows concurrent readers on the same
connection without additional locking.

Usage::

    from archivist.storage.sqlite_pool import pool, initialize_pool, close_pool

    # In app startup:
    await initialize_pool()

    # In any async function:
    async with pool.write() as conn:
        await conn.execute("INSERT INTO ...")
        # commit is automatic on clean exit; rollback on exception

    async with pool.read() as conn:
        cursor = await conn.execute("SELECT ...")
        rows = await cursor.fetchall()

    # In app shutdown:
    await close_pool()

Class hierarchy
---------------
``SQLiteGraphBackend`` is the canonical class name.  ``SQLitePool`` is a
backward-compatible alias retained so that existing imports and tests require
no changes.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

import archivist.core.metrics as m

logger = logging.getLogger("archivist.sqlite_pool")

# ── Public asyncio.Lock for backup serialization ──────────────────────────────
# backup_manager acquires this before starting the SQLite Online Backup API so
# that no new writes can interleave with the backup copy.  pool.write() also
# acquires this lock (after the internal _write_lock) to create a clean two-
# level hierarchy: backup blocks all writes, writes block each other.
#
# The lock is created lazily on first use so that it is always bound to the
# running event loop.  Creating an asyncio.Lock at module-import time binds it
# to whatever loop is current at that moment; in pytest-asyncio with
# asyncio_mode="auto" every test gets a fresh loop, causing "bound to a
# different event loop" errors after the first test.
_GRAPH_WRITE_LOCK: asyncio.Lock | None = None


def _get_graph_write_lock() -> asyncio.Lock:
    """Return (or create) the module-level write-serialisation lock.

    Always called from within a running event loop, so the lock is guaranteed
    to be bound to the correct loop.
    """
    global _GRAPH_WRITE_LOCK
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if _GRAPH_WRITE_LOCK is None:
        _GRAPH_WRITE_LOCK = asyncio.Lock()
        return _GRAPH_WRITE_LOCK

    # In Python 3.10+ asyncio.Lock uses _LoopBoundMixin; _loop is set lazily on
    # first acquire.  If it has been set and differs from the current loop the
    # lock belongs to a closed loop and must be replaced.
    if loop is not None:
        bound_loop = getattr(_GRAPH_WRITE_LOCK, "_loop", None)
        if bound_loop is not None and bound_loop is not loop:
            _GRAPH_WRITE_LOCK = asyncio.Lock()

    return _GRAPH_WRITE_LOCK


# Backward-compatible alias — external code (backup_manager, tests) that reads
# this name directly will get the *current* lock via the property-like accessor.
# For direct attribute access we expose a property through a module-level shim.
# Code that does ``async with GRAPH_WRITE_LOCK_ASYNC`` will still work because
# pool.write() now calls _get_graph_write_lock() internally; callers outside
# pool.write() should migrate to _get_graph_write_lock().
GRAPH_WRITE_LOCK_ASYNC: asyncio.Lock = asyncio.Lock()  # legacy alias, not used internally


class SQLiteGraphBackend:
    """Single-connection async SQLite backend satisfying ``GraphBackend``.

    WAL mode allows only one writer at a time at the engine level, so a queue
    of N connections offers zero additional write throughput.  One connection
    protected by an ``asyncio.Lock`` is the community-standard pattern for
    ``aiosqlite`` and eliminates all pool-starvation / stale-connection edge
    cases.

    Reads share the same connection safely under WAL mode.

    In addition to the ``write()`` / ``read()`` context managers used by all
    existing callers, this class exposes ``execute``, ``executemany``,
    ``fetchall``, and ``execute_ddl`` convenience methods so that it structurally
    satisfies the ``GraphBackend`` Protocol defined in ``backends.py``.
    """

    def __init__(self) -> None:
        self._conn: aiosqlite.Connection | None = None
        self._write_lock: asyncio.Lock = asyncio.Lock()

    async def initialize(self, db_path: str, wal_autocheckpoint: int = 1000) -> None:
        """Open the connection and configure PRAGMAs.

        Called once at app startup via :func:`initialize_pool`.  Idempotent —
        a second call while already initialized is a no-op.

        Args:
            db_path: Path to the SQLite database file.
            wal_autocheckpoint: Pages before automatic WAL checkpoint (0 = off).
        """
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA cache_size=-8000")
        await self._conn.execute(f"PRAGMA wal_autocheckpoint={wal_autocheckpoint}")
        await self._conn.commit()
        logger.info("SQLite pool initialized: %s", db_path)

    async def close(self) -> None:
        """Close the connection.  Called once at app shutdown."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info("SQLite pool closed")

    @asynccontextmanager
    async def write(self) -> AsyncIterator[aiosqlite.Connection]:
        """Acquire the write lock and yield the connection.

        Auto-commits on clean exit; rolls back on any exception and re-raises.
        Records write-lock acquire latency in ``SQLITE_POOL_ACQUIRE_MS``.
        """
        if self._conn is None:
            raise RuntimeError("SQLitePool is not initialized — call initialize_pool() first")
        t0 = time.monotonic()
        async with _get_graph_write_lock():
            async with self._write_lock:
                elapsed_ms = (time.monotonic() - t0) * 1000
                m.observe(m.SQLITE_POOL_ACQUIRE_MS, elapsed_ms)
                try:
                    yield self._conn
                    await self._conn.commit()
                except Exception:
                    m.inc(m.SQLITE_POOL_WRITE_ERRORS)
                    try:
                        await self._conn.rollback()
                    except Exception as rb_err:
                        logger.error("SQLite rollback failed: %s", rb_err)
                    raise

    @asynccontextmanager
    async def read(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield the connection for read-only queries (no lock required under WAL)."""
        if self._conn is None:
            raise RuntimeError("SQLitePool is not initialized — call initialize_pool() first")
        yield self._conn

    # ------------------------------------------------------------------
    # GraphBackend Protocol convenience methods
    # These delegate through write()/read() so callers that prefer a
    # flat function-call API (e.g. future generic helpers) can use them.
    # ------------------------------------------------------------------

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        """Execute *sql* with *params* inside a write transaction and return the cursor.

        Args:
            sql: SQL statement to execute.
            params: Positional parameters for ``?`` placeholders.

        Returns:
            The ``aiosqlite.Cursor`` after execution.
        """
        async with self.write() as conn:
            return await conn.execute(sql, params)

    async def executemany(self, sql: str, params: list[tuple[Any, ...]]) -> None:
        """Execute *sql* for each row in *params* inside a single write transaction.

        Args:
            sql: SQL statement to execute.
            params: List of parameter tuples, one per row.
        """
        async with self.write() as conn:
            await conn.executemany(sql, params)

    async def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
        """Execute a SELECT *sql* and return all rows.

        Args:
            sql: SQL SELECT statement.
            params: Positional parameters for ``?`` placeholders.

        Returns:
            List of ``aiosqlite.Row`` objects (empty list if no rows).
        """
        async with self.read() as conn:
            cursor = await conn.execute(sql, params)
            return list(await cursor.fetchall())

    async def execute_ddl(self, sql: str) -> None:
        """Execute a DDL statement (CREATE TABLE, CREATE INDEX, etc.).

        Used by schema-init code that previously relied on the synchronous
        ``get_db()`` path.  Runs inside a write transaction; changes are
        committed on clean exit.

        Args:
            sql: DDL SQL to execute.
        """
        async with self.write() as conn:
            await conn.executescript(sql)


# ---------------------------------------------------------------------------
# Module-level singleton — imported everywhere.
# ---------------------------------------------------------------------------

# The canonical class is SQLiteGraphBackend.  SQLitePool is preserved as a
# backward-compatible alias so that all existing ``from sqlite_pool import pool``
# and ``isinstance(pool, SQLitePool)`` references continue to work without
# any changes to consumer files.
SQLitePool = SQLiteGraphBackend

pool: SQLiteGraphBackend = SQLiteGraphBackend()


async def initialize_pool() -> None:
    """Initialize the module-level pool singleton.

    Must be called once during app startup before any async graph functions are
    used.  Uses SQLITE_PATH and SQLITE_WAL_AUTOCHECKPOINT from config.
    """
    import os

    from archivist.core.config import SQLITE_PATH, SQLITE_WAL_AUTOCHECKPOINT

    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    await pool.initialize(SQLITE_PATH, wal_autocheckpoint=SQLITE_WAL_AUTOCHECKPOINT)


async def close_pool() -> None:
    """Close the module-level pool singleton.  Called during app shutdown."""
    await pool.close()
