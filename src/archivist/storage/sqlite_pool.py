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
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite

import archivist.core.metrics as m

logger = logging.getLogger("archivist.sqlite_pool")

# ── Public asyncio.Lock for backup serialization ──────────────────────────────
# backup_manager acquires this before starting the SQLite Online Backup API so
# that no new writes can interleave with the backup copy.  pool.write() also
# acquires this lock (after the internal _write_lock) to create a clean two-
# level hierarchy: backup blocks all writes, writes block each other.
GRAPH_WRITE_LOCK_ASYNC: asyncio.Lock = asyncio.Lock()


class SQLitePool:
    """Single-connection async SQLite pool.

    WAL mode allows only one writer at a time at the engine level, so a queue
    of N connections offers zero additional write throughput.  One connection
    protected by an ``asyncio.Lock`` is the community-standard pattern for
    ``aiosqlite`` and eliminates all pool-starvation / stale-connection edge
    cases.

    Reads share the same connection safely under WAL mode.
    """

    def __init__(self) -> None:
        self._conn: aiosqlite.Connection | None = None
        self._write_lock: asyncio.Lock = asyncio.Lock()

    async def initialize(self, db_path: str, wal_autocheckpoint: int = 1000) -> None:
        """Open the connection and configure PRAGMAs.

        Called once at app startup via :func:`initialize_pool`.  Idempotent —
        a second call while already initialized is a no-op.
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
        async with GRAPH_WRITE_LOCK_ASYNC:
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


# Module-level singleton — imported everywhere.
pool: SQLitePool = SQLitePool()


async def initialize_pool() -> None:
    """Initialize the module-level pool singleton.

    Must be called once during app startup before any async graph functions are
    used.  Uses SQLITE_PATH and SQLITE_WAL_AUTOCHECKPOINT from config.
    """
    from archivist.core.config import SQLITE_PATH, SQLITE_WAL_AUTOCHECKPOINT

    import os

    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    await pool.initialize(SQLITE_PATH, wal_autocheckpoint=SQLITE_WAL_AUTOCHECKPOINT)


async def close_pool() -> None:
    """Close the module-level pool singleton.  Called during app shutdown."""
    await pool.close()
