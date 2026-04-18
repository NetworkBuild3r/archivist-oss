"""Factory for creating the active GraphBackend based on configuration.

This module is the single decision point for which storage backend Archivist
uses.  It reads ``GRAPH_BACKEND`` from ``ArchivistSettings`` and returns the
appropriate backend instance.

Usage (app startup)::

    from archivist.storage.backend_factory import create_graph_backend
    import archivist.storage.sqlite_pool as _pool_module

    backend = await create_graph_backend()
    _pool_module.pool = backend          # replace the module singleton
    await _pool_module.initialize_pool() # no-op for postgres (already initialized)

The module-level ``pool`` singleton in ``sqlite_pool`` continues to work
unchanged for all consumers — they never need to know which backend is active.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archivist.storage.asyncpg_backend import AsyncpgGraphBackend
    from archivist.storage.sqlite_pool import SQLiteGraphBackend

logger = logging.getLogger("archivist.backend_factory")


async def create_graph_backend() -> SQLiteGraphBackend | AsyncpgGraphBackend:
    """Instantiate and initialise the graph backend specified in config.

    Reads ``GRAPH_BACKEND`` (default ``"sqlite"``) from
    ``archivist.core.config``.  For ``"postgres"`` it also reads
    ``DATABASE_URL``, ``PG_POOL_MIN``, and ``PG_POOL_MAX``.

    Returns:
        An initialized ``SQLiteGraphBackend`` (default) or
        ``AsyncpgGraphBackend`` (when ``GRAPH_BACKEND=postgres``).

    Raises:
        ValueError: If ``GRAPH_BACKEND`` is an unrecognised value.
        ValueError: If ``GRAPH_BACKEND=postgres`` but ``DATABASE_URL`` is empty.
        ImportError: If ``asyncpg`` is not installed and ``GRAPH_BACKEND=postgres``.
        RuntimeError: If the backend fails to connect during initialisation.
    """
    from archivist.core.config import (
        DATABASE_URL,
        GRAPH_BACKEND,
        PG_POOL_MAX,
        PG_POOL_MIN,
        SQLITE_PATH,
        SQLITE_WAL_AUTOCHECKPOINT,
    )

    backend_name = (GRAPH_BACKEND or "sqlite").strip().lower()

    if backend_name == "sqlite":
        return await _create_sqlite_backend(SQLITE_PATH, SQLITE_WAL_AUTOCHECKPOINT)

    if backend_name == "postgres":
        return await _create_postgres_backend(DATABASE_URL, PG_POOL_MIN, PG_POOL_MAX)

    raise ValueError(
        f"Unknown GRAPH_BACKEND value: {GRAPH_BACKEND!r}. "
        "Supported values are 'sqlite' (default) and 'postgres'."
    )


async def _create_sqlite_backend(
    sqlite_path: str,
    wal_autocheckpoint: int,
) -> SQLiteGraphBackend:
    """Create and initialise the SQLite backend.

    Args:
        sqlite_path: Filesystem path to the SQLite database file.
        wal_autocheckpoint: WAL auto-checkpoint page threshold.

    Returns:
        An initialized ``SQLiteGraphBackend``.
    """
    import os

    from archivist.storage.sqlite_pool import SQLiteGraphBackend

    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    backend = SQLiteGraphBackend()
    await backend.initialize(sqlite_path, wal_autocheckpoint=wal_autocheckpoint)
    logger.info("graph backend: SQLite (%s)", sqlite_path)
    return backend


async def _create_postgres_backend(
    database_url: str,
    pg_pool_min: int,
    pg_pool_max: int,
) -> AsyncpgGraphBackend:
    """Create and initialise the PostgreSQL backend.

    Args:
        database_url: asyncpg-format DSN, e.g.
            ``"postgresql://user:pw@host:5432/dbname"``.
        pg_pool_min: Minimum pool size.
        pg_pool_max: Maximum pool size.

    Returns:
        An initialized ``AsyncpgGraphBackend``.

    Raises:
        ValueError: If *database_url* is empty.
        ImportError: If ``asyncpg`` is not installed.
    """
    from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

    if not database_url:
        raise ValueError(
            "DATABASE_URL must be set when GRAPH_BACKEND=postgres. "
            "Example: DATABASE_URL=postgresql://user:pw@host:5432/archivist"
        )

    backend = AsyncpgGraphBackend()
    await backend.initialize(database_url, min_size=pg_pool_min, max_size=pg_pool_max)
    logger.info("graph backend: PostgreSQL (pool min=%d max=%d)", pg_pool_min, pg_pool_max)
    return backend
