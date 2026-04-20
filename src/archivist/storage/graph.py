"""SQLite-backed temporal knowledge graph for entity/relationship tracking."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

import archivist.core.metrics as m
from archivist.core.config import SQLITE_PATH
from archivist.utils.chunking import NEEDLE_PATTERNS

logger = logging.getLogger("archivist.graph")

# ---------------------------------------------------------------------------
# Backward-compatibility shim: keep the old threading.Lock name so any
# remaining direct imports in backup_manager or test code don't break.
# All internal code now uses the pool pattern.
# ---------------------------------------------------------------------------
GRAPH_WRITE_LOCK = threading.Lock()


def _ensure_dir():
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)


def get_db() -> sqlite3.Connection:
    """Return a new synchronous sqlite3 connection.

    Kept for backup_manager.py (which uses the SQLite Online Backup API that
    requires a synchronous connection) and for test fixtures that work with
    in-memory databases.  All normal application code should use
    ``archivist.storage.sqlite_pool.pool`` instead.

    Deprecation
    -----------
    This function will be removed in the follow-up PR that migrates all callers
    to ``await pool.read()`` / ``await pool.write()``.  When
    ``GRAPH_BACKEND=postgres`` is set this function logs a ``WARNING`` and
    returns a direct synchronous connection to the SQLite path for schema init
    only — callers that perform real data reads/writes against PostgreSQL must
    use the async pool instead.
    """
    from archivist.core.config import GRAPH_BACKEND

    if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
        logging.getLogger("archivist.graph").warning(
            "get_db() is not supported with GRAPH_BACKEND=postgres. "
            "Returning a temporary SQLite connection for schema init only. "
            "Migrate all callers to 'async with pool.read()' or 'async with pool.write()'. "
            "get_db() will be removed in a future release."
        )
    _ensure_dir()
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_conn():
    """Context manager that yields an open SQLite connection and closes it on exit.

    Usage::

        with db_conn() as conn:
            conn.execute(...)
    """
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


def schema_guard(ddl: str):
    """Return a zero-argument callable that runs *ddl* exactly once.

    Uses double-checked locking to avoid the TOCTOU race where two coroutines
    both see ``applied=False`` and both attempt to execute the DDL.

    Usage (module level)::

        _ensure_schema = schema_guard(\"\"\"
            CREATE TABLE IF NOT EXISTS my_table (...);
        \"\"\")

    Then call ``_ensure_schema()`` at the top of each public function that
    needs the schema to be initialised.  Call ``_ensure_schema.reset()`` in
    test fixtures to force re-initialisation against a fresh database.
    """
    _lock = threading.Lock()

    def _ensure():
        if _ensure.applied:
            return
        with _lock:
            # Double-checked: another thread may have run DDL while we waited.
            if _ensure.applied:
                return
            conn = get_db()
            try:
                conn.executescript(ddl)
                conn.commit()
            finally:
                conn.close()
            _ensure.applied = True

    def _reset():
        _ensure.applied = False

    _ensure.applied = False
    _ensure.reset = _reset
    return _ensure


def init_schema():
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL DEFAULT 'unknown',
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            mention_count INTEGER NOT NULL DEFAULT 1,
            metadata TEXT DEFAULT '{}',
            retention_class TEXT NOT NULL DEFAULT 'standard',
            aliases TEXT NOT NULL DEFAULT '[]',
            namespace TEXT NOT NULL DEFAULT 'global',
            actor_id TEXT NOT NULL DEFAULT '',
            actor_type TEXT NOT NULL DEFAULT '',
            UNIQUE(name, namespace)
        );
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
        CREATE INDEX IF NOT EXISTS idx_entities_namespace ON entities(namespace);

        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entity_id INTEGER NOT NULL REFERENCES entities(id),
            target_entity_id INTEGER NOT NULL REFERENCES entities(id),
            relation_type TEXT NOT NULL,
            evidence TEXT NOT NULL,
            agent_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            provenance TEXT NOT NULL DEFAULT 'unknown',
            UNIQUE(source_entity_id, target_entity_id, relation_type)
        );
        CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);
        CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER REFERENCES entities(id),
            fact_text TEXT NOT NULL,
            source_file TEXT,
            agent_id TEXT,
            created_at TEXT NOT NULL,
            superseded_by INTEGER REFERENCES facts(id),
            is_active INTEGER NOT NULL DEFAULT 1,
            retention_class TEXT NOT NULL DEFAULT 'standard',
            valid_from TEXT NOT NULL DEFAULT '',
            valid_until TEXT NOT NULL DEFAULT '',
            memory_id TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity_id);
        CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(is_active);
        CREATE INDEX IF NOT EXISTS idx_facts_valid_from ON facts(valid_from);
        CREATE INDEX IF NOT EXISTS idx_facts_memory_id ON facts(memory_id);

        CREATE TABLE IF NOT EXISTS curator_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            agent_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            operation TEXT NOT NULL,
            parent_versions TEXT DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_memver_memory ON memory_versions(memory_id);
        CREATE INDEX IF NOT EXISTS idx_memver_agent ON memory_versions(agent_id);

        -- BM25 / FTS5 hybrid search tables (v1.2)
        CREATE TABLE IF NOT EXISTS memory_chunks (
            rowid INTEGER PRIMARY KEY,
            qdrant_id TEXT NOT NULL UNIQUE,
            text TEXT NOT NULL,
            file_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            agent_id TEXT NOT NULL DEFAULT '',
            namespace TEXT NOT NULL DEFAULT '',
            date TEXT NOT NULL DEFAULT '',
            memory_type TEXT NOT NULL DEFAULT 'general',
            is_excluded INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_mc_qdrant ON memory_chunks(qdrant_id);
        CREATE INDEX IF NOT EXISTS idx_mc_namespace ON memory_chunks(namespace);
        CREATE INDEX IF NOT EXISTS idx_mc_agent ON memory_chunks(agent_id);

        -- Tracks all Qdrant point IDs created for each memory (Phase 2).
        CREATE TABLE IF NOT EXISTS memory_points (
            memory_id   TEXT NOT NULL,
            qdrant_id   TEXT NOT NULL,
            point_type  TEXT NOT NULL DEFAULT 'primary',
            created_at  TEXT NOT NULL,
            PRIMARY KEY (memory_id, qdrant_id)
        );
        CREATE INDEX IF NOT EXISTS idx_mp_memory ON memory_points(memory_id);
        CREATE INDEX IF NOT EXISTS idx_mp_qdrant ON memory_points(qdrant_id);

        -- Dead-letter queue for failed Qdrant deletes (Phase 2).
        CREATE TABLE IF NOT EXISTS delete_failures (
            id          TEXT PRIMARY KEY,
            memory_id   TEXT NOT NULL,
            qdrant_ids  TEXT NOT NULL,
            error       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            resolved_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_df_memory ON delete_failures(memory_id);
        CREATE INDEX IF NOT EXISTS idx_df_created ON delete_failures(created_at);

        -- Transactional outbox for cross-store writes (Phase 3).
        -- Events are written atomically with SQLite artifacts and applied to
        -- Qdrant by the OutboxProcessor background task.
        CREATE TABLE IF NOT EXISTS outbox (
            id           TEXT PRIMARY KEY,
            event_type   TEXT NOT NULL,
            payload      TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            retry_count  INTEGER NOT NULL DEFAULT 0,
            last_attempt TEXT,
            created_at   TEXT NOT NULL,
            error        TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_outbox_event  ON outbox(event_type, status);
        -- Covering index for drain loop: WHERE status IN ('pending','processing')
        -- filters with last_attempt for backoff, ordered by created_at.
        CREATE INDEX IF NOT EXISTS idx_outbox_drain
            ON outbox(status, last_attempt, created_at)
            WHERE status IN ('pending', 'processing');
        -- Covering index for retention pruning: WHERE status='applied' AND last_attempt < cutoff.
        CREATE INDEX IF NOT EXISTS idx_outbox_prune
            ON outbox(status, last_attempt)
            WHERE status = 'applied';

        -- Needle registry for O(1) structured-token lookup (v2.0).
        -- Also initialised lazily by _ensure_needle_registry; including it here
        -- ensures the table exists before any MemoryTransaction acquires the
        -- pool write-lock (avoids a deadlock when the schema guard fires inside
        -- an open transaction).
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
        conn.commit()
        conn.close()
    # Mark the needle-registry schema guard as already applied so it never tries
    # to acquire a second sync connection while the async pool lock is held.
    _ensure_needle_registry.applied = True  # type: ignore[attr-defined]
    _migrate_schema()
    _migrate_entity_unique_constraint()
    _init_fts5()


def _migrate_schema():
    """Add columns introduced in v1.7+ if upgrading from an older database."""
    _logger = logging.getLogger("archivist.graph")
    migrations = [
        ("facts", "retention_class", "TEXT NOT NULL DEFAULT 'standard'"),
        ("entities", "retention_class", "TEXT NOT NULL DEFAULT 'standard'"),
        ("entities", "aliases", "TEXT NOT NULL DEFAULT '[]'"),
        ("facts", "valid_from", "TEXT NOT NULL DEFAULT ''"),
        ("facts", "valid_until", "TEXT NOT NULL DEFAULT ''"),
        ("relationships", "provenance", "TEXT NOT NULL DEFAULT 'unknown'"),
        ("entities", "namespace", "TEXT NOT NULL DEFAULT 'global'"),
        ("facts", "namespace", "TEXT NOT NULL DEFAULT 'global'"),
        ("relationships", "namespace", "TEXT NOT NULL DEFAULT 'global'"),
        ("facts", "memory_id", "TEXT NOT NULL DEFAULT ''"),
        ("memory_chunks", "is_excluded", "INTEGER NOT NULL DEFAULT 0"),
        # Phase 6: provenance & actor-aware memory
        ("facts", "confidence", "REAL NOT NULL DEFAULT 1.0"),
        ("facts", "provenance", "TEXT NOT NULL DEFAULT 'unknown'"),
        ("facts", "actor_id", "TEXT NOT NULL DEFAULT ''"),
        ("memory_chunks", "actor_id", "TEXT NOT NULL DEFAULT ''"),
        ("memory_chunks", "actor_type", "TEXT NOT NULL DEFAULT ''"),
        ("entities", "actor_id", "TEXT NOT NULL DEFAULT ''"),
        ("entities", "actor_type", "TEXT NOT NULL DEFAULT ''"),
    ]
    # needle_registry may not exist yet (schema_guard creates it lazily),
    # so these ALTER TABLEs are attempted but silently skipped on failure.
    _needle_migrations = [
        ("needle_registry", "actor_id", "TEXT NOT NULL DEFAULT ''"),
        ("needle_registry", "actor_type", "TEXT NOT NULL DEFAULT ''"),
    ]
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_facts_retention ON facts(retention_class)",
        "CREATE INDEX IF NOT EXISTS idx_entities_retention ON entities(retention_class)",
        "CREATE INDEX IF NOT EXISTS idx_facts_valid_from ON facts(valid_from)",
        "CREATE INDEX IF NOT EXISTS idx_entities_namespace ON entities(namespace)",
        "CREATE INDEX IF NOT EXISTS idx_facts_namespace ON facts(namespace)",
        "CREATE INDEX IF NOT EXISTS idx_relationships_namespace ON relationships(namespace)",
        "CREATE INDEX IF NOT EXISTS idx_facts_memory_id ON facts(memory_id)",
        "CREATE INDEX IF NOT EXISTS idx_mc_excluded ON memory_chunks(is_excluded)",
        # Phase 6: provenance indexes
        "CREATE INDEX IF NOT EXISTS idx_facts_actor ON facts(actor_id)",
        "CREATE INDEX IF NOT EXISTS idx_mc_actor ON memory_chunks(actor_id)",
        "CREATE INDEX IF NOT EXISTS idx_mc_actor_type ON memory_chunks(actor_type)",
        "CREATE INDEX IF NOT EXISTS idx_entities_actor ON entities(actor_id)",
    ]
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        for table, column, typedef in migrations:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {typedef}")
                conn.commit()
                _logger.info("Migrated %s: added %s column", table, column)
            except sqlite3.OperationalError:
                pass
        for table, column, typedef in _needle_migrations:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {typedef}")
                conn.commit()
                _logger.info("Migrated %s: added %s column", table, column)
            except (sqlite3.OperationalError, Exception):
                pass
        for ddl in indexes:
            try:
                conn.execute(ddl)
                conn.commit()
            except Exception:
                pass
        conn.close()


def _migrate_entity_unique_constraint():
    """Rebuild entities UNIQUE constraint to include namespace (idempotent).

    The original schema has UNIQUE(name) which collides across namespaces.
    This migration copies to a new table with UNIQUE(name, namespace),
    then swaps in place.  Safe to run multiple times.
    """
    _logger = logging.getLogger("archivist.graph")
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(entities)").fetchall()]
            if "namespace" not in cols:
                return

            idx_info = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='entities' AND sql IS NOT NULL"
            ).fetchall()
            has_ns_unique = any("namespace" in (r[0] or "") for r in idx_info)
            create_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='entities'"
            ).fetchone()
            if create_sql and "UNIQUE" in (create_sql[0] or ""):
                after_unique = create_sql[0].split("UNIQUE", 1)[-1]
                if "namespace" in after_unique:
                    return
            if has_ns_unique:
                return

            conn.execute("DROP TABLE IF EXISTS entities_new")
            conn.execute("""
                CREATE TABLE entities_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT 'unknown',
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT DEFAULT '{}',
                    retention_class TEXT NOT NULL DEFAULT 'standard',
                    aliases TEXT NOT NULL DEFAULT '[]',
                    namespace TEXT NOT NULL DEFAULT 'global',
                    UNIQUE(name, namespace)
                )
            """)
            conn.execute("""
                INSERT INTO entities_new (id, name, entity_type, first_seen, last_seen,
                    mention_count, metadata, retention_class, aliases, namespace)
                SELECT id, name, entity_type, first_seen, last_seen,
                    mention_count, metadata, retention_class, aliases, namespace
                FROM entities
            """)
            conn.execute("DROP TABLE entities")
            conn.execute("ALTER TABLE entities_new RENAME TO entities")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_namespace ON entities(namespace)")
            conn.commit()
            _logger.info("Migrated entities: rebuilt UNIQUE constraint to include namespace")
        except Exception as e:
            _logger.warning(
                "Entity UNIQUE constraint migration failed (may already be done): %s", e
            )
            conn.rollback()
        finally:
            conn.close()


def _init_fts5():
    """Create the FTS5 virtual tables if they don't already exist.

    Separated from init_schema() because FTS5 contentless-delete tables
    need a slightly different DDL path and tolerate 'already exists' gracefully.

    Creates two tables:
      - ``memory_fts``: Porter-stemmed for recall-oriented BM25 search.
      - ``memory_fts_exact``: Non-stemmed (unicode61 only) for exact token matching
        of identifiers, IPs, cron expressions, etc.

    On success we run a trivial read and register ``fts5`` as healthy;
    on failure we register unhealthy so downstream BM25 search can skip FTS.
    """
    import archivist.core.health as health

    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
                "USING fts5(text, content='memory_chunks', content_rowid='rowid', "
                "tokenize='porter unicode61')"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts_exact "
                "USING fts5(text, content='memory_chunks', content_rowid='rowid', "
                "tokenize='unicode61')"
            )
            conn.commit()
            conn.execute("SELECT count(*) FROM memory_fts LIMIT 1")
            conn.execute("SELECT count(*) FROM memory_fts_exact LIMIT 1")
            health.register("fts5", healthy=True)
        except Exception as e:
            health.register("fts5", healthy=False, detail=str(e))
        finally:
            conn.close()


async def init_schema_async() -> None:
    """Async schema initializer that dispatches by active backend.

    Called from ``app/main.py`` startup instead of (or after) the synchronous
    ``init_schema()`` so that Postgres backends can apply the full DDL from
    ``schema_postgres.sql`` without blocking the event loop.

    Behaviour by backend:

    - **SQLite**: delegates synchronously to :func:`init_schema` (unchanged).
    - **Postgres**: reads ``schema_postgres.sql`` from the package data directory
      and executes it via ``pool.execute_ddl()``.  The SQL file uses
      ``IF NOT EXISTS`` guards throughout, so it is safe to run on both fresh
      and pre-existing databases.  After DDL, registers ``fts5`` as healthy
      (the ``fts5`` health key is re-used for Postgres tsvector availability).
    """
    import archivist.core.health as health
    from archivist.core.config import GRAPH_BACKEND

    if (GRAPH_BACKEND or "sqlite").lower() != "postgres":
        init_schema()
        return

    from pathlib import Path

    from archivist.storage.sqlite_pool import pool

    schema_path = Path(__file__).with_name("schema_postgres.sql")
    if not schema_path.exists():
        logging.getLogger("archivist.graph").error(
            "Postgres schema file not found: %s", schema_path
        )
        health.register("fts5", healthy=False, detail="schema_postgres.sql missing")
        return

    ddl = schema_path.read_text()
    try:
        await pool.execute_ddl(ddl)
        health.register("fts5", healthy=True)
        logging.getLogger("archivist.graph").info(
            "Postgres schema applied from %s", schema_path.name
        )
    except Exception as exc:
        logging.getLogger("archivist.graph").error("Postgres schema init failed: %s", exc)
        health.register("fts5", healthy=False, detail=str(exc))
        raise


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
    conn: aiosqlite.Connection | None = None,
):
    """Insert or replace a chunk in memory_chunks and sync to both FTS indexes.

    On SQLite this also maintains the FTS5 shadow-row tables (``memory_fts``
    and ``memory_fts_exact``).  On Postgres the ``fts_vector`` /
    ``fts_vector_simple`` columns are ``GENERATED ALWAYS AS ... STORED``
    so they update automatically when the ``text`` column changes — no
    shadow-row maintenance is needed.

    Args:
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
                "memory_type, actor_id, actor_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (qdrant_id) DO UPDATE SET "
                "text=EXCLUDED.text, file_path=EXCLUDED.file_path, "
                "chunk_index=EXCLUDED.chunk_index, agent_id=EXCLUDED.agent_id, "
                "namespace=EXCLUDED.namespace, date=EXCLUDED.date, "
                "memory_type=EXCLUDED.memory_type, actor_id=EXCLUDED.actor_id, "
                "actor_type=EXCLUDED.actor_type",
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
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, agent_id, namespace, date, memory_type, actor_id, actor_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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


async def _search_fts_sqlite(
    query: str,
    namespace: str = "",
    agent_id: str = "",
    memory_type: str = "",
    limit: int = 30,
    actor_type: str = "",
) -> list[dict]:
    """SQLite FTS5 BM25 search implementation (stemmed / ``memory_fts``)."""
    from archivist.storage.sqlite_pool import pool

    _t0 = time.monotonic()
    try:
        async with pool.read() as conn:
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

            where_sql = " AND " + " AND ".join(where_clauses)

            sql = (
                "SELECT mc.qdrant_id, mc.file_path, mc.chunk_index, mc.agent_id, "
                "mc.namespace, mc.date, mc.memory_type, mc.text, "
                "mc.actor_id, mc.actor_type, "
                "rank AS bm25_rank "
                "FROM memory_fts "
                "JOIN memory_chunks mc ON memory_fts.rowid = mc.rowid "
                f"WHERE memory_fts MATCH ? {where_sql} "
                "ORDER BY rank "
                f"LIMIT ?"
            )
            params = [query] + params + [limit]

            cur = await conn.execute(sql, params)
            results = []
            for row in await cur.fetchall():
                r = dict(row)
                r["bm25_score"] = -r.pop("bm25_rank", 0)
                results.append(r)
        m.observe(
            m.FTS_SEARCH_DURATION_MS, (time.monotonic() - _t0) * 1000.0, {"backend": "sqlite"}
        )
        m.inc(m.FTS_SEARCH_TOTAL, {"backend": "sqlite"})
        return results
    except Exception as e:
        logging.getLogger("archivist.graph").warning("FTS search failed: %s", e)
        return []


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
    from archivist.storage.sqlite_pool import pool

    builder = {
        "or": _pg_tsquery_or,
        "and": _pg_tsquery_and,
        "phrase": _pg_tsquery_phrase,
    }.get(fts_mode, _pg_tsquery_or)

    tsquery_expr = builder(raw_query)
    if not tsquery_expr:
        return []

    _t0 = time.monotonic()
    try:
        async with pool.read() as conn:
            where_clauses = ["mc.is_excluded = 0"]
            params: list = [tsquery_expr]

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

            where_sql = " AND " + " AND ".join(where_clauses)

            # ts_rank_cd returns values in [0,1]; multiply by 32 to normalize
            # into the same ballpark as SQLite FTS5 BM25 scores (~0.5-30 range).
            sql = (
                "SELECT mc.qdrant_id, mc.file_path, mc.chunk_index, mc.agent_id, "
                "mc.namespace, mc.date, mc.memory_type, mc.text, "
                "mc.actor_id, mc.actor_type, "
                "ts_rank_cd(mc.fts_vector, to_tsquery('english', ?)) * 32 AS bm25_rank "
                "FROM memory_chunks mc "
                f"WHERE mc.fts_vector @@ to_tsquery('english', ?) {where_sql} "
                "ORDER BY bm25_rank DESC "
                "LIMIT ?"
            )
            # tsquery_expr used twice: once in SELECT ranking, once in WHERE
            params = [tsquery_expr, tsquery_expr] + params[1:] + [limit]

            cur = await conn.execute(sql, params)
            results = []
            for row in await cur.fetchall():
                r = dict(row)
                r["bm25_score"] = r.pop("bm25_rank", 0)
                results.append(r)
        m.observe(
            m.FTS_SEARCH_DURATION_MS, (time.monotonic() - _t0) * 1000.0, {"backend": "postgres"}
        )
        m.inc(m.FTS_SEARCH_TOTAL, {"backend": "postgres"})
        return results
    except Exception as e:
        logging.getLogger("archivist.graph").warning("FTS Postgres search failed: %s", e)
        return []


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
    from archivist.storage.sqlite_pool import pool

    _t0 = time.monotonic()
    try:
        async with pool.read() as conn:
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

            where_sql = " AND " + " AND ".join(where_clauses)

            sql = (
                "SELECT mc.qdrant_id, mc.file_path, mc.chunk_index, mc.agent_id, "
                "mc.namespace, mc.date, mc.memory_type, mc.text, "
                "mc.actor_id, mc.actor_type, "
                "rank AS bm25_rank "
                "FROM memory_fts_exact "
                "JOIN memory_chunks mc ON memory_fts_exact.rowid = mc.rowid "
                f"WHERE memory_fts_exact MATCH ? {where_sql} "
                "ORDER BY rank "
                f"LIMIT ?"
            )
            params = [query] + params + [limit]

            cur = await conn.execute(sql, params)
            results = []
            for row in await cur.fetchall():
                r = dict(row)
                r["bm25_score"] = -r.pop("bm25_rank", 0)
                results.append(r)
        m.observe(
            m.FTS_SEARCH_DURATION_MS, (time.monotonic() - _t0) * 1000.0, {"backend": "sqlite"}
        )
        m.inc(m.FTS_SEARCH_TOTAL, {"backend": "sqlite"})
        return results
    except Exception as e:
        logging.getLogger("archivist.graph").warning("FTS exact search failed: %s", e)
        return []


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
    from archivist.storage.sqlite_pool import pool

    tsquery_expr = _pg_tsquery_or(raw_query)
    if not tsquery_expr:
        return []

    _t0 = time.monotonic()
    try:
        async with pool.read() as conn:
            where_clauses = ["mc.is_excluded = 0"]
            params: list = [tsquery_expr]

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

            where_sql = " AND " + " AND ".join(where_clauses)

            sql = (
                "SELECT mc.qdrant_id, mc.file_path, mc.chunk_index, mc.agent_id, "
                "mc.namespace, mc.date, mc.memory_type, mc.text, "
                "mc.actor_id, mc.actor_type, "
                "ts_rank_cd(mc.fts_vector_simple, to_tsquery('simple', ?)) * 32 AS bm25_rank "
                "FROM memory_chunks mc "
                f"WHERE mc.fts_vector_simple @@ to_tsquery('simple', ?) {where_sql} "
                "ORDER BY bm25_rank DESC "
                "LIMIT ?"
            )
            params = [tsquery_expr, tsquery_expr] + params[1:] + [limit]

            cur = await conn.execute(sql, params)
            results = []
            for row in await cur.fetchall():
                r = dict(row)
                r["bm25_score"] = r.pop("bm25_rank", 0)
                results.append(r)
        m.observe(
            m.FTS_SEARCH_DURATION_MS, (time.monotonic() - _t0) * 1000.0, {"backend": "postgres"}
        )
        m.inc(m.FTS_SEARCH_TOTAL, {"backend": "postgres"})
        return results
    except Exception as e:
        logging.getLogger("archivist.graph").warning("FTS exact Postgres search failed: %s", e)
        return []


_RETENTION_RANK = {"ephemeral": 0, "standard": 1, "durable": 2, "permanent": 3}


async def upsert_entity(
    name: str,
    entity_type: str = "unknown",
    agent_id: str = "",
    retention_class: str = "standard",
    namespace: str = "global",
    actor_id: str = "",
    actor_type: str = "",
    conn: aiosqlite.Connection | None = None,
) -> int:
    """Insert or update an entity, returning its integer ID.

    Idempotent by design: concurrent calls with the same ``(name, namespace)``
    pair never raise ``IntegrityError``.  Uses a single-statement
    ``INSERT … ON CONFLICT DO UPDATE`` which is atomic on both SQLite (3.24+)
    and Postgres — there is no race window between a SELECT and a separate
    INSERT the way a check-then-act pattern would have.

    When the row already exists the ``mention_count`` is incremented and
    ``retention_class`` is escalated to the higher rank if the incoming class
    outranks the stored one.

    Args:
        conn: Optional open ``aiosqlite.Connection``.  When provided the write
            joins the caller's transaction; when ``None`` a fresh
            ``pool.write()`` lock is acquired.

    Returns:
        The integer primary-key ID of the entity (new or existing).
    """
    import archivist.core.metrics as _m
    from archivist.storage.sqlite_pool import pool

    now = datetime.now(UTC).isoformat()

    async def _run(c: aiosqlite.Connection) -> int:
        # Single atomic UPSERT — no SELECT-then-INSERT race window.
        # ON CONFLICT targets the (name, namespace) composite unique constraint.
        # retention_class comparison uses the _RETENTION_RANK mapping: a string
        # compare would be alphabetical and incorrect, so we embed a CASE
        # expression that mirrors the Python rank dict ordering.
        sql = """
            INSERT INTO entities (
                name, entity_type, first_seen, last_seen,
                retention_class, namespace, actor_id, actor_type,
                mention_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(name, namespace) DO UPDATE SET
                last_seen       = excluded.last_seen,
                mention_count   = entities.mention_count + 1,
                retention_class = CASE
                    WHEN excluded.retention_class = 'permanent'
                         AND entities.retention_class != 'permanent'            THEN 'permanent'
                    WHEN excluded.retention_class = 'durable'
                         AND entities.retention_class NOT IN ('permanent','durable')
                                                                                THEN 'durable'
                    WHEN excluded.retention_class = 'standard'
                         AND entities.retention_class = 'ephemeral'             THEN 'standard'
                    ELSE entities.retention_class
                END
            RETURNING id
        """
        cur = await c.execute(
            sql,
            (name, entity_type, now, now, retention_class, namespace, actor_id, actor_type),
        )
        row = await cur.fetchone()
        entity_id: int = row[0] if row else 0

        if entity_id:
            _m.inc(_m.ENTITY_UPSERT_TOTAL, {"namespace": namespace})

        return entity_id

    if conn is not None:
        return await _run(conn)
    async with pool.write() as c:
        return await _run(c)


async def add_relationship(
    source_id: int,
    target_id: int,
    rel_type: str,
    evidence: str,
    agent_id: str = "",
    provenance: str = "unknown",
    namespace: str = "global",
):
    """Insert or update a relationship between two entities."""
    from archivist.storage.sqlite_pool import pool

    now = datetime.now(UTC).isoformat()
    async with pool.write() as conn:
        await conn.execute(
            """INSERT INTO relationships (source_entity_id, target_entity_id, relation_type,
               evidence, agent_id, created_at, updated_at, provenance, namespace)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(source_entity_id, target_entity_id, relation_type)
               DO UPDATE SET evidence=excluded.evidence, updated_at=excluded.updated_at,
               confidence=min(confidence+0.1, 1.0), provenance=excluded.provenance""",
            (source_id, target_id, rel_type, evidence, agent_id, now, now, provenance, namespace),
        )


def _word_set(text: str) -> set[str]:
    """Extract lowercase word tokens for overlap comparison."""
    return {w for w in text.lower().split() if len(w) >= 2}


_DATE_IN_PATH_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


async def add_fact(
    entity_id: int,
    fact_text: str,
    source_file: str = "",
    agent_id: str = "",
    retention_class: str = "standard",
    valid_from: str = "",
    valid_until: str = "",
    namespace: str = "global",
    memory_id: str = "",
    confidence: float = 1.0,
    provenance: str = "unknown",
    actor_id: str = "",
    conn: aiosqlite.Connection | None = None,
) -> int:
    """Insert a new fact and auto-supersede overlapping existing facts.

    Args:
        conn: Optional open ``aiosqlite.Connection``.  When provided the write
            joins the caller's transaction; when ``None`` a fresh
            ``pool.write()`` lock is acquired.
    """
    from archivist.storage.sqlite_pool import pool

    now = datetime.now(UTC).isoformat()
    new_words = _word_set(fact_text)

    if not valid_from and source_file:
        _m = _DATE_IN_PATH_RE.search(source_file)
        if _m:
            valid_from = _m.group(1)

    async def _run(c: aiosqlite.Connection) -> int:
        cur = await c.execute(
            "INSERT INTO facts (entity_id, fact_text, source_file, agent_id, created_at, "
            "retention_class, valid_from, valid_until, namespace, memory_id, confidence, provenance, actor_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) RETURNING id",
            (
                entity_id,
                fact_text,
                source_file,
                agent_id,
                now,
                retention_class,
                valid_from,
                valid_until,
                namespace,
                memory_id,
                confidence,
                provenance,
                actor_id,
            ),
        )
        fid_row = await cur.fetchone()
        fid = fid_row[0] if fid_row else 0

        if new_words:
            old_facts_cur = await c.execute(
                "SELECT id, fact_text FROM facts "
                "WHERE entity_id=? AND is_active=1 AND id!=? AND superseded_by IS NULL",
                (entity_id, fid),
            )
            old_facts = await old_facts_cur.fetchall()

            superseded_ids = []
            for old in old_facts:
                old_words = _word_set(old["fact_text"])
                if not old_words:
                    continue
                overlap = len(new_words & old_words) / max(len(old_words), 1)
                if overlap >= 0.6:
                    superseded_ids.append(old["id"])

            if superseded_ids:
                placeholders = ",".join("?" for _ in superseded_ids)
                await c.execute(
                    f"UPDATE facts SET superseded_by=? WHERE id IN ({placeholders})",
                    [fid] + superseded_ids,
                )

        return fid

    if conn is not None:
        return await _run(conn)
    async with pool.write() as c:
        return await _run(c)


async def invalidate_fact(fact_id: int, ended: str = ""):
    """Mark a fact as no longer valid by setting ``valid_until``.

    If *ended* is empty the current UTC date is used.
    """
    from archivist.storage.sqlite_pool import pool

    if not ended:
        ended = datetime.now(UTC).strftime("%Y-%m-%d")
    async with pool.write() as conn:
        await conn.execute(
            "UPDATE facts SET valid_until=? WHERE id=?",
            (ended, fact_id),
        )


async def supersede_fact(old_fact_id: int, new_fact_id: int):
    """Explicitly mark an old fact as superseded by a newer one."""
    from archivist.storage.sqlite_pool import pool

    async with pool.write() as conn:
        await conn.execute(
            "UPDATE facts SET superseded_by=? WHERE id=?",
            (new_fact_id, old_fact_id),
        )


def _normalize(text: str) -> str:
    """Lowercase, strip non-alphanumeric except hyphens/underscores."""
    return re.sub(r"[^\w\s\-]", "", text.lower()).strip()


async def search_entities(query: str, limit: int = 10, namespace: str = "") -> list[dict]:
    """Search entities by name or aliases (case-insensitive, normalized)."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        norm_q = _normalize(query)
        if namespace:
            cur = await conn.execute(
                "SELECT * FROM entities "
                "WHERE (name LIKE ? OR aliases LIKE ?) "
                "AND namespace = ? "
                "ORDER BY mention_count DESC LIMIT ?",
                (f"%{query}%", f"%{norm_q}%", namespace, limit),
            )
        else:
            cur = await conn.execute(
                "SELECT * FROM entities "
                "WHERE name LIKE ? OR aliases LIKE ? "
                "ORDER BY mention_count DESC LIMIT ?",
                (f"%{query}%", f"%{norm_q}%", limit),
            )
        return [dict(r) for r in await cur.fetchall()]


async def add_entity_alias(entity_id: int, alias: str):
    """Add an alias to an entity (idempotent)."""
    import json as _json

    from archivist.storage.sqlite_pool import pool

    norm = _normalize(alias)
    if not norm:
        return
    async with pool.write() as conn:
        row = await (
            await conn.execute("SELECT aliases FROM entities WHERE id=?", (entity_id,))
        ).fetchone()
        if row:
            try:
                current = _json.loads(row["aliases"])
            except Exception:
                current = []
            if norm not in current:
                current.append(norm)
                await conn.execute(
                    "UPDATE entities SET aliases=? WHERE id=?",
                    (_json.dumps(current), entity_id),
                )


async def get_entity_by_id(entity_id: int) -> dict | None:
    """Return entity dict by primary key, or None if not found."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        row = await (
            await conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,))
        ).fetchone()
        return dict(row) if row else None


async def get_entity_facts(
    entity_id: int, include_superseded: bool = False, as_of: str = ""
) -> list[dict]:
    """Get active facts for an entity.

    Non-superseded facts come first. Superseded facts are included only when
    ``include_superseded`` is True (useful for history views).

    When ``as_of`` is an ISO-date string (e.g. ``"2025-03-15"``), only facts
    whose validity window contains that date are returned.  Dateless facts
    (empty ``valid_from``) are always included.
    """
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        base = "SELECT * FROM facts WHERE entity_id=? AND is_active=1"
        params: list = [entity_id]

        if not include_superseded:
            base += " AND superseded_by IS NULL"

        if as_of:
            base += " AND (valid_from = '' OR valid_from <= ?)"
            params.append(as_of)
            base += " AND (valid_until = '' OR valid_until > ?)"
            params.append(as_of)

        if include_superseded:
            base += " ORDER BY (superseded_by IS NOT NULL), created_at DESC"
        else:
            base += " ORDER BY created_at DESC"

        cur = await conn.execute(base, params)
        results = []
        for r in await cur.fetchall():
            d = dict(r)
            d["is_current"] = d.get("superseded_by") is None
            results.append(d)
        return results


async def get_entity_relationships(entity_id: int) -> list[dict]:
    """Return all relationships involving the given entity."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        cur = await conn.execute(
            """SELECT r.*, e1.name AS source_name, e2.name AS target_name
               FROM relationships r
               JOIN entities e1 ON r.source_entity_id=e1.id
               JOIN entities e2 ON r.target_entity_id=e2.id
               WHERE r.source_entity_id=? OR r.target_entity_id=?
               ORDER BY r.updated_at DESC""",
            (entity_id, entity_id),
        )
        return [dict(r) for r in await cur.fetchall()]


async def get_entity_facts_bulk(entity_ids: list[int], as_of: str = "") -> dict[int, list[dict]]:
    """Fetch active, non-superseded facts for multiple entities in one query."""
    if not entity_ids:
        return {}
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        placeholders = ",".join("?" for _ in entity_ids)
        base = (
            f"SELECT * FROM facts WHERE entity_id IN ({placeholders}) "
            "AND is_active=1 AND superseded_by IS NULL"
        )
        params: list = list(entity_ids)
        if as_of:
            base += " AND (valid_from = '' OR valid_from <= ?)"
            params.append(as_of)
            base += " AND (valid_until = '' OR valid_until > ?)"
            params.append(as_of)
        base += " ORDER BY entity_id, created_at DESC"
        cur = await conn.execute(base, params)
        result: dict[int, list[dict]] = {eid: [] for eid in entity_ids}
        for r in await cur.fetchall():
            d = dict(r)
            d["is_current"] = True
            result.setdefault(d["entity_id"], []).append(d)
        return result


async def get_entity_relationships_bulk(entity_ids: list[int]) -> dict[int, list[dict]]:
    """Fetch relationships for multiple entities in one query."""
    if not entity_ids:
        return {}
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        placeholders = ",".join("?" for _ in entity_ids)
        params = list(entity_ids) + list(entity_ids)
        cur = await conn.execute(
            f"""SELECT r.*, e1.name AS source_name, e2.name AS target_name
                FROM relationships r
                JOIN entities e1 ON r.source_entity_id=e1.id
                JOIN entities e2 ON r.target_entity_id=e2.id
                WHERE r.source_entity_id IN ({placeholders})
                   OR r.target_entity_id IN ({placeholders})
                ORDER BY r.updated_at DESC""",
            params,
        )
        result: dict[int, list[dict]] = {eid: [] for eid in entity_ids}
        for r in await cur.fetchall():
            d = dict(r)
            if d["source_entity_id"] in result:
                result[d["source_entity_id"]].append(d)
            if d["target_entity_id"] in result and d["target_entity_id"] != d["source_entity_id"]:
                result[d["target_entity_id"]].append(d)
        return result


async def get_curator_state(key: str) -> str | None:
    """Read a single key from the curator_state table."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        cur = await conn.execute("SELECT value FROM curator_state WHERE key=?", (key,))
        row = await cur.fetchone()
        return row["value"] if row else None


async def set_curator_state(key: str, value: str):
    """Upsert a key/value pair in the curator_state table."""
    from archivist.storage.sqlite_pool import pool

    async with pool.write() as conn:
        await conn.execute(
            "INSERT INTO curator_state (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


# ── Deterministic needle registry (v2.0 — 100% recall for structured tokens) ─

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
                "INSERT INTO needle_registry "
                "(token, memory_id, namespace, agent_id, actor_id, actor_type, chunk_text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (token, memory_id) DO UPDATE SET "
                "chunk_text=EXCLUDED.chunk_text, created_at=EXCLUDED.created_at",
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


_BATCH_CHUNK = 500


async def delete_fts_chunks_batch(qdrant_ids: list[str]) -> int:
    """Remove FTS5 entries and memory_chunks rows for multiple Qdrant IDs.

    Internally chunks the ID list into groups of 500 to stay under the
    sqlite3 ~999-parameter limit.  Retries once on ``OperationalError``
    (e.g. "database is locked").
    """
    if not qdrant_ids:
        return 0
    from archivist.storage.sqlite_pool import pool

    for attempt in range(2):
        try:
            total = 0
            async with pool.write() as conn:
                for i in range(0, len(qdrant_ids), _BATCH_CHUNK):
                    chunk = qdrant_ids[i : i + _BATCH_CHUNK]
                    placeholders = ",".join("?" * len(chunk))
                    rows = await (
                        await conn.execute(
                            f"SELECT rowid FROM memory_chunks WHERE qdrant_id IN ({placeholders})",
                            chunk,
                        )
                    ).fetchall()
                    await _delete_fts_rows_async(conn, rows)
                    cur = await conn.execute(
                        f"DELETE FROM memory_chunks WHERE qdrant_id IN ({placeholders})",
                        chunk,
                    )
                    total += cur.rowcount
            return total
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


async def delete_needle_tokens_batch(memory_ids: list[str]) -> int:
    """Remove needle_registry rows for multiple memory IDs.

    Internally chunks the ID list into groups of 500 to stay under the
    sqlite3 ~999-parameter limit.  Retries once on ``OperationalError``
    (e.g. "database is locked").
    """
    if not memory_ids:
        return 0
    from archivist.storage.sqlite_pool import pool

    _ensure_needle_registry()
    for attempt in range(2):
        try:
            total = 0
            async with pool.write() as conn:
                for i in range(0, len(memory_ids), _BATCH_CHUNK):
                    chunk = memory_ids[i : i + _BATCH_CHUNK]
                    placeholders = ",".join("?" * len(chunk))
                    cur = await conn.execute(
                        f"DELETE FROM needle_registry WHERE memory_id IN ({placeholders})",
                        chunk,
                    )
                    total += cur.rowcount
            return total
        except Exception as e:
            if attempt == 0 and "locked" in str(e).lower():
                import asyncio as _asyncio

                await _asyncio.sleep(0.2)
                continue
            raise
    return 0


async def delete_hotness(memory_id: str) -> int:
    """Remove the ``memory_hotness`` row for *memory_id*.

    Returns the number of rows deleted (0 or 1).  Silently returns 0 if the
    ``memory_hotness`` table does not yet exist (it is lazily created by
    ``hotness.refresh_hotness``).
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
        return 0


# ---------------------------------------------------------------------------
# memory_points tracking (Phase 2)
# ---------------------------------------------------------------------------


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
                    "INSERT INTO memory_points "
                    "(memory_id, qdrant_id, point_type, created_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT (memory_id, qdrant_id) DO NOTHING",
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
