"""SQLite/Postgres schema init, migrations, and low-level connection helpers.

Split from the former monolithic ``storage/graph.py`` (INIT-001/SPEC-003).
Provenance: INIT-001/SPEC-003.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from contextlib import contextmanager

logger = logging.getLogger("archivist.graph")

# ---------------------------------------------------------------------------
# Backward-compatibility shim: keep the old threading.Lock name so any
# remaining direct imports in backup_manager or test code don't break.
# All internal code now uses the pool pattern.
# ---------------------------------------------------------------------------
GRAPH_WRITE_LOCK = threading.Lock()

# Shared batch-chunk size for parameterized IN(...) deletes across the FTS
# and needle-registry modules (sqlite3 ~999-parameter limit).
_BATCH_CHUNK = 500


def _is_postgres() -> bool:
    """Return True when the active graph backend is PostgreSQL."""
    from archivist.core.config import GRAPH_BACKEND

    return (GRAPH_BACKEND or "sqlite").lower() == "postgres"


def _ensure_dir():
    from archivist.core.config import SQLITE_PATH

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
    from archivist.core.config import GRAPH_BACKEND, SQLITE_PATH

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

    On **Postgres** the callable is a no-op: all tables (including those created
    by per-module guard strings) are already present in ``schema_postgres.sql``
    which is applied once at startup by :func:`init_schema_async`.  Marking
    the guard as already applied prevents any attempt to run SQLite-only DDL
    strings through a sync ``get_db()`` connection when Postgres is active.

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
            # On Postgres all schema is handled by init_schema_async(); skip.
            if _is_postgres():
                _ensure.applied = True
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
    """Initialize the SQLite schema.

    On Postgres this is a no-op — use :func:`init_schema_async` instead,
    which loads ``schema_postgres.sql`` via the async pool.
    """
    if _is_postgres():
        # All Postgres DDL lives in schema_postgres.sql and is applied by
        # init_schema_async().  Calling get_db() on Postgres would return
        # a SQLite connection, which is wrong.
        logging.getLogger("archivist.graph").debug(
            "init_schema() skipped — Postgres backend active; use init_schema_async() instead"
        )
        return
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            entity_type TEXT NOT NULL DEFAULT 'unknown',
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            mention_count INTEGER NOT NULL DEFAULT 1,
            metadata TEXT DEFAULT '{}',
            retention_class TEXT NOT NULL DEFAULT 'standard',
            aliases TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

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
            is_excluded INTEGER NOT NULL DEFAULT 0,
            actor_id TEXT NOT NULL DEFAULT '',
            actor_type TEXT NOT NULL DEFAULT '',
            importance REAL NOT NULL DEFAULT 0.5,
            tier_label TEXT NOT NULL DEFAULT 'l2',
            ttl_at TEXT,
            decay_rate REAL NOT NULL DEFAULT 0.0
        );
        CREATE INDEX IF NOT EXISTS idx_mc_qdrant ON memory_chunks(qdrant_id);
        CREATE INDEX IF NOT EXISTS idx_mc_namespace ON memory_chunks(namespace);
        CREATE INDEX IF NOT EXISTS idx_mc_agent ON memory_chunks(agent_id);
        CREATE INDEX IF NOT EXISTS idx_mc_importance ON memory_chunks(importance DESC);
        CREATE INDEX IF NOT EXISTS idx_mc_tier ON memory_chunks(tier_label);

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

        -- Hotness scoring table: created eagerly here so FTS LEFT JOINs always succeed.
        CREATE TABLE IF NOT EXISTS memory_hotness (
            memory_id         TEXT PRIMARY KEY,
            score             REAL NOT NULL DEFAULT 0.0,
            retrieval_count   INTEGER NOT NULL DEFAULT 0,
            last_accessed     TEXT,
            updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
            importance_signal REAL NOT NULL DEFAULT 0.5
        );

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

        -- Agent-state checkpoints (Phase 7 / INIT-001/SPEC-007).
        -- Distinct from memory_chunks.tier_label (L0–L2 delivery tiers; GR-002).
        -- payload is JSON text in-row for v1; blob_ref reserved for large payloads later.
        CREATE TABLE IF NOT EXISTS agent_checkpoints (
            id                   TEXT PRIMARY KEY,
            agent_id             TEXT NOT NULL,
            session_id           TEXT NOT NULL,
            namespace             TEXT NOT NULL DEFAULT 'global',
            parent_checkpoint_id TEXT REFERENCES agent_checkpoints(id),
            payload              TEXT NOT NULL DEFAULT '{}',
            blob_ref             TEXT,
            metadata             TEXT NOT NULL DEFAULT '{}',
            created_at           TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_checkpoints_agent_session_time
            ON agent_checkpoints(agent_id, session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_checkpoints_session_time
            ON agent_checkpoints(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_checkpoints_namespace
            ON agent_checkpoints(namespace);
        CREATE INDEX IF NOT EXISTS idx_checkpoints_parent
            ON agent_checkpoints(parent_checkpoint_id);

        -- Memory-as-Product scope version lineage (INIT-001/SPEC-009).
        -- Tracks namespace/agent-scoped snapshots, forks, and export records.
        -- Distinct from per-memory_id memory_versions and from agent_checkpoints.
        -- Rollback: DROP TABLE IF EXISTS memory_scope_versions;
        CREATE TABLE IF NOT EXISTS memory_scope_versions (
            id                TEXT PRIMARY KEY,
            source_namespace  TEXT NOT NULL,
            source_agent_id   TEXT NOT NULL DEFAULT '',
            version           INTEGER NOT NULL,
            label             TEXT NOT NULL DEFAULT '',
            parent_version_id TEXT,
            chunk_count       INTEGER NOT NULL DEFAULT 0,
            point_count       INTEGER NOT NULL DEFAULT 0,
            archive_id        TEXT NOT NULL DEFAULT '',
            operation         TEXT NOT NULL,
            created_by        TEXT NOT NULL DEFAULT '',
            created_at        TEXT NOT NULL,
            lineage_json      TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_msv_namespace_version
            ON memory_scope_versions(source_namespace, version);
        CREATE INDEX IF NOT EXISTS idx_msv_parent
            ON memory_scope_versions(parent_version_id);
        CREATE INDEX IF NOT EXISTS idx_msv_archive
            ON memory_scope_versions(archive_id);

        -- Selective share grants (Phase 10 / INIT-001/SPEC-010).
        -- Consensus v1 = explicit accept/reject + audit; extends handoff (GR-003).
        -- Rollback: DROP TABLE IF EXISTS memory_share_grants;
        CREATE TABLE IF NOT EXISTS memory_share_grants (
            id                  TEXT PRIMARY KEY,
            proposer_agent_id   TEXT NOT NULL,
            recipient_agent_id  TEXT NOT NULL,
            namespace            TEXT NOT NULL,
            memory_ids          TEXT NOT NULL DEFAULT '[]',
            scope               TEXT NOT NULL DEFAULT '',
            status              TEXT NOT NULL DEFAULT 'pending',
            conflict_outcome    TEXT,
            reason              TEXT NOT NULL DEFAULT '',
            metadata            TEXT NOT NULL DEFAULT '{}',
            created_at          TEXT NOT NULL,
            decided_at          TEXT,
            decided_by          TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_share_grants_recipient_status
            ON memory_share_grants(recipient_agent_id, status);
        CREATE INDEX IF NOT EXISTS idx_share_grants_proposer
            ON memory_share_grants(proposer_agent_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_share_grants_namespace
            ON memory_share_grants(namespace);
    """)
        conn.commit()
        conn.close()
    # Mark the needle-registry schema guard as already applied so it never tries
    # to acquire a second sync connection while the async pool lock is held.
    from archivist.storage.graph_needles import _ensure_needle_registry

    _ensure_needle_registry.applied = True  # type: ignore[attr-defined]
    try:
        from archivist.storage.share_grants import _ensure_share_grants_schema

        _ensure_share_grants_schema.applied = True  # type: ignore[attr-defined]
    except ImportError:
        pass
    _migrate_schema()
    _migrate_entity_unique_constraint()
    _init_fts5()


def _migrate_schema():
    """Add columns introduced in v1.7+ if upgrading from an older database.

    On Postgres this is a no-op: ``schema_postgres.sql`` already includes every
    column and index in its final form, so ``ALTER TABLE … ADD COLUMN`` is
    never needed.
    """
    if _is_postgres():
        return
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
        # Phase 1 answer-finder: tiered memory columns
        ("memory_chunks", "importance", "REAL NOT NULL DEFAULT 0.5"),
        ("memory_chunks", "tier_label", "TEXT NOT NULL DEFAULT 'l2'"),
        ("memory_chunks", "ttl_at", "TEXT"),
        ("memory_chunks", "decay_rate", "REAL NOT NULL DEFAULT 0.0"),
        ("memory_hotness", "importance_signal", "REAL NOT NULL DEFAULT 0.5"),
        # Phase 5 answer-finder: token savings in retrieval_logs
        ("retrieval_logs", "tokens_returned", "INTEGER"),
        ("retrieval_logs", "tokens_naive", "INTEGER"),
        ("retrieval_logs", "savings_pct", "REAL"),
        ("retrieval_logs", "pack_policy", "TEXT DEFAULT " + "''"),
    ]
    # needle_registry may not exist yet (schema_guard creates it lazily),
    # so these ALTER TABLEs are attempted but silently skipped on failure.
    _needle_migrations = [
        ("needle_registry", "actor_id", "TEXT NOT NULL DEFAULT " + "''"),
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
        # Phase 1 answer-finder: tier/importance indexes
        "CREATE INDEX IF NOT EXISTS idx_mc_importance ON memory_chunks(importance DESC)",
        "CREATE INDEX IF NOT EXISTS idx_mc_tier ON memory_chunks(tier_label)",
        # Phase 5 answer-finder: pack_policy index on retrieval_logs
        "CREATE INDEX IF NOT EXISTS idx_rl_pack_policy ON retrieval_logs(pack_policy)",
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
            except sqlite3.OperationalError:
                pass
        for ddl in indexes:
            try:
                conn.execute(ddl)
                conn.commit()
            except Exception as e:
                # CREATE INDEX IF NOT EXISTS is idempotent, so a real failure here
                # (e.g. a referenced column missing because an earlier migration
                # failed) should leave a trace instead of vanishing silently.
                if "already exists" not in str(e).lower():
                    _logger.debug("Index creation skipped for %r: %s", ddl, e)
        conn.close()


def _migrate_entity_unique_constraint():
    """Rebuild entities UNIQUE constraint to include namespace (idempotent).

    The original schema has UNIQUE(name) which collides across namespaces.
    This migration copies to a new table with UNIQUE(name, namespace),
    then swaps in place.  Safe to run multiple times.

    On Postgres this is a no-op: ``schema_postgres.sql`` defines the correct
    constraint from the start and uses SQLite-specific ``PRAGMA``/``sqlite_master``
    introspection that is not valid on Postgres.
    """
    if _is_postgres():
        return
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
                    name TEXT NOT NULL COLLATE NOCASE,
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE)"
            )
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

    On Postgres this is a no-op: FTS is provided by ``tsvector``/GIN columns
    defined in ``schema_postgres.sql``.  The ``fts5`` health key is set to
    healthy by :func:`init_schema_async` after the Postgres DDL is applied.
    """
    if _is_postgres():
        return
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
