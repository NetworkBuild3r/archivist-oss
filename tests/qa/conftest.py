"""Shared fixtures for the tests/qa/ suite.

Every fixture here is self-contained — no dependency on the root conftest.py
(which doesn't exist).  Each test module also imports this conftest implicitly
via pytest's conftest resolution.

Key design decisions
--------------------
* ``qa_pool`` — a fresh ``SQLitePool`` per test, backed by a temp-file DB with
  the full Archivist schema applied inline.  ``GRAPH_WRITE_LOCK_ASYNC`` is also
  replaced with a fresh ``asyncio.Lock`` bound to the current event loop so
  tests do not interfere with each other across event-loop boundaries.

* ``OUTBOX_ENABLED=True`` is set as an autouse override so every test in this
  suite exercises the transactional outbox by default.

* ``mock_vector_backend`` — a ``MagicMock`` satisfying the ``VectorBackend``
  protocol with all methods as ``AsyncMock``.  No real Qdrant calls.

* ``memory_factory`` — returns a callable producing realistic memory payload
  dicts whose text contains NEEDLE_PATTERN-matchable tokens (datetimes, ticket
  IDs) so ``register_needle_tokens()`` actually inserts rows.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Schema DDL (mirrors graph.init_schema + _migrate_schema output)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    entity_type TEXT NOT NULL DEFAULT 'unknown',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    metadata TEXT DEFAULT '{}',
    retention_class TEXT NOT NULL DEFAULT 'standard',
    aliases TEXT NOT NULL DEFAULT '[]',
    namespace TEXT NOT NULL DEFAULT 'global',
    actor_id TEXT NOT NULL DEFAULT '',
    actor_type TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE);
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
    namespace TEXT NOT NULL DEFAULT 'global',
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
    memory_id TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    provenance TEXT NOT NULL DEFAULT 'unknown',
    actor_id TEXT NOT NULL DEFAULT '',
    namespace TEXT NOT NULL DEFAULT 'global'
);
CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(is_active);
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
    actor_type TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_mc_qdrant ON memory_chunks(qdrant_id);
CREATE INDEX IF NOT EXISTS idx_mc_namespace ON memory_chunks(namespace);

CREATE TABLE IF NOT EXISTS memory_points (
    memory_id   TEXT NOT NULL,
    qdrant_id   TEXT NOT NULL,
    point_type  TEXT NOT NULL DEFAULT 'primary',
    created_at  TEXT NOT NULL,
    PRIMARY KEY (memory_id, qdrant_id)
);
CREATE INDEX IF NOT EXISTS idx_mp_memory ON memory_points(memory_id);

CREATE TABLE IF NOT EXISTS delete_failures (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    qdrant_ids  TEXT NOT NULL,
    error       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    resolved_at TEXT
);

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

CREATE TABLE IF NOT EXISTS needle_registry (
    token      TEXT NOT NULL,
    memory_id  TEXT NOT NULL,
    namespace  TEXT NOT NULL DEFAULT '',
    agent_id   TEXT NOT NULL DEFAULT '',
    actor_id   TEXT NOT NULL DEFAULT '',
    actor_type TEXT NOT NULL DEFAULT '',
    chunk_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY (token, memory_id)
);
CREATE INDEX IF NOT EXISTS idx_needle_token ON needle_registry(token);

CREATE TABLE IF NOT EXISTS memory_hotness (
    memory_id   TEXT PRIMARY KEY,
    score       REAL NOT NULL DEFAULT 0.0,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_FTS5_SQL = [
    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(text, content='memory_chunks', content_rowid='rowid')",
    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts_exact USING fts5(text, content='memory_chunks', content_rowid='rowid')",
]


async def _build_schema(pool) -> None:
    """Apply the Archivist schema to *pool*'s connection."""
    async with pool.write() as conn:
        await conn.executescript(_SCHEMA_SQL)
        for stmt in _FTS5_SQL:
            try:
                await conn.execute(stmt)
            except Exception:
                pass
        await conn.commit()


# ---------------------------------------------------------------------------
# Core pool fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def qa_pool(tmp_path, monkeypatch):
    """Fresh ``SQLitePool`` per test, backed by a temp-file DB with full schema.

    * Patches ``archivist.storage.sqlite_pool.pool`` so ``MemoryTransaction``
      and ``OutboxProcessor`` use this isolated pool.
    * ``pool.write()`` now calls ``_get_graph_write_lock()`` which is
      loop-aware, so no explicit lock monkeypatching is required.
    * Resets all graph schema guards so the DDL can re-run on the fresh DB.
    """
    from archivist.storage import sqlite_pool as _sp

    p = _sp.SQLitePool()
    db_path = str(tmp_path / "qa_test.db")
    await p.initialize(db_path)

    # Swap the singleton before building the schema.
    original_pool = _sp.pool
    monkeypatch.setattr(_sp, "pool", p)

    # Reset graph schema guards so they don't block re-initialisation.
    from archivist.storage import graph as _graph

    for attr in dir(_graph):
        obj = getattr(_graph, attr, None)
        if hasattr(obj, "reset") and callable(obj.reset):
            obj.reset()
    # Mark the needle guard as applied so it doesn't try to acquire a lock.
    if hasattr(_graph, "_ensure_needle_registry"):
        _graph._ensure_needle_registry.applied = True  # type: ignore[attr-defined]

    await _build_schema(p)

    yield p

    monkeypatch.setattr(_sp, "pool", original_pool)
    await p.close()


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_outbox(monkeypatch):
    """Force OUTBOX_ENABLED=True for every test in tests/qa/."""
    import archivist.core.config as _cfg

    monkeypatch.setattr(_cfg, "OUTBOX_ENABLED", True)
    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 5)
    monkeypatch.setattr(_cfg, "OUTBOX_BATCH_SIZE", 50)
    monkeypatch.setattr(_cfg, "OUTBOX_DRAIN_INTERVAL", 1)


# ---------------------------------------------------------------------------
# Mock vector backend
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vector_backend() -> MagicMock:
    """``VectorBackend``-compatible mock.  No real Qdrant calls."""
    backend = MagicMock()
    backend.upsert = AsyncMock(return_value=None)
    backend.delete = AsyncMock(return_value=None)
    backend.delete_by_filter = AsyncMock(return_value=None)
    backend.set_payload = AsyncMock(return_value=None)
    backend.retrieve = AsyncMock(return_value=[])
    backend.ensure_collection = AsyncMock(return_value=None)
    return backend


# ---------------------------------------------------------------------------
# Data factory
# ---------------------------------------------------------------------------

# Texts contain NEEDLE_PATTERN-matchable tokens (datetimes, ticket IDs, IPs).
_FAKE_TOPICS = [
    "user prefers dark mode — config USER-1234 set 2026-01-17T09:00",
    "project deadline 2026-06-30T00:00 ticket PROJ-5678",
    "deployment runs on 10.0.0.1/24 — ticket OPS-9012",
    "embedding model EMB-3456 at 2026-02-01T12:00 dimension 1536",
    "API rate limit 1000 req/min — TICKET-7890 logged 2026-03-15T08:30",
    "memory retention 90 days — POLICY-2345 since 2025-09-01T00:00",
    "outbox drain KEY=2 — OPS-4567 updated 2026-04-17T10:00",
    "Qdrant 192.168.1.10:6333 — CONFIG-8901 healthy 2026-04-17T11:00",
    "SQLite WAL mode — CONF-1122 at 2026-01-01T00:00",
    "phase 3 rollout FEAT-3344 closed 2026-04-01T17:00",
]


def _make_memory_id() -> str:
    return str(uuid.uuid4())


def _make_qdrant_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def memory_factory():
    """Return a callable producing realistic memory payload dicts."""
    _counter = [0]

    def _factory(
        namespace: str = "default",
        agent_id: str = "qa-agent",
        actor_id: str = "user-qa",
        actor_type: str = "human",
        memory_type: str = "general",
        text: str | None = None,
    ) -> dict[str, Any]:
        idx = _counter[0] % len(_FAKE_TOPICS)
        _counter[0] += 1
        return {
            "memory_id": _make_memory_id(),
            "qdrant_id": _make_qdrant_id(),
            "text": text or _FAKE_TOPICS[idx],
            "namespace": namespace,
            "agent_id": agent_id,
            "actor_id": actor_id,
            "actor_type": actor_type,
            "memory_type": memory_type,
            "file_path": f"qa/{agent_id}/{idx}.md",
            "chunk_index": idx,
            "date": "2026-01-17",
        }

    return _factory


# ---------------------------------------------------------------------------
# Helpers exposed to test modules
# ---------------------------------------------------------------------------


async def count_outbox(pool, status: str | None = None) -> int:
    """Count outbox rows, optionally filtered by *status*."""
    async with pool.read() as conn:
        if status:
            cur = await conn.execute("SELECT COUNT(*) FROM outbox WHERE status=?", (status,))
        else:
            cur = await conn.execute("SELECT COUNT(*) FROM outbox")
        row = await cur.fetchone()
        return row[0]


async def count_table(pool, table: str) -> int:
    """Count all rows in *table*."""
    async with pool.read() as conn:
        cur = await conn.execute(f"SELECT COUNT(*) FROM {table}")
        row = await cur.fetchone()
        return row[0]


async def reset_outbox_backoff(pool) -> None:
    """Force all pending outbox events to be immediately retryable."""
    async with pool.write() as conn:
        await conn.execute(
            "UPDATE outbox SET last_attempt='2000-01-01T00:00:00+00:00' WHERE status='pending'"
        )
