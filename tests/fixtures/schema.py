"""Single source of truth for the Archivist SQLite schema used in tests.

Both the integration-tier pool fixture and the system-tier qa_pool fixture
import ``build_schema`` from here.  Previously the DDL was duplicated across
``conftest.py`` (root) and ``tests/qa/conftest.py``; this module eliminates
that divergence.

Usage::

    from tests.fixtures.schema import build_schema

    async with pool.write() as conn:
        await build_schema(conn)
"""

from __future__ import annotations

_SCHEMA_SQL: str = """
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

CREATE TABLE IF NOT EXISTS annotations (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    annotation  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ann_memory ON annotations(memory_id);

CREATE TABLE IF NOT EXISTS ratings (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    rating      INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rat_memory ON ratings(memory_id);

CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    mcp_endpoint TEXT DEFAULT '',
    current_version TEXT NOT NULL DEFAULT '0.0.0',
    status TEXT NOT NULL DEFAULT 'active',
    description TEXT DEFAULT '',
    registered_by TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    UNIQUE(name, provider)
);
CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
CREATE INDEX IF NOT EXISTS idx_skills_provider ON skills(provider);
CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(status);

CREATE TABLE IF NOT EXISTS skill_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL REFERENCES skills(id),
    version TEXT NOT NULL,
    changelog TEXT DEFAULT '',
    breaking_changes TEXT DEFAULT '',
    observed_at TEXT NOT NULL,
    reported_by TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    UNIQUE(skill_id, version)
);
CREATE INDEX IF NOT EXISTS idx_sv_skill ON skill_versions(skill_id);

CREATE TABLE IF NOT EXISTS skill_lessons (
    id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL REFERENCES skills(id),
    lesson_type TEXT NOT NULL DEFAULT 'general',
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    skill_version TEXT DEFAULT '',
    agent_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    upvotes INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sl_skill ON skill_lessons(skill_id);
CREATE INDEX IF NOT EXISTS idx_sl_type ON skill_lessons(lesson_type);

CREATE TABLE IF NOT EXISTS skill_events (
    id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL REFERENCES skills(id),
    agent_id TEXT NOT NULL,
    event_type TEXT NOT NULL DEFAULT 'invocation',
    outcome TEXT NOT NULL DEFAULT 'unknown',
    skill_version TEXT DEFAULT '',
    duration_ms INTEGER,
    error_message TEXT DEFAULT '',
    trajectory_id TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_se_skill ON skill_events(skill_id);
CREATE INDEX IF NOT EXISTS idx_se_agent ON skill_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_se_outcome ON skill_events(outcome);

CREATE TABLE IF NOT EXISTS skill_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_a TEXT NOT NULL REFERENCES skills(id),
    skill_b TEXT NOT NULL REFERENCES skills(id),
    relation_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    evidence TEXT DEFAULT '',
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sr_a ON skill_relations(skill_a);
CREATE INDEX IF NOT EXISTS idx_sr_b ON skill_relations(skill_b);
CREATE INDEX IF NOT EXISTS idx_sr_type ON skill_relations(relation_type);

CREATE TABLE IF NOT EXISTS trajectories (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    task_description TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT 'unknown',
    created_at TEXT NOT NULL,
    session_id TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_traj_agent ON trajectories(agent_id);

CREATE TABLE IF NOT EXISTS tips (
    id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL REFERENCES trajectories(id),
    agent_id TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    tip_text TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,
    negative_example TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tips_agent ON tips(agent_id);
CREATE INDEX IF NOT EXISTS idx_tips_category ON tips(category);
CREATE INDEX IF NOT EXISTS idx_tips_archived ON tips(archived);
"""

_FTS5_SQL: list[str] = [
    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(text, content='memory_chunks', content_rowid='rowid')",
    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts_exact USING fts5(text, content='memory_chunks', content_rowid='rowid')",
]


async def build_schema(conn) -> None:
    """Apply the full Archivist schema DDL to an open aiosqlite connection.

    Safe to call on an existing database — all statements use IF NOT EXISTS.
    """
    await conn.executescript(_SCHEMA_SQL)
    for stmt in _FTS5_SQL:
        try:
            await conn.execute(stmt)
        except Exception:
            pass
    await conn.commit()
