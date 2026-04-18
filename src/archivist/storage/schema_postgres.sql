-- Archivist PostgreSQL schema
-- Translated from the SQLite DDL in src/archivist/storage/graph.py and
-- other module-level schema_guard() strings.
--
-- Key translation notes:
--   SQLite INTEGER PRIMARY KEY AUTOINCREMENT  →  SERIAL / BIGSERIAL
--   SQLite REAL                               →  DOUBLE PRECISION
--   SQLite INTEGER                            →  INTEGER / BIGINT
--   SQLite TEXT                               →  TEXT
--   COLLATE NOCASE on TEXT                    →  citext extension OR lower() index
--   FTS5 virtual tables                       →  replaced by tsvector / GIN index
--                                                (exact parity is a follow-up)
--   Partial indexes (WHERE clause)            →  supported natively in Postgres
--
-- Usage: psql -d archivist -f schema_postgres.sql
--        (all statements are idempotent via IF NOT EXISTS / CREATE INDEX CONCURRENTLY)

-- ---------------------------------------------------------------------------
-- Extension: case-insensitive text
-- ---------------------------------------------------------------------------
-- citext lets us preserve the NOCASE collation on entities.name.
CREATE EXTENSION IF NOT EXISTS citext;


-- ---------------------------------------------------------------------------
-- Core knowledge-graph tables (from graph.py init_schema)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entities (
    id               SERIAL PRIMARY KEY,
    name             CITEXT NOT NULL,
    entity_type      TEXT NOT NULL DEFAULT 'unknown',
    first_seen       TEXT NOT NULL,
    last_seen        TEXT NOT NULL,
    mention_count    INTEGER NOT NULL DEFAULT 1,
    metadata         TEXT NOT NULL DEFAULT '{}',
    retention_class  TEXT NOT NULL DEFAULT 'standard',
    aliases          TEXT NOT NULL DEFAULT '[]',
    namespace        TEXT NOT NULL DEFAULT 'global',
    actor_id         TEXT NOT NULL DEFAULT '',
    actor_type       TEXT NOT NULL DEFAULT '',
    CONSTRAINT entities_name_unique UNIQUE (name)
);

CREATE INDEX IF NOT EXISTS idx_entities_name       ON entities (name);
CREATE INDEX IF NOT EXISTS idx_entities_type       ON entities (entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_retention  ON entities (retention_class);
CREATE INDEX IF NOT EXISTS idx_entities_namespace  ON entities (namespace);
CREATE INDEX IF NOT EXISTS idx_entities_actor      ON entities (actor_id);


CREATE TABLE IF NOT EXISTS relationships (
    id               SERIAL PRIMARY KEY,
    source_entity_id INTEGER NOT NULL REFERENCES entities (id),
    target_entity_id INTEGER NOT NULL REFERENCES entities (id),
    relation_type    TEXT NOT NULL,
    evidence         TEXT NOT NULL,
    agent_id         TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    confidence       DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    provenance       TEXT NOT NULL DEFAULT 'unknown',
    namespace        TEXT NOT NULL DEFAULT 'global',
    CONSTRAINT rel_unique UNIQUE (source_entity_id, target_entity_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_rel_source              ON relationships (source_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_target              ON relationships (target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_namespace ON relationships (namespace);


CREATE TABLE IF NOT EXISTS facts (
    id               SERIAL PRIMARY KEY,
    entity_id        INTEGER REFERENCES entities (id),
    fact_text        TEXT NOT NULL,
    source_file      TEXT,
    agent_id         TEXT,
    created_at       TEXT NOT NULL,
    superseded_by    INTEGER REFERENCES facts (id),
    is_active        INTEGER NOT NULL DEFAULT 1,
    retention_class  TEXT NOT NULL DEFAULT 'standard',
    valid_from       TEXT NOT NULL DEFAULT '',
    valid_until      TEXT NOT NULL DEFAULT '',
    memory_id        TEXT NOT NULL DEFAULT '',
    namespace        TEXT NOT NULL DEFAULT 'global',
    confidence       DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    provenance       TEXT NOT NULL DEFAULT 'unknown',
    actor_id         TEXT NOT NULL DEFAULT '',
    actor_type       TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_facts_entity      ON facts (entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_active      ON facts (is_active);
CREATE INDEX IF NOT EXISTS idx_facts_valid_from  ON facts (valid_from);
CREATE INDEX IF NOT EXISTS idx_facts_memory_id   ON facts (memory_id);
CREATE INDEX IF NOT EXISTS idx_facts_retention   ON facts (retention_class);
CREATE INDEX IF NOT EXISTS idx_facts_namespace   ON facts (namespace);
CREATE INDEX IF NOT EXISTS idx_facts_actor       ON facts (actor_id);


CREATE TABLE IF NOT EXISTS curator_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS memory_versions (
    id              SERIAL PRIMARY KEY,
    memory_id       TEXT NOT NULL,
    version         INTEGER NOT NULL,
    agent_id        TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    text_hash       TEXT NOT NULL,
    operation       TEXT NOT NULL,
    parent_versions TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_memver_memory ON memory_versions (memory_id);
CREATE INDEX IF NOT EXISTS idx_memver_agent  ON memory_versions (agent_id);


-- ---------------------------------------------------------------------------
-- BM25 / full-text search tables
-- (FTS5 virtual tables replaced by tsvector + GIN; exact parity follow-up)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_chunks (
    id           BIGSERIAL PRIMARY KEY,
    qdrant_id    TEXT NOT NULL,
    text         TEXT NOT NULL,
    file_path    TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    agent_id     TEXT NOT NULL DEFAULT '',
    namespace    TEXT NOT NULL DEFAULT '',
    date         TEXT NOT NULL DEFAULT '',
    memory_type  TEXT NOT NULL DEFAULT 'general',
    is_excluded  INTEGER NOT NULL DEFAULT 0,
    actor_id     TEXT NOT NULL DEFAULT '',
    actor_type   TEXT NOT NULL DEFAULT '',
    -- tsvector column for full-text search (Postgres equivalent of FTS5)
    fts_vector   tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    CONSTRAINT memory_chunks_qdrant_unique UNIQUE (qdrant_id)
);

CREATE INDEX IF NOT EXISTS idx_mc_qdrant      ON memory_chunks (qdrant_id);
CREATE INDEX IF NOT EXISTS idx_mc_namespace   ON memory_chunks (namespace);
CREATE INDEX IF NOT EXISTS idx_mc_agent       ON memory_chunks (agent_id);
CREATE INDEX IF NOT EXISTS idx_mc_excluded    ON memory_chunks (is_excluded);
CREATE INDEX IF NOT EXISTS idx_mc_actor       ON memory_chunks (actor_id);
CREATE INDEX IF NOT EXISTS idx_mc_actor_type  ON memory_chunks (actor_type);
-- GIN index accelerates tsvector full-text search (equivalent of FTS5 BM25 index)
CREATE INDEX IF NOT EXISTS idx_mc_fts         ON memory_chunks USING GIN (fts_vector);


CREATE TABLE IF NOT EXISTS memory_points (
    memory_id  TEXT NOT NULL,
    qdrant_id  TEXT NOT NULL,
    point_type TEXT NOT NULL DEFAULT 'primary',
    created_at TEXT NOT NULL,
    PRIMARY KEY (memory_id, qdrant_id)
);

CREATE INDEX IF NOT EXISTS idx_mp_memory ON memory_points (memory_id);
CREATE INDEX IF NOT EXISTS idx_mp_qdrant ON memory_points (qdrant_id);


CREATE TABLE IF NOT EXISTS delete_failures (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    qdrant_ids  TEXT NOT NULL,
    error       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_df_memory  ON delete_failures (memory_id);
CREATE INDEX IF NOT EXISTS idx_df_created ON delete_failures (created_at);


-- ---------------------------------------------------------------------------
-- Transactional outbox (Phase 3)
-- ---------------------------------------------------------------------------

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

CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox (status, created_at);
CREATE INDEX IF NOT EXISTS idx_outbox_event  ON outbox (event_type, status);
-- Drain-loop index: pending/processing rows ordered for backoff
CREATE INDEX IF NOT EXISTS idx_outbox_drain
    ON outbox (status, last_attempt, created_at)
    WHERE status IN ('pending', 'processing');
-- Retention-pruning index
CREATE INDEX IF NOT EXISTS idx_outbox_prune
    ON outbox (status, last_attempt)
    WHERE status = 'applied';


-- ---------------------------------------------------------------------------
-- Needle registry
-- ---------------------------------------------------------------------------

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

CREATE INDEX IF NOT EXISTS idx_needle_token    ON needle_registry (token);
CREATE INDEX IF NOT EXISTS idx_needle_token_ns ON needle_registry (token, namespace);


-- ---------------------------------------------------------------------------
-- Trajectory / outcome tables (from trajectory.py)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS trajectories (
    id               TEXT PRIMARY KEY,
    agent_id         TEXT NOT NULL,
    session_id       TEXT,
    task_description TEXT NOT NULL,
    task_fingerprint TEXT DEFAULT '',
    actions          TEXT NOT NULL DEFAULT '[]',
    outcome          TEXT NOT NULL DEFAULT 'unknown',
    outcome_score    DOUBLE PRECISION,
    memory_ids_used  TEXT DEFAULT '[]',
    created_at       TEXT NOT NULL,
    metadata         TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_traj_agent       ON trajectories (agent_id);
CREATE INDEX IF NOT EXISTS idx_traj_session     ON trajectories (session_id);
CREATE INDEX IF NOT EXISTS idx_traj_outcome     ON trajectories (outcome);
CREATE INDEX IF NOT EXISTS idx_traj_fingerprint ON trajectories (task_fingerprint);


CREATE TABLE IF NOT EXISTS tips (
    id               TEXT PRIMARY KEY,
    trajectory_id    TEXT NOT NULL REFERENCES trajectories (id),
    agent_id         TEXT NOT NULL,
    category         TEXT NOT NULL,
    tip_text         TEXT NOT NULL,
    context          TEXT,
    negative_example TEXT,
    archived         INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    usage_count      INTEGER NOT NULL DEFAULT 0,
    last_used_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_tips_agent    ON tips (agent_id);
CREATE INDEX IF NOT EXISTS idx_tips_category ON tips (category);
CREATE INDEX IF NOT EXISTS idx_tips_archived ON tips (archived);


CREATE TABLE IF NOT EXISTS annotations (
    id               TEXT PRIMARY KEY,
    memory_id        TEXT NOT NULL,
    agent_id         TEXT NOT NULL,
    annotation_type  TEXT NOT NULL DEFAULT 'note',
    content          TEXT NOT NULL,
    quality_score    DOUBLE PRECISION,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ann_memory ON annotations (memory_id);
CREATE INDEX IF NOT EXISTS idx_ann_agent  ON annotations (agent_id);


CREATE TABLE IF NOT EXISTS ratings (
    id        TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    agent_id  TEXT NOT NULL,
    rating    INTEGER NOT NULL,
    context   TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ratings_memory ON ratings (memory_id);
CREATE INDEX IF NOT EXISTS idx_ratings_agent  ON ratings (agent_id);


CREATE TABLE IF NOT EXISTS memory_outcomes (
    id            SERIAL PRIMARY KEY,
    memory_id     TEXT NOT NULL,
    trajectory_id TEXT NOT NULL REFERENCES trajectories (id),
    influence     TEXT NOT NULL DEFAULT 'medium',
    outcome       TEXT NOT NULL,
    outcome_score DOUBLE PRECISION,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mo_memory ON memory_outcomes (memory_id);


-- ---------------------------------------------------------------------------
-- Skills tables (from skills.py)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS skills (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    provider         TEXT NOT NULL DEFAULT '',
    mcp_endpoint     TEXT DEFAULT '',
    current_version  TEXT NOT NULL DEFAULT '0.0.0',
    status           TEXT NOT NULL DEFAULT 'active',
    description      TEXT DEFAULT '',
    registered_by    TEXT NOT NULL,
    registered_at    TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    metadata         TEXT DEFAULT '{}',
    CONSTRAINT skills_name_provider_unique UNIQUE (name, provider)
);

CREATE INDEX IF NOT EXISTS idx_skills_name     ON skills (name);
CREATE INDEX IF NOT EXISTS idx_skills_provider ON skills (provider);
CREATE INDEX IF NOT EXISTS idx_skills_status   ON skills (status);


CREATE TABLE IF NOT EXISTS skill_versions (
    id           SERIAL PRIMARY KEY,
    skill_id     TEXT NOT NULL REFERENCES skills (id),
    version      TEXT NOT NULL,
    changelog    TEXT DEFAULT '',
    breaking_changes TEXT DEFAULT '',
    observed_at  TEXT NOT NULL,
    reported_by  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active',
    CONSTRAINT sv_unique UNIQUE (skill_id, version)
);

CREATE INDEX IF NOT EXISTS idx_sv_skill ON skill_versions (skill_id);


CREATE TABLE IF NOT EXISTS skill_lessons (
    id           TEXT PRIMARY KEY,
    skill_id     TEXT NOT NULL REFERENCES skills (id),
    lesson_type  TEXT NOT NULL DEFAULT 'general',
    title        TEXT NOT NULL,
    content      TEXT NOT NULL,
    skill_version TEXT DEFAULT '',
    agent_id     TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    upvotes      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sl_skill ON skill_lessons (skill_id);
CREATE INDEX IF NOT EXISTS idx_sl_type  ON skill_lessons (lesson_type);


CREATE TABLE IF NOT EXISTS skill_events (
    id           TEXT PRIMARY KEY,
    skill_id     TEXT NOT NULL REFERENCES skills (id),
    agent_id     TEXT NOT NULL,
    event_type   TEXT NOT NULL DEFAULT 'invocation',
    outcome      TEXT NOT NULL DEFAULT 'unknown',
    skill_version TEXT DEFAULT '',
    duration_ms  INTEGER,
    error_message TEXT DEFAULT '',
    trajectory_id TEXT DEFAULT '',
    created_at   TEXT NOT NULL,
    metadata     TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_se_skill   ON skill_events (skill_id);
CREATE INDEX IF NOT EXISTS idx_se_agent   ON skill_events (agent_id);
CREATE INDEX IF NOT EXISTS idx_se_outcome ON skill_events (outcome);


CREATE TABLE IF NOT EXISTS skill_relations (
    id            SERIAL PRIMARY KEY,
    skill_a       TEXT NOT NULL REFERENCES skills (id),
    skill_b       TEXT NOT NULL REFERENCES skills (id),
    relation_type TEXT NOT NULL,
    confidence    DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    evidence      TEXT DEFAULT '',
    created_by    TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sr_a    ON skill_relations (skill_a);
CREATE INDEX IF NOT EXISTS idx_sr_b    ON skill_relations (skill_b);
CREATE INDEX IF NOT EXISTS idx_sr_type ON skill_relations (relation_type);


-- ---------------------------------------------------------------------------
-- Curator queue (from curator_queue.py)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS curator_queue (
    id         TEXT PRIMARY KEY,
    op_type    TEXT NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}',
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    applied_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_cq_status  ON curator_queue (status);
CREATE INDEX IF NOT EXISTS idx_cq_created ON curator_queue (created_at);


-- ---------------------------------------------------------------------------
-- Retrieval logs (from retrieval_log.py)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id              TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    query           TEXT NOT NULL,
    namespace       TEXT DEFAULT '',
    tier            TEXT DEFAULT 'l2',
    memory_type     TEXT DEFAULT '',
    retrieval_trace TEXT NOT NULL,
    result_count    INTEGER DEFAULT 0,
    cache_hit       INTEGER DEFAULT 0,
    duration_ms     INTEGER,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rl_agent   ON retrieval_logs (agent_id);
CREATE INDEX IF NOT EXISTS idx_rl_created ON retrieval_logs (created_at);


-- ---------------------------------------------------------------------------
-- Audit log (from audit.py)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS audit_log (
    id         TEXT PRIMARY KEY,
    timestamp  TEXT NOT NULL,
    agent_id   TEXT NOT NULL,
    action     TEXT NOT NULL,
    memory_id  TEXT,
    namespace  TEXT,
    text_hash  TEXT,
    version    INTEGER,
    metadata   TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log (timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_agent     ON audit_log (agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_memory    ON audit_log (memory_id);
CREATE INDEX IF NOT EXISTS idx_audit_namespace ON audit_log (namespace);


-- ---------------------------------------------------------------------------
-- Hotness scoring (from hotness.py)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_hotness (
    memory_id        TEXT PRIMARY KEY,
    score            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    retrieval_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed    TEXT,
    updated_at       TEXT NOT NULL
);
