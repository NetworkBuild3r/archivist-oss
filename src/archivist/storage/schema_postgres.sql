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
    CONSTRAINT entities_name_unique UNIQUE (name, namespace)
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
    actor_type       TEXT NOT NULL DEFAULT '',
    -- INIT-003/SPEC-002: coach provenance envelope + suppress
    source           TEXT NOT NULL DEFAULT '',
    subject          TEXT NOT NULL DEFAULT '',
    sensitivity      TEXT NOT NULL DEFAULT 'standard',
    purpose          TEXT NOT NULL DEFAULT '',
    statement_kind   TEXT NOT NULL DEFAULT 'user',
    updated_at       TEXT NOT NULL DEFAULT '',
    is_suppressed    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_facts_entity      ON facts (entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_active      ON facts (is_active);
CREATE INDEX IF NOT EXISTS idx_facts_valid_from  ON facts (valid_from);
CREATE INDEX IF NOT EXISTS idx_facts_memory_id   ON facts (memory_id);
CREATE INDEX IF NOT EXISTS idx_facts_retention   ON facts (retention_class);
CREATE INDEX IF NOT EXISTS idx_facts_namespace   ON facts (namespace);
CREATE INDEX IF NOT EXISTS idx_facts_actor       ON facts (actor_id);
-- INIT-003 provenance indexes (subject/purpose/…) are created AFTER the
-- ALTER TABLE migration block below. On existing DBs, CREATE TABLE IF NOT
-- EXISTS is a no-op and those columns are absent until ALTER runs — early
-- CREATE INDEX here would abort the whole script (UndefinedColumnError).
CREATE INDEX IF NOT EXISTS idx_facts_superseded_by ON facts (superseded_by);


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
-- (FTS5 virtual tables replaced by tsvector + GIN; exact parity delivered via
-- fts_vector_simple using the 'simple' text-search config)
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
    importance   REAL NOT NULL DEFAULT 0.5,
    tier_label   TEXT NOT NULL DEFAULT 'l2',
    ttl_at       TEXT,
    decay_rate   REAL NOT NULL DEFAULT 0.0,
    -- INIT-003/SPEC-002: coach provenance envelope + supersede/suppress
    source           TEXT NOT NULL DEFAULT '',
    subject          TEXT NOT NULL DEFAULT '',
    confidence       DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    sensitivity      TEXT NOT NULL DEFAULT 'standard',
    purpose          TEXT NOT NULL DEFAULT '',
    statement_kind   TEXT NOT NULL DEFAULT 'user',
    created_at       TEXT NOT NULL DEFAULT '',
    updated_at       TEXT NOT NULL DEFAULT '',
    supersedes_id    TEXT NOT NULL DEFAULT '',
    is_suppressed    INTEGER NOT NULL DEFAULT 0,
    -- tsvector for stemmed search (Porter/english -- equivalent of FTS5 'porter unicode61')
    fts_vector        tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    -- tsvector for exact/unstemmed search (equivalent of FTS5 'unicode61' / memory_fts_exact)
    fts_vector_simple tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    CONSTRAINT memory_chunks_qdrant_unique UNIQUE (qdrant_id)
);

CREATE INDEX IF NOT EXISTS idx_mc_qdrant        ON memory_chunks (qdrant_id);
CREATE INDEX IF NOT EXISTS idx_mc_namespace     ON memory_chunks (namespace);
CREATE INDEX IF NOT EXISTS idx_mc_agent         ON memory_chunks (agent_id);
CREATE INDEX IF NOT EXISTS idx_mc_excluded      ON memory_chunks (is_excluded);
CREATE INDEX IF NOT EXISTS idx_mc_actor         ON memory_chunks (actor_id);
CREATE INDEX IF NOT EXISTS idx_mc_actor_type    ON memory_chunks (actor_type);
CREATE INDEX IF NOT EXISTS idx_mc_importance    ON memory_chunks (importance DESC);
CREATE INDEX IF NOT EXISTS idx_mc_tier          ON memory_chunks (tier_label);
CREATE INDEX IF NOT EXISTS idx_mc_ttl           ON memory_chunks (ttl_at) WHERE ttl_at IS NOT NULL;
-- INIT-003 provenance indexes created after ALTER migration (see below).
-- GIN indexes accelerate tsvector full-text search (equivalent of FTS5 BM25 index)
CREATE INDEX IF NOT EXISTS idx_mc_fts           ON memory_chunks USING GIN (fts_vector);
CREATE INDEX IF NOT EXISTS idx_mc_fts_simple    ON memory_chunks USING GIN (fts_vector_simple);

-- ---------------------------------------------------------------------------
-- Migration: add fts_vector_simple to existing Postgres DBs created before
-- this column was introduced (Phase 4 deployments).  Idempotent.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks'
           AND column_name = 'fts_vector_simple'
    ) THEN
        ALTER TABLE memory_chunks
            ADD COLUMN fts_vector_simple
                tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED;
    END IF;
END
$$;

-- CREATE INDEX ... IF NOT EXISTS is idempotent, safe to run on existing DBs.
CREATE INDEX IF NOT EXISTS idx_mc_fts_simple ON memory_chunks USING GIN (fts_vector_simple);

-- ---------------------------------------------------------------------------
-- Migration: add Phase 1 answer-finder columns to existing Postgres DBs.
-- Idempotent — each column is added only if it does not already exist.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'importance'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN importance  REAL NOT NULL DEFAULT 0.5;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'tier_label'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN tier_label  TEXT NOT NULL DEFAULT 'l2';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'ttl_at'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN ttl_at      TEXT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'decay_rate'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN decay_rate  REAL NOT NULL DEFAULT 0.0;
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- Migration: INIT-003/SPEC-002 coach provenance + supersede/suppress columns.
-- Previous head: pre-INIT-003 schema (memory_chunks through decay_rate /
-- facts through actor_type; no coach envelope / is_suppressed / supersedes_id).
-- Idempotent — each column is added only if it does not already exist.
-- Rollback: DROP COLUMN for each added column listed below (see completion
-- summary); indexes drop with IF EXISTS counterparts.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    -- facts envelope
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'facts' AND column_name = 'source'
    ) THEN
        ALTER TABLE facts ADD COLUMN source TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'facts' AND column_name = 'subject'
    ) THEN
        ALTER TABLE facts ADD COLUMN subject TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'facts' AND column_name = 'sensitivity'
    ) THEN
        ALTER TABLE facts ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'standard';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'facts' AND column_name = 'purpose'
    ) THEN
        ALTER TABLE facts ADD COLUMN purpose TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'facts' AND column_name = 'statement_kind'
    ) THEN
        ALTER TABLE facts ADD COLUMN statement_kind TEXT NOT NULL DEFAULT 'user';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'facts' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE facts ADD COLUMN updated_at TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'facts' AND column_name = 'is_suppressed'
    ) THEN
        ALTER TABLE facts ADD COLUMN is_suppressed INTEGER NOT NULL DEFAULT 0;
    END IF;

    -- memory_chunks envelope + supersede/suppress
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'source'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN source TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'subject'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN subject TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'confidence'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'sensitivity'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'standard';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'purpose'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN purpose TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'statement_kind'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN statement_kind TEXT NOT NULL DEFAULT 'user';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'created_at'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN created_at TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN updated_at TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'supersedes_id'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN supersedes_id TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_chunks' AND column_name = 'is_suppressed'
    ) THEN
        ALTER TABLE memory_chunks ADD COLUMN is_suppressed INTEGER NOT NULL DEFAULT 0;
    END IF;
END
$$;

-- INIT-003/SPEC-002 pre-rank indexes (safe on fresh + upgraded DBs)
CREATE INDEX IF NOT EXISTS idx_facts_subject     ON facts (subject);
CREATE INDEX IF NOT EXISTS idx_facts_purpose     ON facts (purpose);
CREATE INDEX IF NOT EXISTS idx_facts_sensitivity ON facts (sensitivity);
CREATE INDEX IF NOT EXISTS idx_facts_statement_kind ON facts (statement_kind);
CREATE INDEX IF NOT EXISTS idx_facts_suppressed  ON facts (is_suppressed);
CREATE INDEX IF NOT EXISTS idx_facts_superseded_by ON facts (superseded_by);
CREATE INDEX IF NOT EXISTS idx_mc_subject       ON memory_chunks (subject);
CREATE INDEX IF NOT EXISTS idx_mc_purpose       ON memory_chunks (purpose);
CREATE INDEX IF NOT EXISTS idx_mc_sensitivity   ON memory_chunks (sensitivity);
CREATE INDEX IF NOT EXISTS idx_mc_statement_kind ON memory_chunks (statement_kind);
CREATE INDEX IF NOT EXISTS idx_mc_suppressed    ON memory_chunks (is_suppressed);
CREATE INDEX IF NOT EXISTS idx_mc_supersedes    ON memory_chunks (supersedes_id);
CREATE INDEX IF NOT EXISTS idx_mc_ns_suppressed ON memory_chunks (namespace, is_suppressed);
CREATE INDEX IF NOT EXISTS idx_mc_created_at    ON memory_chunks (created_at);


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
    created_at      TEXT NOT NULL,
    tokens_returned INTEGER,
    tokens_naive    INTEGER,
    savings_pct     DOUBLE PRECISION,
    pack_policy     TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_rl_agent       ON retrieval_logs (agent_id);
CREATE INDEX IF NOT EXISTS idx_rl_created     ON retrieval_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_rl_pack_policy ON retrieval_logs (pack_policy);


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
    updated_at       TEXT NOT NULL,
    importance_signal REAL NOT NULL DEFAULT 0.5
);

-- Migration: add importance_signal to existing Postgres DBs.  Idempotent.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'memory_hotness' AND column_name = 'importance_signal'
    ) THEN
        ALTER TABLE memory_hotness ADD COLUMN importance_signal REAL NOT NULL DEFAULT 0.5;
    END IF;
END
$$;


-- ---------------------------------------------------------------------------
-- Agent-state checkpoints (Phase 7 / INIT-001/SPEC-007)
-- Distinct from memory_chunks.tier_label (L0–L2 delivery tiers; GR-002).
-- payload is JSON text in-row for v1; blob_ref reserved for large payloads later.
-- Rollback: DROP TABLE IF EXISTS agent_checkpoints; disable checkpoint writers.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS agent_checkpoints (
    id                   TEXT PRIMARY KEY,
    agent_id             TEXT NOT NULL,
    session_id           TEXT NOT NULL,
    namespace             TEXT NOT NULL DEFAULT 'global',
    parent_checkpoint_id TEXT REFERENCES agent_checkpoints (id),
    payload              TEXT NOT NULL DEFAULT '{}',
    blob_ref             TEXT,
    metadata             TEXT NOT NULL DEFAULT '{}',
    created_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_agent_session_time
    ON agent_checkpoints (agent_id, session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_checkpoints_session_time
    ON agent_checkpoints (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_checkpoints_namespace
    ON agent_checkpoints (namespace);
CREATE INDEX IF NOT EXISTS idx_checkpoints_parent
    ON agent_checkpoints (parent_checkpoint_id);


-- ---------------------------------------------------------------------------
-- Memory-as-Product scope version lineage (INIT-001/SPEC-009)
-- Tracks namespace/agent-scoped snapshots, forks, and export records.
-- Distinct from per-memory_id memory_versions and from agent_checkpoints.
-- Rollback: DROP TABLE IF EXISTS memory_scope_versions;
-- ---------------------------------------------------------------------------

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
    ON memory_scope_versions (source_namespace, version);
CREATE INDEX IF NOT EXISTS idx_msv_parent
    ON memory_scope_versions (parent_version_id);
CREATE INDEX IF NOT EXISTS idx_msv_archive
    ON memory_scope_versions (archive_id);


-- ---------------------------------------------------------------------------
-- Selective share grants (Phase 10 / INIT-001/SPEC-010)
-- Consensus v1 = explicit accept/reject + audit; extends handoff (GR-003).
-- Rollback: DROP TABLE IF EXISTS memory_share_grants; disable share MCP tools.
-- ---------------------------------------------------------------------------

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
    ON memory_share_grants (recipient_agent_id, status);
CREATE INDEX IF NOT EXISTS idx_share_grants_proposer
    ON memory_share_grants (proposer_agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_share_grants_namespace
    ON memory_share_grants (namespace);
