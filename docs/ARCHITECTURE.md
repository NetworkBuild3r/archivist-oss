# Archivist architecture

## Overview

Archivist is a memory service for multi-agent fleets. It combines:

| Layer | Technology | Role |
|-------|------------|------|
| **Vectors** | Qdrant | Semantic search over hierarchical chunks |
| **Structured state (default)** | SQLite (aiosqlite pool) | Knowledge graph, FTS5 BM25, needle registry, audit, trajectories, skills, outbox |
| **Structured state (optional)** | PostgreSQL 14+ via `asyncpg` | Drop-in replacement for SQLite graph/FTS when `GRAPH_BACKEND=postgres`; tsvector/tsquery replaces FTS5 |
| **Durability boundary** | Transactional outbox + `MemoryTransaction` | Atomic commit of SQLite artefacts with queued Qdrant work (Phase 3 + 3.5) |
| **Source of truth (optional)** | File system | Markdown under `MEMORY_ROOT` for ingestion |

Retrieval uses the **RLM** (recursive layered memory) pipeline in `rlm_retriever.py`. Writes on hot paths use **`MemoryTransaction`** so FTS5, needle rows, `memory_points`, entity/facts (where applicable), and outbox rows commit together when `OUTBOX_ENABLED=true`. See [`docs/rearchitect_storage_phase3.md`](rearchitect_storage_phase3.md) for the full design.

---

## Storage transaction model (Phase 3.5)

Qdrant and SQLite are different stores; true distributed transactions are not available. Archivist implements a **transactional outbox** in SQLite:

1. Application code enters **`async with MemoryTransaction()`**, which acquires **one** `pool.write()` lock and exposes `txn.conn`.
2. All SQLite mutations for that operation run on that connection—either raw SQL via `txn.execute` / `txn.executemany` / `txn.fetchall`, or via **shims** that forward to `graph.py` helpers with **`conn=txn.conn`** (`upsert_fts_chunk`, `register_needle_tokens`, `upsert_entity`, `add_fact`).
3. **`enqueue_qdrant_upsert`**, **`enqueue_qdrant_delete`**, etc. append to an in-memory list. They do **not** call Qdrant inside the transaction.
4. On successful exit, **`_flush_events()`** inserts one row per event into the **`outbox`** table in the **same** commit as the graph/FTS/needle writes.
5. If any exception propagates, the transaction rolls back: **no** partial SQLite state and **no** outbox rows.
6. A background **`OutboxProcessor`** drains `pending` rows, applies them to **`QdrantVectorBackend`** (idempotent), then marks rows done or dead after retries.

**Why `conn=` matters:** `pool.write()` is backed by a non-reentrant `asyncio.Lock`. Helpers that call `pool.write()` internally would deadlock if invoked from inside an open `MemoryTransaction`. Passing `conn=` avoids nested lock acquisition.

**Default:** `OUTBOX_ENABLED=false` keeps legacy behaviour: Qdrant writes run inline in handlers; enqueue methods are no-ops.

```mermaid
sequenceDiagram
    participant H as Handler / indexer / merge
    participant MT as MemoryTransaction
    participant SQL as SQLite (WAL)
    participant OB as outbox table
    participant OP as OutboxProcessor
    participant Q as Qdrant

    H->>MT: async with MemoryTransaction()
    MT->>SQL: begin (pool.write)
    H->>MT: FTS / needle / memory_points / facts
    H->>MT: enqueue_qdrant_upsert(...)
    MT->>OB: INSERT outbox rows (same commit)
    MT->>SQL: commit
    Note over H,Q: Later, async
    OP->>OB: SELECT pending FOR UPDATE
    OP->>Q: upsert / delete (to_thread)
    OP->>OB: status = done
```

```mermaid
flowchart LR
    subgraph write["Write path (OUTBOX_ENABLED=true)"]
        A[MemoryTransaction] --> B[SQLite artefacts]
        A --> C[outbox INSERT]
        B --> D[(Single commit)]
        C --> D
    end
    subgraph drain["Drain loop"]
        E[OutboxProcessor] --> F[QdrantVectorBackend]
        E --> G[Mark done / retry / dead]
    end
    D -.->|eventually| E
```

---

## Data flow

### Ingestion (file watcher → index)

```mermaid
flowchart TB
    FS[Markdown files MEMORY_ROOT]
    W[File watcher]
    CH[Chunker indexer.py]
    E[Embed API]
    MT[MemoryTransaction]
    SQL[(SQLite FTS needle memory_points facts)]
    OB[outbox]
    Q[(Qdrant)]

    FS --> W --> CH --> E
    CH --> MT
    MT --> SQL
    MT --> OB
    OB -.->|OutboxProcessor| Q
```

When `OUTBOX_ENABLED=false`, the indexer still uses SQLite transaction boundaries where implemented; Qdrant upsert runs inline after SQLite work (same as pre–Phase 3 behaviour for that flag).

### Retrieval (RLM pipeline)

Query execution is unchanged at a high level: coarse vector search → BM25 fusion → graph augmentation → dedupe → decay → hotness → threshold → optional rerank → parent enrichment → optional LLM refinement/synthesis. See the README “How It Works” section for the stage list; detailed stage notes remain in historical release notes below.

---

## Module map (production layout)

Code lives under `src/archivist/`.

| Area | Modules |
|------|---------|
| **App** | `app/main.py`, `app/mcp_server.py`, `app/handlers/*.py` — MCP tools, REST, startup |
| **Storage** | `storage/graph.py`, `storage/sqlite_pool.py`, `storage/fts_search.py`, `storage/transaction.py`, `storage/outbox.py`, `storage/backends.py` (protocols), `storage/backend_factory.py` (backend dispatch), `storage/asyncpg_backend.py` (Postgres impl), `storage/collection_router.py` |
| **Retrieval** | `retrieval/rlm_retriever.py`, `retrieval/graph_retrieval.py`, `retrieval/reranker.py` |
| **Write** | `write/indexer.py`, `write/chunking.py` |
| **Lifecycle** | `lifecycle/memory_lifecycle.py`, `lifecycle/merge.py`, `lifecycle/cascade.py`, `lifecycle/curator_queue.py` |
| **Features** | `features/embeddings.py`, `features/llm.py`, … |
| **Core** | `core/config.py` (`ArchivistSettings`), `core/rbac.py`, `core/audit.py`, `core/metrics.py`, `core/health.py` |

---

## Storage schema (summary)

### Qdrant payload fields

| Field | Type | Purpose |
|-------|------|---------|
| `agent_id` | keyword | Source agent |
| `file_path` | keyword | Relative path from MEMORY_ROOT |
| `file_type` | keyword | daily, durable, system, explicit, merged |
| `team` | keyword | Agent's team |
| `date` | keyword | ISO date |
| `namespace` | keyword | RBAC namespace |
| `text` | text | Chunk content (L2) |
| `l0`, `l1` | text | Tiered summaries |
| `chunk_index` | integer | Position in source file |
| `parent_id` | keyword | Parent chunk reference |
| `is_parent` | bool | Parent/child flag |
| `version` | integer | Monotonic version |
| `importance_score` | float | 0.0–1.0 retention score |
| `ttl_expires_at` | integer | Unix timestamp for expiry |
| `checksum` | keyword | Content hash for dedup |

Additional fields (`source_memory_id`, `is_reverse_hyde`, `thought_type`, actor provenance, etc.) are documented in [`docs/REFERENCE.md`](REFERENCE.md) and storage skills.

### SQLite (representative)

- **Graph** — `entities`, `relationships`, `facts`
- **Search** — `memory_chunks` (and FTS5 virtual tables), `needle_registry`
- **Routing** — `memory_points` (primary / micro_chunk / reverse_hyde linkage)
- **Outbox** — `outbox` (`id`, `event_type`, `payload`, `status`, `retry_count`, …)
- **Operations** — `audit_log`, `memory_versions`, `curator_queue`, `retrieval_logs`, `trajectories`, `skills`, …

Full table inventory: see the Archivist storage-schema skill (`.cursor/skills/archivist-storage-schema/SKILL.md`) when working on schema changes.

---

## PostgreSQL backend (v2.1)

Archivist supports a pluggable graph/FTS storage layer. By default it uses SQLite; PostgreSQL 14+ is available as a drop-in replacement for all graph, FTS, and structured-state operations.

### Activation

Set two environment variables:

```bash
GRAPH_BACKEND=postgres
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

No other changes are needed. The MCP tool signatures, retrieval pipeline, and write paths are identical for both backends.

### Schema initialisation

`AsyncpgGraphBackend.initialize()` auto-runs `src/archivist/storage/schema_postgres.sql` on first connection. The schema creates all tables, FTS indexes (`tsvector` columns, GIN indexes), partial indexes, and the `fts_vector_simple` column used by the simple-dictionary search path. This is idempotent — re-running on an existing database is safe.

### SQL compatibility layer

All SQL in the codebase uses SQLite `?` placeholders. `AsyncpgConnection._translate_sql()` converts `?` → `$1, $2, …` at call time, so no SQL strings need to change when switching backends.

### FTS implementation

| Backend | Full-text search | Exact search |
|---------|-----------------|--------------|
| SQLite | FTS5 virtual tables, BM25 ranking | `LIKE`-based exact match |
| PostgreSQL | `tsvector`/`tsquery`, `ts_rank_cd` | `to_tsvector('simple', …) @@ …` |

Both backends produce results in the same format; the retrieval pipeline is unaware of which backend is active.

### Connection pool

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_POOL_MIN` | `5` | Minimum pool size |
| `PG_POOL_MAX` | `20` | Maximum pool size |

Pool metrics are exposed on `GET /metrics`: `archivist_pg_pool_acquire_ms`, `archivist_pg_pool_query_ms`, `archivist_pg_pool_errors_total`, `archivist_pg_pool_size`.

### Health registration

On successful `initialize()`, the `"postgres"` subsystem is registered in `archivist.core.health` as healthy with its initialisation latency. `GET /health` will include it in the `subsystems` map. On `close()`, it is marked unhealthy.

---

## Observability (v2.1)

### `/health` endpoint

`GET /health` returns a structured JSON document aggregating the live status of all registered subsystems:

```json
{
  "status": "healthy",
  "service": "archivist",
  "version": "2.1.0",
  "timestamp": "2026-04-18T20:00:00+00:00",
  "subsystems": {
    "postgres": { "healthy": true, "detail": "", "since": "…", "latency_ms": 42.3 },
    "qdrant":   { "healthy": true, "detail": "", "since": "…", "latency_ms": 0.0 }
  }
}
```

- **HTTP 200** — all subsystems healthy (or none registered).
- **HTTP 503** — one or more subsystems unhealthy. The container is alive and serving; 503 is a *degraded* state, not a crash. Kubernetes liveness probes should tolerate 503 (set `failureThreshold ≥ 3`).

`GET /health` is always auth-exempt.

### `/debug/config` endpoint

`GET /debug/config` (auth required) returns a read-only snapshot of non-secret configuration and feature-flag states:

```json
{
  "graph_backend": "postgres",
  "metrics_enabled": true,
  "bm25_enabled": true,
  "outbox_enabled": false,
  "reranker_enabled": false,
  "curator_interval_minutes": 30,
  "qdrant_collection": "archivist_memories",
  "vector_dim": 1024,
  "timestamp": "…"
}
```

### Prometheus metrics (`GET /metrics`)

All metric names use the `archivist_` prefix. New in v2.1:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `archivist_pg_pool_acquire_ms` | histogram | — | Pool connection acquire latency |
| `archivist_pg_pool_query_ms` | histogram | — | Per-query execution latency |
| `archivist_pg_pool_errors_total` | counter | — | Pool + query errors |
| `archivist_pg_pool_size` | gauge | — | Current pool size |
| `archivist_fts_search_duration_ms` | histogram | `backend` | FTS search duration (`sqlite`, `postgres`, `bm25`) |
| `archivist_fts_search_total` | counter | `backend` | FTS search call count |
| `archivist_fts_upsert_total` | counter | `backend` | FTS upsert call count |
| `archivist_fts_upsert_errors_total` | counter | `backend` | FTS upsert failures |
| `archivist_index_duration_ms` | histogram | — | Full write-pipeline indexing duration |
| `archivist_curator_extract_duration_ms` | histogram | — | Curator extract phase duration |
| `archivist_curator_decay_duration_ms` | histogram | — | Curator decay phase duration |
| `archivist_subsystem_healthy` | gauge | `subsystem` | 1.0 = healthy, 0.0 = unhealthy per subsystem |

Set `METRICS_AUTH_EXEMPT=true` to allow Prometheus to scrape `/metrics` without an API key.

---

## Further reading

| Document | Content |
|----------|---------|
| [`rearchitect_storage_phase3.md`](rearchitect_storage_phase3.md) | Outbox design, failure modes, config, tests |
| [`MIGRATION.md`](MIGRATION.md) | v2.1 upgrade guide, SQLite→Postgres migration, `/health` 503 behaviour change |
| [`QA.md`](QA.md) | Test commands including `tests/qa/` |
| [`ROADMAP.md`](ROADMAP.md) | Product direction |

---

## Historical operational notes

The following sections record behaviour and operational guidance by release (condensed from earlier architecture docs). New deployments should rely on the storage transaction model above for cross-store write semantics when `OUTBOX_ENABLED=true`.

### v0.4.0

- **HTTP auth** — Set `ARCHIVIST_API_KEY`; clients send `Authorization: Bearer <key>` or `X-API-Key`. `GET /health` is never authenticated (Kubernetes probes).
- **SQLite writes** — Graph mutations, audit inserts, version inserts, and curator fact decay take a process-wide write lock (legacy `GRAPH_WRITE_LOCK`; production uses `sqlite_pool` with async lock).
- **Retrieval trace** — `archivist_search` JSON includes `retrieval_trace`: coarse hit counts, dedupe/threshold/rerank stages, and rerank settings.
- **Store conflicts** — Before `archivist_store`, optional Qdrant similarity vs *other* agents in the same namespace; block or allow via env + `force_skip_conflict_check`.
- **Explicit store IDs** — Qdrant point IDs for explicit stores are UUIDs (not content hashes).

### v0.5.0

- **Tiered context** — On ingest, parent chunks get auto-generated L0 and L1 summaries via LLM. Controlled by `TIERED_CONTEXT_ENABLED`.
- **Graph-augmented retrieval** — Entity mentions matched against the knowledge graph; facts merged into vector results with `GRAPH_RETRIEVAL_WEIGHT`.
- **Temporal decay** — Results weighted by recency; `TEMPORAL_DECAY_HALFLIFE_DAYS=0` disables.
- **Context budget** — `max_tokens`, `tier` (l0/l1/l2).
- **Progressive dereference** — `archivist_deref` for L2 drill-down.
- **Compressed index** — `archivist_index` navigational summary.
- **Contradiction surfacing** — `archivist_contradictions`.
- **Date range filters** — `date_from` / `date_to` on search.

### v0.6.0

- **Trajectory logging** — `archivist_log_trajectory`, outcome-aware retrieval, annotations, ratings, tips, `archivist_session_end`.

### v0.7.0

- **Skill registry** — SQLite tables for skills, versions, lessons, events; skill MCP tools.

### v0.8.0

- **Three-layer hierarchy** — Session → hot cache → long-term (Qdrant + SQLite).
- **`archivist://` URIs** — Parse and resolve.
- **Retrieval trajectory logging** — `retrieval_logs` SQLite table.
- **Consistency config** — `DEFAULT_CONSISTENCY` env var.

### v0.9.0

- **Prometheus metrics** — `GET /metrics`.
- **Webhooks** — HTTP POST on `memory_store`, `memory_conflict`, `skill_event`.
- **Health dashboard** — Aggregated stats and batch heuristic.

### v1.0.0

- **Write-ahead curator queue** — `curator_queue.py` enqueue/drain.
- **LLM-adjudicated dedup** — On store when similarity exceeds threshold.
- **Tip consolidation** — Trajectory tips clustered and merged.
- **`archivist_compress`** — Archives originals; structured summary option.
- **Hotness scoring** — Blended into RLM results.
- **Skill relation graph** — `skill_relations` table.

### v1.0.1

- **MCP server refactor** — Handlers split by domain under `app/handlers/`.
- **Background tasks** — Tracked with done callbacks.

### v1.1.0

- **Token counting** — `tokenizer.py` with tiktoken + fallback.
- **Context manager** — `context_manager.py` for budget checks.
- **Structured compaction** — `compaction.py` Goal/Progress/Decisions/Next Steps.
- **`archivist_context_check`** MCP tool.
- **`archivist_compress`** — `format` flat/structured, `previous_summary`.
