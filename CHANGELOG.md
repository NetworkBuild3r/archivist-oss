# Changelog

All notable changes to Archivist are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- `GET /admin/dashboard` and the `archivist_health_dashboard` / `archivist_savings_dashboard` MCP tools
  raised `RuntimeError: SQLitePool is not initialized` on every call against the default SQLite backend.
  `dashboard.py` and `tools_admin.py` imported the connection-pool singleton at module scope, binding to
  the pre-initialization placeholder instead of the real backend startup rebinds `sqlite_pool.pool` to;
  every other consumer already imported it inside the function that uses it. Both files now follow that
  same pattern.

## [2.3.0] - 2026-04-25

### Added — Answer Finder Architecture

**Phase 1 — Tiered memory schema**
- `memory_chunks` gains four new columns: `importance REAL` (0–1 priority signal), `tier_label TEXT` (`l0`/`l1`/`l2`/`ephemeral`), `ttl_at TEXT` (ISO expiry), `decay_rate REAL` (per-day decay). All default to safe values; existing rows get `tier_label='l2'`.
- `memory_hotness` gains `importance_signal REAL` for blended hotness + outcome + quality scoring.
- New indexes: `idx_mc_importance`, `idx_mc_tier`, `idx_mc_ttl` on `memory_chunks`.
- Backward-compatible `ALTER TABLE` migrations applied on both SQLite and Postgres at startup.
- `upsert_fts_chunk` (and `MemoryTransaction.upsert_fts_chunk`) accept `importance` and `tier_label` kwargs.
- `indexer.py` and `tools_storage.py` forward `importance_score` and `tier_label` at write time.

**Phase 2 — Hybrid retrieval + tier-aware token packing**
- FTS search functions (`_search_fts_sqlite/postgres`, `_search_fts_exact_sqlite/postgres`) join `memory_hotness` and weight `ORDER BY` using `(rank * (1 + 0.3 * COALESCE(mh.importance_signal, 0.5)))`.
- New `retrieval/context_packer.py` with `PackedContext` dataclass and `pack_context()` function supporting three policies: `adaptive` (3-pass L0→L1→L2), `l0_first`, and `l2_first`.
- `rlm_retriever.py` replaces the previous greedy truncation loop with `pack_context()`; `over_budget` is now a top-level key in all search responses.
- New config vars: `CONTEXT_PACK_POLICY` (default `adaptive`), `CONTEXT_L0_BUDGET_SHARE` (0.30), `CONTEXT_MIN_FULL_RESULTS` (3).

**Phase 3 — Compression middleware + ephemeral session tier**
- Auto-compress hook in `rlm_retriever.py`: when `AUTO_COMPRESS_ENABLED=true` and results overflow the budget, dropped chunks are summarized via `compact_flat()` and re-injected as a synthetic L1 result.
- New `retrieval/session_store.py`: `SessionStore` — in-process, TTL-scoped, capacity-capped ephemeral memory. `get_session_store()` returns a process-level singleton.
- `archivist_session_end` flushes the `SessionStore`; promoted entries are stored as durable memories when `persist_ephemeral=true`.
- New config vars: `AUTO_COMPRESS_ENABLED` (false), `AUTO_COMPRESS_THRESHOLD` (0.85), `SESSION_STORE_MAX_ENTRIES` (512), `SESSION_STORE_TTL_SECONDS` (3600).

**Phase 4 — `get_relevant_context()` API + multi-agent handoff**
- New `retrieval/context_api.py`: `get_relevant_context()` assembles token-budgeted context in one call; `RelevantContext` dataclass with `answer`, `sources`, `graph_facts`, `tips`, `total_tokens`, `budget_tokens`, `over_budget`, `tier_distribution`, `token_savings_pct`, `provenance`, `pack_policy`.
- `format_context_for_prompt()` renders a `RelevantContext` as a compact system-prompt prefix.
- `create_handoff_packet()` / `receive_handoff_packet()` for typed goal + memory + knowledge snapshot transfer.
- `HandoffPacket` dataclass: `from_agent`, `to_agent`, `session_summary`, `active_goals`, `open_questions`, `key_memory_ids`, `knowledge_snapshot`, `token_count`, `ephemeral_notes`.
- New MCP tools registered in `_registry.py`:
  - `archivist_get_context` — single-call context assembly (replaces `search + tips + context_check` pattern).
  - `archivist_handoff` — create a `HandoffPacket` at session end.
  - `archivist_receive_handoff` — inject a `HandoffPacket` into the receiving agent's ephemeral session.

**Phase 5 — Token savings observability + benchmark suite**
- `retrieval_logs` gains four new columns: `tokens_returned INTEGER`, `tokens_naive INTEGER`, `savings_pct REAL`, `pack_policy TEXT`. Plus `idx_rl_pack_policy` index.
- `log_retrieval()` accepts and stores the new fields (fully backward-compatible — all default to `NULL`/`''`).
- `get_token_savings_stats()` aggregate query function for dashboard.
- `PackedContext.naive_tokens` field — L2-baseline token count carried out of the packer.
- `rlm_retriever.py` wires `tokens_returned`, `tokens_naive`, `savings_pct`, `pack_policy` into both `log_retrieval()` call sites.
- `dashboard.py`: `build_dashboard()` now includes `token_savings`, `tier_distribution`, `hotness_heatmap` sections, backed by `_token_savings_stats()`, `_tier_distribution_stats()`, `_hotness_heatmap()`.
- New MCP tool: `archivist_savings_dashboard` — token savings analytics: avg/min/max savings %, total tokens saved vs naive full-L2 baseline, per-policy breakdown, hotness heatmap of top-N memories.
- New benchmark: `benchmarks/token_efficiency.py` — 50-query benchmark across 3 domains (engineering / operations / agent-usage), 3 packing policies, competitor reference table. Output: `.benchmarks/token_efficiency_<date>.json`.

**Phase 6 — Documentation**
- `README.md` — "Top-Tier Answer Finder" section with API example, capability table, multi-agent handoff walkthrough, and competitor comparison table (Archivist vs Mem0 / Letta / Zep / Graphiti). Version badge updated to v2.3.0. MCP tool count updated to 40. New "Answer Finder (v2.3)" config reference section. Roadmap milestone marked shipped.
- `.env.example` — documented all seven new Answer Finder config variables with comments.
- `docs/DOCKER.md` — new "Answer Finder configuration" section covering packing policy, auto-compression, ephemeral store, observability, and benchmark run instructions.

### Changed

- `context_packer.PackedContext` — added `naive_tokens: int` field (L2-baseline token count for savings calculations).
- `tools_admin.py` — imports `pool` at module level; `_handle_savings_dashboard` handler added.

### Tests

- `tests/unit/retrieval/test_context_api.py` — 22 tests for `RelevantContext`, `get_relevant_context`, `format_context_for_prompt`, and `HandoffPacket` round-trip.
- `tests/unit/retrieval/test_phase5_observability.py` — 14 tests for `PackedContext.naive_tokens`, `log_retrieval` new kwargs, `get_token_savings_stats`, dashboard helper fallbacks, `build_dashboard` new keys, and the savings dashboard handler.

## [2.2.1] - 2026-04-25

### Fixed

- **`_PgCursorProxy.rowcount`** — 12 crash sites (`storage/graph.py`, `lifecycle/memory_lifecycle.py`, `lifecycle/curator.py`, `storage/outbox.py`) that called `cur.rowcount` after a DML statement now work correctly. `AsyncpgConnection.execute()` captures the asyncpg status string (e.g. `"DELETE 5"`) and exposes the parsed count via a new `rowcount` property on `_PgCursorProxy`. SELECT proxies return -1 (DB-API 2.0 convention).
- **`hotness.py` silent Postgres failure** — `get_hotness_scores()`, `batch_update_hotness()`, and `apply_hotness_to_results()` were calling sync `get_db()` which opened a local SQLite file instead of the Postgres pool. All three are now `async def` using `async with pool.read()/write()`. `datetime('now', '-7 days')` replaced with `NOW() - INTERVAL '7 days'` on Postgres; `json_extract(…, '$.result_ids')` replaced with `retrieval_trace::json->>'result_ids'` on Postgres.
- **`audit.py` Postgres compatibility** — `get_audit_trail()` and `get_agent_activity()` converted to `async def` using the pool; callers in `tools_admin.py` updated with `await`.
- **`retrieval_log.py` Postgres compatibility** — `get_retrieval_logs()` and `get_retrieval_stats()` converted to `async def` using the pool. `datetime('now', ?)` window queries replaced with `NOW() - INTERVAL 'N days'` on Postgres. Callers in `tools_admin.py` and `main.py` updated with `await`.
- **`namespace_inventory.py` Postgres compatibility** — `_fetch_inventory()` and `get_inventory()` converted to `async def` using the pool; `_fetch_top_entities()` helper also made async.
- **`COLLATE NOCASE` in `tools_storage.py`** — Two `SELECT id FROM entities WHERE name = ? COLLATE NOCASE` queries (lines 1049, 1125) are now conditionally guarded: `COLLATE NOCASE` is omitted when `_is_postgres()` since `entities.name` is `CITEXT`.
- **`skill_relations` conflict target** — Added `"skill_relations": ["skill_a", "skill_b", "relation_type"]` to `_conflict_targets` in `asyncpg_backend.py` for correct `ON CONFLICT DO UPDATE` generation if the table is ever upserted.

### Changed

- **Test fixture schema sync** — `tests/fixtures/schema.py` `_SCHEMA_SQL` updated to match `schema_postgres.sql`: `memory_hotness` gains `retrieval_count`/`last_accessed`; `annotations` renamed `annotation` → `content`, added `annotation_type`/`quality_score`; `ratings` gains `context`; `tips` gains `context`/`usage_count`/`last_used_at`. Added missing tables `curator_queue`, `retrieval_logs`, `audit_log`, `memory_outcomes`.
- **4 new rowcount unit tests** in `tests/unit/storage/test_backends.py` verifying DML `DELETE`/`INSERT`/`UPDATE` rowcounts and the SELECT proxy default.
- **5 new dual-backend integration tests** in `tests/integration/storage/test_dual_backend.py` covering DML rowcount, audit log round-trip, retrieval log round-trip, and hotness batch update.

## [2.2.0] - 2026-04-24

### Added

- **PostgreSQL first-class backend** (`GRAPH_BACKEND=postgres`) — all storage hot paths, schema init, FTS, and backup tools now work equally on SQLite and PostgreSQL. Set `GRAPH_BACKEND=postgres` + `DATABASE_URL` to switch; SQLite remains the default and is unaffected.
- **`docker-compose.postgres.yml`** — Compose overlay that starts a `postgres:16-alpine` service, sets `GRAPH_BACKEND=postgres`, and wires `DATABASE_URL` automatically. Provides `PG_USER/PASSWORD/DB/PORT/POOL_MIN/POOL_MAX` variables.
- **`pg_dump` / `pg_restore` backup** (`backup_manager.py`) — `archivist_backup` and `archivist_restore` detect the active backend and use `pg_dump --format=custom` / `pg_restore --clean` for Postgres. Snapshot manifests now include `graph_backend`.
- **`_translate_sql()`** in `asyncpg_backend.py` — transparent SQLite→Postgres SQL rewriting: `INSERT OR IGNORE` → `ON CONFLICT DO NOTHING`, `INSERT OR REPLACE` → `ON CONFLICT DO UPDATE SET`, `COLLATE NOCASE` stripped (Postgres uses `CITEXT`).
- **`fetchval()` on both connection wrappers** — `AsyncpgConnection` and `_WrappedSQLiteConn` both expose `fetchval()` for scalar-result queries (e.g. `INSERT … RETURNING id`).
- **Postgres-aware schema init** — `init_schema_async()` loads `schema_postgres.sql` for Postgres; `init_schema()`, `_migrate_schema()`, `_migrate_entity_unique_constraint()`, `_init_fts5()`, and `schema_guard()` all skip when Postgres is active.
- **`tests/integration/storage/test_dual_backend.py`** — parametrized dual-backend integration tests that run the canonical graph operation suite against both SQLite (always) and Postgres (skipped when `POSTGRES_TEST_DSN` is unset).
- **16 new SQL-translation unit tests** in `tests/unit/storage/test_backends.py` covering all `_translate_sql()` transformations and `fetchval()` contracts.
- **Postgres section in `docs/DOCKER.md`** — compose quickstart, external-Postgres instructions, schema comparison table, backup notes, and integration-test instructions.
- **`.env.example` Postgres variables** — documented `GRAPH_BACKEND`, `DATABASE_URL`, `PG_POOL_MIN/MAX`, and the `docker-compose.postgres.yml` helper variables.

### Changed

- All formerly-synchronous `get_db()` callers across `skills.py`, `trajectory.py`, `curator_queue.py`, `curator.py`, `metrics.py`, `dashboard.py`, and `compressed_index.py` are now `async def` using `async with pool.read()/write()`.
- `graph.py` `upsert_entity()` and `add_fact()` use `conn.fetchval("… RETURNING id")` on Postgres and `cur.lastrowid` on SQLite.
- `README.md` — added "Dual database backends" to the features table, updated the scaling FAQ, added v2.2 release note, and marked the PostgreSQL roadmap milestone as shipped.

## [2.1.0] - 2026-04-17

### Added

- **Transactional outbox** — SQLite `outbox` table plus background `OutboxProcessor` to apply Qdrant operations idempotently after an atomic local commit. Config: `OUTBOX_ENABLED`, `OUTBOX_DRAIN_INTERVAL`, `OUTBOX_BATCH_SIZE`, `OUTBOX_MAX_RETRIES`, `OUTBOX_RETENTION_DAYS` (see `.env.example`).
- **`MemoryTransaction`** — `async with MemoryTransaction()` wraps one `pool.write()` transaction; `enqueue_qdrant_*` methods flush outbox rows in the same commit as SQLite artefacts.
- **`VectorBackend` / `QdrantVectorBackend`** — Thin protocols in `src/archivist/storage/backends.py` for testability and a future non-Qdrant vector store.
- **Connection-passing shims** — `upsert_fts_chunk`, `register_needle_tokens`, `upsert_entity`, and `add_fact` accept optional `conn=` so graph writes join an open transaction without re-entering the async write lock (deadlock-safe).
- **`tests/test_outbox.py`** — Unit, integration, and chaos coverage for the outbox and transaction path.
- **`tests/qa/`** — Dedicated QA package: `test_storage_transactional.py` (atomicity) and `test_chaos_fault_injection.py` (fault injection). Documented in [`tests/qa/README.md`](tests/qa/README.md) and [`docs/QA.md`](docs/QA.md).
- **[`docs/rearchitect_storage_phase3.md`](docs/rearchitect_storage_phase3.md)** — Reference architecture for Phase 3 + 3.5 storage work.

### Changed

- **Write paths** — `archivist_store` (`tools_storage._handle_store`), `index_file`, `merge_memories`, and `delete_memory_complete` (when `OUTBOX_ENABLED=true`) commit FTS5, needle registry, `memory_points`, entity/facts (where applicable), and outbox events in a single transaction where documented.
- **BM25** — `search_bm25` is `async def` and correctly awaits async FTS helpers; `rlm_retriever` updated accordingly.
- **`archivist_delete`** — `_rbac_gate` now receives the `action` argument (write path).

### Fixed

- **Cross-store orphan classes** — Eliminates the prior window where Qdrant could succeed while SQLite artefacts failed (or vice versa) on primary store, indexer, merge, and delete paths when the outbox path is enabled.

## [2.0.1] - 2026-04-17

### Fixed

- **Async write-path data loss** — All graph writes (`upsert_entity`, `add_fact`, `upsert_fts_chunk`, `register_needle_tokens`, `register_memory_points_batch`) in `tools_storage._handle_store` were fire-and-forget coroutines; they are now properly awaited.
- **`merge.py` correctness** — Merged memory now correctly writes FTS chunks, needle tokens, and memory-point batch registrations for the merged ID; Qdrant upsert uses `collection_for(ns)` instead of a hardcoded collection; `record_version` is awaited; originals deletion loop is guarded against `PartialDeletionError`.
- **`sweep_orphans` async** — `sweep_orphans()` is now a proper `async def` using `pool.read()`/`pool.write()`; `delete_fts_chunks_batch` and `delete_needle_tokens_batch` are awaited; `curator.py` call site updated.
- **`retrieval_log.log_retrieval` sync block removed** — Converted to `async def` using `pool.write()`; all callers in `rlm_retriever.py` now await it.
- **`cascade._qdrant_delete` async** — Made async so `log_delete_failure` can be awaited on the failure path.
- **`memory_lifecycle` threading** — Replaced `asyncio.to_thread` wrappers for `_qdrant_delete` and `set_fts_excluded_batch` with direct awaits; removed invalid `conn.close()` inside an `async with pool.write()` block.
- **Dockerfile CMD** — Changed from `python archivist/app/main.py` to `python3 -m archivist.app.main`; the previous form ran the file without the package on `sys.path`, breaking all relative imports.

### Added

- `tests/test_missing_awaits.py` — regression tests asserting all async write-path calls are properly awaited.
- `tests/test_merge_consistency.py` — regression tests for merge correctness (FTS/needle/memory-point propagation, namespace-aware collection routing).

## [2.0.0] - 2026-04-17

### Added

- **Official 2.0 release** — Version **2.0.0** (`v2.0.0` tag): package layout under `src/archivist/`, documented in [`docs/RELEASE_NOTES_v2.0.md`](docs/RELEASE_NOTES_v2.0.md).
- **Pipeline benchmark snapshot (Phase 5)** — Results for `clean_reranker` vs `vector_plus_synth` on `memory_scale=small` (108 queries per variant) documented in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md); reproducible JSON at `.benchmarks/phase5_semantic_chunking.json` when running the harness locally.

### Changed

- **`archivist` package layout** — Production code organized into `core/`, `storage/`, `lifecycle/`, `retrieval/`, `write/`, `features/`, `utils/`, and `app/`; top-level `src/*.py` shims preserve backward-compatible imports.
- **Mypy** — `[tool.mypy]` in `pyproject.toml` sets `mypy_path = "src"` and `explicit_package_bases = true`, with targeted `ignore_missing_imports` for `xgboost` and `yaml` stubs.
- **Version strings** — Health endpoint, startup log, and backup manifest `archivist_version` set to **2.0.0**.

## [1.12.0] - 2026-04-14

### Added

- **Semantic chunking** — `chunk_text_semantic()` for markdown-aware parent chunks; `CHUNKING_STRATEGY` (`semantic` \| `fixed`, default `semantic`) in config; hierarchical indexing passes the strategy through from `indexer.py`.
- **Synthetic question generation** (`synthetic_questions.py`) for optional indexing-time enrichment (feature-flagged).
- **Pipeline benchmark artifacts in-repo** — `benchmarks/pipeline/`, `benchmarks/scripts/`, `benchmarks/academic/` (adapters), `benchmarks/fixtures/questions.json`, and `benchmarks/fixtures/corpus_small/` (including a long-doc sample with **synthetic** hostnames/paths for public use). Large generated corpora remain gitignored.
- **Storage gauges** (background refresh): `archivist_total_memories` (label `namespace`), `archivist_sqlite_size_bytes`, `archivist_qdrant_vectors_total` (label `collection`), `archivist_qdrant_available`, `archivist_sqlite_available` (0/1). Interval: `METRICS_COLLECT_INTERVAL_SECONDS` (default 60s, minimum 5s in the loop).
- **Search result count histogram**: `archivist_search_results` (label `namespace`) — length of the `sources` list returned per search.
- **`METRICS_AUTH_EXEMPT`** — when `true`, `GET /metrics` skips API key auth so Prometheus can scrape without the MCP key.

### Changed

- **`METRICS_ENABLED=false`** now disables all metric recording (no-ops) and returns **HTTP 404** on `GET /metrics`.
- Renamed metrics to the `archivist_*` prefix: `embed_cache_hits_total` → `archivist_embed_cache_hit_total`, `embed_cache_misses_total` → `archivist_embed_cache_miss_total`, `hyde_duration_ms` → `archivist_hyde_duration_ms`, `reverse_hyde_duration_ms` → `archivist_reverse_hyde_duration_ms`, `query_expansion_duration_ms` → `archivist_query_expansion_duration_ms`.
- Histogram rendering supports a second bucket layout for `archivist_search_results` (counts, not milliseconds).
- Version string bumped to **1.12.0** in health endpoint and startup log.

### Removed

- **`TECH_DEBT.md`** — superseded by `docs/ARCHITECTURE.md` and issue-tracker planning; cascade notes folded into README + architecture docs.

## [1.11.0] - 2026-04-10

### Breaking Changes

- **`delete_memory_complete()` now calls `client.scroll()` during cascade.** Any code or test that mocks `qdrant_client` and calls `delete_memory_complete` must add `client.scroll.return_value = ([], None)` to the mock fixture.
- **Embedding cache returns `tuple` instead of `list`.** Code that previously mutated embedding vectors returned by `embed_text()` (e.g., `vec.append(...)`) will fail. Use `list(vec)` if mutation is required.

### Fixed

- **Micro-chunk FTS5 orphan on delete** — `delete_memory_complete()` previously only deleted the primary memory's FTS5 and needle registry entries. Micro-chunk and reverse HyDE child points each have their own `qdrant_id` values that were never cleaned up from `memory_chunks`, `memory_fts`, `memory_fts_exact`, and `needle_registry`. The function now enumerates all child point IDs via `client.scroll()` before deletion and cleans up FTS5 + registry rows for each.
- **Indexer `asyncio.gather` missing `return_exceptions=True`** — Parallel reverse HyDE generation in `indexer.py` used `asyncio.gather()` without `return_exceptions=True`. While internal try/except guards prevented most failures, removing or weakening those guards could cause a single LLM error to cancel all concurrent calls. Now uses `return_exceptions=True` with explicit error logging for each failed batch.
- **FTS5 IDF dilution** — All `upsert_fts_chunk` calls in `tools_storage.py` and `indexer.py` now pass raw text instead of augmented text (which included `[Agent: X | File: Y]` metadata headers), preventing BM25 scoring artifacts.
- **Needle registry error handling** — `delete_needle_tokens_by_memory()` now logs exceptions instead of using bare `except: pass`.

### Added

- **Needle retrieval v2 pipeline** — Deterministic needle registry, reverse HyDE (write-time questions), API micro-chunking, entity auto-extraction, contextual augmentation, multi-query expansion, uniform `ResultCandidate` types, RRF fusion, embedding LRU+TTL cache.
- **`memory_lifecycle.py`** — Unified `delete_memory_complete()` and `archive_memory_complete()` functions that handle all 7 artifact types (Qdrant primary + reverse HyDE + micro-chunks, FTS5 entries, needle registry, entity facts).
- **`result_types.py`** — `RetrievalSource` enum and `ResultCandidate` dataclass with factory methods (`from_qdrant_payload`, `from_registry_hit`, `from_bm25_hit`).
- **`contextual_augment.py`** — `augment_chunk()` for embedding enrichment and `strip_augmentation_header()` for FTS5 raw text extraction.
- **Structured observability** — `store_pipeline.complete` and `retrieval_pipeline.complete` structured log events. Feature flag summary logged at startup.
- **358 unit tests** (up from ~300) covering all new modules and fix paths.

### Changed

- Embedding cache stores vectors as immutable `tuple[float, ...]` instead of `list[float]`, eliminating defensive copies on cache hits.
- Registry hits now flow through RRF → threshold → reranker like all other sources (removed hardcoded `score=0.95` bypass).
- `pre_extract()` call consolidated to single invocation per store/index path (result reused for entities, thought_type, augmentation hints).
- `NEEDLE_PATTERNS` consolidated to `chunking.py` as single source of truth.
- Version string bumped to 1.11.0 in health endpoint and startup log.
