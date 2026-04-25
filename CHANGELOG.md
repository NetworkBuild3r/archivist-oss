# Changelog

All notable changes to Archivist are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
