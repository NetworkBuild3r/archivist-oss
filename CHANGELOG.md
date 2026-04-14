# Changelog

All notable changes to Archivist are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
