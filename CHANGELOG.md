# Changelog

All notable changes to Archivist are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

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
