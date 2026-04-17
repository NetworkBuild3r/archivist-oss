# Archivist 2.0.1 — Storage Hardening Patch

**Release date:** 2026-04-17

## Summary

This patch fixes several silent async correctness bugs in the write path, merge pipeline, and background sweep that could cause data loss or inconsistent state under normal operation. It also corrects the Docker image entry point.

## Fixes

### Async write-path data loss (critical)
`tools_storage._handle_store` was calling async graph functions — `upsert_entity`, `add_fact`, `upsert_fts_chunk`, `register_needle_tokens`, `register_memory_points_batch` — without `await`. The coroutines were silently discarded, meaning graph, FTS, and needle-token indexes were never written. All call sites are now awaited.

### `merge.py` multi-index correctness
After writing the merged Qdrant point, the merge path now also writes FTS chunks, needle tokens, and memory-point batch registrations for the merged memory ID. The Qdrant upsert now uses the namespace-aware `collection_for(ns)` helper instead of a hardcoded collection name. `record_version` is awaited. The originals deletion loop is guarded with `try/except PartialDeletionError` so a partial failure logs a warning but does not abort the merge.

### `sweep_orphans` blocking SQLite
`sweep_orphans()` was synchronous and called `delete_fts_chunks_batch` / `delete_needle_tokens_batch` without `await`. The function is now `async def`, uses `pool.read()`/`pool.write()` throughout, and its call site in `curator.py` awaits it directly.

### `retrieval_log` hot-path blocking
`log_retrieval()` was synchronous and used a blocking `get_db()` call on every search request. It is now `async def` backed by `pool.write()`, and all `rlm_retriever.py` call sites await it.

### `cascade._qdrant_delete` and `memory_lifecycle` threading
`_qdrant_delete` is now `async def` so `log_delete_failure` can be awaited on the failure path. `memory_lifecycle` drops the now-unnecessary `asyncio.to_thread` wrappers for `_qdrant_delete` and `set_fts_excluded_batch`, and removes an invalid synchronous `conn.close()` inside an `async with pool.write()` block.

### Docker CMD entry point
The Dockerfile `CMD` was `python archivist/app/main.py`. Because `COPY src/ ./` places the package at `/app/archivist/`, this ran the file without the `archivist` package on `sys.path`, breaking all package-relative imports at startup. Changed to `python3 -m archivist.app.main` with `WORKDIR /app`.

## New tests
- `tests/test_missing_awaits.py` — asserts async write-path calls are awaited
- `tests/test_merge_consistency.py` — asserts merge propagates FTS/needle/memory-points and uses namespace-correct collections

## Upgrading

Drop-in replacement for 2.0.0. No schema changes, no new environment variables.

```bash
docker pull ghcr.io/networkbuild3r/archivist-oss:2.0.1
```
