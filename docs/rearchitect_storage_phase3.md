# Storage Phase 3 — Transactional Outbox + Pluggable Backend Abstraction

**Status:** Implemented (Phase 3 + Phase 3.5 hardening complete)  
**Branch:** `feature/v2.1-storage-phase1-hardening`  
**Preceded by:** Phase 1 (aiosqlite pool + connection management), Phase 2 (all critical missing-await and async bugs fixed)

---

## 1. Problem Statement

Before Phase 3, Archivist had no cross-store transaction boundary. Qdrant (vectors) and SQLite (knowledge graph, FTS5, needle registry, memory_points, curator_queue, audit_log, etc.) were written independently. Any crash or partial failure during a write, cascade, or curator operation could produce:

| Failure Mode | Description | Affected Path |
|---|---|---|
| A | Qdrant upsert succeeds, `memory_points` INSERT fails → vector orphan | `_handle_store`, `index_file` |
| B | `memory_points` ok, `upsert_fts_chunk` fails → FTS orphan | Same |
| C | `memory_points` + FTS ok, `register_needle_tokens` fails → needle orphan | Same |
| D | Merge: merged point lands in Qdrant, originals deleted, merged `memory_points` never written | `merge_memories` |
| E | Micro-chunks registered but primary point missing → dangling child references | `_handle_store` |
| F | `_delete_qdrant_points` succeeds, SQLite cascade fails → stale FTS/needle rows for deleted point | `delete_memory_complete` |
| G | Outbox row written, crash before commit → Qdrant event applied twice on retry | All write paths |

**Specific file + line references (pre-Phase 3):**

- `src/archivist/app/handlers/tools_storage.py` ~line 312: `client.upsert(...)` followed by separate `upsert_fts_chunk(...)` — no atomic boundary.
- `src/archivist/write/indexer.py` ~line 89: same pattern.
- `src/archivist/lifecycle/merge.py` ~line 95: `client.upsert(...)` then `upsert_fts_chunk(...)` outside any transaction.
- `src/archivist/lifecycle/memory_lifecycle.py` ~line 167: `_delete_qdrant_points(...)` then `_delete_sqlite_artifacts(...)` in separate calls.

---

## 2. Proposed Architecture

### 2.1 Outbox Table (SQLite)

A new `outbox` table holds pending cross-store events atomically with the SQLite writes that produce them. The table lives in the same database as all other Archivist data, so it participates in the same `pool.write()` transaction.

```sql
CREATE TABLE IF NOT EXISTS outbox (
    id           TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    payload      TEXT NOT NULL,         -- JSON
    status       TEXT NOT NULL DEFAULT 'pending',
    retry_count  INTEGER NOT NULL DEFAULT 0,
    last_attempt TEXT,
    created_at   TEXT NOT NULL,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status, created_at);
CREATE INDEX IF NOT EXISTS idx_outbox_event  ON outbox(event_type, status);
```

**Event types:**
- `QDRANT_UPSERT` — upsert one or more `PointStruct` objects.
- `QDRANT_DELETE` — delete a list of point IDs.
- `QDRANT_DELETE_FILTER` — delete by filter expression.
- `QDRANT_SET_PAYLOAD` — update payload fields on a set of IDs.

### 2.2 MemoryTransaction Async Context Manager

`src/archivist/storage/transaction.py` — `MemoryTransaction` wraps a single `pool.write()` context. Inside the block, callers:

1. Execute SQLite writes directly on `txn.conn` (or via the provided shims).
2. Enqueue Qdrant operations via `txn.enqueue_*` methods; these accumulate in-memory.
3. On clean `__aexit__`, all outbox rows are flushed into the `outbox` table before the implicit commit — the SQLite state change and the outbox event land in one fsync.
4. On exception, `pool.write()` rolls back everything — neither artefacts nor outbox rows are written.

**Shim methods available on `MemoryTransaction`:**

| Method | Delegates to |
|---|---|
| `txn.execute(sql, params)` | `self.conn.execute(...)` |
| `txn.executemany(sql, params)` | `self.conn.executemany(...)` |
| `txn.fetchall(sql, params)` | `self.conn.execute(...).fetchall()` |
| `txn.upsert_fts_chunk(...)` | `graph.upsert_fts_chunk(..., conn=self.conn)` |
| `txn.register_needle_tokens(...)` | `graph.register_needle_tokens(..., conn=self.conn)` |
| `txn.upsert_entity(...)` | `graph.upsert_entity(..., conn=self.conn)` |
| `txn.add_fact(...)` | `graph.add_fact(..., conn=self.conn)` |

**Critical design constraint:** `pool.write()` (backed by `asyncio.Lock`) is **not re-entrant**. Any helper called inside `async with MemoryTransaction()` that also tries to acquire `pool.write()` will deadlock. The solution is to pass `conn=self.conn` through to every graph helper that writes SQLite, rather than letting them acquire the lock themselves. All graph write helpers (`upsert_fts_chunk`, `register_needle_tokens`, `upsert_entity`, `add_fact`) accept an optional `conn: aiosqlite.Connection | None = None` for this reason.

### 2.3 Thin Protocol Backends

`src/archivist/storage/backends.py` defines two `Protocol` types:

```python
class VectorBackend(Protocol):
    async def upsert(self, collection: str, points: list[PointStruct]) -> None: ...
    async def delete(self, collection: str, ids: list[str]) -> None: ...
    async def delete_by_filter(self, collection: str, filt: Filter) -> None: ...
    async def set_payload(self, collection: str, payload: dict, ids: list[str]) -> None: ...
    async def search(self, collection: str, vector: list[float], limit: int, filt: Filter | None) -> list[ScoredPoint]: ...
    async def retrieve(self, collection: str, ids: list[str]) -> list[Record]: ...
    async def scroll(self, collection: str, filt: Filter | None, limit: int) -> list[Record]: ...

class GraphBackend(Protocol):
    async def execute(self, sql: str, params: tuple = ()) -> Any: ...
    async def executemany(self, sql: str, params: list[tuple]) -> None: ...
    async def fetchall(self, sql: str, params: tuple = ()) -> list[Any]: ...
    async def fetchone(self, sql: str, params: tuple = ()) -> Any | None: ...
```

`QdrantVectorBackend` implements `VectorBackend`, wrapping the synchronous `QdrantClient` with `asyncio.to_thread`.

### 2.4 OutboxProcessor Background Task

`src/archivist/storage/outbox.py` — `OutboxProcessor` drains the `outbox` table:

1. Poll every `OUTBOX_DRAIN_INTERVAL` seconds (default: 5 s).
2. Fetch up to `OUTBOX_BATCH_SIZE` pending events (default: 50).
3. Apply each event idempotently to the `VectorBackend`.
4. On success: mark `status='done'`.
5. On failure: increment `retry_count`; if `>= OUTBOX_MAX_RETRIES` (default: 5), mark `status='dead'` and copy to `delete_failures` audit table.
6. Concurrent-drain safety: `SELECT ... FOR UPDATE SKIP LOCKED` equivalent via `status='processing'` row locking.
7. Pruning: rows with `status='done'` older than `OUTBOX_RETENTION_DAYS` are deleted on each drain cycle.

---

## 3. Exact Files Changed

### New files
| File | Purpose |
|---|---|
| `src/archivist/storage/backends.py` | `VectorBackend`, `GraphBackend` protocols + `QdrantVectorBackend` adapter |
| `src/archivist/storage/outbox.py` | `EventType`, `OutboxEvent`, `OutboxProcessor` |
| `src/archivist/storage/transaction.py` | `MemoryTransaction` async context manager |
| `tests/test_outbox.py` | 14 unit + integration + chaos tests |
| `docs/rearchitect_storage_phase3.md` | This file |

### Modified files
| File | Change summary |
|---|---|
| `src/archivist/storage/graph.py` | Added `outbox` + `needle_registry` DDL to `init_schema()`; added optional `conn` param to `upsert_fts_chunk`, `register_needle_tokens`, `upsert_entity`, `add_fact` |
| `src/archivist/core/config.py` | Added `OUTBOX_ENABLED`, `OUTBOX_DRAIN_INTERVAL`, `OUTBOX_BATCH_SIZE`, `OUTBOX_MAX_RETRIES`, `OUTBOX_RETENTION_DAYS` |
| `src/archivist/app/main.py` | Start `OutboxProcessor.drain_loop()` as background task in `_startup` |
| `src/archivist/app/handlers/tools_storage.py` | `_handle_store` refactored: FTS5, needle, `memory_points`, and outbox event all inside one `MemoryTransaction`; fixed `_rbac_gate` missing `action` arg in `archivist_delete` |
| `src/archivist/write/indexer.py` | `index_file` refactored: all SQLite writes + outbox event inside one `MemoryTransaction`; entity/fact extraction loop restored inside txn |
| `src/archivist/lifecycle/merge.py` | `merge_memories`: FTS5, needle, `memory_points`, outbox event all inside one `MemoryTransaction` |
| `src/archivist/lifecycle/memory_lifecycle.py` | `delete_memory_complete`: when `OUTBOX_ENABLED=True`, Qdrant-delete enqueue + FTS/needle/entity-facts cleanup all inside one `MemoryTransaction` |
| `src/archivist/retrieval/fts_search.py` | Fixed `search_bm25` sync/async mismatch (was `def`, now `async def`; internal calls to `search_fts` now `await`ed) |
| `src/archivist/retrieval/rlm_retriever.py` | Updated `_bm25_async` to `await search_bm25(...)` directly |

### Files that must NOT change (external API surface)
- All MCP tool signatures in `tools_storage.py`, `tools_search.py`, `tools_graph.py`
- `ARCHITECTURE.md`
- Any public REST or WebSocket endpoint

---

## 4. Configuration Variables

| Variable | Default | Description |
|---|---|---|
| `OUTBOX_ENABLED` | `false` | Master switch. When `false`, all `enqueue_*` calls are no-ops and Qdrant writes run inline (legacy behaviour). |
| `OUTBOX_DRAIN_INTERVAL` | `5` | Seconds between drain cycles. |
| `OUTBOX_BATCH_SIZE` | `50` | Max events per drain cycle. |
| `OUTBOX_MAX_RETRIES` | `5` | Retry attempts before marking event `dead`. |
| `OUTBOX_RETENTION_DAYS` | `7` | Days to retain `done` rows before pruning. |

---

## 5. Atomic Boundary Coverage (Post Phase 3.5)

| Write path | Atomic boundary covers |
|---|---|
| `_handle_store` (primary) | `memory_points` + FTS5 + needle + outbox event |
| `_handle_store` (micro-chunks) | Same — each micro-chunk in its own `MemoryTransaction` |
| `index_file` | `memory_points` + FTS5 + needle + entity/fact loop + outbox event for all chunks in batch |
| `merge_memories` | `memory_points` + FTS5 + needle + outbox event |
| `delete_memory_complete` (`OUTBOX_ENABLED=True`) | FTS cleanup + needle cleanup + entity-facts cleanup + Qdrant-delete outbox event |

**Failure modes closed:** A, B, C, D (partially E), F, G.

---

## 6. Trade-offs

| Concern | Impact | Mitigation |
|---|---|---|
| Latency | +0–2 ms per write (outbox row insert is cheap) | Negligible vs. Qdrant RTT |
| Disk usage | ~100 bytes per event; pruned after `OUTBOX_RETENTION_DAYS` | Configurable retention |
| Complexity | New module surface (`transaction.py`, `outbox.py`, `backends.py`) | Isolated behind `MemoryTransaction` shim; callers see minimal change |
| Re-entrancy | `pool.write()` is not re-entrant → helpers must accept `conn=` param | All write helpers updated; shims enforce correct pattern |
| Eventual consistency | Qdrant writes lag behind SQLite by up to `OUTBOX_DRAIN_INTERVAL` seconds when `OUTBOX_ENABLED=True` | Acceptable for memory systems; inline fallback available |
| PostgreSQL migration | `GraphBackend` protocol isolates SQLite calls | Drop-in `PostgresGraphBackend` when ready |

---

## 7. Backward Compatibility

- `OUTBOX_ENABLED=false` (default): zero behaviour change. All Qdrant writes remain inline. Outbox table exists but is never written to.
- `OUTBOX_ENABLED=true`: opt-in per deployment. Existing memories require no migration; the outbox drains to steady-state on first startup.
- Schema additions (`outbox`, `needle_registry` in `init_schema()`) are `CREATE TABLE IF NOT EXISTS` — zero-downtime on upgrade.

---

## 8. Test Strategy

### Unit tests (`tests/test_outbox.py`)
- `EventType` enum coverage
- `OutboxEvent` JSON serialisation round-trip
- `VectorBackend` protocol structural subtype check
- `MemoryTransaction` commit path: outbox rows flushed atomically
- `MemoryTransaction` rollback path: exception aborts all writes
- `MemoryTransaction` disabled path: `enqueue_*` are no-ops
- `OutboxProcessor` drain: events applied + marked done
- `OutboxProcessor` retry: failed events incremented then dead-lettered
- `OutboxProcessor` concurrent-drain safety
- Chaos test: simulated crash mid-flush → verify no partial state

### Integration tests
- `tests/test_provenance_integration.py::test_store_propagates_to_sqlite` — verifies FTS, needle, and `memory_points` all written by `_handle_store`
- `tests/test_merge_consistency.py` — regression for merge atomicity
- `tests/test_chunk2_cascade.py` — regression for delete cascade

### Chaos scenario
Set `OUTBOX_ENABLED=true`, inject `os.kill(os.getpid(), signal.SIGKILL)` mid-transaction, restart, verify:
1. No Qdrant point exists without a corresponding `memory_points` row.
2. No `memory_points` row exists without FTS5 and needle entries.
3. All pending outbox rows drain successfully on restart.

---

## 9. Deadlock Prevention

The `pool.write()` lock is an `asyncio.Lock` — not re-entrant. Any coroutine that calls `pool.write()` inside an already-held `pool.write()` context will deadlock.

**Prevention strategy:**
1. All graph write helpers accept `conn: aiosqlite.Connection | None = None`.
2. When called from inside a `MemoryTransaction`, callers pass `conn=txn.conn`.
3. When called standalone, `conn=None` triggers a fresh `pool.write()` acquisition.
4. `MemoryTransaction` shims (`txn.upsert_fts_chunk`, `txn.upsert_entity`, etc.) enforce this automatically.
5. `init_schema()` runs the `needle_registry` DDL at startup so the lazy schema guard never fires inside a live transaction.

---

## 10. Known Remaining Gaps (Post Phase 3.5)

- `archive_memory_complete` — archive path does not use `MemoryTransaction`; archive is lower-risk (no hard deletes) but could be tightened in a future pass.
- `soft_delete_memory` — enqueues to `curator_queue` (not outbox); the subsequent hard-cascade via `delete_memory_complete` is now fully atomic when `OUTBOX_ENABLED=True`.
- `add_relationship` in `graph.py` — does not yet accept an optional `conn` param; not on any current hot write path but should be updated before a PostgreSQL migration.
