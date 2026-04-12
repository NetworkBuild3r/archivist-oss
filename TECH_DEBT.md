# Technical Debt — Archivist Cascade & Delete Architecture

This document tracks known architectural limitations in the delete/archive
pipeline and the concrete plan for addressing them.  Items are ordered by
impact.

---

## Phase 1 — Soft-Delete + Background Hard-Cascade

### Problem

`delete_memory_complete()` is a synchronous, multi-step cascade that touches
Qdrant (network) and SQLite (disk) with no distributed transaction.  A process
crash mid-cascade leaves partial state.  While `PartialDeletionError` and the
orphan sweeper provide a practical safety net, the API call that triggered the
delete still blocks for the full cascade duration.

### Solution

Split delete into two phases:

**Hot path** (inline, ~5 ms):
1. `client.set_payload({"deleted": True})` on the primary Qdrant point.
2. Enqueue a `delete_memory` job in `curator_queue` (already exists with
   `pending`/`applied`/`failed` state tracking).
3. Log to `audit_log` with `status="soft_delete_initiated"`.
4. Return immediately to the caller.

**Background worker** (via `curator_queue.drain()`):
- Calls `delete_memory_complete()` for the full hard cascade.
- `curator_queue` already marks jobs `applied` or `failed` and retries on
  failure.
- TTL expiry (`main.py`) and merge cleanup (`merge.py`) are already
  background tasks — they keep calling `delete_memory_complete()` directly.

**New public function** in `memory_lifecycle.py`:
```python
async def soft_delete_memory(memory_id: str, namespace: str) -> dict:
    """Mark deleted + enqueue background cascade.  Returns immediately."""
```

**Notes:**
- No new SQLite table required.  `curator_queue` is the state store.
- `delete_memory_complete()` becomes the background-only hard-delete worker.
- `PartialDeletionError` and the sweeper remain as safety net layers.

### Why Not Done Yet

Requires a new MCP tool (`archivist_delete`) and changes to all callers.
Scoped to a follow-up PR to keep this branch reviewable.

---

## Phase 1b — Fix Archived/Deleted Filter in Retrieval Pipeline

### Problem (CRITICAL)

`archive_memory_complete()` sets `{"archived": True}` on Qdrant point
payloads.  However, **no retrieval path filters on this field**.

- `rlm_retriever.py` — no `must_not` clause for `archived`
- `fts_search.py` — no exclusion in BM25/FTS5 queries
- Needle registry lookup — no exclusion

Result: archived memories continue to appear in every search.  The
`archivist_compress` tool's claim that originals are "excluded from default
search" is aspirational but not implemented.

### Solution

**Vector search** (`rlm_retriever.py`, `search_vectors()`):
```python
must_not_filters.append(
    FieldCondition(key="archived", match=MatchValue(value=True))
)
```
Apply to every call that builds a `must_filters` list.

**BM25/FTS search** (`fts_search.py`):
Add a `deleted`/`archived` flag column to `memory_chunks` (or join against
a SQLite `archived_ids` table populated by `archive_memory_complete`).

**Needle registry** (`graph_retrieval.py`):
Validate retrieved IDs against a `deleted/archived` set before returning.

**Also needed for Phase 1:** add a `deleted` payload filter using the same
pattern, so soft-deleted memories disappear from search immediately after the
hot path.

---

## Phase 2 — `memory_points` Tracking Table

### Problem

There is no SQLite record of which Qdrant point IDs belong to a memory.
The cascade must scroll Qdrant to discover child IDs (micro-chunks, reverse
HyDE) before it can delete them.  This scroll can fail, leaving orphans.
There is also no way to enumerate all points for a memory without querying
Qdrant.

### Solution

Add a `memory_points` table to SQLite:

```sql
CREATE TABLE memory_points (
    memory_id   TEXT NOT NULL,
    qdrant_id   TEXT NOT NULL,
    point_type  TEXT NOT NULL DEFAULT 'primary',  -- primary | micro_chunk | reverse_hyde
    created_at  TEXT NOT NULL,
    PRIMARY KEY (memory_id, qdrant_id)
);
CREATE INDEX idx_mp_memory  ON memory_points(memory_id);
CREATE INDEX idx_mp_qdrant  ON memory_points(qdrant_id);
```

**Write path:** populate on every `client.upsert()` call in
`tools_storage.py`, `indexer.py`, and `hyde.py`.

**Delete path:** look up all `qdrant_id` values for the `memory_id`, delete
them as a batch from Qdrant, then delete the `memory_points` rows atomically
in SQLite.  No Qdrant scroll required.

**Benefit:** hard-delete becomes:
1. `DELETE FROM memory_points WHERE memory_id = ?` (SQLite, atomic)
2. `client.delete(collection_name=col, points_selector=[...])` (bulk)

**Dead-letter queue:** failed deletes log the Qdrant IDs to a
`delete_failures` table for manual inspection and replay.

---

## Completed (This Branch)

- Entity-facts `LIKE '%uuid%'` hack replaced with indexed `memory_id` column.
- `_scroll_all` max-pages guard to prevent infinite pagination.
- `PartialDeletionError` tightened to raise on any Qdrant (data-loss) step.
- Bare `except Exception` narrowed to `(sqlite3.Error, OSError)` for SQLite steps.
- Consistency model documented in `README.md`.
