"""Local performance regression benchmarks for Archivist Phase 3 + 3.5.

Design
------
These tests assert that the new transactional path does **not regress** on
throughput or latency vs. documented budgets from the Phase 3 design doc.
They are runnable locally with no external services (Qdrant is mocked).

Thresholds (conservative, not tight; tighten as the system matures):
    - Single ``MemoryTransaction`` open/close: < 50 ms
    - Write 50 events to outbox inside one transaction: < 100 ms
    - Drain 50 pending outbox events: < 500 ms
    - 100 sequential MemoryTransaction writers: < 5 s
    - ``OutboxEvent.payload_json()`` for 1 000 events: < 10 ms total
    - ``_build_schema`` (full DDL): < 200 ms
    - Lookup 1 000 needle_registry rows: < 50 ms

All measurements use ``time.perf_counter`` (wall-clock, not CPU).
Each test prints a summary line so you can track trends over time.

Run subset::

    pytest tests/qa/test_performance_regression.py -v --tb=short -s
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _make_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Benchmark: MemoryTransaction open/close latency
# ---------------------------------------------------------------------------


async def test_transaction_open_close_under_50ms(qa_pool):
    """Single MemoryTransaction context entry + exit completes in < 50 ms."""
    from archivist.storage.transaction import MemoryTransaction

    t0 = time.perf_counter()
    async with MemoryTransaction(enabled=True) as txn:
        # minimal write to touch the write path
        await txn.execute("SELECT 1")
    elapsed = _ms(t0)
    print(f"\n[perf] txn open/close: {elapsed:.1f} ms")
    assert elapsed < 50, f"MemoryTransaction open/close took {elapsed:.1f} ms (limit 50 ms)"


# ---------------------------------------------------------------------------
# Benchmark: Enqueue 50 events in one transaction
# ---------------------------------------------------------------------------


async def test_enqueue_50_events_under_100ms(qa_pool):
    """Enqueueing 50 events inside a single transaction completes in < 100 ms."""
    from archivist.storage.transaction import MemoryTransaction

    memory_id = _make_id()
    t0 = time.perf_counter()
    async with MemoryTransaction(enabled=True) as txn:
        for i in range(50):
            txn.enqueue_qdrant_upsert("col", [], memory_id=memory_id)
    elapsed = _ms(t0)
    print(f"\n[perf] enqueue 50 events: {elapsed:.1f} ms")
    assert elapsed < 100, f"Enqueue 50 events took {elapsed:.1f} ms (limit 100 ms)"


# ---------------------------------------------------------------------------
# Benchmark: Drain 50 pending events
# ---------------------------------------------------------------------------


async def test_drain_50_events_under_500ms(qa_pool, mock_vector_backend):
    """Draining 50 pending outbox events completes in < 500 ms (mock backend)."""
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    async with MemoryTransaction(enabled=True) as txn:
        for _ in range(50):
            txn.enqueue_qdrant_upsert("col", [], memory_id=_make_id())

    processor = OutboxProcessor(mock_vector_backend)
    t0 = time.perf_counter()
    n = await processor.drain()
    elapsed = _ms(t0)
    print(f"\n[perf] drain 50 events: {elapsed:.1f} ms (applied={n})")
    assert n == 50
    assert elapsed < 500, f"Drain 50 events took {elapsed:.1f} ms (limit 500 ms)"


# ---------------------------------------------------------------------------
# Benchmark: 100 sequential writers
# ---------------------------------------------------------------------------


async def test_100_sequential_txn_writers_under_5s(qa_pool):
    """100 sequential MemoryTransaction writes complete in < 5 000 ms total."""
    from archivist.storage.transaction import MemoryTransaction

    t0 = time.perf_counter()
    for _ in range(100):
        async with MemoryTransaction(enabled=True) as txn:
            mem_id = _make_id()
            qdrant_id = _make_id()
            await txn.execute(
                "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
                " VALUES (?, ?, 'primary', '2026-01-17')",
                (mem_id, qdrant_id),
            )
            txn.enqueue_qdrant_upsert("col", [], memory_id=mem_id)
    elapsed = _ms(t0)
    print(f"\n[perf] 100 sequential txn writes: {elapsed:.1f} ms")
    assert elapsed < 5000, f"100 sequential writers took {elapsed:.1f} ms (limit 5 000 ms)"


# ---------------------------------------------------------------------------
# Benchmark: OutboxEvent.payload_json() — pure-Python hot path
# ---------------------------------------------------------------------------


def test_payload_json_1000_events_under_10ms():
    """Serialising 1 000 OutboxEvents takes < 10 ms total."""
    from archivist.storage.outbox import EventType, OutboxEvent

    events = [
        OutboxEvent(
            event_type=EventType.QDRANT_UPSERT,
            payload={"collection": "col", "points": [], "memory_id": f"m-{i}"},
        )
        for i in range(1000)
    ]
    t0 = time.perf_counter()
    for ev in events:
        ev.payload_json()
    elapsed = _ms(t0)
    print(f"\n[perf] payload_json() ×1000: {elapsed:.1f} ms")
    assert elapsed < 10, f"payload_json ×1000 took {elapsed:.1f} ms (limit 10 ms)"


# ---------------------------------------------------------------------------
# Benchmark: schema DDL performance
# ---------------------------------------------------------------------------


async def test_schema_ddl_under_200ms(tmp_path, monkeypatch):
    """Full schema DDL on a fresh DB completes in < 200 ms."""
    from archivist.storage import sqlite_pool as _sp

    p = _sp.SQLitePool()
    db_path = str(tmp_path / "schema_bench.db")
    await p.initialize(db_path)
    orig = _sp.pool
    monkeypatch.setattr(_sp, "pool", p)

    from tests.qa.conftest import _build_schema

    t0 = time.perf_counter()
    await _build_schema(p)
    elapsed = _ms(t0)
    print(f"\n[perf] full schema DDL: {elapsed:.1f} ms")

    monkeypatch.setattr(_sp, "pool", orig)
    await p.close()

    assert elapsed < 200, f"Schema DDL took {elapsed:.1f} ms (limit 200 ms)"


# ---------------------------------------------------------------------------
# Benchmark: 1 000 needle_registry lookups
# ---------------------------------------------------------------------------


async def test_needle_registry_bulk_insert_and_lookup_under_200ms(qa_pool):
    """Inserting and reading 1 000 needle_registry rows completes in < 200 ms."""
    memory_id = _make_id()

    t0 = time.perf_counter()
    async with qa_pool.write() as conn:
        params = [
            (memory_id, f"token-{i}", "default", "qa-agent", "2026-01-17") for i in range(1000)
        ]
        await conn.executemany(
            "INSERT INTO needle_registry (memory_id, token, namespace, agent_id, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            params,
        )
    elapsed_write = _ms(t0)

    t1 = time.perf_counter()
    async with qa_pool.read() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM needle_registry WHERE memory_id=?", (memory_id,)
        )
        count = (await cur.fetchone())[0]
    elapsed_read = _ms(t1)

    print(f"\n[perf] needle 1000 insert: {elapsed_write:.1f} ms, read: {elapsed_read:.1f} ms")
    assert count == 1000
    assert elapsed_write + elapsed_read < 200, (
        f"Needle bulk ops took {elapsed_write + elapsed_read:.1f} ms (limit 200 ms)"
    )


# ---------------------------------------------------------------------------
# Benchmark: concurrent drain throughput
# ---------------------------------------------------------------------------


async def test_concurrent_drain_throughput(qa_pool):
    """Sequential drain processes 100 events in < 2 s total wall-clock time."""
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    for _ in range(100):
        async with MemoryTransaction(enabled=True) as txn:
            txn.enqueue_qdrant_upsert("col", [], memory_id=_make_id())

    apply_count = 0

    async def _noop_upsert(collection, points):
        nonlocal apply_count
        apply_count += 1

    backend = MagicMock()
    backend.upsert = AsyncMock(side_effect=_noop_upsert)
    processor = OutboxProcessor(backend)

    t0 = time.perf_counter()
    total = 0
    while True:
        n = await processor.drain()
        total += n
        if n == 0:
            break
    elapsed = _ms(t0)

    print(f"\n[perf] sequential drain ×100: {elapsed:.1f} ms (applied={total})")
    assert total == 100
    assert elapsed < 2000, f"Sequential drain took {elapsed:.1f} ms (limit 2 000 ms)"
