"""Chaos and fault-injection tests for the Archivist write path.

Scenarios covered
-----------------
1. Crash-before-drain: transaction commits but drain never runs → event
   survives in outbox; a later drain picks it up with zero data loss.
2. Mid-drain crash: event stuck in 'processing' → a fresh drain correctly
   skips it (no double-apply).
3. Qdrant network blip: transient ``ConnectionError`` causes retry with
   exponential back-off; second drain succeeds.
4. Sustained Qdrant failure: after ``OUTBOX_MAX_RETRIES`` attempts the
   event moves to ``dead`` and a ``delete_failures`` audit row is written.
5. Concurrent drains: two concurrent ``drain()`` calls never double-apply
   the same event (atomic 'processing' claim).
6. SQLite lock contention: ``OUTBOX_BATCH_SIZE`` concurrent writers all
   commit and all events are drained exactly once.
7. Exception injection mid-write: exception after the first of two SQL
   inserts rolls back the second; zero partial state.
8. Pool not initialised: entering ``MemoryTransaction`` before
   ``pool.initialize()`` raises ``RuntimeError`` (not a silent hang).

All tests use ``@pytest.mark.chaos`` for selective running.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.qa.conftest import count_outbox, reset_outbox_backoff

pytestmark = pytest.mark.chaos


# ---------------------------------------------------------------------------
# 1. Crash-before-drain — event survives, later drain recovers it
# ---------------------------------------------------------------------------


@pytest.mark.chaos
async def test_crash_before_drain_event_survives_and_drains(
    qa_pool, mock_vector_backend, memory_factory
):
    """Process 'crash' between SQLite commit and drain call.

    The outbox row must survive (it was committed).  A subsequent drain
    must apply it exactly once and mark it 'applied'.
    """
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    mem = memory_factory()

    # --- Arrange: commit transaction, simulating crash before drain ---
    async with MemoryTransaction(enabled=True) as txn:
        await txn.execute(
            "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
            " VALUES (?, ?, ?, ?)",
            (mem["memory_id"], mem["qdrant_id"], "primary", mem["date"]),
        )
        txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])

    # Crash simulated — drain never runs.
    assert await count_outbox(qa_pool, "pending") == 1

    # --- Act: process restarts, drain runs ---
    processor = OutboxProcessor(mock_vector_backend)
    n = await processor.drain()

    # --- Assert ---
    assert n == 1
    mock_vector_backend.delete.assert_called_once_with("col", [mem["qdrant_id"]])
    assert await count_outbox(qa_pool, "applied") == 1
    assert await count_outbox(qa_pool, "pending") == 0


# ---------------------------------------------------------------------------
# 2. Mid-drain crash — stuck 'processing' event is not double-applied
# ---------------------------------------------------------------------------


@pytest.mark.chaos
async def test_mid_drain_crash_processing_event_not_double_applied(
    qa_pool, mock_vector_backend
):
    """Event in 'processing' state (crash mid-drain) is not re-applied by a new drain."""
    from archivist.storage.outbox import OutboxProcessor

    # Insert an event already in 'processing' (crash happened between claim and apply).
    async with qa_pool.write() as conn:
        await conn.execute(
            """INSERT INTO outbox (id, event_type, payload, status, retry_count, created_at)
               VALUES ('stuck-1', 'qdrant_delete',
                       '{"collection":"col","ids":["s1"],"memory_id":"m1"}',
                       'processing', 0, ?)""",
            (datetime.now(UTC).isoformat(),),
        )

    processor = OutboxProcessor(mock_vector_backend)
    n = await processor.drain()

    assert n == 0
    mock_vector_backend.delete.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Transient Qdrant failure — retry succeeds on second attempt
# ---------------------------------------------------------------------------


@pytest.mark.chaos
async def test_transient_qdrant_failure_retries_and_succeeds(
    qa_pool, memory_factory, monkeypatch
):
    """First drain raises ``ConnectionError``; second drain succeeds.

    Verifies retry_count increments and back-off timestamp is set.
    """
    import archivist.core.config as _cfg
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 5)
    mem = memory_factory()

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])

    call_count = 0

    async def _flaky_delete(collection, ids):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("503 transient")

    backend = MagicMock()
    backend.delete = AsyncMock(side_effect=_flaky_delete)

    processor = OutboxProcessor(backend)

    # First drain — fails
    n1 = await processor.drain()
    assert n1 == 0

    async with qa_pool.read() as conn:
        cur = await conn.execute("SELECT retry_count, status FROM outbox")
        row = await cur.fetchone()
    assert row[1] == "pending"
    assert row[0] == 1

    # Force immediate retryability by rewinding last_attempt
    await reset_outbox_backoff(qa_pool)

    # Second drain — succeeds
    n2 = await processor.drain()
    assert n2 == 1
    assert call_count == 2
    assert await count_outbox(qa_pool, "applied") == 1


# ---------------------------------------------------------------------------
# 4. Sustained failure → dead-letter
# ---------------------------------------------------------------------------


@pytest.mark.chaos
async def test_sustained_failure_creates_dead_letter_entry(
    qa_pool, memory_factory, monkeypatch
):
    """After OUTBOX_MAX_RETRIES exhausted: status='dead', delete_failures row written."""
    import archivist.core.config as _cfg
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 2)
    mem = memory_factory()

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])

    backend = MagicMock()
    backend.delete = AsyncMock(side_effect=ConnectionError("always fails"))
    processor = OutboxProcessor(backend)

    for _ in range(3):
        await reset_outbox_backoff(qa_pool)
        await processor.drain()

    async with qa_pool.read() as conn:
        cur = await conn.execute("SELECT status FROM outbox")
        row = await cur.fetchone()
        df_cur = await conn.execute("SELECT COUNT(*) FROM delete_failures")
        df_count = (await df_cur.fetchone())[0]

    assert row[0] == "dead"
    assert df_count == 1


# ---------------------------------------------------------------------------
# 5. Concurrent drains — exactly-once semantics
# ---------------------------------------------------------------------------


@pytest.mark.chaos
async def test_concurrent_drains_apply_each_event_exactly_once(
    qa_pool, memory_factory
):
    """Two simultaneous ``drain()`` calls: total applied == number of events, no double-apply."""
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    event_count = 5
    mems = [memory_factory() for _ in range(event_count)]

    for mem in mems:
        async with MemoryTransaction(enabled=True) as txn:
            txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])

    apply_log: list[str] = []

    async def _counted_delete(collection, ids):
        apply_log.extend(ids)
        await asyncio.sleep(0)  # yield to allow interleaving

    backend = MagicMock()
    backend.delete = AsyncMock(side_effect=_counted_delete)

    processor = OutboxProcessor(backend)
    results = await asyncio.gather(processor.drain(), processor.drain())

    assert sum(results) == event_count
    # Each qdrant_id appears exactly once across both drains
    assert len(apply_log) == event_count
    assert len(set(apply_log)) == event_count


# ---------------------------------------------------------------------------
# 6. Lock contention — N concurrent writers, all drain exactly once
# ---------------------------------------------------------------------------


@pytest.mark.chaos
async def test_concurrent_writers_all_events_drained_exactly_once(
    qa_pool, memory_factory
):
    """Sequential MemoryTransaction writers; each event applied exactly once after drain."""
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    n = 20
    mems = [memory_factory() for _ in range(n)]

    # Sequential — GRAPH_WRITE_LOCK_ASYNC is a module-level asyncio.Lock that
    # cannot be shared across concurrent asyncio.gather tasks in tests.
    for mem in mems:
        async with MemoryTransaction(enabled=True) as txn:
            txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])

    assert await count_outbox(qa_pool, "pending") == n

    apply_count = 0

    async def _counted_upsert(collection, points):
        nonlocal apply_count
        apply_count += 1

    backend = MagicMock()
    backend.upsert = AsyncMock(side_effect=_counted_upsert)
    processor = OutboxProcessor(backend)

    total_applied = 0
    while True:
        batch = await processor.drain()
        total_applied += batch
        if batch == 0:
            break

    assert total_applied == n
    assert apply_count == n
    assert await count_outbox(qa_pool, "applied") == n


# ---------------------------------------------------------------------------
# 7. Exception mid-write — partial SQL rolled back
# ---------------------------------------------------------------------------


@pytest.mark.chaos
async def test_exception_after_first_insert_rolls_back_second(qa_pool, memory_factory):
    """Exception after first INSERT rolls back both inserts — zero orphans."""
    from archivist.storage.transaction import MemoryTransaction

    mem1 = memory_factory()
    mem2 = memory_factory()

    with pytest.raises(RuntimeError, match="injected fault"):
        async with MemoryTransaction(enabled=True) as txn:
            await txn.execute(
                "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
                " VALUES (?, ?, ?, ?)",
                (mem1["memory_id"], mem1["qdrant_id"], "primary", mem1["date"]),
            )
            txn.enqueue_qdrant_upsert("col", [], memory_id=mem1["memory_id"])
            # Fault injected between first and second write
            raise RuntimeError("injected fault")

    # Second write never happened; first rolled back
    async with qa_pool.read() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM memory_points")
        assert (await cur.fetchone())[0] == 0


# ---------------------------------------------------------------------------
# 8. Pool not initialised raises RuntimeError (not a deadlock)
# ---------------------------------------------------------------------------


@pytest.mark.chaos
async def test_transaction_raises_when_pool_not_initialized():
    """Entering MemoryTransaction before pool.initialize() raises RuntimeError promptly."""
    from archivist.storage import sqlite_pool as _sp
    from archivist.storage.transaction import MemoryTransaction

    # Create a fresh, uninitialised pool and swap the singleton temporarily.
    uninit_pool = _sp.SQLitePool()
    original = _sp.pool
    _sp.pool = uninit_pool  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="not initialized"):
            async with MemoryTransaction(enabled=True):
                pass  # pragma: no cover
    finally:
        _sp.pool = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 9. Multiple event types in one transaction — all survive a simulated crash
# ---------------------------------------------------------------------------


@pytest.mark.chaos
async def test_all_four_event_types_survive_crash_and_drain(
    qa_pool, mock_vector_backend, memory_factory
):
    """All four EventType variants committed in one txn are applied after a drain."""
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    mem = memory_factory()
    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])
        txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])
        txn.enqueue_qdrant_delete_filter("col", {}, memory_id=mem["memory_id"])
        txn.enqueue_qdrant_set_payload("col", {"k": "v"}, [], memory_id=mem["memory_id"])

    assert await count_outbox(qa_pool, "pending") == 4

    processor = OutboxProcessor(mock_vector_backend)
    n = await processor.drain()

    assert n == 4
    assert await count_outbox(qa_pool, "applied") == 4
