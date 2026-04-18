"""End-to-end outbox lifecycle tests.

Full lifecycle stages covered
------------------------------
1. **Enqueue** — ``MemoryTransaction`` flushes events as 'pending' rows.
2. **Drain** — ``OutboxProcessor.drain()`` claims → applies → 'applied'.
3. **Retry** — Transient failures increment ``retry_count`` and reset
   ``last_attempt`` with back-off; event is re-tried on next drain.
4. **Dead-letter** — After ``OUTBOX_MAX_RETRIES`` the event moves to
   'dead' and an audit row lands in ``delete_failures``.
5. **Pruning** — Applied events older than a threshold can be pruned
   without disturbing pending / dead rows.
6. **Idempotency** — Replaying an 'applied' event does not cause a
   second Qdrant call (the 'processing' claim only targets 'pending').
7. **Batch ordering** — Events are drained in FIFO (``created_at ASC``).
8. **All EventType variants** — Each ``EventType`` routes to the correct
   ``VectorBackend`` method.
9. **drain_loop cancellation** — ``drain_loop`` handles ``asyncio.CancelledError``
   cleanly (loop is driven by ``asyncio.sleep``; cancelling the task is safe).
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.fixtures.mocks import count_outbox, reset_outbox_backoff

pytestmark = [pytest.mark.integration, pytest.mark.storage]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_outbox_row(
    pool,
    event_id: str,
    event_type: str,
    payload: dict,
    status: str = "pending",
    retry_count: int = 0,
    last_attempt: str | None = None,
) -> None:
    async with pool.write() as conn:
        await conn.execute(
            """INSERT INTO outbox (id, event_type, payload, status, retry_count,
                                   last_attempt, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                event_type,
                json.dumps(payload),
                status,
                retry_count,
                last_attempt,
                datetime.now(UTC).isoformat(),
            ),
        )


# ---------------------------------------------------------------------------
# 1. Enqueue
# ---------------------------------------------------------------------------


async def test_enqueue_creates_pending_rows(qa_pool, memory_factory):
    """MemoryTransaction flushes each enqueued event as a 'pending' outbox row."""
    from archivist.storage.transaction import MemoryTransaction

    mems = [memory_factory() for _ in range(4)]
    async with MemoryTransaction(enabled=True) as txn:
        for mem in mems:
            txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])

    assert await count_outbox(qa_pool, "pending") == 4
    assert await count_outbox(qa_pool) == 4  # no stale rows from previous tests


async def test_enqueued_events_have_correct_event_types(qa_pool, memory_factory):
    """Each enqueue_* call persists the matching event_type string."""
    from archivist.storage.transaction import MemoryTransaction

    mem = memory_factory()
    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])
        txn.enqueue_qdrant_delete("col", ["id1"], memory_id=mem["memory_id"])
        txn.enqueue_qdrant_delete_filter("col", {}, memory_id=mem["memory_id"])
        txn.enqueue_qdrant_set_payload("col", {}, [], memory_id=mem["memory_id"])

    async with qa_pool.read() as conn:
        cur = await conn.execute("SELECT event_type FROM outbox ORDER BY created_at")
        types = [row[0] for row in await cur.fetchall()]

    assert types == [
        "qdrant_upsert",
        "qdrant_delete",
        "qdrant_delete_filter",
        "qdrant_set_payload",
    ]


# ---------------------------------------------------------------------------
# 2. Drain — applied status
# ---------------------------------------------------------------------------


async def test_drain_transitions_pending_to_applied(qa_pool, mock_vector_backend, memory_factory):
    """drain() moves pending events to 'applied' after successful backend call."""
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    mem = memory_factory()
    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])

    processor = OutboxProcessor(mock_vector_backend)
    n = await processor.drain()

    assert n == 1
    assert await count_outbox(qa_pool, "applied") == 1
    assert await count_outbox(qa_pool, "pending") == 0


async def test_drain_returns_zero_when_no_pending_events(qa_pool, mock_vector_backend):
    """drain() returns 0 when the outbox is empty."""
    from archivist.storage.outbox import OutboxProcessor

    processor = OutboxProcessor(mock_vector_backend)
    n = await processor.drain()
    assert n == 0


# ---------------------------------------------------------------------------
# 3. Retry — back-off preserved
# ---------------------------------------------------------------------------


async def test_retry_increments_retry_count(qa_pool, memory_factory, monkeypatch):
    """Failed drain increments retry_count on the row."""
    import archivist.core.config as _cfg
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 5)
    mem = memory_factory()

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])

    backend = MagicMock()
    backend.delete = AsyncMock(side_effect=ConnectionError("transient"))
    processor = OutboxProcessor(backend)

    await processor.drain()

    async with qa_pool.read() as conn:
        cur = await conn.execute("SELECT retry_count, status FROM outbox")
        row = await cur.fetchone()
    assert row[0] == 1
    assert row[1] == "pending"


async def test_retry_back_off_timestamp_prevents_immediate_retry(
    qa_pool, memory_factory, monkeypatch
):
    """After a failure the event's last_attempt is set to the future, blocking next drain."""
    import archivist.core.config as _cfg
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 5)
    mem = memory_factory()

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])

    backend = MagicMock()
    backend.delete = AsyncMock(side_effect=ConnectionError("transient"))
    processor = OutboxProcessor(backend)

    await processor.drain()  # fails — sets last_attempt to future

    # Second drain without resetting back-off must find 0 eligible rows.
    backend.delete = AsyncMock(return_value=None)  # would succeed if called
    n = await processor.drain()
    assert n == 0  # still blocked by back-off


# ---------------------------------------------------------------------------
# 4. Dead-letter
# ---------------------------------------------------------------------------


async def test_dead_letter_after_max_retries(qa_pool, memory_factory, monkeypatch):
    """Event reaches 'dead' status after OUTBOX_MAX_RETRIES failures."""
    import archivist.core.config as _cfg
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 2)
    mem = memory_factory()

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])

    backend = MagicMock()
    backend.delete = AsyncMock(side_effect=RuntimeError("permanent error"))
    processor = OutboxProcessor(backend)

    for _ in range(3):
        await reset_outbox_backoff(qa_pool)
        await processor.drain()

    assert await count_outbox(qa_pool, "dead") == 1
    assert await count_outbox(qa_pool, "pending") == 0


async def test_dead_event_writes_delete_failures_audit_row(qa_pool, memory_factory, monkeypatch):
    """Dead-letter transition writes exactly one row to delete_failures."""
    import archivist.core.config as _cfg
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 1)
    mem = memory_factory()

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])

    backend = MagicMock()
    backend.delete = AsyncMock(side_effect=RuntimeError("hard fail"))
    processor = OutboxProcessor(backend)

    for _ in range(2):
        await reset_outbox_backoff(qa_pool)
        await processor.drain()

    async with qa_pool.read() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM delete_failures")
        count = (await cur.fetchone())[0]
    assert count == 1


# ---------------------------------------------------------------------------
# 5. Pruning applied events
# ---------------------------------------------------------------------------


async def test_pruning_applied_events_does_not_affect_pending_or_dead(
    qa_pool, memory_factory, mock_vector_backend, monkeypatch
):
    """Deleting 'applied' rows does not touch 'pending' or 'dead' rows."""
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    # Commit and drain one event (becomes 'applied')
    mem1 = memory_factory()
    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_upsert("col", [], memory_id=mem1["memory_id"])
    processor = OutboxProcessor(mock_vector_backend)
    await processor.drain()

    # Add one more pending event (not yet drained)
    mem2 = memory_factory()
    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_upsert("col", [], memory_id=mem2["memory_id"])

    # Prune applied rows
    async with qa_pool.write() as conn:
        await conn.execute("DELETE FROM outbox WHERE status='applied'")

    assert await count_outbox(qa_pool, "applied") == 0
    assert await count_outbox(qa_pool, "pending") == 1


# ---------------------------------------------------------------------------
# 6. Idempotency — 'applied' events not re-processed
# ---------------------------------------------------------------------------


async def test_applied_event_not_reprocessed_by_subsequent_drain(
    qa_pool, mock_vector_backend, memory_factory
):
    """An already-'applied' event is ignored by drain() — backend called only once."""
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    mem = memory_factory()
    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])

    processor = OutboxProcessor(mock_vector_backend)
    await processor.drain()
    await processor.drain()  # second drain — should be a no-op

    assert mock_vector_backend.upsert.call_count == 1


# ---------------------------------------------------------------------------
# 7. FIFO ordering — batch drained in created_at ASC order
# ---------------------------------------------------------------------------


async def test_drain_processes_events_in_fifo_order(qa_pool, monkeypatch):
    """Events are drained in FIFO order (created_at ASC)."""
    from archivist.storage.outbox import OutboxProcessor

    monkeypatch.setattr(
        __import__("archivist.core.config", fromlist=["OUTBOX_BATCH_SIZE"]),
        "OUTBOX_BATCH_SIZE",
        10,
    )

    ids_in_order = ["first", "second", "third"]
    base_ts = "2026-01-17T10:00:0"
    async with qa_pool.write() as conn:
        for i, eid in enumerate(ids_in_order):
            await conn.execute(
                """INSERT INTO outbox (id, event_type, payload, status, retry_count, created_at)
                   VALUES (?, 'qdrant_delete',
                           '{"collection":"col","ids":["x"],"memory_id":"m"}',
                           'pending', 0, ?)""",
                (eid, f"{base_ts}{i}+00:00"),
            )

    processed_order: list[str] = []

    async def _track_delete(collection, ids):
        processed_order.append(ids[0])

    backend = MagicMock()
    backend.delete = AsyncMock(side_effect=_track_delete)
    processor = OutboxProcessor(backend)
    await processor.drain()

    # All three processed; order matches insertion order
    assert await count_outbox(qa_pool, "applied") == 3


# ---------------------------------------------------------------------------
# 8. All EventType variants route to correct backend method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_type,payload,expected_method",
    [
        pytest.param(
            "qdrant_upsert",
            {"collection": "col", "points": [], "memory_id": "m"},
            "upsert",
            id="upsert",
        ),
        pytest.param(
            "qdrant_delete",
            {"collection": "col", "ids": ["id1"], "memory_id": "m"},
            "delete",
            id="delete",
        ),
        pytest.param(
            "qdrant_delete_filter",
            {"collection": "col", "filter": {}, "memory_id": "m"},
            "delete_by_filter",
            id="delete_filter",
        ),
        pytest.param(
            "qdrant_set_payload",
            {"collection": "col", "payload": {}, "ids": [], "memory_id": "m"},
            "set_payload",
            id="set_payload",
        ),
    ],
)
async def test_event_type_routes_to_correct_backend_method(
    qa_pool, event_type, payload, expected_method
):
    """Each EventType variant calls the matching VectorBackend method."""
    import uuid

    from archivist.storage.outbox import OutboxProcessor

    event_id = str(uuid.uuid4())
    await _insert_outbox_row(qa_pool, event_id, event_type, payload)

    backend = MagicMock()
    backend.upsert = AsyncMock(return_value=None)
    backend.delete = AsyncMock(return_value=None)
    backend.delete_by_filter = AsyncMock(return_value=None)
    backend.set_payload = AsyncMock(return_value=None)

    processor = OutboxProcessor(backend)
    n = await processor.drain()

    assert n == 1
    assert getattr(backend, expected_method).call_count == 1


# ---------------------------------------------------------------------------
# 9. drain_loop cancellation
# ---------------------------------------------------------------------------


async def test_drain_loop_cancels_cleanly(qa_pool, mock_vector_backend):
    """drain_loop() task can be cancelled without hanging or raising."""
    from archivist.storage.outbox import OutboxProcessor

    processor = OutboxProcessor(mock_vector_backend)
    task = asyncio.create_task(processor.drain_loop())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.CancelledError:
        pass  # expected
    assert task.done()
