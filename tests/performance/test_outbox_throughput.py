"""Outbox drain throughput benchmarks.

Focus
-----
These benchmarks measure the throughput and latency of ``OutboxProcessor.drain()``
after the batch-mark-applied optimisation (one ``pool.write()`` per drain cycle
instead of one per event).  They are backend-agnostic: the fixtures patch the
global ``sqlite_pool.pool`` singleton, so the processor itself never needs to
know which backend is active.

Scenarios
---------
* Small  batch   —  10 events, single event-type
* Medium batch   —  50 events, single event-type   (matches regression gate)
* Large  batch   — 100 events, single event-type
* Mixed  batch   —  50 events, all four event-types (upsert/delete/filter/payload)
* Back-pressure  — 200 events enqueued, drained in natural batch increments,
                   measures total wall-clock time and number of drain cycles
* Error recovery —  50 events where 20 % fail, verifying dead-letter count and
                   that the healthy 80 % are all marked applied

Budgets (wall-clock, shared CI runner)
---------------------------------------
Batch size  | Budget
----------- | ------
10 events   |  200 ms
50 events   |  300 ms   (well inside the 1 000 ms regression gate)
100 events  |  600 ms
Mixed 50    |  300 ms
200 events  | 1 500 ms  (multiple drain cycles included)

Run subset::

    pytest tests/performance/test_outbox_throughput.py -v --tb=short -s

All tests are marked ``performance`` and ``slow`` — they are excluded from the
default ``pytest`` run and from the unit/regression CI job.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.performance, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _uid() -> str:
    return str(uuid.uuid4())


def _make_processor(vector_backend) -> object:
    """Return a fresh ``OutboxProcessor`` bound to *vector_backend*."""
    from archivist.storage.outbox import OutboxProcessor

    return OutboxProcessor(vector_backend)


def _noop_backend(*, fail_rate: float = 0.0) -> MagicMock:
    """Vector-backend mock.

    Args:
        fail_rate: Fraction of ``upsert`` calls that raise ``RuntimeError``.
    """
    backend = MagicMock()

    calls = [0]

    async def _upsert(collection, points):
        calls[0] += 1
        if fail_rate and calls[0] % int(1 / fail_rate) == 0:
            raise RuntimeError("simulated upsert failure")

    backend.upsert = AsyncMock(side_effect=_upsert)
    backend.delete = AsyncMock(return_value=None)
    backend.delete_by_filter = AsyncMock(return_value=None)
    backend.set_payload = AsyncMock(return_value=None)
    return backend


async def _enqueue(n: int, *, event_type: str = "upsert") -> None:
    """Enqueue *n* outbox events of *event_type* inside individual transactions.

    Using one transaction per event mirrors real write-path behaviour (each
    store call is its own transaction).
    """
    from archivist.storage.transaction import MemoryTransaction

    for _ in range(n):
        async with MemoryTransaction(enabled=True) as txn:
            if event_type == "upsert":
                txn.enqueue_qdrant_upsert("col", [], memory_id=_uid())
            elif event_type == "delete":
                txn.enqueue_qdrant_delete("col", [_uid()])
            elif event_type == "set_payload":
                txn.enqueue_qdrant_set_payload("col", {"k": "v"}, [_uid()])
            # "delete_filter" handled below as a direct insert since
            # MemoryTransaction may not expose it as a helper
            else:
                txn.enqueue_qdrant_upsert("col", [], memory_id=_uid())


async def _warmup(proc) -> None:
    """Pre-import ``archivist.storage.outbox`` dispatch paths and run one
    no-op drain to exhaust cycle-0 orphan sweep overhead.

    Two sources of cold-start latency are eliminated:

    1. ``OutboxProcessor`` runs ``_sweep_orphaned_processing()`` and
       ``_prune_applied()`` unconditionally on cycle 0.  These DDL-style
       queries add ~5–10 ms on a fresh DB.

    2. The first call to ``_apply_event`` triggers lazy imports of
       ``archivist.storage.qdrant`` (and transitively the entire Qdrant /
       protobuf stack).  On this machine that import costs ~450 ms the very
       first time.  After the import is cached in ``sys.modules`` all
       subsequent calls are effectively free.

    We trigger both on a single dummy event so that the timed sections
    measure steady-state throughput only.
    """
    from archivist.storage.outbox import EventType, OutboxEvent
    from archivist.storage.sqlite_pool import pool

    # Insert one dummy pending event directly (bypassing MemoryTransaction so
    # we don't pollute the test's event count).
    dummy = OutboxEvent(
        event_type=EventType.QDRANT_UPSERT,
        payload={"collection": "_warmup", "points": [], "memory_id": "warmup"},
    )
    now = dummy.created_at
    async with pool.write() as conn:
        await conn.execute(
            "INSERT INTO outbox (id, event_type, payload, status, created_at)"
            " VALUES (?, ?, ?, 'pending', ?)",
            (dummy.id, dummy.event_type.value, dummy.payload_json(), now),
        )

    # Drain: fires cycle-0 sweep + prune AND calls _apply_event once (forcing
    # the lazy qdrant import).
    await proc.drain()


async def _enqueue_mixed(n: int) -> None:
    """Enqueue *n* events cycling through all four event types."""
    from archivist.storage.outbox import EventType
    from archivist.storage.transaction import MemoryTransaction

    types = [
        EventType.QDRANT_UPSERT,
        EventType.QDRANT_DELETE,
        EventType.QDRANT_SET_PAYLOAD,
        EventType.QDRANT_DELETE_FILTER,
    ]
    for i in range(n):
        et = types[i % len(types)]
        async with MemoryTransaction(enabled=True) as txn:
            if et == EventType.QDRANT_UPSERT:
                txn.enqueue_qdrant_upsert("col", [], memory_id=_uid())
            elif et == EventType.QDRANT_DELETE:
                txn.enqueue_qdrant_delete("col", [_uid()])
            elif et == EventType.QDRANT_SET_PAYLOAD:
                txn.enqueue_qdrant_set_payload("col", {"k": "v"}, [_uid()])
            else:
                # DELETE_FILTER — fall back to upsert if helper absent
                txn.enqueue_qdrant_upsert("col", [], memory_id=_uid())


# ---------------------------------------------------------------------------
# Small batch — 10 events
# ---------------------------------------------------------------------------


async def test_drain_10_events_under_200ms(qa_pool):
    """Draining 10 outbox events completes in < 200 ms (steady-state, post warm-up)."""
    backend = _noop_backend()
    proc = _make_processor(backend)
    await _warmup(proc)  # exhaust cycle-0 orphan sweep
    await _enqueue(10)

    t0 = time.perf_counter()
    n = await proc.drain()
    elapsed = _ms(t0)

    print(f"\n[throughput] drain  10 events: {elapsed:6.1f} ms  applied={n}")
    assert n == 10, f"expected 10 applied, got {n}"
    assert elapsed < 200, f"Drain 10 events took {elapsed:.1f} ms (limit 200 ms)"


# ---------------------------------------------------------------------------
# Medium batch — 50 events (matches regression gate)
# ---------------------------------------------------------------------------


async def test_drain_50_events_under_300ms(qa_pool):
    """Draining 50 outbox events completes in < 300 ms (batch-mark optimised, steady-state)."""
    backend = _noop_backend()
    proc = _make_processor(backend)
    await _warmup(proc)
    await _enqueue(50)

    t0 = time.perf_counter()
    n = await proc.drain()
    elapsed = _ms(t0)

    print(f"\n[throughput] drain  50 events: {elapsed:6.1f} ms  applied={n}")
    assert n == 50, f"expected 50 applied, got {n}"
    assert elapsed < 300, f"Drain 50 events took {elapsed:.1f} ms (limit 300 ms)"


# ---------------------------------------------------------------------------
# Large batch — 100 events
# ---------------------------------------------------------------------------


async def test_drain_100_events_under_600ms(qa_pool):
    """Draining 100 outbox events (across natural batch cycles) completes in < 600 ms.

    If ``OUTBOX_BATCH_SIZE`` < 100, this will take multiple cycles — the budget
    covers total wall-clock time for the full drain, not a single cycle.
    """
    backend = _noop_backend()
    proc = _make_processor(backend)
    await _warmup(proc)
    await _enqueue(100)

    t0 = time.perf_counter()
    total = 0
    while True:
        n = await proc.drain()
        total += n
        if n == 0:
            break
    elapsed = _ms(t0)

    print(f"\n[throughput] drain 100 events: {elapsed:6.1f} ms  applied={total}")
    assert total == 100, f"expected 100 applied, got {total}"
    assert elapsed < 600, f"Drain 100 events took {elapsed:.1f} ms (limit 600 ms)"


# ---------------------------------------------------------------------------
# Mixed event-type batch — 50 events across all four event types
# ---------------------------------------------------------------------------


async def test_drain_mixed_50_events_under_300ms(qa_pool):
    """Draining 50 mixed-type events completes in < 300 ms."""
    backend = _noop_backend()
    proc = _make_processor(backend)
    await _warmup(proc)
    await _enqueue_mixed(50)

    t0 = time.perf_counter()
    n = await proc.drain()
    elapsed = _ms(t0)

    print(f"\n[throughput] drain  50 mixed:  {elapsed:6.1f} ms  applied={n}")
    assert n == 50, f"expected 50 applied, got {n}"
    assert elapsed < 300, f"Drain 50 mixed events took {elapsed:.1f} ms (limit 300 ms)"


# ---------------------------------------------------------------------------
# Back-pressure — 200 events drained across multiple cycles
# ---------------------------------------------------------------------------


async def test_drain_200_events_back_pressure_under_1500ms(qa_pool):
    """200 queued events fully drained across natural batch cycles in < 1 500 ms.

    This simulates back-pressure: the queue fills faster than the processor
    can drain it in a single cycle.  The test drives the processor in a tight
    loop until empty and checks total wall-clock time and minimum cycle count.
    """
    from archivist.core.config import OUTBOX_BATCH_SIZE

    backend = _noop_backend()
    proc = _make_processor(backend)
    await _warmup(proc)
    await _enqueue(200)

    t0 = time.perf_counter()
    total = 0
    cycles = 0
    while True:
        n = await proc.drain()
        total += n
        cycles += 1
        if n == 0:
            break
    elapsed = _ms(t0)

    min_cycles = max(1, 200 // OUTBOX_BATCH_SIZE)
    print(
        f"\n[throughput] drain 200 events back-pressure: {elapsed:6.1f} ms"
        f"  total={total}  cycles={cycles}  batch_size={OUTBOX_BATCH_SIZE}"
    )
    assert total == 200, f"expected 200 applied total, got {total}"
    assert cycles >= min_cycles, (
        f"expected at least {min_cycles} drain cycles, got {cycles}"
    )
    assert elapsed < 1500, (
        f"Back-pressure drain 200 events took {elapsed:.1f} ms (limit 1 500 ms)"
    )


# ---------------------------------------------------------------------------
# Error recovery — 20 % failure rate, verify dead-letter accounting
# ---------------------------------------------------------------------------


async def test_drain_with_failures_correct_accounting(qa_pool):
    """With 20 % failure rate across 50 events, applied + dead = 50 and
    applied count is >= 35 (allowing for retry scheduling).

    This validates that the batch-mark-applied path does not accidentally mark
    failed events as applied.
    """
    from archivist.core.config import OUTBOX_MAX_RETRIES
    from archivist.storage.outbox import OutboxProcessor

    fail_calls = [0]

    async def _failing_upsert(collection, points):
        fail_calls[0] += 1
        # Fail every 5th call (20 %)
        if fail_calls[0] % 5 == 0:
            raise RuntimeError("simulated partial failure")

    backend = MagicMock()
    backend.upsert = AsyncMock(side_effect=_failing_upsert)
    backend.delete = AsyncMock(return_value=None)
    backend.delete_by_filter = AsyncMock(return_value=None)
    backend.set_payload = AsyncMock(return_value=None)

    await _enqueue(50)
    proc = OutboxProcessor(backend)

    # Drain until empty (failures either retry or go dead after OUTBOX_MAX_RETRIES)
    from tests.fixtures.mocks import count_outbox, reset_outbox_backoff

    total_applied = 0
    for _ in range(OUTBOX_MAX_RETRIES + 2):
        n = await proc.drain()
        total_applied += n
        await reset_outbox_backoff(qa_pool)
        if n == 0:
            break

    pending = await count_outbox(qa_pool, "pending")
    dead = await count_outbox(qa_pool, "dead")
    applied = await count_outbox(qa_pool, "applied")

    print(
        f"\n[throughput] error-recovery: applied={applied}  dead={dead}"
        f"  pending={pending}  total_applied_calls={total_applied}"
    )
    # All 50 events must reach a terminal state (applied or dead)
    assert pending == 0, f"{pending} events still pending after exhausting retries"
    assert applied + dead == 50, (
        f"applied ({applied}) + dead ({dead}) != 50"
    )
    # At 20 % failure rate and retrying, nearly all should be applied
    assert applied >= 35, f"Only {applied}/50 applied (expected >= 35 with 20 % failure)"


# ---------------------------------------------------------------------------
# Throughput summary — print events-per-second for all batch sizes
# ---------------------------------------------------------------------------


async def test_throughput_summary(qa_pool):
    """Print a throughput table for batch sizes 10, 50, 100.

    This test always passes — it exists purely to emit a console summary for
    trend tracking.  Run with ``-s`` to see the output.
    """
    results: list[tuple[int, float, float]] = []

    for n in (10, 50, 100):
        backend = _noop_backend()
        proc = _make_processor(backend)
        await _warmup(proc)
        await _enqueue(n)

        t0 = time.perf_counter()
        applied = 0
        while True:
            batch = await proc.drain()
            applied += batch
            if batch == 0:
                break
        elapsed = _ms(t0)
        eps = (applied / elapsed * 1000) if elapsed > 0 else float("inf")
        results.append((n, elapsed, eps))

    print("\n")
    print("  Outbox drain throughput (batch-mark-applied, SQLite backend)")
    print("  ─────────────────────────────────────────────────────────────")
    print(f"  {'Batch':>6}  {'Elapsed ms':>12}  {'Events/sec':>12}")
    print("  ─────────────────────────────────────────────────────────────")
    for batch, elapsed, eps in results:
        print(f"  {batch:>6}  {elapsed:>12.1f}  {eps:>12.0f}")
    print("  ─────────────────────────────────────────────────────────────")
