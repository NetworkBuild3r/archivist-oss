"""Tests for the transactional outbox — unit, integration, and chaos scenarios.

Test coverage map (from Phase 3 plan):

Unit tests
----------
- test_event_type_serialization            EventType round-trip JSON
- test_outbox_event_payload_json           OutboxEvent.payload_json() compact output
- test_transaction_commit_writes_outbox    MemoryTransaction clean exit → outbox rows
- test_transaction_rollback_no_outbox      Exception inside transaction → no outbox rows
- test_outbox_processor_applies_pending    drain() → VectorBackend mock called, status 'applied'
- test_outbox_retry_on_transient_failure   First drain fails, second drain succeeds
- test_outbox_dead_letter_after_max        Always fails → status='dead', delete_failures row
- test_vector_backend_protocol_conforms    isinstance(QdrantVectorBackend(...), VectorBackend)
- test_enqueue_noop_when_disabled         enqueue_* no-ops when OUTBOX_ENABLED=False

Integration tests (require async_pool)
---------------------------------------
- test_transaction_enqueues_and_drains    Full cycle: enqueue → drain → backend called
- test_delete_outbox_atomicity            Enqueue QDRANT_DELETE → drain → delete called
- test_transaction_rollback_on_exception  Exception in txn body → zero outbox rows in DB

Chaos tests
-----------
- test_crash_before_drain_then_resume     Write txn, crash before drain, resume drain
- test_mid_drain_crash_idempotent         Event stuck in 'processing' → re-drain picks up
- test_concurrent_drains_safe             Two concurrent drains never double-apply
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.storage]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _init_outbox_schema(conn: aiosqlite.Connection) -> None:
    """Create the outbox + memory_points tables used by tests."""
    await conn.executescript("""
        CREATE TABLE IF NOT EXISTS outbox (
            id           TEXT PRIMARY KEY,
            event_type   TEXT NOT NULL,
            payload      TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            retry_count  INTEGER NOT NULL DEFAULT 0,
            last_attempt TEXT,
            created_at   TEXT NOT NULL,
            error        TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status, created_at);

        CREATE TABLE IF NOT EXISTS memory_points (
            memory_id   TEXT NOT NULL,
            qdrant_id   TEXT NOT NULL,
            point_type  TEXT NOT NULL DEFAULT 'primary',
            created_at  TEXT NOT NULL,
            PRIMARY KEY (memory_id, qdrant_id)
        );

        CREATE TABLE IF NOT EXISTS delete_failures (
            id          TEXT PRIMARY KEY,
            memory_id   TEXT NOT NULL,
            qdrant_ids  TEXT NOT NULL,
            error       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            resolved_at TEXT
        );
    """)
    await conn.commit()

async def _count_outbox(conn: aiosqlite.Connection, status: str | None = None) -> int:
    if status:
        cur = await conn.execute("SELECT COUNT(*) FROM outbox WHERE status=?", (status,))
    else:
        cur = await conn.execute("SELECT COUNT(*) FROM outbox")
    row = await cur.fetchone()
    return row[0]

# ---------------------------------------------------------------------------
# Unit tests (no async_pool required — use direct aiosqlite)
# ---------------------------------------------------------------------------

def test_event_type_serialization():
    """EventType values round-trip through JSON without loss."""
    from archivist.storage.outbox import EventType

    for et in EventType:
        serialised = json.dumps(et.value)
        recovered = EventType(json.loads(serialised))
        assert recovered == et

def test_outbox_event_payload_json():
    """OutboxEvent.payload_json() produces compact JSON with no extra whitespace."""
    from archivist.storage.outbox import EventType, OutboxEvent

    ev = OutboxEvent(
        event_type=EventType.QDRANT_UPSERT,
        payload={"collection": "col", "points": [], "memory_id": "abc"},
    )
    raw = ev.payload_json()
    assert isinstance(raw, str)
    parsed = json.loads(raw)
    assert parsed["collection"] == "col"
    # Compact: no trailing spaces in simple values
    assert " " not in raw.split(":")[0]

def test_vector_backend_protocol_conforms():
    """QdrantVectorBackend satisfies the VectorBackend Protocol at runtime."""
    from archivist.storage.backends import QdrantVectorBackend, VectorBackend

    fake_client = MagicMock()
    backend = QdrantVectorBackend(fake_client)
    assert isinstance(backend, VectorBackend)

def test_enqueue_noop_when_disabled():
    """When OUTBOX_ENABLED=False, enqueue_* calls do not accumulate events."""
    from archivist.storage.transaction import MemoryTransaction

    txn = MemoryTransaction(enabled=False)
    txn.enqueue_qdrant_upsert("col", [], memory_id="x")
    txn.enqueue_qdrant_delete("col", ["id1"], memory_id="x")
    txn.enqueue_qdrant_delete_filter("col", {}, memory_id="x")
    txn.enqueue_qdrant_set_payload("col", {}, [], memory_id="x")
    assert txn._events == []

def test_enqueue_accumulates_when_enabled():
    """When OUTBOX_ENABLED=True, enqueue_* calls add events to the queue."""
    from archivist.storage.outbox import EventType
    from archivist.storage.transaction import MemoryTransaction

    txn = MemoryTransaction(enabled=True)
    txn.enqueue_qdrant_upsert("col", [], memory_id="x")
    txn.enqueue_qdrant_delete("col", ["id1"], memory_id="x")
    assert len(txn._events) == 2
    assert txn._events[0].event_type == EventType.QDRANT_UPSERT
    assert txn._events[1].event_type == EventType.QDRANT_DELETE

# ---------------------------------------------------------------------------
# Integration tests (require async_pool fixture with outbox schema)
# ---------------------------------------------------------------------------

@pytest.fixture
async def outbox_pool(async_pool, tmp_path):
    """Extend the async_pool fixture with outbox + memory_points tables."""
    conn = async_pool._conn
    await _init_outbox_schema(conn)
    return async_pool

async def test_transaction_commit_writes_outbox(outbox_pool):
    """Clean MemoryTransaction exit → outbox rows committed with status='pending'."""
    from archivist.storage.transaction import MemoryTransaction

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_upsert("col", [], memory_id="mem1")
        txn.enqueue_qdrant_delete("col", ["id1"], memory_id="mem1")

    async with outbox_pool.read() as conn:
        count = await _count_outbox(conn, "pending")
    assert count == 2

async def test_transaction_rollback_no_outbox(outbox_pool):
    """Exception inside MemoryTransaction body → zero outbox rows."""
    from archivist.storage.transaction import MemoryTransaction

    with pytest.raises(ValueError, match="boom"):
        async with MemoryTransaction(enabled=True) as txn:
            txn.enqueue_qdrant_upsert("col", [], memory_id="mem2")
            raise ValueError("boom")

    async with outbox_pool.read() as conn:
        count = await _count_outbox(conn)
    assert count == 0

async def test_transaction_rollback_on_exception(outbox_pool):
    """Explicit exception in txn body rolls back both SQL and outbox inserts."""
    from archivist.storage.transaction import MemoryTransaction

    with pytest.raises(RuntimeError):
        async with MemoryTransaction(enabled=True) as txn:
            await txn.execute(
                "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at) VALUES (?, ?, ?, ?)",
                ("m1", "q1", "primary", "2026-01-01"),
            )
            txn.enqueue_qdrant_delete("col", ["q1"], memory_id="m1")
            raise RuntimeError("mid-transaction crash")

    async with outbox_pool.read() as conn:
        mp_cur = await conn.execute("SELECT COUNT(*) FROM memory_points")
        mp_count = (await mp_cur.fetchone())[0]
        ob_count = await _count_outbox(conn)
    assert mp_count == 0
    assert ob_count == 0

async def test_outbox_processor_applies_pending(outbox_pool, monkeypatch):
    """OutboxProcessor.drain() calls the VectorBackend and marks events 'applied'."""
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    # Arrange: enqueue one event
    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("test_col", ["id-abc"], memory_id="mem3")

    # Mock VectorBackend
    mock_backend = MagicMock()
    mock_backend.delete = AsyncMock(return_value=None)

    import archivist.core.config as _cfg

    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 5)
    monkeypatch.setattr(_cfg, "OUTBOX_BATCH_SIZE", 50)

    processor = OutboxProcessor(mock_backend)
    n = await processor.drain()

    assert n == 1
    mock_backend.delete.assert_called_once_with("test_col", ["id-abc"])

    async with outbox_pool.read() as conn:
        count = await _count_outbox(conn, "applied")
    assert count == 1

async def test_outbox_retry_on_transient_failure(outbox_pool, monkeypatch):
    """VectorBackend raises once → event retried, second drain succeeds."""
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", ["id-retry"], memory_id="mem4")

    call_count = 0

    async def _delete_side_effect(collection, ids):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("transient 503")

    mock_backend = MagicMock()
    mock_backend.delete = AsyncMock(side_effect=_delete_side_effect)

    import archivist.core.config as _cfg

    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 5)
    monkeypatch.setattr(_cfg, "OUTBOX_BATCH_SIZE", 50)

    processor = OutboxProcessor(mock_backend)

    # First drain: fails, event reset to pending with retry_count=1
    n1 = await processor.drain()
    assert n1 == 0

    async with outbox_pool.read() as conn:
        cur = await conn.execute("SELECT retry_count, status FROM outbox")
        row = await cur.fetchone()
    assert row[0] == 1
    assert row[1] == "pending"

    # Second drain: back-off timestamp must be in the past — manipulate directly
    conn = outbox_pool._conn
    await conn.execute("UPDATE outbox SET last_attempt='2000-01-01T00:00:00+00:00'")
    await conn.commit()

    n2 = await processor.drain()
    assert n2 == 1
    assert call_count == 2

async def test_outbox_dead_letter_after_max_retries(outbox_pool, monkeypatch):
    """After OUTBOX_MAX_RETRIES failures, status='dead' and delete_failures row written."""
    import archivist.core.config as _cfg
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 2)
    monkeypatch.setattr(_cfg, "OUTBOX_BATCH_SIZE", 50)

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", ["id-dead"], memory_id="mem5")

    mock_backend = MagicMock()
    mock_backend.delete = AsyncMock(side_effect=ConnectionError("always fails"))

    processor = OutboxProcessor(mock_backend)

    # Two drains are needed to exhaust OUTBOX_MAX_RETRIES=2 (retry_count hits 2 on drain 2)
    for _ in range(3):
        # Reset last_attempt to force immediate retry on still-pending rows
        conn = outbox_pool._conn
        await conn.execute(
            "UPDATE outbox SET last_attempt='2000-01-01T00:00:00+00:00' WHERE status='pending'"
        )
        await conn.commit()
        await processor.drain()

    async with outbox_pool.read() as conn:
        ob_cur = await conn.execute("SELECT status, retry_count FROM outbox")
        ob_row = await ob_cur.fetchone()
        df_cur = await conn.execute("SELECT COUNT(*) FROM delete_failures")
        df_count = (await df_cur.fetchone())[0]

    assert ob_row[0] == "dead"
    assert df_count == 1

# ---------------------------------------------------------------------------
# Chaos tests
# ---------------------------------------------------------------------------

async def test_crash_before_drain_then_resume(outbox_pool, monkeypatch):
    """Write txn succeeds, drain never runs → event survives; drain picks it up.

    Simulates process crash between SQLite commit and Qdrant apply.
    """
    import archivist.core.config as _cfg
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction

    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 5)
    monkeypatch.setattr(_cfg, "OUTBOX_BATCH_SIZE", 50)

    # Step 1: commit transaction (Qdrant write never happens — simulated crash)
    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", ["orphan-id"], memory_id="mem6")

    # Verify event persisted
    async with outbox_pool.read() as conn:
        count = await _count_outbox(conn, "pending")
    assert count == 1

    # Step 2: process restarts, drain picks up the surviving event
    mock_backend = MagicMock()
    mock_backend.delete = AsyncMock(return_value=None)

    processor = OutboxProcessor(mock_backend)
    n = await processor.drain()

    assert n == 1
    mock_backend.delete.assert_called_once_with("col", ["orphan-id"])

    async with outbox_pool.read() as conn:
        count = await _count_outbox(conn, "applied")
    assert count == 1

async def test_mid_drain_crash_idempotent(outbox_pool, monkeypatch):
    """Event stuck in 'processing' after a crash → re-drain skips it (processing guard).

    In production the stuck 'processing' event would need a watchdog to reset
    it back to 'pending' after a TTL.  This test verifies the processor does
    NOT double-apply events that are already in 'processing' state.
    """
    import archivist.core.config as _cfg
    from archivist.storage.outbox import OutboxProcessor

    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 5)
    monkeypatch.setattr(_cfg, "OUTBOX_BATCH_SIZE", 50)

    # Manually insert an event in 'processing' state (simulating a crash mid-drain)
    from datetime import UTC, datetime

    conn = outbox_pool._conn
    await conn.execute(
        """INSERT INTO outbox (id, event_type, payload, status, retry_count, created_at)
           VALUES ('stuck-event', 'qdrant_delete',
                   '{"collection":"col","ids":["s1"],"memory_id":"m"}',
                   'processing', 0, ?)""",
        (datetime.now(UTC).isoformat(),),
    )
    await conn.commit()

    mock_backend = MagicMock()
    mock_backend.delete = AsyncMock(return_value=None)

    processor = OutboxProcessor(mock_backend)
    n = await processor.drain()

    # Processor should find zero 'pending' rows → 0 applied, backend not called
    assert n == 0
    mock_backend.delete.assert_not_called()

async def test_concurrent_drains_safe(outbox_pool, monkeypatch):
    """Two concurrent drain calls never double-apply the same event.

    The 'processing' status claim is atomic — the second drain sees 0 pending.
    """
    import archivist.core.config as _cfg
    from archivist.storage.outbox import OutboxProcessor
    from archivist.storage.transaction import MemoryTransaction


    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 5)
    monkeypatch.setattr(_cfg, "OUTBOX_BATCH_SIZE", 50)

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", ["concurrent-id"], memory_id="mem7")

    apply_count = 0

    async def _counted_delete(collection, ids):
        nonlocal apply_count
        apply_count += 1
        await asyncio.sleep(0)  # yield to let the other drain run

    mock_backend = MagicMock()
    mock_backend.delete = AsyncMock(side_effect=_counted_delete)

    processor = OutboxProcessor(mock_backend)
    results = await asyncio.gather(
        processor.drain(),
        processor.drain(),
    )

    # One drain applies 1 event, the other finds 0 (event already 'processing')
    assert sum(results) == 1
    assert apply_count == 1
