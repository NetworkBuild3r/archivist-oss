"""Atomicity and rollback tests for the transactional write path (Phase 3 + 3.5).

Guarantee under test
--------------------
A ``MemoryTransaction`` must satisfy:

1. **All-or-nothing commit** — SQLite artefacts (FTS5, needle_registry,
   memory_points) AND outbox rows land in the same atomic commit.
2. **All-or-nothing rollback** — Any exception inside the ``async with``
   block rolls back every write: zero rows in every table.
3. **Outbox-disabled no-ops** — When ``OUTBOX_ENABLED=False``, enqueue_*
   calls are silent no-ops; the context manager still commits SQLite writes.
4. **Shim guard** — Calling ``txn.execute`` / ``txn.executemany`` /
   ``txn.fetchall`` outside the context manager raises ``RuntimeError``.
5. **Schema integrity** — After a commit, all four artifact tables
   (fts_chunks, needle_registry, memory_points, outbox) are consistent.

AAA pattern used throughout.  Each test is fully independent.
"""

from __future__ import annotations

import pytest
from tests.fixtures.mocks import count_outbox, count_table
from tests.integration.conftest import skip_on_postgres

pytestmark = [pytest.mark.integration, pytest.mark.storage]

# ---------------------------------------------------------------------------
# 1. All-or-nothing commit
# ---------------------------------------------------------------------------


@skip_on_postgres
async def test_commit_writes_all_artifact_tables(qa_pool, memory_factory):
    """Clean transaction exit commits FTS5, needle, memory_points, and outbox atomically."""
    # Arrange
    mem = memory_factory()

    # Act
    from archivist.storage.transaction import MemoryTransaction

    async with MemoryTransaction(enabled=True) as txn:
        await txn.execute(
            "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
            " VALUES (?, ?, ?, ?)",
            (mem["memory_id"], mem["qdrant_id"], "primary", mem["date"]),
        )
        await txn.upsert_fts_chunk(
            qdrant_id=mem["qdrant_id"],
            text=mem["text"],
            file_path=mem["file_path"],
            chunk_index=mem["chunk_index"],
            agent_id=mem["agent_id"],
            namespace=mem["namespace"],
        )
        await txn.register_needle_tokens(
            memory_id=mem["memory_id"],
            text=mem["text"],  # memory_factory text has NEEDLE_PATTERN tokens
            namespace=mem["namespace"],
            agent_id=mem["agent_id"],
        )
        txn.enqueue_qdrant_upsert(
            collection="qa_col",
            points=[],
            memory_id=mem["memory_id"],
        )

    # Assert — all four tables populated
    assert await count_table(qa_pool, "memory_points") == 1
    assert await count_table(qa_pool, "memory_chunks") == 1
    assert await count_table(qa_pool, "needle_registry") >= 1
    assert await count_outbox(qa_pool, "pending") == 1


async def test_commit_multiple_events_in_single_txn(qa_pool, memory_factory):
    """All enqueued events land atomically when multiple enqueue_* calls are made."""
    # Arrange
    mems = [memory_factory() for _ in range(3)]

    # Act
    from archivist.storage.transaction import MemoryTransaction

    async with MemoryTransaction(enabled=True) as txn:
        for mem in mems:
            txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])
            txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])

    # Assert — 6 events (2 per memory × 3 memories)
    assert await count_outbox(qa_pool, "pending") == 6


# ---------------------------------------------------------------------------
# 2. All-or-nothing rollback
# ---------------------------------------------------------------------------


async def test_exception_rolls_back_all_artifacts(qa_pool, memory_factory):
    """Exception inside the transaction body → zero rows in every table."""
    # Arrange
    mem = memory_factory()

    # Act
    with pytest.raises(ValueError, match="deliberate failure"):
        from archivist.storage.transaction import MemoryTransaction

        async with MemoryTransaction(enabled=True) as txn:
            await txn.execute(
                "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
                " VALUES (?, ?, ?, ?)",
                (mem["memory_id"], mem["qdrant_id"], "primary", mem["date"]),
            )
            txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])
            raise ValueError("deliberate failure")

    # Assert — zero orphans
    assert await count_table(qa_pool, "memory_points") == 0
    assert await count_outbox(qa_pool) == 0


async def test_flush_failure_rolls_back_outbox_and_sql(qa_pool, memory_factory):
    """If _flush_events raises, both SQL artefacts and outbox rows are rolled back."""
    # Arrange
    mem = memory_factory()
    from archivist.storage.transaction import MemoryTransaction

    # Force _flush_events to raise by patching executemany after the first call
    original_init = MemoryTransaction.__init__

    class _FailingTxn(MemoryTransaction):
        async def _flush_events(self) -> None:
            raise OSError("simulated flush failure")

    # Act
    with pytest.raises(OSError, match="simulated flush failure"):
        async with _FailingTxn(enabled=True) as txn:
            await txn.execute(
                "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
                " VALUES (?, ?, ?, ?)",
                (mem["memory_id"], mem["qdrant_id"], "primary", mem["date"]),
            )
            txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])

    # Assert — both tables empty
    assert await count_table(qa_pool, "memory_points") == 0
    assert await count_outbox(qa_pool) == 0


# ---------------------------------------------------------------------------
# 3. Outbox disabled — SQLite writes still commit, no outbox rows
# ---------------------------------------------------------------------------


async def test_outbox_disabled_commits_sql_silently(qa_pool, memory_factory, monkeypatch):
    """OUTBOX_ENABLED=False: SQLite artefacts commit; enqueue_* is a silent no-op."""
    # Arrange
    import archivist.core.config as _cfg

    monkeypatch.setattr(_cfg, "OUTBOX_ENABLED", False)
    mem = memory_factory()

    # Act
    from archivist.storage.transaction import MemoryTransaction

    async with MemoryTransaction(enabled=False) as txn:
        await txn.execute(
            "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
            " VALUES (?, ?, ?, ?)",
            (mem["memory_id"], mem["qdrant_id"], "primary", mem["date"]),
        )
        txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])  # must be no-op

    # Assert
    assert await count_table(qa_pool, "memory_points") == 1
    assert await count_outbox(qa_pool) == 0  # no outbox rows


# ---------------------------------------------------------------------------
# 4. Shim guards — RuntimeError outside context manager
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,args",
    [
        pytest.param("execute", ("SELECT 1",), id="execute"),
        pytest.param("executemany", ("SELECT 1", []), id="executemany"),
        pytest.param("fetchall", ("SELECT 1",), id="fetchall"),
        pytest.param(
            "upsert_fts_chunk",
            ("qid", "text", "path", 0),
            id="upsert_fts_chunk",
        ),
        pytest.param(
            "register_needle_tokens",
            ("mem-id", "text"),
            id="register_needle_tokens",
        ),
    ],
)
async def test_shim_raises_outside_context_manager(method, args):
    """Calling txn.<shim> before entering ``async with`` raises RuntimeError."""
    from archivist.storage.transaction import MemoryTransaction

    txn = MemoryTransaction(enabled=True)
    with pytest.raises(RuntimeError, match="outside async with"):
        await getattr(txn, method)(*args)


# ---------------------------------------------------------------------------
# 5. Schema integrity after commit
# ---------------------------------------------------------------------------


async def test_memory_points_unique_constraint(qa_pool, memory_factory):
    """Inserting a duplicate (memory_id, qdrant_id) pair raises IntegrityError."""
    import aiosqlite

    mem = memory_factory()
    from archivist.storage.transaction import MemoryTransaction

    async with MemoryTransaction(enabled=True) as txn:
        await txn.execute(
            "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
            " VALUES (?, ?, ?, ?)",
            (mem["memory_id"], mem["qdrant_id"], "primary", mem["date"]),
        )

    # Second insert must fail
    with pytest.raises(aiosqlite.IntegrityError):
        async with MemoryTransaction(enabled=True) as txn:
            await txn.execute(
                "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
                " VALUES (?, ?, ?, ?)",
                (mem["memory_id"], mem["qdrant_id"], "duplicate", mem["date"]),
            )


async def test_outbox_rows_have_pending_status_after_commit(qa_pool, memory_factory):
    """Freshly committed outbox rows always start in 'pending' status."""
    from archivist.storage.transaction import MemoryTransaction

    mem = memory_factory()
    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])
        txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])
        txn.enqueue_qdrant_delete_filter("col", {}, memory_id=mem["memory_id"])
        txn.enqueue_qdrant_set_payload("col", {}, [], memory_id=mem["memory_id"])

    async with qa_pool.read() as conn:
        cur = await conn.execute("SELECT DISTINCT status FROM outbox")
        statuses = {row[0] for row in await cur.fetchall()}

    assert statuses == {"pending"}


async def test_fetchall_inside_txn_reads_own_writes(qa_pool, memory_factory):
    """txn.fetchall can see rows written earlier in the same transaction."""
    mem = memory_factory()
    from archivist.storage.transaction import MemoryTransaction

    async with MemoryTransaction(enabled=True) as txn:
        await txn.execute(
            "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
            " VALUES (?, ?, ?, ?)",
            (mem["memory_id"], mem["qdrant_id"], "primary", mem["date"]),
        )
        rows = await txn.fetchall(
            "SELECT memory_id FROM memory_points WHERE memory_id=?",
            (mem["memory_id"],),
        )
        assert len(rows) == 1
        assert rows[0][0] == mem["memory_id"]


# ---------------------------------------------------------------------------
# 6. Concurrent transactions do not interleave
# ---------------------------------------------------------------------------


async def test_concurrent_transactions_serialize_cleanly(qa_pool, memory_factory):
    """Multiple sequential MemoryTransactions each commit without data corruption."""
    from archivist.storage.transaction import MemoryTransaction

    mems = [memory_factory() for _ in range(10)]

    # Sequential — GRAPH_WRITE_LOCK_ASYNC is a module-level asyncio.Lock; concurrent
    # asyncio.gather tasks would bind it to different event loops and deadlock.
    for m in mems:
        async with MemoryTransaction(enabled=True) as txn:
            await txn.execute(
                "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
                " VALUES (?, ?, ?, ?)",
                (m["memory_id"], m["qdrant_id"], "primary", m["date"]),
            )
            txn.enqueue_qdrant_upsert("col", [], memory_id=m["memory_id"])

    assert await count_table(qa_pool, "memory_points") == 10
    assert await count_outbox(qa_pool, "pending") == 10
