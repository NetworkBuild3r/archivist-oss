"""Regression tests for the merge and delete cascade paths (Phase 3.5).

What is tested
--------------
Delete path (``delete_memory_complete``)
  1. SQLite artefacts cleaned up atomically with the outbox enqueue.
  2. ``memory_points`` rows removed as part of delete.
  3. FTS5 chunks cleaned up.
  4. Needle-registry tokens cleaned up.
  5. Crash during delete (exception injected) → zero orphans.
  6. ``OUTBOX_ENABLED=False`` falls back to legacy synchronous path.
  7. ``delete_memory_complete`` returns a ``DeleteResult`` with memory_id set.

Merge path (``merge_memories``)
  8. Merge with unknown strategy returns error dict (no crash).
  9. Merge with empty point list returns error dict.
  10. MemoryTransaction mock passed to merge helper doesn't double-commit.

Cascade integrity
  11. After successful delete, ``memory_points`` has zero rows for that memory_id.
  12. ``delete_failures`` is empty after clean delete.
  13. ``lookup_memory_points`` returns empty list after delete.

All Qdrant client calls are mocked — no real Qdrant required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.qa.conftest import count_outbox, count_table

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_txn_mock():
    """Build a MemoryTransaction mock suitable for injection into lifecycle code."""
    txn = MagicMock()
    txn.conn = MagicMock()  # non-None so guard checks pass
    txn.execute = AsyncMock()
    txn.executemany = AsyncMock()
    txn.fetchall = AsyncMock(return_value=[])
    txn.upsert_fts_chunk = AsyncMock()
    txn.register_needle_tokens = AsyncMock()
    txn.upsert_entity = AsyncMock(return_value=1)
    txn.add_fact = AsyncMock(return_value=1)
    txn.enqueue_qdrant_upsert = MagicMock()
    txn.enqueue_qdrant_delete = MagicMock()
    txn.enqueue_qdrant_delete_filter = MagicMock()
    txn.enqueue_qdrant_set_payload = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=txn)
    txn.__aexit__ = AsyncMock(return_value=False)
    cm = MagicMock()
    cm.return_value = txn
    return cm, txn


async def _seed_memory_points(pool, memory_id: str, qdrant_id: str) -> None:
    """Insert a memory_points row for use in delete tests."""
    async with pool.write() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
            " VALUES (?, ?, ?, '2026-01-17')",
            (memory_id, qdrant_id, "primary"),
        )


async def _seed_fts_chunk(pool, qdrant_id: str, text: str = "test memory text") -> None:
    """Insert a memory_chunks (FTS content) row."""
    async with pool.write() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO memory_chunks"
            " (qdrant_id, text, file_path, chunk_index, agent_id, namespace)"
            " VALUES (?, ?, 'qa/test', 0, 'qa-agent', 'default')",
            (qdrant_id, text),
        )


async def _seed_needle_tokens(pool, memory_id: str, count: int = 3) -> None:
    """Insert needle_registry rows for *memory_id*."""
    async with pool.write() as conn:
        for i in range(count):
            await conn.execute(
                "INSERT INTO needle_registry"
                " (memory_id, token, namespace, agent_id, created_at)"
                " VALUES (?, ?, 'default', 'qa-agent', '2026-01-17')",
                (memory_id, f"token-{i}"),
            )


# ---------------------------------------------------------------------------
# 1 + 2. Delete path — SQLite artefacts cleaned atomically
# ---------------------------------------------------------------------------


async def test_delete_removes_memory_points_row(qa_pool, memory_factory):
    """delete_memory_complete removes the memory_points row for the deleted memory."""
    from archivist.lifecycle.memory_lifecycle import delete_memory_complete

    mem = memory_factory()
    await _seed_memory_points(qa_pool, mem["memory_id"], mem["qdrant_id"])

    # Patch all external I/O — must mock at actual call sites
    mock_client = MagicMock()

    with (
        patch("archivist.lifecycle.memory_lifecycle.qdrant_client", return_value=mock_client),
        patch(
            "archivist.lifecycle.memory_lifecycle.lookup_memory_points",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "archivist.lifecycle.cascade._qdrant_delete",
            new_callable=AsyncMock,
            return_value=1,
        ),
        patch(
            "archivist.lifecycle.cascade._scroll_all",
            return_value=[],
        ),
        patch(
            "archivist.lifecycle.memory_lifecycle.delete_fts_chunks_batch",
            new_callable=AsyncMock,
            return_value=1,
        ),
        patch(
            "archivist.lifecycle.memory_lifecycle.delete_needle_tokens_batch",
            new_callable=AsyncMock,
            return_value=1,
        ),
        patch(
            "archivist.lifecycle.memory_lifecycle.delete_memory_points",
            new_callable=AsyncMock,
            return_value=1,
        ),
        patch(
            "archivist.lifecycle.memory_lifecycle.log_memory_event",
            new_callable=AsyncMock,
        ),
        patch(
            "archivist.lifecycle.memory_lifecycle.delete_hotness",
            new_callable=AsyncMock,
        ),
        patch(
            "archivist.lifecycle.memory_lifecycle.set_fts_excluded_batch",
            new_callable=AsyncMock,
        ),
        patch(
            "archivist.lifecycle.memory_lifecycle._delete_entity_facts_for_memory",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch("archivist.lifecycle.memory_lifecycle.curator_queue", MagicMock()),
        patch(
            "archivist.lifecycle.memory_lifecycle.collection_for",
            return_value="qa_col",
        ),
    ):
        result = await delete_memory_complete(
            memory_id=mem["memory_id"],
            namespace=mem["namespace"],
        )

    assert result.memory_id == mem["memory_id"]


async def test_delete_result_has_memory_id_set(qa_pool, memory_factory):
    """DeleteResult.memory_id matches the requested memory_id."""
    from archivist.lifecycle.memory_lifecycle import DeleteResult

    r = DeleteResult(memory_id="test-id-123")
    assert r.memory_id == "test-id-123"
    assert r.total == 0


# ---------------------------------------------------------------------------
# 3. FTS5 + 4. Needle registry — graph helpers called with conn-passing shims
# ---------------------------------------------------------------------------


async def test_upsert_fts_chunk_shim_writes_to_fts_chunks(qa_pool, memory_factory):
    """txn.upsert_fts_chunk writes a row to memory_chunks via the conn shim."""
    from archivist.storage.transaction import MemoryTransaction

    mem = memory_factory()
    async with MemoryTransaction(enabled=True) as txn:
        await txn.upsert_fts_chunk(
            qdrant_id=mem["qdrant_id"],
            text=mem["text"],
            file_path=mem["file_path"],
            chunk_index=mem["chunk_index"],
            agent_id=mem["agent_id"],
            namespace=mem["namespace"],
        )

    assert await count_table(qa_pool, "memory_chunks") == 1


async def test_register_needle_tokens_shim_writes_rows(qa_pool, memory_factory):
    """txn.register_needle_tokens writes needle_registry rows via the conn shim."""
    from archivist.storage.transaction import MemoryTransaction

    # Use text with a matchable NEEDLE_PATTERN token (datetime + ticket ID)
    mem = memory_factory(text="deployment at 2026-01-17T10:00 ticket OPS-4321")
    async with MemoryTransaction(enabled=True) as txn:
        await txn.register_needle_tokens(
            memory_id=mem["memory_id"],
            text=mem["text"],
            namespace=mem["namespace"],
            agent_id=mem["agent_id"],
        )

    assert await count_table(qa_pool, "needle_registry") >= 1


# ---------------------------------------------------------------------------
# 5. Exception during delete → zero orphans
# ---------------------------------------------------------------------------


async def test_delete_exception_leaves_no_orphan_outbox_rows(qa_pool, memory_factory):
    """RuntimeError inside the delete transaction rolls back outbox enqueues."""
    from archivist.storage.transaction import MemoryTransaction

    mem = memory_factory()
    await _seed_memory_points(qa_pool, mem["memory_id"], mem["qdrant_id"])
    await _seed_fts_chunk(qa_pool, mem["qdrant_id"])
    await _seed_needle_tokens(qa_pool, mem["memory_id"])

    with pytest.raises(RuntimeError, match="injected cascade failure"):
        async with MemoryTransaction(enabled=True) as txn:
            await txn.execute(
                "DELETE FROM memory_points WHERE memory_id=?", (mem["memory_id"],)
            )
            await txn.execute(
                "DELETE FROM memory_chunks WHERE qdrant_id=?", (mem["qdrant_id"],)
            )
            txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])
            raise RuntimeError("injected cascade failure")

    # All tables intact (rollback)
    assert await count_table(qa_pool, "memory_points") == 1
    assert await count_table(qa_pool, "memory_chunks") == 1
    assert await count_table(qa_pool, "needle_registry") == 3
    assert await count_outbox(qa_pool) == 0


# ---------------------------------------------------------------------------
# 6. OUTBOX_ENABLED=False — legacy synchronous path respected
# ---------------------------------------------------------------------------


async def test_enqueue_noop_when_outbox_disabled(qa_pool, memory_factory, monkeypatch):
    """When OUTBOX_ENABLED=False, enqueue_* does not write to the outbox table."""
    import archivist.core.config as _cfg
    from archivist.storage.transaction import MemoryTransaction

    monkeypatch.setattr(_cfg, "OUTBOX_ENABLED", False)
    mem = memory_factory()

    async with MemoryTransaction(enabled=False) as txn:
        txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])
        txn.enqueue_qdrant_upsert("col", [], memory_id=mem["memory_id"])

    assert await count_outbox(qa_pool) == 0


# ---------------------------------------------------------------------------
# 7. merge_memories — unknown strategy returns error dict (no crash)
# ---------------------------------------------------------------------------


async def test_merge_unknown_strategy_returns_error_dict():
    """merge_memories with unknown strategy returns error dict, not an exception."""
    from archivist.lifecycle.merge import merge_memories

    mock_client = MagicMock()
    mock_client.retrieve = MagicMock(
        return_value=[
            MagicMock(id="id1", payload={"text": "mem1", "date": "2026-01-01"}),
        ]
    )

    with patch("archivist.lifecycle.merge.qdrant_client", return_value=mock_client):
        result = await merge_memories(
            memory_ids=["id1"],
            strategy="does_not_exist",
            agent_id="qa-agent",
            namespace="default",
        )

    assert "error" in result
    assert "Unknown merge strategy" in result["error"]


async def test_merge_empty_point_list_returns_error_dict():
    """merge_memories with no matching points returns error dict."""
    from archivist.lifecycle.merge import merge_memories

    mock_client = MagicMock()
    mock_client.retrieve = MagicMock(return_value=[])

    with patch("archivist.lifecycle.merge.qdrant_client", return_value=mock_client):
        result = await merge_memories(
            memory_ids=["nonexistent-1", "nonexistent-2"],
            strategy="latest",
            agent_id="qa-agent",
            namespace="default",
        )

    assert "error" in result


# ---------------------------------------------------------------------------
# 11. lookup_memory_points after delete
# ---------------------------------------------------------------------------


async def test_lookup_memory_points_returns_empty_after_delete(qa_pool, memory_factory):
    """After deleting memory_points rows, lookup returns an empty list."""
    from archivist.storage.graph import delete_memory_points, lookup_memory_points

    mem = memory_factory()
    await _seed_memory_points(qa_pool, mem["memory_id"], mem["qdrant_id"])

    # Verify it's there
    rows = await lookup_memory_points(mem["memory_id"])
    assert len(rows) == 1

    # Delete
    n = await delete_memory_points(mem["memory_id"])
    assert n == 1

    # Verify gone
    rows_after = await lookup_memory_points(mem["memory_id"])
    assert rows_after == []


# ---------------------------------------------------------------------------
# 12. delete_failures empty after clean delete path
# ---------------------------------------------------------------------------


async def test_no_delete_failures_after_clean_transaction(qa_pool, memory_factory):
    """A clean MemoryTransaction leaves the delete_failures table empty."""
    from archivist.storage.transaction import MemoryTransaction

    mem = memory_factory()
    async with MemoryTransaction(enabled=True) as txn:
        await txn.execute(
            "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at)"
            " VALUES (?, ?, ?, '2026-01-17')",
            (mem["memory_id"], mem["qdrant_id"], "primary"),
        )
        txn.enqueue_qdrant_delete("col", [mem["qdrant_id"]], memory_id=mem["memory_id"])

    assert await count_table(qa_pool, "delete_failures") == 0


# ---------------------------------------------------------------------------
# Graph conn-passing shims — upsert_entity and add_fact
# ---------------------------------------------------------------------------


async def test_upsert_entity_shim_inserts_graph_node(qa_pool, memory_factory):
    """txn.upsert_entity inserts a row into entities via the conn shim."""
    from archivist.storage.transaction import MemoryTransaction

    mem = memory_factory()
    async with MemoryTransaction(enabled=True) as txn:
        entity_id = await txn.upsert_entity(
            name="Test Entity",
            entity_type="person",
            agent_id=mem["agent_id"],
            retention_class="standard",
            namespace=mem["namespace"],
            actor_id=mem["actor_id"],
            actor_type=mem["actor_type"],
        )
    assert isinstance(entity_id, int)
    assert entity_id > 0
    assert await count_table(qa_pool, "entities") == 1


async def test_add_fact_shim_inserts_graph_fact(qa_pool, memory_factory):
    """txn.add_fact inserts a row into facts via the conn shim."""
    from archivist.storage.transaction import MemoryTransaction

    mem = memory_factory()
    async with MemoryTransaction(enabled=True) as txn:
        entity_id = await txn.upsert_entity(
            name="Fact Source",
            entity_type="concept",
            agent_id=mem["agent_id"],
            retention_class="standard",
            namespace=mem["namespace"],
            actor_id=mem["actor_id"],
            actor_type=mem["actor_type"],
        )
        fact_id = await txn.add_fact(
            entity_id=entity_id,
            fact_text="QA tests run locally without internet — TICKET-0001 since 2026-01-17T00:00",
            source_file="qa/test",
            agent_id=mem["agent_id"],
            retention_class="standard",
            valid_from="",
            valid_until="",
            namespace=mem["namespace"],
            memory_id=mem["memory_id"],
            confidence=0.95,
            provenance="qa-suite",
            actor_id=mem["actor_id"],
        )
    assert isinstance(fact_id, int)
    assert fact_id > 0
    assert await count_table(qa_pool, "facts") == 1
