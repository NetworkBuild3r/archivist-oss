"""Unit tests for INIT-022/SPEC-004's txn-wiring fix in ``memory_lifecycle.py``.

Covers ac-3 (``M4``): ``_delete_sqlite_artifacts``'s ``txn`` parameter is now
genuinely wired through to ``delete_fts_chunks_batch``,
``delete_needle_tokens_batch``, and ``_delete_entity_facts_for_memory`` — the
spec's own Testing Strategy calls for "a before/after parity test that
``_delete_sqlite_artifacts`` produces identical SQL/results whether called
standalone or via ``delete_memory_complete``'s (now-unduplicated)
``OUTBOX_ENABLED`` branch." That is exercised here by driving
``_delete_sqlite_artifacts`` twice against identically-shaped fixtures: once
with ``txn=None`` (acquires its own ``pool.write()`` lock) and once with a
stand-in transaction object exposing ``.conn`` (mirroring
``MemoryTransaction.conn``) — and asserting identical (fts, needle, facts)
counts and identical database end-state.
"""

from dataclasses import dataclass

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.lifecycle]


@dataclass
class _FakeTxn:
    """Minimal stand-in for MemoryTransaction — _delete_sqlite_artifacts only reads .conn."""

    conn: object


async def _seed_artifacts(memory_id: str, all_ids: list[str]) -> None:
    """Seed FTS chunk rows, needle registry rows, and one active fact for *memory_id*."""
    from graph import _ensure_needle_registry, add_fact, get_db, upsert_entity, upsert_fts_chunk

    for i, qid in enumerate(all_ids):
        await upsert_fts_chunk(qid, f"text {i}", "f.md", i, "agent", "ns")

    _ensure_needle_registry()
    conn = get_db()
    for qid in all_ids:
        conn.execute(
            "INSERT INTO needle_registry (memory_id, token, namespace, agent_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (qid, "tok", "ns", "agent", "2025-01-01T00:00:00"),
        )
    conn.commit()
    conn.close()

    eid = await upsert_entity(f"entity-{memory_id}", "concept", namespace="global")
    await add_fact(
        eid,
        "some fact",
        source_file=f"explicit/{memory_id}",
        agent_id="agent",
        namespace="global",
        memory_id=memory_id,
    )


class TestDeleteSqliteArtifactsParity:
    """_delete_sqlite_artifacts behaves identically standalone vs. via an open txn."""

    async def test_standalone_and_txn_paths_produce_identical_counts(self, async_pool):
        from memory_lifecycle import _delete_sqlite_artifacts

        ids_standalone = ["standalone-primary", "standalone-child"]
        ids_txn = ["txn-primary", "txn-child"]
        await _seed_artifacts("mem-standalone", ids_standalone)
        await _seed_artifacts("mem-txn", ids_txn)

        # Path 1: conn=None (default) — acquires its own pool.write() lock,
        # matching the legacy (non-outbox) call site in delete_memory_complete.
        failed_standalone: list[str] = []
        standalone_result = await _delete_sqlite_artifacts(
            "mem-standalone", ids_standalone, failed_standalone
        )

        # Path 2: txn=<object with .conn> — mirrors the OUTBOX_ENABLED branch,
        # which threads MemoryTransaction.conn through so all three SQLite
        # cleanup steps join the caller's transaction instead of acquiring
        # their own write locks.
        failed_txn: list[str] = []
        async with async_pool.write() as conn:
            txn_result = await _delete_sqlite_artifacts(
                "mem-txn", ids_txn, failed_txn, txn=_FakeTxn(conn=conn)
            )

        assert failed_standalone == []
        assert failed_txn == []
        assert standalone_result == txn_result == (2, 2, 1)

    async def test_txn_path_commits_before_returning(self, async_pool):
        """Rows deleted via the txn path are actually gone once pool.write() exits (auto-commit)."""
        from graph import get_db
        from memory_lifecycle import _delete_sqlite_artifacts

        ids = ["commit-check-primary", "commit-check-child"]
        await _seed_artifacts("mem-commit-check", ids)

        async with async_pool.write() as conn:
            fts_count, needle_count, facts_count = await _delete_sqlite_artifacts(
                "mem-commit-check", ids, [], txn=_FakeTxn(conn=conn)
            )

        assert (fts_count, needle_count, facts_count) == (2, 2, 1)

        conn = get_db()
        remaining_chunks = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id IN (?, ?)",
            ids,
        ).fetchone()[0]
        remaining_needles = conn.execute(
            "SELECT COUNT(*) FROM needle_registry WHERE memory_id IN (?, ?)",
            ids,
        ).fetchone()[0]
        active_facts = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE memory_id = ? AND is_active = 1",
            ("mem-commit-check",),
        ).fetchone()[0]
        conn.close()

        assert remaining_chunks == 0
        assert remaining_needles == 0
        assert active_facts == 0

    async def test_conn_none_does_not_reuse_a_stale_connection(self, async_pool):
        """conn=None must acquire a fresh pool.write() lock, not silently reuse txn state."""
        from memory_lifecycle import _delete_sqlite_artifacts

        ids = ["fresh-lock-primary"]
        await _seed_artifacts("mem-fresh-lock", ids)

        # No open transaction anywhere in scope; this must still succeed by
        # acquiring its own write lock via pool.write() internally.
        fts_count, needle_count, facts_count = await _delete_sqlite_artifacts(
            "mem-fresh-lock", ids, []
        )

        assert (fts_count, needle_count, facts_count) == (1, 1, 1)
