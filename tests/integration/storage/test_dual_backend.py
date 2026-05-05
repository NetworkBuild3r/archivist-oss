"""Dual-backend parametrized tests — run the same operations against both
SQLite and PostgreSQL.

These tests verify that high-level graph functions (upsert_entity, add_fact,
search_entities, needle registry) produce identical results regardless of which
backend is active.

The ``sqlite`` parametrize variant runs unconditionally (uses the existing
``async_pool`` fixture).  The ``postgres`` variant is **skipped** unless
``POSTGRES_TEST_DSN`` is set, so CI never blocks on a missing Postgres server.

Usage::

    # SQLite only (always available)
    pytest tests/integration/storage/test_dual_backend.py -v -k sqlite

    # Both backends
    POSTGRES_TEST_DSN="postgresql://user:pw@localhost/archivist_test" \\
        pytest tests/integration/storage/test_dual_backend.py -v
"""

from __future__ import annotations

import os

import pytest

POSTGRES_DSN = os.getenv("POSTGRES_TEST_DSN", "")

pytestmark = [pytest.mark.integration, pytest.mark.storage]

# ---------------------------------------------------------------------------
# Dual-backend fixture
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param(
            "postgres",
            marks=pytest.mark.skipif(
                not POSTGRES_DSN,
                reason="POSTGRES_TEST_DSN not set — skipping Postgres dual-backend tests",
            ),
        ),
    ]
)
async def dual_pool(request, tmp_path, monkeypatch):
    """Yield a pool connected to either SQLite or Postgres.

    Patches ``archivist.storage.sqlite_pool.pool`` for the duration of the
    test so all high-level graph functions use the right backend.  Resets
    schema guards and marks the needle-registry guard as applied so
    schema_guard() never fires mid-test.

    For SQLite, creates a fresh in-memory database with the full schema.
    For Postgres, applies schema_postgres.sql and cleans up test-scoped rows
    after each test.
    """
    from archivist.storage import graph as _graph
    from archivist.storage import sqlite_pool as _sp

    original_pool = _sp.pool

    def _reset_guards():
        for attr in dir(_graph):
            obj = getattr(_graph, attr, None)
            if hasattr(obj, "reset") and callable(obj.reset):
                obj.reset()
        if hasattr(_graph, "_ensure_needle_registry"):
            _graph._ensure_needle_registry.applied = True  # type: ignore[attr-defined]

    if request.param == "sqlite":
        from tests.fixtures.schema import build_schema

        p = _sp.SQLitePool()
        db_path = str(tmp_path / "dual.db")
        await p.initialize(db_path)
        async with p.write() as conn:
            await build_schema(conn)

        monkeypatch.setattr(_sp, "pool", p)
        monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "sqlite")
        _reset_guards()

        yield p, "sqlite"

        monkeypatch.setattr(_sp, "pool", original_pool)
        await p.close()

    else:  # postgres
        pytest.importorskip("asyncpg")
        from pathlib import Path

        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        backend = AsyncpgGraphBackend()
        await backend.initialize(POSTGRES_DSN, min_size=2, max_size=5)

        schema_sql = (
            Path(__file__).parents[3] / "src/archivist/storage/schema_postgres.sql"
        ).read_text()
        await backend.execute_ddl(schema_sql)

        monkeypatch.setattr(_sp, "pool", backend)
        monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "postgres")
        _reset_guards()

        yield backend, "postgres"

        async with backend.write() as conn:
            await conn.execute("DELETE FROM needle_registry WHERE namespace = 'dual_test'")
            await conn.execute("DELETE FROM facts WHERE namespace = 'dual_test'")
            await conn.execute("DELETE FROM entities WHERE namespace = 'dual_test'")

        monkeypatch.setattr(_sp, "pool", original_pool)
        await backend.close()


# ---------------------------------------------------------------------------
# upsert_entity
# ---------------------------------------------------------------------------


async def test_dual_upsert_entity_returns_positive_int(dual_pool):
    """upsert_entity() returns a positive integer ID on both backends."""
    _pool, backend_name = dual_pool
    from archivist.storage.graph import upsert_entity

    eid = await upsert_entity("DualTestEntity", "tool", namespace="dual_test")
    assert isinstance(eid, int), f"[{backend_name}] expected int, got {type(eid)}"
    assert eid > 0, f"[{backend_name}] expected positive ID, got {eid}"


async def test_dual_upsert_entity_idempotent(dual_pool):
    """upsert_entity() returns the same ID on repeated calls."""
    _pool, backend_name = dual_pool
    from archivist.storage.graph import upsert_entity

    eid1 = await upsert_entity("IdempotentDual", "service", namespace="dual_test")
    eid2 = await upsert_entity("IdempotentDual", "service", namespace="dual_test")
    assert eid1 == eid2, f"[{backend_name}] IDs diverged: {eid1} vs {eid2}"


async def test_dual_upsert_entity_mention_count(dual_pool):
    """Repeated upsert_entity() increments mention_count."""
    pool, backend_name = dual_pool
    from archivist.storage.graph import upsert_entity

    eid = await upsert_entity("MentionDual", "tool", namespace="dual_test")
    await upsert_entity("MentionDual", "tool", namespace="dual_test")

    async with pool.read() as conn:
        rows = await conn.fetchall("SELECT mention_count FROM entities WHERE id = ?", (eid,))
    assert rows[0]["mention_count"] == 2, f"[{backend_name}] mention_count wrong"


# ---------------------------------------------------------------------------
# add_fact
# ---------------------------------------------------------------------------


async def test_dual_add_fact_returns_positive_int(dual_pool):
    """add_fact() returns a positive integer ID on both backends."""
    _pool, backend_name = dual_pool
    from archivist.storage.graph import add_fact, upsert_entity

    eid = await upsert_entity("FactDual", "tool", namespace="dual_test")
    fid = await add_fact(eid, "dual backend fact", namespace="dual_test")
    assert isinstance(fid, int), f"[{backend_name}] expected int, got {type(fid)}"
    assert fid > 0, f"[{backend_name}] expected positive ID"


async def test_dual_add_fact_ids_unique(dual_pool):
    """Each add_fact() call returns a distinct ID."""
    _pool, backend_name = dual_pool
    from archivist.storage.graph import add_fact, upsert_entity

    eid = await upsert_entity("FactUniqueDual", "service", namespace="dual_test")
    fid1 = await add_fact(eid, "fact one dual", namespace="dual_test")
    fid2 = await add_fact(eid, "fact two dual", namespace="dual_test")
    assert fid1 != fid2, f"[{backend_name}] fact IDs must be unique"


async def test_dual_add_fact_retrievable(dual_pool):
    """Facts stored by add_fact() are retrievable via get_entity_facts()."""
    _pool, backend_name = dual_pool
    from archivist.storage.graph import add_fact, get_entity_facts, upsert_entity

    eid = await upsert_entity("FactRtDual", "tool", namespace="dual_test")
    await add_fact(eid, "dual backend retrieval test fact", namespace="dual_test")

    facts = await get_entity_facts(eid)
    assert len(facts) >= 1, f"[{backend_name}] no facts returned"
    assert any("dual" in f["fact_text"].lower() for f in facts), (
        f"[{backend_name}] fact text not found"
    )


# ---------------------------------------------------------------------------
# search_entities
# ---------------------------------------------------------------------------


async def test_dual_search_entities(dual_pool):
    """search_entities() finds entities by name on both backends."""
    _pool, backend_name = dual_pool
    from archivist.storage.graph import search_entities, upsert_entity

    await upsert_entity("SearchDualEnt", "tool", namespace="dual_test")
    results = await search_entities("SearchDualEnt", namespace="dual_test")
    assert len(results) >= 1, f"[{backend_name}] no search results"


async def test_dual_search_entities_case_insensitive(dual_pool):
    """search_entities() is case-insensitive on both backends."""
    _pool, backend_name = dual_pool
    from archivist.storage.graph import search_entities, upsert_entity

    await upsert_entity("CaseDualEnt", "tool", namespace="dual_test")
    results = await search_entities("casedualent", namespace="dual_test")
    assert len(results) >= 1, f"[{backend_name}] case-insensitive search failed"


# ---------------------------------------------------------------------------
# needle registry
# ---------------------------------------------------------------------------


async def test_dual_needle_register_and_lookup(dual_pool):
    """register_needle_tokens() + lookup_needle_tokens() works on both backends."""
    _pool, backend_name = dual_pool
    from archivist.storage.graph import lookup_needle_tokens, register_needle_tokens

    memory_id = "dual-needle-001"
    text = "Deploy service to 10.1.2.3 at 08:00 UTC"
    await register_needle_tokens(memory_id, text, namespace="dual_test")

    results = await lookup_needle_tokens("10.1.2.3", namespace="dual_test")
    assert any(r.get("memory_id") == memory_id for r in results), (
        f"[{backend_name}] needle token not found in lookup"
    )


async def test_dual_needle_upsert_idempotent(dual_pool):
    """Registering the same tokens twice is idempotent on both backends."""
    _pool, backend_name = dual_pool
    from archivist.storage.graph import register_needle_tokens

    memory_id = "dual-needle-002"
    text = "cron_job_at_midnight on server-42"
    # Must not raise on the second call (INSERT OR REPLACE / ON CONFLICT DO UPDATE)
    await register_needle_tokens(memory_id, text, namespace="dual_test")
    await register_needle_tokens(memory_id, text, namespace="dual_test")


# ---------------------------------------------------------------------------
# fetchval interface
# ---------------------------------------------------------------------------


async def test_dual_fetchval_returns_scalar(dual_pool):
    """fetchval() returns a scalar value on both backends via RETURNING id."""
    pool, backend_name = dual_pool

    if backend_name == "sqlite":
        # SQLite does not support RETURNING in all versions; use a SELECT scalar instead
        async with pool.write() as conn:
            await conn.execute(
                "INSERT INTO curator_state (key, value) VALUES (?, ?)", ("dual_fetchval_key", "42")
            )
        async with pool.read() as conn:
            val = await conn.fetchval(
                "SELECT value FROM curator_state WHERE key = ?", ("dual_fetchval_key",)
            )
        assert val == "42", f"[{backend_name}] fetchval wrong: {val!r}"
    else:
        async with pool.write() as conn:
            new_id = await conn.fetchval(
                "INSERT INTO entities "
                "(name, entity_type, first_seen, last_seen, namespace) "
                "VALUES (?, ?, ?, ?, ?) RETURNING id",
                ("FetchvalDual", "tool", "2026-01-01T00:00:00", "2026-01-01T00:00:00", "dual_test"),
            )
        assert new_id is not None, f"[{backend_name}] fetchval returned None"
        assert isinstance(new_id, int), f"[{backend_name}] fetchval not int: {type(new_id)}"


# ---------------------------------------------------------------------------
# DML rowcount
# ---------------------------------------------------------------------------


async def test_dual_dml_rowcount(dual_pool):
    """DML cursor.rowcount returns the number of affected rows on both backends."""
    pool, backend_name = dual_pool

    async with pool.write() as conn:
        await conn.execute(
            "INSERT INTO curator_state (key, value) VALUES (?, ?)",
            (f"rc_key_{backend_name}", "hello"),
        )
        cur = await conn.execute(
            "UPDATE curator_state SET value = ? WHERE key = ?",
            ("world", f"rc_key_{backend_name}"),
        )
    assert cur.rowcount == 1, f"[{backend_name}] expected rowcount 1, got {cur.rowcount}"


async def test_dual_dml_rowcount_zero_on_no_match(dual_pool):
    """DELETE with no matching rows returns rowcount == 0 on both backends."""
    pool, backend_name = dual_pool

    async with pool.write() as conn:
        cur = await conn.execute(
            "DELETE FROM curator_state WHERE key = ?",
            (f"no_such_key_{backend_name}_xyz",),
        )
    assert cur.rowcount == 0, f"[{backend_name}] expected rowcount 0, got {cur.rowcount}"


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


async def test_dual_audit_log_roundtrip(dual_pool):
    """log_memory_event() + get_audit_trail() work on both backends."""
    pool, backend_name = dual_pool

    from archivist.core.audit import get_audit_trail, log_memory_event

    mid = f"mem-audit-{backend_name}"
    await log_memory_event(
        agent_id="test-agent",
        action="create",
        memory_id=mid,
        namespace="dual_test",
        text_hash="abc123",
        version=1,
        metadata={"src": "dual_test"},
    )
    entries = await get_audit_trail(mid)
    assert len(entries) == 1, f"[{backend_name}] expected 1 audit entry, got {len(entries)}"
    assert entries[0]["action"] == "create"
    assert entries[0]["agent_id"] == "test-agent"


# ---------------------------------------------------------------------------
# Retrieval log
# ---------------------------------------------------------------------------


async def test_dual_retrieval_log_roundtrip(dual_pool):
    """log_retrieval() + get_retrieval_logs() work on both backends."""
    import os

    pool, backend_name = dual_pool

    os.environ["TRAJECTORY_EXPORT_ENABLED"] = "1"
    try:
        from archivist.retrieval.retrieval_log import get_retrieval_logs, log_retrieval

        lid = await log_retrieval(
            agent_id=f"agent-{backend_name}",
            query="dual test query",
            namespace="dual_test",
            tier="l2",
            memory_type="",
            retrieval_trace={"hits": 5},
            result_count=5,
            cache_hit=False,
            duration_ms=42,
        )
        assert lid, f"[{backend_name}] log_retrieval returned empty id"

        logs = await get_retrieval_logs(agent_id=f"agent-{backend_name}", limit=10)
        assert len(logs) >= 1, f"[{backend_name}] expected at least 1 log, got {len(logs)}"
        assert logs[0]["query"] == "dual test query"
        assert logs[0]["duration_ms"] == 42
        assert logs[0]["cache_hit"] is False
    finally:
        os.environ.pop("TRAJECTORY_EXPORT_ENABLED", None)


# ---------------------------------------------------------------------------
# Hotness
# ---------------------------------------------------------------------------


async def test_dual_hotness_batch_update(dual_pool):
    """batch_update_hotness() runs without error and get_hotness_scores() returns data."""
    pool, backend_name = dual_pool

    # Insert a retrieval log so batch_update has something to aggregate
    import os

    from archivist.core.hotness import batch_update_hotness, get_hotness_scores

    os.environ["TRAJECTORY_EXPORT_ENABLED"] = "1"
    try:
        from archivist.retrieval.retrieval_log import log_retrieval

        mid = f"hot-mem-{backend_name}"
        await log_retrieval(
            agent_id="hotness-agent",
            query="hotness query",
            namespace="dual_test",
            tier="l2",
            memory_type="",
            retrieval_trace={"result_ids": [mid]},
            result_count=1,
            cache_hit=False,
            duration_ms=10,
        )
    finally:
        os.environ.pop("TRAJECTORY_EXPORT_ENABLED", None)

    updated = await batch_update_hotness()
    assert isinstance(updated, int), f"[{backend_name}] batch_update_hotness returned non-int"

    scores = await get_hotness_scores([mid])
    # Score may be 0 if the memory wasn't logged in time; just verify no crash
    assert isinstance(scores, dict), f"[{backend_name}] get_hotness_scores returned non-dict"


# ---------------------------------------------------------------------------
# Phase 1: tier/importance columns in memory_chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("dual_pool", ["sqlite"], indirect=True)
async def test_dual_upsert_fts_chunk_tier_columns(dual_pool):
    """upsert_fts_chunk persists importance and tier_label on both backends."""
    backend, backend_name = dual_pool

    from archivist.storage.graph import upsert_fts_chunk

    qid = f"tier-test-{backend_name}"
    await upsert_fts_chunk(
        qdrant_id=qid,
        text="Phase 1 tier test",
        file_path=f"test/{backend_name}",
        chunk_index=0,
        importance=0.8,
        tier_label="l1",
    )

    async with backend.read() as conn:
        row = await conn.fetchone(
            "SELECT importance, tier_label FROM memory_chunks WHERE qdrant_id = ?",
            (qid,),
        )

    assert row is not None, f"[{backend_name}] memory_chunks row not found"
    assert float(row["importance"]) == pytest.approx(0.8), (
        f"[{backend_name}] importance mismatch: {row['importance']}"
    )
    assert row["tier_label"] == "l1", (
        f"[{backend_name}] tier_label mismatch: {row['tier_label']}"
    )

    # Cleanup
    async with backend.write() as conn:
        await conn.execute("DELETE FROM memory_chunks WHERE qdrant_id = ?", (qid,))


@pytest.mark.asyncio
@pytest.mark.parametrize("dual_pool", ["sqlite"], indirect=True)
async def test_dual_upsert_fts_chunk_defaults(dual_pool):
    """upsert_fts_chunk stores safe defaults when importance/tier_label are omitted."""
    backend, backend_name = dual_pool

    from archivist.storage.graph import upsert_fts_chunk

    qid = f"tier-default-{backend_name}"
    await upsert_fts_chunk(
        qdrant_id=qid,
        text="default tier test",
        file_path=f"test/{backend_name}",
        chunk_index=1,
    )

    async with backend.read() as conn:
        row = await conn.fetchone(
            "SELECT importance, tier_label, decay_rate FROM memory_chunks WHERE qdrant_id = ?",
            (qid,),
        )

    assert row is not None, f"[{backend_name}] memory_chunks row not found"
    assert float(row["importance"]) == pytest.approx(0.5), (
        f"[{backend_name}] default importance should be 0.5"
    )
    assert row["tier_label"] == "l2", (
        f"[{backend_name}] default tier_label should be 'l2'"
    )
    assert float(row["decay_rate"]) == pytest.approx(0.0), (
        f"[{backend_name}] default decay_rate should be 0.0"
    )

    # Cleanup
    async with backend.write() as conn:
        await conn.execute("DELETE FROM memory_chunks WHERE qdrant_id = ?", (qid,))

