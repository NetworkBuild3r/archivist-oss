"""Integration tests for the PostgreSQL backend.

These tests require a real PostgreSQL server.  They are **skipped** unless the
``POSTGRES_TEST_DSN`` environment variable is set, so they never block CI for
SQLite-only deployments.

To run them locally::

    POSTGRES_TEST_DSN="postgresql://user:pw@localhost/archivist_test" \\
        pytest tests/integration/storage/test_postgres_backend.py -v

Requires: ``asyncpg`` installed (``pip install asyncpg``).
"""

from __future__ import annotations

import os

import pytest

POSTGRES_DSN = os.getenv("POSTGRES_TEST_DSN", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.storage,
    pytest.mark.skipif(
        not POSTGRES_DSN,
        reason="POSTGRES_TEST_DSN not set — skipping PostgreSQL integration tests",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def pg_backend():
    """Yield an initialized AsyncpgGraphBackend against the test database.

    Creates a fresh schema before the test and drops all test tables after.
    """
    pytest.importorskip("asyncpg")
    from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

    backend = AsyncpgGraphBackend()
    await backend.initialize(POSTGRES_DSN, min_size=2, max_size=5)

    # Minimal schema for tests (subset of schema_postgres.sql)
    async with backend.write() as conn:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS _test_entities (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                entity_type TEXT NOT NULL DEFAULT 'unknown'
            );
            CREATE TABLE IF NOT EXISTS _test_facts (
                id SERIAL PRIMARY KEY,
                entity_id INTEGER NOT NULL REFERENCES _test_entities (id),
                fact_text TEXT NOT NULL
            );
        """)

    yield backend

    # Teardown: drop test tables so the database stays clean.
    async with backend.write() as conn:
        await conn.execute("DROP TABLE IF EXISTS _test_facts")
        await conn.execute("DROP TABLE IF EXISTS _test_entities")

    await backend.close()


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


async def test_insert_and_fetchall(pg_backend):
    """Insert a row then fetch it back."""
    async with pg_backend.write() as conn:
        await conn.execute(
            "INSERT INTO _test_entities (name, entity_type) VALUES (?, ?)",
            ("Alice", "person"),
        )

    async with pg_backend.read() as conn:
        rows = await conn.fetchall(
            "SELECT name, entity_type FROM _test_entities WHERE name = ?",
            ("Alice",),
        )

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["entity_type"] == "person"


async def test_rollback_on_error(pg_backend):
    """A write that raises inside the context manager is rolled back."""
    with pytest.raises(ValueError, match="simulated error"):
        async with pg_backend.write() as conn:
            await conn.execute(
                "INSERT INTO _test_entities (name) VALUES (?)",
                ("RollbackTarget",),
            )
            raise ValueError("simulated error")

    async with pg_backend.read() as conn:
        rows = await conn.fetchall(
            "SELECT name FROM _test_entities WHERE name = ?",
            ("RollbackTarget",),
        )
    assert rows == []


async def test_executemany(pg_backend):
    """executemany inserts multiple rows in a single transaction."""
    rows_to_insert = [("Bob", "person"), ("Carol", "person"), ("Dave", "bot")]
    async with pg_backend.write() as conn:
        await conn.executemany(
            "INSERT INTO _test_entities (name, entity_type) VALUES (?, ?)",
            rows_to_insert,
        )

    async with pg_backend.read() as conn:
        rows = await conn.fetchall("SELECT name FROM _test_entities ORDER BY name")

    names = [r["name"] for r in rows]
    assert "Bob" in names
    assert "Carol" in names
    assert "Dave" in names


async def test_fetchone(pg_backend):
    """fetchone returns a single row."""
    async with pg_backend.write() as conn:
        await conn.execute(
            "INSERT INTO _test_entities (name, entity_type) VALUES (?, ?)",
            ("Erin", "org"),
        )

    async with pg_backend.read() as conn:
        row = await conn.fetchone(
            "SELECT name, entity_type FROM _test_entities WHERE name = ?",
            ("Erin",),
        )

    assert row is not None
    assert row["name"] == "Erin"


async def test_fk_constraint_enforced(pg_backend):
    """Foreign key constraint prevents orphan facts."""
    import asyncpg

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        async with pg_backend.write() as conn:
            await conn.execute(
                "INSERT INTO _test_facts (entity_id, fact_text) VALUES (?, ?)",
                (999999, "orphan fact"),
            )


async def test_concurrent_writes_no_lock_contention(pg_backend):
    """Multiple concurrent write() calls succeed without deadlock (MVCC)."""
    import asyncio

    async def _write_entity(name: str) -> None:
        async with pg_backend.write() as conn:
            await conn.execute(
                "INSERT INTO _test_entities (name) VALUES (?) ON CONFLICT (name) DO NOTHING",
                (name,),
            )

    names = [f"concurrent_{i}" for i in range(10)]
    await asyncio.gather(*[_write_entity(n) for n in names])

    async with pg_backend.read() as conn:
        rows = await conn.fetchall("SELECT name FROM _test_entities WHERE name LIKE 'concurrent_%'")
    assert len(rows) == 10


# ---------------------------------------------------------------------------
# FTS parity integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
async def pg_fts_backend():
    """Yield an AsyncpgGraphBackend with the full memory_chunks FTS schema."""
    pytest.importorskip("asyncpg")
    from pathlib import Path

    from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

    backend = AsyncpgGraphBackend()
    await backend.initialize(POSTGRES_DSN, min_size=2, max_size=5)

    schema_sql = (
        Path(__file__).parents[3] / "src/archivist/storage/schema_postgres.sql"
    ).read_text()

    await backend.execute_ddl(schema_sql)

    yield backend

    # Clean up the FTS test rows inserted during the test.
    async with backend.write() as conn:
        await conn.execute("DELETE FROM memory_chunks WHERE namespace = 'fts_parity_test'")

    await backend.close()


async def test_fts_tsvector_auto_generated(pg_fts_backend):
    """Inserting a row auto-populates fts_vector and fts_vector_simple."""
    async with pg_fts_backend.write() as conn:
        await conn.execute(
            "INSERT INTO memory_chunks "
            "(qdrant_id, text, file_path, chunk_index, namespace) "
            "VALUES (?, ?, ?, ?, ?)",
            ("fts-pg-01", "kubernetes pod deployment", "/f1", 0, "fts_parity_test"),
        )

    async with pg_fts_backend.read() as conn:
        rows = await conn.fetchall(
            "SELECT fts_vector, fts_vector_simple FROM memory_chunks WHERE qdrant_id = ?",
            ("fts-pg-01",),
        )

    assert len(rows) == 1
    assert rows[0]["fts_vector"] is not None
    assert rows[0]["fts_vector_simple"] is not None


async def test_fts_stemmed_search(pg_fts_backend):
    """fts_vector (english stemmed) matches stemmed variants."""
    async with pg_fts_backend.write() as conn:
        for i, text in enumerate(
            [
                "kubernetes pod deployment running",
                "unrelated content about cats and dogs",
                "deploy application to kubernetes cluster",
            ]
        ):
            await conn.execute(
                "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, namespace) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"fts-stem-{i}", text, f"/f{i}", i, "fts_parity_test"),
            )

    async with pg_fts_backend.read() as conn:
        rows = await conn.fetchall(
            "SELECT mc.qdrant_id, ts_rank_cd(fts_vector, query) * 32 AS score "
            "FROM memory_chunks mc, to_tsquery('english', 'kubernetes') query "
            "WHERE fts_vector @@ query AND namespace = 'fts_parity_test' "
            "ORDER BY score DESC"
        )

    qdrant_ids = [r["qdrant_id"] for r in rows]
    assert "fts-stem-0" in qdrant_ids
    assert "fts-stem-2" in qdrant_ids
    assert "fts-stem-1" not in qdrant_ids


async def test_fts_simple_exact_search(pg_fts_backend):
    """fts_vector_simple (unstemmed) matches exact tokens like IP addresses.

    The 'simple' config preserves tokens without stemming. An IP address like
    192.168.1.100 is stored as a single token; the query must use the full token.
    """
    async with pg_fts_backend.write() as conn:
        await conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, namespace) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "fts-exact-1",
                "server at 192.168.1.100 failed health check",
                "/e1",
                0,
                "fts_parity_test",
            ),
        )
        await conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, namespace) "
            "VALUES (?, ?, ?, ?, ?)",
            ("fts-exact-2", "no ip address mentioned here", "/e2", 0, "fts_parity_test"),
        )

    async with pg_fts_backend.read() as conn:
        rows = await conn.fetchall(
            "SELECT mc.qdrant_id "
            "FROM memory_chunks mc "
            # plainto_tsquery matches the full IP token stored by 'simple' config
            "WHERE fts_vector_simple @@ plainto_tsquery('simple', '192.168.1.100') "
            "AND namespace = 'fts_parity_test'"
        )

    qdrant_ids = [r["qdrant_id"] for r in rows]
    assert "fts-exact-1" in qdrant_ids
    assert "fts-exact-2" not in qdrant_ids


async def test_fts_scoring_in_bm25_range(pg_fts_backend):
    """ts_rank_cd * 32 produces scores in a range comparable to FTS5 BM25."""
    async with pg_fts_backend.write() as conn:
        await conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, namespace) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "fts-score-1",
                "kubernetes kubernetes kubernetes deployment",
                "/sc1",
                0,
                "fts_parity_test",
            ),
        )

    async with pg_fts_backend.read() as conn:
        rows = await conn.fetchall(
            "SELECT ts_rank_cd(fts_vector, to_tsquery('english', 'kubernetes')) * 32 AS score "
            "FROM memory_chunks WHERE qdrant_id = 'fts-score-1'"
        )

    assert len(rows) == 1
    score = float(rows[0]["score"])
    # FTS5 BM25 typically produces values in 0.5-30 range;
    # ts_rank_cd is 0-1 * 32 = 0-32.  We just verify it's in a reasonable range.
    assert 0 < score <= 32


# ---------------------------------------------------------------------------
# High-level graph function tests (entity/fact CRUD, needle registry)
# These tests exercise upsert_entity(), add_fact(), search_entities(), etc.
# against a real Postgres backend by patching the module-level pool singleton
# with the test backend.
# ---------------------------------------------------------------------------


@pytest.fixture()
async def pg_graph(monkeypatch):
    """Initialize a full Postgres graph schema and patch the pool singleton.

    After the test, reverts the pool and closes the connection.
    """
    pytest.importorskip("asyncpg")
    from pathlib import Path

    from archivist.storage import graph as _graph
    from archivist.storage import sqlite_pool as _sp
    from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

    backend = AsyncpgGraphBackend()
    await backend.initialize(POSTGRES_DSN, min_size=2, max_size=5)

    schema_sql = (
        Path(__file__).parents[3] / "src/archivist/storage/schema_postgres.sql"
    ).read_text()
    await backend.execute_ddl(schema_sql)

    # Pre-clean any stale test data from previous runs (FK order: children first)
    async with backend.write() as conn:
        await conn.execute("DELETE FROM relationships WHERE namespace = 'pg_graph_test'")
        await conn.execute("DELETE FROM needle_registry WHERE namespace = 'pg_graph_test'")
        await conn.execute("DELETE FROM facts WHERE namespace = 'pg_graph_test'")
        await conn.execute("DELETE FROM entities WHERE namespace = 'pg_graph_test'")

    original_pool = _sp.pool
    monkeypatch.setattr(_sp, "pool", backend)
    monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "postgres")

    # Reset schema guards so they don't try to call get_db()
    for attr in dir(_graph):
        obj = getattr(_graph, attr, None)
        if hasattr(obj, "reset") and callable(obj.reset):
            obj.reset()
    # Mark needle registry guard applied so it doesn't re-run SQLite DDL
    if hasattr(_graph, "_ensure_needle_registry"):
        _graph._ensure_needle_registry.applied = True  # type: ignore[attr-defined]

    yield backend

    # Clean up test data — FK order: children before parents
    async with backend.write() as conn:
        await conn.execute("DELETE FROM relationships WHERE namespace = 'pg_graph_test'")
        await conn.execute("DELETE FROM needle_registry WHERE namespace = 'pg_graph_test'")
        await conn.execute("DELETE FROM facts WHERE namespace = 'pg_graph_test'")
        await conn.execute("DELETE FROM entities WHERE namespace = 'pg_graph_test'")

    monkeypatch.setattr(_sp, "pool", original_pool)
    await backend.close()


async def test_upsert_entity_returns_id(pg_graph):
    """upsert_entity() returns a valid integer ID on Postgres."""
    from archivist.storage.graph import upsert_entity

    eid = await upsert_entity("PostgresTestEntity", "tool", namespace="pg_graph_test")
    assert isinstance(eid, int)
    assert eid > 0


async def test_upsert_entity_idempotent(pg_graph):
    """upsert_entity() returns the same ID on repeated calls (RETURNING id path)."""
    from archivist.storage.graph import upsert_entity

    eid1 = await upsert_entity("IdempotentEnt", "service", namespace="pg_graph_test")
    eid2 = await upsert_entity("IdempotentEnt", "service", namespace="pg_graph_test")
    assert eid1 == eid2
    assert eid1 > 0


async def test_upsert_entity_increments_mention_count(pg_graph):
    """upsert_entity() increments mention_count on repeated calls."""
    from archivist.storage.graph import upsert_entity

    eid = await upsert_entity("MentionEnt", "tool", namespace="pg_graph_test")
    await upsert_entity("MentionEnt", "tool", namespace="pg_graph_test")

    async with pg_graph.read() as conn:
        rows = await conn.fetchall(
            "SELECT mention_count FROM entities WHERE id = ?", (eid,)
        )
    assert rows[0]["mention_count"] == 2


async def test_upsert_entity_citext_case_insensitive(pg_graph):
    """citext column makes entity names case-insensitive on Postgres."""
    from archivist.storage.graph import upsert_entity

    eid_lower = await upsert_entity("citext_test_entity", "org", namespace="pg_graph_test")
    eid_upper = await upsert_entity("CITEXT_TEST_ENTITY", "org", namespace="pg_graph_test")
    # citext UNIQUE constraint treats both as the same row
    assert eid_lower == eid_upper


async def test_add_fact_returns_id(pg_graph):
    """add_fact() returns a valid integer ID on Postgres (RETURNING id path)."""
    from archivist.storage.graph import add_fact, upsert_entity

    eid = await upsert_entity("FactEntity", "tool", namespace="pg_graph_test")
    fid = await add_fact(eid, "Postgres is production-ready", namespace="pg_graph_test")
    assert isinstance(fid, int)
    assert fid > 0


async def test_add_fact_multiple_ids_unique(pg_graph):
    """Each add_fact() call returns a distinct ID."""
    from archivist.storage.graph import add_fact, upsert_entity

    eid = await upsert_entity("MultiFact", "service", namespace="pg_graph_test")
    fid1 = await add_fact(eid, "fact one", namespace="pg_graph_test")
    fid2 = await add_fact(eid, "fact two", namespace="pg_graph_test")
    assert fid1 != fid2


async def test_get_entity_facts_round_trip(pg_graph):
    """add_fact() + get_entity_facts() correctly stores and retrieves fact text."""
    from archivist.storage.graph import add_fact, get_entity_facts, upsert_entity

    eid = await upsert_entity("RoundTripEnt", "tool", namespace="pg_graph_test")
    await add_fact(
        eid,
        "Archivist supports PostgreSQL as a first-class backend",
        namespace="pg_graph_test",
    )

    facts = await get_entity_facts(eid)
    assert len(facts) >= 1
    texts = [f["fact_text"] for f in facts]
    assert any("PostgreSQL" in t for t in texts)


async def test_search_entities_returns_results(pg_graph):
    """search_entities() finds entities by name on Postgres (citext LIKE)."""
    from archivist.storage.graph import search_entities, upsert_entity

    await upsert_entity("SearchableEntity", "tool", namespace="pg_graph_test")
    results = await search_entities("SearchableEntity", namespace="pg_graph_test")
    assert len(results) >= 1
    assert any(r["name"].lower() == "searchableentity" for r in results)


async def test_search_entities_case_insensitive(pg_graph):
    """search_entities() is case-insensitive on Postgres via citext."""
    from archivist.storage.graph import search_entities, upsert_entity

    await upsert_entity("CaseSensTest", "tool", namespace="pg_graph_test")
    results = await search_entities("casesenstest", namespace="pg_graph_test")
    assert len(results) >= 1


async def test_add_relationship(pg_graph):
    """add_relationship() creates a relationship between two entities."""
    from archivist.storage.graph import add_relationship, upsert_entity

    src = await upsert_entity("RelSrc", "service", namespace="pg_graph_test")
    tgt = await upsert_entity("RelTgt", "service", namespace="pg_graph_test")
    # Should not raise
    await add_relationship(src, tgt, "depends_on", "RelSrc depends on RelTgt",
                           namespace="pg_graph_test")

    async with pg_graph.read() as conn:
        rows = await conn.fetchall(
            "SELECT relation_type FROM relationships "
            "WHERE source_entity_id = ? AND target_entity_id = ?",
            (src, tgt),
        )
    assert len(rows) >= 1
    assert rows[0]["relation_type"] == "depends_on"


async def test_fetchval_returning_id(pg_graph):
    """AsyncpgConnection.fetchval() correctly retrieves a RETURNING id value."""
    async with pg_graph.write() as conn:
        new_id = await conn.fetchval(
            "INSERT INTO entities "
            "(name, entity_type, first_seen, last_seen, namespace) "
            "VALUES (?, ?, ?, ?, ?) RETURNING id",
            (
                "FetchvalTestEnt",
                "tool",
                "2026-01-01T00:00:00",
                "2026-01-01T00:00:00",
                "pg_graph_test",
            ),
        )
    assert new_id is not None
    assert isinstance(new_id, int)
    assert new_id > 0


# ---------------------------------------------------------------------------
# Needle registry tests on Postgres
# ---------------------------------------------------------------------------


async def test_register_needle_tokens_upsert(pg_graph):
    """register_needle_tokens() is idempotent (INSERT OR REPLACE → ON CONFLICT DO UPDATE)."""
    from archivist.storage.graph import register_needle_tokens

    memory_id = "pg-needle-mem-001"
    text = "Deploy to 192.168.1.100 at 09:30 UTC cron job"
    await register_needle_tokens(memory_id, text, namespace="pg_graph_test")
    # Run twice — second call must not raise a uniqueness error
    await register_needle_tokens(memory_id, text, namespace="pg_graph_test")

    async with pg_graph.read() as conn:
        rows = await conn.fetchall(
            "SELECT token FROM needle_registry WHERE memory_id = ? AND namespace = ?",
            (memory_id, "pg_graph_test"),
        )
    assert len(rows) >= 1


async def test_lookup_needle_tokens(pg_graph):
    """lookup_needle_tokens() returns rows for registered tokens."""
    from archivist.storage.graph import lookup_needle_tokens, register_needle_tokens

    memory_id = "pg-needle-mem-002"
    text = "Server at 10.0.0.55 restarted at 14:45 UTC"
    await register_needle_tokens(memory_id, text, namespace="pg_graph_test")

    results = await lookup_needle_tokens("10.0.0.55", namespace="pg_graph_test")
    assert any(r.get("memory_id") == memory_id for r in results)


# ---------------------------------------------------------------------------
# INSERT OR IGNORE path via pool
# ---------------------------------------------------------------------------


async def test_insert_or_ignore_via_pool(pg_graph):
    """INSERT OR IGNORE is correctly translated to ON CONFLICT DO NOTHING on Postgres."""
    async with pg_graph.write() as conn:
        await conn.execute(
            "INSERT INTO entities "
            "(name, entity_type, first_seen, last_seen, namespace) "
            "VALUES (?, ?, ?, ?, ?)",
            ("IgnoreTestEnt", "tool", "2026-01-01T00:00:00", "2026-01-01T00:00:00", "pg_graph_test"),
        )

    # Second insert via OR IGNORE — must not raise
    async with pg_graph.write() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO entities "
            "(name, entity_type, first_seen, last_seen, namespace) "
            "VALUES (?, ?, ?, ?, ?)",
            ("IgnoreTestEnt", "tool", "2026-01-01T00:00:00", "2026-01-01T00:00:00", "pg_graph_test"),
        )

    async with pg_graph.read() as conn:
        rows = await conn.fetchall(
            "SELECT id FROM entities WHERE name = ? AND namespace = ?",
            ("IgnoreTestEnt", "pg_graph_test"),
        )
    # Exactly one row — the second insert was silently ignored
    assert len(rows) == 1
