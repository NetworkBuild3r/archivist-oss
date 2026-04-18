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
    """fts_vector_simple (unstemmed) matches exact tokens like IP addresses."""
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
            "WHERE fts_vector_simple @@ to_tsquery('simple', '192') "
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
