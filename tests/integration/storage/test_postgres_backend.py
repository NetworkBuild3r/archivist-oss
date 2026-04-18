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
        rows = await conn.fetchall(
            "SELECT name FROM _test_entities ORDER BY name"
        )

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
                "INSERT INTO _test_entities (name) VALUES (?)"
                " ON CONFLICT (name) DO NOTHING",
                (name,),
            )

    names = [f"concurrent_{i}" for i in range(10)]
    await asyncio.gather(*[_write_entity(n) for n in names])

    async with pg_backend.read() as conn:
        rows = await conn.fetchall(
            "SELECT name FROM _test_entities WHERE name LIKE 'concurrent_%'"
        )
    assert len(rows) == 10
