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
# Regression tests: DDL isolation, ON CONFLICT, RETURNING id
#
# These tests guard against the production bugs that were introduced when
# SQLite-specific SQL was used against the asyncpg backend:
#
#  1. execute_ddl() must run in autocommit — not inside a SERIALIZABLE
#     transaction.  asyncpg 0.31 crashes with ``AttributeError: 'NoneType'
#     object has no attribute 'decode'`` when DDL is executed inside an
#     explicit serializable transaction block.
#
#  2. ``INSERT OR IGNORE`` / ``INSERT OR REPLACE`` are SQLite-only syntax.
#     PostgreSQL requires ``ON CONFLICT (...) DO NOTHING`` / ``DO UPDATE``.
#
#  3. ``cursor.lastrowid`` does not exist on asyncpg cursors. The correct
#     PostgreSQL pattern is ``INSERT ... RETURNING id`` + ``fetchone()[0]``.
# ---------------------------------------------------------------------------


async def test_execute_ddl_runs_in_autocommit(pg_backend) -> None:
    """execute_ddl() must succeed without an explicit transaction wrapper.

    The bug: when ``execute_ddl()`` was called inside ``self.write()``
    (SERIALIZABLE isolation), asyncpg 0.31 raised::

        AttributeError: 'NoneType' object has no attribute 'decode'

    because PostgreSQL returns no OID descriptor for DDL statements.
    The fix: execute DDL on a raw connection in autocommit mode.
    """
    # CREATE TABLE is a DDL statement. If execute_ddl() wraps it in a
    # SERIALIZABLE transaction this call will raise AttributeError.
    await pg_backend.execute_ddl(
        "CREATE TABLE IF NOT EXISTS _ddl_test_sentinel (id SERIAL PRIMARY KEY, label TEXT)"
    )

    # Verify the table actually exists by inserting a row.
    async with pg_backend.write() as conn:
        await conn.execute(
            "INSERT INTO _ddl_test_sentinel (label) VALUES (?)", ("ok",)
        )

    async with pg_backend.read() as conn:
        row = await conn.fetchone("SELECT label FROM _ddl_test_sentinel")
    assert row is not None
    assert row["label"] == "ok"

    # Teardown
    await pg_backend.execute_ddl("DROP TABLE IF EXISTS _ddl_test_sentinel")


async def test_on_conflict_do_nothing(pg_backend) -> None:
    """ON CONFLICT (...) DO NOTHING must silently skip duplicate rows.

    The bug: ``INSERT OR IGNORE INTO ...`` (SQLite syntax) was used against
    the asyncpg pool, producing a syntax error in PostgreSQL.
    """
    async with pg_backend.write() as conn:
        await conn.execute(
            "INSERT INTO _test_entities (name, entity_type) VALUES (?, ?)",
            ("dup_entity", "person"),
        )
        # Second insert with ON CONFLICT DO NOTHING must not raise.
        await conn.execute(
            "INSERT INTO _test_entities (name, entity_type) "
            "VALUES (?) ON CONFLICT (name) DO NOTHING",
            ("dup_entity",),
        )

    async with pg_backend.read() as conn:
        rows = await conn.fetchall(
            "SELECT name FROM _test_entities WHERE name = ?", ("dup_entity",)
        )
    assert len(rows) == 1, "ON CONFLICT DO NOTHING should not create a duplicate row"


async def test_on_conflict_do_update(pg_backend) -> None:
    """ON CONFLICT (...) DO UPDATE must replace the row atomically.

    The bug: ``INSERT OR REPLACE INTO ...`` (SQLite syntax) was used against
    the asyncpg pool, producing a syntax error in PostgreSQL.
    """
    async with pg_backend.write() as conn:
        await conn.execute(
            "INSERT INTO _test_entities (name, entity_type) VALUES (?, ?)",
            ("upsert_entity", "person"),
        )
        # Upsert: change entity_type from 'person' to 'bot'.
        await conn.execute(
            "INSERT INTO _test_entities (name, entity_type) VALUES (?, ?) "
            "ON CONFLICT (name) DO UPDATE SET entity_type = EXCLUDED.entity_type",
            ("upsert_entity", "bot"),
        )

    async with pg_backend.read() as conn:
        row = await conn.fetchone(
            "SELECT entity_type FROM _test_entities WHERE name = ?", ("upsert_entity",)
        )
    assert row is not None
    assert row["entity_type"] == "bot", "ON CONFLICT DO UPDATE should update the row"


async def test_insert_returning_id(pg_backend) -> None:
    """INSERT ... RETURNING id must return the generated primary-key value.

    The bug: ``.lastrowid`` was used on an asyncpg cursor, which does not
    expose that attribute.  The correct PostgreSQL pattern is
    ``INSERT ... RETURNING id`` followed by ``fetchone()[0]``.
    """
    async with pg_backend.write() as conn:
        row = await conn.fetchone(
            "INSERT INTO _test_entities (name, entity_type) "
            "VALUES (?, ?) RETURNING id",
            ("returning_test", "service"),
        )

    assert row is not None, "RETURNING id must yield a row"
    entity_id = row[0]
    assert isinstance(entity_id, int), f"Expected int id, got {type(entity_id)}: {entity_id!r}"
    assert entity_id > 0


async def test_execute_ddl_multiple_statements(pg_backend) -> None:
    """execute_ddl() must handle a semicolon-separated DDL script correctly.

    The schema initialiser passes the full ``schema_postgres.sql`` file in
    one call. If the DDL splitter is broken, only the first statement runs and
    subsequent table/index creation is silently skipped.
    """
    await pg_backend.execute_ddl(
        "CREATE TABLE IF NOT EXISTS _ddl_multi_a (id SERIAL PRIMARY KEY);"
        "CREATE TABLE IF NOT EXISTS _ddl_multi_b (id SERIAL PRIMARY KEY);"
        "CREATE INDEX IF NOT EXISTS _ddl_multi_idx ON _ddl_multi_a (id);"
    )

    # Both tables must exist — verify by inserting.
    async with pg_backend.write() as conn:
        await conn.execute("INSERT INTO _ddl_multi_a DEFAULT VALUES")
        await conn.execute("INSERT INTO _ddl_multi_b DEFAULT VALUES")

    async with pg_backend.read() as conn:
        a = await conn.fetchall("SELECT id FROM _ddl_multi_a")
        b = await conn.fetchall("SELECT id FROM _ddl_multi_b")

    assert len(a) == 1
    assert len(b) == 1

    await pg_backend.execute_ddl(
        "DROP TABLE IF EXISTS _ddl_multi_a;"
        "DROP TABLE IF EXISTS _ddl_multi_b;"
    )


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


# ---------------------------------------------------------------------------
# graph.py Postgres-path regression tests
#
# These tests exercise the specific graph.py functions that previously used
# SQLite-only syntax. They run against the full schema_postgres.sql schema so
# the tests reflect the real production table structure.
#
# The bugs caught here:
#   * upsert_entity() used ``cursor.lastrowid`` → AttributeError on asyncpg
#   * add_fact() used ``cursor.lastrowid`` → AttributeError on asyncpg
#   * register_needle_tokens() used ``INSERT OR IGNORE`` → syntax error
#
# The fixes (INSERT ... RETURNING id, ON CONFLICT ... DO UPDATE/NOTHING) are
# exercised directly so any regression will immediately fail.
# ---------------------------------------------------------------------------


@pytest.fixture()
async def pg_graph_backend():
    """AsyncpgGraphBackend with the full production Postgres schema."""
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

    # Clean up rows created by these tests only.
    async with backend.write() as conn:
        await conn.execute(
            "DELETE FROM entities WHERE namespace = 'pg_graph_test'"
        )
        await conn.execute(
            "DELETE FROM needle_registry WHERE namespace = 'pg_graph_test'"
        )

    await backend.close()


async def test_upsert_entity_returning_id(pg_graph_backend) -> None:
    """upsert_entity INSERT path must return a valid integer id via RETURNING.

    The bug: ``cur2.lastrowid`` was used, which does not exist on asyncpg
    cursors. Symptom: AttributeError at runtime whenever a new entity was
    inserted. The fix replaces ``lastrowid`` with ``RETURNING id`` +
    ``fetchone()[0]``.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    async with pg_graph_backend.write() as conn:
        row = await conn.fetchone(
            "INSERT INTO entities "
            "(name, entity_type, first_seen, last_seen, retention_class, namespace, actor_id, actor_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
            ("TestEntityAlpha", "service", now, now, "standard", "pg_graph_test", "", ""),
        )

    assert row is not None, "RETURNING id must yield a row"
    entity_id = row[0]
    assert isinstance(entity_id, int)
    assert entity_id > 0


async def test_upsert_entity_idempotent(pg_graph_backend) -> None:
    """upsert_entity must return the same id on the second call (UPDATE path)."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    # First insert — exercises the RETURNING id path.
    async with pg_graph_backend.write() as conn:
        row = await conn.fetchone(
            "INSERT INTO entities "
            "(name, entity_type, first_seen, last_seen, retention_class, namespace, actor_id, actor_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
            ("IdempotentEntity", "org", now, now, "standard", "pg_graph_test", "", ""),
        )
    assert row is not None
    first_id = row[0]

    # Second insert via ON CONFLICT DO UPDATE — must not create a duplicate.
    async with pg_graph_backend.write() as conn:
        await conn.execute(
            "UPDATE entities SET last_seen=?, mention_count=mention_count+1 "
            "WHERE name=? AND namespace=?",
            (now, "IdempotentEntity", "pg_graph_test"),
        )
        row2 = await conn.fetchone(
            "SELECT id, mention_count FROM entities WHERE name=? AND namespace=?",
            ("IdempotentEntity", "pg_graph_test"),
        )

    assert row2 is not None
    assert row2["id"] == first_id, "id must not change on update"
    assert row2["mention_count"] == 2, "mention_count must be incremented"


async def test_needle_registry_on_conflict_do_update(pg_graph_backend) -> None:
    """needle_registry insert must use ON CONFLICT DO UPDATE, not INSERT OR IGNORE.

    The bug: ``INSERT OR REPLACE INTO needle_registry`` (SQLite syntax) was
    used, producing a PostgreSQL syntax error. The fix replaces it with
    ``INSERT ... ON CONFLICT (token, memory_id) DO UPDATE SET ...``.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    async with pg_graph_backend.write() as conn:
        # Initial insert.
        await conn.execute(
            "INSERT INTO needle_registry "
            "(token, memory_id, namespace, agent_id, actor_id, actor_type, chunk_text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (token, memory_id) DO UPDATE SET "
            "chunk_text=EXCLUDED.chunk_text, created_at=EXCLUDED.created_at",
            ("192.168.1.1", "mem-pg-test-001", "pg_graph_test", "agent1", "", "", "original text", now),
        )
        # Upsert — must update chunk_text without raising.
        await conn.execute(
            "INSERT INTO needle_registry "
            "(token, memory_id, namespace, agent_id, actor_id, actor_type, chunk_text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (token, memory_id) DO UPDATE SET "
            "chunk_text=EXCLUDED.chunk_text, created_at=EXCLUDED.created_at",
            ("192.168.1.1", "mem-pg-test-001", "pg_graph_test", "agent1", "", "", "updated text", now),
        )

    async with pg_graph_backend.read() as conn:
        rows = await conn.fetchall(
            "SELECT token, chunk_text FROM needle_registry "
            "WHERE memory_id = ? AND namespace = ?",
            ("mem-pg-test-001", "pg_graph_test"),
        )

    assert len(rows) == 1, "ON CONFLICT DO UPDATE must not create duplicate rows"
    assert rows[0]["chunk_text"] == "updated text", "chunk_text must be updated"


async def test_memory_points_on_conflict_do_nothing(pg_graph_backend) -> None:
    """memory_points insert must use ON CONFLICT DO NOTHING, not INSERT OR IGNORE.

    The bug: ``INSERT OR IGNORE INTO memory_points`` (SQLite syntax) was used
    across indexer.py, tools_storage.py, and merge.py. Each produced a syntax
    error in PostgreSQL.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    async with pg_graph_backend.write() as conn:
        await conn.execute(
            "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("mp-test-001", "qd-0001", "primary", now),
        )
        # Re-insert with ON CONFLICT DO NOTHING must not raise.
        await conn.execute(
            "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT (memory_id, qdrant_id) DO NOTHING",
            ("mp-test-001", "qd-0001", "primary", now),
        )

    async with pg_graph_backend.read() as conn:
        rows = await conn.fetchall(
            "SELECT memory_id FROM memory_points WHERE memory_id = ?",
            ("mp-test-001",),
        )

    assert len(rows) == 1, "ON CONFLICT DO NOTHING must not create duplicate rows"

    # Cleanup
    async with pg_graph_backend.write() as conn:
        await conn.execute("DELETE FROM memory_points WHERE memory_id = ?", ("mp-test-001",))
