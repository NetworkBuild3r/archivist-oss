"""Unit tests for storage/graph.py hygiene fixes (INIT-022/SPEC-003).

Covers:
  - delete_hotness(): "no such table" short-circuit unchanged, other exceptions
    now logged at warning (H7).
  - The four ``_search_fts_*`` variants: shared helper extraction preserves
    per-backend SQL semantics and result shape (M14).
  - Schema-migration exception handling: narrowed ``except`` on the ALTER TABLE
    loops, and non-silent (debug-logged) ``except`` on the index-creation loop
    (M16).
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]


# ---------------------------------------------------------------------------
# Shared fakes — no real I/O, matching the unit-tier conftest.py contract.
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Minimal stand-in for an aiosqlite cursor."""

    def __init__(self, rows: list[dict] | None = None, rowcount: int = 0):
        self._rows = rows or []
        self.rowcount = rowcount

    async def fetchall(self):
        return list(self._rows)


class _FakeAsyncConn:
    """Minimal stand-in for the pooled aiosqlite connection.

    Raises *exc* on execute() if set; otherwise returns a cursor built from
    *rows* (for reads) or *rowcount* (for writes).
    """

    def __init__(self, rows: list[dict] | None = None, rowcount: int = 0, exc: Exception | None = None):
        self._rows = rows
        self._rowcount = rowcount
        self._exc = exc
        self.executed: list[tuple[str, tuple]] = []

    async def execute(self, sql: str, params: tuple = ()):
        self.executed.append((sql, params))
        if self._exc is not None:
            raise self._exc
        return _FakeCursor(rows=self._rows, rowcount=self._rowcount)


class _FakePool:
    """Minimal stand-in for ``archivist.storage.sqlite_pool.pool``."""

    def __init__(self, conn: _FakeAsyncConn):
        self._conn = conn

    @asynccontextmanager
    async def read(self):
        yield self._conn

    @asynccontextmanager
    async def write(self):
        yield self._conn


def _patch_pool(monkeypatch, conn: _FakeAsyncConn) -> None:
    monkeypatch.setattr("archivist.storage.sqlite_pool.pool", _FakePool(conn))


# Fixed fixture used by every FTS parity test — same shape the four variants
# all consume (only the score-sign convention differs by backend family).
_FIXTURE_ROWS = [
    {
        "qdrant_id": "mem-1",
        "file_path": "notes/a.md",
        "chunk_index": 0,
        "agent_id": "agent-1",
        "namespace": "ns1",
        "date": "2024-01-01",
        "memory_type": "note",
        "text": "the quick brown fox",
        "actor_id": "actor-1",
        "actor_type": "user",
        "importance": 0.7,
        "tier_label": "l2",
        "bm25_rank": -2.5,
    },
    {
        "qdrant_id": "mem-2",
        "file_path": "notes/b.md",
        "chunk_index": 1,
        "agent_id": "agent-1",
        "namespace": "ns1",
        "date": "2024-01-02",
        "memory_type": "note",
        "text": "jumps over the lazy dog",
        "actor_id": "actor-1",
        "actor_type": "user",
        "importance": 0.4,
        "tier_label": "l3",
        "bm25_rank": -0.8,
    },
]


class _MetricsRecorder:
    """Captures m.observe/m.inc calls without touching real metrics state."""

    def __init__(self):
        self.observed: list[tuple[str, float, dict | None]] = []
        self.incremented: list[tuple[str, dict | None]] = []

    def observe(self, name, value, labels=None):
        self.observed.append((name, value, labels))

    def inc(self, name, labels=None, value=1.0):
        self.incremented.append((name, labels))


@pytest.fixture
def metrics_recorder(monkeypatch):
    rec = _MetricsRecorder()
    monkeypatch.setattr("archivist.storage.graph.m.observe", rec.observe)
    monkeypatch.setattr("archivist.storage.graph.m.inc", rec.inc)
    return rec


# ---------------------------------------------------------------------------
# ac-1 (H7): delete_hotness() exception visibility
# ---------------------------------------------------------------------------


class TestDeleteHotness:
    async def test_no_such_table_still_returns_0_silently(self, monkeypatch, caplog):
        """Existing short-circuit behavior is unchanged: no log, returns 0."""
        from archivist.storage.graph import delete_hotness

        conn = _FakeAsyncConn(exc=sqlite3.OperationalError("no such table: memory_hotness"))
        _patch_pool(monkeypatch, conn)

        with caplog.at_level(logging.WARNING, logger="archivist.graph"):
            result = await delete_hotness("mem-1")

        assert result == 0
        assert "delete_hotness failed" not in caplog.text

    async def test_other_exception_logs_warning_and_returns_0(self, monkeypatch, caplog):
        """ac-1: a non-'no such table' error must be visible at warning level."""
        from archivist.storage.graph import delete_hotness

        conn = _FakeAsyncConn(exc=sqlite3.OperationalError("disk I/O error"))
        _patch_pool(monkeypatch, conn)

        with caplog.at_level(logging.WARNING, logger="archivist.graph"):
            result = await delete_hotness("mem-1")

        assert result == 0
        assert "delete_hotness failed" in caplog.text
        assert "mem-1" in caplog.text
        assert "disk I/O error" in caplog.text
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    async def test_successful_delete_returns_rowcount(self, monkeypatch, caplog):
        """Unrelated success path is unaffected by the logging fix."""
        from archivist.storage.graph import delete_hotness

        conn = _FakeAsyncConn(rowcount=1)
        _patch_pool(monkeypatch, conn)

        with caplog.at_level(logging.WARNING, logger="archivist.graph"):
            result = await delete_hotness("mem-1")

        assert result == 1
        assert "delete_hotness failed" not in caplog.text


# ---------------------------------------------------------------------------
# ac-2 (M14): shared FTS search helper — behavioral parity across variants
# ---------------------------------------------------------------------------


class TestFtsSearchSqliteFamily:
    async def test_search_fts_sqlite_negates_rank_into_bm25_score(
        self, monkeypatch, metrics_recorder
    ):
        from archivist.storage.graph import _search_fts_sqlite

        conn = _FakeAsyncConn(rows=_FIXTURE_ROWS)
        _patch_pool(monkeypatch, conn)

        results = await _search_fts_sqlite("fox", namespace="ns1")

        assert len(results) == 2
        assert results[0]["bm25_score"] == 2.5
        assert results[1]["bm25_score"] == 0.8
        assert "bm25_rank" not in results[0]
        # Query text and the filter param were threaded through correctly.
        sql, params = conn.executed[0]
        assert "memory_fts" in sql
        assert "MATCH ?" in sql
        assert params[0] == "fox"
        assert "ns1" in params

    async def test_search_fts_exact_sqlite_uses_exact_table_same_shape(
        self, monkeypatch, metrics_recorder
    ):
        from archivist.storage.graph import _search_fts_exact_sqlite

        conn = _FakeAsyncConn(rows=_FIXTURE_ROWS)
        _patch_pool(monkeypatch, conn)

        results = await _search_fts_exact_sqlite("fox", namespace="ns1")

        assert len(results) == 2
        # Same negate-score convention as the stemmed variant (shared helper).
        assert results[0]["bm25_score"] == 2.5
        assert results[1]["bm25_score"] == 0.8
        sql, _ = conn.executed[0]
        assert "memory_fts_exact" in sql

    async def test_sqlite_variants_produce_identical_result_shape(self, monkeypatch, metrics_recorder):
        """ac-2 parity check: stemmed and exact sqlite variants must agree on
        shape/score-sign for the same fixture, proving the shared helper didn't
        introduce a divergence between the two call sites."""
        from archivist.storage.graph import _search_fts_exact_sqlite, _search_fts_sqlite

        conn_a = _FakeAsyncConn(rows=_FIXTURE_ROWS)
        _patch_pool(monkeypatch, conn_a)
        stemmed = await _search_fts_sqlite("fox", namespace="ns1")

        conn_b = _FakeAsyncConn(rows=_FIXTURE_ROWS)
        _patch_pool(monkeypatch, conn_b)
        exact = await _search_fts_exact_sqlite("fox", namespace="ns1")

        assert [r["bm25_score"] for r in stemmed] == [r["bm25_score"] for r in exact]
        assert [r["qdrant_id"] for r in stemmed] == [r["qdrant_id"] for r in exact]
        assert set(stemmed[0].keys()) == set(exact[0].keys())

    async def test_metrics_recorded_with_sqlite_backend_label(self, monkeypatch, metrics_recorder):
        from archivist.storage.graph import _search_fts_sqlite

        conn = _FakeAsyncConn(rows=_FIXTURE_ROWS)
        _patch_pool(monkeypatch, conn)

        await _search_fts_sqlite("fox")

        assert len(metrics_recorder.observed) == 1
        assert metrics_recorder.observed[0][2] == {"backend": "sqlite"}
        assert metrics_recorder.incremented == [("archivist_fts_search_total", {"backend": "sqlite"})]

    async def test_sqlite_search_error_logs_warning_and_returns_empty(self, monkeypatch, caplog):
        from archivist.storage.graph import _search_fts_sqlite

        conn = _FakeAsyncConn(exc=RuntimeError("db locked"))
        _patch_pool(monkeypatch, conn)

        with caplog.at_level(logging.WARNING, logger="archivist.graph"):
            results = await _search_fts_sqlite("fox")

        assert results == []
        assert "FTS search failed" in caplog.text
        assert "db locked" in caplog.text

    async def test_exact_sqlite_search_error_uses_distinct_error_context(self, monkeypatch, caplog):
        from archivist.storage.graph import _search_fts_exact_sqlite

        conn = _FakeAsyncConn(exc=RuntimeError("db locked"))
        _patch_pool(monkeypatch, conn)

        with caplog.at_level(logging.WARNING, logger="archivist.graph"):
            results = await _search_fts_exact_sqlite("fox")

        assert results == []
        assert "FTS exact search failed" in caplog.text


class TestFtsSearchPostgresFamily:
    async def test_search_fts_postgres_keeps_rank_sign_and_uses_english_config(
        self, monkeypatch, metrics_recorder
    ):
        from archivist.storage.graph import _search_fts_postgres

        conn = _FakeAsyncConn(rows=_FIXTURE_ROWS)
        _patch_pool(monkeypatch, conn)

        results = await _search_fts_postgres(raw_query="fox", fts_mode="or", namespace="ns1")

        assert len(results) == 2
        # Postgres family does NOT negate the rank (unlike the sqlite family).
        assert results[0]["bm25_score"] == -2.5
        assert results[1]["bm25_score"] == -0.8
        sql, _ = conn.executed[0]
        assert "fts_vector" in sql
        assert "to_tsquery('english'" in sql

    async def test_search_fts_exact_postgres_uses_simple_config_and_simple_column(
        self, monkeypatch, metrics_recorder
    ):
        from archivist.storage.graph import _search_fts_exact_postgres

        conn = _FakeAsyncConn(rows=_FIXTURE_ROWS)
        _patch_pool(monkeypatch, conn)

        results = await _search_fts_exact_postgres(raw_query="fox")

        assert len(results) == 2
        assert results[0]["bm25_score"] == -2.5
        sql, _ = conn.executed[0]
        assert "fts_vector_simple" in sql
        assert "to_tsquery('simple'" in sql

    async def test_postgres_variants_produce_identical_result_shape(self, monkeypatch, metrics_recorder):
        """ac-2 parity check: stemmed and exact postgres variants agree on
        shape/score-sign for the same fixture."""
        from archivist.storage.graph import _search_fts_exact_postgres, _search_fts_postgres

        conn_a = _FakeAsyncConn(rows=_FIXTURE_ROWS)
        _patch_pool(monkeypatch, conn_a)
        stemmed = await _search_fts_postgres(raw_query="fox", fts_mode="or")

        conn_b = _FakeAsyncConn(rows=_FIXTURE_ROWS)
        _patch_pool(monkeypatch, conn_b)
        exact = await _search_fts_exact_postgres(raw_query="fox")

        assert [r["bm25_score"] for r in stemmed] == [r["bm25_score"] for r in exact]
        assert [r["qdrant_id"] for r in stemmed] == [r["qdrant_id"] for r in exact]

    async def test_empty_tsquery_short_circuits_without_touching_pool(self, monkeypatch, metrics_recorder):
        """raw_query that reduces to an empty tsquery must return [] without a
        pool round-trip (existing behavior, unchanged by the refactor)."""
        from archivist.storage.graph import _search_fts_postgres

        conn = _FakeAsyncConn(rows=_FIXTURE_ROWS)
        _patch_pool(monkeypatch, conn)

        # A single stopword-only query yields "" from the "and" builder.
        results = await _search_fts_postgres(raw_query="the", fts_mode="and")

        assert results == []
        assert conn.executed == []
        assert metrics_recorder.observed == []

    async def test_metrics_recorded_with_postgres_backend_label(self, monkeypatch, metrics_recorder):
        from archivist.storage.graph import _search_fts_postgres

        conn = _FakeAsyncConn(rows=_FIXTURE_ROWS)
        _patch_pool(monkeypatch, conn)

        await _search_fts_postgres(raw_query="fox", fts_mode="or")

        assert len(metrics_recorder.observed) == 1
        assert metrics_recorder.observed[0][2] == {"backend": "postgres"}
        assert metrics_recorder.incremented == [
            ("archivist_fts_search_total", {"backend": "postgres"})
        ]

    async def test_postgres_search_error_logs_warning_and_returns_empty(self, monkeypatch, caplog):
        from archivist.storage.graph import _search_fts_postgres

        conn = _FakeAsyncConn(exc=RuntimeError("connection reset"))
        _patch_pool(monkeypatch, conn)

        with caplog.at_level(logging.WARNING, logger="archivist.graph"):
            results = await _search_fts_postgres(raw_query="fox", fts_mode="or")

        assert results == []
        assert "FTS Postgres search failed" in caplog.text

    async def test_exact_postgres_search_error_uses_distinct_error_context(self, monkeypatch, caplog):
        from archivist.storage.graph import _search_fts_exact_postgres

        conn = _FakeAsyncConn(exc=RuntimeError("connection reset"))
        _patch_pool(monkeypatch, conn)

        with caplog.at_level(logging.WARNING, logger="archivist.graph"):
            results = await _search_fts_exact_postgres(raw_query="fox")

        assert results == []
        assert "FTS exact Postgres search failed" in caplog.text


class TestFtsSearchEdgeCases:
    """Existing edge-case behavior must survive the refactor untouched."""

    async def test_empty_result_set_returns_empty_list(self, monkeypatch, metrics_recorder):
        from archivist.storage.graph import _search_fts_sqlite

        conn = _FakeAsyncConn(rows=[])
        _patch_pool(monkeypatch, conn)

        results = await _search_fts_sqlite("no-match-anywhere")

        assert results == []


# ---------------------------------------------------------------------------
# ac-3 (M16): schema-migration exception handling
# ---------------------------------------------------------------------------


class _FakeMigrateConn:
    """Sync stand-in for the sqlite3.Connection used inside _migrate_schema()."""

    def __init__(self, raise_on_prefix: str | None = None, raise_exc: Exception | None = None):
        self.raise_on_prefix = raise_on_prefix
        self.raise_exc = raise_exc
        self.executed: list[str] = []

    def execute(self, sql: str, *args):
        self.executed.append(sql)
        if self.raise_on_prefix is not None and sql.startswith(self.raise_on_prefix) and self.raise_exc:
            raise self.raise_exc

    def commit(self):
        pass

    def close(self):
        pass


class TestMigrateSchemaExceptionNarrowing:
    def test_operational_error_on_alter_table_is_still_swallowed(self, monkeypatch):
        """Existing idempotent-DDL behavior is unchanged for the expected case."""
        from archivist.storage.graph import _migrate_schema

        conn = _FakeMigrateConn(
            raise_on_prefix="ALTER TABLE", raise_exc=sqlite3.OperationalError("duplicate column")
        )
        monkeypatch.setattr("archivist.storage.graph.get_db", lambda: conn)

        _migrate_schema()  # must not raise

        assert any(sql.startswith("ALTER TABLE") for sql in conn.executed)
        assert any(sql.startswith("CREATE INDEX") for sql in conn.executed)

    def test_non_operational_error_on_alter_table_now_propagates(self, monkeypatch):
        """ac-3: the redundant bare ``Exception`` arm is gone — a genuinely
        different error class is no longer silently swallowed alongside the
        expected OperationalError case."""
        from archivist.storage.graph import _migrate_schema

        conn = _FakeMigrateConn(raise_on_prefix="ALTER TABLE", raise_exc=TypeError("boom"))
        monkeypatch.setattr("archivist.storage.graph.get_db", lambda: conn)

        with pytest.raises(TypeError, match="boom"):
            _migrate_schema()

    def test_index_creation_already_exists_error_is_silent(self, monkeypatch, caplog):
        from archivist.storage.graph import _migrate_schema

        conn = _FakeMigrateConn(
            raise_on_prefix="CREATE INDEX", raise_exc=Exception("index already exists")
        )
        monkeypatch.setattr("archivist.storage.graph.get_db", lambda: conn)

        with caplog.at_level(logging.DEBUG, logger="archivist.graph"):
            _migrate_schema()  # must not raise

        assert "Index creation skipped" not in caplog.text

    def test_index_creation_real_failure_logs_debug(self, monkeypatch, caplog):
        """ac-3: a genuine index-creation failure (not 'already exists') must
        leave a trace instead of vanishing silently."""
        from archivist.storage.graph import _migrate_schema

        conn = _FakeMigrateConn(
            raise_on_prefix="CREATE INDEX", raise_exc=Exception("no such column: bogus")
        )
        monkeypatch.setattr("archivist.storage.graph.get_db", lambda: conn)

        with caplog.at_level(logging.DEBUG, logger="archivist.graph"):
            _migrate_schema()  # must not raise — index loop only logs, never re-raises

        assert "Index creation skipped" in caplog.text
        assert "no such column: bogus" in caplog.text
        assert any(rec.levelno == logging.DEBUG for rec in caplog.records)
