"""Unit tests for the pluggable GraphBackend infrastructure (Phase 4).

Covers:
  - SQL placeholder translation (? → $N)
  - SQLiteGraphBackend class renamed correctly + alias preserved
  - SQLiteGraphBackend satisfies GraphBackend protocol
  - AsyncpgGraphBackend satisfies GraphBackend protocol (mocked asyncpg)
  - create_graph_backend() factory dispatches based on GRAPH_BACKEND config
  - get_db() emits deprecation warning when GRAPH_BACKEND=postgres
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]


# ---------------------------------------------------------------------------
# SQL placeholder translation
# ---------------------------------------------------------------------------


class TestTranslateSql:
    """Tests for _translate_sql() in asyncpg_backend."""

    def _translate(self, sql: str) -> str:
        from archivist.storage.asyncpg_backend import _translate_sql

        return _translate_sql(sql)

    def test_no_placeholders(self):
        assert self._translate("SELECT 1") == "SELECT 1"

    def test_single_placeholder(self):
        assert self._translate("SELECT * FROM t WHERE id = ?") == "SELECT * FROM t WHERE id = $1"

    def test_multiple_placeholders(self):
        result = self._translate("INSERT INTO t (a, b, c) VALUES (?, ?, ?)")
        assert result == "INSERT INTO t (a, b, c) VALUES ($1, $2, $3)"

    def test_two_placeholders(self):
        result = self._translate("SELECT * FROM t WHERE a = ? AND b = ?")
        assert result == "SELECT * FROM t WHERE a = $1 AND b = $2"

    def test_placeholder_in_string_unchanged(self):
        """? inside a string literal should still be replaced (safe to translate)."""
        result = self._translate("SELECT ? as val")
        assert result == "SELECT $1 as val"

    def test_six_placeholders(self):
        sql = "INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)"
        result = self._translate(sql)
        assert result == "INSERT INTO t VALUES ($1, $2, $3, $4, $5, $6)"


# ---------------------------------------------------------------------------
# SQL dialect translation (Phase 2 additions)
# ---------------------------------------------------------------------------


class TestTranslateSqlDialects:
    """Tests for Phase 2 additions to _translate_sql(): INSERT OR IGNORE/REPLACE,
    COLLATE NOCASE stripping, and fetchval interface."""

    def _t(self, sql: str) -> str:
        from archivist.storage.asyncpg_backend import _translate_sql

        return _translate_sql(sql)

    # --- INSERT OR IGNORE ---

    def test_insert_or_ignore_basic(self):
        result = self._t("INSERT OR IGNORE INTO t (a) VALUES (?)")
        assert "INSERT OR IGNORE" not in result
        assert "INSERT INTO" in result
        assert "ON CONFLICT DO NOTHING" in result
        assert "$1" in result

    def test_insert_or_ignore_multi_col(self):
        result = self._t(
            "INSERT OR IGNORE INTO memory_points "
            "(memory_id, qdrant_id, point_type, created_at) VALUES (?, ?, ?, ?)"
        )
        assert "ON CONFLICT DO NOTHING" in result
        assert "$4" in result
        assert "$5" not in result

    def test_insert_or_ignore_case_insensitive(self):
        result = self._t("insert or ignore into t (x) values (?)")
        assert "ON CONFLICT DO NOTHING" in result

    # --- INSERT OR REPLACE ---

    def test_insert_or_replace_needle_registry(self):
        result = self._t(
            "INSERT OR REPLACE INTO needle_registry "
            "(token, memory_id, namespace, agent_id, actor_id, actor_type, chunk_text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        assert "INSERT OR REPLACE" not in result
        assert "INSERT INTO needle_registry" in result
        assert "ON CONFLICT (token, memory_id) DO UPDATE SET" in result
        # Conflict-target columns must NOT appear in SET clause
        assert "token=EXCLUDED" not in result
        assert "memory_id=EXCLUDED" not in result
        # Non-PK columns must appear
        assert "namespace=EXCLUDED.namespace" in result
        assert "chunk_text=EXCLUDED.chunk_text" in result

    def test_insert_or_replace_memory_hotness(self):
        result = self._t(
            "INSERT OR REPLACE INTO memory_hotness "
            "(memory_id, score, retrieval_count, last_accessed, updated_at) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        assert "ON CONFLICT (memory_id) DO UPDATE SET" in result
        assert "memory_id=EXCLUDED" not in result
        assert "score=EXCLUDED.score" in result
        assert "updated_at=EXCLUDED.updated_at" in result

    def test_insert_or_replace_no_columns_falls_back_to_nothing(self):
        """When no column list is parseable, fall back to DO NOTHING."""
        result = self._t("INSERT OR REPLACE INTO t VALUES (?, ?)")
        assert "ON CONFLICT" in result

    # --- COLLATE NOCASE ---

    def test_collate_nocase_stripped_from_where(self):
        result = self._t("SELECT * FROM t WHERE name = ? COLLATE NOCASE")
        assert "COLLATE NOCASE" not in result
        assert "$1" in result

    def test_collate_nocase_stripped_from_like(self):
        result = self._t(
            "SELECT * FROM entities "
            "WHERE name LIKE ? COLLATE NOCASE OR aliases LIKE ? COLLATE NOCASE "
            "ORDER BY mention_count DESC LIMIT ?"
        )
        assert "COLLATE NOCASE" not in result
        assert "$1" in result
        assert "$2" in result
        assert "$3" in result

    def test_collate_nocase_case_insensitive(self):
        result = self._t("WHERE x = ? collate nocase")
        assert "collate nocase" not in result.lower()

    def test_collate_nocase_in_create_table_not_stripped(self):
        """COLLATE NOCASE in DDL strings inside schema_guard is SQLite-only;
        the translator is not applied to DDL (only to runtime queries via conn methods),
        but we verify it strips correctly if it were applied."""
        result = self._t("CREATE TABLE t (name TEXT COLLATE NOCASE)")
        assert "COLLATE NOCASE" not in result

    # --- Combined transformations ---

    def test_insert_or_ignore_with_collate_nocase(self):
        """Both transforms apply in the same statement."""
        result = self._t(
            "INSERT OR IGNORE INTO t (name) SELECT name COLLATE NOCASE FROM src WHERE active = ?"
        )
        assert "INSERT OR IGNORE" not in result
        assert "COLLATE NOCASE" not in result
        assert "ON CONFLICT DO NOTHING" in result

    def test_plain_insert_unchanged(self):
        """Normal INSERT is not modified."""
        result = self._t("INSERT INTO t (a, b) VALUES (?, ?)")
        assert "ON CONFLICT" not in result
        assert result == "INSERT INTO t (a, b) VALUES ($1, $2)"

    # --- fetchval interface ---

    def test_asyncpg_connection_has_fetchval(self):
        from archivist.storage.asyncpg_backend import AsyncpgConnection

        assert hasattr(AsyncpgConnection, "fetchval")
        assert callable(AsyncpgConnection.fetchval)

    def test_wrapped_sqlite_conn_has_fetchval(self):
        from archivist.storage.sqlite_pool import _WrappedSQLiteConn

        assert hasattr(_WrappedSQLiteConn, "fetchval")
        assert callable(_WrappedSQLiteConn.fetchval)

    async def test_wrapped_sqlite_conn_fetchval_returns_scalar(self, tmp_path):
        """fetchval returns the first column of the first row."""
        import aiosqlite

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as raw_conn:
            await raw_conn.execute("CREATE TABLE x (val INTEGER)")
            await raw_conn.execute("INSERT INTO x VALUES (42)")
            await raw_conn.commit()

            from archivist.storage.sqlite_pool import _WrappedSQLiteConn

            wrapped = _WrappedSQLiteConn(raw_conn)
            result = await wrapped.fetchval("SELECT val FROM x")
            assert result == 42

    async def test_wrapped_sqlite_conn_fetchval_none_when_empty(self, tmp_path):
        """fetchval returns None when query returns no rows."""
        import aiosqlite

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as raw_conn:
            await raw_conn.execute("CREATE TABLE x (val INTEGER)")

            from archivist.storage.sqlite_pool import _WrappedSQLiteConn

            wrapped = _WrappedSQLiteConn(raw_conn)
            result = await wrapped.fetchval("SELECT val FROM x WHERE val = 999")
            assert result is None


# ---------------------------------------------------------------------------
# SQLiteGraphBackend class name and alias
# ---------------------------------------------------------------------------


class TestSQLiteGraphBackendAlias:
    """SQLitePool alias preserves backward compatibility."""

    def test_alias_is_same_class(self):
        from archivist.storage.sqlite_pool import SQLiteGraphBackend, SQLitePool

        assert SQLitePool is SQLiteGraphBackend

    def test_isinstance_check_works(self):
        from archivist.storage.sqlite_pool import SQLiteGraphBackend, SQLitePool

        instance = SQLitePool()
        assert isinstance(instance, SQLiteGraphBackend)

    def test_pool_singleton_is_sqlite_graph_backend(self):
        from archivist.storage.sqlite_pool import SQLiteGraphBackend, pool

        assert isinstance(pool, SQLiteGraphBackend)


# ---------------------------------------------------------------------------
# GraphBackend protocol conformance
# ---------------------------------------------------------------------------


class TestSQLiteGraphBackendProtocol:
    """SQLiteGraphBackend satisfies GraphBackend protocol."""

    def test_satisfies_protocol(self):
        from archivist.storage.backends import GraphBackend
        from archivist.storage.sqlite_pool import SQLiteGraphBackend

        backend = SQLiteGraphBackend()
        assert isinstance(backend, GraphBackend)

    def test_has_execute(self):
        from archivist.storage.sqlite_pool import SQLiteGraphBackend

        assert callable(getattr(SQLiteGraphBackend, "execute", None))

    def test_has_executemany(self):
        from archivist.storage.sqlite_pool import SQLiteGraphBackend

        assert callable(getattr(SQLiteGraphBackend, "executemany", None))

    def test_has_fetchall(self):
        from archivist.storage.sqlite_pool import SQLiteGraphBackend

        assert callable(getattr(SQLiteGraphBackend, "fetchall", None))

    def test_has_execute_ddl(self):
        from archivist.storage.sqlite_pool import SQLiteGraphBackend

        assert callable(getattr(SQLiteGraphBackend, "execute_ddl", None))

    def test_has_write(self):
        from archivist.storage.sqlite_pool import SQLiteGraphBackend

        assert callable(getattr(SQLiteGraphBackend, "write", None))

    def test_has_read(self):
        from archivist.storage.sqlite_pool import SQLiteGraphBackend

        assert callable(getattr(SQLiteGraphBackend, "read", None))


class TestAsyncpgGraphBackendProtocol:
    """AsyncpgGraphBackend satisfies GraphBackend protocol (structurally)."""

    def test_has_execute(self):
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        assert callable(getattr(AsyncpgGraphBackend, "execute", None))

    def test_has_executemany(self):
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        assert callable(getattr(AsyncpgGraphBackend, "executemany", None))

    def test_has_fetchall(self):
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        assert callable(getattr(AsyncpgGraphBackend, "fetchall", None))

    def test_has_execute_ddl(self):
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        assert callable(getattr(AsyncpgGraphBackend, "execute_ddl", None))

    def test_has_write(self):
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        assert callable(getattr(AsyncpgGraphBackend, "write", None))

    def test_has_read(self):
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        assert callable(getattr(AsyncpgGraphBackend, "read", None))


# ---------------------------------------------------------------------------
# AsyncpgGraphBackend — mocked asyncpg
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_asyncpg_pool():
    """Return a MagicMock that mimics asyncpg.Pool."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.executemany = AsyncMock()
    conn.fetch = AsyncMock(return_value=[{"id": 1}])
    conn.fetchrow = AsyncMock(return_value={"id": 1})

    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)

    pool_ctx = AsyncMock()
    pool_ctx.__aenter__ = AsyncMock(return_value=conn)
    pool_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=pool_ctx)

    return pool


class TestAsyncpgGraphBackendMocked:
    """Test AsyncpgGraphBackend using a mocked asyncpg pool."""

    async def test_write_yields_connection(self, mock_asyncpg_pool):
        from archivist.storage.asyncpg_backend import AsyncpgConnection, AsyncpgGraphBackend

        backend = AsyncpgGraphBackend()
        backend._pool = mock_asyncpg_pool

        async with backend.write() as conn:
            assert isinstance(conn, AsyncpgConnection)

    async def test_read_yields_connection(self, mock_asyncpg_pool):
        from archivist.storage.asyncpg_backend import AsyncpgConnection, AsyncpgGraphBackend

        backend = AsyncpgGraphBackend()
        backend._pool = mock_asyncpg_pool

        async with backend.read() as conn:
            assert isinstance(conn, AsyncpgConnection)

    async def test_execute_translates_placeholders(self, mock_asyncpg_pool):
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        backend = AsyncpgGraphBackend()
        backend._pool = mock_asyncpg_pool

        # For non-SELECT statements execute() calls raw conn.execute directly.
        await backend.execute("INSERT INTO t (a) VALUES (?)", ("val",))
        raw_conn = mock_asyncpg_pool.acquire().__aenter__.return_value
        call_args = raw_conn.execute.call_args
        assert call_args is not None
        assert "$1" in call_args[0][0]
        assert "?" not in call_args[0][0]

    async def test_fetchall_translates_placeholders(self, mock_asyncpg_pool):
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        backend = AsyncpgGraphBackend()
        backend._pool = mock_asyncpg_pool

        rows = await backend.fetchall("SELECT id FROM t WHERE name = ?", ("alice",))
        raw_conn = mock_asyncpg_pool.acquire().__aenter__.return_value
        call_args = raw_conn.fetch.call_args
        assert "$1" in call_args[0][0]
        assert isinstance(rows, list)

    async def test_write_raises_without_init(self):
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        backend = AsyncpgGraphBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            async with backend.write():
                pass

    async def test_read_raises_without_init(self):
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        backend = AsyncpgGraphBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            async with backend.read():
                pass

    async def test_initialize_raises_without_asyncpg(self):
        """ImportError is raised with a helpful message when asyncpg is missing."""
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        backend = AsyncpgGraphBackend()
        with patch("builtins.__import__", side_effect=ImportError("No module named 'asyncpg'")):
            with pytest.raises(ImportError, match="asyncpg is required"):
                await backend.initialize("postgresql://localhost/test")


# ---------------------------------------------------------------------------
# AsyncpgConnection
# ---------------------------------------------------------------------------


class TestAsyncpgConnection:
    """Unit tests for AsyncpgConnection wrapper methods."""

    def _make_conn(self) -> tuple:
        from archivist.storage.asyncpg_backend import AsyncpgConnection

        raw = AsyncMock()
        raw.execute = AsyncMock(return_value="OK")
        raw.executemany = AsyncMock()
        raw.fetch = AsyncMock(return_value=[])
        raw.fetchrow = AsyncMock(return_value=None)
        return AsyncpgConnection(raw), raw

    async def test_execute_translates(self):
        conn, raw = self._make_conn()
        await conn.execute("UPDATE t SET v = ? WHERE id = ?", (99, 1))
        raw.execute.assert_called_once_with("UPDATE t SET v = $1 WHERE id = $2", 99, 1)

    async def test_executemany_translates(self):
        conn, raw = self._make_conn()
        await conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
        raw.executemany.assert_called_once_with("INSERT INTO t VALUES ($1)", [(1,), (2,)])

    async def test_fetchall_translates(self):
        conn, raw = self._make_conn()
        raw.fetch.return_value = [{"id": 5}]
        result = await conn.fetchall("SELECT * FROM t WHERE id = ?", (5,))
        raw.fetch.assert_called_once_with("SELECT * FROM t WHERE id = $1", 5)
        assert result == [{"id": 5}]

    async def test_fetchone_translates(self):
        conn, raw = self._make_conn()
        raw.fetchrow.return_value = {"id": 7}
        result = await conn.fetchone("SELECT * FROM t WHERE id = ?", (7,))
        raw.fetchrow.assert_called_once_with("SELECT * FROM t WHERE id = $1", 7)
        assert result == {"id": 7}

    async def test_executescript_passes_full_script(self):
        conn, raw = self._make_conn()
        script = "CREATE TABLE a (id INT); CREATE TABLE b (id INT);"
        await conn.executescript(script)
        # executescript passes the full script as a single call
        raw.execute.assert_called_once_with(script)

    async def test_execute_dml_rowcount_parsed(self):
        """DML execute() returns a proxy with rowcount parsed from asyncpg status string."""
        from archivist.storage.asyncpg_backend import AsyncpgConnection

        raw = AsyncMock()
        raw.execute = AsyncMock(return_value="DELETE 5")
        conn = AsyncpgConnection(raw)
        cur = await conn.execute("DELETE FROM t WHERE id = ?", (42,))
        assert cur.rowcount == 5

    async def test_execute_insert_rowcount_parsed(self):
        """INSERT status 'INSERT 0 1' → rowcount == 1."""
        from archivist.storage.asyncpg_backend import AsyncpgConnection

        raw = AsyncMock()
        raw.execute = AsyncMock(return_value="INSERT 0 1")
        conn = AsyncpgConnection(raw)
        cur = await conn.execute("INSERT INTO t (v) VALUES (?)", (1,))
        assert cur.rowcount == 1

    async def test_execute_select_rowcount_is_minus_one(self):
        """SELECT proxy has rowcount == -1 (no DML run yet, matches DB-API 2.0)."""
        from archivist.storage.asyncpg_backend import AsyncpgConnection

        raw = AsyncMock()
        raw.fetch = AsyncMock(return_value=[])
        conn = AsyncpgConnection(raw)
        cur = await conn.execute("SELECT * FROM t")
        assert cur.rowcount == -1

    async def test_execute_dml_rowcount_zero_on_bad_status(self):
        """Unparseable status string does not crash; rowcount falls back to 0."""
        from archivist.storage.asyncpg_backend import AsyncpgConnection

        raw = AsyncMock()
        raw.execute = AsyncMock(return_value="COMMAND OK")
        conn = AsyncpgConnection(raw)
        cur = await conn.execute("UPDATE t SET v=1")
        assert cur.rowcount == 0

    async def test_commit_is_noop(self):
        conn, raw = self._make_conn()
        await conn.commit()  # should not raise

    async def test_rollback_is_noop(self):
        conn, raw = self._make_conn()
        await conn.rollback()  # should not raise


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


class TestCreateGraphBackend:
    """Tests for create_graph_backend() factory dispatch."""

    async def test_sqlite_backend_is_default(self, tmp_path):
        """GRAPH_BACKEND=sqlite (default) returns SQLiteGraphBackend."""
        db_path = str(tmp_path / "test.db")

        # The factory imports from archivist.core.config at call time, so we
        # patch the config module's re-exported names.
        with (
            patch("archivist.core.config.GRAPH_BACKEND", "sqlite"),
            patch("archivist.core.config.SQLITE_PATH", db_path),
            patch("archivist.core.config.SQLITE_WAL_AUTOCHECKPOINT", 1000),
        ):
            from archivist.storage.backend_factory import create_graph_backend
            from archivist.storage.sqlite_pool import SQLiteGraphBackend

            backend = await create_graph_backend()
            assert isinstance(backend, SQLiteGraphBackend)
            await backend.close()

    async def test_postgres_backend_dispatched_when_configured(self):
        """GRAPH_BACKEND=postgres calls _create_postgres_backend with correct args."""
        mock_backend = AsyncMock()

        async def _fake_create_pg(dsn, pg_pool_min, pg_pool_max):
            assert dsn == "postgresql://user:pw@host/db"
            assert pg_pool_min == 2
            assert pg_pool_max == 5
            return mock_backend

        with (
            patch("archivist.core.config.GRAPH_BACKEND", "postgres"),
            patch("archivist.core.config.DATABASE_URL", "postgresql://user:pw@host/db"),
            patch("archivist.core.config.PG_POOL_MIN", 2),
            patch("archivist.core.config.PG_POOL_MAX", 5),
            patch(
                "archivist.storage.backend_factory._create_postgres_backend",
                side_effect=_fake_create_pg,
            ),
        ):
            from archivist.storage.backend_factory import create_graph_backend

            backend = await create_graph_backend()

        assert backend is mock_backend

    async def test_unknown_backend_raises(self):
        """An unrecognised GRAPH_BACKEND value raises ValueError."""
        with (
            patch("archivist.core.config.GRAPH_BACKEND", "duckdb"),
            patch("archivist.core.config.DATABASE_URL", ""),
            patch("archivist.core.config.SQLITE_PATH", "/dev/null"),
            patch("archivist.core.config.SQLITE_WAL_AUTOCHECKPOINT", 1000),
            patch("archivist.core.config.PG_POOL_MIN", 5),
            patch("archivist.core.config.PG_POOL_MAX", 20),
        ):
            from archivist.storage.backend_factory import create_graph_backend

            with pytest.raises(ValueError, match="Unknown GRAPH_BACKEND"):
                await create_graph_backend()

    async def test_postgres_without_url_raises(self):
        """GRAPH_BACKEND=postgres without DATABASE_URL raises ValueError."""
        with (
            patch("archivist.core.config.GRAPH_BACKEND", "postgres"),
            patch("archivist.core.config.DATABASE_URL", ""),
            patch("archivist.core.config.PG_POOL_MIN", 5),
            patch("archivist.core.config.PG_POOL_MAX", 20),
        ):
            from archivist.storage.backend_factory import create_graph_backend

            with pytest.raises(ValueError, match="DATABASE_URL must be set"):
                await create_graph_backend()


# ---------------------------------------------------------------------------
# get_db() deprecation warning
# ---------------------------------------------------------------------------


class TestGetDbDeprecation:
    """get_db() should warn when GRAPH_BACKEND=postgres."""

    def test_warns_when_postgres(self, tmp_path, monkeypatch, caplog):
        import logging

        monkeypatch.setattr("archivist.storage.graph.SQLITE_PATH", str(tmp_path / "test.db"))
        # get_db() imports GRAPH_BACKEND from archivist.core.config at call time
        with patch("archivist.core.config.GRAPH_BACKEND", "postgres"):
            from archivist.storage.graph import get_db

            with caplog.at_level(logging.WARNING, logger="archivist.graph"):
                conn = get_db()
                conn.close()

        assert "get_db() is not supported with GRAPH_BACKEND=postgres" in caplog.text

    def test_no_warning_when_sqlite(self, tmp_path, monkeypatch, caplog):
        import logging

        monkeypatch.setattr("archivist.storage.graph.SQLITE_PATH", str(tmp_path / "test.db"))
        with patch("archivist.core.config.GRAPH_BACKEND", "sqlite"):
            from archivist.storage.graph import get_db

            with caplog.at_level(logging.WARNING, logger="archivist.graph"):
                conn = get_db()
                conn.close()

        assert "get_db() is not supported" not in caplog.text


# ---------------------------------------------------------------------------
# FTS backend dispatch (unit tests — mocked pool)
# ---------------------------------------------------------------------------


class TestFtsBackendDispatch:
    """search_fts() and search_fts_exact() route to the right implementation."""

    async def test_search_fts_calls_sqlite_impl_by_default(self, monkeypatch):
        """With GRAPH_BACKEND unset (sqlite), the SQLite FTS5 path is taken."""
        calls = []

        async def _fake_sqlite_search(**kwargs):
            calls.append(("sqlite", kwargs))
            return []

        monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "")
        monkeypatch.setattr("archivist.storage.graph._search_fts_sqlite", _fake_sqlite_search)

        from archivist.storage.graph import search_fts

        await search_fts(query='"k8s"', namespace="ns1")
        assert len(calls) == 1
        assert calls[0][0] == "sqlite"
        assert calls[0][1]["namespace"] == "ns1"

    async def test_search_fts_calls_postgres_impl_when_configured(self, monkeypatch):
        """With GRAPH_BACKEND=postgres, the Postgres tsvector path is taken."""
        calls = []

        async def _fake_pg_search(**kwargs):
            calls.append(("postgres", kwargs))
            return []

        monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "postgres")
        monkeypatch.setattr("archivist.storage.graph._search_fts_postgres", _fake_pg_search)

        from archivist.storage.graph import search_fts

        await search_fts(query='"k8s"', raw_query="k8s", fts_mode="or", namespace="ns1")
        assert len(calls) == 1
        assert calls[0][0] == "postgres"
        assert calls[0][1]["fts_mode"] == "or"
        assert calls[0][1]["namespace"] == "ns1"

    async def test_search_fts_exact_calls_sqlite_impl_by_default(self, monkeypatch):
        calls = []

        async def _fake_sqlite_exact(**kwargs):
            calls.append(("sqlite", kwargs))
            return []

        monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "sqlite")
        monkeypatch.setattr("archivist.storage.graph._search_fts_exact_sqlite", _fake_sqlite_exact)

        from archivist.storage.graph import search_fts_exact

        await search_fts_exact(query='"192.168.1.1"')
        assert calls[0][0] == "sqlite"

    async def test_search_fts_exact_calls_postgres_impl_when_configured(self, monkeypatch):
        calls = []

        async def _fake_pg_exact(**kwargs):
            calls.append(("postgres", kwargs))
            return []

        monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "postgres")
        monkeypatch.setattr("archivist.storage.graph._search_fts_exact_postgres", _fake_pg_exact)

        from archivist.storage.graph import search_fts_exact

        await search_fts_exact(query='"192.168.1.1"', raw_query="192.168.1.1")
        assert calls[0][0] == "postgres"


class TestUpsertFtsChunkNoopOnPostgres:
    """upsert_fts_chunk() skips shadow-row ops on Postgres."""

    async def test_postgres_path_calls_postgres_impl(self, monkeypatch):
        pg_calls = []
        sqlite_calls = []

        async def _fake_pg(**kwargs):
            pg_calls.append(kwargs)

        async def _fake_sqlite(**kwargs):
            sqlite_calls.append(kwargs)

        monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "postgres")
        monkeypatch.setattr("archivist.storage.graph._upsert_fts_chunk_postgres", _fake_pg)
        monkeypatch.setattr("archivist.storage.graph._upsert_fts_chunk_sqlite", _fake_sqlite)

        from archivist.storage.graph import upsert_fts_chunk

        await upsert_fts_chunk(
            qdrant_id="abc",
            text="hello world",
            file_path="/f",
            chunk_index=0,
        )

        assert len(pg_calls) == 1
        assert len(sqlite_calls) == 0
        assert pg_calls[0]["qdrant_id"] == "abc"

    async def test_sqlite_path_calls_sqlite_impl(self, monkeypatch):
        pg_calls = []
        sqlite_calls = []

        async def _fake_pg(**kwargs):
            pg_calls.append(kwargs)

        async def _fake_sqlite(**kwargs):
            sqlite_calls.append(kwargs)

        monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "sqlite")
        monkeypatch.setattr("archivist.storage.graph._upsert_fts_chunk_postgres", _fake_pg)
        monkeypatch.setattr("archivist.storage.graph._upsert_fts_chunk_sqlite", _fake_sqlite)

        from archivist.storage.graph import upsert_fts_chunk

        await upsert_fts_chunk(
            qdrant_id="xyz",
            text="some text",
            file_path="/g",
            chunk_index=1,
        )

        assert len(sqlite_calls) == 1
        assert len(pg_calls) == 0


class TestDeleteFtsRowsAsyncNoopOnPostgres:
    """_delete_fts_rows_async() is a no-op on Postgres."""

    async def test_no_sql_executed_when_postgres(self, monkeypatch):
        executed = []

        class MockConn:
            async def execute(self, sql, *args):
                executed.append(sql)

        monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "postgres")

        from archivist.storage.graph import _delete_fts_rows_async

        rows = [{"rowid": 1}, {"rowid": 2}]
        await _delete_fts_rows_async(MockConn(), rows)

        assert executed == [], "Expected no SQL execution on Postgres"

    async def test_sql_executed_when_sqlite(self, monkeypatch):
        executed = []

        class MockConn:
            async def execute(self, sql, *args):
                executed.append(sql)

        monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "sqlite")

        from archivist.storage.graph import _delete_fts_rows_async

        rows = [{"rowid": 1}]
        await _delete_fts_rows_async(MockConn(), rows)

        assert len(executed) > 0
