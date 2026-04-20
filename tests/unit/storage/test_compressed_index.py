"""Unit tests for the archivist_index fetch-error fix.

Regression guard for the bug where build_namespace_index() used the deprecated
synchronous get_db() instead of the async pool, causing every archivist_index
call to fail with a generic "fetch error" when:
  - GRAPH_BACKEND=postgres (get_db() returns a temp SQLite conn with no data)
  - The SQLite file doesn't exist / isn't initialised yet
  - Any database error occurs at query time

The tests here cover:
  1. Empty database → returns the "No indexed knowledge" sentinel (not an exception)
  2. Populated database → returns well-formed index text
  3. Database error → _handle_index returns a structured error response (not a raise)
  4. Cache hit → second call skips the DB entirely
  5. Cache invalidation → invalidate_index_cache causes next call to re-query
  6. Async guard → build_namespace_index is a coroutine function (prevents regression
     to a sync implementation)
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rows(*dicts: dict):
    """Return a list of mock Row objects that behave like aiosqlite.Row.

    aiosqlite.Row supports dict(row) via the mapping protocol (keys + __getitem__).
    We simulate this so that ``[dict(r) for r in rows]`` works correctly.
    """
    rows = []
    for d in dicts:
        row = MagicMock()
        row.__getitem__ = lambda self, k, _d=d: _d[k]
        row.keys = MagicMock(return_value=list(d.keys()))
        # Support dict(row) via __iter__ yielding (key, value) pairs
        row.__iter__ = lambda self, _d=d: iter(_d.keys())
        # aiosqlite.Row is actually a sqlite3.Row, which maps keys
        # dict() on an aiosqlite.Row calls row.keys() then row[key].
        # We override __class__ approach; simplest is to make it a plain dict subclass:
        rows.append(d)  # return the plain dict — aiosqlite.Row converts to dict the same way
    return rows


def _make_conn(entities_rows=None, facts_rows=None):
    """Build a mock aiosqlite connection that returns controlled query results."""
    entities_rows = entities_rows or []
    facts_rows = facts_rows or []

    async def _execute(sql, params=None):
        cursor = AsyncMock()
        sql_lower = sql.strip().lower()
        if "from facts" in sql_lower and "join entities" not in sql_lower[:sql_lower.index("from facts")]:
            cursor.fetchall = AsyncMock(return_value=facts_rows)
        elif "from entities" in sql_lower or "distinct e." in sql_lower:
            cursor.fetchall = AsyncMock(return_value=entities_rows)
        else:
            # memory_chunks count, last_seen, tips queries
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, k: 0 if k == "c" else ""
            cursor.fetchone = AsyncMock(return_value=mock_row)
            cursor.fetchall = AsyncMock(return_value=[])
        return cursor

    conn = AsyncMock()
    conn.execute = _execute
    return conn


@asynccontextmanager
async def _pool_read_ctx(conn):
    yield conn


def _patch_pool(monkeypatch, conn):
    """Patch _sqlite_pool.pool.read() to return a controlled connection.

    compressed_index.py references pool through the module object
    (``_sqlite_pool.pool.read()``), so we patch the pool attribute on the
    module-level singleton, not on the imported name.
    """
    from archivist.storage import compressed_index as ci

    class _MockPool:
        def read(self):
            return _pool_read_ctx(conn)

    monkeypatch.setattr(ci._sqlite_pool, "pool", _MockPool())


# ---------------------------------------------------------------------------
# Test 6: build_namespace_index is a coroutine function (regression guard)
# ---------------------------------------------------------------------------


class TestAsyncGuard:
    """Regression guard: build_namespace_index must remain a coroutine function.

    If someone accidentally reverts it to a synchronous function the pool.read()
    await calls will break and the fetch-error will return.
    """

    def test_build_namespace_index_is_coroutine(self):
        from archivist.storage.compressed_index import build_namespace_index

        assert asyncio.iscoroutinefunction(build_namespace_index), (
            "build_namespace_index must be async — reverting to sync breaks the "
            "async pool and causes the archivist_index fetch-error regression."
        )

    def test_build_wake_up_context_is_coroutine(self):
        from archivist.storage.compressed_index import build_wake_up_context

        assert asyncio.iscoroutinefunction(build_wake_up_context), (
            "build_wake_up_context must be async — it calls build_namespace_index "
            "and uses pool.read() internally."
        )


# ---------------------------------------------------------------------------
# Test 1: Empty database → "No indexed knowledge" sentinel
# ---------------------------------------------------------------------------


class TestBuildNamespaceIndexEmpty:
    """Empty DB must return the sentinel string, not raise or return garbage."""

    @pytest.mark.asyncio
    async def test_empty_db_returns_sentinel(self, monkeypatch):
        from archivist.storage import compressed_index as ci

        _patch_pool(monkeypatch, _make_conn(entities_rows=[]))
        ci._index_cache.clear()

        result = await ci.build_namespace_index("test_ns")

        assert "No indexed knowledge" in result
        assert "test_ns" in result

    @pytest.mark.asyncio
    async def test_empty_db_does_not_raise(self, monkeypatch):
        from archivist.storage import compressed_index as ci

        _patch_pool(monkeypatch, _make_conn(entities_rows=[]))
        ci._index_cache.clear()

        try:
            await ci.build_namespace_index("empty_ns")
        except Exception as exc:
            pytest.fail(f"build_namespace_index raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# Test 2: Populated database → well-formed index text
# ---------------------------------------------------------------------------


class TestBuildNamespaceIndexPopulated:
    """Populated DB returns entity categories, key facts, and top topics."""

    @pytest.mark.asyncio
    async def test_populated_returns_index(self, monkeypatch):
        from archivist.storage import compressed_index as ci

        entity_rows = _make_rows(
            {
                "id": 1,
                "name": "Alice",
                "entity_type": "person",
                "mention_count": 10,
                "retention_class": "durable",
                "last_seen": "2026-01-01",
            },
            {
                "id": 2,
                "name": "ProjectX",
                "entity_type": "project",
                "mention_count": 5,
                "retention_class": "standard",
                "last_seen": "2026-01-02",
            },
        )
        fact_rows = _make_rows(
            {
                "entity_id": 1,
                "fact_text": "Alice is the lead engineer",
                "retention_class": "durable",
            }
        )
        _patch_pool(monkeypatch, _make_conn(entities_rows=entity_rows, facts_rows=fact_rows))
        ci._index_cache.clear()

        result = await ci.build_namespace_index("prod")

        assert "Memory Index" in result
        assert "Alice" in result
        assert "ProjectX" in result

    @pytest.mark.asyncio
    async def test_populated_contains_entity_type_section(self, monkeypatch):
        from archivist.storage import compressed_index as ci

        entity_rows = _make_rows(
            {
                "id": 1,
                "name": "CoreSystem",
                "entity_type": "system",
                "mention_count": 20,
                "retention_class": "permanent",
                "last_seen": "2026-01-05",
            }
        )
        _patch_pool(monkeypatch, _make_conn(entities_rows=entity_rows))
        ci._index_cache.clear()

        result = await ci.build_namespace_index("prod2")

        assert "Memory Index" in result
        assert "CoreSystem" in result


# ---------------------------------------------------------------------------
# Test 3: Database error → _handle_index returns structured error response
# ---------------------------------------------------------------------------


class TestHandleIndexDatabaseError:
    """DB errors must produce a structured error response, never an unhandled exception.

    This is the exact scenario that caused the 'fetch error' wall: any exception from
    build_namespace_index propagated through dispatch_tool as a generic tool_error.
    Now the handler catches it and returns an actionable error response.
    """

    @pytest.mark.asyncio
    async def test_db_error_returns_error_response(self, monkeypatch):
        from archivist.app.handlers import tools_search
        from archivist.app.handlers.tools_search import _handle_index
        from archivist.storage import compressed_index as ci

        async def _raise_db_error(namespace, agent_ids=None):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(ci, "build_namespace_index", _raise_db_error)
        # Bypass RBAC so we reach the index build path
        monkeypatch.setattr(tools_search, "is_permissive_mode", lambda: True)

        result = await _handle_index({"agent_id": "agent_a", "namespace": "ns_a"})

        assert len(result) == 1
        content = result[0].text
        parsed = json.loads(content)
        assert parsed.get("error") == "index_unavailable"
        assert "next_steps" in parsed
        assert isinstance(parsed["next_steps"], list)
        assert len(parsed["next_steps"]) > 0

    @pytest.mark.asyncio
    async def test_db_error_increments_tool_errors_metric(self, monkeypatch):
        import archivist.core.metrics as m
        from archivist.app.handlers import tools_search
        from archivist.app.handlers.tools_search import _handle_index
        from archivist.storage import compressed_index as ci

        async def _raise(namespace, agent_ids=None):
            raise Exception("connection refused")

        monkeypatch.setattr(ci, "build_namespace_index", _raise)
        monkeypatch.setattr(tools_search, "is_permissive_mode", lambda: True)

        before = m._counters.get(f'{m.TOOL_ERRORS}{{tool="archivist_index"}}', 0)
        await _handle_index({"agent_id": "agent_b", "namespace": "ns_b"})
        after = m._counters.get(f'{m.TOOL_ERRORS}{{tool="archivist_index"}}', 0)

        assert after > before, "TOOL_ERRORS metric must be incremented on DB failure"

    @pytest.mark.asyncio
    async def test_db_error_does_not_raise_to_caller(self, monkeypatch):
        from archivist.app.handlers import tools_search
        from archivist.app.handlers.tools_search import _handle_index
        from archivist.storage import compressed_index as ci

        async def _raise(namespace, agent_ids=None):
            raise Exception("no such table: entities")

        monkeypatch.setattr(ci, "build_namespace_index", _raise)
        monkeypatch.setattr(tools_search, "is_permissive_mode", lambda: True)

        try:
            result = await _handle_index({"agent_id": "agent_c", "namespace": "ns_c"})
        except Exception as exc:
            pytest.fail(
                f"_handle_index raised unexpectedly — this is the fetch-error regression: {exc}"
            )
        assert result is not None


# ---------------------------------------------------------------------------
# Test 4: Cache hit skips the DB
# ---------------------------------------------------------------------------


class TestIndexCacheHit:
    """Second call for the same namespace must be served from cache (no DB query)."""

    @pytest.mark.asyncio
    async def test_second_call_is_cache_hit(self, monkeypatch):
        import archivist.core.metrics as m
        from archivist.storage import compressed_index as ci

        call_count = 0

        class _CountingPool:
            def read(self):
                nonlocal call_count
                call_count += 1
                return _pool_read_ctx(_make_conn(entities_rows=[]))

        monkeypatch.setattr(ci._sqlite_pool, "pool", _CountingPool())
        ci._index_cache.clear()

        await ci.build_namespace_index("cache_ns")
        assert call_count == 1, "First call should hit the DB"

        await ci.build_namespace_index("cache_ns")
        assert call_count == 1, "Second call must not hit the DB — cache miss regression"

    @pytest.mark.asyncio
    async def test_cache_hit_increments_hit_metric(self, monkeypatch):
        import archivist.core.metrics as m
        from archivist.storage import compressed_index as ci

        _patch_pool(monkeypatch, _make_conn(entities_rows=[]))
        ci._index_cache.clear()

        await ci.build_namespace_index("hit_ns")
        before = m._counters.get(m.INDEX_CACHE_HIT, 0)
        await ci.build_namespace_index("hit_ns")
        after = m._counters.get(m.INDEX_CACHE_HIT, 0)

        assert after > before, "INDEX_CACHE_HIT must be incremented on cache hit"


# ---------------------------------------------------------------------------
# Test 5: Cache invalidation causes re-query
# ---------------------------------------------------------------------------


class TestIndexCacheInvalidation:
    """invalidate_index_cache must force a DB re-query on next call."""

    @pytest.mark.asyncio
    async def test_invalidation_causes_cache_miss(self, monkeypatch):
        from archivist.storage import compressed_index as ci

        call_count = 0

        class _CountingPool:
            def read(self):
                nonlocal call_count
                call_count += 1
                return _pool_read_ctx(_make_conn(entities_rows=[]))

        monkeypatch.setattr(ci._sqlite_pool, "pool", _CountingPool())
        ci._index_cache.clear()

        await ci.build_namespace_index("inv_ns")
        assert call_count == 1

        ci.invalidate_index_cache("inv_ns")

        await ci.build_namespace_index("inv_ns")
        assert call_count == 2, (
            "After invalidation, build_namespace_index must re-query the DB"
        )

    def test_invalidation_only_affects_target_namespace(self):
        from archivist.storage import compressed_index as ci

        ci._index_cache.clear()
        # Seed two namespaces
        ci._index_cache_set(ci._index_cache_key("ns_a", None), "index_a")
        ci._index_cache_set(ci._index_cache_key("ns_b", None), "index_b")

        ci.invalidate_index_cache("ns_a")

        assert ci._index_cache_get(ci._index_cache_key("ns_a", None)) is None
        assert ci._index_cache_get(ci._index_cache_key("ns_b", None)) == "index_b", (
            "Invalidating ns_a must not evict ns_b from the cache"
        )
