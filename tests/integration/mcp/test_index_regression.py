"""Integration and system tests for the archivist_index fetch-error fix.

These tests use a real (in-memory) SQLite database to verify the full pipeline
from build_namespace_index() through _handle_index() to dispatch_tool().

Regression guard: previously build_namespace_index() used get_db() (sync,
deprecated) which bypassed the async pool. Any exception propagated to
dispatch_tool() as a generic tool_error, rendered as "fetch error" by agents.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Test 7: Integration — build_namespace_index end-to-end with real SQLite
# ---------------------------------------------------------------------------


class TestBuildNamespaceIndexIntegration:
    """Real SQLite pool — no mocks — verifies the full async pool path."""

    @pytest.mark.asyncio
    async def test_empty_namespace_returns_sentinel(self, integration_pool):
        from archivist.storage.compressed_index import _index_cache, build_namespace_index

        _index_cache.clear()
        result = await build_namespace_index("integration_empty")
        assert "No indexed knowledge" in result

    @pytest.mark.asyncio
    async def test_populated_namespace_returns_index(self, integration_pool):
        """Insert an entity + fact then verify they appear in the index text."""
        from archivist.storage import sqlite_pool as _sp
        from archivist.storage.compressed_index import _index_cache, build_namespace_index

        _index_cache.clear()

        # Insert test data directly into the pool
        async with _sp.pool.write() as conn:
            await conn.execute(
                """INSERT INTO entities (name, entity_type, first_seen, last_seen,
                       mention_count, retention_class, namespace)
                   VALUES (?, ?, '2026-01-01', '2026-01-01', 5, 'durable', ?)""",
                ("IntegrationEntity", "concept", "integration_ns"),
            )
            entity_id_cur = await conn.execute(
                "SELECT id FROM entities WHERE name = ?", ("IntegrationEntity",)
            )
            row = await entity_id_cur.fetchone()
            entity_id = row[0]
            await conn.execute(
                """INSERT INTO facts (entity_id, fact_text, created_at, is_active,
                       retention_class, namespace)
                   VALUES (?, ?, '2026-01-01T00:00:00', 1, 'durable', ?)""",
                (entity_id, "IntegrationEntity is a test concept", "integration_ns"),
            )

        result = await build_namespace_index("integration_ns")

        assert "IntegrationEntity" in result
        assert "Memory Index" in result

    @pytest.mark.asyncio
    async def test_agent_scoped_query(self, integration_pool):
        """Scoping by agent_id filters entities correctly."""
        from archivist.storage import sqlite_pool as _sp
        from archivist.storage.compressed_index import _index_cache, build_namespace_index

        _index_cache.clear()

        async with _sp.pool.write() as conn:
            await conn.execute(
                """INSERT INTO entities (name, entity_type, first_seen, last_seen,
                       mention_count, retention_class, namespace)
                   VALUES (?, 'person', '2026-01-01', '2026-01-01', 3, 'standard', 'scoped_ns')""",
                ("AgentScopedEntity",),
            )
            eid_cur = await conn.execute(
                "SELECT id FROM entities WHERE name = ?", ("AgentScopedEntity",)
            )
            eid_row = await eid_cur.fetchone()
            eid = eid_row[0]
            await conn.execute(
                """INSERT INTO facts (entity_id, fact_text, created_at, is_active,
                       retention_class, agent_id, namespace)
                   VALUES (?, 'agent fact', '2026-01-01T00:00:00', 1, 'standard', 'agent_x', 'scoped_ns')""",
                (eid,),
            )

        result = await build_namespace_index("scoped_ns", agent_ids=["agent_x"])
        assert "AgentScopedEntity" in result


# ---------------------------------------------------------------------------
# Test 8: System — archivist_index via dispatch_tool (end-to-end round-trip)
# ---------------------------------------------------------------------------


class TestDispatchToolRoundTrip:
    """Verify that dispatch_tool('archivist_index', ...) never returns a tool_error.

    This is the exact failure mode that caused agents to see 'fetch error':
    dispatch_tool caught the exception from _handle_index and returned a generic
    tool_error JSON payload. With the fix, the handler returns valid TextContent.
    """

    @pytest.mark.asyncio
    async def test_dispatch_index_returns_text_content(self, integration_pool, monkeypatch):
        import json

        from archivist.app.handlers._registry import dispatch_tool

        monkeypatch.setenv("RBAC_PERMISSIVE_MODE", "true")

        from archivist.storage.compressed_index import _index_cache

        _index_cache.clear()

        result = await dispatch_tool("archivist_index", {"agent_id": "system_test_agent"})

        assert result, "dispatch_tool must return a non-empty list"
        content = result[0].text
        # Must NOT be a raw exception string or generic tool_error
        assert "coroutine" not in content, (
            "Result contains 'coroutine' — build_namespace_index was called without await"
        )
        # If it's JSON, must not be a tool_error
        try:
            parsed = json.loads(content)
            assert parsed.get("error") != "tool_error", (
                "dispatch_tool returned a generic tool_error — "
                "the fetch-error regression has returned"
            )
        except (json.JSONDecodeError, TypeError):
            # Plain text index is the normal success case
            pass

    @pytest.mark.asyncio
    async def test_dispatch_index_with_namespace(self, integration_pool, monkeypatch):
        """Explicit namespace argument must work without raising."""
        import json

        from archivist.app.handlers._registry import dispatch_tool
        from archivist.storage.compressed_index import _index_cache

        monkeypatch.setenv("RBAC_PERMISSIVE_MODE", "true")
        _index_cache.clear()

        result = await dispatch_tool(
            "archivist_index", {"agent_id": "system_test_agent", "namespace": "integration_ns"}
        )

        assert result
        content = result[0].text
        assert "coroutine" not in content
        try:
            parsed = json.loads(content)
            assert parsed.get("error") != "tool_error"
        except (json.JSONDecodeError, TypeError):
            pass
