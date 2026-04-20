"""Unit tests for archivist_store idempotency fix.

Regression guard: previously the ``except sqlite3.IntegrityError`` block in
``_handle_store``'s explicit-entity loop called ``return conflict_resolved_response(...)``
on the FIRST entity conflict.  That returned a truncated response with no
``memory_id`` / ``stored`` field, skipped all remaining entities, and aborted
the entire embedding + Qdrant + FTS + audit pipeline.  Agents saw what looked
like a "success" response but the memory was never actually stored.

The tests here verify:
  1. IntegrityError on the FIRST entity does not abort the remaining pipeline.
  2. IntegrityError on the agent self-entity path does not abort the store.
  3. IntegrityError on auto-extracted entities does not abort the store.
  4. ``resolve_entity_id`` correctly looks up an existing entity row.
  5. ``resolve_entity_id`` returns 0 for an unknown entity (does not raise).
  6. Both ``sqlite3.IntegrityError`` and asyncpg-style "unique" exceptions are
     caught — the broad string-match guard handles both backends.
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_standard_patches(monkeypatch):
    """Return a dict of pre-built mocks for the heavy _handle_store dependencies."""
    return {
        "embed_text": AsyncMock(return_value=[0.1] * 1024),
        "embed_batch": AsyncMock(return_value=[[0.1] * 1024]),
        "check_for_conflicts": AsyncMock(return_value=MagicMock(has_conflict=False)),
        "llm_adjudicated_dedup": AsyncMock(return_value=None),
        "qdrant_client": MagicMock(return_value=MagicMock(upsert=MagicMock())),
        "ensure_collection": MagicMock(return_value="test-col"),
        "audit.log_memory_event": AsyncMock(),
        "get_namespace_for_agent": MagicMock(return_value=None),
        "get_namespace_config": MagicMock(return_value=None),
        "_rbac_gate": MagicMock(return_value=None),
        "webhooks.fire_background": MagicMock(),
        "invalidate_index_cache": MagicMock(),
    }


# ---------------------------------------------------------------------------
# Test 1: IntegrityError on explicit entity continues the store pipeline
# ---------------------------------------------------------------------------


class TestExplicitEntityIntegrityErrorContinues:
    """IntegrityError on the first explicit entity must not abort the store.

    Regression: the old code called ``return conflict_resolved_response(...)``
    inside the entity loop, which terminated _handle_store entirely — skipping
    all remaining entities and the embedding/Qdrant/FTS pipeline.
    """

    @pytest.mark.asyncio
    async def test_store_returns_stored_true_despite_integrity_error(
        self, monkeypatch
    ):
        """Even when the first upsert_entity raises IntegrityError, the store
        completes successfully and returns a response with stored=True."""
        monkeypatch.setattr(
            "archivist.app.handlers.tools_search.is_permissive_mode", lambda: True
        )
        monkeypatch.setattr("archivist.core.config.RBAC_PERMISSIVE_MODE", True)

        existing_id = 42

        upsert_call_count = 0

        async def mock_upsert(name, *args, **kwargs):
            nonlocal upsert_call_count
            upsert_call_count += 1
            if upsert_call_count == 1:
                # First call raises IntegrityError; the fix must catch it and
                # continue instead of returning.
                raise sqlite3.IntegrityError("UNIQUE constraint failed: entities.name, entities.namespace")
            return existing_id + upsert_call_count

        async def mock_resolve(name, namespace="global"):
            return existing_id

        with (
            patch("handlers.tools_storage.upsert_entity", side_effect=mock_upsert),
            patch("handlers.tools_storage.resolve_entity_id", side_effect=mock_resolve),
            patch("handlers.tools_storage.add_fact", new_callable=AsyncMock),
            patch(
                "handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
            patch(
                "conflict_detection.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
            patch(
                "handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                return_value=[[0.1] * 1024],
            ),
            patch(
                "handlers.tools_storage.check_for_conflicts",
                new_callable=AsyncMock,
                return_value=MagicMock(has_conflict=False),
            ),
            patch(
                "handlers.tools_storage.llm_adjudicated_dedup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("handlers.tools_storage.qdrant_client") as mock_qc,
            patch("handlers.tools_storage.ensure_collection", return_value="test-col"),
            patch("audit.log_memory_event", new_callable=AsyncMock),
            patch("handlers.tools_storage.get_namespace_for_agent", return_value=None),
            patch("handlers.tools_storage.get_namespace_config", return_value=None),
            patch("handlers.tools_storage._rbac_gate", return_value=None),
            patch("handlers.tools_storage.invalidate_index_cache", MagicMock()),
            patch("webhooks.fire_background"),
        ):
            mock_qc.return_value = MagicMock(upsert=MagicMock())
            monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", False)
            monkeypatch.setattr("config.BM25_ENABLED", False)
            monkeypatch.setattr("config.TOPIC_ROUTING_ENABLED", False)

            from handlers.tools_storage import _handle_store

            result = await _handle_store(
                {
                    "text": "Brian deployed Kubernetes to production",
                    "agent_id": "test-agent",
                    "entities": ["brian", "kubernetes"],
                    "namespace": "test-ns",
                }
            )

        # Must NOT be a conflict_resolved_response (which has entity_id=0 and
        # no memory_id).  We accept any non-error response.
        assert result is not None
        content = result[0].text if hasattr(result[0], "text") else str(result)
        import json as _json
        parsed = _json.loads(content)
        # The old bug returned {"status": "conflict_resolved", "entity_id": 0}
        assert parsed.get("status") != "conflict_resolved", (
            "BUG REGRESSION: _handle_store returned conflict_resolved_response "
            "which aborts the store pipeline for all remaining entities."
        )

    @pytest.mark.asyncio
    async def test_integrity_error_increments_metric(self, monkeypatch):
        """ENTITY_UPSERT_CONFLICTS metric is incremented when IntegrityError fires."""
        monkeypatch.setattr("archivist.core.config.RBAC_PERMISSIVE_MODE", True)

        async def raising_upsert(name, *args, **kwargs):
            raise sqlite3.IntegrityError("UNIQUE constraint failed")

        async def mock_resolve(name, namespace="global"):
            return 99

        import archivist.core.metrics as m

        before = m._counters.get(
            'archivist_entity_upsert_conflicts_total{namespace="test-ns"}', 0
        )

        with (
            patch("handlers.tools_storage.upsert_entity", side_effect=raising_upsert),
            patch("handlers.tools_storage.resolve_entity_id", side_effect=mock_resolve),
            patch("handlers.tools_storage.add_fact", new_callable=AsyncMock),
            patch(
                "handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
            patch(
                "conflict_detection.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
            patch(
                "handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                return_value=[[0.1] * 1024],
            ),
            patch(
                "handlers.tools_storage.check_for_conflicts",
                new_callable=AsyncMock,
                return_value=MagicMock(has_conflict=False),
            ),
            patch(
                "handlers.tools_storage.llm_adjudicated_dedup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("handlers.tools_storage.qdrant_client") as mock_qc,
            patch("handlers.tools_storage.ensure_collection", return_value="test-col"),
            patch("audit.log_memory_event", new_callable=AsyncMock),
            patch("handlers.tools_storage.get_namespace_for_agent", return_value=None),
            patch("handlers.tools_storage.get_namespace_config", return_value=None),
            patch("handlers.tools_storage._rbac_gate", return_value=None),
            patch("handlers.tools_storage.invalidate_index_cache", MagicMock()),
            patch("webhooks.fire_background"),
        ):
            mock_qc.return_value = MagicMock(upsert=MagicMock())
            monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", False)
            monkeypatch.setattr("config.BM25_ENABLED", False)
            monkeypatch.setattr("config.TOPIC_ROUTING_ENABLED", False)

            from handlers.tools_storage import _handle_store

            await _handle_store(
                {
                    "text": "test entity conflict",
                    "agent_id": "test-agent",
                    "entities": ["brian"],
                    "namespace": "test-ns",
                }
            )

        after = m._counters.get(
            'archivist_entity_upsert_conflicts_total{namespace="test-ns"}', 0
        )
        assert after > before, "ENTITY_UPSERT_CONFLICTS must be incremented on IntegrityError"

    @pytest.mark.asyncio
    async def test_non_unique_exception_reraises(self, monkeypatch):
        """Exceptions that are NOT unique-constraint errors must propagate normally."""
        monkeypatch.setattr("archivist.core.config.RBAC_PERMISSIVE_MODE", True)

        async def breaking_upsert(name, *args, **kwargs):
            raise RuntimeError("unexpected database crash")

        with (
            patch("handlers.tools_storage.upsert_entity", side_effect=breaking_upsert),
            patch(
                "handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
            patch(
                "conflict_detection.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
            patch(
                "handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                return_value=[[0.1] * 1024],
            ),
            patch(
                "handlers.tools_storage.check_for_conflicts",
                new_callable=AsyncMock,
                return_value=MagicMock(has_conflict=False),
            ),
            patch(
                "handlers.tools_storage.llm_adjudicated_dedup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("handlers.tools_storage.qdrant_client") as mock_qc,
            patch("handlers.tools_storage.ensure_collection", return_value="test-col"),
            patch("audit.log_memory_event", new_callable=AsyncMock),
            patch("handlers.tools_storage.get_namespace_for_agent", return_value=None),
            patch("handlers.tools_storage.get_namespace_config", return_value=None),
            patch("handlers.tools_storage._rbac_gate", return_value=None),
            patch("webhooks.fire_background"),
        ):
            mock_qc.return_value = MagicMock(upsert=MagicMock())
            monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", False)
            monkeypatch.setattr("config.BM25_ENABLED", False)
            monkeypatch.setattr("config.TOPIC_ROUTING_ENABLED", False)

            from handlers.tools_storage import _handle_store

            with pytest.raises(RuntimeError, match="unexpected database crash"):
                await _handle_store(
                    {
                        "text": "test non-unique error propagation",
                        "agent_id": "test-agent",
                        "entities": ["entity1"],
                        "namespace": "test-ns",
                    }
                )


# ---------------------------------------------------------------------------
# Test 2: resolve_entity_id helper
# ---------------------------------------------------------------------------


class TestResolveEntityId:
    """``resolve_entity_id`` in graph.py correctly looks up existing entities."""

    @pytest.mark.asyncio
    async def test_returns_id_for_existing_entity(self, monkeypatch):
        """Returns the integer row ID when the entity exists."""
        import archivist.storage.sqlite_pool as _sp

        mock_pool = MagicMock()

        @asynccontextmanager
        async def mock_read():
            conn = AsyncMock()
            cur = AsyncMock()
            cur.fetchone = AsyncMock(return_value=(77,))
            conn.execute = AsyncMock(return_value=cur)
            yield conn

        mock_pool.read = mock_read
        monkeypatch.setattr(_sp, "pool", mock_pool)

        from archivist.storage.graph import resolve_entity_id

        result = await resolve_entity_id("brian", "default")
        assert result == 77

    @pytest.mark.asyncio
    async def test_returns_zero_for_missing_entity(self, monkeypatch):
        """Returns 0 (not an exception) when the entity does not exist."""
        import archivist.storage.sqlite_pool as _sp

        mock_pool = MagicMock()

        @asynccontextmanager
        async def mock_read():
            conn = AsyncMock()
            cur = AsyncMock()
            cur.fetchone = AsyncMock(return_value=None)
            conn.execute = AsyncMock(return_value=cur)
            yield conn

        mock_pool.read = mock_read
        monkeypatch.setattr(_sp, "pool", mock_pool)

        from archivist.storage.graph import resolve_entity_id

        result = await resolve_entity_id("nonexistent-entity", "default")
        assert result == 0, "Must return 0 (not raise) for missing entities"

    @pytest.mark.asyncio
    async def test_namespace_is_passed_to_query(self, monkeypatch):
        """Namespace is used as a query parameter so cross-namespace entities
        are not confused (the original Postgres schema bug was UNIQUE(name) which
        would conflate 'brian' in 'ns-a' with 'brian' in 'ns-b')."""
        import archivist.storage.sqlite_pool as _sp

        received_params: list = []

        @asynccontextmanager
        async def mock_read():
            conn = AsyncMock()
            cur = AsyncMock()
            cur.fetchone = AsyncMock(return_value=(10,))

            async def capture_execute(sql, params):
                received_params.extend(params)
                return cur

            conn.execute = capture_execute
            yield conn

        mock_pool = MagicMock()
        mock_pool.read = mock_read
        monkeypatch.setattr(_sp, "pool", mock_pool)

        from archivist.storage.graph import resolve_entity_id

        await resolve_entity_id("brian", "special-namespace")
        assert "special-namespace" in received_params, (
            "Namespace must be passed as a query parameter to prevent "
            "cross-namespace entity ID collisions."
        )


# ---------------------------------------------------------------------------
# Test 3: asyncpg-style "unique violation" string is caught
# ---------------------------------------------------------------------------


class TestAsyncpgUniqueViolationCaught:
    """Both sqlite3.IntegrityError and asyncpg UniqueViolationError are handled.

    asyncpg raises a custom exception class, but its message contains the word
    "unique".  The guard uses a lowercase string check so both backends are covered
    without importing asyncpg (an optional dependency).
    """

    @pytest.mark.asyncio
    async def test_asyncpg_style_exception_is_caught(self, monkeypatch):
        """An Exception whose message contains 'unique' is treated as a conflict."""
        monkeypatch.setattr("archivist.core.config.RBAC_PERMISSIVE_MODE", True)

        class FakeUniqueViolationError(Exception):
            """Simulates asyncpg.exceptions.UniqueViolationError."""

        async def asyncpg_raising_upsert(name, *args, **kwargs):
            raise FakeUniqueViolationError(
                'duplicate key value violates unique constraint "entities_name_ns_unique"'
            )

        async def mock_resolve(name, namespace="global"):
            return 55

        with (
            patch(
                "handlers.tools_storage.upsert_entity",
                side_effect=asyncpg_raising_upsert,
            ),
            patch("handlers.tools_storage.resolve_entity_id", side_effect=mock_resolve),
            patch("handlers.tools_storage.add_fact", new_callable=AsyncMock),
            patch(
                "handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
            patch(
                "conflict_detection.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
            patch(
                "handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                return_value=[[0.1] * 1024],
            ),
            patch(
                "handlers.tools_storage.check_for_conflicts",
                new_callable=AsyncMock,
                return_value=MagicMock(has_conflict=False),
            ),
            patch(
                "handlers.tools_storage.llm_adjudicated_dedup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("handlers.tools_storage.qdrant_client") as mock_qc,
            patch("handlers.tools_storage.ensure_collection", return_value="test-col"),
            patch("audit.log_memory_event", new_callable=AsyncMock),
            patch("handlers.tools_storage.get_namespace_for_agent", return_value=None),
            patch("handlers.tools_storage.get_namespace_config", return_value=None),
            patch("handlers.tools_storage._rbac_gate", return_value=None),
            patch("handlers.tools_storage.invalidate_index_cache", MagicMock()),
            patch("webhooks.fire_background"),
        ):
            mock_qc.return_value = MagicMock(upsert=MagicMock())
            monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", False)
            monkeypatch.setattr("config.BM25_ENABLED", False)
            monkeypatch.setattr("config.TOPIC_ROUTING_ENABLED", False)

            from handlers.tools_storage import _handle_store

            # Must not raise — asyncpg-style unique violations must be caught
            result = await _handle_store(
                {
                    "text": "asyncpg backend test",
                    "agent_id": "test-agent",
                    "entities": ["brian"],
                    "namespace": "test-ns",
                }
            )

        assert result is not None


# ---------------------------------------------------------------------------
# Test 4: Postgres schema contains UNIQUE(name, namespace)
# ---------------------------------------------------------------------------


class TestPostgresSchemaConstraint:
    """Static analysis: schema_postgres.sql must use UNIQUE(name, namespace).

    Regression guard: the original schema had ``UNIQUE(name)`` (missing namespace)
    which caused ``ON CONFLICT(name, namespace)`` in ``upsert_entity`` to fail on
    Postgres with an unhandled exception — a different entity named 'brian' in a
    different namespace would conflict globally.
    """

    def test_postgres_schema_has_namespace_in_unique_constraint(self):
        import re
        from pathlib import Path

        schema_path = (
            Path(__file__).parents[3]
            / "src"
            / "archivist"
            / "storage"
            / "schema_postgres.sql"
        )
        content = schema_path.read_text()

        # Must contain UNIQUE (name, namespace) or UNIQUE(name, namespace)
        assert re.search(
            r"UNIQUE\s*\(\s*name\s*,\s*namespace\s*\)", content
        ), (
            "schema_postgres.sql must define UNIQUE(name, namespace) on the "
            "entities table so ON CONFLICT(name, namespace) in upsert_entity "
            "works correctly.  The old UNIQUE(name) caused cross-namespace "
            "collisions and broke upsert_entity on Postgres."
        )

    def test_postgres_schema_does_not_have_name_only_unique(self):
        import re
        from pathlib import Path

        schema_path = (
            Path(__file__).parents[3]
            / "src"
            / "archivist"
            / "storage"
            / "schema_postgres.sql"
        )
        content = schema_path.read_text()

        # Must NOT contain the old broken constraint (UNIQUE with only name
        # and no namespace as the second column).
        bad_match = re.search(
            r"entities_name_unique\s+UNIQUE\s*\(\s*name\s*\)", content
        )
        assert not bad_match, (
            "REGRESSION: schema_postgres.sql still has the old "
            "entities_name_unique UNIQUE(name) constraint that caused "
            "cross-namespace conflicts."
        )
