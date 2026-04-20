"""Integration tests for archivist_store idempotency — migration re-store scenario.

These tests use a real (in-memory) SQLite database to verify the full entity
upsert pipeline is idempotent when the same entity names are stored multiple
times (the exact pattern that occurs during agent migration runs).

Regression guard: the original handler-level ``except sqlite3.IntegrityError``
block *returned* ``conflict_resolved_response`` on the first entity conflict,
silently aborting the entire store (no embedding, no Qdrant, no FTS, no audit).
These integration tests confirm that all paths through _handle_store succeed
even when the same entities are stored repeatedly.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mcp, pytest.mark.storage]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _common_patches():
    """Context managers that stub out heavy I/O for integration tests."""
    return [
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
        patch("handlers.tools_storage.qdrant_client"),
        patch("handlers.tools_storage.ensure_collection", return_value="test-col"),
        patch("audit.log_memory_event", new_callable=AsyncMock),
        patch("handlers.tools_storage.get_namespace_for_agent", return_value=None),
        patch("handlers.tools_storage.get_namespace_config", return_value=None),
        patch("handlers.tools_storage._rbac_gate", return_value=None),
        patch("handlers.tools_storage.invalidate_index_cache", MagicMock()),
        patch("webhooks.fire_background"),
    ]


def _apply_patches(patches):
    """Enter a list of context managers and return the stack."""
    for p in patches:
        p.__enter__()
    return patches


def _exit_patches(patches):
    for p in patches:
        p.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Test: upsert_entity itself is idempotent end-to-end
# ---------------------------------------------------------------------------


class TestUpsertEntityIdempotency:
    """upsert_entity with ON CONFLICT DO UPDATE is idempotent at the storage layer."""

    @pytest.mark.asyncio
    async def test_upsert_same_entity_twice_returns_same_id(self, integration_pool):
        """Storing the same (name, namespace) twice should return the same row ID."""
        from archivist.storage.graph import upsert_entity

        id1 = await upsert_entity(
            "brian",
            "person",
            namespace="test-ns",
            actor_id="agent-1",
            actor_type="agent",
        )
        id2 = await upsert_entity(
            "brian",
            "person",
            namespace="test-ns",
            actor_id="agent-1",
            actor_type="agent",
        )
        assert isinstance(id1, int) and id1 > 0
        assert id1 == id2, (
            "upsert_entity must return the same ID for the same (name, namespace) "
            "pair on repeated calls — ON CONFLICT DO UPDATE must be working."
        )

    @pytest.mark.asyncio
    async def test_same_name_different_namespace_gets_different_id(
        self, integration_pool
    ):
        """Entities with the same name in different namespaces must be distinct rows.

        Regression guard for the Postgres schema bug where UNIQUE(name) (without
        namespace) would have caused this to fail.
        """
        from archivist.storage.graph import upsert_entity

        id_ns1 = await upsert_entity(
            "brian",
            "person",
            namespace="ns-alpha",
            actor_id="agent-1",
            actor_type="agent",
        )
        id_ns2 = await upsert_entity(
            "brian",
            "person",
            namespace="ns-beta",
            actor_id="agent-1",
            actor_type="agent",
        )
        assert id_ns1 != id_ns2, (
            "Same entity name in different namespaces must produce different row IDs. "
            "If these are equal, the unique constraint is not scoped by namespace — "
            "which was the Postgres schema bug (UNIQUE(name) instead of "
            "UNIQUE(name, namespace))."
        )


# ---------------------------------------------------------------------------
# Test: resolve_entity_id end-to-end with real SQLite
# ---------------------------------------------------------------------------


class TestResolveEntityIdIntegration:
    """resolve_entity_id correctly finds entities in a real SQLite database."""

    @pytest.mark.asyncio
    async def test_resolve_finds_previously_upserted_entity(self, integration_pool):
        from archivist.storage.graph import resolve_entity_id, upsert_entity

        created_id = await upsert_entity(
            "kubernetes",
            "technology",
            namespace="infra-ns",
            actor_id="agent-1",
            actor_type="agent",
        )

        resolved_id = await resolve_entity_id("kubernetes", "infra-ns")
        assert resolved_id == created_id, (
            "resolve_entity_id must return the same ID that upsert_entity created."
        )

    @pytest.mark.asyncio
    async def test_resolve_returns_zero_for_nonexistent_entity(self, integration_pool):
        from archivist.storage.graph import resolve_entity_id

        result = await resolve_entity_id("does-not-exist", "infra-ns")
        assert result == 0

    @pytest.mark.asyncio
    async def test_resolve_is_namespace_scoped(self, integration_pool):
        """resolve_entity_id must not find an entity in a different namespace."""
        from archivist.storage.graph import resolve_entity_id, upsert_entity

        await upsert_entity(
            "argocd",
            "tool",
            namespace="prod",
            actor_id="agent-1",
            actor_type="agent",
        )

        # Searching in a different namespace must return 0
        result = await resolve_entity_id("argocd", "staging")
        assert result == 0, (
            "resolve_entity_id must be namespace-scoped; an entity in 'prod' "
            "must not be found when searching in 'staging'."
        )


# ---------------------------------------------------------------------------
# Test: Migration scenario — re-store same entities in batch
# ---------------------------------------------------------------------------


class TestMigrationReStoreScenario:
    """Full migration scenario: store the same 5 entities repeatedly.

    This is the exact pattern that was breaking agents during migration runs:
    re-storing legacy memories that reference entities already in the DB.
    """

    @pytest.mark.asyncio
    async def test_migration_restore_all_succeed(
        self, integration_pool, monkeypatch
    ):
        """Storing entities twice must never cause a pipeline abort.

        Regression guard: the old code ``return``ed ``conflict_resolved_response``
        on the first entity conflict, silently dropping the embedding + Qdrant
        pipeline for ALL subsequent operations in the batch.
        """
        monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", False)
        monkeypatch.setattr("config.BM25_ENABLED", False)
        monkeypatch.setattr("config.TOPIC_ROUTING_ENABLED", False)

        entities = ["brian", "kubernetes", "argocd", "grafana", "prometheus"]
        namespace = "migration-ns"

        import json

        from handlers.tools_storage import _handle_store

        patches = _common_patches()
        for ctx in patches:
            ctx.__enter__()

        try:
            # First pass — seeds the entities
            result1 = await _handle_store(
                {
                    "text": "Initial migration store of legacy data",
                    "agent_id": "migration-agent",
                    "entities": entities,
                    "namespace": namespace,
                }
            )
            content1 = json.loads(result1[0].text)
            assert content1.get("status") != "conflict_resolved", (
                "First store must not return conflict_resolved."
            )

            # Second pass — all entities already exist; must still complete cleanly
            result2 = await _handle_store(
                {
                    "text": "Re-store of same legacy data (migration retry)",
                    "agent_id": "migration-agent",
                    "entities": entities,
                    "namespace": namespace,
                }
            )
            content2 = json.loads(result2[0].text)
            assert content2.get("status") != "conflict_resolved", (
                "BUG REGRESSION: second store of same entities returned "
                "conflict_resolved_response, aborting the store pipeline. "
                "archivist_store must be idempotent and migration-safe."
            )
        finally:
            _exit_patches(patches)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entity_count", [1, 3, 5])
    async def test_repeated_store_with_varying_entity_counts(
        self, integration_pool, monkeypatch, entity_count
    ):
        """Parametrized: works for 1, 3, and 5 entities re-stored."""
        monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", False)
        monkeypatch.setattr("config.BM25_ENABLED", False)
        monkeypatch.setattr("config.TOPIC_ROUTING_ENABLED", False)

        import json

        entities = [f"entity-{i}" for i in range(entity_count)]
        namespace = f"param-ns-{entity_count}"

        from handlers.tools_storage import _handle_store

        patches = _common_patches()
        for ctx in patches:
            ctx.__enter__()

        try:
            # First write
            await _handle_store(
                {
                    "text": f"Parametrized store pass 1 with {entity_count} entities",
                    "agent_id": "test-agent",
                    "entities": entities,
                    "namespace": namespace,
                }
            )

            # Second write — must succeed regardless of entity_count
            result = await _handle_store(
                {
                    "text": f"Parametrized store pass 2 with {entity_count} entities",
                    "agent_id": "test-agent",
                    "entities": entities,
                    "namespace": namespace,
                }
            )
            parsed = json.loads(result[0].text)
            assert parsed.get("status") != "conflict_resolved", (
                f"Re-storing {entity_count} entities returned conflict_resolved. "
                "archivist_store must be idempotent."
            )
        finally:
            _exit_patches(patches)
