import pytest
from tests.integration.conftest import skip_on_postgres

pytestmark = [pytest.mark.integration, pytest.mark.storage]
"""Tests for graph.py — entity, fact, relationship CRUD and FTS5 with real SQLite."""


class TestEntityOperations:
    async def test_upsert_entity_creates(self, async_pool):
        from graph import search_entities, upsert_entity

        eid = await upsert_entity("Kubernetes", "tool")
        assert eid > 0

        results = await search_entities("kubernetes")
        assert len(results) >= 1
        assert results[0]["name"] == "Kubernetes"

    async def test_upsert_entity_increments_mention_count(self, async_pool):
        from graph import get_db, upsert_entity

        eid1 = await upsert_entity("ArgoCD", "tool")
        eid2 = await upsert_entity("ArgoCD", "tool")
        assert eid1 == eid2

        conn = get_db()
        row = conn.execute("SELECT mention_count FROM entities WHERE id = ?", (eid1,)).fetchone()
        conn.close()
        assert row["mention_count"] == 2

    async def test_upsert_entity_case_insensitive(self, async_pool):
        from graph import upsert_entity

        eid1 = await upsert_entity("grafana")
        eid2 = await upsert_entity("Grafana")
        assert eid1 == eid2


class TestFactOperations:
    async def test_add_and_retrieve_fact(self, async_pool):
        from graph import add_fact, get_entity_facts, upsert_entity

        eid = await upsert_entity("PostgreSQL", "database")
        fid = await add_fact(eid, "Migration approved for Q2 2026", "test.md", "chief")
        assert fid > 0

        facts = await get_entity_facts(eid)
        assert len(facts) == 1
        assert "Migration approved" in facts[0]["fact_text"]

    async def test_multiple_facts_for_entity(self, async_pool):
        from graph import add_fact, get_entity_facts, upsert_entity

        eid = await upsert_entity("Redis")
        await add_fact(eid, "Used for caching", "a.md", "chief")
        await add_fact(eid, "Version 7.2 deployed", "b.md", "argo")

        facts = await get_entity_facts(eid)
        assert len(facts) == 2


class TestRelationshipOperations:
    async def test_add_relationship(self, async_pool):
        from graph import add_relationship, get_entity_relationships, upsert_entity

        eid1 = await upsert_entity("ArgoCD")
        eid2 = await upsert_entity("Kubernetes")
        await add_relationship(eid1, eid2, "deploys_to", "ArgoCD deploys to K8s", "chief")

        rels = await get_entity_relationships(eid1)
        assert len(rels) == 1
        assert rels[0]["relation_type"] == "deploys_to"

    async def test_relationship_upsert_updates_evidence(self, async_pool):
        from graph import add_relationship, get_db, upsert_entity

        eid1 = await upsert_entity("A")
        eid2 = await upsert_entity("B")
        await add_relationship(eid1, eid2, "uses", "evidence1", "agent1")
        await add_relationship(eid1, eid2, "uses", "evidence2", "agent1")

        conn = get_db()
        row = conn.execute(
            "SELECT confidence, evidence FROM relationships WHERE source_entity_id=? AND target_entity_id=?",
            (eid1, eid2),
        ).fetchone()
        conn.close()
        assert row["confidence"] == 1.0  # capped at 1.0 by min(confidence+0.1, 1.0)
        assert row["evidence"] == "evidence2"  # updated to latest


class TestSearchEntities:
    async def test_search_by_partial_name(self, async_pool):
        from graph import search_entities, upsert_entity

        await upsert_entity("GitLab CI/CD", "tool")
        results = await search_entities("gitlab")
        assert len(results) >= 1

    async def test_search_limit(self, async_pool):
        from graph import search_entities, upsert_entity

        for i in range(20):
            await upsert_entity(f"entity_{i}", "test")
        results = await search_entities("entity", limit=5)
        assert len(results) == 5

    async def test_search_empty_query(self, async_pool):
        from graph import search_entities

        results = await search_entities("")
        assert isinstance(results, list)


class TestCuratorState:
    async def test_set_and_get(self, async_pool):
        from graph import get_curator_state, set_curator_state

        await set_curator_state("test_key", "test_value")
        assert await get_curator_state("test_key") == "test_value"

    async def test_overwrite(self, async_pool):
        from graph import get_curator_state, set_curator_state

        await set_curator_state("k", "v1")
        await set_curator_state("k", "v2")
        assert await get_curator_state("k") == "v2"

    async def test_missing_key(self, async_pool):
        from graph import get_curator_state

        assert await get_curator_state("nonexistent") is None


@skip_on_postgres
class TestFTS5:
    async def test_upsert_and_search(self, async_pool):
        from graph import search_fts, upsert_fts_chunk

        await upsert_fts_chunk(
            qdrant_id="abc-123",
            text="The deployment pipeline uses ArgoCD for continuous delivery",
            file_path="agents/argo/2026-03-21.md",
            chunk_index=0,
            agent_id="argo",
            namespace="deployer",
        )

        results = await search_fts("deployment pipeline")
        assert len(results) >= 1
        assert results[0]["qdrant_id"] == "abc-123"

    async def test_search_with_namespace_filter(self, async_pool):
        from graph import search_fts, upsert_fts_chunk

        await upsert_fts_chunk("id1", "kubernetes cluster health", "a.md", 0, "argo", "deployer")
        await upsert_fts_chunk(
            "id2", "kubernetes monitoring dashboards", "b.md", 0, "grafgreg", "pipeline"
        )

        results = await search_fts("kubernetes", namespace="deployer")
        assert all(r["namespace"] == "deployer" for r in results)

    async def test_delete_by_file(self, async_pool):
        from graph import delete_fts_chunks_by_file, search_fts, upsert_fts_chunk

        await upsert_fts_chunk("id1", "some text content", "file_a.md", 0)
        await upsert_fts_chunk("id2", "other text content", "file_a.md", 1)
        await upsert_fts_chunk("id3", "different file content", "file_b.md", 0)

        await delete_fts_chunks_by_file("file_a.md")

        results = await search_fts("content")
        ids = [r["qdrant_id"] for r in results]
        assert "id1" not in ids
        assert "id2" not in ids
        assert "id3" in ids


class TestEntityUpsertIdempotency:
    """Regression tests for the SELECT-then-INSERT race condition (v1.12 fix)."""

    @pytest.mark.integration
    async def test_concurrent_entity_upsert_no_integrity_error(self, async_pool):
        """10 concurrent upserts of the exact same entity must all succeed with the same ID.

        This is the canonical regression test for the duplicate key race condition
        that caused 'UNIQUE constraint failed: entities.name' IntegrityErrors during
        memory migration when multiple agents upserted the same entity simultaneously.
        """
        import asyncio

        from graph import upsert_entity

        async def upsert_brian():
            return await upsert_entity(
                name="brian",
                entity_type="person",
                namespace="test-concurrent",
            )

        results = await asyncio.gather(*[upsert_brian() for _ in range(10)], return_exceptions=True)

        exceptions = [r for r in results if isinstance(r, Exception)]
        assert not exceptions, f"Got unexpected exceptions: {exceptions}"

        ids = [r for r in results if isinstance(r, int)]
        assert len(ids) == 10, f"Expected 10 results, got {len(ids)}"
        assert len(set(ids)) == 1, f"Expected all same ID, got distinct IDs: {set(ids)}"
        assert all(i > 0 for i in ids), f"All IDs must be positive integers, got: {ids}"

    @pytest.mark.integration
    async def test_concurrent_upsert_across_namespaces(self, async_pool):
        """Same entity name in different namespaces must produce different IDs."""
        import asyncio

        from graph import upsert_entity

        ns_a_id, ns_b_id = await asyncio.gather(
            upsert_entity("shared-entity", "system", namespace="ns-alpha"),
            upsert_entity("shared-entity", "system", namespace="ns-beta"),
        )
        assert ns_a_id > 0
        assert ns_b_id > 0
        assert ns_a_id != ns_b_id, "Same name in different namespaces must be separate entities"

    @pytest.mark.integration
    async def test_upsert_retention_class_escalation(self, async_pool):
        """Higher retention class must win on conflict, lower must not downgrade."""
        from graph import get_db, upsert_entity

        eid = await upsert_entity("raven", "agent", retention_class="ephemeral", namespace="rc-test")
        await upsert_entity("raven", "agent", retention_class="permanent", namespace="rc-test")
        await upsert_entity("raven", "agent", retention_class="standard", namespace="rc-test")

        conn = get_db()
        row = conn.execute(
            "SELECT retention_class, mention_count FROM entities WHERE name=? AND namespace=?",
            ("raven", "rc-test"),
        ).fetchone()
        conn.close()

        assert row["retention_class"] == "permanent", (
            f"permanent should have won, got: {row['retention_class']}"
        )
        assert row["mention_count"] == 3, f"Expected 3 mentions, got {row['mention_count']}"
