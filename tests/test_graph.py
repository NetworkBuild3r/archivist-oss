"""Tests for graph.py — entity, fact, relationship CRUD and FTS5 with real SQLite."""

import pytest


class TestEntityOperations:
    def test_upsert_entity_creates(self, graph_db):
        from graph import upsert_entity, search_entities

        eid = upsert_entity("Kubernetes", "tool")
        assert eid > 0

        results = search_entities("kubernetes")
        assert len(results) >= 1
        assert results[0]["name"] == "Kubernetes"

    def test_upsert_entity_increments_mention_count(self, graph_db):
        from graph import upsert_entity, get_db

        eid1 = upsert_entity("ArgoCD", "tool")
        eid2 = upsert_entity("ArgoCD", "tool")
        assert eid1 == eid2

        conn = get_db()
        row = conn.execute("SELECT mention_count FROM entities WHERE id = ?", (eid1,)).fetchone()
        conn.close()
        assert row["mention_count"] == 2

    def test_upsert_entity_case_insensitive(self, graph_db):
        from graph import upsert_entity

        eid1 = upsert_entity("grafana")
        eid2 = upsert_entity("Grafana")
        assert eid1 == eid2


class TestFactOperations:
    def test_add_and_retrieve_fact(self, graph_db):
        from graph import upsert_entity, add_fact, get_entity_facts

        eid = upsert_entity("PostgreSQL", "database")
        fid = add_fact(eid, "Migration approved for Q2 2026", "test.md", "chief")
        assert fid > 0

        facts = get_entity_facts(eid)
        assert len(facts) == 1
        assert "Migration approved" in facts[0]["fact_text"]

    def test_multiple_facts_for_entity(self, graph_db):
        from graph import upsert_entity, add_fact, get_entity_facts

        eid = upsert_entity("Redis")
        add_fact(eid, "Used for caching", "a.md", "chief")
        add_fact(eid, "Version 7.2 deployed", "b.md", "argo")

        facts = get_entity_facts(eid)
        assert len(facts) == 2


class TestRelationshipOperations:
    def test_add_relationship(self, graph_db):
        from graph import upsert_entity, add_relationship, get_entity_relationships

        eid1 = upsert_entity("ArgoCD")
        eid2 = upsert_entity("Kubernetes")
        add_relationship(eid1, eid2, "deploys_to", "ArgoCD deploys to K8s", "chief")

        rels = get_entity_relationships(eid1)
        assert len(rels) == 1
        assert rels[0]["relation_type"] == "deploys_to"

    def test_relationship_upsert_updates_evidence(self, graph_db):
        from graph import upsert_entity, add_relationship, get_db

        eid1 = upsert_entity("A")
        eid2 = upsert_entity("B")
        add_relationship(eid1, eid2, "uses", "evidence1", "agent1")
        add_relationship(eid1, eid2, "uses", "evidence2", "agent1")

        conn = get_db()
        row = conn.execute(
            "SELECT confidence, evidence FROM relationships WHERE source_entity_id=? AND target_entity_id=?",
            (eid1, eid2),
        ).fetchone()
        conn.close()
        assert row["confidence"] == 1.0  # capped at 1.0 by min(confidence+0.1, 1.0)
        assert row["evidence"] == "evidence2"  # updated to latest


class TestSearchEntities:
    def test_search_by_partial_name(self, graph_db):
        from graph import upsert_entity, search_entities

        upsert_entity("GitLab CI/CD", "tool")
        results = search_entities("gitlab")
        assert len(results) >= 1

    def test_search_limit(self, graph_db):
        from graph import upsert_entity, search_entities

        for i in range(20):
            upsert_entity(f"entity_{i}", "test")
        results = search_entities("entity", limit=5)
        assert len(results) == 5

    def test_search_empty_query(self, graph_db):
        from graph import search_entities
        results = search_entities("")
        assert isinstance(results, list)


class TestCuratorState:
    def test_set_and_get(self, graph_db):
        from graph import set_curator_state, get_curator_state

        set_curator_state("test_key", "test_value")
        assert get_curator_state("test_key") == "test_value"

    def test_overwrite(self, graph_db):
        from graph import set_curator_state, get_curator_state

        set_curator_state("k", "v1")
        set_curator_state("k", "v2")
        assert get_curator_state("k") == "v2"

    def test_missing_key(self, graph_db):
        from graph import get_curator_state
        assert get_curator_state("nonexistent") is None


class TestFTS5:
    def test_upsert_and_search(self, graph_db):
        from graph import upsert_fts_chunk, search_fts

        upsert_fts_chunk(
            qdrant_id="abc-123",
            text="The deployment pipeline uses ArgoCD for continuous delivery",
            file_path="agents/argo/2026-03-21.md",
            chunk_index=0,
            agent_id="argo",
            namespace="deployer",
        )

        results = search_fts("deployment pipeline")
        assert len(results) >= 1
        assert results[0]["qdrant_id"] == "abc-123"

    def test_search_with_namespace_filter(self, graph_db):
        from graph import upsert_fts_chunk, search_fts

        upsert_fts_chunk("id1", "kubernetes cluster health", "a.md", 0, "argo", "deployer")
        upsert_fts_chunk("id2", "kubernetes monitoring dashboards", "b.md", 0, "grafgreg", "pipeline")

        results = search_fts("kubernetes", namespace="deployer")
        assert all(r["namespace"] == "deployer" for r in results)

    def test_delete_by_file(self, graph_db):
        from graph import upsert_fts_chunk, delete_fts_chunks_by_file, search_fts

        upsert_fts_chunk("id1", "some text content", "file_a.md", 0)
        upsert_fts_chunk("id2", "other text content", "file_a.md", 1)
        upsert_fts_chunk("id3", "different file content", "file_b.md", 0)

        delete_fts_chunks_by_file("file_a.md")

        results = search_fts("content")
        ids = [r["qdrant_id"] for r in results]
        assert "id1" not in ids
        assert "id2" not in ids
        assert "id3" in ids
