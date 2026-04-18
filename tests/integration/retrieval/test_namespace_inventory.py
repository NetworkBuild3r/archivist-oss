"""v1.6 memory awareness: namespace inventory, query classifier, compaction multi_agent."""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.retrieval, pytest.mark.rbac]

class TestNamespaceInventory:
    async def test_inventory_counts_by_type_and_ttl_invalidate(self, async_pool, monkeypatch):
        import namespace_inventory as ni
        from graph import upsert_fts_chunk

        ni.invalidate_all()
        monkeypatch.setattr(ni, "INVENTORY_TTL_SECONDS", 3600)

        for i in range(30):
            await upsert_fts_chunk(
                f"id-s-{i}", "skill text", f"f{i}.md", i, "a1", "ns1", "2026-01-01", "skill"
            )
        for i in range(20):
            await upsert_fts_chunk(
                f"id-e-{i}",
                "experience text",
                f"e{i}.md",
                i,
                "a1",
                "ns1",
                "2026-01-01",
                "experience",
            )

        inv = ni.get_inventory("ns1")
        assert inv.total_memories == 50
        assert inv.by_type.get("skill") == 30
        assert inv.by_type.get("experience") == 20

        inv2 = ni.get_inventory("ns1")
        assert inv2.total_memories == 50
        assert inv2 is inv

        ni.invalidate("ns1")
        inv3 = ni.get_inventory("ns1")
        assert inv3.total_memories == 50

    def test_inventory_empty_namespace_string(self):
        import namespace_inventory as ni

        ni.invalidate_all()
        inv = ni.get_inventory("")
        assert isinstance(inv.total_memories, int)

class TestQueryClassifier:
    async def test_skip_small_namespace(self):
        from namespace_inventory import NamespaceInventory
        from query_classifier import classify_query, invalidate_all_cache

        invalidate_all_cache()

        inv = NamespaceInventory(
            namespace="x",
            total_memories=10,
            by_type={"skill": 10},
            top_entities=[],
            has_fleet_tips=False,
            cached_at=0.0,
        )
        assert await classify_query("how do I deploy", inv) == ""

    async def test_single_type_short_circuits(self):
        from namespace_inventory import NamespaceInventory
        from query_classifier import classify_query, invalidate_all_cache

        invalidate_all_cache()

        inv = NamespaceInventory(
            namespace="x",
            total_memories=80,
            by_type={"skill": 80},
            top_entities=[],
            has_fleet_tips=False,
            cached_at=0.0,
        )
        assert await classify_query("anything", inv) == "skill"

    async def test_skew_heuristic_no_llm(self):
        from namespace_inventory import NamespaceInventory
        from query_classifier import classify_query, invalidate_all_cache

        invalidate_all_cache()

        inv = NamespaceInventory(
            namespace="x",
            total_memories=100,
            by_type={"skill": 91, "general": 9},
            top_entities=[],
            has_fleet_tips=False,
            cached_at=0.0,
        )
        assert await classify_query("mixed question", inv) == "skill"

    async def test_llm_classification_respects_inventory(self, mock_llm):
        from namespace_inventory import NamespaceInventory
        from query_classifier import classify_query, invalidate_all_cache

        invalidate_all_cache()

        inv = NamespaceInventory(
            namespace="x",
            total_memories=100,
            by_type={"skill": 40, "experience": 40, "general": 20},
            top_entities=[],
            has_fleet_tips=False,
            cached_at=0.0,
        )
        mock_llm.return_value = "experience"
        assert await classify_query("what happened yesterday", inv) == "experience"

    async def test_llm_returns_empty_type_overridden(self, mock_llm):
        from namespace_inventory import NamespaceInventory
        from query_classifier import classify_query, invalidate_all_cache

        invalidate_all_cache()

        inv = NamespaceInventory(
            namespace="x",
            total_memories=100,
            by_type={"skill": 50, "general": 50},
            top_entities=[],
            has_fleet_tips=False,
            cached_at=0.0,
        )
        mock_llm.return_value = "experience"
        assert await classify_query("q", inv) == ""

class TestCompactionMultiAgent:
    async def test_structured_multi_agent_prompt(self, mock_llm):
        import json

        from compaction import compact_structured

        mock_llm.return_value = json.dumps(
            {
                "goal": "g",
                "progress": ["p"],
                "decisions": [],
                "next_steps": [],
                "critical_context": "",
            }
        )
        await compact_structured([("a", "t")], multi_agent=True)
        assert mock_llm.call_count >= 1
        call_kw = mock_llm.call_args[1]
        assert "multiple agents" in call_kw.get("system", "")

    async def test_flat_multi_agent_prompt(self, mock_llm):
        from compaction import compact_flat


        mock_llm.return_value = "summary"
        await compact_flat([("a", "text")], multi_agent=True)
        assert mock_llm.called
        call_kw = mock_llm.call_args[1]
        assert "multiple agents" in call_kw.get("system", "")
