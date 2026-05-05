"""Unit tests for tiering, temporal decay, contradiction detection, and compressed index."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.retrieval]


class TestTiering:
    def test_select_tier_l0(self):
        from tiering import select_tier

        hit = {"text": "full text here", "l0": "short", "l1": "medium overview"}
        assert select_tier(hit, "l0") == "short"

    def test_select_tier_l1(self):
        from tiering import select_tier

        hit = {"text": "full text here", "l0": "short", "l1": "medium overview"}
        assert select_tier(hit, "l1") == "medium overview"

    def test_select_tier_l2(self):
        from tiering import select_tier

        hit = {"text": "full text here", "l0": "short", "l1": "medium overview"}
        assert select_tier(hit, "l2") == "full text here"

    def test_select_tier_missing_l0_falls_back(self):
        from tiering import select_tier

        hit = {"text": "full text here"}
        result = select_tier(hit, "l0")
        assert "full text" in result


class TestTemporalDecay:
    def test_temporal_decay_recent_scores_higher(self):
        from graph_retrieval import apply_temporal_decay

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        results = [
            {"score": 0.9, "date": today, "content_date": today},
            {"score": 0.9, "date": "2020-01-01", "content_date": "2020-01-01"},
        ]
        decayed = apply_temporal_decay(results, halflife_days=30)
        assert decayed[0]["score"] > decayed[1]["score"]

    def test_temporal_decay_preserves_original(self):
        from graph_retrieval import apply_temporal_decay

        results = [{"score": 0.8, "date": "2025-06-01", "content_date": "2025-06-01"}]
        decayed = apply_temporal_decay(results, halflife_days=30)
        assert "original_score" in decayed[0]
        assert decayed[0]["original_score"] == 0.8

    def test_temporal_decay_skips_without_content_date(self):
        """Results without content_date (inferred index date) should not be decayed."""
        from graph_retrieval import apply_temporal_decay

        results = [{"score": 0.9, "date": "2020-01-01"}]
        decayed = apply_temporal_decay(results, halflife_days=30)
        assert decayed[0]["score"] == 0.9
        assert "original_score" not in decayed[0]


class TestContradictionDetection:
    async def test_detect_contradictions_opposing_keywords(self):
        import graph_retrieval

        mock_facts = [
            {
                "fact_text": "Service is enabled and running",
                "agent_id": "agent-a",
                "created_at": "2026-01-01",
            },
            {
                "fact_text": "Service is disabled",
                "agent_id": "agent-b",
                "created_at": "2026-01-02",
            },
        ]
        original_fn = graph_retrieval.get_entity_facts
        graph_retrieval.get_entity_facts = AsyncMock(return_value=mock_facts)
        try:
            contras = await graph_retrieval.detect_contradictions(1)
            assert len(contras) >= 1
            assert "enabled" in contras[0]["trigger"] or "disabled" in contras[0]["trigger"]
        finally:
            graph_retrieval.get_entity_facts = original_fn

    async def test_detect_contradictions_same_agent_skipped(self):
        import graph_retrieval

        mock_facts = [
            {
                "fact_text": "Service enabled",
                "agent_id": "agent-a",
                "created_at": "2026-01-01",
            },
            {
                "fact_text": "Service disabled",
                "agent_id": "agent-a",
                "created_at": "2026-01-02",
            },
        ]
        original_fn = graph_retrieval.get_entity_facts
        graph_retrieval.get_entity_facts = AsyncMock(return_value=mock_facts)
        try:
            contras = await graph_retrieval.detect_contradictions(1)
            assert len(contras) == 0
        finally:
            graph_retrieval.get_entity_facts = original_fn


class TestRetrievalTraceV05:
    def test_retrieval_trace_v05_fields(self):
        from rlm_retriever import _retrieval_trace

        trace = _retrieval_trace(
            vector_limit=64,
            coarse_count=50,
            deduped_count=45,
            threshold=0.65,
            after_threshold_count=30,
            after_rerank_count=10,
            parent_enriched=True,
            refinement_chunks=10,
            graph_entities_found=3,
            graph_context_items=8,
            temporal_decay_applied=True,
            tier="l1",
        )
        assert trace["graph_retrieval_enabled"] is not None
        assert trace["graph_entities_found"] == 3
        assert trace["graph_context_items"] == 8
        assert trace["temporal_decay_applied"] is True
        assert trace["tier"] == "l1"


class TestCompressedIndex:
    @pytest.mark.asyncio
    async def test_compressed_index_empty_namespace(self, async_pool):
        from compressed_index import build_namespace_index

        result = await build_namespace_index("test-ns-empty-xyz-unique")
        assert "No indexed knowledge" in result
