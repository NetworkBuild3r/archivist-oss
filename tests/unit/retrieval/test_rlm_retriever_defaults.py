"""Unit tests for INIT-023/SPEC-002 defaults, HyDE gating, temporal decay, and dead-code cleanup."""

from __future__ import annotations

import importlib.util
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.unit]


class TestHyDEReadGating:
    """Read-time HyDE must be strictly gated behind HYDE_READ_ENABLED (default False)."""

    @pytest.mark.asyncio
    async def test_generate_hypothetical_document_returns_none_when_disabled(self, monkeypatch):
        import archivist.core.config as cfg
        import archivist.write.hyde as hyde

        monkeypatch.setattr(cfg, "HYDE_READ_ENABLED", False)
        with patch("archivist.write.hyde.llm_query", new=AsyncMock()) as mock_llm:
            out = await hyde.generate_hypothetical_document("What is the exact backup window?")
            assert out is None
            mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_recursive_retrieve_makes_zero_llm_calls_on_needle_query_by_default(
        self, monkeypatch
    ):
        import archivist.core.config as cfg
        import archivist.retrieval.rlm_retriever as rlm

        monkeypatch.setattr(cfg, "HYDE_READ_ENABLED", False)
        monkeypatch.setattr(rlm, "HYDE_READ_ENABLED", False)
        monkeypatch.setattr(rlm, "QUERY_EXPANSION_ENABLED", False)
        monkeypatch.setattr(rlm, "RERANKER_ENABLED", False)
        monkeypatch.setattr(rlm, "BM25_ENABLED", False)
        monkeypatch.setattr(rlm, "GRAPH_RETRIEVAL_ENABLED", False)

        from archivist.write.hyde import is_needle_query

        needle_q = "What is the exact backup window?"
        assert is_needle_query(needle_q) is True

        fake_hit = {
            "id": "chunk-1",
            "score": 0.91,
            "vector_score": 0.91,
            "text": "The exact backup window is 04:15 UTC every Sunday.",
            "file_path": "ops/backup.md",
            "date": "2026-01-15",
            "agent_id": "ops-bot",
            "namespace": "global",
        }

        async def _identity(items, *args, **kwargs):
            return items

        with (
            patch.object(rlm, "get_inventory", new=AsyncMock(return_value=None)),
            patch.object(rlm, "embed_batch", new=AsyncMock(return_value=[[0.1] * 8])),
            patch.object(rlm, "search_vectors", new=AsyncMock(return_value=[fake_hit])),
            patch.object(rlm, "extract_entity_mentions", new=AsyncMock(return_value=[])),
            patch.object(rlm, "graph_context_for_entities", new=AsyncMock(return_value=[])),
            patch.object(rlm, "build_entity_fact_results", new=AsyncMock(return_value=[])),
            patch.object(rlm, "apply_hotness_to_results", new=_identity),
            patch(
                "archivist.write.hyde.generate_hypothetical_document", new=AsyncMock()
            ) as mock_hyde,
            patch("archivist.features.llm.llm_query", new=AsyncMock()) as mock_llm,
            patch.object(rlm, "llm_query", new=AsyncMock()) as mock_rlm_llm,
        ):
            res = await rlm.recursive_retrieve(
                query=needle_q,
                refine=False,
                limit=5,
                namespace="global",
            )
            assert res["sources"]
            mock_hyde.assert_not_called()
            mock_llm.assert_not_called()
            mock_rlm_llm.assert_not_called()


class TestTemporalDecayDefaultsAndReferenceTime:
    """TEMPORAL_DECAY_HALFLIFE_DAYS=0 bypasses decay; reference_time overrides now()."""

    def test_halflife_zero_bypasses_temporal_decay(self):
        from archivist.storage.graph_retrieval import apply_temporal_decay

        items = [
            {"id": "old", "score": 0.80, "date": "2020-01-01", "content_date": "2020-01-01"},
            {"id": "new", "score": 0.70, "date": "2026-11-01", "content_date": "2026-11-01"},
        ]
        out = apply_temporal_decay(items, halflife_days=0)
        assert out[0]["id"] == "old"
        assert out[0]["score"] == pytest.approx(0.80)
        assert out[1]["score"] == pytest.approx(0.70)

    def test_default_halflife_zero_from_config_bypasses_decay(self, monkeypatch):
        import archivist.storage.graph_retrieval as gr

        monkeypatch.setattr(gr, "TEMPORAL_DECAY_HALFLIFE_DAYS", 0)
        items = [{"id": "old", "score": 0.85, "date": "2021-06-01", "content_date": "2021-06-01"}]
        out = gr.apply_temporal_decay(items)
        assert out[0]["score"] == pytest.approx(0.85)

    def test_reference_time_anchors_decay_computation(self):
        from archivist.storage.graph_retrieval import apply_temporal_decay

        items_same = [
            {"id": "fact", "score": 1.0, "date": "2023-05-01", "content_date": "2023-05-01"}
        ]
        out_same_day = apply_temporal_decay(
            items_same, halflife_days=30, reference_time="2023-05-01"
        )
        assert out_same_day[0]["score"] == pytest.approx(1.0, abs=1e-4)

        items_30d = [
            {"id": "fact", "score": 1.0, "date": "2023-05-01", "content_date": "2023-05-01"}
        ]
        out_30d = apply_temporal_decay(
            items_30d, halflife_days=30, reference_time="2023/05/31 (Wed) 12:00"
        )
        assert out_30d[0]["score"] == pytest.approx(0.5, abs=0.05)


class TestTieringZeroLLMByDefault:
    """generate_tiers skips LLM calls when TIERED_CONTEXT_ENABLED=False."""

    @pytest.mark.asyncio
    async def test_generate_tiers_deterministic_without_llm_when_disabled(self, monkeypatch):
        import archivist.write.tiering as tiering

        monkeypatch.setattr(tiering, "TIERED_CONTEXT_ENABLED", False)
        with patch("archivist.features.llm.llm_query", new=AsyncMock()) as mock_llm:
            tiers = await tiering.generate_tiers(
                "First sentence states the policy clearly. Second sentence gives details. " * 10
            )
            assert tiers["l0"]
            assert tiers["l1"]
            assert tiers["l2"]
            mock_llm.assert_not_called()


class TestDeadModulesRemoved:
    """Unused XGBoost LTR stub modules must be removed."""

    def test_ranker_modules_absent(self):
        assert importlib.util.find_spec("archivist.retrieval.ranker") is None
        assert importlib.util.find_spec("archivist.retrieval.ranker_train") is None
