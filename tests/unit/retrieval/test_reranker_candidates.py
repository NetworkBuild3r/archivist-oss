"""Unit tests for the cross-encoder reranker — scoring, sorting, pair building."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.retrieval]


class TestRerankerCandidates:
    """Tests for src/reranker.py (Phase 2 cross-encoder reranker)."""

    def test_basic_scoring_adds_reranker_score(self):
        """rerank_candidates adds a reranker_score key to each candidate."""
        from reranker import rerank_candidates

        candidates = [
            {"text": "The server IP is 10.0.0.5", "score": 0.8},
            {"text": "Unrelated weather info", "score": 0.9},
        ]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95, 0.12]

        with patch("reranker._get_model", return_value=mock_model):
            result = asyncio.get_event_loop().run_until_complete(
                rerank_candidates("What is the server IP?", candidates, top_k=10)
            )
        assert all("reranker_score" in r for r in result)
        assert result[0]["reranker_score"] == 0.95

    def test_parent_text_included_in_pair(self):
        """_build_pair includes parent_text when present."""
        from reranker import _build_pair

        candidate = {
            "text": "Backup runs at 2am",
            "parent_text": "Server maintenance schedule for prod-us-east-1",
        }
        pair_text = _build_pair("When does backup run?", candidate)
        assert "Backup runs at 2am" in pair_text
        assert "prod-us-east-1" in pair_text
        assert "Parent context:" in pair_text

    def test_sorting_by_reranker_score(self):
        """Candidates are sorted descending by reranker_score."""
        from reranker import rerank_candidates

        candidates = [
            {"text": "low relevance", "score": 0.99},
            {"text": "medium relevance", "score": 0.50},
            {"text": "high relevance", "score": 0.30},
        ]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.5, 0.9]

        with patch("reranker._get_model", return_value=mock_model):
            result = asyncio.get_event_loop().run_until_complete(
                rerank_candidates("query", candidates, top_k=10)
            )
        scores = [r["reranker_score"] for r in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0]["text"] == "high relevance"

    def test_empty_list_returns_empty(self):
        """rerank_candidates with empty list returns empty."""
        from reranker import rerank_candidates

        result = asyncio.get_event_loop().run_until_complete(
            rerank_candidates("anything", [], top_k=10)
        )
        assert result == []
