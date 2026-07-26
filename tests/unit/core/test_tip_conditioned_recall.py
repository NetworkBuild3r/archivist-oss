"""INIT-007/SPEC-003 — conditioned tip ranking helpers (no embed)."""

from __future__ import annotations

from archivist.core.trajectory import (
    rank_tip_rows,
    score_tip_relevance,
    tokenize_tip_query,
)


class TestTokenizeTipQuery:
    def test_strips_stopwords(self):
        tokens = tokenize_tip_query("Always use express shipping for orders")
        assert "express" in tokens
        assert "shipping" in tokens
        assert "always" not in tokens
        assert "for" not in tokens


class TestScoreTipRelevance:
    def test_relevant_scores_higher(self):
        relevant = {
            "tip_text": "Prefer express shipping for rush orders",
            "context": "",
            "category": "strategy",
        }
        irrelevant = {"tip_text": "Retry OAuth on 401", "context": "", "category": "recovery"}
        q = "express shipping order"
        assert score_tip_relevance(relevant, q) > score_tip_relevance(irrelevant, q)

    def test_empty_query_scores_zero(self):
        row = {"tip_text": "Prefer express shipping"}
        assert score_tip_relevance(row, "") == 0.0
        assert score_tip_relevance(row, "the and") == 0.0


class TestRankTipRows:
    def test_relevant_beats_newer_irrelevant(self):
        # Newer irrelevant tip would win under pure recency
        rows = [
            {
                "id": "new",
                "tip_text": "Rotate API keys monthly",
                "context": "",
                "category": "optimization",
                "created_at": "2026-07-26T12:00:00+00:00",
            },
            {
                "id": "old",
                "tip_text": "Prefer express shipping for rush orders",
                "context": "",
                "category": "strategy",
                "created_at": "2026-07-01T12:00:00+00:00",
            },
        ]
        ranked = rank_tip_rows(rows, "express shipping rush", limit=2)
        assert ranked[0]["id"] == "old"
        assert ranked[1]["id"] == "new"

    def test_empty_query_preserves_recency_order(self):
        rows = [
            {"id": "a", "tip_text": "first", "created_at": "2026-07-26"},
            {"id": "b", "tip_text": "second", "created_at": "2026-07-01"},
        ]
        ranked = rank_tip_rows(rows, "", limit=2)
        assert [r["id"] for r in ranked] == ["a", "b"]

    def test_limit_truncates(self):
        rows = [
            {"id": "1", "tip_text": "express shipping", "created_at": "2026-07-26"},
            {"id": "2", "tip_text": "express shipping also", "created_at": "2026-07-25"},
            {"id": "3", "tip_text": "oauth retry", "created_at": "2026-07-24"},
        ]
        ranked = rank_tip_rows(rows, "express shipping", limit=1)
        assert len(ranked) == 1
        assert "express" in ranked[0]["tip_text"]
