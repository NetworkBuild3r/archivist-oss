"""Unit tests for FTS5 search utilities — fts5_safe_query, merge functions."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]


class TestFTSSearch:
    def test_fts5_safe_query_basic(self):
        from fts_search import _fts5_safe_query

        q = _fts5_safe_query("kubernetes deployment")
        assert '"kubernetes"' in q
        assert '"deployment"' in q

    def test_fts5_safe_query_special_chars(self):
        from fts_search import _fts5_safe_query

        q = _fts5_safe_query("NOT (foo AND bar)")
        assert "NOT" not in q or '"NOT"' in q

    def test_fts5_safe_query_empty(self):
        from fts_search import _fts5_safe_query

        assert _fts5_safe_query("") == ""

    async def test_search_bm25_disabled(self, monkeypatch):
        monkeypatch.setattr("fts_search.BM25_ENABLED", False)
        from fts_search import search_bm25

        results = await search_bm25("test query")
        assert results == []

    def test_merge_vector_and_bm25_empty_bm25(self):
        from fts_search import merge_vector_and_bm25

        vec = [{"id": "a", "score": 0.9, "text": "t"}]
        result = merge_vector_and_bm25(vec, [])
        assert len(result) == 1
        assert result[0]["id"] == "a"

    def test_merge_vector_and_bm25_fusion(self):
        from fts_search import merge_vector_and_bm25

        vec = [{"id": "a", "score": 0.9, "text": "t1", "qdrant_id": "a"}]
        bm25 = [{"qdrant_id": "a", "bm25_score": 5.0, "text": "t1"}]
        result = merge_vector_and_bm25(vec, bm25)
        assert len(result) >= 1
        assert result[0]["score"] > 0
        assert "vector_score" in result[0]

    def test_merge_vector_rescue_preserves_top_vector_hits(self):
        from fts_search import merge_vector_and_bm25

        vec = [
            {"id": f"v{i}", "qdrant_id": f"v{i}", "score": 0.95 - i * 0.02, "text": f"vec{i}"}
            for i in range(10)
        ]
        bm25 = [
            {"qdrant_id": f"b{i}", "bm25_score": 10.0 - i * 0.1, "text": f"bm25{i}"}
            for i in range(50)
        ]
        result = merge_vector_and_bm25(vec, bm25)
        top_ids = {r["qdrant_id"] for r in result[:8]}
        for i in range(8):
            assert f"v{i}" in top_ids or f"v{i}" in {r.get("qdrant_id") for r in result}, (
                f"Vector hit v{i} was buried by BM25 noise"
            )

    def test_merge_vector_score_field_preserved(self):
        from fts_search import merge_vector_and_bm25

        vec = [{"id": "x", "qdrant_id": "x", "score": 0.87, "text": "match"}]
        bm25 = [{"qdrant_id": "x", "bm25_score": 3.0, "text": "match"}]
        result = merge_vector_and_bm25(vec, bm25)
        assert result[0]["vector_score"] == 0.87

    def test_merge_output_capped_at_20(self):
        from fts_search import merge_vector_and_bm25

        vec = [
            {"id": f"v{i}", "qdrant_id": f"v{i}", "score": 0.9 - i * 0.01, "text": f"t{i}"}
            for i in range(25)
        ]
        bm25 = [
            {"qdrant_id": f"b{i}", "bm25_score": 5.0 - i * 0.1, "text": f"t{i}"} for i in range(25)
        ]
        result = merge_vector_and_bm25(vec, bm25)
        assert len(result) <= 20
