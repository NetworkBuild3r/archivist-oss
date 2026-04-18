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


class TestPgQueryBuilders:
    """Tests for Postgres tsquery builder functions."""

    def test_pg_tsquery_or_basic(self):
        from fts_search import _pg_tsquery_or

        result = _pg_tsquery_or("kubernetes deployment")
        assert "'kubernetes'" in result
        assert "'deployment'" in result
        assert "|" in result

    def test_pg_tsquery_or_single_token(self):
        from fts_search import _pg_tsquery_or

        result = _pg_tsquery_or("kubernetes")
        assert "'kubernetes'" in result
        assert "|" not in result

    def test_pg_tsquery_or_empty(self):
        from fts_search import _pg_tsquery_or

        assert _pg_tsquery_or("") == ""
        assert _pg_tsquery_or("   ") == ""

    def test_pg_tsquery_or_strips_punctuation(self):
        from fts_search import _pg_tsquery_or

        result = _pg_tsquery_or("(foo) [bar]!")
        assert "(" not in result
        assert "[" not in result

    def test_pg_tsquery_and_basic(self):
        from fts_search import _pg_tsquery_and

        result = _pg_tsquery_and("kubernetes pod deployment")
        assert "'kubernetes'" in result
        assert "'pod'" in result
        assert "&" in result

    def test_pg_tsquery_and_removes_stopwords(self):
        from fts_search import _pg_tsquery_and

        result = _pg_tsquery_and("the kubernetes pod")
        assert "'the'" not in result
        assert "'kubernetes'" in result

    def test_pg_tsquery_and_fewer_than_two_tokens_returns_empty(self):
        from fts_search import _pg_tsquery_and

        # Single content word after stopword removal
        assert _pg_tsquery_and("the kubernetes") == ""

    def test_pg_tsquery_and_empty(self):
        from fts_search import _pg_tsquery_and

        assert _pg_tsquery_and("") == ""

    def test_pg_tsquery_and_all_stopwords(self):
        from fts_search import _pg_tsquery_and

        assert _pg_tsquery_and("the is a an") == ""

    def test_pg_tsquery_phrase_basic(self):
        from fts_search import _pg_tsquery_phrase

        result = _pg_tsquery_phrase("kubernetes pod restart")
        assert "'kubernetes'" in result
        assert "'pod'" in result
        assert "'restart'" in result
        assert "<->" in result

    def test_pg_tsquery_phrase_removes_stopwords(self):
        from fts_search import _pg_tsquery_phrase

        result = _pg_tsquery_phrase("the kubernetes pod")
        assert "'the'" not in result

    def test_pg_tsquery_phrase_fewer_than_two_tokens_returns_empty(self):
        from fts_search import _pg_tsquery_phrase

        assert _pg_tsquery_phrase("kubernetes") == ""
        assert _pg_tsquery_phrase("the kubernetes") == ""

    def test_pg_tsquery_phrase_empty(self):
        from fts_search import _pg_tsquery_phrase

        assert _pg_tsquery_phrase("") == ""

    def test_pg_tsquery_or_ip_address(self):
        from fts_search import _pg_tsquery_or

        result = _pg_tsquery_or("192.168.1.1")
        assert result != ""

    def test_pg_tsquery_or_uuid_prefix(self):
        from fts_search import _pg_tsquery_or

        result = _pg_tsquery_or("abc12345-dead")
        assert result != ""

    def test_pg_tsquery_phrase_order(self):
        """Tokens appear in original order separated by <->."""
        from fts_search import _pg_tsquery_phrase

        result = _pg_tsquery_phrase("kubernetes pod restart")
        parts = result.split(" <-> ")
        assert len(parts) == 3
        assert parts[0] == "'kubernetes'"
        assert parts[1] == "'pod'"
        assert parts[2] == "'restart'"
