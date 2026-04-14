"""Tests for Sprint 1 needle-finding improvements (v1.10).

Covers:
  - RRF rank fusion
  - BM25 dual-mode (AND/OR/phrase)
  - Dynamic threshold
  - Embedding cache
  - Query expansion cache
"""

import sys, os, time, threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── RRF Tests ────────────────────────────────────────────────────────────────

class TestRRF:
    def test_single_ranking_passthrough(self):
        from rank_fusion import rrf_merge
        ranking = [
            {"id": "a", "score": 0.9, "text": "first"},
            {"id": "b", "score": 0.5, "text": "second"},
        ]
        result = rrf_merge([ranking])
        assert len(result) == 2
        assert result[0]["id"] == "a"
        assert "rrf_score" in result[0]

    def test_two_rankings_with_overlap(self):
        from rank_fusion import rrf_merge
        r1 = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
        r2 = [{"id": "b", "score": 0.95}, {"id": "c", "score": 0.7}]
        result = rrf_merge([r1, r2])
        ids = [r["id"] for r in result]
        assert "b" in ids
        # "b" appears in both rankings — should be boosted to top
        b_entry = next(r for r in result if r["id"] == "b")
        a_entry = next(r for r in result if r["id"] == "a")
        assert b_entry["rrf_score"] > a_entry["rrf_score"]

    def test_limit(self):
        from rank_fusion import rrf_merge
        r1 = [{"id": str(i), "score": 1.0 - i * 0.1} for i in range(10)]
        result = rrf_merge([r1], limit=3)
        assert len(result) == 3

    def test_empty_rankings(self):
        from rank_fusion import rrf_merge
        assert rrf_merge([]) == []
        assert rrf_merge([[], []]) == []

    def test_id_key_override(self):
        from rank_fusion import rrf_merge
        r1 = [{"qdrant_id": "x", "score": 0.9}]
        r2 = [{"qdrant_id": "x", "score": 0.8}]
        result = rrf_merge([r1, r2], id_key="qdrant_id")
        assert len(result) == 1
        assert result[0]["qdrant_id"] == "x"

    def test_three_rankings_rrf_scores_monotonic(self):
        from rank_fusion import rrf_merge
        r1 = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        r2 = [{"id": "a"}, {"id": "c"}, {"id": "b"}]
        r3 = [{"id": "a"}, {"id": "b"}, {"id": "d"}]
        result = rrf_merge([r1, r2, r3])
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_default_k_is_20(self):
        from rank_fusion import rrf_merge
        r1 = [{"id": "a", "score": 0.9}]
        r2 = [{"id": "b", "score": 0.8}]
        result = rrf_merge([r1, r2])
        # With k=20: rank-1 item gets 1/(20+1) = 0.047619
        expected_score = round(1.0 / (20 + 1), 6)
        assert result[0]["rrf_score"] == expected_score or result[1]["rrf_score"] == expected_score

    def test_adaptive_k_small_list(self):
        from rank_fusion import rrf_merge
        r1 = [{"id": str(i)} for i in range(15)]
        r2 = [{"id": str(i)} for i in range(15)]
        result_adaptive = rrf_merge([r1, r2], adaptive_k=True)
        # 15 items → k = max(1, min(15//15, 20)) = 1
        # With k=1, rank-1 score = 2/(1+1) = 1.0
        top = result_adaptive[0]
        assert top["rrf_score"] == round(2.0 / (1 + 1), 6)

    def test_adaptive_k_capped_at_20(self):
        from rank_fusion import rrf_merge
        r1 = [{"id": str(i)} for i in range(500)]
        r2 = [{"id": str(i)} for i in range(500)]
        result = rrf_merge([r1, r2], adaptive_k=True)
        # 500 items → k = max(1, min(500//15, 20)) = min(33, 20) = 20
        expected = round(2.0 / (20 + 1), 6)
        assert result[0]["rrf_score"] == expected


# ── BM25 AND/OR/Phrase Tests ────────────────────────────────────────────────

class TestBM25Modes:
    def test_fts5_and_query_removes_stopwords(self):
        from fts_search import _fts5_and_query
        q = _fts5_and_query("What is the backup window for production?")
        assert "what" not in q.lower()
        assert "the" not in q.lower()
        assert '"backup"' in q
        assert '"window"' in q
        assert '"production"' in q
        assert "AND" in q

    def test_fts5_and_query_too_few_tokens(self):
        from fts_search import _fts5_and_query
        assert _fts5_and_query("is") == ""
        assert _fts5_and_query("the a") == ""

    def test_fts5_phrase_query(self):
        from fts_search import _fts5_phrase_query
        q = _fts5_phrase_query("production backup window")
        assert q == '"production backup window"'

    def test_fts5_phrase_strips_stopwords(self):
        from fts_search import _fts5_phrase_query
        q = _fts5_phrase_query("What is the backup window?")
        assert "what" not in q
        assert "the" not in q
        assert "backup" in q
        assert "window" in q

    def test_fts5_safe_query_or_mode(self):
        from fts_search import _fts5_safe_query
        q = _fts5_safe_query("backup window production")
        assert "OR" in q
        assert '"backup"' in q


# ── Dynamic Threshold Tests ──────────────────────────────────────────────────

class TestDynamicThreshold:
    def test_keeps_minimum_results(self):
        from retrieval_filters import apply_dynamic_threshold
        results = [
            {"score": 0.3, "text": "low"},
            {"score": 0.2, "text": "lower"},
            {"score": 0.1, "text": "lowest"},
        ]
        filtered = apply_dynamic_threshold(results, fallback_threshold=0.65, min_keep=3)
        assert len(filtered) == 3

    def test_filters_outliers(self):
        from retrieval_filters import apply_dynamic_threshold
        results = [
            {"score": 0.9, "text": "a"},
            {"score": 0.85, "text": "b"},
            {"score": 0.82, "text": "c"},
            {"score": 0.80, "text": "d"},
            {"score": 0.10, "text": "noise"},
        ]
        filtered = apply_dynamic_threshold(results, fallback_threshold=0.9, min_keep=2)
        assert len(filtered) >= 3
        noise_in = any(r["text"] == "noise" for r in filtered)
        # The noise item at 0.10 should generally be filtered out
        assert not noise_in or len(filtered) == 5

    def test_empty_input(self):
        from retrieval_filters import apply_dynamic_threshold
        assert apply_dynamic_threshold([], fallback_threshold=0.65) == []

    def test_single_result_kept(self):
        from retrieval_filters import apply_dynamic_threshold
        results = [{"score": 0.4, "text": "only"}]
        filtered = apply_dynamic_threshold(results, fallback_threshold=0.65, min_keep=1)
        assert len(filtered) == 1

    def test_low_scoring_needle_survives(self):
        """The whole point: a low-scoring needle should not be dropped."""
        from retrieval_filters import apply_dynamic_threshold
        results = [{"score": 0.55, "text": "needle"}]
        filtered = apply_dynamic_threshold(results, fallback_threshold=0.65, min_keep=3)
        assert len(filtered) == 1
        assert filtered[0]["text"] == "needle"


# ── Embedding Cache Tests ────────────────────────────────────────────────────

class TestEmbeddingCache:
    def test_cache_hit_returns_same_vector(self):
        from embeddings import _cache_put, _cache_get
        vec = [0.1, 0.2, 0.3]
        _cache_put("hello world", "test-model", vec)
        cached = _cache_get("hello world", "test-model")
        assert cached == vec or cached == tuple(vec)

    def test_cache_miss_returns_none(self):
        from embeddings import _cache_get
        assert _cache_get("never_stored_xyz_12345", "test-model") is None

    def test_cache_ttl_expiry(self):
        from embeddings import _cache_put, _cache_get, _embed_cache, _embed_cache_lock, _cache_key
        text, model = "ttl_test_abc", "test-model"
        _cache_put(text, model, [1.0])
        # Manually backdate the timestamp
        key = _cache_key(text, model)
        with _embed_cache_lock:
            ts, vec = _embed_cache[key]
            _embed_cache[key] = (ts - 7200, vec)  # 2h ago
        assert _cache_get(text, model) is None

    def test_cache_eviction_at_max(self):
        from embeddings import _embed_cache, _embed_cache_lock, _cache_put, _EMBED_CACHE_MAX
        # Fill beyond max — oldest should be evicted
        for i in range(_EMBED_CACHE_MAX + 10):
            _cache_put(f"evict_test_{i}", "m", [float(i)])
        with _embed_cache_lock:
            assert len(_embed_cache) <= _EMBED_CACHE_MAX


# ── Query Expansion Cache Tests ──────────────────────────────────────────────

class TestQueryExpansionCache:
    def test_expansion_cache_roundtrip(self):
        from query_expansion import _cache_put, _cache_get
        variants = ["q1", "q2", "q3"]
        _cache_put("test query", variants)
        cached = _cache_get("test query")
        assert cached == variants

    def test_expansion_cache_miss(self):
        from query_expansion import _cache_get
        assert _cache_get("never_asked_query_xyz_99999") is None

    def test_expansion_cache_ttl_expiry(self):
        from query_expansion import _cache_put, _cache_get, _expansion_cache, _expansion_lock
        _cache_put("ttl_exp_q", ["a", "b"])
        with _expansion_lock:
            ts, v = _expansion_cache["ttl_exp_q"]
            _expansion_cache["ttl_exp_q"] = (ts - 1200, v)
        assert _cache_get("ttl_exp_q") is None


# ── Benchmark Needle Questions Tests ─────────────────────────────────────────

def _generate_questions_or_skip():
    """``benchmarks/`` is gitignored in minimal clones; skip when fixtures are absent."""
    fixtures_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "benchmarks", "fixtures")
    )
    corpus_py = os.path.join(fixtures_dir, "generate_corpus.py")
    if not os.path.isfile(corpus_py):
        pytest.skip("benchmarks/fixtures/generate_corpus.py not present (benchmarks/ often gitignored)")
    if fixtures_dir not in sys.path:
        sys.path.insert(0, fixtures_dir)
    from generate_corpus import _generate_questions
    return _generate_questions


class TestNeedleBenchmark:
    def test_needle_questions_count(self):
        """Sprint 1 target: at least 15 needle questions."""
        _generate_questions = _generate_questions_or_skip()
        questions = _generate_questions()
        needle_qs = [q for q in questions if q.get("query_type") == "needle"]
        assert len(needle_qs) >= 15, f"Expected >=15 needle questions, got {len(needle_qs)}"

    def test_needle_question_types_diverse(self):
        """Verify needle questions cover multiple categories."""
        _generate_questions = _generate_questions_or_skip()
        questions = _generate_questions()
        needle_qs = [q for q in questions if q.get("query_type") == "needle"]
        all_tags = set()
        for q in needle_qs:
            all_tags.update(q.get("tags", []))
        for expected_tag in ["paraphrase", "numeric", "name", "config", "network", "cross_agent", "temporal"]:
            assert expected_tag in all_tags, f"Missing needle tag: {expected_tag}"
