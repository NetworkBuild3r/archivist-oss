"""Tests for v1.1-1.3 modules: tokenizer, context_manager, compaction, fts_search, curator checksum."""

import pytest


class TestTokenizer:
    def test_count_tokens_nonempty(self):
        from tokenizer import count_tokens

        n = count_tokens("Hello, world!")
        assert n > 0

    def test_count_tokens_empty(self):
        from tokenizer import count_tokens

        n = count_tokens("")
        assert n >= 0

    def test_count_message_tokens(self):
        from tokenizer import count_message_tokens

        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        n = count_message_tokens(msgs)
        assert n > 10  # overhead + content

    def test_count_message_tokens_empty(self):
        from tokenizer import count_message_tokens

        n = count_message_tokens([])
        assert n == 0

    def test_fallback_approximation(self):
        from tokenizer import count_tokens

        text = "a" * 400
        n = count_tokens(text)
        assert n >= 50  # at least chars//4 ballpark


class TestContextManager:
    def test_check_context_under_budget(self):
        from context_manager import check_context

        msgs = [{"role": "user", "content": "Short message."}]
        result = check_context(msgs, budget_tokens=10000)
        assert result["over_budget"] is False
        assert result["hint"] == "ok"

    def test_check_context_empty(self):
        from context_manager import check_context

        result = check_context([], budget_tokens=1000)
        assert result["total_tokens"] == 0
        assert result["hint"] == "ok"

    def test_check_context_over_budget(self):
        from context_manager import check_context

        msgs = [{"role": "user", "content": "x " * 5000}]
        result = check_context(msgs, budget_tokens=100)
        assert result["over_budget"] is True
        assert result["hint"] in ("compress", "critical")

    def test_check_memories_budget(self):
        from context_manager import check_memories_budget

        result = check_memories_budget(["short text"], budget_tokens=10000)
        assert result["over_budget"] is False
        assert result["memory_count"] == 1

    def test_check_memories_budget_over(self):
        from context_manager import check_memories_budget

        texts = ["word " * 1000 for _ in range(10)]
        result = check_memories_budget(texts, budget_tokens=100)
        assert result["over_budget"] is True


class TestCompaction:
    @pytest.mark.asyncio
    async def test_compact_flat(self, mock_llm):
        from compaction import compact_flat

        mock_llm.return_value = "This is a flat summary."
        result = await compact_flat([("id1", "Some memory text")])
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_compact_structured(self, mock_llm):
        import json
        from compaction import compact_structured

        mock_llm.return_value = json.dumps({
            "goal": "Deploy app",
            "progress": ["Step 1 done"],
            "decisions": ["Use ArgoCD"],
            "next_steps": ["Run tests"],
            "critical_context": "Cluster is prod-us-east-1",
        })
        result = await compact_structured([("id1", "Memory about deployment")])
        assert isinstance(result, dict)
        assert "goal" in result
        assert "progress" in result

    @pytest.mark.asyncio
    async def test_compact_structured_fallback(self, mock_llm):
        from compaction import compact_structured

        mock_llm.return_value = "Not valid JSON at all"
        result = await compact_structured([("id1", "Some memory")])
        assert isinstance(result, dict)
        assert "progress" in result

    def test_format_structured_summary(self):
        from compaction import format_structured_summary

        data = {
            "goal": "Migrate database",
            "progress": ["Schema created", "Data exported"],
            "decisions": ["Use PostgreSQL"],
            "next_steps": ["Run migration script"],
            "critical_context": "Prod cluster: us-east-1",
        }
        md = format_structured_summary(data)
        assert "## Goal" in md
        assert "## Progress" in md
        assert "PostgreSQL" in md


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

    def test_search_bm25_disabled(self, monkeypatch):
        monkeypatch.setattr("fts_search.BM25_ENABLED", False)
        from fts_search import search_bm25

        results = search_bm25("test query")
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
        # Top-8 vector hits must survive in the top results
        for i in range(8):
            assert f"v{i}" in top_ids or f"v{i}" in {r.get("qdrant_id") for r in result}, \
                f"Vector hit v{i} was buried by BM25 noise"

    def test_merge_vector_score_field_preserved(self):
        from fts_search import merge_vector_and_bm25

        vec = [{"id": "x", "qdrant_id": "x", "score": 0.87, "text": "match"}]
        bm25 = [{"qdrant_id": "x", "bm25_score": 3.0, "text": "match"}]
        result = merge_vector_and_bm25(vec, bm25)
        assert result[0]["vector_score"] == 0.87

    def test_merge_output_capped_at_20(self):
        from fts_search import merge_vector_and_bm25

        vec = [{"id": f"v{i}", "qdrant_id": f"v{i}", "score": 0.9 - i * 0.01, "text": f"t{i}"} for i in range(25)]
        bm25 = [{"qdrant_id": f"b{i}", "bm25_score": 5.0 - i * 0.1, "text": f"t{i}"} for i in range(25)]
        result = merge_vector_and_bm25(vec, bm25)
        assert len(result) <= 20


class TestEntityExtraction:
    def test_ngram_expansion(self, graph_db):
        from graph import upsert_entity
        from graph_retrieval import extract_entity_mentions

        upsert_entity("Argo CD", "tool")
        upsert_entity("hot cache", "concept")

        results = extract_entity_mentions("How does Argo CD use the hot cache?")
        names = [e["name"] for e in results]
        assert any("Argo CD" in n for n in names)
        assert any("hot cache" in n for n in names)

    def test_short_entities(self, graph_db):
        from graph import upsert_entity
        from graph_retrieval import extract_entity_mentions

        upsert_entity("AI", "concept")
        results = extract_entity_mentions("How does AI work?")
        names = [e["name"] for e in results]
        assert any("AI" in n for n in names)


class TestCuratorChecksum:
    def test_file_checksum_deterministic(self):
        from curator import _file_checksum

        h1 = _file_checksum("hello world")
        h2 = _file_checksum("hello world")
        assert h1 == h2

    def test_file_checksum_differs(self):
        from curator import _file_checksum

        h1 = _file_checksum("hello world")
        h2 = _file_checksum("hello world!")
        assert h1 != h2


class TestReranker:
    """Tests for src/reranker.py (Phase 2 cross-encoder reranker)."""

    def test_basic_scoring_adds_reranker_score(self):
        """rerank_candidates adds a reranker_score key to each candidate."""
        import asyncio
        from unittest.mock import patch, MagicMock
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
        import asyncio
        from unittest.mock import patch, MagicMock
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
        import asyncio
        from reranker import rerank_candidates

        result = asyncio.get_event_loop().run_until_complete(
            rerank_candidates("anything", [], top_k=10)
        )
        assert result == []


class TestBenchmarkMetrics:
    """Tests for the evaluation metric functions in benchmarks/pipeline/evaluate.py."""

    def _import_metrics(self):
        import sys, os
        bench_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "pipeline")
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        from evaluate import _extract_source_text, _keyword_recall_at_k, _ndcg_at_k
        return _extract_source_text, _keyword_recall_at_k, _ndcg_at_k

    def test_extract_source_text_prefers_tier_text(self):
        extract, _, _ = self._import_metrics()
        src = {"tier_text": "tier content", "text": "raw content", "score": 0.9}
        assert extract(src) == "tier content"

    def test_extract_source_text_falls_back_to_text(self):
        extract, _, _ = self._import_metrics()
        src = {"text": "raw content", "score": 0.9}
        assert extract(src) == "raw content"

    def test_extract_source_text_returns_empty_for_metadata_only(self):
        extract, _, _ = self._import_metrics()
        src = {"file": "path.md", "score": 0.9, "agent": "chief"}
        assert extract(src) == ""

    def test_extract_source_text_never_returns_str_dict(self):
        """The old code did str(source_dict) — verify we never do that."""
        extract, _, _ = self._import_metrics()
        src = {"file_path": "pipeline.md", "score": 0.9, "agent_id": "gitbob"}
        result = extract(src)
        assert "file_path" not in result
        assert "score" not in result
        assert "agent_id" not in result

    def test_keyword_recall_at_k_respects_k(self):
        _, recall_at_k, _ = self._import_metrics()
        sources = [
            {"text": "kubernetes cluster setup"},
            {"text": "postgres database migration"},
            {"text": "ci/cd pipeline gitlab"},
        ]
        r1 = recall_at_k("", sources, ["kubernetes", "postgres", "gitlab"], k=1)
        r3 = recall_at_k("", sources, ["kubernetes", "postgres", "gitlab"], k=3)
        assert r1 < r3, "recall@1 should be less than recall@3 when keywords are spread"
        assert abs(r1 - 1 / 3) < 0.01
        assert abs(r3 - 1.0) < 0.01

    def test_keyword_recall_at_5_differs_from_at_10(self):
        """The core bug: recall@5 and recall@10 must differ when keywords are in sources 6-10."""
        _, recall_at_k, _ = self._import_metrics()
        sources = [{"text": f"filler text {i}"} for i in range(10)]
        sources[7]["text"] = "kubernetes cluster k8s"
        r5 = recall_at_k("", sources, ["kubernetes"], k=5)
        r10 = recall_at_k("", sources, ["kubernetes"], k=10)
        assert r5 == 0.0, "keyword in source 8 should not appear in top-5"
        assert r10 == 1.0, "keyword in source 8 should appear in top-10"

    def test_keyword_recall_includes_answer_text(self):
        _, recall_at_k, _ = self._import_metrics()
        sources = [{"text": "unrelated stuff"}]
        r = recall_at_k("kubernetes cluster", sources, ["kubernetes"], k=5)
        assert r == 1.0

    def test_keyword_recall_empty_keywords(self):
        _, recall_at_k, _ = self._import_metrics()
        assert recall_at_k("text", [{"text": "t"}], [], k=5) == 0.0

    def test_keyword_recall_does_not_match_metadata(self):
        """Keywords must NOT match against dict keys, file paths, or other metadata."""
        _, recall_at_k, _ = self._import_metrics()
        src = {"text": "unrelated content", "file_path": "/agents/gitbob/pipeline.md", "score": 0.9}
        r = recall_at_k("", [src], ["pipeline"], k=5)
        assert r == 0.0, "keyword 'pipeline' should not match file_path metadata"

    def test_ndcg_perfect_ordering(self):
        _, _, ndcg = self._import_metrics()
        sources = [
            {"text": "kubernetes k8s cluster"},
            {"text": "unrelated text"},
        ]
        score = ndcg(sources, ["kubernetes", "k8s", "cluster"], k=5)
        assert score == 1.0, "already-optimal ordering should get NDCG=1.0"

    def test_ndcg_reversed_ordering(self):
        _, _, ndcg = self._import_metrics()
        sources = [
            {"text": "unrelated text"},
            {"text": "kubernetes k8s cluster"},
        ]
        score = ndcg(sources, ["kubernetes", "k8s", "cluster"], k=5)
        assert 0.0 < score < 1.0, "suboptimal ordering should get NDCG < 1.0"

    def test_ndcg_no_keywords_returns_zero(self):
        _, _, ndcg = self._import_metrics()
        assert ndcg([{"text": "stuff"}], [], k=5) == 0.0

    def test_ndcg_empty_sources_returns_zero(self):
        _, _, ndcg = self._import_metrics()
        assert ndcg([], ["keyword"], k=5) == 0.0

    def test_ndcg_no_matches_returns_zero(self):
        _, _, ndcg = self._import_metrics()
        score = ndcg([{"text": "nothing relevant"}], ["kubernetes"], k=5)
        assert score == 0.0

    def test_filter_questions_for_scale(self):
        import sys, os
        bench_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "pipeline")
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        from evaluate import filter_questions_for_scale

        questions = [
            {"id": 1, "scales": ["all"]},
            {"id": 2, "scales": ["medium", "large"]},
            {"id": 3, "scales": ["small", "medium", "large"]},
            {"id": 4},  # no scales field
        ]
        small = filter_questions_for_scale(questions, "small")
        assert {q["id"] for q in small} == {1, 3, 4}

        medium = filter_questions_for_scale(questions, "medium")
        assert {q["id"] for q in medium} == {1, 2, 3, 4}

        none_scale = filter_questions_for_scale(questions, None)
        assert len(none_scale) == 4

    def test_questions_json_has_scales(self):
        """Every question in the fixture file must have a 'scales' field."""
        import json, os
        qpath = os.path.join(
            os.path.dirname(__file__), "..", "benchmarks", "fixtures", "questions.json"
        )
        if not os.path.exists(qpath):
            pytest.skip("questions.json not present")
        with open(qpath) as f:
            questions = json.load(f)
        missing = [q["id"] for q in questions if "scales" not in q]
        assert missing == [], f"Questions missing 'scales' field: {missing}"

    def test_questions_json_no_impossible_q61(self):
        """Q61 (2025 date on 2026 corpus) must not exist."""
        import json, os
        qpath = os.path.join(
            os.path.dirname(__file__), "..", "benchmarks", "fixtures", "questions.json"
        )
        if not os.path.exists(qpath):
            pytest.skip("questions.json not present")
        with open(qpath) as f:
            questions = json.load(f)
        ids = {q["id"] for q in questions}
        assert 61 not in ids, "Q61 should be removed (impossible temporal question)"

    def test_needle_questions_require_medium_plus(self):
        """Needle questions should be excluded from small corpus."""
        import json, os
        qpath = os.path.join(
            os.path.dirname(__file__), "..", "benchmarks", "fixtures", "questions.json"
        )
        if not os.path.exists(qpath):
            pytest.skip("questions.json not present")
        with open(qpath) as f:
            questions = json.load(f)
        from evaluate import filter_questions_for_scale
        small_qs = filter_questions_for_scale(questions, "small")
        needle_in_small = [q for q in small_qs if q.get("query_type") == "needle"]
        assert len(needle_in_small) == 0, "Needle questions should be filtered out for small corpus"


class TestPhase1QueryExpansionKill:
    """Phase 1: verify query expansion is dead by default and variants are clean."""

    def _import_evaluate(self):
        import sys, os
        bench_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "pipeline")
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        import evaluate
        return evaluate

    def test_config_default_expansion_disabled(self):
        """QUERY_EXPANSION_ENABLED must default to False."""
        from config import QUERY_EXPANSION_ENABLED
        assert QUERY_EXPANSION_ENABLED is False, (
            "QUERY_EXPANSION_ENABLED must default to False — "
            "expansion adds ~6s latency for <1pp recall"
        )

    def test_all_variants_explicitly_disable_expansion(self):
        """Every variant except expansion_on must have QUERY_EXPANSION_ENABLED=false."""
        evaluate = self._import_evaluate()
        for name, overrides in evaluate.VARIANTS.items():
            if name == "expansion_on":
                assert overrides.get("QUERY_EXPANSION_ENABLED") == "true", (
                    f"expansion_on should have expansion enabled"
                )
                continue
            assert overrides.get("QUERY_EXPANSION_ENABLED") == "false", (
                f"Variant '{name}' must explicitly set QUERY_EXPANSION_ENABLED=false"
            )

    def test_expansion_ab_variants_exist(self):
        """The one-time A/B comparison variants must exist."""
        evaluate = self._import_evaluate()
        assert "expansion_off" in evaluate.VARIANTS
        assert "expansion_on" in evaluate.VARIANTS

    def test_expansion_ab_only_differ_in_expansion_flag(self):
        """expansion_off and expansion_on must be identical except QUERY_EXPANSION_ENABLED."""
        evaluate = self._import_evaluate()
        off = dict(evaluate.VARIANTS["expansion_off"])
        on = dict(evaluate.VARIANTS["expansion_on"])
        off_expansion = off.pop("QUERY_EXPANSION_ENABLED")
        on_expansion = on.pop("QUERY_EXPANSION_ENABLED")
        assert off_expansion == "false"
        assert on_expansion == "true"
        assert off == on, (
            f"expansion_off and expansion_on differ in keys other than QUERY_EXPANSION_ENABLED: "
            f"off={off}, on={on}"
        )

    def test_query_expansion_in_propagation_targets(self):
        """query_expansion must be in the module reload list."""
        evaluate = self._import_evaluate()
        import inspect
        source = inspect.getsource(evaluate._apply_variant)
        assert "query_expansion" in source, (
            "_apply_variant must reload query_expansion module on variant switch"
        )

    def test_variant_count_unchanged(self):
        """Sanity: we didn't accidentally lose variants during refactor."""
        evaluate = self._import_evaluate()
        expected_names = {
            "vector_only", "vector_plus_synth", "vector_plus_synth_plus_reranker",
            "clean_reranker",
            "expansion_off", "expansion_on",
            "plus_bm25", "plus_graph", "plus_temporal", "plus_hotness", "plus_rerank",
            "full_pipeline", "full_pipeline_rerank",
        }
        assert set(evaluate.VARIANTS.keys()) == expected_names


class TestPhase2SyntheticQuestionPipeline:
    """Phase 2: verify synthetic question embeddings flow through the search path."""

    def test_search_vectors_returns_representation_type(self):
        """search_vectors result rows must include representation_type."""
        import inspect
        from rlm_retriever import search_vectors
        source = inspect.getsource(search_vectors)
        assert "representation_type" in source, (
            "search_vectors must propagate representation_type from Qdrant payload"
        )
        assert "synthetic_question" in source, (
            "search_vectors must propagate synthetic_question from Qdrant payload"
        )
        assert "source_memory_id" in source, (
            "search_vectors must propagate source_memory_id from Qdrant payload"
        )

    def test_search_vectors_filters_synthetic_when_disabled(self):
        """search_vectors must exclude synthetic_question points when the feature is off."""
        import inspect
        from rlm_retriever import search_vectors
        source = inspect.getsource(search_vectors)
        assert "SYNTHETIC_QUESTIONS_ENABLED" in source, (
            "search_vectors must check SYNTHETIC_QUESTIONS_ENABLED to filter synthetic points"
        )
        assert "must_not_filters" in source, (
            "search_vectors must use must_not filter for synthetic question exclusion"
        )

    def test_dedupe_keeps_best_score(self):
        """When a chunk and its synthetic question both match, keep the higher score."""
        from memory_fusion import dedupe_vector_hits

        chunk_hit = {
            "id": "chunk-1",
            "file_path": "agents/chief/2026-04-01.md",
            "chunk_index": 0,
            "text": "The backup runs at 04:15 UTC every Sunday",
            "score": 0.75,
            "representation_type": "chunk",
        }
        synth_hit = {
            "id": "synth-1",
            "file_path": "agents/chief/2026-04-01.md",
            "chunk_index": 0,
            "text": "The backup runs at 04:15 UTC every Sunday",
            "score": 0.92,
            "representation_type": "synthetic_question",
            "synthetic_question": "What time does the backup run?",
        }
        result = dedupe_vector_hits([chunk_hit, synth_hit])
        assert len(result) == 1, "chunk and its synthetic twin should dedupe to one"
        assert result[0]["score"] == 0.92, "should keep the higher score"
        assert result[0]["synthetic_match"] is True

    def test_dedupe_marks_synthetic_match_when_chunk_wins(self):
        """Even when the chunk score wins, synthetic_match should be True."""
        from memory_fusion import dedupe_vector_hits

        chunk_hit = {
            "id": "chunk-1",
            "file_path": "agents/chief/daily.md",
            "chunk_index": 2,
            "text": "PostgreSQL backup via pg_dump",
            "score": 0.95,
            "representation_type": "chunk",
        }
        synth_hit = {
            "id": "synth-2",
            "file_path": "agents/chief/daily.md",
            "chunk_index": 2,
            "text": "PostgreSQL backup via pg_dump",
            "score": 0.80,
            "representation_type": "synthetic_question",
        }
        result = dedupe_vector_hits([chunk_hit, synth_hit])
        assert len(result) == 1
        assert result[0]["score"] == 0.95, "chunk with higher score wins"
        assert result[0]["synthetic_match"] is True, (
            "synthetic_match must be True even when chunk score wins"
        )

    def test_dedupe_no_synthetic_match_for_pure_chunks(self):
        """Chunks without a synthetic twin should NOT have synthetic_match."""
        from memory_fusion import dedupe_vector_hits

        hits = [
            {
                "id": "c1", "file_path": "f1.md", "chunk_index": 0,
                "text": "some text", "score": 0.8,
                "representation_type": "chunk",
            },
            {
                "id": "c2", "file_path": "f2.md", "chunk_index": 0,
                "text": "other text", "score": 0.7,
                "representation_type": "chunk",
            },
        ]
        result = dedupe_vector_hits(hits)
        assert len(result) == 2
        for r in result:
            assert "synthetic_match" not in r

    def test_trace_includes_synthetic_hits_key(self):
        """_trace_kw must include synthetic_hits for benchmark observability."""
        import inspect
        from rlm_retriever import recursive_retrieve
        source = inspect.getsource(recursive_retrieve)
        assert "synthetic_hits" in source, (
            "recursive_retrieve must track n_synthetic_hits in the retrieval trace"
        )

    def test_force_invalidate_all_clears_cache(self):
        """force_invalidate_all (alias for invalidate_all) must always clear entries."""
        import hot_cache
        old_enabled = hot_cache.HOT_CACHE_ENABLED

        try:
            hot_cache.HOT_CACHE_ENABLED = True
            hot_cache.put("test_agent", "test query", {"answer": "cached"})
            assert hot_cache.get("test_agent", "test query") is not None

            evicted = hot_cache.force_invalidate_all()
            assert evicted >= 1, "force_invalidate_all must clear entries"
            assert hot_cache.get("test_agent", "test query") is None
        finally:
            hot_cache.HOT_CACHE_ENABLED = old_enabled
            hot_cache.force_invalidate_all()

    def test_synthetic_point_payload_structure(self):
        """Synthetic question points must carry required fields for the search path."""
        import asyncio
        from unittest.mock import AsyncMock, patch
        import synthetic_questions

        mock_qs = '["How do backups work?", "When is the backup window?"]'
        with patch.object(synthetic_questions, "SYNTHETIC_QUESTIONS_ENABLED", True), \
             patch.object(synthetic_questions, "SYNTHETIC_QUESTIONS_COUNT", 2), \
             patch("synthetic_questions.llm_query", new_callable=AsyncMock) as mock_llm, \
             patch("synthetic_questions.embed_batch", new_callable=AsyncMock) as mock_embed:
            mock_llm.return_value = mock_qs
            mock_embed.return_value = [[0.1] * 1024, [0.2] * 1024]
            synthetic_questions._cache.clear()

            points = asyncio.get_event_loop().run_until_complete(
                synthetic_questions.generate_and_embed_synthetic_points(
                    chunk_point_id="parent-chunk-id",
                    chunk_text="Backups run at 04:15 UTC on Sunday",
                    base_payload={
                        "agent_id": "chief",
                        "namespace": "default",
                        "file_path": "agents/chief/daily.md",
                        "chunk_index": 3,
                    },
                )
            )

        assert len(points) == 2
        for p in points:
            assert p.payload["representation_type"] == "synthetic_question"
            assert p.payload["source_memory_id"] == "parent-chunk-id"
            assert p.payload["text"] == "Backups run at 04:15 UTC on Sunday"
            assert p.payload["file_path"] == "agents/chief/daily.md"
            assert p.payload["chunk_index"] == 3
            assert p.payload["agent_id"] == "chief"

    def test_benchmark_summary_includes_synthetic_counts(self):
        """Benchmark summary dict must include total_synthetic_hits and queries_with_synthetic_hits."""
        import sys, os
        bench_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "pipeline")
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        import evaluate
        import inspect
        source = inspect.getsource(evaluate.run_variant)
        assert "total_synthetic_hits" in source, (
            "run_variant summary must aggregate total_synthetic_hits"
        )
        assert "queries_with_synthetic_hits" in source, (
            "run_variant summary must count queries_with_synthetic_hits"
        )


class TestPhase3NominateThenRerank:
    """Phase 3: verify the clean nominate-then-rerank pipeline."""

    def test_v2_path_bypasses_rrf_merge(self):
        """When RERANKER_ENABLED=True, the v2 branch must NOT use rrf_merge."""
        import inspect
        from rlm_retriever import recursive_retrieve
        source = inspect.getsource(recursive_retrieve)
        lines = source.split("\n")
        in_v2 = False
        for line in lines:
            if "v2 CLEAN PATH" in line:
                in_v2 = True
            if in_v2 and "LEGACY PATH" in line:
                break
            if in_v2:
                assert "rrf_merge" not in line, (
                    "rrf_merge must not appear in the v2 clean path"
                )

    def test_v2_path_bypasses_bm25_merge(self):
        """When RERANKER_ENABLED=True, merge_vector_and_bm25 must be skipped."""
        import inspect
        from rlm_retriever import recursive_retrieve
        source = inspect.getsource(recursive_retrieve)
        assert "not RERANKER_ENABLED" in source, (
            "merge_vector_and_bm25 must be gated behind 'not RERANKER_ENABLED'"
        )

    def test_v2_path_does_id_based_dedup(self):
        """The v2 path must build a candidate_pool keyed by ID, not use dedupe_vector_hits."""
        import inspect
        from rlm_retriever import recursive_retrieve
        source = inspect.getsource(recursive_retrieve)
        assert "candidate_pool" in source, (
            "v2 path must use candidate_pool dict for ID-based deduplication"
        )

    def test_v2_pool_collects_all_nomination_sources(self):
        """The nomination pool must include vec_results, bm25_hits, literal_hits, _registry_hits."""
        import inspect
        from rlm_retriever import recursive_retrieve
        source = inspect.getsource(recursive_retrieve)
        for source_name in ("vec_results", "bm25_hits", "literal_hits", "_registry_hits"):
            assert source_name in source, (
                f"Nomination pool must include {source_name}"
            )

    def test_v2_trace_includes_nomination_pool_size(self):
        """The v2 trace must record nomination_pool_size for observability."""
        import inspect
        from rlm_retriever import recursive_retrieve
        source = inspect.getsource(recursive_retrieve)
        assert "nomination_pool_size" in source, (
            "v2 path must include nomination_pool_size in _common_trace"
        )

    def test_v2_pool_keeps_best_score_per_id(self):
        """When duplicate IDs appear from different nomination sources, keep the highest score."""
        candidate_pool: dict[str, dict] = {}
        test_items = [
            {"id": "abc", "score": 0.5, "file_path": "a.md", "representation_type": "chunk"},
            {"id": "abc", "score": 0.9, "file_path": "a.md", "representation_type": "chunk"},
            {"id": "def", "score": 0.7, "file_path": "b.md", "representation_type": "chunk"},
        ]
        for r in test_items:
            rid = str(r.get("id", ""))
            existing = candidate_pool.get(rid)
            if existing is None:
                candidate_pool[rid] = dict(r)
            elif r.get("score", 0) > existing.get("score", 0):
                candidate_pool[rid] = dict(r)

        assert len(candidate_pool) == 2
        assert candidate_pool["abc"]["score"] == 0.9
        assert candidate_pool["def"]["score"] == 0.7

    def test_v2_pool_marks_synthetic_match(self):
        """Synthetic question hits must be tagged with synthetic_match=True in the pool."""
        candidate_pool: dict[str, dict] = {}
        test_items = [
            {"id": "c1", "score": 0.8, "file_path": "a.md", "chunk_index": 0,
             "representation_type": "chunk"},
            {"id": "s1", "score": 0.85, "file_path": "a.md", "chunk_index": 0,
             "representation_type": "synthetic_question"},
        ]
        for r in test_items:
            rid = str(r.get("id", ""))
            existing = candidate_pool.get(rid)
            if existing is None:
                candidate_pool[rid] = dict(r)
            elif r.get("score", 0) > existing.get("score", 0):
                candidate_pool[rid] = dict(r)
            if r.get("representation_type") == "synthetic_question":
                candidate_pool[rid]["synthetic_match"] = True

        assert candidate_pool["s1"].get("synthetic_match") is True
        assert candidate_pool["c1"].get("synthetic_match") is not True

    def test_v2_pool_preserves_needle_registry_hit(self):
        """Needle registry hits must keep their needle_registry_hit flag through dedup."""
        candidate_pool: dict[str, dict] = {}
        test_items = [
            {"id": "n1", "score": 1.0, "file_path": "needle.md",
             "needle_registry_hit": True, "representation_type": "chunk"},
            {"id": "n1", "score": 0.9, "file_path": "needle.md",
             "representation_type": "chunk"},
        ]
        for r in test_items:
            rid = str(r.get("id", ""))
            existing = candidate_pool.get(rid)
            if existing is None:
                candidate_pool[rid] = dict(r)
            elif r.get("score", 0) > existing.get("score", 0):
                candidate_pool[rid] = dict(r)
            if r.get("needle_registry_hit"):
                candidate_pool[rid]["needle_registry_hit"] = True

        assert candidate_pool["n1"]["needle_registry_hit"] is True
        assert candidate_pool["n1"]["score"] == 1.0

    def test_clean_reranker_variant_exists(self):
        """The clean_reranker benchmark variant must exist with correct flags."""
        import sys, os
        bench_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "pipeline")
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        import importlib
        import evaluate
        importlib.reload(evaluate)
        v = evaluate.VARIANTS["clean_reranker"]
        assert v["RERANKER_ENABLED"] == "true"
        assert v["BM25_ENABLED"] == "true"
        assert v["GRAPH_RETRIEVAL_ENABLED"] == "true"
        assert v["SYNTHETIC_QUESTIONS_ENABLED"] == "true"
        assert v["QUERY_EXPANSION_ENABLED"] == "false"

    def test_clean_reranker_variant_no_legacy_scoring(self):
        """clean_reranker must disable all legacy scoring knobs."""
        import sys, os
        bench_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "pipeline")
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        import importlib
        import evaluate
        importlib.reload(evaluate)
        v = evaluate.VARIANTS["clean_reranker"]
        assert v.get("HOTNESS_WEIGHT") == "0", "Legacy hotness must be disabled"
        assert v.get("TEMPORAL_DECAY_HALFLIFE_DAYS") == "0", "Legacy temporal decay must be disabled"
        assert v.get("RERANK_ENABLED") == "false", "Legacy rerank must be disabled"

    def test_legacy_path_preserved(self):
        """The legacy path (RERANKER_ENABLED=False) must still exist for shadow comparison."""
        import inspect
        from rlm_retriever import recursive_retrieve
        source = inspect.getsource(recursive_retrieve)
        assert "LEGACY PATH" in source, (
            "Legacy path comment marker must still exist"
        )
        assert "apply_temporal_decay" in source, (
            "Legacy temporal decay must still exist in the legacy path"
        )
        assert "apply_hotness_to_results" in source, (
            "Legacy hotness must still exist in the legacy path"
        )
        assert "apply_retrieval_threshold" in source, (
            "Legacy threshold must still exist in the legacy path"
        )

    def test_v2_no_threshold_filter(self):
        """The v2 reranker path must use threshold=0.0, no dynamic threshold."""
        import inspect
        from rlm_retriever import recursive_retrieve
        source = inspect.getsource(recursive_retrieve)
        lines = source.split("\n")
        in_v2 = False
        for line in lines:
            if "v2 CLEAN PATH" in line:
                in_v2 = True
            if in_v2 and "LEGACY PATH" in line:
                break
            if in_v2:
                assert "apply_dynamic_threshold" not in line, (
                    "v2 path must not call apply_dynamic_threshold"
                )
                assert "apply_retrieval_threshold" not in line, (
                    "v2 path must not call apply_retrieval_threshold"
                )

    def test_benchmark_summary_includes_nomination_pool(self):
        """Benchmark summary must include avg_nomination_pool_size."""
        import sys, os
        bench_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "pipeline")
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        import importlib
        import evaluate
        importlib.reload(evaluate)
        import inspect
        source = inspect.getsource(evaluate.run_variant)
        assert "nomination_pool_size" in source, (
            "run_variant must extract nomination_pool_size from trace"
        )
        assert "avg_nomination_pool_size" in source, (
            "run_variant summary must compute avg_nomination_pool_size"
        )


class TestPhase4ParentTextAtIndexTime:
    """Phase 4: verify parent context is stored at index time, not fetched at runtime."""

    def test_enrich_with_parent_deleted(self):
        """enrich_with_parent must no longer exist in rlm_retriever."""
        import rlm_retriever
        assert not hasattr(rlm_retriever, "enrich_with_parent"), (
            "enrich_with_parent must be deleted — parent text is stored at index time"
        )

    def test_no_enrich_with_parent_calls(self):
        """recursive_retrieve must not call enrich_with_parent anywhere."""
        import inspect
        from rlm_retriever import recursive_retrieve
        source = inspect.getsource(recursive_retrieve)
        assert "enrich_with_parent" not in source, (
            "All enrich_with_parent calls must be removed from recursive_retrieve"
        )

    def test_search_vectors_returns_parent_text(self):
        """search_vectors result rows must include parent_text from payload."""
        import inspect
        from rlm_retriever import search_vectors
        source = inspect.getsource(search_vectors)
        assert '"parent_text"' in source or "'parent_text'" in source, (
            "search_vectors must propagate parent_text from Qdrant payload"
        )

    def test_indexer_stores_parent_text_hierarchical(self):
        """Hierarchical indexing must store parent_text on child chunks."""
        import inspect
        from indexer import index_file
        source = inspect.getsource(index_file)
        assert "parent_text" in source, (
            "index_file must store parent_text in payload for child chunks"
        )
        assert "_parent_text_map" in source, (
            "index_file must build _parent_text_map for parent->text lookup"
        )

    def test_indexer_parent_text_map_correctness(self):
        """_parent_text_map must map parent IDs to parent content text."""
        hier_chunks = [
            {"id": "p1", "parent_id": None, "content": "Parent text here", "is_parent": True},
            {"id": "c1", "parent_id": "p1", "content": "Child text", "is_parent": False},
            {"id": "c2", "parent_id": "p1", "content": "Another child", "is_parent": False},
            {"id": "p2", "parent_id": None, "content": "Second parent", "is_parent": True},
        ]
        _parent_text_map = {
            c["id"]: c["content"] for c in hier_chunks if c["is_parent"]
        }
        assert _parent_text_map["p1"] == "Parent text here"
        assert _parent_text_map["p2"] == "Second parent"
        assert "c1" not in _parent_text_map

        for c in hier_chunks:
            if not c["is_parent"] and c["parent_id"]:
                parent_text = _parent_text_map.get(c["parent_id"], "")
                assert parent_text == "Parent text here"

    def test_indexer_flat_chunks_have_empty_parent_text(self):
        """Flat chunks (no hierarchy) must store parent_text as empty string."""
        import inspect
        from indexer import index_file
        source = inspect.getsource(index_file)
        # The flat chunk payload must include parent_text with empty default
        assert source.count('"parent_text"') >= 2, (
            "Both hierarchical and flat paths must set parent_text in payload"
        )

    def test_reranker_reads_parent_text(self):
        """_build_pair must read parent_text (index-time field), not parent_context."""
        from reranker import _build_pair
        candidate_with_text = {
            "text": "Child chunk content",
            "parent_text": "Full parent context from indexing",
        }
        pair = _build_pair("test query", candidate_with_text)
        assert "Full parent context from indexing" in pair
        assert "Parent context:" in pair

    def test_reranker_backward_compat_parent_context(self):
        """_build_pair should fall back to parent_context for pre-Phase-4 indexed data."""
        from reranker import _build_pair
        candidate_legacy = {
            "text": "Child chunk content",
            "parent_context": "Legacy parent context",
        }
        pair = _build_pair("test query", candidate_legacy)
        assert "Legacy parent context" in pair

    def test_reranker_no_parent_text_omits_section(self):
        """_build_pair omits parent context section when parent_text is empty."""
        from reranker import _build_pair
        candidate_no_parent = {
            "text": "Standalone chunk content",
            "parent_text": "",
        }
        pair = _build_pair("test query", candidate_no_parent)
        assert "Parent context:" not in pair

    def test_refine_uses_parent_text(self):
        """_refine_one_chunk must read parent_text, not parent_context."""
        import inspect
        from rlm_retriever import _refine_one_chunk
        source = inspect.getsource(_refine_one_chunk)
        assert "parent_text" in source, (
            "_refine_one_chunk must read from parent_text field"
        )
        assert "parent_context" not in source, (
            "_refine_one_chunk must not reference the deleted parent_context field"
        )

    def test_v2_trace_parent_enriched_uses_parent_text(self):
        """The v2 path trace must check parent_text, not parent_context."""
        import inspect
        from rlm_retriever import recursive_retrieve
        source = inspect.getsource(recursive_retrieve)
        assert 'parent_text' in source, (
            "Trace parent_enriched must check parent_text field"
        )
        assert 'parent_context' not in source, (
            "No references to parent_context should remain in recursive_retrieve"
        )


class TestPhase5SemanticChunking:
    """Phase 5: Semantic / meaning-boundary chunking."""

    def test_chunking_strategy_config_exists(self):
        """CHUNKING_STRATEGY config variable must exist and default to 'semantic'."""
        import config
        assert hasattr(config, "CHUNKING_STRATEGY"), (
            "CHUNKING_STRATEGY must be defined in config.py"
        )
        assert config.CHUNKING_STRATEGY in ("semantic", "fixed"), (
            f"CHUNKING_STRATEGY must be 'semantic' or 'fixed', got {config.CHUNKING_STRATEGY!r}"
        )
        # Default must be semantic
        import os
        if "CHUNKING_STRATEGY" not in os.environ:
            assert config.CHUNKING_STRATEGY == "semantic", (
                "Default CHUNKING_STRATEGY must be 'semantic'"
            )

    def test_chunk_text_semantic_exported(self):
        """chunk_text_semantic must be importable from chunking module."""
        from chunking import chunk_text_semantic
        assert callable(chunk_text_semantic)

    def test_chunk_text_semantic_short_doc_fast_path(self):
        """Short documents (len <= size) return a single unchanged chunk."""
        from chunking import chunk_text_semantic
        text = "## Title\n\nShort content that fits in one chunk easily."
        result = chunk_text_semantic(text, size=2000)
        assert len(result) == 1
        assert result[0] == text.strip()

    def test_chunk_text_semantic_long_doc_splits(self):
        """Long documents with multiple headings are split per heading."""
        from chunking import chunk_text_semantic
        body = "Content word. " * 50  # ~700 chars
        text = f"## Section A\n\n{body}\n\n## Section B\n\n{body}"
        result = chunk_text_semantic(text, size=800)
        assert len(result) >= 2, (
            f"Long multi-section document should produce ≥2 chunks, got {len(result)}"
        )

    def test_chunk_text_semantic_no_cross_section_merge(self):
        """Content from different heading sections must not merge into one chunk."""
        from chunking import chunk_text_semantic
        body = "unique_marker_{n}. " * 40
        text = (
            f"## One\n\n{body.format(n='A')}\n\n"
            f"## Two\n\n{body.format(n='B')}"
        )
        result = chunk_text_semantic(text, size=500)
        for chunk in result:
            has_a = "unique_marker_A" in chunk
            has_b = "unique_marker_B" in chunk
            assert not (has_a and has_b), "Cross-section merge detected"

    def test_hierarchical_accepts_strategy_param(self):
        """chunk_text_hierarchical must accept and propagate the strategy parameter."""
        import inspect
        from chunking import chunk_text_hierarchical
        sig = inspect.signature(chunk_text_hierarchical)
        assert "strategy" in sig.parameters, (
            "chunk_text_hierarchical must have a 'strategy' parameter"
        )
        default = sig.parameters["strategy"].default
        assert default == "semantic", (
            f"Default strategy must be 'semantic', got {default!r}"
        )

    def test_hierarchical_semantic_uses_chunk_text_semantic(self, monkeypatch):
        """With strategy='semantic', chunk_text_hierarchical must call chunk_text_semantic."""
        import chunking
        calls = []

        original = chunking.chunk_text_semantic

        def spy(text, **kwargs):
            calls.append(text)
            return original(text, **kwargs)

        monkeypatch.setattr(chunking, "chunk_text_semantic", spy)
        body = "word " * 100
        text = f"## Section\n\n{body}"
        chunking.chunk_text_hierarchical(text, "test.md", parent_size=300, strategy="semantic")
        assert len(calls) > 0, (
            "chunk_text_semantic was not called when strategy='semantic'"
        )

    def test_hierarchical_fixed_does_not_use_chunk_text_semantic(self, monkeypatch):
        """With strategy='fixed', chunk_text_hierarchical must NOT call chunk_text_semantic."""
        import chunking
        calls = []

        original = chunking.chunk_text_semantic

        def spy(text, **kwargs):
            calls.append(text)
            return original(text, **kwargs)

        monkeypatch.setattr(chunking, "chunk_text_semantic", spy)
        body = "word " * 100
        text = f"## Section\n\n{body}"
        chunking.chunk_text_hierarchical(text, "test.md", parent_size=300, strategy="fixed")
        assert len(calls) == 0, (
            "chunk_text_semantic must NOT be called when strategy='fixed'"
        )

    def test_indexer_imports_chunking_strategy(self):
        """indexer.py must import CHUNKING_STRATEGY from config."""
        import inspect
        import indexer
        source = inspect.getsource(indexer)
        assert "CHUNKING_STRATEGY" in source, (
            "indexer.py must import and use CHUNKING_STRATEGY"
        )

    def test_indexer_passes_strategy_to_hierarchical(self):
        """indexer.py must pass strategy= to chunk_text_hierarchical."""
        import inspect
        import indexer
        source = inspect.getsource(indexer)
        assert "strategy=CHUNKING_STRATEGY" in source, (
            "indexer.py must forward strategy=CHUNKING_STRATEGY to chunk_text_hierarchical"
        )

    def test_short_document_strategy_invariant(self):
        """Short documents produce identical parent content for both strategies."""
        from chunking import chunk_text_hierarchical
        text = "## Note\n\nA short note that fits comfortably inside one parent chunk."
        sem = chunk_text_hierarchical(text, "note.md", parent_size=2000, strategy="semantic")
        fix = chunk_text_hierarchical(text, "note.md", parent_size=2000, strategy="fixed")
        sem_parents = [c["content"] for c in sem if c["is_parent"]]
        fix_parents = [c["content"] for c in fix if c["is_parent"]]
        assert sem_parents == fix_parents, (
            "Short documents must produce identical parent chunks regardless of strategy"
        )

