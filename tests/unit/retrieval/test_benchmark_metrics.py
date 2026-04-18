"""Unit tests for benchmark evaluation metrics and fixture integrity."""

import json
import os
import sys

import pytest

pytestmark = [pytest.mark.unit]

_BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks", "pipeline")


def _import_metrics():
    if _BENCH_DIR not in sys.path:
        sys.path.insert(0, _BENCH_DIR)
    from evaluate import _extract_source_text, _keyword_recall_at_k, _ndcg_at_k

    return _extract_source_text, _keyword_recall_at_k, _ndcg_at_k


class TestBenchmarkMetrics:
    """Tests for evaluation metric functions in benchmarks/pipeline/evaluate.py."""

    def test_extract_source_text_prefers_tier_text(self):
        extract, _, _ = _import_metrics()
        src = {"tier_text": "tier content", "text": "raw content", "score": 0.9}
        assert extract(src) == "tier content"

    def test_extract_source_text_falls_back_to_text(self):
        extract, _, _ = _import_metrics()
        src = {"text": "raw content", "score": 0.9}
        assert extract(src) == "raw content"

    def test_extract_source_text_returns_empty_for_metadata_only(self):
        extract, _, _ = _import_metrics()
        src = {"file": "path.md", "score": 0.9, "agent": "chief"}
        assert extract(src) == ""

    def test_extract_source_text_never_returns_str_dict(self):
        extract, _, _ = _import_metrics()
        src = {"file_path": "pipeline.md", "score": 0.9, "agent_id": "gitbob"}
        result = extract(src)
        assert "file_path" not in result
        assert "score" not in result
        assert "agent_id" not in result

    def test_keyword_recall_at_k_respects_k(self):
        _, recall_at_k, _ = _import_metrics()
        sources = [
            {"text": "kubernetes cluster setup"},
            {"text": "postgres database migration"},
            {"text": "ci/cd pipeline gitlab"},
        ]
        r1 = recall_at_k("", sources, ["kubernetes", "postgres", "gitlab"], k=1)
        r3 = recall_at_k("", sources, ["kubernetes", "postgres", "gitlab"], k=3)
        assert r1 < r3
        assert abs(r1 - 1 / 3) < 0.01
        assert abs(r3 - 1.0) < 0.01

    def test_keyword_recall_at_5_differs_from_at_10(self):
        _, recall_at_k, _ = _import_metrics()
        sources = [{"text": f"filler text {i}"} for i in range(10)]
        sources[7]["text"] = "kubernetes cluster k8s"
        r5 = recall_at_k("", sources, ["kubernetes"], k=5)
        r10 = recall_at_k("", sources, ["kubernetes"], k=10)
        assert r5 == 0.0
        assert r10 == 1.0

    def test_keyword_recall_includes_answer_text(self):
        _, recall_at_k, _ = _import_metrics()
        sources = [{"text": "unrelated stuff"}]
        r = recall_at_k("kubernetes cluster", sources, ["kubernetes"], k=5)
        assert r == 1.0

    def test_keyword_recall_empty_keywords(self):
        _, recall_at_k, _ = _import_metrics()
        assert recall_at_k("text", [{"text": "t"}], [], k=5) == 0.0

    def test_keyword_recall_does_not_match_metadata(self):
        _, recall_at_k, _ = _import_metrics()
        src = {"text": "unrelated content", "file_path": "/agents/gitbob/pipeline.md", "score": 0.9}
        r = recall_at_k("", [src], ["pipeline"], k=5)
        assert r == 0.0

    def test_ndcg_perfect_ordering(self):
        _, _, ndcg = _import_metrics()
        sources = [
            {"text": "kubernetes k8s cluster"},
            {"text": "unrelated text"},
        ]
        score = ndcg(sources, ["kubernetes", "k8s", "cluster"], k=5)
        assert score == 1.0

    def test_ndcg_reversed_ordering(self):
        _, _, ndcg = _import_metrics()
        sources = [
            {"text": "unrelated text"},
            {"text": "kubernetes k8s cluster"},
        ]
        score = ndcg(sources, ["kubernetes", "k8s", "cluster"], k=5)
        assert 0.0 < score < 1.0

    def test_ndcg_no_keywords_returns_zero(self):
        _, _, ndcg = _import_metrics()
        assert ndcg([{"text": "stuff"}], [], k=5) == 0.0

    def test_ndcg_empty_sources_returns_zero(self):
        _, _, ndcg = _import_metrics()
        assert ndcg([], ["keyword"], k=5) == 0.0

    def test_ndcg_no_matches_returns_zero(self):
        _, _, ndcg = _import_metrics()
        score = ndcg([{"text": "nothing relevant"}], ["kubernetes"], k=5)
        assert score == 0.0

    def test_filter_questions_for_scale(self):
        if _BENCH_DIR not in sys.path:
            sys.path.insert(0, _BENCH_DIR)
        from evaluate import filter_questions_for_scale

        questions = [
            {"id": 1, "scales": ["all"]},
            {"id": 2, "scales": ["medium", "large"]},
            {"id": 3, "scales": ["small", "medium", "large"]},
            {"id": 4},
        ]
        small = filter_questions_for_scale(questions, "small")
        assert {q["id"] for q in small} == {1, 3, 4}

        medium = filter_questions_for_scale(questions, "medium")
        assert {q["id"] for q in medium} == {1, 2, 3, 4}

        none_scale = filter_questions_for_scale(questions, None)
        assert len(none_scale) == 4

    def test_questions_json_has_scales(self):
        qpath = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "benchmarks", "fixtures", "questions.json"
        )
        if not os.path.exists(qpath):
            pytest.skip("questions.json not present")
        with open(qpath) as f:
            questions = json.load(f)
        missing = [q["id"] for q in questions if "scales" not in q]
        assert missing == []

    def test_questions_json_no_impossible_q61(self):
        qpath = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "benchmarks", "fixtures", "questions.json"
        )
        if not os.path.exists(qpath):
            pytest.skip("questions.json not present")
        with open(qpath) as f:
            questions = json.load(f)
        ids = {q["id"] for q in questions}
        assert 61 not in ids

    def test_needle_questions_require_medium_plus(self):
        if _BENCH_DIR not in sys.path:
            sys.path.insert(0, _BENCH_DIR)
        qpath = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "benchmarks", "fixtures", "questions.json"
        )
        if not os.path.exists(qpath):
            pytest.skip("questions.json not present")
        with open(qpath) as f:
            questions = json.load(f)
        from evaluate import filter_questions_for_scale

        small_qs = filter_questions_for_scale(questions, "small")
        needle_in_small = [q for q in small_qs if q.get("query_type") == "needle"]
        assert len(needle_in_small) == 0


class TestPhase1QueryExpansionKill:
    """Phase 1: verify query expansion is dead by default and variants are clean."""

    def _import_evaluate(self):
        import importlib

        if _BENCH_DIR not in sys.path:
            sys.path.insert(0, _BENCH_DIR)
        import evaluate

        return evaluate

    def test_config_default_expansion_disabled(self):
        from config import QUERY_EXPANSION_ENABLED

        assert QUERY_EXPANSION_ENABLED is False

    def test_all_variants_explicitly_disable_expansion(self):
        import inspect

        evaluate = self._import_evaluate()
        for name, overrides in evaluate.VARIANTS.items():
            if name == "expansion_on":
                assert overrides.get("QUERY_EXPANSION_ENABLED") == "true"
                continue
            assert overrides.get("QUERY_EXPANSION_ENABLED") == "false"

    def test_expansion_ab_variants_exist(self):
        evaluate = self._import_evaluate()
        assert "expansion_off" in evaluate.VARIANTS
        assert "expansion_on" in evaluate.VARIANTS

    def test_expansion_ab_only_differ_in_expansion_flag(self):
        evaluate = self._import_evaluate()
        off = dict(evaluate.VARIANTS["expansion_off"])
        on = dict(evaluate.VARIANTS["expansion_on"])
        off_expansion = off.pop("QUERY_EXPANSION_ENABLED")
        on_expansion = on.pop("QUERY_EXPANSION_ENABLED")
        assert off_expansion == "false"
        assert on_expansion == "true"
        assert off == on

    def test_query_expansion_in_propagation_targets(self):
        import inspect

        evaluate = self._import_evaluate()
        source = inspect.getsource(evaluate._apply_variant)
        assert "query_expansion" in source

    def test_variant_count_unchanged(self):
        evaluate = self._import_evaluate()
        expected_names = {
            "vector_only",
            "vector_plus_synth",
            "vector_plus_synth_plus_reranker",
            "clean_reranker",
            "expansion_off",
            "expansion_on",
            "plus_bm25",
            "plus_graph",
            "plus_temporal",
            "plus_hotness",
            "plus_rerank",
            "full_pipeline",
            "full_pipeline_rerank",
        }
        assert set(evaluate.VARIANTS.keys()) == expected_names
