"""Benchmark: vector+BM25 score fusion and merge overhead."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_helpers import make_vector_results, make_bm25_results


@pytest.mark.parametrize("n_results", [10, 50, 200])
def test_merge_fusion_latency(benchmark, n_results, monkeypatch):
    """Measure merge_vector_and_bm25 at various result set sizes."""
    import fts_search

    monkeypatch.setattr(fts_search, "VECTOR_WEIGHT", 0.7)
    monkeypatch.setattr(fts_search, "BM25_WEIGHT", 0.3)

    v_results = make_vector_results(n_results)
    b_results = make_bm25_results(n_results)

    benchmark(fts_search.merge_vector_and_bm25, v_results, b_results)


def test_merge_fusion_with_overlap(benchmark, monkeypatch):
    """Measure fusion when 50% of BM25 results share qdrant_id with vector results."""
    import fts_search

    monkeypatch.setattr(fts_search, "VECTOR_WEIGHT", 0.7)
    monkeypatch.setattr(fts_search, "BM25_WEIGHT", 0.3)

    v_results = make_vector_results(50)
    b_results = make_bm25_results(50)

    for i in range(25):
        b_results[i]["qdrant_id"] = str(v_results[i]["id"])

    benchmark(fts_search.merge_vector_and_bm25, v_results, b_results)


def test_merge_vector_only(benchmark, monkeypatch):
    """Measure fusion fast path when BM25 returns empty."""
    import fts_search

    v_results = make_vector_results(50)

    benchmark(fts_search.merge_vector_and_bm25, v_results, [])


def test_merge_bm25_only(benchmark, monkeypatch):
    """Measure fusion fast path when vector returns empty."""
    import fts_search

    monkeypatch.setattr(fts_search, "BM25_WEIGHT", 0.3)
    b_results = make_bm25_results(50)

    benchmark(fts_search.merge_vector_and_bm25, [], b_results)
