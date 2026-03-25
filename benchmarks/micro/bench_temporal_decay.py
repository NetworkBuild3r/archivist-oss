"""Benchmark: temporal decay computation on result sets."""

import sys
import os
from datetime import datetime, timezone, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _make_dated_results(n: int) -> list[dict]:
    """Generate N results with dates spread across the last 90 days."""
    now = datetime.now(timezone.utc).date()
    results = []
    for i in range(n):
        d = now - timedelta(days=i % 90)
        results.append({
            "id": f"id-{i}",
            "score": 0.9 - (i * 0.005),
            "date": d.isoformat(),
            "text": f"Memory chunk {i}",
        })
    return results


@pytest.mark.parametrize("n_results", [20, 100, 500])
def test_temporal_decay_latency(benchmark, n_results):
    """Measure apply_temporal_decay at various result set sizes."""
    from graph_retrieval import apply_temporal_decay

    results = _make_dated_results(n_results)
    benchmark(apply_temporal_decay, results, halflife_days=30)


@pytest.mark.parametrize("halflife", [7, 30, 90, 365])
def test_temporal_decay_halflife_sweep(benchmark, halflife):
    """Measure decay with different halflife values (100 results)."""
    from graph_retrieval import apply_temporal_decay

    results = _make_dated_results(100)
    decayed = benchmark(apply_temporal_decay, results, halflife_days=halflife)
    assert decayed[0]["score"] >= decayed[-1]["score"]


def test_temporal_decay_preserves_order_for_same_date(benchmark):
    """Verify same-date results maintain original score order after decay."""
    from graph_retrieval import apply_temporal_decay

    today = datetime.now(timezone.utc).date().isoformat()
    results = [{"id": f"id-{i}", "score": 1.0 - i * 0.1, "date": today, "text": ""} for i in range(10)]

    decayed = benchmark(apply_temporal_decay, results, halflife_days=30)
    scores = [r["score"] for r in decayed]
    assert scores == sorted(scores, reverse=True)
