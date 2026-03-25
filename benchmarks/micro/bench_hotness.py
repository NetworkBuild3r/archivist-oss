"""Benchmark: hotness scoring — compute_hotness and batch apply."""

import sys
import os
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def test_compute_hotness_single(benchmark, monkeypatch):
    """Measure single hotness computation (pure math)."""
    from hotness import compute_hotness

    benchmark(compute_hotness, retrieval_count=50, days_since_last_access=3.5, halflife=7)


@pytest.mark.parametrize("n_scores", [100, 1000, 10000])
def test_compute_hotness_batch(benchmark, n_scores, monkeypatch):
    """Measure batch hotness computation (N scores in a loop)."""
    from hotness import compute_hotness

    counts = list(range(1, n_scores + 1))
    days = [i * 0.5 for i in range(n_scores)]

    def _batch():
        return [compute_hotness(c, d, halflife=7) for c, d in zip(counts, days)]

    results = benchmark(_batch)
    assert len(results) == n_scores


def test_apply_hotness_to_results(benchmark, monkeypatch):
    """Measure apply_hotness_to_results with pre-populated hotness table."""
    from hotness import apply_hotness_to_results, _ensure_schema, compute_hotness
    from graph import get_db, GRAPH_WRITE_LOCK

    _ensure_schema()

    ids = [str(uuid.uuid4()) for _ in range(50)]
    conn = get_db()
    with GRAPH_WRITE_LOCK:
        for i, mid in enumerate(ids):
            score = compute_hotness(i + 1, i * 0.5, halflife=7)
            conn.execute(
                "INSERT OR REPLACE INTO memory_hotness (memory_id, score, retrieval_count, last_accessed, updated_at) "
                "VALUES (?, ?, ?, datetime('now'), datetime('now'))",
                (mid, score, i + 1),
            )
        conn.commit()
    conn.close()

    results = [
        {"id": mid, "score": 0.8 - i * 0.01, "text": f"chunk {i}"}
        for i, mid in enumerate(ids)
    ]

    benchmark(apply_hotness_to_results, results, weight=0.15)
