"""Benchmark: Prometheus metrics render() latency at various observation counts."""

import sys
import os
import random

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _seed_metrics(n_observations: int):
    """Populate metrics with N histogram observations and counters."""
    import metrics

    metrics._counters.clear()
    metrics._histogram_obs.clear()
    metrics._gauges.clear()

    for i in range(n_observations):
        metrics.observe("archivist_search_duration_ms", random.uniform(5, 5000))
        metrics.inc("archivist_search_total")
    for i in range(n_observations // 5):
        metrics.observe("archivist_llm_duration_ms", random.uniform(100, 10000))
        metrics.inc("archivist_llm_call_total")
    metrics.gauge_set("archivist_cache_entries", 42.0)
    metrics.gauge_set("archivist_curator_queue_depth", 7.0)


@pytest.mark.parametrize("n_obs", [100, 1000, 10000])
def test_metrics_render_latency(benchmark, n_obs):
    """Measure render() latency scaling with observation count."""
    import metrics

    _seed_metrics(n_obs)

    output = benchmark(metrics.render)
    assert "archivist_search_duration_ms" in output
    assert "# TYPE" in output


def test_metrics_render_cold(benchmark):
    """Measure render() with minimal data (near-empty state)."""
    import metrics

    metrics._counters.clear()
    metrics._histogram_obs.clear()
    metrics._gauges.clear()
    metrics.inc("archivist_search_total", value=1)

    output = benchmark(metrics.render)
    assert "archivist_search_total" in output


def test_metrics_inc_throughput(benchmark):
    """Measure counter increment throughput (hot path on every request)."""
    import metrics

    metrics._counters.clear()
    benchmark(metrics.inc, "archivist_search_total")


def test_metrics_observe_throughput(benchmark):
    """Measure histogram observe throughput (hot path on every request)."""
    import metrics

    metrics._histogram_obs.clear()
    benchmark(metrics.observe, "archivist_search_duration_ms", 42.5)
