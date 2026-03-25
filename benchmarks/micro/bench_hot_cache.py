"""Benchmark: hot_cache — hit/miss latency, LRU eviction, namespace invalidation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def test_cache_put_latency(benchmark, monkeypatch):
    """Measure per-entry insertion cost."""
    import hot_cache
    monkeypatch.setattr("config.HOT_CACHE_ENABLED", True)
    monkeypatch.setattr(hot_cache, "HOT_CACHE_ENABLED", True)
    hot_cache.invalidate_all()

    i = 0

    def _put():
        nonlocal i
        hot_cache.put(
            f"agent-{i % 10}",
            f"query about deployment pipeline run {i}",
            {"answer": f"result {i}", "_cache_namespace": "bench"},
            namespace="bench",
        )
        i += 1

    benchmark(_put)
    hot_cache.invalidate_all()


def test_cache_hit_latency(benchmark, monkeypatch):
    """Measure latency on a cache hit (LRU move-to-end)."""
    import hot_cache
    monkeypatch.setattr("config.HOT_CACHE_ENABLED", True)
    monkeypatch.setattr(hot_cache, "HOT_CACHE_ENABLED", True)
    hot_cache.invalidate_all()

    hot_cache.put("agent-0", "stable query", {"answer": "cached", "_cache_namespace": "bench"}, namespace="bench")

    def _get():
        hot_cache.get("agent-0", "stable query", namespace="bench")

    benchmark(_get)
    hot_cache.invalidate_all()


def test_cache_miss_latency(benchmark, monkeypatch):
    """Measure latency on a cache miss."""
    import hot_cache
    monkeypatch.setattr("config.HOT_CACHE_ENABLED", True)
    monkeypatch.setattr(hot_cache, "HOT_CACHE_ENABLED", True)
    hot_cache.invalidate_all()

    def _miss():
        hot_cache.get("agent-0", "query that is not cached", namespace="bench")

    benchmark(_miss)


def test_cache_lru_eviction(benchmark, monkeypatch):
    """Measure put cost when LRU eviction is triggered on every insert."""
    import hot_cache
    monkeypatch.setattr("config.HOT_CACHE_ENABLED", True)
    monkeypatch.setattr(hot_cache, "HOT_CACHE_ENABLED", True)
    monkeypatch.setattr("config.HOT_CACHE_MAX_PER_AGENT", 10)
    monkeypatch.setattr(hot_cache, "HOT_CACHE_MAX_PER_AGENT", 10)
    hot_cache.invalidate_all()

    for j in range(10):
        hot_cache.put("agent-0", f"fill-{j}", {"answer": f"v{j}", "_cache_namespace": "bench"}, namespace="bench")

    i = 0

    def _evict():
        nonlocal i
        hot_cache.put("agent-0", f"overflow-{i}", {"answer": f"new-{i}", "_cache_namespace": "bench"}, namespace="bench")
        i += 1

    benchmark(_evict)
    hot_cache.invalidate_all()


def test_invalidate_namespace_100_agents(benchmark, monkeypatch):
    """Measure invalidate_namespace scanning 100 agents x 50 entries each."""
    import hot_cache
    monkeypatch.setattr("config.HOT_CACHE_ENABLED", True)
    monkeypatch.setattr(hot_cache, "HOT_CACHE_ENABLED", True)
    monkeypatch.setattr("config.HOT_CACHE_MAX_PER_AGENT", 128)
    monkeypatch.setattr(hot_cache, "HOT_CACHE_MAX_PER_AGENT", 128)
    hot_cache.invalidate_all()

    for a in range(100):
        for q in range(50):
            ns = "target" if q % 2 == 0 else "other"
            hot_cache.put(
                f"agent-{a}", f"query-{q}",
                {"answer": f"r-{a}-{q}", "_cache_namespace": ns},
                namespace=ns,
            )

    benchmark(hot_cache.invalidate_namespace, "target")
    hot_cache.invalidate_all()
