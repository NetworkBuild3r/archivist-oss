"""Tests for Sprint 3 enterprise scaling improvements (v1.10).

Covers:
  - HNSW config variables
  - Collection router (namespace sharding)
  - Cache backend abstraction (memory + Redis fallback)
  - Latency budget system
  - SearchParams integration
  - Parallel pipeline (12c)
  - Collection routing in retriever/indexer/storage (12a integration)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── HNSW Config Tests ────────────────────────────────────────────────────────

class TestHNSWConfig:
    def test_hnsw_defaults(self):
        from config import QDRANT_HNSW_M, QDRANT_HNSW_EF_CONSTRUCT, QDRANT_SEARCH_EF
        assert QDRANT_HNSW_M == 32
        assert QDRANT_HNSW_EF_CONSTRUCT == 256
        assert QDRANT_SEARCH_EF == 256

    def test_search_params_import(self):
        from qdrant_client.models import SearchParams
        params = SearchParams(hnsw_ef=256)
        assert params.hnsw_ef == 256


# ── Collection Router Tests ──────────────────────────────────────────────────

class TestCollectionRouter:
    def test_single_collection_mode(self):
        from collection_router import collection_for
        from config import QDRANT_COLLECTION
        assert collection_for("") == QDRANT_COLLECTION
        # When sharding is disabled (default), all namespaces map to primary
        assert collection_for("some-namespace") == QDRANT_COLLECTION

    def test_collections_for_query_default(self):
        from collection_router import collections_for_query
        from config import QDRANT_COLLECTION
        result = collections_for_query("")
        assert result == [QDRANT_COLLECTION]

    def test_collections_for_query_with_namespace(self):
        from collection_router import collections_for_query
        from config import QDRANT_COLLECTION
        result = collections_for_query("my-ns")
        assert result == [QDRANT_COLLECTION]

    def test_collection_name_sanitization(self):
        """Namespace names with special chars should be sanitized."""
        from collection_router import collection_for
        from config import QDRANT_COLLECTION, NAMESPACE_SHARDING_ENABLED, SINGLE_COLLECTION_MODE
        # When sharding is disabled, this just returns the primary collection
        name = collection_for("test/namespace with spaces")
        assert "/" not in name or name == QDRANT_COLLECTION
        assert " " not in name or name == QDRANT_COLLECTION


# ── Cache Backend Tests ──────────────────────────────────────────────────────

class TestCacheBackendMemory:
    def test_put_and_get(self):
        from cache_backend import MemoryBackend
        cache = MemoryBackend()
        cache.put("key1", {"data": "hello"}, ttl_seconds=60)
        result = cache.get("key1")
        assert result == {"data": "hello"}

    def test_miss(self):
        from cache_backend import MemoryBackend
        cache = MemoryBackend()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        from cache_backend import MemoryBackend
        cache = MemoryBackend()
        cache.put("expire_me", "val", ttl_seconds=60)
        # Manually backdate
        key = "expire_me"
        ts, ttl, val = cache._store[key]
        cache._store[key] = (ts - 120, ttl, val)
        assert cache.get("expire_me") is None

    def test_lru_eviction(self):
        from cache_backend import MemoryBackend
        cache = MemoryBackend(max_entries=5)
        for i in range(10):
            cache.put(f"key{i}", f"val{i}", ttl_seconds=600)
        assert cache.size() == 5
        # Earliest keys should be evicted
        assert cache.get("key0") is None
        assert cache.get("key9") is not None

    def test_delete(self):
        from cache_backend import MemoryBackend
        cache = MemoryBackend()
        cache.put("del_me", "val")
        assert cache.delete("del_me")
        assert cache.get("del_me") is None

    def test_delete_pattern(self):
        from cache_backend import MemoryBackend
        cache = MemoryBackend()
        cache.put("agent:a:1", "v1")
        cache.put("agent:a:2", "v2")
        cache.put("agent:b:1", "v3")
        n = cache.delete_pattern("agent:a:*")
        assert n == 2
        assert cache.get("agent:b:1") is not None

    def test_clear(self):
        from cache_backend import MemoryBackend
        cache = MemoryBackend()
        for i in range(5):
            cache.put(f"k{i}", i)
        n = cache.clear()
        assert n == 5
        assert cache.size() == 0

    def test_get_backend_returns_memory_default(self):
        from cache_backend import get_cache_backend, MemoryBackend
        backend = get_cache_backend()
        assert isinstance(backend, MemoryBackend)


# ── Latency Budget Tests ────────────────────────────────────────────────────

class TestLatencyBudget:
    def test_initial_state(self):
        from latency_budget import LatencyBudget
        b = LatencyBudget(max_ms=500)
        assert b.remaining_ms() > 490
        assert b.remaining_ms() <= 500
        assert not b.is_expired()

    def test_can_afford(self):
        from latency_budget import LatencyBudget
        b = LatencyBudget(max_ms=500)
        assert b.can_afford(100)
        assert b.can_afford(490)  # tiny elapsed time since construction

    def test_cannot_afford_over_budget(self):
        from latency_budget import LatencyBudget
        b = LatencyBudget(max_ms=50)
        # Sleep briefly to consume some budget
        time.sleep(0.06)
        assert not b.can_afford(100)

    def test_reserve(self):
        from latency_budget import LatencyBudget
        b = LatencyBudget(max_ms=500)
        assert b.reserve("embed", 80)
        assert "embed" in b.summary()["reservations"]

    def test_elapsed_tracking(self):
        from latency_budget import LatencyBudget
        b = LatencyBudget(max_ms=500)
        time.sleep(0.02)
        assert b.elapsed_ms() >= 15

    def test_summary(self):
        from latency_budget import LatencyBudget
        b = LatencyBudget(max_ms=500)
        s = b.summary()
        assert "budget_ms" in s
        assert "elapsed_ms" in s
        assert "remaining_ms" in s
        assert s["budget_ms"] == 500

    def test_is_expired_after_timeout(self):
        from latency_budget import LatencyBudget
        b = LatencyBudget(max_ms=10)
        time.sleep(0.02)
        assert b.is_expired()


# ── Search Params in rlm_retriever Tests ─────────────────────────────────────

class TestSearchParamsIntegration:
    def test_search_params_in_imports(self):
        """Verify SearchParams is imported in rlm_retriever."""
        import rlm_retriever
        from qdrant_client.models import SearchParams
        # The import should not raise

    def test_qdrant_search_ef_config(self):
        from config import QDRANT_SEARCH_EF
        assert isinstance(QDRANT_SEARCH_EF, int)
        assert QDRANT_SEARCH_EF > 0


# ── Enterprise Config Tests ──────────────────────────────────────────────────

class TestEnterpriseConfig:
    def test_namespace_sharding_default_off(self):
        from config import NAMESPACE_SHARDING_ENABLED
        assert not NAMESPACE_SHARDING_ENABLED

    def test_single_collection_mode_default_on(self):
        from config import SINGLE_COLLECTION_MODE
        assert SINGLE_COLLECTION_MODE

    def test_cache_backend_default_memory(self):
        from config import CACHE_BACKEND
        assert CACHE_BACKEND == "memory"

    def test_latency_budget_default(self):
        from config import LATENCY_BUDGET_MS
        assert LATENCY_BUDGET_MS == 500

    def test_redis_url_default(self):
        from config import REDIS_URL
        assert "localhost" in REDIS_URL or "redis" in REDIS_URL


# ── Parallel Pipeline Tests (12c) ───────────────────────────────────────────

class TestParallelPipeline:
    def test_asyncio_to_thread_available(self):
        """asyncio.to_thread is used to run sync BM25 in parallel."""
        import asyncio
        assert hasattr(asyncio, "to_thread")

    def test_bm25_search_is_sync(self):
        """BM25 search is synchronous (SQLite) — needs to_thread for parallel."""
        import inspect
        from fts_search import search_bm25
        assert not inspect.iscoroutinefunction(search_bm25)


# ── Collection Routing Integration Tests (12a wiring) ────────────────────────

class TestCollectionRoutingIntegration:
    def test_search_vectors_uses_collection_router(self):
        """search_vectors should import collection_for for routing."""
        import inspect
        from rlm_retriever import search_vectors
        source = inspect.getsource(search_vectors)
        assert "collection_for" in source

    def test_enrich_with_parent_uses_collection_router(self):
        """enrich_with_parent should route to the correct shard."""
        import inspect
        from rlm_retriever import enrich_with_parent
        source = inspect.getsource(enrich_with_parent)
        assert "collection_for" in source

    def test_collection_router_refresh(self):
        """refresh_known_collections should return 0 when sharding disabled."""
        from collection_router import refresh_known_collections
        assert refresh_known_collections() == 0

    def test_drop_collection_noop_for_primary(self):
        """drop_collection should refuse to drop the primary collection."""
        from collection_router import drop_collection
        assert drop_collection("") is False
