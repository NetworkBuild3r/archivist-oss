"""Phase 5 tests — hot cache, archivist:// URIs, retrieval log, consistency."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── URI parsing ──────────────────────────────────────────────────────────────


def test_uri_roundtrip():
    from archivist_uri import entity_uri, memory_uri, namespace_uri, parse_uri, skill_uri

    m = memory_uri("agents-nova", "abc-123")
    assert m == "archivist://agents-nova/memory/abc-123"
    parsed = parse_uri(m)
    assert parsed is not None
    assert parsed.namespace == "agents-nova"
    assert parsed.resource_type == "memory"
    assert parsed.resource_id == "abc-123"
    assert parsed.is_memory

    e = entity_uri("shared", "42", hop="2")
    parsed_e = parse_uri(e)
    assert parsed_e.is_entity
    assert parsed_e.resource_id == "42"
    assert parsed_e.params.get("hop") == "2"

    n = namespace_uri("team-alpha")
    parsed_n = parse_uri(n)
    assert parsed_n.is_namespace

    s = skill_uri("global", "web_search")
    parsed_s = parse_uri(s)
    assert parsed_s.is_skill
    assert parsed_s.resource_id == "web_search"


def test_uri_parse_invalid():
    from archivist_uri import parse_uri

    assert parse_uri("http://not-archivist/foo") is None
    assert parse_uri("archivist://ns/bad_type/123") is None
    assert parse_uri("") is None


# ── Hot cache ────────────────────────────────────────────────────────────────


def test_cache_put_get():
    import hot_cache

    hot_cache.invalidate_all()

    result = {"answer": "test", "sources": [], "_cache_namespace": "ns1"}
    hot_cache.put("agent-a", "what is X?", result, namespace="ns1")

    cached = hot_cache.get("agent-a", "what is X?", namespace="ns1")
    assert cached is not None
    assert cached["answer"] == "test"

    miss = hot_cache.get("agent-a", "different query", namespace="ns1")
    assert miss is None

    miss2 = hot_cache.get("agent-b", "what is X?", namespace="ns1")
    assert miss2 is None

    hot_cache.invalidate_all()


def test_cache_lru_eviction():
    import hot_cache

    hot_cache.invalidate_all()

    original_max = hot_cache.HOT_CACHE_MAX_PER_AGENT
    import config

    old_val = config.HOT_CACHE_MAX_PER_AGENT
    try:
        hot_cache.HOT_CACHE_MAX_PER_AGENT = 3
        for i in range(5):
            hot_cache.put(
                "agent-lru",
                f"query-{i}",
                {"answer": f"r{i}", "_cache_namespace": "ns"},
                namespace="ns",
            )

        assert hot_cache.get("agent-lru", "query-0", namespace="ns") is None
        assert hot_cache.get("agent-lru", "query-1", namespace="ns") is None
        assert hot_cache.get("agent-lru", "query-4", namespace="ns") is not None
    finally:
        hot_cache.HOT_CACHE_MAX_PER_AGENT = original_max
        config.HOT_CACHE_MAX_PER_AGENT = old_val
        hot_cache.invalidate_all()


def test_cache_namespace_invalidation():
    import hot_cache

    hot_cache.invalidate_all()

    hot_cache.put("a1", "q1", {"answer": "x", "_cache_namespace": "ns-alpha"}, namespace="ns-alpha")
    hot_cache.put("a1", "q2", {"answer": "y", "_cache_namespace": "ns-beta"}, namespace="ns-beta")

    evicted = hot_cache.invalidate_namespace("ns-alpha")
    assert evicted == 1
    assert hot_cache.get("a1", "q1", namespace="ns-alpha") is None
    assert hot_cache.get("a1", "q2", namespace="ns-beta") is not None

    hot_cache.invalidate_all()


def test_cache_stats():
    import hot_cache

    hot_cache.invalidate_all()

    hot_cache.put("s1", "q1", {"answer": "ok", "_cache_namespace": "ns"}, namespace="ns")
    s = hot_cache.stats()
    assert s["enabled"] is True
    assert s["total_entries"] >= 1
    assert "s1" in s["per_agent"]

    hot_cache.invalidate_all()


# ── Retrieval log ────────────────────────────────────────────────────────────


def test_retrieval_log_roundtrip():
    from retrieval_log import get_retrieval_logs, log_retrieval

    lid = log_retrieval(
        agent_id="agent-x",
        query="test query",
        namespace="ns1",
        tier="l2",
        memory_type="",
        retrieval_trace={"coarse_hits": 10, "after_threshold": 5},
        result_count=3,
        cache_hit=False,
        duration_ms=150,
    )
    assert lid

    logs = get_retrieval_logs(agent_id="agent-x")
    assert len(logs) == 1
    assert logs[0]["query"] == "test query"
    assert logs[0]["duration_ms"] == 150
    assert logs[0]["retrieval_trace"]["coarse_hits"] == 10
    assert logs[0]["cache_hit"] is False


def test_retrieval_stats():
    from retrieval_log import get_retrieval_stats, log_retrieval

    for i in range(5):
        log_retrieval(
            "a1",
            f"q{i}",
            "ns",
            "l2",
            "",
            {"hits": i},
            i,
            cache_hit=(i % 2 == 0),
            duration_ms=100 + i * 10,
        )

    stats = get_retrieval_stats("a1")
    assert stats["total"] == 5
    assert stats["cache_hits"] == 3
    assert stats["cache_hit_rate"] == 0.6
    assert stats["avg_duration_ms"] is not None
    assert len(stats["top_agents"]) >= 1


# ── Consistency semantics ────────────────────────────────────────────────────


def test_default_consistency_config():
    from config import DEFAULT_CONSISTENCY

    assert DEFAULT_CONSISTENCY in ("eventual", "session", "strong")
