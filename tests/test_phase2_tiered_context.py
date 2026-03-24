"""Tests for Phase 2 (v0.5.0) — tiering, graph retrieval, compressed index, contradictions."""

import math
from datetime import datetime, timezone

# ── tiering ──────────────────────────────────────────────────────────────────

def test_select_tier_l0():
    from tiering import select_tier
    hit = {"text": "full text here", "l0": "short", "l1": "medium overview"}
    assert select_tier(hit, "l0") == "short"

def test_select_tier_l1():
    from tiering import select_tier
    hit = {"text": "full text here", "l0": "short", "l1": "medium overview"}
    assert select_tier(hit, "l1") == "medium overview"

def test_select_tier_l2():
    from tiering import select_tier
    hit = {"text": "full text here", "l0": "short", "l1": "medium overview"}
    assert select_tier(hit, "l2") == "full text here"

def test_select_tier_missing_l0_falls_back():
    from tiering import select_tier
    hit = {"text": "full text here"}
    result = select_tier(hit, "l0")
    assert "full text" in result

# ── temporal decay ───────────────────────────────────────────────────────────

def test_temporal_decay_recent_scores_higher():
    from graph_retrieval import apply_temporal_decay
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results = [
        {"score": 0.9, "date": today},
        {"score": 0.9, "date": "2020-01-01"},
    ]
    decayed = apply_temporal_decay(results, halflife_days=30)
    assert decayed[0]["score"] > decayed[1]["score"]

def test_temporal_decay_preserves_original():
    from graph_retrieval import apply_temporal_decay
    results = [{"score": 0.8, "date": "2025-06-01"}]
    decayed = apply_temporal_decay(results, halflife_days=30)
    assert "original_score" in decayed[0]
    assert decayed[0]["original_score"] == 0.8

# ── contradiction detection ──────────────────────────────────────────────────

def test_detect_contradictions_opposing_keywords():
    import graph_retrieval

    original_fn = graph_retrieval.get_entity_facts
    graph_retrieval.get_entity_facts = lambda eid: [
        {"fact_text": "Service is enabled and running", "agent_id": "agent-a", "created_at": "2026-01-01"},
        {"fact_text": "Service is disabled", "agent_id": "agent-b", "created_at": "2026-01-02"},
    ]
    try:
        contras = graph_retrieval.detect_contradictions(1)
        assert len(contras) >= 1
        assert "enabled" in contras[0]["trigger"] or "disabled" in contras[0]["trigger"]
    finally:
        graph_retrieval.get_entity_facts = original_fn

def test_detect_contradictions_same_agent_skipped():
    import graph_retrieval

    original_fn = graph_retrieval.get_entity_facts
    graph_retrieval.get_entity_facts = lambda eid: [
        {"fact_text": "Service enabled", "agent_id": "agent-a", "created_at": "2026-01-01"},
        {"fact_text": "Service disabled", "agent_id": "agent-a", "created_at": "2026-01-02"},
    ]
    try:
        contras = graph_retrieval.detect_contradictions(1)
        assert len(contras) == 0
    finally:
        graph_retrieval.get_entity_facts = original_fn

# ── retrieval trace new fields ───────────────────────────────────────────────

def test_retrieval_trace_v05_fields():
    from rlm_retriever import _retrieval_trace
    trace = _retrieval_trace(
        vector_limit=64, coarse_count=50, deduped_count=45, threshold=0.65,
        after_threshold_count=30, after_rerank_count=10, parent_enriched=True,
        refinement_chunks=10, graph_entities_found=3, graph_context_items=8,
        temporal_decay_applied=True, tier="l1",
    )
    assert trace["graph_retrieval_enabled"] is not None
    assert trace["graph_entities_found"] == 3
    assert trace["graph_context_items"] == 8
    assert trace["temporal_decay_applied"] is True
    assert trace["tier"] == "l1"

# ── compressed index ─────────────────────────────────────────────────────────

def test_compressed_index_empty_namespace():
    from compressed_index import build_namespace_index
    import graph

    original_get_db = graph.get_db
    import sqlite3

    class FakeConn:
        def __init__(self):
            self._db = sqlite3.connect(":memory:")
            self._db.row_factory = sqlite3.Row
            self._db.execute("CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT, entity_type TEXT, mention_count INTEGER)")
        def execute(self, *args, **kwargs):
            return self._db.execute(*args, **kwargs)
        def close(self):
            self._db.close()

    fake = FakeConn()
    graph.get_db = lambda: fake
    try:
        result = build_namespace_index("test-ns")
        assert "No indexed knowledge" in result
    finally:
        graph.get_db = original_get_db
        fake.close()
