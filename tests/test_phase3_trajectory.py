"""Tests for Phase 3 (v0.6.0) — trajectory, annotations, ratings, outcome-aware retrieval."""


def test_outcome_adjustments_empty():
    from trajectory import get_outcome_adjustments
    assert get_outcome_adjustments([]) == {}


def test_add_annotation_and_retrieve():
    from trajectory import _ensure_trajectory_schema, add_annotation, get_annotations

    _ensure_trajectory_schema()

    ann_id = add_annotation("mem-1", "agent-a", "This fact is outdated", "stale", 0.3)
    assert ann_id

    anns = get_annotations("mem-1")
    assert len(anns) == 1
    assert anns[0]["content"] == "This fact is outdated"
    assert anns[0]["annotation_type"] == "stale"
    assert anns[0]["quality_score"] == 0.3


def test_add_rating_and_summary():
    from trajectory import _ensure_trajectory_schema, add_rating, get_rating_summary

    _ensure_trajectory_schema()

    add_rating("mem-1", "agent-a", 1, "very helpful")
    add_rating("mem-1", "agent-b", 1)
    add_rating("mem-1", "agent-c", -1, "outdated")

    summary = get_rating_summary("mem-1")
    assert summary["total"] == 3
    assert summary["up"] == 2
    assert summary["down"] == 1


def test_search_tips_empty():
    from trajectory import _ensure_trajectory_schema, search_tips

    _ensure_trajectory_schema()
    tips = search_tips("agent-x", category="strategy")
    assert tips == []


def test_retrieval_trace_v06_fields():
    from rlm_retriever import _retrieval_trace
    trace = _retrieval_trace(
        vector_limit=64, coarse_count=50, deduped_count=45, threshold=0.65,
        after_threshold_count=30, after_rerank_count=10, parent_enriched=True,
        refinement_chunks=10, graph_entities_found=3, graph_context_items=8,
        temporal_decay_applied=True, tier="l2", outcome_adjustments=5,
    )
    assert trace["outcome_adjustments"] == 5
    assert "graph_retrieval_enabled" in trace


def test_rating_clamp():
    from trajectory import _ensure_trajectory_schema, add_rating, get_rating_summary

    _ensure_trajectory_schema()
    add_rating("mem-2", "agent-a", 5)   # clamps to 1
    add_rating("mem-2", "agent-b", -10)  # clamps to -1

    summary = get_rating_summary("mem-2")
    assert summary["up"] == 1
    assert summary["down"] == 1
