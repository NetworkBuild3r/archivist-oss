"""Tests for Phase 3 (v0.6.0) — trajectory, annotations, ratings, outcome-aware retrieval."""

import sqlite3


def _make_test_db():
    """Create an in-memory SQLite with the same config as graph.get_db."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def _patch_dbs(mem_db):
    """Patch both graph.get_db AND trajectory.get_db so the local binding is updated."""
    import graph
    import trajectory
    orig_graph = graph.get_db
    orig_traj = trajectory.get_db
    graph.get_db = lambda: mem_db
    trajectory.get_db = lambda: mem_db
    trajectory._SCHEMA_APPLIED = False
    return orig_graph, orig_traj


def _unpatch_dbs(orig_graph, orig_traj, mem_db):
    import graph
    import trajectory
    graph.get_db = orig_graph
    trajectory.get_db = orig_traj
    trajectory._SCHEMA_APPLIED = False
    mem_db.close()


def test_outcome_adjustments_empty():
    from trajectory import get_outcome_adjustments
    assert get_outcome_adjustments([]) == {}


def test_add_annotation_and_retrieve():
    from trajectory import _ensure_trajectory_schema, add_annotation, get_annotations

    mem_db = _make_test_db()
    orig_graph, orig_traj = _patch_dbs(mem_db)

    try:
        _ensure_trajectory_schema()

        ann_id = add_annotation("mem-1", "agent-a", "This fact is outdated", "stale", 0.3)
        assert ann_id

        anns = get_annotations("mem-1")
        assert len(anns) == 1
        assert anns[0]["content"] == "This fact is outdated"
        assert anns[0]["annotation_type"] == "stale"
        assert anns[0]["quality_score"] == 0.3
    finally:
        _unpatch_dbs(orig_graph, orig_traj, mem_db)


def test_add_rating_and_summary():
    from trajectory import _ensure_trajectory_schema, add_rating, get_rating_summary

    mem_db = _make_test_db()
    orig_graph, orig_traj = _patch_dbs(mem_db)

    try:
        _ensure_trajectory_schema()

        add_rating("mem-1", "agent-a", 1, "very helpful")
        add_rating("mem-1", "agent-b", 1)
        add_rating("mem-1", "agent-c", -1, "outdated")

        summary = get_rating_summary("mem-1")
        assert summary["total"] == 3
        assert summary["up"] == 2
        assert summary["down"] == 1
    finally:
        _unpatch_dbs(orig_graph, orig_traj, mem_db)


def test_search_tips_empty():
    from trajectory import _ensure_trajectory_schema, search_tips

    mem_db = _make_test_db()
    orig_graph, orig_traj = _patch_dbs(mem_db)

    try:
        _ensure_trajectory_schema()
        tips = search_tips("agent-x", category="strategy")
        assert tips == []
    finally:
        _unpatch_dbs(orig_graph, orig_traj, mem_db)


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

    mem_db = _make_test_db()
    orig_graph, orig_traj = _patch_dbs(mem_db)

    try:
        _ensure_trajectory_schema()
        add_rating("mem-2", "agent-a", 5)   # clamps to 1
        add_rating("mem-2", "agent-b", -10)  # clamps to -1

        summary = get_rating_summary("mem-2")
        assert summary["up"] == 1
        assert summary["down"] == 1
    finally:
        _unpatch_dbs(orig_graph, orig_traj, mem_db)
