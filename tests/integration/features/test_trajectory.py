import pytest

pytestmark = [pytest.mark.integration]
"""Tests for Phase 3 (v0.6.0) — trajectory, annotations, ratings, outcome-aware retrieval."""


async def test_outcome_adjustments_empty(async_pool):
    from archivist.core.trajectory import get_outcome_adjustments

    assert await get_outcome_adjustments([]) == {}


async def test_add_annotation_and_retrieve(async_pool):
    from archivist.core.trajectory import _ensure_trajectory_schema, add_annotation, get_annotations

    _ensure_trajectory_schema()

    ann_id = await add_annotation("mem-1", "agent-a", "This fact is outdated", "stale", 0.3)
    assert ann_id

    anns = await get_annotations("mem-1")
    assert len(anns) == 1
    assert anns[0]["content"] == "This fact is outdated"
    assert anns[0]["annotation_type"] == "stale"
    assert anns[0]["quality_score"] == 0.3


async def test_add_rating_and_summary(async_pool):
    from archivist.core.trajectory import _ensure_trajectory_schema, add_rating, get_rating_summary

    _ensure_trajectory_schema()

    await add_rating("mem-1", "agent-a", 1, "very helpful")
    await add_rating("mem-1", "agent-b", 1)
    await add_rating("mem-1", "agent-c", -1, "outdated")

    summary = await get_rating_summary("mem-1")
    assert summary["total"] == 3
    assert summary["up"] == 2
    assert summary["down"] == 1


async def test_search_tips_empty(async_pool):
    from archivist.core.trajectory import _ensure_trajectory_schema, search_tips

    _ensure_trajectory_schema()
    tips = await search_tips("agent-x", category="strategy")
    assert tips == []


async def test_search_tips_query_ranks_relevant_and_bumps_usage(async_pool):
    """INIT-007/SPEC-003: query ranks relevant tip over newer irrelevant; usage bumps."""
    import uuid
    from datetime import UTC, datetime

    from archivist.core.trajectory import _ensure_trajectory_schema, search_tips
    from archivist.storage.sqlite_pool import pool

    _ensure_trajectory_schema()
    agent = "agent-tip-rank"
    traj_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    tip_new = str(uuid.uuid4())
    tip_old = str(uuid.uuid4())

    async with pool.write() as conn:
        await conn.execute(
            """INSERT INTO trajectories
               (id, agent_id, session_id, task_description, task_fingerprint,
                actions, outcome, outcome_score, memory_ids_used, created_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                traj_id,
                agent,
                "s1",
                "seed",
                "fp",
                "[]",
                "success",
                1.0,
                "[]",
                now,
                "{}",
            ),
        )
        await conn.execute(
            """INSERT INTO tips
               (id, trajectory_id, agent_id, category, tip_text, context,
                archived, created_at, usage_count)
               VALUES (?,?,?,?,?,?,0,?,0)""",
            (
                tip_new,
                traj_id,
                agent,
                "optimization",
                "Rotate API keys monthly",
                "",
                "2026-07-26T12:00:00+00:00",
            ),
        )
        await conn.execute(
            """INSERT INTO tips
               (id, trajectory_id, agent_id, category, tip_text, context,
                archived, created_at, usage_count)
               VALUES (?,?,?,?,?,?,0,?,0)""",
            (
                tip_old,
                traj_id,
                agent,
                "strategy",
                "Prefer express shipping for rush orders",
                "",
                "2026-07-01T12:00:00+00:00",
            ),
        )

    # Recency-only (empty query): newer tip first
    recent = await search_tips(agent, limit=2, query="")
    assert recent[0]["id"] == tip_new

    # Conditioned: older express tip wins
    ranked = await search_tips(
        agent,
        limit=2,
        query="express shipping rush order",
        record_usage=True,
    )
    assert ranked[0]["id"] == tip_old
    assert ranked[0]["usage_count"] >= 0  # pre-bump snapshot; re-fetch below

    async with pool.read() as conn:
        row = await conn.fetchone(
            "SELECT usage_count, last_used_at FROM tips WHERE id=?", (tip_old,)
        )
    assert row is not None
    assert int(row["usage_count"]) >= 1
    assert row["last_used_at"]

    # Agent scoping: other agent sees nothing
    other = await search_tips("other-agent", limit=5, query="express shipping")
    assert other == []


def test_retrieval_trace_v06_fields():
    from archivist.retrieval.rlm_retriever import _retrieval_trace

    trace = _retrieval_trace(
        vector_limit=64,
        coarse_count=50,
        deduped_count=45,
        threshold=0.65,
        after_threshold_count=30,
        after_rerank_count=10,
        parent_enriched=True,
        refinement_chunks=10,
        graph_entities_found=3,
        graph_context_items=8,
        temporal_decay_applied=True,
        tier="l2",
        outcome_adjustments=5,
    )
    assert trace["outcome_adjustments"] == 5
    assert "graph_retrieval_enabled" in trace


async def test_rating_clamp(async_pool):
    from archivist.core.trajectory import _ensure_trajectory_schema, add_rating, get_rating_summary

    _ensure_trajectory_schema()
    await add_rating("mem-2", "agent-a", 5)  # clamps to 1
    await add_rating("mem-2", "agent-b", -10)  # clamps to -1

    summary = await get_rating_summary("mem-2")
    assert summary["up"] == 1
    assert summary["down"] == 1
