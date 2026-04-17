"""Tests for Phase 7 — Memory Intelligence Layer (v1.0.0).

Tests cover: curator_queue, LLM dedup dataclasses, hotness scoring, skill relations,
tip consolidation schema, context-status signaling, and metrics names.
"""

import os
import sys
import uuid

import pytest

src = os.path.join(os.path.dirname(__file__), "..", "src")
if src not in sys.path:
    sys.path.insert(0, src)


# ── Helpers ──────────────────────────────────────────────────────────────────
# DB isolation is handled by conftest._isolate_env (temp file DB, schema reset).


# ── Curator Queue ────────────────────────────────────────────────────────────


class TestCuratorQueue:
    def test_enqueue_returns_uuid(self):
        from curator_queue import enqueue

        op_id = enqueue("merge_memory", {"ids": ["a", "b"]})
        assert len(op_id) == 36  # UUID format

    def test_enqueue_invalid_op_type(self):
        from curator_queue import enqueue

        with pytest.raises(ValueError, match="Invalid op_type"):
            enqueue("invalid_op", {})

    def test_stats_counts_pending(self):
        from curator_queue import enqueue, stats

        enqueue("merge_memory", {})
        enqueue("archive_memory", {})
        s = stats()
        assert s["pending"] >= 2
        assert s["total"] >= 2

    @pytest.mark.asyncio
    async def test_drain_applies_ops(self, async_pool):
        from curator_queue import drain, enqueue, stats

        enqueue("skip_store", {"reason": "test"})
        applied = await drain(limit=10)
        assert len(applied) >= 1
        assert applied[0]["status"] == "applied"
        s = stats()
        assert s["pending"] == 0

    @pytest.mark.asyncio
    async def test_drain_empty_returns_empty(self, async_pool):
        from curator_queue import drain

        applied = await drain(limit=10)
        assert applied == []


# ── Conflict Detection (LLM dedup dataclasses) ──────────────────────────────


class TestDedupResult:
    def test_dedup_result_dataclass(self):
        from conflict_detection import DedupResult

        r = DedupResult(
            action="skip",
            existing_ids=["abc"],
            decisions=[{"existing_id": "abc", "decision": "skip"}],
            max_similarity=0.92,
        )
        assert r.action == "skip"
        assert r.max_similarity == 0.92


# ── Hotness Scoring ──────────────────────────────────────────────────────────


class TestHotnessScoring:
    def test_compute_hotness_fresh(self):
        from hotness import compute_hotness

        score = compute_hotness(retrieval_count=10, days_since_last_access=0, halflife=7)
        assert 0.0 < score <= 1.0

    def test_compute_hotness_decays(self):
        from hotness import compute_hotness

        fresh = compute_hotness(retrieval_count=10, days_since_last_access=0, halflife=7)
        old = compute_hotness(retrieval_count=10, days_since_last_access=30, halflife=7)
        assert fresh > old

    def test_compute_hotness_zero_count(self):
        from hotness import compute_hotness

        score = compute_hotness(retrieval_count=0, days_since_last_access=0, halflife=7)
        assert 0.4 < score < 0.6  # sigmoid(log1p(0)) = sigmoid(0) = 0.5

    def test_sigmoid_bounds(self):
        from hotness import _sigmoid

        assert _sigmoid(0) == 0.5
        assert _sigmoid(100) > 0.99
        assert _sigmoid(-100) < 0.01


# ── Skill Relations ──────────────────────────────────────────────────────────


class TestSkillRelations:
    def test_add_and_get_relation(self):
        from skills import add_skill_relation, get_skill_relations, register_skill

        r1 = register_skill(name="kubectl", provider="k8s", registered_by="test")
        r2 = register_skill(name="helm", provider="k8s", registered_by="test")

        rel_id = add_skill_relation(
            skill_a_id=r1["skill_id"],
            skill_b_id=r2["skill_id"],
            relation_type="compose_with",
            confidence=0.9,
            evidence="both manage k8s resources",
            created_by="test",
        )
        assert rel_id > 0

        rels = get_skill_relations(r1["skill_id"])
        assert len(rels) >= 1
        assert rels[0]["relation_type"] == "compose_with"

    def test_invalid_relation_type(self):
        from skills import add_skill_relation

        with pytest.raises(ValueError, match="Invalid relation_type"):
            add_skill_relation("a", "b", "invalid_type", created_by="test")

    def test_get_substitutes(self):
        from skills import add_skill_relation, get_skill_substitutes, register_skill

        r1 = register_skill(name="docker", provider="oci", registered_by="test")
        r2 = register_skill(name="podman", provider="oci", registered_by="test")

        add_skill_relation(
            skill_a_id=r1["skill_id"],
            skill_b_id=r2["skill_id"],
            relation_type="similar_to",
            confidence=0.95,
            created_by="test",
        )

        subs = get_skill_substitutes(r1["skill_id"])
        assert len(subs) >= 1
        assert subs[0]["name"] == "podman"


# ── Trajectory (tip consolidation schema) ────────────────────────────────────


class TestTipConsolidationSchema:
    def test_tips_table_has_negative_example_column(self):
        from trajectory import _ensure_trajectory_schema

        _ensure_trajectory_schema()

        from graph import get_db

        conn = get_db()
        info = conn.execute("PRAGMA table_info(tips)").fetchall()
        columns = {row["name"] for row in info}
        assert "negative_example" in columns
        assert "archived" in columns

    def test_search_tips_excludes_archived(self):
        from graph import get_db
        from trajectory import _ensure_trajectory_schema, search_tips

        _ensure_trajectory_schema()

        conn = get_db()
        tid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO trajectories (id, agent_id, task_description, outcome, created_at) "
            "VALUES (?, 'test', 'test task', 'success', '2026-01-01T00:00:00')",
            (tid,),
        )
        conn.execute(
            "INSERT INTO tips (id, trajectory_id, agent_id, category, tip_text, archived, created_at) "
            "VALUES (?, ?, 'test', 'strategy', 'active tip', 0, '2026-01-01')",
            (str(uuid.uuid4()), tid),
        )
        conn.execute(
            "INSERT INTO tips (id, trajectory_id, agent_id, category, tip_text, archived, created_at) "
            "VALUES (?, ?, 'test', 'strategy', 'archived tip', 1, '2026-01-01')",
            (str(uuid.uuid4()), tid),
        )
        conn.commit()

        tips = search_tips("test")
        assert len(tips) == 1
        assert tips[0]["tip_text"] == "active tip"


# ── Metrics ──────────────────────────────────────────────────────────────────


class TestCuratorMetrics:
    def test_curator_metric_names_exist(self):
        import metrics as m

        assert hasattr(m, "CURATOR_QUEUE_DEPTH")
        assert hasattr(m, "CURATOR_DEDUP_DECISION")
        assert hasattr(m, "CURATOR_TIP_CONSOLIDATIONS")
        assert hasattr(m, "CURATOR_LLM_CALLS")
        assert hasattr(m, "CURATOR_DRAIN_DURATION")

    def test_curator_metrics_emit(self):
        import metrics as m

        m.inc(m.CURATOR_LLM_CALLS)
        m.inc(m.CURATOR_DEDUP_DECISION, {"decision": "skip"})
        m.inc(m.CURATOR_TIP_CONSOLIDATIONS)
        m.gauge_set(m.CURATOR_QUEUE_DEPTH, 5)
        m.observe(m.CURATOR_DRAIN_DURATION, 42.0)

        rendered = m.render()
        assert "archivist_curator_llm_calls_total" in rendered
        assert "archivist_curator_queue_depth" in rendered
        assert "archivist_curator_drain_duration_ms" in rendered


# ── Config ───────────────────────────────────────────────────────────────────


class TestCuratorConfig:
    def test_config_defaults(self):
        from config import (
            CURATOR_QUEUE_DRAIN_INTERVAL,
            CURATOR_TIP_BUDGET,
            DEDUP_LLM_ENABLED,
            DEDUP_LLM_THRESHOLD,
            HOTNESS_HALFLIFE_DAYS,
            HOTNESS_WEIGHT,
        )

        assert isinstance(DEDUP_LLM_ENABLED, bool)
        assert 0.0 <= DEDUP_LLM_THRESHOLD <= 1.0
        assert CURATOR_TIP_BUDGET > 0
        assert CURATOR_QUEUE_DRAIN_INTERVAL > 0
        assert 0.0 <= HOTNESS_WEIGHT <= 1.0
        assert HOTNESS_HALFLIFE_DAYS > 0
