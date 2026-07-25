"""Unit tests for Phase 8 contradiction resolution (INIT-001/SPEC-006).

Covers rule matrix without live LLM (ac-4), documented actions (ac-1),
and safe default flags (ac-3).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.lifecycle]


def _fact(
    fid: int,
    text: str,
    *,
    agent: str,
    created_at: str,
    namespace: str = "ns-a",
    entity_id: int = 1,
) -> dict:
    return {
        "id": fid,
        "entity_id": entity_id,
        "fact_text": text,
        "agent_id": agent,
        "created_at": created_at,
        "namespace": namespace,
        "superseded_by": None,
        "is_active": 1,
    }


class TestRuleMatrix:
    """Deterministic supersede / merge / keep_both paths (no LLM)."""

    def test_temporal_supersede_opposing_keywords(self):
        from archivist.lifecycle.contradiction_resolve import (
            ConflictPair,
            propose_resolution_rules,
        )

        pair = ConflictPair(
            entity_id=1,
            namespace="ns-a",
            fact_a=_fact(
                10,
                "Feature X is enabled in prod",
                agent="alice",
                created_at="2026-01-01T00:00:00+00:00",
            ),
            fact_b=_fact(
                11,
                "Feature X is disabled in prod",
                agent="bob",
                created_at="2026-02-01T00:00:00+00:00",
            ),
            trigger="enabled/disabled",
        )
        prop = propose_resolution_rules(pair)
        assert prop.action == "supersede"
        assert prop.winner_fact_id == 11
        assert prop.loser_fact_id == 10
        assert prop.rule == "temporal_supersede"
        assert prop.reason
        assert prop.resolution_id

    def test_near_duplicate_merge(self):
        from archivist.lifecycle.contradiction_resolve import (
            ConflictPair,
            propose_resolution_rules,
        )

        pair = ConflictPair(
            entity_id=1,
            namespace="ns-a",
            fact_a=_fact(
                20,
                "Primary database host is db-01 in cluster east",
                agent="alice",
                created_at="2026-01-01T00:00:00+00:00",
            ),
            fact_b=_fact(
                21,
                "Primary database host is db-01 in cluster east region",
                agent="bob",
                created_at="2026-01-02T00:00:00+00:00",
            ),
            trigger="near_duplicate",
        )
        prop = propose_resolution_rules(pair)
        assert prop.action == "merge"
        assert prop.merge_text
        assert prop.rule == "near_duplicate_merge"

    def test_ambiguous_keep_both_same_timestamp(self):
        from archivist.lifecycle.contradiction_resolve import (
            ConflictPair,
            propose_resolution_rules,
        )

        ts = "2026-01-01T00:00:00+00:00"
        pair = ConflictPair(
            entity_id=1,
            namespace="ns-a",
            fact_a=_fact(30, "Service is up and healthy", agent="alice", created_at=ts),
            fact_b=_fact(31, "Service is down for maintenance", agent="bob", created_at=ts),
            trigger="up/down",
        )
        prop = propose_resolution_rules(pair)
        assert prop.action == "keep_both"
        assert prop.rule == "ambiguous_keep_both"


class TestDetectPairsNamespace:
    def test_skips_cross_namespace_pairs(self):
        from archivist.lifecycle.contradiction_resolve import detect_conflict_pairs

        facts = [
            _fact(
                1,
                "Feature enabled",
                agent="a",
                created_at="2026-01-01T00:00:00+00:00",
                namespace="ns-a",
            ),
            _fact(
                2,
                "Feature disabled",
                agent="b",
                created_at="2026-02-01T00:00:00+00:00",
                namespace="ns-b",
            ),
        ]
        assert detect_conflict_pairs(facts) == []

    def test_namespace_filter(self):
        from archivist.lifecycle.contradiction_resolve import detect_conflict_pairs

        facts = [
            _fact(
                1,
                "Feature enabled",
                agent="a",
                created_at="2026-01-01T00:00:00+00:00",
                namespace="ns-a",
            ),
            _fact(
                2,
                "Feature disabled",
                agent="b",
                created_at="2026-02-01T00:00:00+00:00",
                namespace="ns-a",
            ),
            _fact(
                3,
                "Feature enabled",
                agent="c",
                created_at="2026-01-01T00:00:00+00:00",
                namespace="ns-b",
            ),
            _fact(
                4,
                "Feature disabled",
                agent="d",
                created_at="2026-02-01T00:00:00+00:00",
                namespace="ns-b",
            ),
        ]
        pairs = detect_conflict_pairs(facts, namespace="ns-a")
        assert len(pairs) == 1
        assert pairs[0].namespace == "ns-a"


class TestRedactSensitive:
    def test_redacts_api_key_style_markers(self):
        from archivist.lifecycle.contradiction_resolve import redact_sensitive

        text = "config has api_key=sk-live-SECRET123 and token: abc.def"
        redacted = redact_sensitive(text)
        assert "sk-live-SECRET123" not in redacted
        assert "REDACTED" in redacted


class TestLlmVerdictContainment:
    """SEC-012-03: an LLM verdict may only name facts from its own pair."""

    def test_out_of_pair_ids_are_dropped(self):
        from archivist.lifecycle.contradiction_resolve import _coerce_pair_fact_ids

        assert _coerce_pair_fact_ids(999, 1000, 1, 2) == (None, None)
        assert _coerce_pair_fact_ids(2, 1, 1, 2) == (2, 1)
        assert _coerce_pair_fact_ids("2", "1", 1, 2) == (2, 1)
        assert _coerce_pair_fact_ids(None, 1, 1, 2) == (None, 1)
        # Same fact cannot both win and lose.
        assert _coerce_pair_fact_ids(1, 1, 1, 2) == (None, None)

    @pytest.mark.asyncio
    async def test_llm_supersede_outside_pair_falls_back_to_rules(self):
        import json
        from unittest.mock import AsyncMock, patch

        from archivist.lifecycle.contradiction_resolve import ConflictPair, propose_resolution

        pair = ConflictPair(
            entity_id=1,
            namespace="ns-a",
            fact_a=_fact(1, "service is enabled", agent="a", created_at="2026-01-01T00:00:00Z"),
            fact_b=_fact(2, "service is disabled", agent="b", created_at="2026-02-01T00:00:00Z"),
            trigger="enabled/disabled",
        )
        hostile = json.dumps(
            {
                "action": "supersede",
                "winner_fact_id": 4242,
                "loser_fact_id": 9001,
                "reason": "injected",
            }
        )
        with (
            patch(
                "archivist.lifecycle.contradiction_resolve.CONTRADICTION_RESOLVE_LLM_ENABLED",
                True,
            ),
            patch(
                "archivist.features.llm.llm_query",
                new=AsyncMock(return_value=hostile),
            ),
        ):
            proposal = await propose_resolution(pair)

        assert proposal.rule == "temporal_supersede"
        assert {proposal.winner_fact_id, proposal.loser_fact_id} == {1, 2}

    @pytest.mark.asyncio
    async def test_apply_rejects_supersede_outside_pair(self):
        from unittest.mock import AsyncMock, patch

        from archivist.lifecycle.contradiction_resolve import (
            ResolutionProposal,
            apply_resolution,
        )

        tampered = ResolutionProposal(
            action="supersede",
            entity_id=1,
            namespace="ns-a",
            fact_a_id=1,
            fact_b_id=2,
            winner_fact_id=4242,
            loser_fact_id=9001,
            merge_text=None,
            reason="tampered",
            rule="llm_adjudicate",
            trigger="enabled/disabled",
        )
        supersede = AsyncMock()
        with (
            patch("archivist.storage.graph.supersede_fact", new=supersede),
            patch("archivist.storage.graph.add_fact", new=AsyncMock()),
            patch(
                "archivist.lifecycle.contradiction_resolve.log_memory_event",
                new=AsyncMock(),
            ),
            patch(
                "archivist.lifecycle.contradiction_resolve._already_applied",
                new=AsyncMock(return_value=False),
            ),
            pytest.raises(ValueError, match="conflict pair"),
        ):
            await apply_resolution(tampered, dry_run=False)

        supersede.assert_not_awaited()


class TestSafeFlagDefaults:
    def test_resolve_and_reflection_flags_default_safe(self):
        from archivist.core.config import ArchivistSettings

        s = ArchivistSettings.model_construct()
        assert s.contradiction_resolve_enabled is False
        assert s.contradiction_resolve_dry_run is True
        assert s.contradiction_resolve_llm_enabled is False
        assert s.reflection_enabled is False
        assert s.reflection_dry_run is True


class TestReflectionDeterministic:
    def test_build_reflection_success_strategy(self):
        from archivist.lifecycle.reflection import build_reflection_from_outcome

        art = build_reflection_from_outcome(
            trajectory_id="t1",
            agent_id="agent-a",
            task_description="deploy api",
            outcome="success",
            outcome_score=0.9,
            namespace="team-x",
            actions=[{"step": 1}],
        )
        assert art.category == "strategy"
        assert "deploy api" in art.tip_text
        assert art.namespace == "team-x"
        assert "trajectory_id=t1" in art.context

    def test_build_reflection_failure_recovery(self):
        from archivist.lifecycle.reflection import build_reflection_from_outcome

        art = build_reflection_from_outcome(
            trajectory_id="t2",
            agent_id="agent-b",
            task_description="migrate db",
            outcome="failure",
            outcome_score=0.1,
            namespace="global",
        )
        assert art.category == "recovery"
        assert "outcome=failure" in art.tip_text


@pytest.mark.asyncio
async def test_apply_supersede_writes_audit_and_mutates(async_pool):
    """ac-1: documented resolution with audit; apply path (dry_run=False)."""
    from archivist.core.audit import get_audit_trail
    from archivist.lifecycle.contradiction_resolve import (
        ConflictPair,
        apply_resolution,
        propose_resolution_rules,
    )
    from archivist.storage.graph import add_fact, get_entity_facts, upsert_entity

    eid = await upsert_entity("svc-resolve-test", "service", namespace="ns-resolve")
    # Distinct wording keeps add_fact auto-supersede (0.6 overlap) from firing.
    fa = await add_fact(
        eid,
        "Canary rollout path is enabled for beta cohort",
        agent_id="alice",
        namespace="ns-resolve",
    )
    from archivist.storage.sqlite_pool import pool

    async with pool.write() as conn:
        await conn.execute(
            "UPDATE facts SET created_at=? WHERE id=?",
            ("2026-01-01T00:00:00+00:00", fa),
        )
    fb = await add_fact(
        eid,
        "Canary rollout path is disabled after incident",
        agent_id="bob",
        namespace="ns-resolve",
    )
    async with pool.write() as conn:
        await conn.execute(
            "UPDATE facts SET created_at=? WHERE id=?",
            ("2026-02-01T00:00:00+00:00", fb),
        )

    facts = await get_entity_facts(eid)
    by_id = {int(f["id"]): f for f in facts}
    pair = ConflictPair(
        entity_id=eid,
        namespace="ns-resolve",
        fact_a=by_id[fa],
        fact_b=by_id[fb],
        trigger="enabled/disabled",
    )
    prop = propose_resolution_rules(pair)
    assert prop.action == "supersede"
    result = await apply_resolution(prop, dry_run=False)
    assert result.applied is True

    facts_after = await get_entity_facts(eid, include_superseded=True)
    loser = next(f for f in facts_after if int(f["id"]) == prop.loser_fact_id)
    assert loser.get("superseded_by") == prop.winner_fact_id

    trail = await get_audit_trail(result.resolution_id)
    assert any(e["action"] == "contradiction_resolved" for e in trail)
    assert all(
        e.get("namespace") == "ns-resolve" for e in trail if e["action"] == "contradiction_resolved"
    )


@pytest.mark.asyncio
async def test_apply_dry_run_does_not_mutate(async_pool):
    from archivist.lifecycle.contradiction_resolve import (
        ConflictPair,
        apply_resolution,
        propose_resolution_rules,
    )
    from archivist.storage.graph import add_fact, get_entity_facts, upsert_entity

    eid = await upsert_entity("svc-dry-run", "service", namespace="ns-dry")
    fa = await add_fact(
        eid, "Auth middleware toggle is enabled in edge", agent_id="a1", namespace="ns-dry"
    )
    fb = await add_fact(
        eid, "Auth middleware toggle is disabled during outage", agent_id="a2", namespace="ns-dry"
    )
    facts = {int(f["id"]): f for f in await get_entity_facts(eid)}
    pair = ConflictPair(
        entity_id=eid,
        namespace="ns-dry",
        fact_a=facts[fa],
        fact_b=facts[fb],
        trigger="enabled/disabled",
    )
    prop = propose_resolution_rules(pair)
    result = await apply_resolution(prop, dry_run=True)
    assert result.applied is False
    assert result.dry_run is True
    active = await get_entity_facts(eid)
    assert all(f.get("superseded_by") is None for f in active)


@pytest.mark.asyncio
async def test_reflection_writes_tip_artifact(async_pool):
    """ac-2: reflection hook writes structured tip from trajectory outcome."""
    import json
    import uuid
    from datetime import UTC, datetime

    from archivist.core.trajectory import _ensure_trajectory_schema, search_tips
    from archivist.lifecycle.reflection import reflect_from_trajectory
    from archivist.storage.sqlite_pool import pool

    _ensure_trajectory_schema()
    tid = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    async with pool.write() as conn:
        await conn.execute(
            """INSERT INTO trajectories
               (id, agent_id, session_id, task_description, actions, outcome, outcome_score, created_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                tid,
                "agent-reflect",
                "sess-1",
                "roll out canary",
                json.dumps([{"action": "deploy"}]),
                "success",
                0.95,
                now,
                json.dumps({"namespace": "ops"}),
            ),
        )

    art = await reflect_from_trajectory(tid, namespace="ops", dry_run=False)
    assert art is not None
    assert art.applied is True
    assert art.tip_id
    tips = await search_tips("agent-reflect", category="reflection")
    assert any(t["id"] == art.tip_id for t in tips)
    assert "canary" in tips[0]["tip_text"]

    # Idempotent second call
    again = await reflect_from_trajectory(tid, namespace="ops", dry_run=False)
    assert again is not None
    assert again.metadata.get("idempotent_skip") is True
    assert again.tip_id == art.tip_id


@pytest.mark.asyncio
async def test_resolve_cycle_noop_when_disabled(async_pool, monkeypatch):
    import archivist.lifecycle.contradiction_resolve as cr

    monkeypatch.setattr(cr, "CONTRADICTION_RESOLVE_ENABLED", False)
    out = await cr.resolve_contradictions_cycle()
    assert out["enabled"] is False
    assert out["proposed"] == 0
