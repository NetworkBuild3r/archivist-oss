"""Unit tests for memory_share_grants state machine (INIT-001/SPEC-010)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]


@pytest.mark.asyncio
async def test_create_requires_memory_ids_or_scope_or_tip_ids(async_pool):
    from archivist.storage import share_grants as sg

    with pytest.raises(ValueError, match="memory_ids, scope, or tip_ids"):
        await sg.create_share_grant(
            proposer_agent_id="a",
            recipient_agent_id="b",
            namespace="ns",
        )


@pytest.mark.asyncio
async def test_create_accepts_tip_ids_metadata_only(async_pool):
    """INIT-009: tip_ids alone are a valid selective-share target."""
    from archivist.storage import share_grants as sg

    grant = await sg.create_share_grant(
        proposer_agent_id="agent-a",
        recipient_agent_id="agent-b",
        namespace="ns-a",
        memory_ids=[],
        metadata={"tip_ids": ["tip-1"], "lesson_channel": "tips"},
    )
    assert grant.status == "pending"
    assert grant.memory_ids == []
    assert grant.metadata["tip_ids"] == ["tip-1"]


@pytest.mark.asyncio
async def test_propose_accept_idempotent(async_pool):
    from archivist.storage import share_grants as sg

    grant = await sg.create_share_grant(
        proposer_agent_id="agent-a",
        recipient_agent_id="agent-b",
        namespace="ns-a",
        memory_ids=["m1", "m2"],
        scope="s1",
    )
    assert grant.status == "pending"

    accepted = await sg.decide_share_grant(
        grant.id, namespace="ns-a", status="accepted", decided_by="agent-b"
    )
    assert accepted is not None
    assert accepted.status == "accepted"
    assert accepted.decided_by == "agent-b"

    again = await sg.decide_share_grant(
        grant.id, namespace="ns-a", status="accepted", decided_by="agent-b"
    )
    assert again is not None
    assert again.status == "accepted"
    assert again.decided_at == accepted.decided_at


@pytest.mark.asyncio
async def test_reject_then_cannot_accept(async_pool):
    from archivist.storage import share_grants as sg

    grant = await sg.create_share_grant(
        proposer_agent_id="agent-a",
        recipient_agent_id="agent-b",
        namespace="ns-a",
        memory_ids=["m1"],
    )
    rejected = await sg.decide_share_grant(
        grant.id, namespace="ns-a", status="rejected", decided_by="agent-b"
    )
    assert rejected is not None
    assert rejected.status == "rejected"

    with pytest.raises(ValueError, match="already rejected"):
        await sg.decide_share_grant(
            grant.id, namespace="ns-a", status="accepted", decided_by="agent-b"
        )


@pytest.mark.asyncio
async def test_attach_conflict_outcome_spec006_actions(async_pool):
    from archivist.storage import share_grants as sg

    grant = await sg.create_share_grant(
        proposer_agent_id="agent-a",
        recipient_agent_id="agent-b",
        namespace="ns-a",
        scope="conflict-scope",
    )
    updated = await sg.attach_conflict_outcome(
        grant.id,
        namespace="ns-a",
        outcome={"action": "supersede", "winner_fact_id": 1, "loser_fact_id": 2},
    )
    assert updated is not None
    assert updated.conflict_outcome is not None
    assert updated.conflict_outcome["action"] == "supersede"


@pytest.mark.asyncio
async def test_get_scoped_by_namespace(async_pool):
    from archivist.storage import share_grants as sg

    grant = await sg.create_share_grant(
        proposer_agent_id="agent-a",
        recipient_agent_id="agent-b",
        namespace="ns-a",
        memory_ids=["m1"],
    )
    assert await sg.get_share_grant(grant.id, namespace="ns-a") is not None
    assert await sg.get_share_grant(grant.id, namespace="other-ns") is None
