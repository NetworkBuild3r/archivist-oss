"""Unit tests for Diff #6 relevance forget + resolve product path (INIT-010/SPEC-003)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import TextContent

pytestmark = [pytest.mark.unit, pytest.mark.lifecycle]


def _parse(result) -> dict:
    assert isinstance(result, list) and result
    return json.loads(result[0].text)


def _grant(**overrides):
    base = {
        "id": "grant-1",
        "proposer_agent_id": "agent-a",
        "recipient_agent_id": "agent-b",
        "namespace": "ns-a",
        "memory_ids": ["m1", "m2"],
        "scope": "scope-v1",
        "status": "pending",
        "conflict_outcome": None,
        "reason": "need context",
        "metadata": {},
        "created_at": "2026-07-25T00:00:00+00:00",
        "decided_at": None,
        "decided_by": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_forget_cycle_disabled_is_noop():
    from archivist.lifecycle import relevance_forget as mod

    with patch.object(mod, "RELEVANCE_FORGET_ENABLED", False):
        out = await mod.relevance_forget_cycle()
    assert out["enabled"] is False
    assert out["proposed"] == 0
    assert out["applied"] == 0


@pytest.mark.asyncio
async def test_forget_dry_run_proposes_and_audits_without_suppress():
    from archivist.lifecycle import relevance_forget as mod
    from archivist.lifecycle.relevance_forget import ForgetProposal

    candidates = [
        ForgetProposal(
            memory_id="mem-cold-1",
            namespace="ns-a",
            agent_id="agent-a",
            hotness=0.01,
            importance=0.2,
        )
    ]
    audit = AsyncMock()
    suppress = AsyncMock()
    with (
        patch.object(mod, "RELEVANCE_FORGET_ENABLED", True),
        patch.object(mod, "RELEVANCE_FORGET_DRY_RUN", True),
        patch.object(mod, "_candidate_chunks", AsyncMock(return_value=candidates)),
        patch.object(mod, "log_memory_event", audit),
        patch("archivist.lifecycle.correct.suppress_memory", suppress),
        patch.object(mod, "_qdrant_retention_class", AsyncMock(return_value=None)),
    ):
        out = await mod.relevance_forget_cycle(dry_run=True)

    assert out["enabled"] is True
    assert out["proposed"] == 1
    assert out["applied"] == 0
    assert out["dry_run"] is True
    suppress.assert_not_awaited()
    assert audit.await_count == 1
    assert audit.await_args.kwargs["action"] == "relevance_forget_proposed"
    assert audit.await_args.kwargs["namespace"] == "ns-a"


@pytest.mark.asyncio
async def test_forget_skips_high_importance_pin_floor():
    """SEC-010-01: importance >= 0.9 is protected (archivist_pin sync)."""
    from archivist.lifecycle import relevance_forget as mod
    from archivist.lifecycle.relevance_forget import ForgetProposal

    proposal = ForgetProposal(
        memory_id="mem-pinned",
        namespace="ns-p",
        agent_id="agent-p",
        hotness=0.0,
        importance=1.0,
    )
    audit = AsyncMock()
    suppress = AsyncMock()
    with (
        patch.object(mod, "log_memory_event", audit),
        patch("archivist.lifecycle.correct.suppress_memory", suppress),
        patch.object(mod, "_qdrant_retention_class", AsyncMock(return_value=None)),
    ):
        out = await mod.apply_relevance_forget(proposal, dry_run=False)

    assert out.applied is False
    assert out.metadata.get("skipped") == "high_importance"
    suppress.assert_not_awaited()
    assert audit.await_args.kwargs["action"] == "relevance_forget_skipped"


@pytest.mark.asyncio
async def test_forget_skips_qdrant_permanent_retention():
    """SEC-010-01: Qdrant retention_class=permanent blocks suppress."""
    from archivist.lifecycle import relevance_forget as mod
    from archivist.lifecycle.relevance_forget import ForgetProposal

    proposal = ForgetProposal(
        memory_id="mem-qpin",
        namespace="ns-q",
        agent_id="agent-q",
        hotness=0.0,
        importance=0.1,
    )
    audit = AsyncMock()
    suppress = AsyncMock()
    with (
        patch.object(mod, "log_memory_event", audit),
        patch("archivist.lifecycle.correct.suppress_memory", suppress),
        patch.object(mod, "_qdrant_retention_class", AsyncMock(return_value="permanent")),
    ):
        out = await mod.apply_relevance_forget(proposal, dry_run=False)

    assert out.applied is False
    assert out.metadata.get("skipped") == "retention_permanent"
    suppress.assert_not_awaited()
    assert audit.await_args.kwargs["action"] == "relevance_forget_skipped"


@pytest.mark.asyncio
async def test_forget_apply_suppresses_same_namespace():
    from archivist.lifecycle import relevance_forget as mod
    from archivist.lifecycle.relevance_forget import ForgetProposal

    proposal = ForgetProposal(
        memory_id="mem-cold-2",
        namespace="ns-b",
        agent_id="agent-b",
        hotness=0.0,
        importance=0.1,
    )
    audit = AsyncMock()
    suppress = AsyncMock(return_value={"status": "suppressed"})
    with (
        patch.object(mod, "RELEVANCE_FORGET_DRY_RUN", False),
        patch.object(mod, "log_memory_event", audit),
        patch("archivist.lifecycle.correct.suppress_memory", suppress),
        patch.object(mod, "_qdrant_retention_class", AsyncMock(return_value=None)),
    ):
        out = await mod.apply_relevance_forget(proposal, dry_run=False)

    assert out.applied is True
    suppress.assert_awaited_once()
    args, kwargs = suppress.await_args
    assert args[0] == "mem-cold-2"
    assert args[1] == "ns-b"
    assert kwargs["agent_id"] == "agent-b"
    assert audit.await_args.kwargs["action"] == "relevance_forget_applied"


@pytest.mark.asyncio
async def test_forget_passes_proposal_namespace_only():
    """Suppress always uses proposal.namespace — no cross-ns rewrite."""
    from archivist.lifecycle import relevance_forget as mod
    from archivist.lifecycle.relevance_forget import ForgetProposal

    proposal = ForgetProposal(
        memory_id="mem-x",
        namespace="ns-owner",
        agent_id="agent-x",
        hotness=0.0,
        importance=0.1,
    )
    suppress = AsyncMock(return_value={"status": "suppressed", "rows_updated": 0})
    with (
        patch.object(mod, "log_memory_event", AsyncMock()),
        patch("archivist.lifecycle.correct.suppress_memory", suppress),
        patch.object(mod, "_qdrant_retention_class", AsyncMock(return_value=None)),
    ):
        await mod.apply_relevance_forget(proposal, dry_run=False)

    assert suppress.await_args.args[1] == "ns-owner"


@pytest.mark.asyncio
async def test_curator_loop_invokes_forget_and_resolve():
    from archivist.lifecycle import curator as curator_mod

    forget = AsyncMock(return_value={"enabled": False, "proposed": 0, "applied": 0})
    resolve = AsyncMock(
        return_value={"enabled": True, "proposed": 1, "applied": 0, "dry_run": True}
    )
    with (
        patch.object(curator_mod, "CURATOR_INTERVAL_MINUTES", 60),
        patch.object(
            curator_mod,
            "curate_cycle",
            new=AsyncMock(return_value={"processed": 0, "skipped": 0}),
        ),
        patch.object(curator_mod, "reinforce_durable_entities", new=AsyncMock()),
        patch.object(
            curator_mod,
            "decay_old_entries",
            new=AsyncMock(return_value={"total": 0, "aged_out": 0, "superseded_out": 0}),
        ),
        patch.object(curator_mod, "batch_update_hotness", new=AsyncMock(return_value=0)),
        patch.object(
            curator_mod,
            "consolidate_tips",
            new=AsyncMock(return_value={"consolidated": 0}),
        ),
        patch("archivist.lifecycle.relevance_forget.relevance_forget_cycle", new=forget),
        patch(
            "archivist.lifecycle.reconsolidation.reconsolidation_cycle",
            new=AsyncMock(return_value={"enabled": False}),
        ),
        patch(
            "archivist.lifecycle.contradiction_resolve.resolve_contradictions_cycle",
            new=resolve,
        ),
        patch(
            "archivist.lifecycle.reflection.reflection_cycle",
            new=AsyncMock(return_value={"enabled": False}),
        ),
        patch.object(curator_mod, "_refresh_wake_up_caches", new=AsyncMock(return_value=0)),
        patch.object(curator_mod.asyncio, "sleep", new=AsyncMock(side_effect=StopAsyncIteration)),
    ):
        with pytest.raises(StopAsyncIteration):
            await curator_mod.curator_loop()

    forget.assert_awaited_once()
    resolve.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_cycle_respects_adr_defaults(monkeypatch):
    """ADR-010: ENABLED=false by default; DRY_RUN=true when enabled."""
    import archivist.lifecycle.contradiction_resolve as cr

    monkeypatch.setattr(cr, "CONTRADICTION_RESOLVE_ENABLED", False)
    out = await cr.resolve_contradictions_cycle()
    assert out["enabled"] is False
    assert out["dry_run"] is True

    monkeypatch.setattr(cr, "CONTRADICTION_RESOLVE_ENABLED", True)
    monkeypatch.setattr(cr, "CONTRADICTION_RESOLVE_DRY_RUN", True)
    monkeypatch.setattr(cr, "_candidate_entity_ids", AsyncMock(return_value=[]))
    out2 = await cr.resolve_contradictions_cycle()
    assert out2["enabled"] is True
    assert out2["dry_run"] is True
    assert out2["applied"] == 0


@pytest.mark.asyncio
async def test_share_attach_mutating_requires_resolve_enabled():
    """ac-3 / GR-SHARE-001: SEC-009-04 still gates mutating apply."""
    from archivist.app.handlers.tools_coordination import _handle_share_attach_conflict

    pending = _grant()
    with (
        patch("archivist.app.handlers.tools_coordination.require_caller", return_value=None),
        patch("archivist.app.handlers.tools_coordination.require_rbac", return_value=None),
        patch("archivist.core.config.CONTRADICTION_RESOLVE_ENABLED", False),
        patch(
            "archivist.storage.share_grants.get_share_grant",
            new=AsyncMock(return_value=pending),
        ),
        patch(
            "archivist.storage.share_grants.attach_conflict_outcome",
            new=AsyncMock(),
        ) as attach,
        patch(
            "archivist.lifecycle.contradiction_resolve.apply_resolution",
            new=AsyncMock(),
        ) as apply_mock,
    ):
        result = await _handle_share_attach_conflict(
            {
                "agent_id": "agent-a",
                "grant_id": "grant-1",
                "namespace": "ns-a",
                "action": "supersede",
                "apply": True,
                "dry_run": False,
                "entity_id": 7,
                "winner_fact_id": 1,
                "loser_fact_id": 2,
            }
        )
    assert _parse(result)["error"] == "resolve_disabled"
    attach.assert_not_awaited()
    apply_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_share_attach_mutating_requires_write_rbac():
    """ac-3: mutating apply still requires namespace write (SEC-009-01)."""
    from archivist.app.handlers.tools_coordination import _handle_share_attach_conflict

    pending = _grant()
    denied = [TextContent(type="text", text=json.dumps({"error": "access_denied"}))]

    def _rbac(_caller, action, _ns):
        if action == "write":
            return denied
        return None

    with (
        patch("archivist.app.handlers.tools_coordination.require_caller", return_value=None),
        patch(
            "archivist.app.handlers.tools_coordination.require_rbac",
            side_effect=_rbac,
        ),
        patch("archivist.core.config.CONTRADICTION_RESOLVE_ENABLED", True),
        patch(
            "archivist.storage.share_grants.get_share_grant",
            new=AsyncMock(return_value=pending),
        ),
        patch(
            "archivist.storage.share_grants.attach_conflict_outcome",
            new=AsyncMock(),
        ) as attach,
        patch(
            "archivist.lifecycle.contradiction_resolve.apply_resolution",
            new=AsyncMock(),
        ) as apply_mock,
    ):
        result = await _handle_share_attach_conflict(
            {
                "agent_id": "agent-a",
                "grant_id": "grant-1",
                "namespace": "ns-a",
                "action": "supersede",
                "apply": True,
                "dry_run": False,
                "entity_id": 7,
                "winner_fact_id": 1,
                "loser_fact_id": 2,
            }
        )
    assert _parse(result)["error"] == "access_denied"
    attach.assert_not_awaited()
    apply_mock.assert_not_awaited()
