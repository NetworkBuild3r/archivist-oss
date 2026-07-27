"""Unit tests for Diff #6 reconsolidation product path (INIT-010/SPEC-002)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.lifecycle]


@pytest.mark.asyncio
async def test_cycle_disabled_is_noop():
    from archivist.lifecycle import reconsolidation as mod

    with patch.object(mod, "RECONSOLIDATION_ENABLED", False):
        out = await mod.reconsolidation_cycle()
    assert out["enabled"] is False
    assert out["proposed"] == 0
    assert out["applied"] == 0


@pytest.mark.asyncio
async def test_dry_run_proposes_and_audits_without_upsert():
    from archivist.lifecycle import reconsolidation as mod
    from archivist.lifecycle.reconsolidation import ReconsolidationProposal

    groups = [
        {
            "namespace": "ns-a",
            "agent_id": "agent-a",
            "chunks": [
                {"qdrant_id": "c1", "text": "alpha " * 40},
                {"qdrant_id": "c2", "text": "beta " * 40},
                {"qdrant_id": "c3", "text": "gamma " * 40},
            ],
        }
    ]
    audit = AsyncMock()
    upsert = AsyncMock()
    with (
        patch.object(mod, "RECONSOLIDATION_ENABLED", True),
        patch.object(mod, "RECONSOLIDATION_DRY_RUN", True),
        patch.object(mod, "RECONSOLIDATION_MAX_GROUPS_PER_CYCLE", 5),
        patch.object(mod, "RECONSOLIDATION_MIN_CHUNKS", 3),
        patch.object(mod, "RECONSOLIDATION_MAX_CHUNKS_PER_GROUP", 8),
        patch.object(mod, "_candidate_groups", new=AsyncMock(return_value=groups)),
        patch.object(mod, "_summarize_group", new=AsyncMock(return_value="L1 overview summary")),
        patch("archivist.lifecycle.reconsolidation.log_memory_event", new=audit),
        patch("archivist.storage.graph_fts.upsert_fts_chunk", new=upsert),
    ):
        out = await mod.reconsolidation_cycle()

    assert out["enabled"] is True
    assert out["proposed"] == 1
    assert out["applied"] == 0
    assert out["dry_run"] is True
    assert out["results"][0]["namespace"] == "ns-a"
    assert out["results"][0]["agent_id"] == "agent-a"
    audit.assert_awaited()
    assert audit.await_args.kwargs["action"] == "reconsolidation_proposed"
    assert audit.await_args.kwargs["namespace"] == "ns-a"
    upsert.assert_not_awaited()
    # type sanity
    assert isinstance(
        ReconsolidationProposal(
            namespace="ns-a", agent_id="agent-a", source_qdrant_ids=["c1"]
        ).to_dict()["source_qdrant_ids"],
        list,
    )


@pytest.mark.asyncio
async def test_apply_writes_l1_chunk_same_agent_namespace():
    from archivist.lifecycle import reconsolidation as mod
    from archivist.lifecycle.reconsolidation import ReconsolidationProposal

    proposal = ReconsolidationProposal(
        namespace="ns-b",
        agent_id="agent-b",
        source_qdrant_ids=["x1", "x2", "x3"],
        summary_text="Consolidated overview of three memories.",
    )
    upsert = AsyncMock()
    audit = AsyncMock()
    with (
        patch.object(mod, "RECONSOLIDATION_DRY_RUN", False),
        patch("archivist.storage.graph_fts.upsert_fts_chunk", new=upsert),
        patch("archivist.lifecycle.reconsolidation.log_memory_event", new=audit),
    ):
        out = await mod.apply_reconsolidation(proposal, dry_run=False)

    assert out.applied is True
    assert out.dry_run is False
    upsert.assert_awaited_once()
    kwargs = upsert.await_args.kwargs
    assert kwargs["agent_id"] == "agent-b"
    assert kwargs["namespace"] == "ns-b"
    assert kwargs["tier_label"] == "l1"
    assert kwargs["memory_type"] == "reconsolidation"
    assert kwargs["text"].startswith("Consolidated")
    assert audit.await_args.kwargs["action"] == "reconsolidation_applied"


@pytest.mark.asyncio
async def test_curator_loop_invokes_reconsolidation_cycle():
    """ac-1: curator product loop wires reconsolidation_cycle."""
    from archivist.lifecycle import curator as curator_mod

    recon = AsyncMock(return_value={"enabled": True, "proposed": 1, "applied": 0, "dry_run": True})
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
        patch(
            "archivist.lifecycle.reconsolidation.reconsolidation_cycle",
            new=recon,
        ),
        patch(
            "archivist.lifecycle.contradiction_resolve.resolve_contradictions_cycle",
            new=AsyncMock(return_value={"enabled": False}),
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

    recon.assert_awaited_once()


def test_no_core_mcp_tools_added_for_reconsolidation():
    """ac-3: reconsolidation is lifecycle-flag driven, not a new core MCP tool."""
    from archivist.app.handlers._registry import CORE_TOOL_NAMES

    assert not any("reconsolidat" in n for n in CORE_TOOL_NAMES)
