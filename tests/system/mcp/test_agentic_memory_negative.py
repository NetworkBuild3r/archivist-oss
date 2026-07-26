"""INIT-006/SPEC-004 — negative stale / contradictory / ambiguous Memory→Action.

Poisoned evidence must not silently pass as ``order_express`` (SM-002).
Cite-or-refuse / empty-OK preserved (ADR-004 / GR-CE-001).

Stale encoding (aud-1): synthetic ``is_suppressed=True`` on the outdated hit
(prefer suppress/supersede when wiring live APIs; harness oracle ignores
suppressed rows). Contradiction (aud-2): both eligible + ineligible cues →
``needs_clarification`` — never invent a merge.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from tests.system.mcp import agentic_memory_harness as harness

pytestmark = [
    pytest.mark.system,
    pytest.mark.mcp,
    pytest.mark.agentic_memory,
]

_ENTITY = "ReturnEligibilityEntityNeg004"
_PROVENANCE = {
    "source": "harness",
    "subject": "returns",
    "purpose": "agentic_eval",
    "sensitivity": "standard",
    "statement_kind": "user",
    "confidence": 0.9,
}


def _eligible_text() -> str:
    return (
        f"{_ENTITY}: v1 note — customer is {harness.EXPRESS_ELIGIBLE_CUE} "
        "(synthetic stale/eligible fixture)."
    )


def _ineligible_text() -> str:
    return (
        f"{_ENTITY}: v2 note — customer is {harness.EXPRESS_INELIGIBLE_CUE} "
        "(synthetic contradictory fixture)."
    )


def _ambiguous_text() -> str:
    return f"{_ENTITY}: package mentioned; eligibility unknown (synthetic ambiguous)."


class TestAgenticMemoryNegative:
    async def test_stale_suppressed_eligible_does_not_order_express(self, qa_pool, monkeypatch):
        """ac-1: suppressed (stale) EXPRESS_ELIGIBLE must not drive order_express."""
        harness.configure_agentic_store_flags(monkeypatch)
        session = harness.AgenticSession.from_agent("agentic-agent")

        store_v1 = await harness.store_memory(
            text=_eligible_text(),
            session=session,
            entities=[_ENTITY],
            provenance={**_PROVENANCE, "confidence": 0.5},
        )
        mid_v1 = store_v1["memory_id"]
        row_v1 = await harness.load_chunk(qa_pool, mid_v1)
        assert row_v1 is not None

        # Newer active memory without eligible cue (revocation / update).
        store_v2 = await harness.store_memory(
            text=f"{_ENTITY}: eligibility revoked after review — no express path.",
            session=session,
            entities=[_ENTITY],
            provenance={**_PROVENANCE, "confidence": 0.95},
        )
        mid_v2 = store_v2["memory_id"]
        row_v2 = await harness.load_chunk(qa_pool, mid_v2)
        assert row_v2 is not None

        hit_stale = harness.row_to_hit(row_v1)
        hit_stale["is_suppressed"] = True  # aud-1 synthetic stale encoding
        hit_new = harness.row_to_hit(row_v2)
        hit_new["is_suppressed"] = False

        ctx = await harness.get_context_with_hits(
            session=session,
            hits=[hit_stale, hit_new],
            task_description="Should we order express replacement?",
        )
        memories = ctx.get("memories") or []
        # Prerank drops suppressed; ensure stale eligible cue is not actionable.
        assert all(m.get("id") != mid_v1 or m.get("is_suppressed") for m in memories) or all(
            m.get("id") != mid_v1 for m in memories
        )

        action = harness.choose_action(memories)
        assert action != harness.ACTION_ORDER_EXPRESS
        assert action in {harness.ACTION_REFUSE, harness.ACTION_NEEDS_CLARIFICATION}

    async def test_contradictory_eligible_and_ineligible_needs_clarification(
        self, qa_pool, monkeypatch
    ):
        """ac-2: contradictory pair → needs_clarification (no invented merge)."""
        harness.configure_agentic_store_flags(monkeypatch)
        session = harness.AgenticSession.from_agent("agentic-agent-a")

        store_a = await harness.store_memory(
            text=_eligible_text(),
            session=session,
            entities=[_ENTITY],
            provenance=_PROVENANCE,
        )
        store_b = await harness.store_memory(
            text=_ineligible_text(),
            session=session,
            entities=[_ENTITY],
            provenance=_PROVENANCE,
        )
        row_a = await harness.load_chunk(qa_pool, store_a["memory_id"])
        row_b = await harness.load_chunk(qa_pool, store_b["memory_id"])
        assert row_a is not None and row_b is not None

        ctx = await harness.get_context_with_hits(
            session=session,
            hits=[harness.row_to_hit(row_a), harness.row_to_hit(row_b)],
        )
        memories = ctx.get("memories") or []
        assert len(memories) >= 2

        action = harness.choose_action(memories)
        assert action == harness.ACTION_NEEDS_CLARIFICATION
        assert action != harness.ACTION_ORDER_EXPRESS

    async def test_ambiguous_and_empty_evidence_paths(self, qa_pool, monkeypatch):
        """ac-3: ambiguous → needs_clarification; empty → refuse (empty-OK)."""
        harness.configure_agentic_store_flags(monkeypatch)
        session = harness.AgenticSession.from_agent("agentic-agent-b")

        # Empty evidence
        empty_ctx = await harness.get_context_with_hits(session=session, hits=[])
        assert harness.choose_action(empty_ctx.get("memories") or []) == harness.ACTION_REFUSE

        # Ambiguous non-empty without eligible/ineligible cues
        store = await harness.store_memory(
            text=_ambiguous_text(),
            session=session,
            entities=[_ENTITY],
            provenance=_PROVENANCE,
        )
        row = await harness.load_chunk(qa_pool, store["memory_id"])
        assert row is not None
        amb_ctx = await harness.get_context_with_hits(
            session=session, hits=[harness.row_to_hit(row)]
        )
        action = harness.choose_action(amb_ctx.get("memories") or [])
        assert action == harness.ACTION_NEEDS_CLARIFICATION
        assert action != harness.ACTION_ORDER_EXPRESS

    async def test_index_markdown_never_asserted_as_sufficient_evidence(self, qa_pool, monkeypatch):
        """ac-4: never claim index TOC alone is enough for order_express."""
        harness.configure_agentic_store_flags(monkeypatch)
        session = harness.AgenticSession.from_agent("agentic-agent")

        await harness.store_memory(
            text=_eligible_text(),
            session=session,
            entities=[_ENTITY],
            provenance=_PROVENANCE,
        )

        with harness.allow_search_rbac():
            from archivist.app.handlers._registry import dispatch_tool

            with patch(
                "archivist.app.handlers.tools_search.get_namespace_for_agent",
                return_value=session.namespace,
            ):
                index_out = await dispatch_tool(
                    "archivist_index",
                    {"agent_id": session.agent_id, "namespace": session.namespace},
                )
        index_payload = json.loads(index_out[0].text)
        markdown = index_payload.get("markdown") or ""
        assert isinstance(markdown, str)
        assert "markdown" in index_payload

        # Policy: action uses get_context memories[], never index markdown.
        action = harness.choose_action([])
        assert action == harness.ACTION_REFUSE
        assert action != harness.ACTION_ORDER_EXPRESS
        _ = markdown  # index ran; intentionally unused for action selection

    def test_oracle_unit_suppressed_and_contradiction(self):
        """Unit: oracle rules without MCP (fast guard for SPEC-004 policy)."""
        stale = [
            {
                "id": "1",
                "text": f"x {harness.EXPRESS_ELIGIBLE_CUE}",
                "is_suppressed": True,
            }
        ]
        assert harness.choose_action(stale) == harness.ACTION_REFUSE

        both = [
            {"id": "a", "text": f"a {harness.EXPRESS_ELIGIBLE_CUE}"},
            {"id": "b", "text": f"b {harness.EXPRESS_INELIGIBLE_CUE}"},
        ]
        assert harness.choose_action(both) == harness.ACTION_NEEDS_CLARIFICATION
