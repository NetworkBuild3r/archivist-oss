"""INIT-007/SPEC-004 — procedure→action agentic_memory scenarios (ADR-007).

Tips from get_context must change discrete action selection. Memories / index
TOC alone must not unlock procedure actions when require_tip_evidence=True.
"""

from __future__ import annotations

import pytest
from tests.system.mcp import agentic_memory_harness as harness

pytestmark = [
    pytest.mark.system,
    pytest.mark.mcp,
    pytest.mark.agentic_memory,
]

_TASK = "Should we order express replacement using procedure tips?"
_PROCEDURE_TIP = (
    f"When rush replacement is approved, apply {harness.PROCEDURE_EXPRESS_CUE} "
    "shipping path (synthetic procedure eval fixture)."
)
_IRRELEVANT_TIP = "Rotate API keys monthly on the ops calendar (synthetic)."
_MEMORY_WITH_ELIGIBLE = (
    f"Customer note: {harness.EXPRESS_ELIGIBLE_CUE} for express — "
    "must NOT unlock procedure action without tip evidence."
)


class TestAgenticMemoryProcedureAction:
    async def test_tips_present_orders_express(self, qa_pool, monkeypatch):
        """ac-1: seeded tip surfaces via get_context → order_express."""
        harness.configure_agentic_store_flags(monkeypatch)
        session = harness.AgenticSession.from_agent("agentic-agent-a")

        tip_id = await harness.seed_tip(
            session=session,
            tip_text=_PROCEDURE_TIP,
            category="strategy",
            context="express shipping rush replacement",
        )
        assert tip_id

        ctx = await harness.get_context_with_hits(
            session=session,
            hits=[],
            task_description=_TASK,
            include_tips=True,
        )
        tips = ctx.get("tips") or []
        assert any(harness.PROCEDURE_EXPRESS_CUE in t for t in tips if isinstance(t, str)), ctx

        action = harness.choose_action(context=ctx, require_tip_evidence=True)
        assert action == harness.ACTION_ORDER_EXPRESS

    async def test_omit_tips_refuses(self, qa_pool, monkeypatch):
        """ac-2: no tip seed → refuse even if memories contain EXPRESS_ELIGIBLE."""
        harness.configure_agentic_store_flags(monkeypatch)
        session = harness.AgenticSession.from_agent("agentic-agent-a")

        # Memory-only eligible cue must not substitute for tip evidence.
        hit = {
            "id": "mem-proc-eligible-only",
            "text": _MEMORY_WITH_ELIGIBLE,
            "score": 0.95,
            "namespace": session.namespace,
            "agent_id": session.agent_id,
            "subject": "returns",
            "purpose": "agentic_eval",
            "sensitivity": "standard",
            "source": "harness",
            "statement_kind": "user",
            "confidence": 0.9,
            "is_suppressed": False,
        }
        ctx = await harness.get_context_with_hits(
            session=session,
            hits=[hit],
            task_description=_TASK,
            include_tips=True,
        )
        tips = ctx.get("tips") or []
        assert not any(harness.PROCEDURE_EXPRESS_CUE in t for t in tips if isinstance(t, str)), tips

        action = harness.choose_action(context=ctx, require_tip_evidence=True)
        assert action == harness.ACTION_REFUSE
        assert action != harness.ACTION_ORDER_EXPRESS

        # Sanity: memory-only path without require_tip_evidence still sees eligible
        mem_action = harness.choose_action(context=ctx, require_tip_evidence=False)
        assert mem_action == harness.ACTION_ORDER_EXPRESS

    async def test_archived_tip_refuses(self, qa_pool, monkeypatch):
        """ac-2 variant: archived tip is omitted by search_tips → refuse."""
        harness.configure_agentic_store_flags(monkeypatch)
        session = harness.AgenticSession.from_agent("agentic-agent-a")

        await harness.seed_tip(
            session=session,
            tip_text=_PROCEDURE_TIP,
            category="strategy",
            context="express shipping",
            archived=1,
        )
        ctx = await harness.get_context_with_hits(
            session=session,
            hits=[],
            task_description=_TASK,
            include_tips=True,
        )
        tips = ctx.get("tips") or []
        assert tips == [] or not any(
            harness.PROCEDURE_EXPRESS_CUE in t for t in tips if isinstance(t, str)
        )
        action = harness.choose_action(context=ctx, require_tip_evidence=True)
        assert action == harness.ACTION_REFUSE

    async def test_irrelevant_tip_alone_refuses(self, qa_pool, monkeypatch):
        """Optional SPEC-003 tie-in: irrelevant tip lacks PROCEDURE cue → refuse."""
        harness.configure_agentic_store_flags(monkeypatch)
        session = harness.AgenticSession.from_agent("agentic-agent-a")

        await harness.seed_tip(
            session=session,
            tip_text=_IRRELEVANT_TIP,
            category="optimization",
            context="api keys",
        )
        ctx = await harness.get_context_with_hits(
            session=session,
            hits=[],
            task_description=_TASK,
            include_tips=True,
        )
        action = harness.choose_action(context=ctx, require_tip_evidence=True)
        assert action == harness.ACTION_REFUSE

    async def test_cross_agent_tip_does_not_leak(self, qa_pool, monkeypatch):
        """Security AC: tip for agent-b does not unlock agent-a procedure action."""
        harness.configure_agentic_store_flags(monkeypatch)
        session_a = harness.AgenticSession.from_agent("agentic-agent-a")
        session_b = harness.AgenticSession.from_agent("agentic-agent-b")

        await harness.seed_tip(
            session=session_b,
            tip_text=_PROCEDURE_TIP,
            category="strategy",
            context="express shipping rush",
        )
        ctx = await harness.get_context_with_hits(
            session=session_a,
            hits=[],
            task_description=_TASK,
            include_tips=True,
        )
        tips = ctx.get("tips") or []
        assert not any(harness.PROCEDURE_EXPRESS_CUE in t for t in tips if isinstance(t, str)), tips
        action = harness.choose_action(context=ctx, require_tip_evidence=True)
        assert action == harness.ACTION_REFUSE
