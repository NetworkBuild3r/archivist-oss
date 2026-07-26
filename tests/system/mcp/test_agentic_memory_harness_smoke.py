"""INIT-006/SPEC-002 — smoke: agentic_memory harness under SQLite + fake embed.

No live Qdrant. Marker is sibling to coach_core (ADR-006 GR-COACH-001).
"""

from __future__ import annotations

import pytest
from tests.system.mcp import agentic_memory_harness as harness

pytestmark = [
    pytest.mark.system,
    pytest.mark.mcp,
    pytest.mark.agentic_memory,
]


class TestAgenticMemoryHarnessSmoke:
    def test_choose_action_empty_refuses(self):
        assert harness.choose_action([]) == harness.ACTION_REFUSE
        assert harness.choose_action(None) == harness.ACTION_REFUSE
        assert harness.choose_action(context={"memories": []}) == harness.ACTION_REFUSE

    def test_choose_action_express_cue(self):
        memories = [
            {
                "id": "m1",
                "text": f"Customer is {harness.EXPRESS_ELIGIBLE_CUE} for replacement",
                "provenance": {"source": "harness"},
            }
        ]
        assert harness.choose_action(memories) == harness.ACTION_ORDER_EXPRESS

    def test_choose_action_ambiguous_needs_clarification(self):
        memories = [{"id": "m2", "text": "Customer mentioned a package once.", "provenance": {}}]
        assert harness.choose_action(memories) == harness.ACTION_NEEDS_CLARIFICATION

    def test_namespace_fail_closed_unknown_agent(self):
        assert harness.namespace_for_agent("not-a-synthetic-agent") == ""
        with pytest.raises(ValueError, match="fail-closed"):
            harness.AgenticSession.from_agent("not-a-synthetic-agent")

    async def test_store_under_sqlite_fake_embed_no_live_qdrant(self, qa_pool, monkeypatch):
        """ac-3: store via harness completes without live Qdrant."""
        harness.configure_agentic_store_flags(monkeypatch)
        session = harness.AgenticSession.from_agent("agentic-agent")
        body = f"Return policy note: {harness.EXPRESS_ELIGIBLE_CUE} within 7 days"

        data = await harness.store_memory(
            text=body,
            session=session,
            entities=["ReturnPolicy"],
            provenance={
                "source": "harness",
                "subject": "returns",
                "purpose": "agentic_eval",
                "sensitivity": "standard",
                "statement_kind": "user",
            },
        )
        assert data.get("stored") is True, data
        mid = data["memory_id"]
        assert mid

        # Oracle sees provenance-bearing memory text (not index TOC).
        action = harness.choose_action(
            [{"id": mid, "text": body, "provenance": {"source": "harness"}}]
        )
        assert action == harness.ACTION_ORDER_EXPRESS

        # Confirm row landed in SQLite (durable graph path).
        async with qa_pool.read() as conn:
            cur = await conn.execute(
                "SELECT qdrant_id, text, namespace FROM memory_chunks WHERE qdrant_id = ?",
                (mid,),
            )
            row = await cur.fetchone()
        assert row is not None
        assert dict(row)["namespace"] == session.namespace
        assert harness.EXPRESS_ELIGIBLE_CUE in dict(row)["text"]
