"""INIT-010/SPEC-004 — agentic Memory→Action after relevance forget (Diff #6).

Self-curation spirit: cold/stale EXPRESS_ELIGIBLE evidence suppressed by the
product forget path must not unlock ``order_express`` (cite-or-refuse / empty-OK).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from tests.system.mcp import agentic_memory_harness as harness

pytestmark = [
    pytest.mark.system,
    pytest.mark.mcp,
    pytest.mark.agentic_memory,
    pytest.mark.lifecycle,
]

_ENTITY = "ReturnEligibilityEntityCurate010"
_PROVENANCE = {
    "source": "harness",
    "subject": "returns",
    "purpose": "agentic_self_curation",
    "sensitivity": "standard",
    "statement_kind": "user",
    "confidence": 0.4,
}


def _eligible_text() -> str:
    return (
        f"{_ENTITY}: stale note — customer is {harness.EXPRESS_ELIGIBLE_CUE} "
        "(synthetic relevance-forget fixture)."
    )


class TestAgenticSelfCurationForget:
    async def test_relevance_forget_suppress_blocks_order_express(self, qa_pool, monkeypatch):
        """Forget apply → suppressed chunk → choose_action must not order_express."""
        import archivist.lifecycle.relevance_forget as forget_mod
        from archivist.lifecycle.relevance_forget import ForgetProposal
        from archivist.storage.sqlite_pool import pool

        harness.configure_agentic_store_flags(monkeypatch)
        session = harness.AgenticSession.from_agent("agentic-agent")

        store = await harness.store_memory(
            text=_eligible_text(),
            session=session,
            entities=[_ENTITY],
            provenance=_PROVENANCE,
        )
        mid = store["memory_id"]
        row = await harness.load_chunk(qa_pool, mid)
        assert row is not None

        # Age + cold hotness so product forget would select it in a full cycle.
        old = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        now = datetime.now(UTC).isoformat()
        async with pool.write() as conn:
            await conn.execute(
                "UPDATE memory_chunks SET created_at=?, importance=? WHERE qdrant_id=?",
                (old, 0.2, mid),
            )
            await conn.execute(
                "INSERT INTO memory_hotness "
                "(memory_id, score, retrieval_count, last_accessed, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(memory_id) DO UPDATE SET score=excluded.score",
                (mid, 0.0, 0, old, now),
            )

        proposal = ForgetProposal(
            memory_id=mid,
            namespace=session.namespace,
            agent_id=session.agent_id,
            hotness=0.0,
            importance=0.2,
            rule="cold_low_importance",
        )
        with patch(
            "archivist.lifecycle.correct._best_effort_qdrant_payload",
            new=AsyncMock(return_value=[]),
        ):
            with patch(
                "archivist.lifecycle.relevance_forget._qdrant_retention_class",
                new=AsyncMock(return_value=None),
            ):
                applied = await forget_mod.apply_relevance_forget(proposal, dry_run=False)
        assert applied.applied is True

        row_after = await harness.load_chunk(qa_pool, mid)
        assert row_after is not None
        assert int(row_after.get("is_suppressed") or 0) == 1

        hit = harness.row_to_hit(row_after)
        ctx = await harness.get_context_with_hits(
            session=session,
            hits=[hit],
            task_description="Should we order express replacement?",
        )
        memories = ctx.get("memories") or []
        action = harness.choose_action(memories)
        assert action != harness.ACTION_ORDER_EXPRESS
        assert action in {harness.ACTION_REFUSE, harness.ACTION_NEEDS_CLARIFICATION}
