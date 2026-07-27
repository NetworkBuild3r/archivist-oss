"""INIT-009/SPEC-003 — multi-agent tip share / handoff lesson scenarios.

Proves Diff #5 lesson path: tips travel via handoff to unlock procedure
action for the recipient; selective share tip_ids inject into SessionStore.
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

_TASK = "Should we order express replacement using procedure tips?"
_PROCEDURE_TIP = (
    f"When rush replacement is approved, apply {harness.PROCEDURE_EXPRESS_CUE} "
    "shipping path (synthetic multi-agent tip-share fixture)."
)


class TestAgenticMultiAgentTipCoordination:
    async def test_tip_via_handoff_unlocks_recipient_procedure(self, qa_pool, monkeypatch):
        """ac-4: handoff carries procedure tip text → recipient choose_action express."""
        from archivist.retrieval.context_api import HandoffPacket, receive_handoff_packet
        from archivist.retrieval.session_store import SessionStore

        harness.configure_agentic_store_flags(monkeypatch)
        session_a = harness.AgenticSession.from_agent("agentic-agent-a")
        session_b = harness.AgenticSession.from_agent("agentic-agent-b")

        # Seed tip on sender (honesty: tip exists for A before handoff).
        tip_id = await harness.seed_tip(
            session=session_a,
            tip_text=_PROCEDURE_TIP,
            category="strategy",
            context="express shipping rush replacement",
        )
        assert tip_id

        pkt = HandoffPacket(
            from_agent=session_a.agent_id,
            to_agent=session_b.agent_id,
            session_summary="Handing off rush replacement procedure",
            active_goals=["complete express path"],
            open_questions=[_PROCEDURE_TIP],
            key_memory_ids=[],
            knowledge_snapshot={},
            token_count=40,
            created_at="2026-07-26T00:00:00+00:00",
        )
        store = SessionStore()
        with patch("archivist.retrieval.context_api.get_session_store", return_value=store):
            result = await receive_handoff_packet(pkt, session_b.agent_id, "sess-b")

        assert "handoff_recovery_0" in result["injected_keys"]
        recovered = store.get(session_b.agent_id, "sess-b", "handoff_recovery_0")
        assert harness.PROCEDURE_EXPRESS_CUE in (recovered or "")

        # Recipient uses handed-off tip strings as tip evidence (GR-HANDOFF-001).
        action = harness.choose_action(
            context={"tips": [recovered]},
            require_tip_evidence=True,
        )
        assert action == harness.ACTION_ORDER_EXPRESS

    async def test_share_tip_ids_propose_accept_round_trip(self, async_pool, monkeypatch):
        """ac-4: propose tip_ids → accept injects share_tip_ids for recipient.

        Uses ``async_pool`` (graph.init_schema includes share grants) rather than
        ``qa_pool``, whose build_schema may omit share DDL until a guard reset.
        """
        from archivist.app.handlers.tools_coordination import (
            _handle_share_accept,
            _handle_share_propose,
        )
        from archivist.retrieval.session_store import SessionStore
        from archivist.storage import share_grants as sg

        harness.configure_agentic_store_flags(monkeypatch)
        # Force share-grant DDL against this test's pool (schema_guard is sticky).
        sg._ensure_share_grants_schema.reset()
        sg._ensure_share_grants_schema()

        session_a = harness.AgenticSession.from_agent("agentic-agent-a")
        session_b = harness.AgenticSession(
            agent_id="agentic-agent-b",
            namespace=session_a.namespace,  # shared tenant for grant isolation
        )
        tip_id = "tip-proc-shared-001"

        with (
            patch("archivist.app.handlers.tools_coordination.require_caller", return_value=None),
            patch("archivist.app.handlers.tools_coordination.require_rbac", return_value=None),
        ):
            propose = await _handle_share_propose(
                {
                    "agent_id": session_a.agent_id,
                    "recipient_agent_id": session_b.agent_id,
                    "namespace": session_a.namespace,
                    "tip_ids": [str(tip_id)],
                    "reason": "share procedure tip ids",
                }
            )
        grant = json.loads(propose[0].text)["grant"]
        assert grant["metadata"]["tip_ids"] == [str(tip_id)]
        assert grant["metadata"]["lesson_channel"] == "tips"

        store = SessionStore()
        with (
            patch("archivist.app.handlers.tools_coordination.require_caller", return_value=None),
            patch("archivist.app.handlers.tools_coordination.require_rbac", return_value=None),
            patch(
                "archivist.retrieval.session_store.get_session_store",
                return_value=store,
            ),
        ):
            accepted = await _handle_share_accept(
                {
                    "agent_id": session_b.agent_id,
                    "grant_id": grant["id"],
                    "namespace": session_a.namespace,
                    "session_id": "sess-share-b",
                }
            )
        payload = json.loads(accepted[0].text)
        assert payload["grant"]["status"] == "accepted"
        assert "share_tip_ids" in payload["injected_keys"]
        raw = store.get(session_b.agent_id, "sess-share-b", "share_tip_ids")
        assert json.loads(raw) == [str(tip_id)]
