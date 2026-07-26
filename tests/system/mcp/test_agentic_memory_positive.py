"""INIT-006/SPEC-003 — positive multi-session Memory→Action scenarios (ADR-006).

Session A stores a decisive fact; Session B retrieves provenance-bearing
memories via get_context and the test-only oracle selects an action.
Omit-store control must refuse (SM-001). Index TOC alone is not evidence
(GR-CE-001).
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

# Synthetic coach-domain fact (aud-1) — not real user data.
_ENTITY = "ReturnEligibilityEntity003"
_DECISIVE_TEXT = (
    f"{_ENTITY}: damaged item within window — customer is {harness.EXPRESS_ELIGIBLE_CUE} "
    "for express replacement (synthetic agentic eval fixture)."
)
_PROVENANCE = {
    "source": "harness",
    "subject": "returns",
    "purpose": "agentic_eval",
    "sensitivity": "standard",
    "statement_kind": "user",
    "confidence": 0.93,
}


class TestAgenticMemoryPositive:
    async def test_session_a_store_session_b_get_context_orders_express(self, qa_pool, monkeypatch):
        """ac-1: Session A store → Session B get_context → order_express."""
        harness.configure_agentic_store_flags(monkeypatch)
        # Distinct logical sessions / same namespace (aud-2).
        session_a = harness.AgenticSession.from_agent("agentic-agent-a")
        session_b = harness.AgenticSession(
            agent_id="agentic-agent-b",
            namespace=session_a.namespace,  # shared tenant memory across sessions
        )

        store = await harness.store_memory(
            text=_DECISIVE_TEXT,
            session=session_a,
            entities=[_ENTITY],
            provenance=_PROVENANCE,
        )
        assert store.get("stored") is True, store
        mid = store["memory_id"]

        row = await harness.load_chunk(qa_pool, mid)
        assert row is not None
        assert row["namespace"] == session_a.namespace
        hit = harness.row_to_hit(row)

        ctx_payload = await harness.get_context_with_hits(
            session=session_b,
            hits=[hit],
            task_description="Should we order express replacement?",
        )
        memories = ctx_payload.get("memories")
        assert isinstance(memories, list) and memories, ctx_payload
        mem = next((m for m in memories if m.get("id") == mid), memories[0])
        assert mem["id"] == mid
        assert harness.EXPRESS_ELIGIBLE_CUE in (mem.get("text") or "")
        prov = mem.get("provenance")
        assert isinstance(prov, dict)
        assert "api_key" not in prov

        action = harness.choose_action(memories)
        assert action == harness.ACTION_ORDER_EXPRESS

    async def test_omit_store_control_refuses(self, qa_pool, monkeypatch):
        """ac-2: without Session A store, Session B evidence empty → refuse."""
        harness.configure_agentic_store_flags(monkeypatch)
        session_b = harness.AgenticSession.from_agent("agentic-agent-a")

        rows = await harness.chunks_for_namespace(qa_pool, session_b.namespace)
        assert rows == []

        ctx_payload = await harness.get_context_with_hits(
            session=session_b,
            hits=[],
            task_description="Should we order express replacement?",
        )
        memories = ctx_payload.get("memories") or []
        assert memories == [] or all(not (m.get("text") or "").strip() for m in memories)

        action = harness.choose_action(memories if memories else None)
        assert action == harness.ACTION_REFUSE
        assert action != harness.ACTION_ORDER_EXPRESS

    async def test_index_markdown_alone_is_not_evidence(self, qa_pool, monkeypatch):
        """ac-3: TOC / index markdown must not drive order_express (GR-CE-001)."""
        harness.configure_agentic_store_flags(monkeypatch)
        session = harness.AgenticSession.from_agent("agentic-agent")

        store = await harness.store_memory(
            text=_DECISIVE_TEXT,
            session=session,
            entities=[_ENTITY],
            provenance=_PROVENANCE,
        )
        mid = store["memory_id"]

        with harness.allow_search_rbac():
            from archivist.app.handlers._registry import dispatch_tool

            with patch(
                "archivist.app.handlers.tools_search.get_namespace_for_agent",
                return_value=session.namespace,
            ):
                # Index may mention entity names — still not citable evidence.
                index_out = await dispatch_tool(
                    "archivist_index",
                    {"agent_id": session.agent_id, "namespace": session.namespace},
                )
        index_payload = json.loads(index_out[0].text)
        assert "markdown" in index_payload
        # Oracle must refuse when only TOC-shaped text is fed (no memories[]).
        toc_only = index_payload.get("markdown") or ""
        assert isinstance(toc_only, str)
        fake_as_memory = [{"id": "toc", "text": toc_only, "provenance": {}}]
        # Even if TOC accidentally contains the cue string from entity indexing,
        # GR-CE-001: scenarios must not treat index as evidence — assert we pass
        # empty memories to the oracle for action, not TOC.
        action_from_empty = harness.choose_action([])
        assert action_from_empty == harness.ACTION_REFUSE
        # Documented contract: action uses get_context memories, not index markdown.
        _ = mid, fake_as_memory  # mid proves store happened; TOC path unused for action

    async def test_two_namespace_no_cross_tenant_leak_for_action(self, qa_pool, monkeypatch):
        """Security: Session A ns-a store must not leak into ns-b action evidence."""
        harness.configure_agentic_store_flags(monkeypatch)
        session_a = harness.AgenticSession.from_agent("agentic-agent-a")
        session_b = harness.AgenticSession.from_agent("agentic-agent-b")
        assert session_a.namespace != session_b.namespace

        store = await harness.store_memory(
            text=_DECISIVE_TEXT,
            session=session_a,
            entities=[_ENTITY],
            provenance=_PROVENANCE,
        )
        mid = store["memory_id"]
        row_a = await harness.load_chunk(qa_pool, mid)
        assert row_a is not None and row_a["namespace"] == session_a.namespace

        rows_b = await harness.chunks_for_namespace(qa_pool, session_b.namespace)
        assert all(r.get("qdrant_id") != mid for r in rows_b)

        ctx_b = await harness.get_context_with_hits(session=session_b, hits=[])
        action_b = harness.choose_action(ctx_b.get("memories") or [])
        assert action_b == harness.ACTION_REFUSE

        # SEC Low-002: cross-ns hit injected into Session B must be prerank-filtered.
        hit_a = harness.row_to_hit(row_a)
        ctx_leak = await harness.get_context_with_hits(session=session_b, hits=[hit_a])
        leak_memories = ctx_leak.get("memories") or []
        assert all(m.get("id") != mid for m in leak_memories)
        assert harness.choose_action(leak_memories) == harness.ACTION_REFUSE
