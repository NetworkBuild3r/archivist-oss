"""Unit tests for retrieval/context_api.py — get_relevant_context, create_handoff_packet, receive_handoff_packet."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from archivist.retrieval.context_api import (
    ContextChunk,
    HandoffPacket,
    RelevantContext,
    create_handoff_packet,
    format_context_for_prompt,
    get_relevant_context,
    receive_handoff_packet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(idx: int, tier: str = "l2", score: float = 0.9) -> dict:
    return {
        "id": f"mem-{idx}",
        "qdrant_id": f"mem-{idx}",
        "text": f"memory content number {idx}",
        "tier_text": f"[{tier.upper()}] memory content number {idx}",
        "score": score,
        "file_path": f"/notes/{idx}.md",
        "date": "2026-04-01",
        "agent_id": "alice",
        "_packed_tier": tier,
    }


def _mock_retrieval_result(sources=None, answer="", over_budget=False):
    return {
        "sources": sources or [_make_source(i) for i in range(3)],
        "answer": answer,
        "over_budget": over_budget,
        "retrieval_trace": {
            "context_status": {
                "tier_distribution": {"l2": 3},
                "token_savings_pct": 12.5,
            },
            "graph_context": [
                {"text": "Alice owns project Archivist"},
                {"fact": "Archivist supports Postgres"},
            ],
        },
    }


# ---------------------------------------------------------------------------
# RelevantContext and ContextChunk dataclass basics
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_context_chunk_fields(self):
        chunk = ContextChunk(
            memory_id="m1",
            text="hello",
            score=0.9,
            tier="l2",
            file_path="/f.md",
            date="2026-01-01",
            agent_id="alice",
        )
        assert chunk.memory_id == "m1"
        assert chunk.tier == "l2"

    def test_relevant_context_defaults(self):
        ctx = RelevantContext(
            answer="",
            sources=[],
            graph_facts=[],
            tips=[],
            total_tokens=0,
            budget_tokens=8000,
            over_budget=False,
            tier_distribution={},
            token_savings_pct=0.0,
            provenance=[],
            pack_policy="adaptive",
        )
        assert ctx.over_budget is False
        assert ctx.pack_policy == "adaptive"

    def test_handoff_packet_fields(self):
        pkt = HandoffPacket(
            from_agent="alice",
            to_agent="bob",
            session_summary="did stuff",
            active_goals=["finish task"],
            open_questions=["why did X fail?"],
            key_memory_ids=["m1", "m2"],
            knowledge_snapshot={"entities": [], "facts": []},
            token_count=100,
            created_at="2026-04-25T00:00:00+00:00",
        )
        assert pkt.from_agent == "alice"
        assert pkt.to_agent == "bob"
        assert pkt.ephemeral_notes == []


# ---------------------------------------------------------------------------
# get_relevant_context
# ---------------------------------------------------------------------------


class TestGetRelevantContext:
    @pytest.mark.asyncio
    async def test_returns_relevant_context_struct(self):
        mock_result = _mock_retrieval_result()
        with (
            patch("archivist.retrieval.context_api.recursive_retrieve", new=AsyncMock(return_value=mock_result)),
            patch("archivist.retrieval.context_api.count_tokens", return_value=50),
            patch("archivist.retrieval.context_api.search_tips", new=AsyncMock(return_value=[])),
        ):
            ctx = await get_relevant_context(
                agent_id="alice",
                task_description="what is archivist?",
                max_tokens=8000,
            )

        assert isinstance(ctx, RelevantContext)
        assert len(ctx.sources) == 3
        assert ctx.budget_tokens == 8000
        assert ctx.pack_policy == "adaptive"

    @pytest.mark.asyncio
    async def test_graph_facts_extracted_from_retrieval_trace(self):
        mock_result = _mock_retrieval_result(answer="Archivist is a memory system.")
        with (
            patch("archivist.retrieval.context_api.recursive_retrieve", new=AsyncMock(return_value=mock_result)),
            patch("archivist.retrieval.context_api.count_tokens", return_value=50),
            patch("archivist.retrieval.context_api.search_tips", new=AsyncMock(return_value=[])),
        ):
            ctx = await get_relevant_context("alice", "archivist")

        assert "Alice owns project Archivist" in ctx.graph_facts
        assert "Archivist supports Postgres" in ctx.graph_facts
        assert ctx.answer == "Archivist is a memory system."

    @pytest.mark.asyncio
    async def test_tips_are_included(self):
        tip_rows = [{"content": "always call session_end at task end"}]
        mock_result = _mock_retrieval_result()
        with (
            patch("archivist.retrieval.context_api.recursive_retrieve", new=AsyncMock(return_value=mock_result)),
            patch("archivist.retrieval.context_api.count_tokens", return_value=50),
            patch("archivist.retrieval.context_api.search_tips", new=AsyncMock(return_value=tip_rows)),
        ):
            ctx = await get_relevant_context("alice", "some task", include_tips=True)

        assert "always call session_end at task end" in ctx.tips

    @pytest.mark.asyncio
    async def test_include_tips_false_skips_tips(self):
        mock_result = _mock_retrieval_result()
        with (
            patch("archivist.retrieval.context_api.recursive_retrieve", new=AsyncMock(return_value=mock_result)),
            patch("archivist.retrieval.context_api.count_tokens", return_value=50),
        ):
            ctx = await get_relevant_context("alice", "some task", include_tips=False)

        assert ctx.tips == []

    @pytest.mark.asyncio
    async def test_include_graph_false_skips_graph_facts(self):
        mock_result = _mock_retrieval_result()
        with (
            patch("archivist.retrieval.context_api.recursive_retrieve", new=AsyncMock(return_value=mock_result)),
            patch("archivist.retrieval.context_api.count_tokens", return_value=50),
        ):
            ctx = await get_relevant_context("alice", "some task", include_graph=False)

        assert ctx.graph_facts == []

    @pytest.mark.asyncio
    async def test_extra_memory_ids_are_injected(self):
        mock_result = _mock_retrieval_result(sources=[_make_source(0)])
        mock_packed = MagicMock()
        mock_packed.sources = [_make_source(0), {"id": "pinned-m99", "qdrant_id": "pinned-m99", "text": "[pinned memory: pinned-m99]", "tier_text": "[pinned memory: pinned-m99]", "score": 0.0, "file_path": "", "date": "", "agent_id": "alice", "_packed_tier": "l2"}]
        mock_packed.total_tokens = 40
        mock_packed.over_budget = False
        mock_packed.tier_distribution = {"l2": 2}
        mock_packed.token_savings_pct = 0.0

        with (
            patch("archivist.retrieval.context_api.recursive_retrieve", new=AsyncMock(return_value=mock_result)),
            patch("archivist.retrieval.context_api.pack_context", return_value=mock_packed),
            patch("archivist.retrieval.context_api.count_tokens", return_value=20),
            patch("archivist.retrieval.context_api.search_tips", new=AsyncMock(return_value=[])),
        ):
            ctx = await get_relevant_context(
                "alice", "task", extra_memory_ids=["pinned-m99"]
            )

        memory_ids = [c.memory_id for c in ctx.sources]
        assert "pinned-m99" in memory_ids

    @pytest.mark.asyncio
    async def test_provenance_contains_all_source_ids(self):
        sources = [_make_source(i) for i in range(4)]
        mock_result = _mock_retrieval_result(sources=sources)
        with (
            patch("archivist.retrieval.context_api.recursive_retrieve", new=AsyncMock(return_value=mock_result)),
            patch("archivist.retrieval.context_api.count_tokens", return_value=50),
        ):
            ctx = await get_relevant_context("alice", "task", include_tips=False, include_graph=False)

        for i in range(4):
            assert f"mem-{i}" in ctx.provenance

    @pytest.mark.asyncio
    async def test_over_budget_propagated(self):
        mock_result = _mock_retrieval_result(over_budget=True)
        with (
            patch("archivist.retrieval.context_api.recursive_retrieve", new=AsyncMock(return_value=mock_result)),
            patch("archivist.retrieval.context_api.count_tokens", return_value=50),
        ):
            ctx = await get_relevant_context("alice", "task", include_tips=False, include_graph=False)

        assert ctx.over_budget is True


# ---------------------------------------------------------------------------
# format_context_for_prompt
# ---------------------------------------------------------------------------


class TestFormatContextForPrompt:
    def _make_ctx(self, answer="", sources=None, graph_facts=None, tips=None, savings=55.0) -> RelevantContext:
        chunks = sources or [
            ContextChunk("m1", "memory text one", 0.9, "l0", "/f.md", "2026-01-01", "alice"),
            ContextChunk("m2", "memory text two", 0.7, "l2", "", "", ""),
        ]
        return RelevantContext(
            answer=answer,
            sources=chunks,
            graph_facts=graph_facts or [],
            tips=tips or [],
            total_tokens=500,
            budget_tokens=8000,
            over_budget=False,
            tier_distribution={"l0": 1, "l2": 1},
            token_savings_pct=savings,
            provenance=["m1", "m2"],
            pack_policy="adaptive",
        )

    def test_answer_section_present_when_non_empty(self):
        ctx = self._make_ctx(answer="This is the synthesized answer.")
        text = format_context_for_prompt(ctx)
        assert "## Memory Answer" in text
        assert "This is the synthesized answer." in text

    def test_no_answer_section_when_empty(self):
        ctx = self._make_ctx(answer="")
        text = format_context_for_prompt(ctx)
        assert "## Memory Answer" not in text

    def test_relevant_memories_section_present(self):
        ctx = self._make_ctx()
        text = format_context_for_prompt(ctx)
        assert "## Relevant Memories" in text
        assert "memory text one" in text

    def test_tier_shown_in_memories(self):
        ctx = self._make_ctx()
        text = format_context_for_prompt(ctx)
        assert "[L0]" in text

    def test_graph_facts_section(self):
        ctx = self._make_ctx(graph_facts=["Alice knows Bob", "Bob owns project X"])
        text = format_context_for_prompt(ctx)
        assert "## Knowledge Graph Facts" in text
        assert "Alice knows Bob" in text

    def test_tips_section_present_when_include_tips_true(self):
        ctx = self._make_ctx(tips=["always end sessions"])
        text = format_context_for_prompt(ctx, include_tips=True)
        assert "## Procedural Tips" in text
        assert "always end sessions" in text

    def test_tips_hidden_when_include_tips_false(self):
        ctx = self._make_ctx(tips=["always end sessions"])
        text = format_context_for_prompt(ctx, include_tips=False)
        assert "## Procedural Tips" not in text

    def test_token_footer_present(self):
        ctx = self._make_ctx(savings=55.0)
        text = format_context_for_prompt(ctx)
        assert "500/8000 tokens" in text
        assert "55.0% savings" in text


# ---------------------------------------------------------------------------
# receive_handoff_packet
# ---------------------------------------------------------------------------


class TestReceiveHandoffPacket:
    @pytest.mark.asyncio
    async def test_injects_summary_and_goals(self):
        pkt = HandoffPacket(
            from_agent="alice",
            to_agent="bob",
            session_summary="Alice completed X",
            active_goals=["finish Y", "review Z"],
            open_questions=["why did A fail?"],
            key_memory_ids=["m1"],
            knowledge_snapshot={},
            token_count=50,
            created_at="2026-04-25T00:00:00+00:00",
        )

        from archivist.retrieval.session_store import SessionStore

        mock_ss = SessionStore()
        with patch("archivist.retrieval.context_api.get_session_store", return_value=mock_ss):
            result = await receive_handoff_packet(pkt, "bob", "session-99")

        assert result["from_agent"] == "alice"
        assert result["to_agent"] == "bob"
        assert "handoff_summary" in result["injected_keys"]
        assert "handoff_goal_0" in result["injected_keys"]
        assert "handoff_goal_1" in result["injected_keys"]
        assert "handoff_recovery_0" in result["injected_keys"]
        assert mock_ss.get("bob", "session-99", "handoff_summary") == "Alice completed X"
        assert mock_ss.get("bob", "session-99", "handoff_goal_0") == "finish Y"

    @pytest.mark.asyncio
    async def test_key_memory_ids_returned(self):
        pkt = HandoffPacket(
            from_agent="alice",
            to_agent="bob",
            session_summary="",
            active_goals=[],
            open_questions=[],
            key_memory_ids=["m1", "m2", "m3"],
            knowledge_snapshot={"entities": [{"name": "Archivist"}], "facts": []},
            token_count=10,
            created_at="2026-04-25T00:00:00+00:00",
        )

        from archivist.retrieval.session_store import SessionStore

        mock_ss = SessionStore()
        with patch("archivist.retrieval.context_api.get_session_store", return_value=mock_ss):
            result = await receive_handoff_packet(pkt, "bob", "sess-1")

        assert result["key_memory_ids"] == ["m1", "m2", "m3"]
        assert result["knowledge_snapshot"]["entities"][0]["name"] == "Archivist"

    @pytest.mark.asyncio
    async def test_ephemeral_notes_injected(self):
        pkt = HandoffPacket(
            from_agent="alice",
            to_agent="bob",
            session_summary="",
            active_goals=[],
            open_questions=[],
            key_memory_ids=[],
            knowledge_snapshot={},
            token_count=0,
            created_at="2026-04-25T00:00:00+00:00",
            ephemeral_notes=[{"key": "goal_hint", "value": "focus on auth"}],
        )

        from archivist.retrieval.session_store import SessionStore

        mock_ss = SessionStore()
        with patch("archivist.retrieval.context_api.get_session_store", return_value=mock_ss):
            result = await receive_handoff_packet(pkt, "bob", "sess-1")

        assert "handoff_note_goal_hint" in result["injected_keys"]
        assert mock_ss.get("bob", "sess-1", "handoff_note_goal_hint") == "focus on auth"
