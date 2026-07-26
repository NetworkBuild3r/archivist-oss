"""INIT-004/SPEC-004 — get_context coach budgets, mode=bootstrap, empty-OK."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from archivist.app.handlers import tools_context
from archivist.retrieval.context_api import (
    BOOTSTRAP_DEFAULT_MAX_TOKENS,
    NORMAL_DEFAULT_MAX_TOKENS,
    RelevantContext,
    get_relevant_context,
    resolve_context_max_tokens,
)


class TestResolveContextMaxTokens:
    def test_normal_default_is_coach_oriented(self):
        assert resolve_context_max_tokens("normal") == NORMAL_DEFAULT_MAX_TOKENS
        assert NORMAL_DEFAULT_MAX_TOKENS < 8000
        assert NORMAL_DEFAULT_MAX_TOKENS == 2000

    def test_bootstrap_default_in_spirit_range(self):
        assert resolve_context_max_tokens("bootstrap") == BOOTSTRAP_DEFAULT_MAX_TOKENS
        assert 200 <= BOOTSTRAP_DEFAULT_MAX_TOKENS <= 400

    def test_explicit_max_tokens_wins(self):
        assert resolve_context_max_tokens("normal", max_tokens=1234) == 1234
        assert resolve_context_max_tokens("bootstrap", max_tokens=50) == 50


class TestBootstrapMode:
    @pytest.mark.asyncio
    async def test_bootstrap_returns_compact_payload_under_ceiling(self):
        wake = {
            "l0_identity": "Namespace: ns-a; Agent: alice; Core entities: Alice",
            "l1_critical": "[Alice] prefers morning workouts",
            "namespace_toc": "## MEMORY INDEX\n- Alice (person)",
            "fleet_tips": [],
            "total_memories": 3,
            "last_activity": "2026-07-26",
            "top_entities": ["Alice"],
        }
        with (
            patch(
                "archivist.storage.compressed_index.build_wake_up_context",
                new=AsyncMock(return_value=wake),
            ),
            patch(
                "archivist.core.rbac.list_accessible_namespaces",
                return_value=[{"namespace": "ns-a", "can_read": True, "can_write": True}],
            ),
        ):
            ctx = await get_relevant_context(
                agent_id="alice",
                task_description="session start",
                namespace="ns-a",
                mode="bootstrap",
            )

        assert ctx.mode == "bootstrap"
        assert ctx.pack_policy == "bootstrap"
        assert ctx.budget_tokens == BOOTSTRAP_DEFAULT_MAX_TOKENS
        assert ctx.total_tokens <= BOOTSTRAP_DEFAULT_MAX_TOKENS
        assert "Bootstrap Context" in ctx.answer or "Identity" in ctx.answer
        # Do not invent memories from the map (GR-CE-003)
        assert ctx.memories == []
        assert ctx.sources == []
        assert ctx.provenance == []

    @pytest.mark.asyncio
    async def test_bootstrap_respects_explicit_max_tokens(self):
        wake = {
            "l0_identity": "Namespace: ns-a",
            "l1_critical": "No facts recorded yet.",
            "namespace_toc": "",
            "fleet_tips": [],
            "total_memories": 0,
            "last_activity": "",
            "top_entities": [],
        }
        with (
            patch(
                "archivist.storage.compressed_index.build_wake_up_context",
                new=AsyncMock(return_value=wake),
            ),
            patch(
                "archivist.core.rbac.list_accessible_namespaces",
                return_value=[],
            ),
        ):
            ctx = await get_relevant_context(
                agent_id="alice",
                task_description="boot",
                namespace="ns-a",
                mode="bootstrap",
                max_tokens=80,
            )
        assert ctx.budget_tokens == 80
        assert ctx.total_tokens <= 80


class TestNormalModeBudgets:
    @pytest.mark.asyncio
    async def test_normal_default_budget_applied(self):
        mock_result = {
            "sources": [],
            "answer": "",
            "over_budget": False,
            "retrieval_trace": {"context_status": {}, "graph_context": []},
        }
        with (
            patch(
                "archivist.retrieval.context_api.recursive_retrieve",
                new=AsyncMock(return_value=mock_result),
            ) as retrieve,
            patch("archivist.retrieval.context_api.search_tips", new=AsyncMock(return_value=[])),
        ):
            ctx = await get_relevant_context(
                agent_id="alice",
                task_description="no hits",
                mode="normal",
            )
        assert ctx.budget_tokens == NORMAL_DEFAULT_MAX_TOKENS
        assert retrieve.await_args.kwargs["max_tokens"] == NORMAL_DEFAULT_MAX_TOKENS
        assert ctx.mode == "normal"

    @pytest.mark.asyncio
    async def test_explicit_max_tokens_passed_to_retrieve(self):
        mock_result = {
            "sources": [],
            "answer": "",
            "over_budget": False,
            "retrieval_trace": {"context_status": {}, "graph_context": []},
        }
        with (
            patch(
                "archivist.retrieval.context_api.recursive_retrieve",
                new=AsyncMock(return_value=mock_result),
            ) as retrieve,
            patch("archivist.retrieval.context_api.search_tips", new=AsyncMock(return_value=[])),
        ):
            ctx = await get_relevant_context(
                agent_id="alice",
                task_description="task",
                max_tokens=777,
            )
        assert ctx.budget_tokens == 777
        assert retrieve.await_args.kwargs["max_tokens"] == 777


class TestEmptyMemoriesOk:
    @pytest.mark.asyncio
    async def test_empty_hits_return_success_shape_without_inventing(self):
        mock_result = {
            "sources": [],
            "answer": "",
            "over_budget": False,
            "retrieval_trace": {"context_status": {}, "graph_context": []},
        }
        with (
            patch(
                "archivist.retrieval.context_api.recursive_retrieve",
                new=AsyncMock(return_value=mock_result),
            ),
            patch("archivist.retrieval.context_api.search_tips", new=AsyncMock(return_value=[])),
        ):
            ctx = await get_relevant_context(
                agent_id="alice",
                task_description="nothing matches",
                include_graph=False,
                include_tips=False,
            )
        assert ctx.memories == []
        assert ctx.sources == []
        assert ctx.provenance == []
        assert isinstance(ctx, RelevantContext)


class TestGetContextHandlerBootstrap:
    @pytest.mark.asyncio
    async def test_handler_bootstrap_mode_and_rbac(self):
        ctx = RelevantContext(
            answer="## Bootstrap Context\n**Identity:** Namespace: ns-a",
            sources=[],
            graph_facts=[],
            tips=[],
            total_tokens=40,
            budget_tokens=400,
            over_budget=False,
            tier_distribution={},
            token_savings_pct=0.0,
            provenance=[],
            pack_policy="bootstrap",
            memories=[],
            mode="bootstrap",
        )
        with (
            patch(
                "archivist.retrieval.context_api.get_relevant_context",
                new=AsyncMock(return_value=ctx),
            ) as get_ctx,
            patch(
                "archivist.app.handlers.tools_context.require_rbac",
                return_value=None,
            ) as rbac,
        ):
            out = await tools_context._handle_get_context(
                {
                    "agent_id": "alice",
                    "task_description": "session start",
                    "namespace": "ns-a",
                    "mode": "bootstrap",
                }
            )
        rbac.assert_called_once_with("alice", "read", "ns-a")
        assert get_ctx.await_args.kwargs["mode"] == "bootstrap"
        assert get_ctx.await_args.kwargs["max_tokens"] == BOOTSTRAP_DEFAULT_MAX_TOKENS
        payload = json.loads(out[0].text)
        assert payload["mode"] == "bootstrap"
        assert payload["memories"] == []
        assert payload["context_status"]["budget_tokens"] == 400
        assert "error" not in payload

    @pytest.mark.asyncio
    async def test_handler_empty_memories_success(self):
        ctx = RelevantContext(
            answer="",
            sources=[],
            graph_facts=[],
            tips=[],
            total_tokens=0,
            budget_tokens=NORMAL_DEFAULT_MAX_TOKENS,
            over_budget=False,
            tier_distribution={},
            token_savings_pct=0.0,
            provenance=[],
            pack_policy="adaptive",
            memories=[],
            mode="normal",
        )
        with patch(
            "archivist.retrieval.context_api.get_relevant_context",
            new=AsyncMock(return_value=ctx),
        ):
            out = await tools_context._handle_get_context(
                {
                    "agent_id": "alice",
                    "task_description": "no hits",
                }
            )
        payload = json.loads(out[0].text)
        assert payload["memories"] == []
        assert "error" not in payload
        assert payload["context_status"]["budget_tokens"] == NORMAL_DEFAULT_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_handler_explicit_max_tokens_wins(self):
        ctx = RelevantContext(
            answer="",
            sources=[],
            graph_facts=[],
            tips=[],
            total_tokens=0,
            budget_tokens=999,
            over_budget=False,
            tier_distribution={},
            token_savings_pct=0.0,
            provenance=[],
            pack_policy="adaptive",
            memories=[],
            mode="normal",
        )
        with patch(
            "archivist.retrieval.context_api.get_relevant_context",
            new=AsyncMock(return_value=ctx),
        ) as get_ctx:
            await tools_context._handle_get_context(
                {
                    "agent_id": "alice",
                    "task_description": "task",
                    "max_tokens": 999,
                }
            )
        assert get_ctx.await_args.kwargs["max_tokens"] == 999

    @pytest.mark.asyncio
    async def test_bootstrap_rbac_denied(self):
        denied = [MagicMock(text='{"error":"access_denied"}')]
        with patch(
            "archivist.app.handlers.tools_context.require_rbac",
            return_value=denied,
        ):
            out = await tools_context._handle_get_context(
                {
                    "agent_id": "alice",
                    "task_description": "boot",
                    "namespace": "ns-other",
                    "mode": "bootstrap",
                }
            )
        assert out is denied

    def test_tool_schema_documents_modes_and_budgets(self):
        tool = next(t for t in tools_context.TOOLS if t.name == "archivist_get_context")
        assert "bootstrap" in tool.description
        assert "empty" in tool.description.lower() or "memories[]" in tool.description
        props = tool.inputSchema["properties"]
        assert props["mode"]["enum"] == ["normal", "bootstrap"]
        assert "2000" in props["max_tokens"]["description"]
        assert "400" in props["max_tokens"]["description"]
        # No schema default that would override omission → mode defaults apply
        assert "default" not in props["max_tokens"]
