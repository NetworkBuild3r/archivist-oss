"""INIT-003/SPEC-005 — stable recall shape + pre-rank filters + namespace isolation."""

from __future__ import annotations

import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from archivist.app.handlers import tools_context, tools_search
from archivist.retrieval.context_api import ContextChunk, RelevantContext
from archivist.retrieval.retrieval_filters import (
    apply_prerank_filters,
    build_stable_memories,
    sanitize_provenance,
)

pytestmark = [pytest.mark.unit, pytest.mark.retrieval]


def _hit(
    mid: str,
    *,
    text: str = "usable memory text",
    score: float = 0.9,
    namespace: str = "ns-a",
    subject: str = "",
    purpose: str = "",
    sensitivity: str = "standard",
    is_suppressed: bool = False,
    is_superseded: bool = False,
    superseded_by: str | None = None,
    api_key: str | None = None,
) -> dict:
    row = {
        "id": mid,
        "text": text,
        "score": score,
        "namespace": namespace,
        "subject": subject,
        "purpose": purpose,
        "sensitivity": sensitivity,
        "is_suppressed": is_suppressed,
        "is_superseded": is_superseded,
        "agent_id": "alice",
        "date": "2026-07-25",
        "source": "user",
        "confidence": 0.8,
        "statement_kind": "user",
    }
    if superseded_by is not None:
        row["superseded_by"] = superseded_by
    if api_key is not None:
        row["api_key"] = api_key
    return row


class TestBuildStableMemories:
    def test_non_empty_text_when_hits_exist(self):
        memories = build_stable_memories([_hit("m1"), _hit("m2", text="second")])
        assert len(memories) == 2
        assert all(m["text"].strip() for m in memories)
        assert memories[0]["id"] == "m1"
        assert "score" in memories[0]
        assert "provenance" in memories[0]

    def test_uses_tier_text_fallback(self):
        row = _hit("m1", text="")
        row["tier_text"] = "tier body"
        memories = build_stable_memories([row])
        assert memories[0]["text"] == "tier body"

    def test_provenance_strips_secrets(self):
        row = _hit("m1", api_key="sk-secret-should-not-leak")
        row["password"] = "hunter2"
        row["provenance"] = {"token": "abc", "subject": "coach"}
        memories = build_stable_memories([row])
        prov = memories[0]["provenance"]
        assert "api_key" not in prov
        assert "password" not in prov
        assert "token" not in prov
        assert prov.get("subject") == "coach"
        assert "sk-secret" not in json.dumps(memories)


class TestSanitizeProvenance:
    def test_drops_secret_keys(self):
        cleaned = sanitize_provenance(
            {
                "subject": "ok",
                "api_key": "nope",
                "authorization": "Bearer x",
                "namespace": "ns-a",
            }
        )
        assert cleaned == {"subject": "ok", "namespace": "ns-a"}


class TestPrerankFilters:
    def test_omits_suppressed(self):
        rows = [_hit("ok"), _hit("bad", is_suppressed=True)]
        out = apply_prerank_filters(rows, namespace="ns-a")
        assert [r["id"] for r in out] == ["ok"]

    def test_omits_superseded_losers(self):
        rows = [
            _hit("winner"),
            _hit("loser", is_superseded=True),
            _hit("loser2", superseded_by="winner"),
            _hit("loser3"),
        ]
        out = apply_prerank_filters(
            rows,
            namespace="ns-a",
            known_superseded_ids={"loser3"},
        )
        assert [r["id"] for r in out] == ["winner"]

    def test_namespace_isolation_critical(self):
        rows = [
            _hit("a", namespace="ns-a", text="tenant A secret plan"),
            _hit("b", namespace="ns-b", text="tenant B secret plan"),
        ]
        out = apply_prerank_filters(rows, namespace="ns-a")
        assert len(out) == 1
        assert out[0]["id"] == "a"
        assert all(r["namespace"] == "ns-a" for r in out)

    def test_subject_filter_isolates(self):
        rows = [
            _hit("a", subject="user-1"),
            _hit("b", subject="user-2"),
        ]
        out = apply_prerank_filters(rows, namespace="ns-a", subject="user-1")
        assert [r["id"] for r in out] == ["a"]

    def test_purpose_default_no_restriction(self):
        rows = [
            _hit("a", purpose="coaching"),
            _hit("b", purpose="ops"),
        ]
        out = apply_prerank_filters(rows, namespace="ns-a", purpose="")
        assert len(out) == 2

    def test_purpose_and_sensitivity_narrow(self):
        rows = [
            _hit("a", purpose="coaching", sensitivity="standard"),
            _hit("b", purpose="coaching", sensitivity="sensitive"),
            _hit("c", purpose="ops", sensitivity="standard"),
        ]
        out = apply_prerank_filters(
            rows,
            namespace="ns-a",
            purpose="coaching",
            sensitivity="standard",
        )
        assert [r["id"] for r in out] == ["a"]

    def test_no_caller_bypass_to_widen_tenants(self):
        """apply_prerank_filters has no include_all / skip_namespace widen knobs."""
        sig = inspect.signature(apply_prerank_filters)
        forbidden = {
            "include_suppressed",
            "include_superseded",
            "skip_namespace",
            "include_all_namespaces",
            "disable_filters",
            "widen",
        }
        assert forbidden.isdisjoint(sig.parameters.keys())


class TestSearchHandlerStableShape:
    @pytest.mark.asyncio
    async def test_search_returns_memories_with_text(self):
        mock_result = {
            "answer": "",
            "sources": [_hit("m1", text="recall body")],
            "memories": build_stable_memories([_hit("m1", text="recall body")]),
        }
        with (
            patch(
                "archivist.app.handlers.tools_search.recursive_retrieve",
                new=AsyncMock(return_value=mock_result),
            ),
            patch("archivist.app.handlers.tools_search.require_rbac", return_value=None),
        ):
            out = await tools_search._handle_search(
                {
                    "query": "what?",
                    "agent_id": "alice",
                    "namespace": "ns-a",
                    "refine": False,
                }
            )
        payload = json.loads(out[0].text)
        assert payload["answer"] == ""
        assert payload["memories"]
        assert payload["memories"][0]["text"] == "recall body"
        assert "sources" in payload

    @pytest.mark.asyncio
    async def test_search_passes_prerank_args(self):
        mock_retrieve = AsyncMock(return_value={"answer": "", "sources": [], "memories": []})
        with (
            patch(
                "archivist.app.handlers.tools_search.recursive_retrieve",
                new=mock_retrieve,
            ),
            patch("archivist.app.handlers.tools_search.require_rbac", return_value=None),
        ):
            await tools_search._handle_search(
                {
                    "query": "q",
                    "agent_id": "alice",
                    "namespace": "ns-a",
                    "subject": "user-1",
                    "purpose": "coaching",
                    "sensitivity": "standard",
                }
            )
        kwargs = mock_retrieve.await_args.kwargs
        assert kwargs["subject"] == "user-1"
        assert kwargs["purpose"] == "coaching"
        assert kwargs["sensitivity"] == "standard"
        assert "include_suppressed" not in kwargs


class TestGetContextHandler:
    @pytest.mark.asyncio
    async def test_get_context_returns_memories_and_rbac(self):
        ctx = RelevantContext(
            answer="",
            sources=[
                ContextChunk(
                    memory_id="m1",
                    text="context body",
                    score=0.91,
                    tier="l2",
                    file_path="/x.md",
                    date="2026-07-25",
                    agent_id="alice",
                    namespace="ns-a",
                    subject="user-1",
                )
            ],
            graph_facts=[],
            tips=[],
            total_tokens=10,
            budget_tokens=8000,
            over_budget=False,
            tier_distribution={"l2": 1},
            token_savings_pct=0.0,
            provenance=["m1"],
            pack_policy="adaptive",
            memories=build_stable_memories([_hit("m1", text="context body", subject="user-1")]),
        )
        with (
            patch(
                "archivist.retrieval.context_api.get_relevant_context",
                new=AsyncMock(return_value=ctx),
            ),
            patch(
                "archivist.app.handlers.tools_context.require_rbac",
                return_value=None,
            ) as rbac,
        ):
            out = await tools_context._handle_get_context(
                {
                    "agent_id": "alice",
                    "task_description": "help",
                    "namespace": "ns-a",
                }
            )
        rbac.assert_called_once_with("alice", "read", "ns-a")
        payload = json.loads(out[0].text)
        assert payload["memories"][0]["text"] == "context body"
        assert payload["sources"][0]["text"] == "context body"
        assert payload["provenance"] == ["m1"]

    @pytest.mark.asyncio
    async def test_get_context_rbac_denied(self):
        denied = [MagicMock(text='{"error":"access_denied"}')]
        with patch(
            "archivist.app.handlers.tools_context.require_rbac",
            return_value=denied,
        ):
            out = await tools_context._handle_get_context(
                {
                    "agent_id": "alice",
                    "task_description": "help",
                    "namespace": "ns-other",
                }
            )
        assert out is denied


class TestToolDescriptionsDocumentShape:
    def test_search_tool_mentions_memories(self):
        tool = next(t for t in tools_search.TOOLS if t.name == "archivist_search")
        assert "memories" in tool.description
        props = tool.inputSchema["properties"]
        assert "subject" in props
        assert "purpose" in props
        assert "sensitivity" in props

    def test_get_context_tool_mentions_memories(self):
        tool = next(t for t in tools_context.TOOLS if t.name == "archivist_get_context")
        assert "memories" in tool.description
        props = tool.inputSchema["properties"]
        assert "subject" in props
        assert "purpose" in props
