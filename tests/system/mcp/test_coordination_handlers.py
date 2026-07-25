"""System tests for selective-share MCP tools (INIT-001/SPEC-010)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from mcp.types import TextContent

pytestmark = [pytest.mark.system, pytest.mark.mcp]


def _assert_text_response(result) -> None:
    assert isinstance(result, list) and len(result) > 0
    first = result[0]
    assert first.type == "text"
    assert first.text
    assert "Traceback" not in first.text


def _parse(result) -> dict:
    _assert_text_response(result)
    return json.loads(result[0].text)


@pytest.fixture
def allow_rbac():
    with (
        patch("archivist.app.handlers.tools_coordination.require_caller", return_value=None),
        patch("archivist.app.handlers.tools_coordination.require_rbac", return_value=None),
    ):
        yield


class TestShareHappyPath:
    @pytest.mark.asyncio
    async def test_propose_accept_conflict_round_trip(self, async_pool, allow_rbac):
        from archivist.app.handlers.tools_coordination import (
            _handle_share_accept,
            _handle_share_attach_conflict,
            _handle_share_get,
            _handle_share_propose,
        )
        from archivist.retrieval.session_store import SessionStore

        propose = await _handle_share_propose(
            {
                "agent_id": "agent-a",
                "recipient_agent_id": "agent-b",
                "namespace": "coord-ns",
                "memory_ids": ["mem-1", "mem-2"],
                "scope": "export-v1",
                "reason": "handoff follow-up",
            }
        )
        grant_id = _parse(propose)["grant"]["id"]
        assert _parse(propose)["grant"]["status"] == "pending"

        with patch(
            "archivist.retrieval.session_store.get_session_store",
            return_value=SessionStore(),
        ):
            accepted = _parse(
                await _handle_share_accept(
                    {
                        "agent_id": "agent-b",
                        "grant_id": grant_id,
                        "namespace": "coord-ns",
                        "session_id": "sess-b",
                    }
                )
            )
        assert accepted["grant"]["status"] == "accepted"
        assert "share_memory_ids" in accepted["injected_keys"]

        conflict = _parse(
            await _handle_share_attach_conflict(
                {
                    "agent_id": "agent-a",
                    "grant_id": grant_id,
                    "namespace": "coord-ns",
                    "action": "merge",
                    "merge_text": "combined view",
                    "reason": "consensus v1",
                }
            )
        )
        assert conflict["grant"]["conflict_outcome"]["action"] == "merge"

        got = _parse(
            await _handle_share_get(
                {
                    "agent_id": "agent-b",
                    "grant_id": grant_id,
                    "namespace": "coord-ns",
                }
            )
        )
        assert got["grant"]["id"] == grant_id
        assert got["grant"]["conflict_outcome"]["action"] == "merge"


class TestUnauthorizedShare:
    @pytest.mark.asyncio
    async def test_propose_denied_without_read(self, async_pool):
        from archivist.app.handlers.tools_coordination import _handle_share_propose

        denied = [
            TextContent(
                type="text",
                text=json.dumps({"error": "access_denied", "reason": "no read"}),
            )
        ]
        with (
            patch(
                "archivist.app.handlers.tools_coordination.require_caller",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_coordination.require_rbac",
                return_value=denied,
            ),
        ):
            result = await _handle_share_propose(
                {
                    "agent_id": "outsider",
                    "recipient_agent_id": "agent-b",
                    "namespace": "private-ns",
                    "memory_ids": ["m1"],
                }
            )
        assert _parse(result)["error"] == "access_denied"

    @pytest.mark.asyncio
    async def test_accept_by_non_recipient_denied(self, async_pool, allow_rbac):
        from archivist.app.handlers.tools_coordination import (
            _handle_share_accept,
            _handle_share_propose,
        )

        propose = await _handle_share_propose(
            {
                "agent_id": "agent-a",
                "recipient_agent_id": "agent-b",
                "namespace": "coord-ns",
                "memory_ids": ["m1"],
            }
        )
        grant_id = _parse(propose)["grant"]["id"]

        denied = _parse(
            await _handle_share_accept(
                {
                    "agent_id": "agent-c",
                    "grant_id": grant_id,
                    "namespace": "coord-ns",
                }
            )
        )
        assert denied["error"] == "access_denied"

    @pytest.mark.asyncio
    async def test_accept_materialize_unauthorized_ns(self, async_pool):
        from archivist.app.handlers.tools_coordination import (
            _handle_share_accept,
            _handle_share_propose,
        )

        with (
            patch(
                "archivist.app.handlers.tools_coordination.require_caller",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_coordination.require_rbac",
                return_value=None,
            ),
        ):
            propose = await _handle_share_propose(
                {
                    "agent_id": "agent-a",
                    "recipient_agent_id": "agent-b",
                    "namespace": "coord-ns",
                    "memory_ids": ["m1"],
                }
            )
        grant_id = _parse(propose)["grant"]["id"]

        denied = [
            TextContent(
                type="text",
                text=json.dumps({"error": "access_denied", "reason": "no write"}),
            )
        ]
        with (
            patch(
                "archivist.app.handlers.tools_coordination.require_caller",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_coordination.require_rbac",
                return_value=denied,
            ),
        ):
            result = await _handle_share_accept(
                {
                    "agent_id": "agent-b",
                    "grant_id": grant_id,
                    "namespace": "coord-ns",
                    "materialize_namespace": "forbidden-ns",
                }
            )
        assert _parse(result)["error"] == "access_denied"


class TestHandoffBackwardCompatible:
    @pytest.mark.asyncio
    async def test_handoff_tools_still_dispatch(self, async_pool):
        from archivist.app.handlers._registry import TOOL_REGISTRY, dispatch_tool
        from archivist.app.handlers.tools_context import HANDLERS as CONTEXT_HANDLERS

        assert "archivist_handoff" in TOOL_REGISTRY
        assert "archivist_receive_handoff" in TOOL_REGISTRY
        assert "archivist_share_propose" in TOOL_REGISTRY
        # Context handlers unchanged by coordination module.
        assert CONTEXT_HANDLERS["archivist_handoff"] is TOOL_REGISTRY["archivist_handoff"]

        # Missing required fields → stable error, not crash (backward compatible path).
        result = await dispatch_tool(
            "archivist_handoff",
            {"agent_id": "agent-a", "session_id": "", "receiving_agent_id": "agent-b"},
        )
        data = _parse(result)
        assert "error" in data
