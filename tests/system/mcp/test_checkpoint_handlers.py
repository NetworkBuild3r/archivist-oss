"""System smoke tests for checkpoint MCP tools (INIT-001/SPEC-008)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.system, pytest.mark.mcp]


def _assert_text_response(result) -> None:
    assert isinstance(result, list) and len(result) > 0
    first = result[0]
    assert first.type == "text"
    assert first.text
    assert "Traceback" not in first.text
    assert "coroutine" not in first.text


def _parse(result) -> dict:
    _assert_text_response(result)
    return json.loads(result[0].text)


@pytest.fixture
def allow_rbac():
    with (
        patch("archivist.app.handlers.tools_checkpoint.require_caller", return_value=None),
        patch("archivist.app.handlers.tools_checkpoint.require_rbac", return_value=None),
    ):
        yield


class TestCheckpointHandlerSmoke:
    @pytest.mark.asyncio
    async def test_save_list_get_resume_replay_round_trip(self, async_pool, allow_rbac):
        from archivist.app.handlers.tools_checkpoint import (
            _handle_checkpoint_get,
            _handle_checkpoint_list,
            _handle_checkpoint_replay,
            _handle_checkpoint_resume,
            _handle_checkpoint_save,
        )
        from archivist.retrieval.session_store import SessionStore

        save = await _handle_checkpoint_save(
            {
                "agent_id": "sys-agent",
                "session_id": "sys-sess",
                "namespace": "sys-ns",
                "payload": {
                    "summary": "mid-task",
                    "active_goals": ["finish SPEC-008"],
                    "extra_memory_ids": ["mem-1"],
                },
                "metadata": {"step": 1},
            }
        )
        save_data = _parse(save)
        ckpt_id = save_data["checkpoint"]["id"]

        child = await _handle_checkpoint_save(
            {
                "agent_id": "sys-agent",
                "session_id": "sys-sess",
                "namespace": "sys-ns",
                "payload": {"summary": "child", "n": 2},
                "parent_checkpoint_id": ckpt_id,
            }
        )
        child_id = _parse(child)["checkpoint"]["id"]

        listed = _parse(
            await _handle_checkpoint_list(
                {
                    "agent_id": "sys-agent",
                    "session_id": "sys-sess",
                    "namespace": "sys-ns",
                }
            )
        )
        assert listed["count"] == 2

        got = _parse(
            await _handle_checkpoint_get(
                {
                    "agent_id": "sys-agent",
                    "checkpoint_id": ckpt_id,
                    "namespace": "sys-ns",
                }
            )
        )
        assert got["checkpoint"]["id"] == ckpt_id

        # Cross-namespace get must not leak.
        missing = _parse(
            await _handle_checkpoint_get(
                {
                    "agent_id": "sys-agent",
                    "checkpoint_id": ckpt_id,
                    "namespace": "other-ns",
                }
            )
        )
        assert missing["error"] == "not_found"

        store = SessionStore()
        other = SessionStore()
        other.put("peer", "peer-sess", "untouched", "yes")
        with patch(
            "archivist.retrieval.session_store.get_session_store",
            return_value=store,
        ):
            resume = _parse(
                await _handle_checkpoint_resume(
                    {
                        "agent_id": "sys-agent",
                        "session_id": "sys-sess",
                        "namespace": "sys-ns",
                        "checkpoint_id": child_id,
                    }
                )
            )
        assert resume["resume_packet"]["checkpoint_id"] == child_id
        assert store.get("sys-agent", "sys-sess", "checkpoint_id") == child_id
        assert other.get("peer", "peer-sess", "untouched") == "yes"

        replay = _parse(
            await _handle_checkpoint_replay(
                {
                    "agent_id": "sys-agent",
                    "checkpoint_id": child_id,
                    "namespace": "sys-ns",
                }
            )
        )
        assert replay["readonly"] is True
        assert [c["id"] for c in replay["chain"]] == [ckpt_id, child_id]

    @pytest.mark.asyncio
    async def test_unauthorized_namespace_denied(self, async_pool):
        from mcp.types import TextContent

        from archivist.app.handlers.tools_checkpoint import _handle_checkpoint_list

        denied = [
            TextContent(
                type="text",
                text=json.dumps({"error": "access_denied", "reason": "no_access"}),
            )
        ]
        with (
            patch("archivist.app.handlers.tools_checkpoint.require_caller", return_value=None),
            patch(
                "archivist.app.handlers.tools_checkpoint.require_rbac",
                return_value=denied,
            ),
        ):
            result = await _handle_checkpoint_list(
                {
                    "agent_id": "sys-agent",
                    "session_id": "sys-sess",
                    "namespace": "forbidden",
                }
            )
        assert result is denied
        assert _parse(result)["error"] == "access_denied"


class TestCheckpointToolsRegistered:
    def test_five_checkpoint_tools_in_registry(self):
        from archivist.app.handlers._registry import ALL_TOOLS, TOOL_REGISTRY

        expected = {
            "archivist_checkpoint_save",
            "archivist_checkpoint_list",
            "archivist_checkpoint_get",
            "archivist_checkpoint_resume",
            "archivist_checkpoint_replay",
        }
        assert expected <= {t.name for t in ALL_TOOLS}
        assert expected <= set(TOOL_REGISTRY)
        # Every advertised tool must have exactly one handler — the absolute
        # count moves as later specs add tools, the 1:1 invariant does not.
        assert len(ALL_TOOLS) == len(TOOL_REGISTRY)
