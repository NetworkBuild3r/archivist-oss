"""System smoke tests for checkpoint MCP tools.

Provenance: INIT-001/SPEC-008; INIT-012/SPEC-004 (branch + HITL product bar).
"""

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
    async def test_branch_interrupt_approve_resume_replay(self, async_pool, allow_rbac):
        """INIT-012/SPEC-004 ac-2: save → branch → interrupt → approve → resume → replay."""
        from archivist.app.handlers.tools_checkpoint import (
            _handle_checkpoint_approve,
            _handle_checkpoint_branch,
            _handle_checkpoint_interrupt,
            _handle_checkpoint_replay,
            _handle_checkpoint_resume,
            _handle_checkpoint_save,
        )
        from archivist.retrieval.session_store import SessionStore

        root = _parse(
            await _handle_checkpoint_save(
                {
                    "agent_id": "sys-agent",
                    "session_id": "sys-sess",
                    "namespace": "sys-ns",
                    "payload": {"summary": "root", "step": 0},
                }
            )
        )["checkpoint"]["id"]

        branched = _parse(
            await _handle_checkpoint_branch(
                {
                    "agent_id": "sys-agent",
                    "parent_checkpoint_id": root,
                    "namespace": "sys-ns",
                    "payload": {"summary": "fork", "step": 1},
                }
            )
        )
        branch_id = branched["checkpoint"]["id"]
        assert branched["checkpoint"]["parent_checkpoint_id"] == root

        interrupted = _parse(
            await _handle_checkpoint_interrupt(
                {
                    "agent_id": "sys-agent",
                    "checkpoint_id": branch_id,
                    "namespace": "sys-ns",
                    "reason": "human review",
                }
            )
        )
        assert interrupted["checkpoint"]["metadata"]["hitl_status"] == "interrupted"

        store = SessionStore()
        with patch(
            "archivist.retrieval.session_store.get_session_store",
            return_value=store,
        ):
            blocked = _parse(
                await _handle_checkpoint_resume(
                    {
                        "agent_id": "sys-agent",
                        "session_id": "sys-sess",
                        "namespace": "sys-ns",
                        "checkpoint_id": branch_id,
                    }
                )
            )
        assert blocked["error"] == "hitl_interrupted"
        assert store.get("sys-agent", "sys-sess", "checkpoint_payload") is None

        approved = _parse(
            await _handle_checkpoint_approve(
                {
                    "agent_id": "sys-agent",
                    "checkpoint_id": branch_id,
                    "namespace": "sys-ns",
                }
            )
        )
        assert approved["checkpoint"]["metadata"]["hitl_status"] == "approved"

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
                        "checkpoint_id": branch_id,
                    }
                )
            )
        assert resume["resume_packet"]["checkpoint_id"] == branch_id
        assert store.get("sys-agent", "sys-sess", "checkpoint_id") == branch_id

        replay = _parse(
            await _handle_checkpoint_replay(
                {
                    "agent_id": "sys-agent",
                    "checkpoint_id": branch_id,
                    "namespace": "sys-ns",
                }
            )
        )
        assert [c["id"] for c in replay["chain"]] == [root, branch_id]

    @pytest.mark.asyncio
    async def test_unauthorized_namespace_denied(self, async_pool):
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

    @pytest.mark.asyncio
    async def test_branch_hitl_rbac_deny(self, async_pool):
        """SPEC-004 security: branch/HITL fail closed without write."""
        from archivist.app.handlers.tools_checkpoint import (
            _handle_checkpoint_approve,
            _handle_checkpoint_branch,
            _handle_checkpoint_interrupt,
        )

        denied = [
            TextContent(
                type="text",
                text=json.dumps({"error": "access_denied", "reason": "no write"}),
            )
        ]
        with (
            patch("archivist.app.handlers.tools_checkpoint.require_caller", return_value=None),
            patch(
                "archivist.app.handlers.tools_checkpoint.require_rbac",
                return_value=denied,
            ),
        ):
            for handler, args in (
                (
                    _handle_checkpoint_branch,
                    {
                        "agent_id": "sys-agent",
                        "parent_checkpoint_id": "any",
                        "namespace": "forbidden",
                    },
                ),
                (
                    _handle_checkpoint_interrupt,
                    {
                        "agent_id": "sys-agent",
                        "checkpoint_id": "any",
                        "namespace": "forbidden",
                    },
                ),
                (
                    _handle_checkpoint_approve,
                    {
                        "agent_id": "sys-agent",
                        "checkpoint_id": "any",
                        "namespace": "forbidden",
                    },
                ),
            ):
                result = await handler(args)
                assert result is denied
                assert _parse(result)["error"] == "access_denied"


class TestCheckpointToolsRegistered:
    def test_eight_checkpoint_tools_in_registry(self):
        from archivist.app.handlers._registry import ALL_TOOLS, TOOL_REGISTRY

        expected = {
            "archivist_checkpoint_save",
            "archivist_checkpoint_list",
            "archivist_checkpoint_get",
            "archivist_checkpoint_resume",
            "archivist_checkpoint_replay",
            "archivist_checkpoint_branch",
            "archivist_checkpoint_interrupt",
            "archivist_checkpoint_approve",
        }
        assert expected <= {t.name for t in ALL_TOOLS}
        assert expected <= set(TOOL_REGISTRY)
        # Every advertised tool must have exactly one handler — the absolute
        # count moves as later specs add tools, the 1:1 invariant does not.
        assert len(ALL_TOOLS) == len(TOOL_REGISTRY)

    def test_smoke_inventory_includes_branch_hitl(self):
        """ac-4: smoke expected-tool list includes new tool names."""
        from tests.system.mcp.test_smoke_all_handlers import _ALL_EXPECTED_TOOLS

        from archivist.app.handlers._registry import TOOL_REGISTRY

        expected = {
            "archivist_checkpoint_branch",
            "archivist_checkpoint_interrupt",
            "archivist_checkpoint_approve",
        }
        assert expected <= set(_ALL_EXPECTED_TOOLS)
        assert expected <= set(TOOL_REGISTRY)
