"""INIT-012/SPEC-004 — agentic checkpoint ops profile + branch/HITL path.

Diff #7 product bar: ops exposes checkpoint tools; core does not;
save→branch→interrupt→approve→resume via MCP handlers.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

pytestmark = [
    pytest.mark.system,
    pytest.mark.mcp,
    pytest.mark.agentic_memory,
]


def _parse(result) -> dict:
    assert isinstance(result, list) and result
    return json.loads(result[0].text)


@pytest.fixture
def allow_rbac():
    with (
        patch("archivist.app.handlers.tools_checkpoint.require_caller", return_value=None),
        patch("archivist.app.handlers.tools_checkpoint.require_rbac", return_value=None),
    ):
        yield


class TestCheckpointProfileAndHitl:
    def test_core_excludes_checkpoint_ops_includes(self, monkeypatch):
        """ac-1: core excludes archivist_checkpoint_*; ops includes branch/HITL."""
        import archivist.core.config as config
        from archivist.app.handlers._registry import allowed_tool_names, get_all_tools

        monkeypatch.setattr(config, "TOOL_PROFILE", "core")
        core_names = {t.name for t in get_all_tools()}
        assert not any(n.startswith("archivist_checkpoint_") for n in core_names)
        assert not any(n.startswith("archivist_checkpoint_") for n in allowed_tool_names("core"))

        monkeypatch.setattr(config, "TOOL_PROFILE", "ops")
        ops_names = {t.name for t in get_all_tools()}
        assert "archivist_checkpoint_save" in ops_names
        assert "archivist_checkpoint_branch" in ops_names
        assert "archivist_checkpoint_interrupt" in ops_names
        assert "archivist_checkpoint_approve" in ops_names
        assert "archivist_checkpoint_branch" in allowed_tool_names("ops")

    @pytest.mark.asyncio
    async def test_agentic_branch_hitl_resume(self, async_pool, allow_rbac):
        """ac-2: agentic path exercises branch + HITL resume gate."""
        from archivist.app.handlers.tools_checkpoint import (
            _handle_checkpoint_approve,
            _handle_checkpoint_branch,
            _handle_checkpoint_interrupt,
            _handle_checkpoint_resume,
            _handle_checkpoint_save,
        )
        from archivist.retrieval.session_store import SessionStore

        root = _parse(
            await _handle_checkpoint_save(
                {
                    "agent_id": "agentic-a",
                    "session_id": "sess-a",
                    "namespace": "agentic-ns",
                    "payload": {"summary": "plan", "step": 0},
                }
            )
        )["checkpoint"]["id"]

        branch_id = _parse(
            await _handle_checkpoint_branch(
                {
                    "agent_id": "agentic-a",
                    "parent_checkpoint_id": root,
                    "namespace": "agentic-ns",
                    "payload": {"summary": "forked plan", "step": 1},
                }
            )
        )["checkpoint"]["id"]

        await _handle_checkpoint_interrupt(
            {
                "agent_id": "agentic-a",
                "checkpoint_id": branch_id,
                "namespace": "agentic-ns",
                "reason": "wait",
            }
        )

        store = SessionStore()
        with patch(
            "archivist.retrieval.session_store.get_session_store",
            return_value=store,
        ):
            blocked = _parse(
                await _handle_checkpoint_resume(
                    {
                        "agent_id": "agentic-a",
                        "session_id": "sess-a",
                        "namespace": "agentic-ns",
                        "checkpoint_id": branch_id,
                    }
                )
            )
        assert blocked["error"] == "hitl_interrupted"

        await _handle_checkpoint_approve(
            {
                "agent_id": "agentic-a",
                "checkpoint_id": branch_id,
                "namespace": "agentic-ns",
            }
        )

        with patch(
            "archivist.retrieval.session_store.get_session_store",
            return_value=store,
        ):
            resume = _parse(
                await _handle_checkpoint_resume(
                    {
                        "agent_id": "agentic-a",
                        "session_id": "sess-a",
                        "namespace": "agentic-ns",
                        "checkpoint_id": branch_id,
                    }
                )
            )
        assert resume["resume_packet"]["checkpoint_id"] == branch_id
        assert store.get("agentic-a", "sess-a", "checkpoint_summary") == "forked plan"
