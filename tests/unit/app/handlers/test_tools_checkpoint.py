"""Unit tests for checkpoint MCP handlers (INIT-001/SPEC-008)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.mcp]


def _parse(result) -> dict:
    assert isinstance(result, list) and result
    return json.loads(result[0].text)


def _record(**overrides):
    base = {
        "id": "ckpt-1",
        "agent_id": "agent-a",
        "session_id": "sess-1",
        "namespace": "ns-a",
        "parent_checkpoint_id": None,
        "payload": {"summary": "hello", "active_goals": ["g1"], "extra_memory_ids": ["m1"]},
        "blob_ref": None,
        "metadata": {"step": 1},
        "created_at": "2026-07-25T00:00:00+00:00",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def allow_rbac():
    with (
        patch("archivist.app.handlers.tools_checkpoint.require_caller", return_value=None),
        patch("archivist.app.handlers.tools_checkpoint.require_rbac", return_value=None),
    ):
        yield


class TestCheckpointSaveValidation:
    @pytest.mark.asyncio
    async def test_rejects_oversized_payload(self, allow_rbac):
        from archivist.app.handlers.tools_checkpoint import (
            _MAX_PAYLOAD_BYTES,
            _handle_checkpoint_save,
        )

        huge = {"blob": "x" * (_MAX_PAYLOAD_BYTES + 1)}
        result = await _handle_checkpoint_save(
            {
                "agent_id": "agent-a",
                "session_id": "sess-1",
                "namespace": "ns-a",
                "payload": huge,
            }
        )
        data = _parse(result)
        assert data["error"] == "payload_too_large"

    @pytest.mark.asyncio
    async def test_requires_namespace(self, allow_rbac):
        from archivist.app.handlers.tools_checkpoint import _handle_checkpoint_save

        result = await _handle_checkpoint_save(
            {"agent_id": "agent-a", "session_id": "sess-1", "namespace": ""}
        )
        data = _parse(result)
        assert data["error"] == "namespace_required"

    @pytest.mark.asyncio
    async def test_save_round_trip_shape(self, allow_rbac):
        from archivist.app.handlers.tools_checkpoint import _handle_checkpoint_save

        created = _record()
        with patch(
            "archivist.storage.checkpoints.create_checkpoint",
            new=AsyncMock(return_value=created),
        ):
            result = await _handle_checkpoint_save(
                {
                    "agent_id": "agent-a",
                    "session_id": "sess-1",
                    "namespace": "ns-a",
                    "payload": {"summary": "hello"},
                }
            )
        data = _parse(result)
        assert data["checkpoint"]["id"] == "ckpt-1"
        assert data["checkpoint"]["namespace"] == "ns-a"


class TestCheckpointGetIsolation:
    @pytest.mark.asyncio
    async def test_missing_or_wrong_namespace_is_not_found(self, allow_rbac):
        from archivist.app.handlers.tools_checkpoint import _handle_checkpoint_get

        with patch(
            "archivist.storage.checkpoints.get_checkpoint",
            new=AsyncMock(return_value=None),
        ):
            result = await _handle_checkpoint_get(
                {
                    "agent_id": "agent-a",
                    "checkpoint_id": "ckpt-1",
                    "namespace": "other-ns",
                }
            )
        data = _parse(result)
        assert data["error"] == "not_found"

    @pytest.mark.asyncio
    async def test_rbac_denied(self):
        from mcp.types import TextContent

        from archivist.app.handlers.tools_checkpoint import _handle_checkpoint_get

        denied = [TextContent(type="text", text=json.dumps({"error": "access_denied"}))]
        with (
            patch("archivist.app.handlers.tools_checkpoint.require_caller", return_value=None),
            patch("archivist.app.handlers.tools_checkpoint.require_rbac", return_value=denied),
        ):
            result = await _handle_checkpoint_get(
                {
                    "agent_id": "agent-a",
                    "checkpoint_id": "ckpt-1",
                    "namespace": "ns-a",
                }
            )
        assert result is denied


class TestCheckpointListAndReplay:
    @pytest.mark.asyncio
    async def test_list_returns_count(self, allow_rbac):
        from archivist.app.handlers.tools_checkpoint import _handle_checkpoint_list

        rows = [_record(id="a"), _record(id="b", parent_checkpoint_id="a")]
        with patch(
            "archivist.storage.checkpoints.list_checkpoints_by_session",
            new=AsyncMock(return_value=rows),
        ):
            result = await _handle_checkpoint_list(
                {
                    "agent_id": "agent-a",
                    "session_id": "sess-1",
                    "namespace": "ns-a",
                }
            )
        data = _parse(result)
        assert data["count"] == 2
        assert [c["id"] for c in data["checkpoints"]] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_replay_root_to_leaf(self, allow_rbac):
        from archivist.app.handlers.tools_checkpoint import _handle_checkpoint_replay

        leaf = _record(id="leaf", parent_checkpoint_id="root", payload={"n": 2})
        root = _record(id="root", parent_checkpoint_id=None, payload={"n": 1})

        async def _get(cid, *, namespace):
            assert namespace == "ns-a"
            return {"leaf": leaf, "root": root}.get(cid)

        with patch(
            "archivist.storage.checkpoints.get_checkpoint",
            new=AsyncMock(side_effect=_get),
        ):
            result = await _handle_checkpoint_replay(
                {
                    "agent_id": "agent-a",
                    "checkpoint_id": "leaf",
                    "namespace": "ns-a",
                }
            )
        data = _parse(result)
        assert data["readonly"] is True
        assert [c["id"] for c in data["chain"]] == ["root", "leaf"]


class TestCheckpointResumeSessionIsolation:
    @pytest.mark.asyncio
    async def test_resume_injects_only_caller_session(self, allow_rbac):
        from archivist.app.handlers.tools_checkpoint import _handle_checkpoint_resume
        from archivist.retrieval.session_store import SessionStore

        store = SessionStore(max_entries=64, default_ttl_seconds=3600)
        # Pre-seed another agent's key — must remain untouched.
        store.put("other-agent", "other-sess", "secret", "keep-me")

        with (
            patch(
                "archivist.storage.checkpoints.get_checkpoint",
                new=AsyncMock(return_value=_record()),
            ),
            patch(
                "archivist.retrieval.session_store.get_session_store",
                return_value=store,
            ),
        ):
            result = await _handle_checkpoint_resume(
                {
                    "agent_id": "agent-a",
                    "session_id": "sess-1",
                    "namespace": "ns-a",
                    "checkpoint_id": "ckpt-1",
                }
            )
        data = _parse(result)
        packet = data["resume_packet"]
        assert "checkpoint_payload" in packet["injected_keys"]
        assert packet["extra_memory_ids"] == ["m1"]
        assert store.get("agent-a", "sess-1", "checkpoint_id") == "ckpt-1"
        assert store.get("agent-a", "sess-1", "checkpoint_summary") == "hello"
        assert store.get("other-agent", "other-sess", "secret") == "keep-me"

    @pytest.mark.asyncio
    async def test_resume_denies_other_agents_checkpoint(self, allow_rbac):
        """SEC-008-01: namespace read must not permit resuming another agent."""
        from archivist.app.handlers.tools_checkpoint import _handle_checkpoint_resume
        from archivist.retrieval.session_store import SessionStore

        store = SessionStore(max_entries=64, default_ttl_seconds=3600)
        with (
            patch(
                "archivist.storage.checkpoints.get_checkpoint",
                new=AsyncMock(return_value=_record(agent_id="victim-agent")),
            ),
            patch(
                "archivist.retrieval.session_store.get_session_store",
                return_value=store,
            ),
        ):
            result = await _handle_checkpoint_resume(
                {
                    "agent_id": "attacker",
                    "session_id": "sess-x",
                    "namespace": "ns-a",
                    "checkpoint_id": "ckpt-1",
                }
            )
        data = _parse(result)
        assert data["error"] == "access_denied"
        assert store.get("attacker", "sess-x", "checkpoint_payload") is None


class TestRegistryWiring:
    def test_tools_registered(self):
        from archivist.app.handlers._registry import ALL_TOOLS, TOOL_REGISTRY

        names = {
            "archivist_checkpoint_save",
            "archivist_checkpoint_list",
            "archivist_checkpoint_get",
            "archivist_checkpoint_resume",
            "archivist_checkpoint_replay",
        }
        tool_names = {t.name for t in ALL_TOOLS}
        assert names <= tool_names
        assert names <= set(TOOL_REGISTRY)

    def test_handoff_tools_intact(self):
        """GR-003: checkpoint tools must not remove existing handoff tools."""
        from archivist.app.handlers._registry import TOOL_REGISTRY

        assert "archivist_handoff" in TOOL_REGISTRY
        assert "archivist_receive_handoff" in TOOL_REGISTRY
        assert "archivist_get_context" in TOOL_REGISTRY
