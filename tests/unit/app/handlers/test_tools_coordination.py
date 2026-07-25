"""Unit tests for selective-share MCP handlers (INIT-001/SPEC-010)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import TextContent

pytestmark = [pytest.mark.unit, pytest.mark.mcp]


def _parse(result) -> dict:
    assert isinstance(result, list) and result
    return json.loads(result[0].text)


def _grant(**overrides):
    base = {
        "id": "grant-1",
        "proposer_agent_id": "agent-a",
        "recipient_agent_id": "agent-b",
        "namespace": "ns-a",
        "memory_ids": ["m1", "m2"],
        "scope": "scope-v1",
        "status": "pending",
        "conflict_outcome": None,
        "reason": "need context",
        "metadata": {},
        "created_at": "2026-07-25T00:00:00+00:00",
        "decided_at": None,
        "decided_by": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def allow_rbac():
    with (
        patch("archivist.app.handlers.tools_coordination.require_caller", return_value=None),
        patch("archivist.app.handlers.tools_coordination.require_rbac", return_value=None),
    ):
        yield


class TestSharePropose:
    @pytest.mark.asyncio
    async def test_requires_memory_ids_or_scope(self, allow_rbac):
        from archivist.app.handlers.tools_coordination import _handle_share_propose

        result = await _handle_share_propose(
            {
                "agent_id": "agent-a",
                "recipient_agent_id": "agent-b",
                "namespace": "ns-a",
            }
        )
        data = _parse(result)
        assert data["error"] == "share_target_required"

    @pytest.mark.asyncio
    async def test_propose_happy_path(self, allow_rbac):
        from archivist.app.handlers.tools_coordination import _handle_share_propose

        created = _grant()
        with (
            patch(
                "archivist.storage.share_grants.create_share_grant",
                new=AsyncMock(return_value=created),
            ),
            patch(
                "archivist.storage.share_grants.record_to_dict",
                return_value={"id": "grant-1", "status": "pending"},
            ),
            patch(
                "archivist.app.handlers.tools_coordination._audit_share",
                new=AsyncMock(),
            ) as audit,
        ):
            result = await _handle_share_propose(
                {
                    "agent_id": "agent-a",
                    "recipient_agent_id": "agent-b",
                    "namespace": "ns-a",
                    "memory_ids": ["m1"],
                }
            )
        data = _parse(result)
        assert data["grant"]["id"] == "grant-1"
        audit.assert_awaited_once()
        assert audit.await_args.kwargs["action"] == "share_propose"

    @pytest.mark.asyncio
    async def test_propose_rbac_denied(self):
        from archivist.app.handlers.tools_coordination import _handle_share_propose

        denied = [TextContent(type="text", text=json.dumps({"error": "access_denied"}))]
        with (
            patch("archivist.app.handlers.tools_coordination.require_caller", return_value=None),
            patch("archivist.app.handlers.tools_coordination.require_rbac", return_value=denied),
        ):
            result = await _handle_share_propose(
                {
                    "agent_id": "agent-a",
                    "recipient_agent_id": "agent-b",
                    "namespace": "secret-ns",
                    "memory_ids": ["m1"],
                }
            )
        assert _parse(result)["error"] == "access_denied"

    @pytest.mark.asyncio
    async def test_propose_rejects_mismatched_caller_identity(self, allow_rbac):
        """SEC-012-09: agent_id cannot impersonate a different proposer."""
        from archivist.app.handlers.tools_coordination import _handle_share_propose

        create = AsyncMock()
        with patch("archivist.storage.share_grants.create_share_grant", new=create):
            result = await _handle_share_propose(
                {
                    "agent_id": "victim-proposer",
                    "caller_agent_id": "attacker",
                    "recipient_agent_id": "agent-b",
                    "namespace": "ns-a",
                    "memory_ids": ["m1"],
                }
            )
        assert _parse(result)["error"] == "access_denied"
        create.assert_not_awaited()


class TestShareAcceptReject:
    @pytest.mark.asyncio
    async def test_accept_wrong_recipient_denied(self, allow_rbac):
        from archivist.app.handlers.tools_coordination import _handle_share_accept

        with patch(
            "archivist.storage.share_grants.get_share_grant",
            new=AsyncMock(return_value=_grant()),
        ):
            result = await _handle_share_accept(
                {
                    "agent_id": "agent-evil",
                    "grant_id": "grant-1",
                    "namespace": "ns-a",
                }
            )
        assert _parse(result)["error"] == "access_denied"

    @pytest.mark.asyncio
    async def test_accept_materialize_requires_write(self):
        from archivist.app.handlers.tools_coordination import _handle_share_accept

        denied = [TextContent(type="text", text=json.dumps({"error": "access_denied"}))]
        with (
            patch("archivist.app.handlers.tools_coordination.require_caller", return_value=None),
            patch(
                "archivist.app.handlers.tools_coordination.require_rbac",
                return_value=denied,
            ) as rbac,
        ):
            result = await _handle_share_accept(
                {
                    "agent_id": "agent-b",
                    "grant_id": "grant-1",
                    "namespace": "ns-a",
                    "materialize_namespace": "other-ns",
                }
            )
        assert _parse(result)["error"] == "access_denied"
        rbac.assert_called_once_with("agent-b", "write", "other-ns")

    @pytest.mark.asyncio
    async def test_accept_requires_namespace_read(self):
        """SEC-012-05: being named recipient does not grant namespace read."""
        from archivist.app.handlers.tools_coordination import _handle_share_accept

        denied = [TextContent(type="text", text=json.dumps({"error": "access_denied"}))]
        get_grant = AsyncMock(return_value=_grant())
        with (
            patch("archivist.app.handlers.tools_coordination.require_caller", return_value=None),
            patch(
                "archivist.app.handlers.tools_coordination.require_rbac",
                return_value=denied,
            ) as rbac,
            patch("archivist.storage.share_grants.get_share_grant", new=get_grant),
        ):
            result = await _handle_share_accept(
                {"agent_id": "agent-b", "grant_id": "grant-1", "namespace": "ns-a"}
            )
        assert _parse(result)["error"] == "access_denied"
        rbac.assert_called_once_with("agent-b", "read", "ns-a")
        get_grant.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reject_requires_namespace_read(self):
        """SEC-012-05: reject is also a namespace-scoped grant read."""
        from archivist.app.handlers.tools_coordination import _handle_share_reject

        denied = [TextContent(type="text", text=json.dumps({"error": "access_denied"}))]
        get_grant = AsyncMock(return_value=_grant())
        with (
            patch("archivist.app.handlers.tools_coordination.require_caller", return_value=None),
            patch(
                "archivist.app.handlers.tools_coordination.require_rbac",
                return_value=denied,
            ) as rbac,
            patch("archivist.storage.share_grants.get_share_grant", new=get_grant),
        ):
            result = await _handle_share_reject(
                {"agent_id": "agent-b", "grant_id": "grant-1", "namespace": "ns-a"}
            )
        assert _parse(result)["error"] == "access_denied"
        rbac.assert_called_once_with("agent-b", "read", "ns-a")
        get_grant.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_accept_happy_path(self, allow_rbac):
        from archivist.app.handlers.tools_coordination import _handle_share_accept

        pending = _grant()
        accepted = _grant(status="accepted", decided_by="agent-b", decided_at="t")
        with (
            patch(
                "archivist.storage.share_grants.get_share_grant",
                new=AsyncMock(return_value=pending),
            ),
            patch(
                "archivist.storage.share_grants.decide_share_grant",
                new=AsyncMock(return_value=accepted),
            ),
            patch(
                "archivist.storage.share_grants.record_to_dict",
                return_value={"id": "grant-1", "status": "accepted"},
            ),
            patch(
                "archivist.app.handlers.tools_coordination._audit_share",
                new=AsyncMock(),
            ) as audit,
        ):
            result = await _handle_share_accept(
                {
                    "agent_id": "agent-b",
                    "grant_id": "grant-1",
                    "namespace": "ns-a",
                }
            )
        data = _parse(result)
        assert data["grant"]["status"] == "accepted"
        assert audit.await_args.kwargs["action"] == "share_accept"

    @pytest.mark.asyncio
    async def test_reject_audited(self, allow_rbac):
        from archivist.app.handlers.tools_coordination import _handle_share_reject

        pending = _grant()
        rejected = _grant(status="rejected")
        with (
            patch(
                "archivist.storage.share_grants.get_share_grant",
                new=AsyncMock(return_value=pending),
            ),
            patch(
                "archivist.storage.share_grants.decide_share_grant",
                new=AsyncMock(return_value=rejected),
            ),
            patch(
                "archivist.storage.share_grants.record_to_dict",
                return_value={"id": "grant-1", "status": "rejected"},
            ),
            patch(
                "archivist.app.handlers.tools_coordination._audit_share",
                new=AsyncMock(),
            ) as audit,
        ):
            result = await _handle_share_reject(
                {
                    "agent_id": "agent-b",
                    "grant_id": "grant-1",
                    "namespace": "ns-a",
                    "reason": "no thanks",
                }
            )
        assert _parse(result)["grant"]["status"] == "rejected"
        assert audit.await_args.kwargs["action"] == "share_reject"


class TestShareConflictOutcome:
    @pytest.mark.asyncio
    async def test_invalid_action_rejected(self, allow_rbac):
        from archivist.app.handlers.tools_coordination import _handle_share_attach_conflict

        result = await _handle_share_attach_conflict(
            {
                "agent_id": "agent-a",
                "grant_id": "grant-1",
                "namespace": "ns-a",
                "action": "paxos",
            }
        )
        assert _parse(result)["error"] == "invalid_resolution_action"

    @pytest.mark.asyncio
    async def test_attach_keep_both(self, allow_rbac):
        from archivist.app.handlers.tools_coordination import _handle_share_attach_conflict

        pending = _grant()
        updated = _grant(
            conflict_outcome={"action": "keep_both", "reason": "both valid"},
        )
        with (
            patch(
                "archivist.storage.share_grants.get_share_grant",
                new=AsyncMock(return_value=pending),
            ),
            patch(
                "archivist.storage.share_grants.attach_conflict_outcome",
                new=AsyncMock(return_value=updated),
            ),
            patch(
                "archivist.storage.share_grants.record_to_dict",
                return_value={
                    "id": "grant-1",
                    "conflict_outcome": {"action": "keep_both"},
                },
            ),
            patch(
                "archivist.app.handlers.tools_coordination._audit_share",
                new=AsyncMock(),
            ),
        ):
            result = await _handle_share_attach_conflict(
                {
                    "agent_id": "agent-a",
                    "grant_id": "grant-1",
                    "namespace": "ns-a",
                    "action": "keep_both",
                    "reason": "both valid",
                }
            )
        assert _parse(result)["grant"]["conflict_outcome"]["action"] == "keep_both"

    @pytest.mark.asyncio
    async def test_attach_rejects_spoofed_party_via_agent_id(self, allow_rbac):
        """SEC-012-08: spoofing agent_id as a party must not authorize attach."""
        from archivist.app.handlers.tools_coordination import _handle_share_attach_conflict

        attach = AsyncMock()
        with (
            patch(
                "archivist.storage.share_grants.get_share_grant",
                new=AsyncMock(return_value=_grant()),
            ),
            patch(
                "archivist.storage.share_grants.attach_conflict_outcome",
                new=attach,
            ),
        ):
            result = await _handle_share_attach_conflict(
                {
                    "agent_id": "agent-a",  # real party
                    "caller_agent_id": "attacker",  # RBAC identity (non-party)
                    "grant_id": "grant-1",
                    "namespace": "ns-a",
                    "action": "keep_both",
                }
            )
        assert _parse(result)["error"] == "access_denied"
        attach.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_attach_rejects_oversized_merge_text(self, allow_rbac):
        """SEC-012-10: merge_text is size-capped like other share payloads."""
        from archivist.app.handlers.tools_coordination import (
            _MAX_MERGE_TEXT_CHARS,
            _handle_share_attach_conflict,
        )

        attach = AsyncMock()
        with (
            patch(
                "archivist.storage.share_grants.get_share_grant",
                new=AsyncMock(return_value=_grant()),
            ),
            patch(
                "archivist.storage.share_grants.attach_conflict_outcome",
                new=attach,
            ),
        ):
            result = await _handle_share_attach_conflict(
                {
                    "agent_id": "agent-a",
                    "grant_id": "grant-1",
                    "namespace": "ns-a",
                    "action": "merge",
                    "merge_text": "x" * (_MAX_MERGE_TEXT_CHARS + 1),
                }
            )
        assert _parse(result)["error"] == "merge_text_too_large"
        attach.assert_not_awaited()


class TestShareGetPartyGate:
    @pytest.mark.asyncio
    async def test_get_rejects_spoofed_party_via_agent_id(self, allow_rbac):
        """SEC-012-08: spoofing agent_id as a party must not authorize get."""
        from archivist.app.handlers.tools_coordination import _handle_share_get

        with patch(
            "archivist.storage.share_grants.get_share_grant",
            new=AsyncMock(return_value=_grant()),
        ):
            result = await _handle_share_get(
                {
                    "agent_id": "agent-b",  # real party
                    "caller_agent_id": "attacker",  # RBAC identity (non-party)
                    "grant_id": "grant-1",
                    "namespace": "ns-a",
                }
            )
        assert _parse(result)["error"] == "access_denied"

    @pytest.mark.asyncio
    async def test_get_allows_real_party(self, allow_rbac):
        from archivist.app.handlers.tools_coordination import _handle_share_get

        with (
            patch(
                "archivist.storage.share_grants.get_share_grant",
                new=AsyncMock(return_value=_grant()),
            ),
            patch(
                "archivist.storage.share_grants.record_to_dict",
                return_value={"id": "grant-1", "status": "pending"},
            ),
        ):
            result = await _handle_share_get(
                {
                    "agent_id": "agent-a",
                    "grant_id": "grant-1",
                    "namespace": "ns-a",
                }
            )
        assert _parse(result)["grant"]["id"] == "grant-1"


class TestHandoffStillRegistered:
    def test_handoff_tools_unchanged_in_context_module(self):
        from archivist.app.handlers.tools_context import HANDLERS, TOOLS

        names = {t.name for t in TOOLS}
        assert "archivist_handoff" in names
        assert "archivist_receive_handoff" in names
        assert "archivist_handoff" in HANDLERS
        assert "archivist_receive_handoff" in HANDLERS
