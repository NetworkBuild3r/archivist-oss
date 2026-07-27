"""Direct unit tests for INIT-022/SPEC-009's shared `require_rbac` helper.

`require_rbac` replaces ~12 hand-copied RBAC-gate wrap-and-return call sites
across `tools_storage.py`, `tools_search.py`, and `tools_admin.py` (M9) --
it is now the single authorization chokepoint for every handler, so it gets
direct coverage of both the allowed and denied paths here, independent of
any one handler.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from mcp.types import TextContent

from archivist.app.handlers._common import require_rbac
from archivist.core.rbac import AccessPolicy

pytestmark = [pytest.mark.unit, pytest.mark.rbac]


class TestRequireRbacAllowed:
    def test_returns_none_when_access_allowed(self):
        with patch(
            "archivist.app.handlers._common.check_access",
            return_value=AccessPolicy(allowed=True),
        ):
            result = require_rbac("agent-1", "write", "agents-agent-1")

        assert result is None


class TestRequireRbacDenied:
    def test_returns_text_content_list_when_access_denied(self):
        denied_policy = AccessPolicy(
            allowed=False,
            reason="namespace_not_found",
            hint="Did you mean 'agents-agent-1'?",
        )
        with patch("archivist.app.handlers._common.check_access", return_value=denied_policy):
            result = require_rbac("agent-1", "write", "agents-agnt-1")

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].type == "text"

    def test_denial_payload_carries_reason_and_hint(self):
        denied_policy = AccessPolicy(
            allowed=False,
            reason="access_denied",
            hint="Request access via archivist_request_namespace_access.",
        )
        with patch("archivist.app.handlers._common.check_access", return_value=denied_policy):
            result = require_rbac("agent-1", "read", "restricted-ns")

        assert result is not None
        text = result[0].text
        assert "access_denied" in text
        assert "Request access" in text

    def test_matches_rbac_gate_shape_exactly(self):
        """require_rbac must wrap _rbac_gate's raw string, never diverge from it."""
        from archivist.app.handlers._common import _rbac_gate

        denied_policy = AccessPolicy(allowed=False, reason="access_denied")
        with patch("archivist.app.handlers._common.check_access", return_value=denied_policy):
            raw = _rbac_gate("agent-1", "write", "restricted-ns")
            wrapped = require_rbac("agent-1", "write", "restricted-ns")

        assert wrapped == [TextContent(type="text", text=raw)]
