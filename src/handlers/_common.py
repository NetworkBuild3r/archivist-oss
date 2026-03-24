"""Shared helpers for MCP tool handlers."""

import json
import logging

from mcp.types import TextContent

from rbac import check_access

logger = logging.getLogger("archivist.mcp")


def _rbac_gate(agent_id: str, action: str, namespace: str) -> str | None:
    """Return error JSON string if access denied, None if allowed."""
    policy = check_access(agent_id, action, namespace)
    if not policy.allowed:
        return json.dumps({"error": "access_denied", "reason": policy.reason})
    return None


def error_response(data: dict, **json_kw) -> list[TextContent]:
    """Return a single-element TextContent list with a JSON error payload."""
    return [TextContent(type="text", text=json.dumps(data, **json_kw))]


def success_response(data: dict, **json_kw) -> list[TextContent]:
    """Return a single-element TextContent list with a JSON success payload."""
    json_kw.setdefault("indent", 2)
    return [TextContent(type="text", text=json.dumps(data, **json_kw))]
