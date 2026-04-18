"""Shared helpers for MCP tool handlers."""

import json
import logging

from mcp.types import TextContent

from archivist.core.rbac import check_access, is_permissive_mode

logger = logging.getLogger("archivist.mcp")

# Always-present guidance appended to every error response so agents can
# self-correct without human intervention.
_UNIVERSAL_NEXT_STEP = (
    "Call archivist_get_reference_docs() (or archivist_get_reference_docs"
    "(section='<topic>')) to read the full tool reference for this operation."
)


def _rbac_gate(agent_id: str, action: str, namespace: str) -> str | None:
    """Return error JSON string if access denied, None if allowed.

    The JSON payload includes:
    - ``hint``              — one-line actionable message
    - ``similar_namespaces`` — fuzzy-matched alternative namespace names
    - ``next_steps``        — ordered list of tool calls the agent should make
                              to diagnose and fix the problem
    - ``get_help``          — always points to archivist_get_reference_docs
    """
    policy = check_access(agent_id, action, namespace)
    if not policy.allowed:
        payload: dict = {"error": "access_denied", "reason": policy.reason}
        if policy.hint:
            payload["hint"] = policy.hint
        if policy.similar_namespaces:
            payload["similar_namespaces"] = policy.similar_namespaces
        if policy.next_steps:
            payload["next_steps"] = policy.next_steps
        payload["get_help"] = _UNIVERSAL_NEXT_STEP
        return json.dumps(payload)
    return None


def resolve_caller(arguments: dict) -> str:
    """Return the effective caller agent_id from tool arguments.

    Prefers ``caller_agent_id`` over ``agent_id`` so that a delegating agent
    can supply the original caller identity explicitly.
    """
    agent_id = (arguments.get("agent_id") or "").strip()
    return (arguments.get("caller_agent_id") or "").strip() or agent_id


def resolve_actor(arguments: dict) -> tuple[str, str]:
    """Return ``(actor_id, actor_type)`` from tool arguments.

    ``actor_id`` defaults to ``agent_id`` for backward compatibility.
    ``actor_type`` defaults to ``"agent"`` when not provided.
    """
    agent_id = (arguments.get("agent_id") or "").strip()
    actor_id = (arguments.get("actor_id") or "").strip() or agent_id
    actor_type = (arguments.get("actor_type") or "").strip() or "agent"
    return actor_id, actor_type


def require_caller(caller: str) -> list[TextContent] | None:
    """Return an error response when caller is missing in strict RBAC mode.

    Returns ``None`` when the caller is present or when permissive mode is
    active (so callers can do ``if err := require_caller(caller): return err``).
    """
    if not is_permissive_mode() and not caller:
        return error_response(
            {
                "error": "caller_required",
                "reason": "agent_id is required — Archivist needs to know which agent is making this call.",
                "hint": "Add agent_id='<your_agent_id>' to your tool call arguments.",
                "next_steps": _MISSING_CALLER_NEXT_STEPS,
                "get_help": _UNIVERSAL_NEXT_STEP,
            }
        )
    return None


_MISSING_CALLER_NEXT_STEPS = [
    "Re-issue the call with agent_id='<your_agent_id>' set to your unique agent identifier.",
    "Call archivist_get_reference_docs() to see required parameters for every tool.",
]


def error_response(data: dict, **json_kw) -> list[TextContent]:
    """Return a single-element TextContent list with a JSON error payload."""
    return [TextContent(type="text", text=json.dumps(data, **json_kw))]


def success_response(data: dict, **json_kw) -> list[TextContent]:
    """Return a single-element TextContent list with a JSON success payload."""
    json_kw.setdefault("indent", 2)
    return [TextContent(type="text", text=json.dumps(data, **json_kw))]
