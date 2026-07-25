"""MCP tool handlers — Phase 10 selective share + consensus (beyond handoff).

Extends multi-agent coordination without replacing ``archivist_handoff`` /
``archivist_receive_handoff`` (GR-003). Consensus v1 = explicit accept/reject
plus audit trail; conflict outcomes use SPEC-006 resolution action types.

Security: propose cannot escalate beyond proposer's read rights; accept cannot
write into an unauthorized namespace; every mutation is audited.

Provenance: INIT-001/SPEC-010.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.types import TextContent, Tool

from ._common import (
    error_response,
    require_caller,
    require_rbac,
    resolve_caller,
    success_response,
)

logger = logging.getLogger("archivist.mcp")

# SPEC-006 ResolutionAction set — keep in sync with contradiction_resolve.
_RESOLUTION_ACTIONS = frozenset({"supersede", "merge", "keep_both"})
_MAX_MEMORY_IDS = 500
_MAX_REASON_CHARS = 4000
_MAX_METADATA_BYTES = 64 * 1024
_MAX_MERGE_TEXT_CHARS = 64 * 1024

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="archivist_share_propose",
        description=(
            "Propose a selective share of memory IDs and/or a scope to another agent. "
            "Creates a pending grant; the recipient must accept or reject. "
            "Does not replace archivist_handoff (GR-003). "
            "Proposer must have read access to the source namespace."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Proposing agent (required).",
                },
                "recipient_agent_id": {
                    "type": "string",
                    "description": "Agent invited to receive the share (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Source namespace for the shared memories (required).",
                },
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Selective memory IDs to share (optional if scope set).",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scope label (Memory-as-Product / SPEC-009).",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this share is proposed (audited).",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional non-secret metadata.",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "recipient_agent_id", "namespace"],
        },
    ),
    Tool(
        name="archivist_share_accept",
        description=(
            "Accept a pending selective share grant. Audited and RBAC-enforced. "
            "Only the named recipient may accept. Idempotent if already accepted. "
            "Optional materialize_namespace requires write permission (cannot write "
            "unauthorized namespaces). Injects shared memory IDs into the recipient "
            "SessionStore for the given session_id."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Accepting agent — must be the grant recipient (required).",
                },
                "grant_id": {
                    "type": "string",
                    "description": "Share grant id to accept (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace that owns the grant (required).",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session for SessionStore injection of shared memory IDs.",
                },
                "materialize_namespace": {
                    "type": "string",
                    "description": (
                        "Optional namespace to mark as write target for materialization. "
                        "Requires write RBAC; omit for session-only accept."
                    ),
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "grant_id", "namespace"],
        },
    ),
    Tool(
        name="archivist_share_reject",
        description=(
            "Reject a pending selective share grant. Audited. "
            "Only the named recipient may reject. Idempotent if already rejected."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Rejecting agent — must be the grant recipient (required).",
                },
                "grant_id": {
                    "type": "string",
                    "description": "Share grant id to reject (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace that owns the grant (required).",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional rejection reason (audited).",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "grant_id", "namespace"],
        },
    ),
    Tool(
        name="archivist_share_attach_conflict",
        description=(
            "Attach a conflict/consensus outcome to a share grant using SPEC-006 "
            "resolution action types: supersede | merge | keep_both. "
            "Proposer or recipient may attach; namespace read required."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Calling agent (proposer or recipient) (required).",
                },
                "grant_id": {
                    "type": "string",
                    "description": "Share grant id (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace that owns the grant (required).",
                },
                "action": {
                    "type": "string",
                    "enum": ["supersede", "merge", "keep_both"],
                    "description": "SPEC-006 ResolutionAction (required).",
                },
                "reason": {
                    "type": "string",
                    "description": "Human-readable rationale for the outcome.",
                },
                "winner_fact_id": {
                    "type": "integer",
                    "description": "Winner fact id when action=supersede.",
                },
                "loser_fact_id": {
                    "type": "integer",
                    "description": "Loser fact id when action=supersede.",
                },
                "merge_text": {
                    "type": "string",
                    "description": "Merged text when action=merge.",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional extra outcome fields.",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "grant_id", "namespace", "action"],
        },
    ),
    Tool(
        name="archivist_share_get",
        description=(
            "Fetch one share grant by id scoped to namespace. "
            "Visible to proposer or recipient with namespace read access."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Calling agent (required).",
                },
                "grant_id": {
                    "type": "string",
                    "description": "Share grant id (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace scope (required).",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "grant_id", "namespace"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_namespace(namespace: str) -> list[TextContent] | None:
    if not namespace:
        return error_response(
            {
                "error": "namespace_required",
                "reason": "namespace is required for share grant isolation",
            }
        )
    return None


def _json_size_bytes(obj: Any) -> int:
    return len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def _normalize_memory_ids(raw: Any) -> tuple[list[str] | None, list[TextContent] | None]:
    """Return ``(ids, None)`` on success or ``(None, error_response)`` on failure."""
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return None, error_response(
            {"error": "invalid_memory_ids", "reason": "memory_ids must be an array of strings"}
        )
    if len(raw) > _MAX_MEMORY_IDS:
        return None, error_response(
            {
                "error": "memory_ids_too_many",
                "reason": f"memory_ids exceeds {_MAX_MEMORY_IDS}",
                "count": len(raw),
                "max": _MAX_MEMORY_IDS,
            }
        )
    out: list[str] = []
    for item in raw:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out, None


async def _audit_share(
    *,
    agent_id: str,
    action: str,
    grant_id: str,
    namespace: str,
    metadata: dict[str, Any],
) -> None:
    from archivist.core.audit import log_memory_event

    await log_memory_event(
        agent_id=agent_id,
        action=action,
        memory_id=grant_id,
        namespace=namespace,
        text_hash="",
        version=0,
        metadata=metadata,
    )


def _party_of_grant(caller: str, record: Any) -> bool:
    return caller in (record.proposer_agent_id, record.recipient_agent_id)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_share_propose(arguments: dict) -> list[TextContent]:
    from archivist.storage import share_grants as sg

    agent_id = (arguments.get("agent_id") or "").strip()
    recipient = (arguments.get("recipient_agent_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if not recipient:
        return error_response({"error": "recipient_agent_id is required"})
    if err := _require_namespace(namespace):
        return err
    # Proposer identity must match the RBAC identity. Allowing agent_id to
    # diverge from caller_agent_id would stamp grants/audits as another agent
    # (INIT-001/SPEC-012, SEC-012-09).
    if caller != agent_id:
        return error_response(
            {
                "error": "access_denied",
                "reason": "agent_id must match the effective caller identity",
            }
        )
    # Share cannot escalate beyond proposer's read rights.
    if denied := require_rbac(caller, "read", namespace):
        return denied

    memory_ids, mem_err = _normalize_memory_ids(arguments.get("memory_ids"))
    if mem_err is not None:
        return mem_err
    assert memory_ids is not None

    scope = (arguments.get("scope") or "").strip()
    if not memory_ids and not scope:
        return error_response(
            {
                "error": "share_target_required",
                "reason": "Provide memory_ids and/or scope for a selective share",
            }
        )

    reason = (arguments.get("reason") or "")[:_MAX_REASON_CHARS]
    metadata = arguments.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            return error_response(
                {"error": "invalid_metadata", "reason": "metadata must be a JSON object"}
            )
        if _json_size_bytes(metadata) > _MAX_METADATA_BYTES:
            return error_response({"error": "metadata_too_large"})
    else:
        metadata = {}

    try:
        record = await sg.create_share_grant(
            proposer_agent_id=caller,
            recipient_agent_id=recipient,
            namespace=namespace,
            memory_ids=memory_ids,
            scope=scope,
            reason=reason,
            metadata=metadata,
        )
    except ValueError as exc:
        return error_response({"error": "invalid_request", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_share_propose failed")
        return error_response({"error": str(exc)})

    await _audit_share(
        agent_id=caller,
        action="share_propose",
        grant_id=record.id,
        namespace=namespace,
        metadata={
            "recipient_agent_id": recipient,
            "memory_id_count": len(record.memory_ids),
            "scope": scope,
            "status": record.status,
        },
    )
    return success_response({"grant": sg.record_to_dict(record)}, default=str)


async def _handle_share_accept(arguments: dict) -> list[TextContent]:
    from archivist.storage import share_grants as sg

    agent_id = (arguments.get("agent_id") or "").strip()
    grant_id = (arguments.get("grant_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    session_id = (arguments.get("session_id") or "").strip()
    materialize_ns = (arguments.get("materialize_namespace") or "").strip()
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if not grant_id:
        return error_response({"error": "grant_id is required"})
    if err := _require_namespace(namespace):
        return err

    # Accept cannot write into unauthorized namespace when materializing.
    if materialize_ns:
        if denied := require_rbac(caller, "write", materialize_ns):
            return denied
    # A grant is still namespace-scoped data: reading it requires namespace read
    # access, not merely being named as recipient (INIT-001/SPEC-012, SEC-012-05).
    if denied := require_rbac(caller, "read", namespace):
        return denied

    try:
        existing = await sg.get_share_grant(grant_id, namespace=namespace)
    except ValueError as exc:
        return error_response({"error": "invalid_request", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_share_accept failed")
        return error_response({"error": str(exc)})

    if existing is None:
        return error_response(
            {"error": "not_found", "reason": "share grant not found in namespace"}
        )

    # Only the named recipient may accept (effective caller identity).
    if caller != existing.recipient_agent_id:
        return error_response(
            {
                "error": "access_denied",
                "reason": "only the grant recipient may accept this share",
            }
        )

    try:
        record = await sg.decide_share_grant(
            grant_id,
            namespace=namespace,
            status="accepted",
            decided_by=caller,
        )
    except ValueError as exc:
        return error_response({"error": "invalid_state", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_share_accept failed")
        return error_response({"error": str(exc)})

    if record is None:
        return error_response(
            {"error": "not_found", "reason": "share grant not found in namespace"}
        )

    injected: list[str] = []
    if session_id:
        from archivist.retrieval.session_store import get_session_store

        # Inject only into the recipient agent's session (never the proposer's).
        recipient = record.recipient_agent_id
        ss = get_session_store()
        ss.put(recipient, session_id, "share_grant_id", record.id)
        injected.append("share_grant_id")
        ss.put(recipient, session_id, "share_namespace", record.namespace)
        injected.append("share_namespace")
        if record.memory_ids:
            ss.put(
                recipient,
                session_id,
                "share_memory_ids",
                json.dumps(record.memory_ids, separators=(",", ":")),
            )
            injected.append("share_memory_ids")
        if record.scope:
            ss.put(recipient, session_id, "share_scope", record.scope)
            injected.append("share_scope")

    await _audit_share(
        agent_id=caller,
        action="share_accept",
        grant_id=record.id,
        namespace=namespace,
        metadata={
            "status": record.status,
            "materialize_namespace": materialize_ns or None,
            "injected_keys": injected,
            "idempotent": existing.status == "accepted",
        },
    )
    return success_response(
        {
            "grant": sg.record_to_dict(record),
            "injected_keys": injected,
            "hint": (
                "Pass share_memory_ids to archivist_get_context as extra_memory_ids; "
                "existing archivist_handoff / archivist_receive_handoff remain unchanged."
            ),
        },
        default=str,
    )


async def _handle_share_reject(arguments: dict) -> list[TextContent]:
    from archivist.storage import share_grants as sg

    agent_id = (arguments.get("agent_id") or "").strip()
    grant_id = (arguments.get("grant_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    reason = (arguments.get("reason") or "")[:_MAX_REASON_CHARS]
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if not grant_id:
        return error_response({"error": "grant_id is required"})
    if err := _require_namespace(namespace):
        return err
    if denied := require_rbac(caller, "read", namespace):
        return denied

    try:
        existing = await sg.get_share_grant(grant_id, namespace=namespace)
    except ValueError as exc:
        return error_response({"error": "invalid_request", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_share_reject failed")
        return error_response({"error": str(exc)})

    if existing is None:
        return error_response(
            {"error": "not_found", "reason": "share grant not found in namespace"}
        )
    if caller != existing.recipient_agent_id:
        return error_response(
            {
                "error": "access_denied",
                "reason": "only the grant recipient may reject this share",
            }
        )

    try:
        record = await sg.decide_share_grant(
            grant_id,
            namespace=namespace,
            status="rejected",
            decided_by=caller,
        )
    except ValueError as exc:
        return error_response({"error": "invalid_state", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_share_reject failed")
        return error_response({"error": str(exc)})

    if record is None:
        return error_response(
            {"error": "not_found", "reason": "share grant not found in namespace"}
        )

    await _audit_share(
        agent_id=caller,
        action="share_reject",
        grant_id=record.id,
        namespace=namespace,
        metadata={
            "status": record.status,
            "reason": reason,
            "idempotent": existing.status == "rejected",
        },
    )
    return success_response({"grant": sg.record_to_dict(record)}, default=str)


async def _handle_share_attach_conflict(arguments: dict) -> list[TextContent]:
    from archivist.storage import share_grants as sg

    agent_id = (arguments.get("agent_id") or "").strip()
    grant_id = (arguments.get("grant_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    action = (arguments.get("action") or "").strip()
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if not grant_id:
        return error_response({"error": "grant_id is required"})
    if err := _require_namespace(namespace):
        return err
    if denied := require_rbac(caller, "read", namespace):
        return denied
    if action not in _RESOLUTION_ACTIONS:
        return error_response(
            {
                "error": "invalid_resolution_action",
                "reason": "action must be one of: supersede, merge, keep_both",
                "allowed": sorted(_RESOLUTION_ACTIONS),
            }
        )

    try:
        existing = await sg.get_share_grant(grant_id, namespace=namespace)
    except ValueError as exc:
        return error_response({"error": "invalid_request", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_share_attach_conflict failed")
        return error_response({"error": str(exc)})

    if existing is None:
        return error_response(
            {"error": "not_found", "reason": "share grant not found in namespace"}
        )
    # Party authorization uses the RBAC identity only. Trusting a separate
    # agent_id argument let non-parties spoof proposer/recipient while
    # authenticating as themselves (INIT-001/SPEC-012, SEC-012-08).
    if not _party_of_grant(caller, existing):
        return error_response(
            {
                "error": "access_denied",
                "reason": "only proposer or recipient may attach a conflict outcome",
            }
        )

    merge_text = arguments.get("merge_text")
    if merge_text is not None:
        if not isinstance(merge_text, str):
            return error_response(
                {"error": "invalid_merge_text", "reason": "merge_text must be a string"}
            )
        if len(merge_text) > _MAX_MERGE_TEXT_CHARS:
            return error_response(
                {
                    "error": "merge_text_too_large",
                    "reason": f"merge_text exceeds {_MAX_MERGE_TEXT_CHARS} characters",
                }
            )

    outcome: dict[str, Any] = {
        "action": action,
        "reason": (arguments.get("reason") or "")[:_MAX_REASON_CHARS],
        "attached_by": caller,
    }
    if arguments.get("winner_fact_id") is not None:
        outcome["winner_fact_id"] = arguments.get("winner_fact_id")
    if arguments.get("loser_fact_id") is not None:
        outcome["loser_fact_id"] = arguments.get("loser_fact_id")
    if merge_text is not None:
        outcome["merge_text"] = merge_text
    extra = arguments.get("metadata")
    if isinstance(extra, dict):
        if _json_size_bytes(extra) > _MAX_METADATA_BYTES:
            return error_response({"error": "metadata_too_large"})
        outcome["metadata"] = extra
    if _json_size_bytes(outcome) > _MAX_METADATA_BYTES:
        return error_response(
            {
                "error": "conflict_outcome_too_large",
                "reason": f"conflict_outcome exceeds {_MAX_METADATA_BYTES} bytes",
            }
        )

    try:
        record = await sg.attach_conflict_outcome(grant_id, namespace=namespace, outcome=outcome)
    except ValueError as exc:
        return error_response({"error": "invalid_request", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_share_attach_conflict failed")
        return error_response({"error": str(exc)})

    if record is None:
        return error_response(
            {"error": "not_found", "reason": "share grant not found in namespace"}
        )

    await _audit_share(
        agent_id=caller,
        action="share_attach_conflict",
        grant_id=record.id,
        namespace=namespace,
        metadata={"resolution_action": action, "grant_status": record.status},
    )
    return success_response({"grant": sg.record_to_dict(record)}, default=str)


async def _handle_share_get(arguments: dict) -> list[TextContent]:
    from archivist.storage import share_grants as sg

    agent_id = (arguments.get("agent_id") or "").strip()
    grant_id = (arguments.get("grant_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if not grant_id:
        return error_response({"error": "grant_id is required"})
    if err := _require_namespace(namespace):
        return err
    if denied := require_rbac(caller, "read", namespace):
        return denied

    try:
        record = await sg.get_share_grant(grant_id, namespace=namespace)
    except ValueError as exc:
        return error_response({"error": "invalid_request", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_share_get failed")
        return error_response({"error": str(exc)})

    if record is None:
        return error_response(
            {"error": "not_found", "reason": "share grant not found in namespace"}
        )
    # Same single-identity party gate as attach_conflict (SEC-012-08).
    if not _party_of_grant(caller, record):
        return error_response(
            {
                "error": "access_denied",
                "reason": "only proposer or recipient may view this grant",
            }
        )
    return success_response({"grant": sg.record_to_dict(record)}, default=str)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, object] = {
    "archivist_share_propose": _handle_share_propose,
    "archivist_share_accept": _handle_share_accept,
    "archivist_share_reject": _handle_share_reject,
    "archivist_share_attach_conflict": _handle_share_attach_conflict,
    "archivist_share_get": _handle_share_get,
}
