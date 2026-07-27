"""MCP tool handlers — Phase 7 agent-state checkpoint resume/replay/branch/HITL.

Exposes ``archivist_checkpoint_*`` tools over the checkpoint store.
Keeps existing handoff tools intact (GR-003). Resume injects into the
caller's SessionStore only; replay is read-only chain reconstruction.
Branch + thin HITL interrupt/approve per ADR-012 (INIT-012/SPEC-003).

Provenance: INIT-001/SPEC-008; INIT-012/SPEC-003.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.types import TextContent, Tool

from ._common import error_response, require_caller, require_rbac, resolve_caller, success_response

logger = logging.getLogger("archivist.mcp")

# Hard cap on JSON-serialized checkpoint payload (external input size limit).
_MAX_PAYLOAD_BYTES = 256 * 1024
_MAX_METADATA_BYTES = 64 * 1024
_MAX_REPLAY_DEPTH = 100
_MAX_LIST_LIMIT = 500

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="archivist_checkpoint_save",
        description=(
            "Persist an agent-state checkpoint for later resume/time-travel. "
            "Stores a JSON payload scoped to agent_id + session_id + namespace. "
            "Optional parent_checkpoint_id links a chain for replay. "
            "This is Phase-7 agent state — not L0–L2 memory tiers (GR-002)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Owning agent for this checkpoint (required).",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session the checkpoint belongs to (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace scope; required for RBAC isolation.",
                },
                "payload": {
                    "type": "object",
                    "description": (
                        "Checkpoint state object (goals, step, notes, memory_ids, etc.). "
                        f"JSON size limited to {_MAX_PAYLOAD_BYTES} bytes."
                    ),
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional non-state metadata (labels, step index).",
                },
                "parent_checkpoint_id": {
                    "type": "string",
                    "description": "Optional parent checkpoint id in the same namespace.",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "session_id", "namespace"],
        },
    ),
    Tool(
        name="archivist_checkpoint_list",
        description=(
            "List checkpoints for an agent session in a namespace (oldest first). "
            "Requires namespace — never lists across tenants."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Agent whose session checkpoints to list (required).",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session id to list (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace scope (required).",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max rows to return (default 100, max {_MAX_LIST_LIMIT}).",
                    "default": 100,
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "session_id", "namespace"],
        },
    ),
    Tool(
        name="archivist_checkpoint_get",
        description=(
            "Fetch one checkpoint by id, scoped to namespace. "
            "Id alone is never enough — cross-namespace get is denied."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Calling agent (required for RBAC).",
                },
                "checkpoint_id": {
                    "type": "string",
                    "description": "Checkpoint id to fetch (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace that must own the checkpoint (required).",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "checkpoint_id", "namespace"],
        },
    ),
    Tool(
        name="archivist_checkpoint_resume",
        description=(
            "Load a checkpoint and inject its resume packet into the caller's "
            "SessionStore for the given session_id only (does not mutate other agents). "
            "Only the owning agent_id may resume a checkpoint, even within a shared "
            "namespace; use archivist_handoff to transfer state between agents. "
            "Returns a packet suitable for archivist_get_context (injected_keys, "
            "extra_memory_ids, summary). Existing handoff tools remain unchanged."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Agent session that will resume (required).",
                },
                "session_id": {
                    "type": "string",
                    "description": "Target session for SessionStore injection (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace of the checkpoint (required).",
                },
                "checkpoint_id": {
                    "type": "string",
                    "description": "Checkpoint to resume from (required).",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "session_id", "namespace", "checkpoint_id"],
        },
    ),
    Tool(
        name="archivist_checkpoint_replay",
        description=(
            "Read-only reconstruction of a checkpoint parent chain (leaf → root). "
            "Returns ordered metadata + payloads without mutating SessionStore or "
            "other agents' data."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Calling agent (required for RBAC).",
                },
                "checkpoint_id": {
                    "type": "string",
                    "description": "Leaf checkpoint id to walk parents from (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace scope (required).",
                },
                "max_depth": {
                    "type": "integer",
                    "description": f"Max chain length (default {_MAX_REPLAY_DEPTH}).",
                    "default": _MAX_REPLAY_DEPTH,
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "checkpoint_id", "namespace"],
        },
    ),
    Tool(
        name="archivist_checkpoint_branch",
        description=(
            "Create a child checkpoint from a required parent in the same namespace "
            "(explicit branch UX). Owner-agent bind applies — only the parent owner "
            "may branch. Optional payload overrides the parent copy; default copies "
            "parent payload/session. Agent-state only — not L0–L2 tiers."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Owning agent (must match parent owner; required).",
                },
                "parent_checkpoint_id": {
                    "type": "string",
                    "description": "Parent checkpoint id in the same namespace (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace scope (required).",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session override; defaults to parent session.",
                },
                "payload": {
                    "type": "object",
                    "description": (
                        "Optional child payload; when omitted, copies parent payload. "
                        f"JSON size limited to {_MAX_PAYLOAD_BYTES} bytes."
                    ),
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional child metadata (HITL keys managed by interrupt/approve).",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "parent_checkpoint_id", "namespace"],
        },
    ),
    Tool(
        name="archivist_checkpoint_interrupt",
        description=(
            "Mark a checkpoint as HITL-interrupted (metadata-only). "
            "Resume fails closed until archivist_checkpoint_approve. "
            "Requires namespace write + owner-agent bind."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Owning agent (must match checkpoint owner; required).",
                },
                "checkpoint_id": {
                    "type": "string",
                    "description": "Checkpoint to interrupt (required).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace scope (required).",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional human-readable interrupt reason.",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Delegating caller identity when distinct from agent_id.",
                },
            },
            "required": ["agent_id", "checkpoint_id", "namespace"],
        },
    ),
    Tool(
        name="archivist_checkpoint_approve",
        description=(
            "Clear HITL interrupt on a checkpoint (idempotent if already approved). "
            "Required before resume when the checkpoint is interrupted. "
            "Requires namespace write + owner-agent bind."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Owning agent (must match checkpoint owner; required).",
                },
                "checkpoint_id": {
                    "type": "string",
                    "description": "Checkpoint to approve (required).",
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
            "required": ["agent_id", "checkpoint_id", "namespace"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_to_dict(record: Any) -> dict[str, Any]:
    return {
        "id": record.id,
        "agent_id": record.agent_id,
        "session_id": record.session_id,
        "namespace": record.namespace,
        "parent_checkpoint_id": record.parent_checkpoint_id,
        "payload": record.payload,
        "blob_ref": record.blob_ref,
        "metadata": record.metadata,
        "created_at": record.created_at,
    }


def _json_size_bytes(obj: Any) -> int:
    return len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def _validate_payload(payload: Any) -> list[TextContent] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return error_response(
            {"error": "invalid_payload", "reason": "payload must be a JSON object"}
        )
    size = _json_size_bytes(payload)
    if size > _MAX_PAYLOAD_BYTES:
        return error_response(
            {
                "error": "payload_too_large",
                "reason": f"payload JSON exceeds {_MAX_PAYLOAD_BYTES} bytes",
                "size_bytes": size,
                "max_bytes": _MAX_PAYLOAD_BYTES,
            }
        )
    return None


def _validate_metadata(metadata: Any) -> list[TextContent] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        return error_response(
            {"error": "invalid_metadata", "reason": "metadata must be a JSON object"}
        )
    size = _json_size_bytes(metadata)
    if size > _MAX_METADATA_BYTES:
        return error_response(
            {
                "error": "metadata_too_large",
                "reason": f"metadata JSON exceeds {_MAX_METADATA_BYTES} bytes",
                "size_bytes": size,
                "max_bytes": _MAX_METADATA_BYTES,
            }
        )
    return None


def _require_namespace(namespace: str) -> list[TextContent] | None:
    if not namespace:
        return error_response(
            {
                "error": "namespace_required",
                "reason": "namespace is required for checkpoint isolation",
            }
        )
    return None


def _map_checkpoint_error(exc: Exception) -> list[TextContent]:
    """Map storage checkpoint errors to stable MCP error shapes (no payload echo)."""
    from archivist.storage import checkpoints as ckpt

    if isinstance(exc, ckpt.CheckpointAuthzError):
        return error_response(
            {
                "error": exc.code,
                "reason": str(exc),
                "hint": "Use your own checkpoints, or archivist_handoff to transfer state.",
            }
        )
    if isinstance(exc, ckpt.CheckpointNotFoundError):
        return error_response({"error": exc.code, "reason": str(exc)})
    if isinstance(exc, ckpt.CheckpointConflictError):
        return error_response({"error": exc.code, "reason": str(exc)})
    if isinstance(exc, ckpt.CheckpointError):
        return error_response({"error": getattr(exc, "code", "checkpoint_error"), "reason": str(exc)})
    if isinstance(exc, ValueError):
        return error_response({"error": "invalid_request", "reason": str(exc)})
    raise exc


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_checkpoint_save(arguments: dict) -> list[TextContent]:
    from archivist.storage import checkpoints as ckpt

    agent_id = (arguments.get("agent_id") or "").strip()
    session_id = (arguments.get("session_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if not session_id:
        return error_response({"error": "session_id is required"})
    if err := _require_namespace(namespace):
        return err
    if denied := require_rbac(caller, "write", namespace):
        return denied

    payload = arguments.get("payload")
    if payload is None:
        payload = {}
    if err := _validate_payload(payload):
        return err
    metadata = arguments.get("metadata")
    if err := _validate_metadata(metadata):
        return err

    parent = (arguments.get("parent_checkpoint_id") or "").strip() or None
    try:
        if parent is not None:
            parent_rec = await ckpt.get_checkpoint(parent, namespace=namespace)
            if parent_rec is None:
                return error_response(
                    {
                        "error": "parent_not_found",
                        "reason": "parent_checkpoint_id not found in namespace",
                    }
                )
        record = await ckpt.create_checkpoint(
            agent_id=agent_id,
            session_id=session_id,
            namespace=namespace,
            payload=payload,
            metadata=metadata,
            parent_checkpoint_id=parent,
        )
    except ValueError as exc:
        return error_response({"error": "invalid_request", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_checkpoint_save failed")
        return error_response({"error": str(exc)})

    return success_response({"checkpoint": _record_to_dict(record)}, default=str)


async def _handle_checkpoint_list(arguments: dict) -> list[TextContent]:
    from archivist.storage import checkpoints as ckpt

    agent_id = (arguments.get("agent_id") or "").strip()
    session_id = (arguments.get("session_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if not session_id:
        return error_response({"error": "session_id is required"})
    if err := _require_namespace(namespace):
        return err
    if denied := require_rbac(caller, "read", namespace):
        return denied

    try:
        limit = int(arguments.get("limit") or 100)
    except (TypeError, ValueError):
        return error_response({"error": "invalid_limit", "reason": "limit must be an integer"})
    if limit < 1:
        return error_response({"error": "invalid_limit", "reason": "limit must be >= 1"})
    limit = min(limit, _MAX_LIST_LIMIT)

    try:
        rows = await ckpt.list_checkpoints_by_session(
            agent_id=agent_id,
            session_id=session_id,
            namespace=namespace,
            limit=limit,
        )
    except ValueError as exc:
        return error_response({"error": "invalid_request", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_checkpoint_list failed")
        return error_response({"error": str(exc)})

    return success_response(
        {
            "agent_id": agent_id,
            "session_id": session_id,
            "namespace": namespace,
            "count": len(rows),
            "checkpoints": [_record_to_dict(r) for r in rows],
        },
        default=str,
    )


async def _handle_checkpoint_get(arguments: dict) -> list[TextContent]:
    from archivist.storage import checkpoints as ckpt

    checkpoint_id = (arguments.get("checkpoint_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not checkpoint_id:
        return error_response({"error": "checkpoint_id is required"})
    if err := _require_namespace(namespace):
        return err
    if denied := require_rbac(caller, "read", namespace):
        return denied

    try:
        record = await ckpt.get_checkpoint(checkpoint_id, namespace=namespace)
    except ValueError as exc:
        return error_response({"error": "invalid_request", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_checkpoint_get failed")
        return error_response({"error": str(exc)})

    if record is None:
        # Same shape for missing vs wrong-namespace (no cross-tenant oracle).
        return error_response(
            {
                "error": "not_found",
                "reason": "checkpoint not found in namespace",
            }
        )
    return success_response({"checkpoint": _record_to_dict(record)}, default=str)


async def _handle_checkpoint_resume(arguments: dict) -> list[TextContent]:
    """Inject checkpoint resume packet into caller's SessionStore only."""
    from archivist.retrieval.session_store import get_session_store
    from archivist.storage import checkpoints as ckpt

    agent_id = (arguments.get("agent_id") or "").strip()
    session_id = (arguments.get("session_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    checkpoint_id = (arguments.get("checkpoint_id") or "").strip()
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if not session_id:
        return error_response({"error": "session_id is required"})
    if not checkpoint_id:
        return error_response({"error": "checkpoint_id is required"})
    if err := _require_namespace(namespace):
        return err
    if denied := require_rbac(caller, "read", namespace):
        return denied

    try:
        record = await ckpt.get_checkpoint(checkpoint_id, namespace=namespace)
    except ValueError as exc:
        return error_response({"error": "invalid_request", "reason": str(exc)})
    except Exception as exc:
        logger.exception("archivist_checkpoint_resume failed")
        return error_response({"error": str(exc)})

    if record is None:
        return error_response(
            {
                "error": "not_found",
                "reason": "checkpoint not found in namespace",
            }
        )

    # Owner bind + HITL interrupt gate (SEC-008-01 / ADR-012).
    try:
        ckpt.ensure_resume_allowed(record, agent_id=agent_id)
    except (ckpt.CheckpointAuthzError, ckpt.CheckpointConflictError) as exc:
        return _map_checkpoint_error(exc)

    payload = record.payload if isinstance(record.payload, dict) else {}
    summary = ""
    if isinstance(payload.get("summary"), str):
        summary = payload["summary"]
    elif isinstance(payload.get("session_summary"), str):
        summary = payload["session_summary"]

    extra_memory_ids: list[str] = []
    raw_ids = payload.get("extra_memory_ids") or payload.get("key_memory_ids") or []
    if isinstance(raw_ids, list):
        extra_memory_ids = [str(x) for x in raw_ids if x]

    # Inject only into the resume agent's session — never other agents' keys.
    ss = get_session_store()
    injected: list[str] = []
    ss.put(agent_id, session_id, "checkpoint_id", record.id)
    injected.append("checkpoint_id")
    ss.put(agent_id, session_id, "checkpoint_namespace", record.namespace)
    injected.append("checkpoint_namespace")
    if summary:
        ss.put(agent_id, session_id, "checkpoint_summary", summary)
        injected.append("checkpoint_summary")
    # Compact payload for get_context / ephemeral packer (string value API).
    ss.put(
        agent_id,
        session_id,
        "checkpoint_payload",
        json.dumps(payload, separators=(",", ":"), default=str),
    )
    injected.append("checkpoint_payload")

    goals = payload.get("active_goals") or payload.get("goals") or []
    if isinstance(goals, list):
        for i, goal in enumerate(goals):
            if not isinstance(goal, str) or not goal:
                continue
            key = f"checkpoint_goal_{i}"
            ss.put(agent_id, session_id, key, goal)
            injected.append(key)

    packet = {
        "checkpoint_id": record.id,
        "agent_id": agent_id,
        "session_id": session_id,
        "namespace": record.namespace,
        "parent_checkpoint_id": record.parent_checkpoint_id,
        "summary": summary,
        "injected_keys": injected,
        "extra_memory_ids": extra_memory_ids,
        "payload": payload,
        "metadata": record.metadata,
        "created_at": record.created_at,
        "hint": (
            "Pass extra_memory_ids to archivist_get_context to pin resumed memories; "
            "ephemeral keys are in SessionStore for this agent/session only."
        ),
    }
    return success_response({"resume_packet": packet}, default=str)


async def _handle_checkpoint_replay(arguments: dict) -> list[TextContent]:
    """Read-only walk of parent chain: leaf → root."""
    from archivist.storage import checkpoints as ckpt

    checkpoint_id = (arguments.get("checkpoint_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not checkpoint_id:
        return error_response({"error": "checkpoint_id is required"})
    if err := _require_namespace(namespace):
        return err
    if denied := require_rbac(caller, "read", namespace):
        return denied

    try:
        max_depth = int(arguments.get("max_depth") or _MAX_REPLAY_DEPTH)
    except (TypeError, ValueError):
        return error_response(
            {"error": "invalid_max_depth", "reason": "max_depth must be an integer"}
        )
    if max_depth < 1:
        return error_response({"error": "invalid_max_depth", "reason": "max_depth must be >= 1"})
    max_depth = min(max_depth, _MAX_REPLAY_DEPTH)

    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_id: str | None = checkpoint_id
    try:
        while current_id and len(chain) < max_depth:
            if current_id in seen:
                return error_response(
                    {
                        "error": "cycle_detected",
                        "reason": "parent chain contains a cycle",
                        "checkpoint_id": current_id,
                    }
                )
            seen.add(current_id)
            record = await ckpt.get_checkpoint(current_id, namespace=namespace)
            if record is None:
                if not chain:
                    return error_response(
                        {
                            "error": "not_found",
                            "reason": "checkpoint not found in namespace",
                        }
                    )
                break
            chain.append(_record_to_dict(record))
            current_id = record.parent_checkpoint_id
    except Exception as exc:
        logger.exception("archivist_checkpoint_replay failed")
        return error_response({"error": str(exc)})

    # Present root → leaf for chronological reconstruction.
    chain_root_to_leaf = list(reversed(chain))
    return success_response(
        {
            "namespace": namespace,
            "leaf_checkpoint_id": checkpoint_id,
            "depth": len(chain_root_to_leaf),
            "readonly": True,
            "chain": chain_root_to_leaf,
        },
        default=str,
    )


async def _handle_checkpoint_branch(arguments: dict) -> list[TextContent]:
    """Create a child checkpoint from a required parent (ADR-012)."""
    from archivist.storage import checkpoints as ckpt

    agent_id = (arguments.get("agent_id") or "").strip()
    parent_id = (arguments.get("parent_checkpoint_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    session_id = (arguments.get("session_id") or "").strip() or None
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if not parent_id:
        return error_response({"error": "parent_checkpoint_id is required"})
    if err := _require_namespace(namespace):
        return err
    if denied := require_rbac(caller, "write", namespace):
        return denied

    payload = arguments.get("payload")
    if payload is not None:
        if err := _validate_payload(payload):
            return err
    metadata = arguments.get("metadata")
    if err := _validate_metadata(metadata):
        return err

    try:
        record = await ckpt.branch_checkpoint(
            parent_checkpoint_id=parent_id,
            namespace=namespace,
            agent_id=agent_id,
            session_id=session_id,
            payload=payload,
            metadata=metadata,
        )
    except (
        ckpt.CheckpointAuthzError,
        ckpt.CheckpointNotFoundError,
        ckpt.CheckpointConflictError,
        ckpt.CheckpointError,
        ValueError,
    ) as exc:
        return _map_checkpoint_error(exc)
    except Exception as exc:
        logger.exception("archivist_checkpoint_branch failed")
        return error_response({"error": str(exc)})

    return success_response({"checkpoint": _record_to_dict(record)}, default=str)


async def _handle_checkpoint_interrupt(arguments: dict) -> list[TextContent]:
    """Mark checkpoint HITL-interrupted (metadata-only)."""
    from archivist.storage import checkpoints as ckpt

    agent_id = (arguments.get("agent_id") or "").strip()
    checkpoint_id = (arguments.get("checkpoint_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    reason = arguments.get("reason")
    if reason is not None and not isinstance(reason, str):
        return error_response({"error": "invalid_reason", "reason": "reason must be a string"})
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if not checkpoint_id:
        return error_response({"error": "checkpoint_id is required"})
    if err := _require_namespace(namespace):
        return err
    if denied := require_rbac(caller, "write", namespace):
        return denied

    try:
        record = await ckpt.interrupt_checkpoint(
            checkpoint_id,
            namespace=namespace,
            agent_id=agent_id,
            reason=reason,
            actor=caller,
        )
    except (
        ckpt.CheckpointAuthzError,
        ckpt.CheckpointNotFoundError,
        ckpt.CheckpointConflictError,
        ckpt.CheckpointError,
        ValueError,
    ) as exc:
        return _map_checkpoint_error(exc)
    except Exception as exc:
        logger.exception("archivist_checkpoint_interrupt failed")
        return error_response({"error": str(exc)})

    return success_response({"checkpoint": _record_to_dict(record)}, default=str)


async def _handle_checkpoint_approve(arguments: dict) -> list[TextContent]:
    """Clear HITL interrupt so resume may proceed."""
    from archivist.storage import checkpoints as ckpt

    agent_id = (arguments.get("agent_id") or "").strip()
    checkpoint_id = (arguments.get("checkpoint_id") or "").strip()
    namespace = (arguments.get("namespace") or "").strip()
    caller = resolve_caller(arguments)

    if err := require_caller(caller):
        return err
    if not agent_id:
        return error_response({"error": "agent_id is required"})
    if not checkpoint_id:
        return error_response({"error": "checkpoint_id is required"})
    if err := _require_namespace(namespace):
        return err
    if denied := require_rbac(caller, "write", namespace):
        return denied

    try:
        record = await ckpt.approve_checkpoint(
            checkpoint_id,
            namespace=namespace,
            agent_id=agent_id,
            actor=caller,
        )
    except (
        ckpt.CheckpointAuthzError,
        ckpt.CheckpointNotFoundError,
        ckpt.CheckpointConflictError,
        ckpt.CheckpointError,
        ValueError,
    ) as exc:
        return _map_checkpoint_error(exc)
    except Exception as exc:
        logger.exception("archivist_checkpoint_approve failed")
        return error_response({"error": str(exc)})

    return success_response({"checkpoint": _record_to_dict(record)}, default=str)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, object] = {
    "archivist_checkpoint_save": _handle_checkpoint_save,
    "archivist_checkpoint_list": _handle_checkpoint_list,
    "archivist_checkpoint_get": _handle_checkpoint_get,
    "archivist_checkpoint_resume": _handle_checkpoint_resume,
    "archivist_checkpoint_replay": _handle_checkpoint_replay,
    "archivist_checkpoint_branch": _handle_checkpoint_branch,
    "archivist_checkpoint_interrupt": _handle_checkpoint_interrupt,
    "archivist_checkpoint_approve": _handle_checkpoint_approve,
}
