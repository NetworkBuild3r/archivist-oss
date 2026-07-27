"""MCP tool handlers — Phase 10 selective share + Diff #5 productize.

Extends multi-agent coordination without replacing ``archivist_handoff`` /
``archivist_receive_handoff`` (GR-HANDOFF-001 / GR-003). Consensus v1 =
explicit accept/reject plus audit trail; conflict outcomes use
``contradiction_resolve.ResolutionAction`` types and may optionally apply
a resolution (dry-run by default).

Procedural lessons: tips travel in ``HandoffPacket``; selective share may
also carry ``tip_ids`` (metadata) alongside ``memory_ids`` / ``scope``.

Security: propose cannot escalate beyond proposer's read rights; accept cannot
write into an unauthorized namespace; every mutation is audited.

Provenance: INIT-001/SPEC-010; INIT-009/SPEC-002 (ops profile + conflict wire);
INIT-009/SPEC-005 (apply write gate + tip_ids metadata hardening).
"""

from __future__ import annotations

import json
import logging
from typing import Any, get_args

from mcp.types import TextContent, Tool

from archivist.lifecycle.contradiction_resolve import ResolutionAction

from ._common import (
    error_response,
    require_caller,
    require_rbac,
    resolve_caller,
    success_response,
)

logger = logging.getLogger("archivist.mcp")

# Single source of truth with lifecycle/contradiction_resolve (INIT-009/SPEC-002).
_RESOLUTION_ACTIONS = frozenset(get_args(ResolutionAction))
_MAX_MEMORY_IDS = 500
_MAX_TIP_IDS = 500
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
            "Does not replace archivist_handoff (GR-HANDOFF-001). "
            "Procedural lessons: tip strings travel in HandoffPacket; optional tip_ids "
            "on this propose carry tip identifiers for selective lesson share. "
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
                    "description": "Selective memory IDs to share (optional if scope/tip_ids set).",
                },
                "tip_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional tip/lesson IDs to share (stored on grant metadata; "
                        "handoff remains the primary tip-transfer channel)."
                    ),
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
            "Attach a conflict/consensus outcome to a share grant using "
            "contradiction_resolve ResolutionAction types: supersede | merge | keep_both. "
            "Proposer or recipient may attach; namespace read required. "
            "Optional apply=true builds a ResolutionProposal and calls apply_resolution "
            "(dry_run defaults true). Mutating apply (dry_run=false) requires namespace "
            "write, CONTRADICTION_RESOLVE_ENABLED, and facts bound to entity_id+namespace."
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
                    "description": "ResolutionAction (required).",
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
                "entity_id": {
                    "type": "integer",
                    "description": "Entity id required when apply=true.",
                },
                "apply": {
                    "type": "boolean",
                    "description": (
                        "When true, invoke contradiction_resolve.apply_resolution "
                        "with a proposal built from this outcome (default false)."
                    ),
                },
                "dry_run": {
                    "type": "boolean",
                    "description": (
                        "When apply=true, dry-run the resolution (default true). "
                        "dry_run=false requires namespace write + resolve enabled."
                    ),
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
    return _normalize_id_list(raw, field="memory_ids", max_count=_MAX_MEMORY_IDS)


def _normalize_id_list(
    raw: Any,
    *,
    field: str,
    max_count: int,
) -> tuple[list[str] | None, list[TextContent] | None]:
    """Normalize a string-id array field (memory_ids / tip_ids)."""
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return None, error_response(
            {
                "error": f"invalid_{field}",
                "reason": f"{field} must be an array of strings",
            }
        )
    if len(raw) > max_count:
        return None, error_response(
            {
                "error": f"{field}_too_many",
                "reason": f"{field} exceeds {max_count}",
                "count": len(raw),
                "max": max_count,
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


async def _facts_bound_to_entity_namespace(
    fact_ids: list[int],
    *,
    entity_id: int,
    namespace: str,
) -> list[TextContent] | None:
    """Reject apply when any fact is missing or outside entity_id+namespace (SEC-009-01)."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        for fid in fact_ids:
            row = await conn.fetchone(
                "SELECT id, entity_id, namespace FROM facts WHERE id=?",
                (fid,),
            )
            if row is None:
                return error_response(
                    {
                        "error": "fact_not_found",
                        "reason": f"fact id {fid} does not exist",
                        "fact_id": fid,
                    }
                )
            if int(row["entity_id"]) != int(entity_id):
                return error_response(
                    {
                        "error": "fact_entity_mismatch",
                        "reason": "fact does not belong to entity_id",
                        "fact_id": fid,
                        "entity_id": entity_id,
                    }
                )
            if (row["namespace"] or "") != namespace:
                return error_response(
                    {
                        "error": "fact_namespace_mismatch",
                        "reason": "fact does not belong to grant namespace",
                        "fact_id": fid,
                        "namespace": namespace,
                    }
                )
    return None


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

    tip_ids, tip_err = _normalize_id_list(
        arguments.get("tip_ids"),
        field="tip_ids",
        max_count=_MAX_TIP_IDS,
    )
    if tip_err is not None:
        return tip_err
    assert tip_ids is not None

    scope = (arguments.get("scope") or "").strip()
    if not memory_ids and not scope and not tip_ids:
        return error_response(
            {
                "error": "share_target_required",
                "reason": "Provide memory_ids, tip_ids, and/or scope for a selective share",
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
        metadata = dict(metadata)
        # SEC-009-02: tip_ids are server-set only from the tip_ids argument —
        # ignore client-supplied metadata.tip_ids to prevent limit bypass.
        metadata.pop("tip_ids", None)
        metadata.pop("lesson_channel", None)
    else:
        metadata = {}

    # Tip/lesson share path (INIT-009/SPEC-002): tip_ids ride on grant metadata;
    # handoff remains the primary tip-transfer channel (GR-HANDOFF-001).
    if tip_ids:
        metadata["tip_ids"] = tip_ids
        metadata["lesson_channel"] = "tips"

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
            "tip_id_count": len(tip_ids),
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
        tip_ids = record.metadata.get("tip_ids") if isinstance(record.metadata, dict) else None
        if isinstance(tip_ids, list) and tip_ids:
            ss.put(
                recipient,
                session_id,
                "share_tip_ids",
                json.dumps(tip_ids, separators=(",", ":")),
            )
            injected.append("share_tip_ids")

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
                "Pass share_memory_ids to archivist_get_context as extra_memory_ids. "
                "Procedural lessons: tips travel via archivist_handoff / "
                "archivist_receive_handoff; tip_ids on the grant metadata are the "
                "selective tip-share channel (INIT-009)."
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

    apply = bool(arguments.get("apply", False))
    resolution_payload: dict[str, Any] | None = None
    dry_run_effective: bool | None = None
    pending_apply: dict[str, Any] | None = None
    if apply:
        from archivist.core.config import CONTRADICTION_RESOLVE_ENABLED

        entity_raw = arguments.get("entity_id")
        try:
            entity_id = int(entity_raw)
        except (TypeError, ValueError):
            return error_response(
                {
                    "error": "entity_id_required",
                    "reason": "entity_id (int) is required when apply=true",
                }
            )
        fact_a = arguments.get("winner_fact_id")
        fact_b = arguments.get("loser_fact_id")
        if action == "merge" and not (merge_text or "").strip():
            return error_response(
                {
                    "error": "merge_text_required",
                    "reason": "merge_text is required when apply=true and action=merge",
                }
            )
        try:
            fact_a_id = int(fact_a)
            fact_b_id = int(fact_b)
        except (TypeError, ValueError):
            return error_response(
                {
                    "error": "fact_ids_required",
                    "reason": (
                        "winner_fact_id and loser_fact_id (int pair members) "
                        "are required when apply=true"
                    ),
                }
            )

        dry_run_effective = (
            True if arguments.get("dry_run") is None else bool(arguments.get("dry_run"))
        )
        # SEC-009-01 / SEC-009-04: mutating apply needs write RBAC + master resolve flag.
        if not dry_run_effective:
            if denied := require_rbac(caller, "write", namespace):
                return denied
            if not CONTRADICTION_RESOLVE_ENABLED:
                return error_response(
                    {
                        "error": "resolve_disabled",
                        "reason": (
                            "CONTRADICTION_RESOLVE_ENABLED must be true for "
                            "apply=true with dry_run=false"
                        ),
                    }
                )
            if bound_err := await _facts_bound_to_entity_namespace(
                [fact_a_id, fact_b_id],
                entity_id=entity_id,
                namespace=namespace,
            ):
                return bound_err

        pending_apply = {
            "action": action,
            "entity_id": entity_id,
            "fact_a_id": fact_a_id,
            "fact_b_id": fact_b_id,
            "merge_text": merge_text,
            "dry_run": dry_run_effective,
        }
        outcome["resolution"] = {
            "pending_apply": True,
            "dry_run": dry_run_effective,
            "entity_id": entity_id,
            "fact_a_id": fact_a_id,
            "fact_b_id": fact_b_id,
        }

    # SEC-009-03: persist conflict outcome before any graph mutation.
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

    if pending_apply is not None:
        from archivist.lifecycle.contradiction_resolve import (
            ResolutionProposal,
            apply_resolution,
        )

        proposal = ResolutionProposal(
            action=pending_apply["action"],  # type: ignore[arg-type]
            entity_id=pending_apply["entity_id"],
            namespace=namespace,
            fact_a_id=pending_apply["fact_a_id"],
            fact_b_id=pending_apply["fact_b_id"],
            winner_fact_id=(
                pending_apply["fact_a_id"] if pending_apply["action"] == "supersede" else None
            ),
            loser_fact_id=(
                pending_apply["fact_b_id"] if pending_apply["action"] == "supersede" else None
            ),
            merge_text=(
                pending_apply["merge_text"] if pending_apply["action"] == "merge" else None
            ),
            reason=outcome["reason"] or "share_attach_conflict apply",
            rule="share_attach_conflict",
            trigger="share_grant",
        )
        try:
            applied = await apply_resolution(
                proposal,
                dry_run=pending_apply["dry_run"],
                agent_id=caller,
            )
        except Exception as exc:
            logger.exception("share_attach_conflict apply_resolution failed")
            return error_response(
                {
                    "error": "resolution_apply_failed",
                    "reason": str(exc),
                    "grant": sg.record_to_dict(record),
                    "partial": "conflict_outcome_attached",
                }
            )
        resolution_payload = applied.to_dict()

    await _audit_share(
        agent_id=caller,
        action="share_attach_conflict",
        grant_id=record.id,
        namespace=namespace,
        metadata={
            "resolution_action": action,
            "grant_status": record.status,
            "apply": apply,
            "dry_run": dry_run_effective,
        },
    )
    payload: dict[str, Any] = {"grant": sg.record_to_dict(record)}
    if resolution_payload is not None:
        payload["resolution"] = resolution_payload
    return success_response(payload, default=str)


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
