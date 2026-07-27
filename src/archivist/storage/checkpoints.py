"""Agent-state checkpoint store (Phase 7).

Persists LangGraph-style resume/time-travel checkpoints in the graph backend.
This store is **not** the Answer Finder L0–L2 ``tier_label`` taxonomy (GR-002).

v1 stores checkpoint payloads as JSON text in-row; ``blob_ref`` is reserved for
optional out-of-row large blobs later (aud-1).

All reads that accept a namespace require an explicit namespace filter so there
is no default public cross-namespace listing. Callers (SPEC-008 MCP tools /
RBAC) must pass the authorized namespace.

Branch + thin HITL interrupt/approve helpers: INIT-012/SPEC-002 (ADR-012).

Provenance: INIT-001/SPEC-007; INIT-012/SPEC-002.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("archivist.checkpoints")

# ADR-012 HITL metadata keys (metadata-only; no DDL).
HITL_STATUS_KEY = "hitl_status"
HITL_STATUS_NONE = "none"
HITL_STATUS_INTERRUPTED = "interrupted"
HITL_STATUS_APPROVED = "approved"
HITL_REASON_KEY = "hitl_reason"
HITL_INTERRUPTED_AT_KEY = "hitl_interrupted_at"
HITL_APPROVED_AT_KEY = "hitl_approved_at"
HITL_ACTOR_KEY = "hitl_actor"

_HITL_METADATA_KEYS: frozenset[str] = frozenset(
    {
        HITL_STATUS_KEY,
        HITL_REASON_KEY,
        HITL_INTERRUPTED_AT_KEY,
        HITL_APPROVED_AT_KEY,
        HITL_ACTOR_KEY,
    }
)


def _sanitize_client_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Drop client-supplied HITL keys (SEC-012-02) — only interrupt/approve may set them."""
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items() if k not in _HITL_METADATA_KEYS}


class CheckpointError(Exception):
    """Base checkpoint service error (stable ``code`` for MCP mapping)."""

    code = "checkpoint_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class CheckpointNotFoundError(LookupError, CheckpointError):
    """Checkpoint missing or not visible in the requested namespace."""

    code = "not_found"


class CheckpointAuthzError(PermissionError, CheckpointError):
    """Owner-agent bind failure (SEC-008-01 parity)."""

    code = "access_denied"


class CheckpointConflictError(CheckpointError):
    """Illegal state transition (e.g. resume while HITL interrupted)."""

    code = "conflict"


@dataclass(frozen=True)
class CheckpointRecord:
    """One row from ``agent_checkpoints``."""

    id: str
    agent_id: str
    session_id: str
    namespace: str
    parent_checkpoint_id: str | None
    payload: dict[str, Any]
    blob_ref: str | None
    metadata: dict[str, Any]
    created_at: str


def hitl_status(record: CheckpointRecord) -> str:
    """Return normalized HITL status from metadata (default ``none``)."""
    raw = record.metadata.get(HITL_STATUS_KEY) if isinstance(record.metadata, dict) else None
    if raw in (HITL_STATUS_INTERRUPTED, HITL_STATUS_APPROVED, HITL_STATUS_NONE):
        return str(raw)
    return HITL_STATUS_NONE


def ensure_resume_allowed(record: CheckpointRecord, *, agent_id: str) -> None:
    """Fail closed when resume is blocked by owner bind or HITL interrupt.

    Raises:
        CheckpointAuthzError: ``agent_id`` is not the checkpoint owner.
        CheckpointConflictError: ``hitl_status`` is ``interrupted``.
    """
    _require_owner(record, agent_id)
    if hitl_status(record) == HITL_STATUS_INTERRUPTED:
        raise CheckpointConflictError(
            "checkpoint is interrupted; approve before resume",
            code="hitl_interrupted",
        )


def _require_owner(record: CheckpointRecord, agent_id: str) -> None:
    if not agent_id or not agent_id.strip():
        raise ValueError("agent_id is required")
    if record.agent_id and record.agent_id != agent_id:
        raise CheckpointAuthzError(
            "checkpoint belongs to another agent in this namespace",
        )


def _parse_json_object(raw: str | None, *, field: str, checkpoint_id: str) -> dict[str, Any]:
    """Parse a JSON object column; never log the raw payload body."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "checkpoint %s: invalid JSON in %s (len=%d)",
            checkpoint_id,
            field,
            len(raw),
        )
        return {}
    if not isinstance(value, dict):
        logger.warning(
            "checkpoint %s: %s is not a JSON object (type=%s)",
            checkpoint_id,
            field,
            type(value).__name__,
        )
        return {}
    return value


def _row_to_record(row: Any) -> CheckpointRecord:
    data = dict(row)
    cid = data["id"]
    return CheckpointRecord(
        id=cid,
        agent_id=data["agent_id"],
        session_id=data["session_id"],
        namespace=data["namespace"],
        parent_checkpoint_id=data.get("parent_checkpoint_id"),
        payload=_parse_json_object(data.get("payload"), field="payload", checkpoint_id=cid),
        blob_ref=data.get("blob_ref"),
        metadata=_parse_json_object(data.get("metadata"), field="metadata", checkpoint_id=cid),
        created_at=data["created_at"],
    )


async def create_checkpoint(
    *,
    agent_id: str,
    session_id: str,
    namespace: str,
    payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    parent_checkpoint_id: str | None = None,
    blob_ref: str | None = None,
    checkpoint_id: str | None = None,
) -> CheckpointRecord:
    """Insert a checkpoint row and return the stored record.

    ``namespace`` is required (RBAC-ready). Payload/metadata are serialised as
    compact JSON text; neither is logged at info level.
    """
    from archivist.storage.sqlite_pool import pool

    if not agent_id or not session_id or not namespace:
        raise ValueError("agent_id, session_id, and namespace are required")

    cid = checkpoint_id or str(uuid.uuid4())
    created_at = datetime.now(UTC).isoformat()
    safe_metadata = _sanitize_client_metadata(metadata)
    safe_metadata.setdefault(HITL_STATUS_KEY, HITL_STATUS_NONE)
    payload_json = json.dumps(payload or {}, separators=(",", ":"))
    metadata_json = json.dumps(safe_metadata, separators=(",", ":"))

    async with pool.write() as conn:
        await conn.execute(
            """
            INSERT INTO agent_checkpoints (
                id, agent_id, session_id, namespace, parent_checkpoint_id,
                payload, blob_ref, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cid,
                agent_id,
                session_id,
                namespace,
                parent_checkpoint_id,
                payload_json,
                blob_ref,
                metadata_json,
                created_at,
            ),
        )

    logger.debug(
        "created checkpoint id=%s agent_id=%s session_id=%s namespace=%s parent=%s",
        cid,
        agent_id,
        session_id,
        namespace,
        parent_checkpoint_id,
    )
    return CheckpointRecord(
        id=cid,
        agent_id=agent_id,
        session_id=session_id,
        namespace=namespace,
        parent_checkpoint_id=parent_checkpoint_id,
        payload=payload or {},
        blob_ref=blob_ref,
        metadata=safe_metadata,
        created_at=created_at,
    )


async def get_checkpoint(
    checkpoint_id: str,
    *,
    namespace: str,
) -> CheckpointRecord | None:
    """Fetch one checkpoint by id, scoped to *namespace*.

    Returns ``None`` when the id is missing or belongs to another namespace
    (no cross-namespace leakage).
    """
    from archivist.storage.sqlite_pool import pool

    if not checkpoint_id or not namespace:
        raise ValueError("checkpoint_id and namespace are required")

    async with pool.read() as conn:
        cur = await conn.execute(
            """
            SELECT id, agent_id, session_id, namespace, parent_checkpoint_id,
                   payload, blob_ref, metadata, created_at
            FROM agent_checkpoints
            WHERE id = ? AND namespace = ?
            """,
            (checkpoint_id, namespace),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    return _row_to_record(row)


async def list_checkpoints_by_session(
    *,
    agent_id: str,
    session_id: str,
    namespace: str,
    limit: int = 100,
) -> list[CheckpointRecord]:
    """List checkpoints for an agent session in created_at order (oldest first).

    Requires *namespace* so listing never defaults to a public cross-namespace read.
    Uses ``idx_checkpoints_agent_session_time``.
    """
    from archivist.storage.sqlite_pool import pool

    if not agent_id or not session_id or not namespace:
        raise ValueError("agent_id, session_id, and namespace are required")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    async with pool.read() as conn:
        cur = await conn.execute(
            """
            SELECT id, agent_id, session_id, namespace, parent_checkpoint_id,
                   payload, blob_ref, metadata, created_at
            FROM agent_checkpoints
            WHERE agent_id = ? AND session_id = ? AND namespace = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (agent_id, session_id, namespace, limit),
        )
        rows = await cur.fetchall()
    return [_row_to_record(r) for r in rows]


async def link_parent(
    checkpoint_id: str,
    parent_checkpoint_id: str,
    *,
    namespace: str,
) -> CheckpointRecord | None:
    """Set ``parent_checkpoint_id`` on an existing checkpoint (same namespace).

    Both rows must exist in *namespace*. Returns the updated child record, or
    ``None`` if the child is missing / out of namespace. Raises ``ValueError``
    if the parent is missing, out of namespace, or equals the child id.
    """
    from archivist.storage.sqlite_pool import pool

    if not checkpoint_id or not parent_checkpoint_id or not namespace:
        raise ValueError("checkpoint_id, parent_checkpoint_id, and namespace are required")
    if checkpoint_id == parent_checkpoint_id:
        raise ValueError("checkpoint cannot be its own parent")

    parent = await get_checkpoint(parent_checkpoint_id, namespace=namespace)
    if parent is None:
        raise ValueError("parent checkpoint not found in namespace")

    async with pool.write() as conn:
        cur = await conn.execute(
            """
            UPDATE agent_checkpoints
            SET parent_checkpoint_id = ?
            WHERE id = ? AND namespace = ?
            """,
            (parent_checkpoint_id, checkpoint_id, namespace),
        )
        if cur.rowcount == 0:
            return None

    logger.debug(
        "linked checkpoint id=%s parent=%s namespace=%s",
        checkpoint_id,
        parent_checkpoint_id,
        namespace,
    )
    return await get_checkpoint(checkpoint_id, namespace=namespace)


async def _write_metadata(
    checkpoint_id: str,
    *,
    namespace: str,
    metadata: dict[str, Any],
) -> CheckpointRecord:
    """Replace metadata JSON for a checkpoint in *namespace* (ids-only logging)."""
    from archivist.storage.sqlite_pool import pool

    metadata_json = json.dumps(metadata, separators=(",", ":"))
    async with pool.write() as conn:
        cur = await conn.execute(
            """
            UPDATE agent_checkpoints
            SET metadata = ?
            WHERE id = ? AND namespace = ?
            """,
            (metadata_json, checkpoint_id, namespace),
        )
        if cur.rowcount == 0:
            raise CheckpointNotFoundError("checkpoint not found in namespace")

    updated = await get_checkpoint(checkpoint_id, namespace=namespace)
    if updated is None:
        raise CheckpointNotFoundError("checkpoint not found in namespace")
    return updated


async def branch_checkpoint(
    *,
    parent_checkpoint_id: str,
    namespace: str,
    agent_id: str,
    session_id: str | None = None,
    payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> CheckpointRecord:
    """Create a child checkpoint from a required parent (ADR-012 branch UX).

    Parent must exist in *namespace*. ``agent_id`` must match the parent's owner
    (SEC-008-01 parity). Default session and payload copy from the parent when
    omitted. Never logs payload bodies.
    """
    if not parent_checkpoint_id or not namespace or not agent_id:
        raise ValueError("parent_checkpoint_id, namespace, and agent_id are required")

    parent = await get_checkpoint(parent_checkpoint_id, namespace=namespace)
    if parent is None:
        raise CheckpointNotFoundError("parent checkpoint not found in namespace")

    _require_owner(parent, agent_id)

    child_session = (session_id or parent.session_id or "").strip()
    if not child_session:
        raise ValueError("session_id is required when parent has no session_id")

    child_payload = parent.payload if payload is None else payload
    child_meta = _sanitize_client_metadata(metadata)
    child_meta.setdefault(HITL_STATUS_KEY, HITL_STATUS_NONE)

    child = await create_checkpoint(
        agent_id=parent.agent_id,
        session_id=child_session,
        namespace=namespace,
        payload=child_payload,
        metadata=child_meta,
        parent_checkpoint_id=parent.id,
    )
    logger.info(
        "branched checkpoint id=%s parent=%s agent_id=%s namespace=%s",
        child.id,
        parent.id,
        parent.agent_id,
        namespace,
    )
    return child


async def interrupt_checkpoint(
    checkpoint_id: str,
    *,
    namespace: str,
    agent_id: str,
    reason: str | None = None,
    actor: str | None = None,
) -> CheckpointRecord:
    """Mark a checkpoint HITL-interrupted (metadata-only; ADR-012)."""
    if not checkpoint_id or not namespace or not agent_id:
        raise ValueError("checkpoint_id, namespace, and agent_id are required")

    record = await get_checkpoint(checkpoint_id, namespace=namespace)
    if record is None:
        raise CheckpointNotFoundError("checkpoint not found in namespace")

    _require_owner(record, agent_id)

    now = datetime.now(UTC).isoformat()
    meta = dict(record.metadata or {})
    meta[HITL_STATUS_KEY] = HITL_STATUS_INTERRUPTED
    meta[HITL_INTERRUPTED_AT_KEY] = now
    if reason is not None:
        meta[HITL_REASON_KEY] = reason
    meta[HITL_ACTOR_KEY] = actor or agent_id

    updated = await _write_metadata(checkpoint_id, namespace=namespace, metadata=meta)
    logger.info(
        "interrupted checkpoint id=%s agent_id=%s namespace=%s hitl_status=%s",
        checkpoint_id,
        agent_id,
        namespace,
        HITL_STATUS_INTERRUPTED,
    )
    return updated


async def approve_checkpoint(
    checkpoint_id: str,
    *,
    namespace: str,
    agent_id: str,
    actor: str | None = None,
) -> CheckpointRecord:
    """Clear HITL interrupt (idempotent if already approved / none).

    Owner-agent bind required. Resume is allowed after this clears
    ``hitl_status=interrupted``.
    """
    if not checkpoint_id or not namespace or not agent_id:
        raise ValueError("checkpoint_id, namespace, and agent_id are required")

    record = await get_checkpoint(checkpoint_id, namespace=namespace)
    if record is None:
        raise CheckpointNotFoundError("checkpoint not found in namespace")

    _require_owner(record, agent_id)

    status = hitl_status(record)
    if status == HITL_STATUS_APPROVED:
        return record

    now = datetime.now(UTC).isoformat()
    meta = dict(record.metadata or {})
    meta[HITL_STATUS_KEY] = HITL_STATUS_APPROVED
    meta[HITL_APPROVED_AT_KEY] = now
    meta[HITL_ACTOR_KEY] = actor or agent_id

    updated = await _write_metadata(checkpoint_id, namespace=namespace, metadata=meta)
    logger.info(
        "approved checkpoint id=%s agent_id=%s namespace=%s prior_hitl_status=%s",
        checkpoint_id,
        agent_id,
        namespace,
        status,
    )
    return updated
