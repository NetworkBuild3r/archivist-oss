"""Agent-state checkpoint store (Phase 7).

Persists LangGraph-style resume/time-travel checkpoints in the graph backend.
This store is **not** the Answer Finder L0–L2 ``tier_label`` taxonomy (GR-002).

v1 stores checkpoint payloads as JSON text in-row; ``blob_ref`` is reserved for
optional out-of-row large blobs later (aud-1).

All reads that accept a namespace require an explicit namespace filter so there
is no default public cross-namespace listing. Callers (SPEC-008 MCP tools /
RBAC) must pass the authorized namespace.

Provenance: INIT-001/SPEC-007.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("archivist.checkpoints")


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
    payload_json = json.dumps(payload or {}, separators=(",", ":"))
    metadata_json = json.dumps(metadata or {}, separators=(",", ":"))

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
        metadata=metadata or {},
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
