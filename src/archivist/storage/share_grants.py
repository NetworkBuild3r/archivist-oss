"""Selective memory share grants (Phase 10 coordination).

Minimal persistence for propose / accept / reject / conflict-outcome
attachments. Extends handoff — does not replace ``HandoffPacket`` (GR-003).

Consensus v1 = explicit accept/reject + audit (not distributed Paxos).

Provenance: INIT-001/SPEC-010.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from archivist.storage.graph_schema import schema_guard

logger = logging.getLogger("archivist.share_grants")

ShareStatus = Literal["pending", "accepted", "rejected"]

# Idempotent SQLite init; Postgres uses schema_postgres.sql via init_schema_async.
_ensure_share_grants_schema = schema_guard("""
    CREATE TABLE IF NOT EXISTS memory_share_grants (
        id                  TEXT PRIMARY KEY,
        proposer_agent_id   TEXT NOT NULL,
        recipient_agent_id  TEXT NOT NULL,
        namespace            TEXT NOT NULL,
        memory_ids          TEXT NOT NULL DEFAULT '[]',
        scope               TEXT NOT NULL DEFAULT '',
        status              TEXT NOT NULL DEFAULT 'pending',
        conflict_outcome    TEXT,
        reason              TEXT NOT NULL DEFAULT '',
        metadata            TEXT NOT NULL DEFAULT '{}',
        created_at          TEXT NOT NULL,
        decided_at          TEXT,
        decided_by          TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_share_grants_recipient_status
        ON memory_share_grants(recipient_agent_id, status);
    CREATE INDEX IF NOT EXISTS idx_share_grants_proposer
        ON memory_share_grants(proposer_agent_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_share_grants_namespace
        ON memory_share_grants(namespace);
""")


@dataclass(frozen=True)
class ShareGrantRecord:
    """One row from ``memory_share_grants``."""

    id: str
    proposer_agent_id: str
    recipient_agent_id: str
    namespace: str
    memory_ids: list[str]
    scope: str
    status: str
    conflict_outcome: dict[str, Any] | None
    reason: str
    metadata: dict[str, Any]
    created_at: str
    decided_at: str | None
    decided_by: str | None


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if x is not None and str(x)]


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _row_to_record(row: Any) -> ShareGrantRecord:
    data = dict(row)
    outcome_raw = data.get("conflict_outcome")
    outcome: dict[str, Any] | None
    if outcome_raw is None or outcome_raw == "":
        outcome = None
    else:
        parsed = _parse_json_object(outcome_raw)
        outcome = parsed or None
    return ShareGrantRecord(
        id=data["id"],
        proposer_agent_id=data["proposer_agent_id"],
        recipient_agent_id=data["recipient_agent_id"],
        namespace=data["namespace"],
        memory_ids=_parse_json_list(data.get("memory_ids")),
        scope=data.get("scope") or "",
        status=data["status"],
        conflict_outcome=outcome,
        reason=data.get("reason") or "",
        metadata=_parse_json_object(data.get("metadata")),
        created_at=data["created_at"],
        decided_at=data.get("decided_at"),
        decided_by=data.get("decided_by"),
    )


def ensure_share_grants_schema() -> None:
    """Public hook for tests / callers that need explicit schema init."""
    _ensure_share_grants_schema()


async def create_share_grant(
    *,
    proposer_agent_id: str,
    recipient_agent_id: str,
    namespace: str,
    memory_ids: list[str] | None = None,
    scope: str = "",
    reason: str = "",
    metadata: dict[str, Any] | None = None,
    grant_id: str | None = None,
) -> ShareGrantRecord:
    """Insert a pending share grant."""
    from archivist.storage.sqlite_pool import pool

    _ensure_share_grants_schema()
    if not proposer_agent_id or not recipient_agent_id or not namespace:
        raise ValueError("proposer_agent_id, recipient_agent_id, and namespace are required")

    ids = [str(x) for x in (memory_ids or []) if x is not None and str(x)]
    scope_s = (scope or "").strip()
    if not ids and not scope_s:
        raise ValueError("memory_ids or scope is required")

    gid = grant_id or str(uuid.uuid4())
    created_at = datetime.now(UTC).isoformat()
    memory_json = json.dumps(ids, separators=(",", ":"))
    meta_json = json.dumps(metadata or {}, separators=(",", ":"))

    async with pool.write() as conn:
        await conn.execute(
            """
            INSERT INTO memory_share_grants (
                id, proposer_agent_id, recipient_agent_id, namespace,
                memory_ids, scope, status, conflict_outcome, reason,
                metadata, created_at, decided_at, decided_by
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, ?, ?, ?, NULL, NULL)
            """,
            (
                gid,
                proposer_agent_id,
                recipient_agent_id,
                namespace,
                memory_json,
                scope_s,
                reason or "",
                meta_json,
                created_at,
            ),
        )

    logger.debug(
        "created share grant id=%s proposer=%s recipient=%s namespace=%s",
        gid,
        proposer_agent_id,
        recipient_agent_id,
        namespace,
    )
    return ShareGrantRecord(
        id=gid,
        proposer_agent_id=proposer_agent_id,
        recipient_agent_id=recipient_agent_id,
        namespace=namespace,
        memory_ids=ids,
        scope=scope_s,
        status="pending",
        conflict_outcome=None,
        reason=reason or "",
        metadata=metadata or {},
        created_at=created_at,
        decided_at=None,
        decided_by=None,
    )


async def get_share_grant(grant_id: str, *, namespace: str) -> ShareGrantRecord | None:
    """Fetch one grant by id scoped to *namespace* (no cross-tenant leak)."""
    from archivist.storage.sqlite_pool import pool

    _ensure_share_grants_schema()
    if not grant_id or not namespace:
        raise ValueError("grant_id and namespace are required")

    async with pool.read() as conn:
        cur = await conn.execute(
            """
            SELECT id, proposer_agent_id, recipient_agent_id, namespace,
                   memory_ids, scope, status, conflict_outcome, reason,
                   metadata, created_at, decided_at, decided_by
            FROM memory_share_grants
            WHERE id = ? AND namespace = ?
            """,
            (grant_id, namespace),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    return _row_to_record(row)


async def decide_share_grant(
    grant_id: str,
    *,
    namespace: str,
    status: ShareStatus,
    decided_by: str,
) -> ShareGrantRecord | None:
    """Accept or reject a pending grant. Idempotent when already in *status*."""
    from archivist.storage.sqlite_pool import pool

    _ensure_share_grants_schema()
    if status not in ("accepted", "rejected"):
        raise ValueError("status must be 'accepted' or 'rejected'")
    if not grant_id or not namespace or not decided_by:
        raise ValueError("grant_id, namespace, and decided_by are required")

    existing = await get_share_grant(grant_id, namespace=namespace)
    if existing is None:
        return None
    if existing.status == status:
        return existing  # idempotent re-accept / re-reject
    if existing.status != "pending":
        raise ValueError(f"grant is already {existing.status}")

    decided_at = datetime.now(UTC).isoformat()
    async with pool.write() as conn:
        await conn.execute(
            """
            UPDATE memory_share_grants
            SET status = ?, decided_at = ?, decided_by = ?
            WHERE id = ? AND namespace = ? AND status = 'pending'
            """,
            (status, decided_at, decided_by, grant_id, namespace),
        )

    return await get_share_grant(grant_id, namespace=namespace)


async def attach_conflict_outcome(
    grant_id: str,
    *,
    namespace: str,
    outcome: dict[str, Any],
) -> ShareGrantRecord | None:
    """Attach a SPEC-006 resolution-shaped conflict outcome to a grant."""
    from archivist.storage.sqlite_pool import pool

    _ensure_share_grants_schema()
    if not grant_id or not namespace:
        raise ValueError("grant_id and namespace are required")
    if not isinstance(outcome, dict) or not outcome:
        raise ValueError("outcome must be a non-empty object")

    existing = await get_share_grant(grant_id, namespace=namespace)
    if existing is None:
        return None

    outcome_json = json.dumps(outcome, separators=(",", ":"), default=str)
    async with pool.write() as conn:
        await conn.execute(
            """
            UPDATE memory_share_grants
            SET conflict_outcome = ?
            WHERE id = ? AND namespace = ?
            """,
            (outcome_json, grant_id, namespace),
        )

    return await get_share_grant(grant_id, namespace=namespace)


def record_to_dict(record: ShareGrantRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "proposer_agent_id": record.proposer_agent_id,
        "recipient_agent_id": record.recipient_agent_id,
        "namespace": record.namespace,
        "memory_ids": list(record.memory_ids),
        "scope": record.scope,
        "status": record.status,
        "conflict_outcome": record.conflict_outcome,
        "reason": record.reason,
        "metadata": dict(record.metadata),
        "created_at": record.created_at,
        "decided_at": record.decided_at,
        "decided_by": record.decided_by,
    }
