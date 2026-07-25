"""Memory version tracking — per-memory and scope lineage (INIT-001/SPEC-009)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("archivist.versioning")


async def record_version(
    memory_id: str,
    agent_id: str,
    text_hash: str,
    operation: str,
    parent_versions: list[int] | None = None,
) -> int:
    """Record a new version for a memory_id. Returns the new version number."""
    from archivist.storage.sqlite_pool import pool

    now = datetime.now(UTC).isoformat()
    async with pool.write() as conn:
        cur = await conn.execute(
            "SELECT MAX(version) as max_ver FROM memory_versions WHERE memory_id = ?",
            (memory_id,),
        )
        row = await cur.fetchone()
        current = row["max_ver"] if row and row["max_ver"] is not None else 0
        new_version = current + 1
        parents_json = json.dumps(parent_versions or [current] if current > 0 else [])
        await conn.execute(
            """INSERT INTO memory_versions (memory_id, version, agent_id, timestamp, text_hash, operation, parent_versions)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (memory_id, new_version, agent_id, now, text_hash, operation, parents_json),
        )
    return new_version


async def next_scope_version(namespace: str, agent_id: str = "") -> int:
    """Return the next monotonic version number for a memory scope.

    Scope key is ``(source_namespace, source_agent_id)``. Empty ``agent_id``
    means the whole-namespace scope.
    """
    from archivist.storage.sqlite_pool import pool

    if not namespace:
        raise ValueError("namespace is required")
    async with pool.read() as conn:
        cur = await conn.execute(
            """
            SELECT MAX(version) AS max_ver
            FROM memory_scope_versions
            WHERE source_namespace = ? AND source_agent_id = ?
            """,
            (namespace, agent_id or ""),
        )
        row = await cur.fetchone()
    current = row["max_ver"] if row and row["max_ver"] is not None else 0
    return int(current) + 1


async def insert_scope_version_row(
    *,
    version_id: str,
    namespace: str,
    agent_id: str,
    version: int,
    label: str,
    parent_version_id: str | None,
    chunk_count: int,
    point_count: int,
    archive_id: str,
    operation: str,
    created_by: str,
    created_at: str,
    lineage: dict[str, Any],
) -> None:
    """Persist a ``memory_scope_versions`` lineage row (Memory-as-Product)."""
    from archivist.storage.sqlite_pool import pool

    lineage_json = json.dumps(lineage or {}, separators=(",", ":"))
    async with pool.write() as conn:
        await conn.execute(
            """
            INSERT INTO memory_scope_versions (
                id, source_namespace, source_agent_id, version, label,
                parent_version_id, chunk_count, point_count, archive_id,
                operation, created_by, created_at, lineage_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                namespace,
                agent_id or "",
                version,
                label or "",
                parent_version_id,
                chunk_count,
                point_count,
                archive_id or "",
                operation,
                created_by or "",
                created_at,
                lineage_json,
            ),
        )
    logger.debug(
        "scope version recorded id=%s ns=%s version=%d op=%s parent=%s",
        version_id,
        namespace,
        version,
        operation,
        parent_version_id,
    )
