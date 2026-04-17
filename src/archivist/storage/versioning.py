"""Memory version tracking — monotonically increasing versions per memory_id."""

import json
import logging
from datetime import datetime, timezone

from archivist.storage.graph import get_db, GRAPH_WRITE_LOCK

logger = logging.getLogger("archivist.versioning")


def record_version(
    memory_id: str,
    agent_id: str,
    text_hash: str,
    operation: str,
    parent_versions: list[int] | None = None,
) -> int:
    """Record a new version for a memory_id. Returns the new version number."""
    now = datetime.now(timezone.utc).isoformat()
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            cur = conn.execute(
                "SELECT MAX(version) as max_ver FROM memory_versions WHERE memory_id = ?",
                (memory_id,),
            )
            row = cur.fetchone()
            current = row["max_ver"] if row and row["max_ver"] is not None else 0
            new_version = current + 1
            parents_json = json.dumps(parent_versions or [current] if current > 0 else [])
            conn.execute(
                """INSERT INTO memory_versions (memory_id, version, agent_id, timestamp, text_hash, operation, parent_versions)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (memory_id, new_version, agent_id, now, text_hash, operation, parents_json),
            )
            conn.commit()
        finally:
            conn.close()
    return new_version


