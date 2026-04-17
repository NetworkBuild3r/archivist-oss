"""Immutable audit logging for all memory operations."""

import hashlib
import logging
import json
import uuid
from datetime import datetime, timezone

from graph import get_db, GRAPH_WRITE_LOCK, schema_guard

logger = logging.getLogger("archivist.audit")

_ensure_audit_schema = schema_guard("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        action TEXT NOT NULL,
        memory_id TEXT,
        namespace TEXT,
        text_hash TEXT,
        version INTEGER,
        metadata TEXT DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_id);
    CREATE INDEX IF NOT EXISTS idx_audit_memory ON audit_log(memory_id);
    CREATE INDEX IF NOT EXISTS idx_audit_namespace ON audit_log(namespace);
""")


async def log_memory_event(
    agent_id: str,
    action: str,
    memory_id: str,
    namespace: str,
    text_hash: str,
    version: int = 0,
    metadata: dict | None = None,
):
    """Append an immutable entry to the audit log."""
    _ensure_audit_schema()
    entry_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(metadata or {})

    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            conn.execute(
                """INSERT INTO audit_log (id, timestamp, agent_id, action, memory_id, namespace, text_hash, version, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry_id, now, agent_id, action, memory_id, namespace, text_hash, version, meta_json),
            )
            conn.commit()
        except Exception as e:
            logger.error("Failed to write audit log: %s", e)
        finally:
            conn.close()


def get_audit_trail(memory_id: str, limit: int = 50) -> list[dict]:
    """Query audit log for a specific memory ID."""
    _ensure_audit_schema()
    conn = get_db()
    cur = conn.execute(
        "SELECT * FROM audit_log WHERE memory_id = ? ORDER BY timestamp DESC LIMIT ?",
        (memory_id, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_agent_activity(agent_id: str, limit: int = 50) -> list[dict]:
    """Query audit log for agent activity."""
    _ensure_audit_schema()
    conn = get_db()
    if agent_id:
        cur = conn.execute(
            "SELECT * FROM audit_log WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?",
            (agent_id, limit),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
