"""Write-ahead curator queue — stages curation operations for batched application.

Operations are enqueued by the on-write dedup pipeline and batch curator cycle,
then drained by a periodic applicator that runs during idle periods. This avoids
GRAPH_WRITE_LOCK contention on the hot path.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone

from archivist.storage.graph import get_db, GRAPH_WRITE_LOCK, schema_guard
import archivist.core.metrics as m

logger = logging.getLogger("archivist.curator_queue")

VALID_OP_TYPES = {
    "merge_memory", "delete_memory", "consolidate_tips",
    "update_hotness", "skip_store", "archive_memory",
}

_ensure_schema = schema_guard("""
    CREATE TABLE IF NOT EXISTS curator_queue (
        id TEXT PRIMARY KEY,
        op_type TEXT NOT NULL,
        payload TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        applied_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_cq_status ON curator_queue(status);
    CREATE INDEX IF NOT EXISTS idx_cq_created ON curator_queue(created_at);
""")


def enqueue(op_type: str, payload: dict) -> str:
    """Add a curation operation to the queue. Returns the operation ID."""
    _ensure_schema()
    if op_type not in VALID_OP_TYPES:
        raise ValueError(f"Invalid op_type: {op_type}. Must be one of {VALID_OP_TYPES}")

    op_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with GRAPH_WRITE_LOCK:
        conn = get_db()
        conn.execute(
            "INSERT INTO curator_queue (id, op_type, payload, status, created_at) VALUES (?, ?, ?, 'pending', ?)",
            (op_id, op_type, json.dumps(payload), now),
        )
        conn.commit()
        conn.close()

    _update_depth_gauge()
    logger.debug("Enqueued %s operation %s", op_type, op_id)
    return op_id


async def drain(limit: int = 50) -> list[dict]:
    """Apply pending operations up to limit. Returns list of applied ops."""
    _ensure_schema()
    start = time.time()
    applied = []

    with GRAPH_WRITE_LOCK:
        conn = get_db()
        rows = conn.execute(
            "SELECT id, op_type, payload FROM curator_queue WHERE status = 'pending' ORDER BY created_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()

    now = datetime.now(timezone.utc).isoformat()

    for row in rows:
        op_id, op_type, payload_json = row
        try:
            payload = json.loads(payload_json) if payload_json else {}
            await _apply_op(op_type, payload)
            _mark_op(op_id, "applied", now)
            applied.append({"id": op_id, "op_type": op_type, "status": "applied"})
        except Exception as e:
            logger.error("Failed to apply curator op %s (%s): %s", op_id, op_type, e)
            _mark_op(op_id, "failed", now)
            applied.append({"id": op_id, "op_type": op_type, "status": "failed", "error": str(e)})

    elapsed = (time.time() - start) * 1000
    m.observe(m.CURATOR_DRAIN_DURATION, elapsed)
    _update_depth_gauge()

    if applied:
        logger.info("Curator queue drain: %d ops applied in %.0fms", len(applied), elapsed)
    return applied


def _mark_op(op_id: str, status: str, timestamp: str):
    """Update a queue entry's status under the write lock."""
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        conn.execute(
            "UPDATE curator_queue SET status = ?, applied_at = ? WHERE id = ?",
            (status, timestamp, op_id),
        )
        conn.commit()
        conn.close()


async def _apply_op(op_type: str, payload: dict):
    """Execute a single queued operation."""
    if op_type == "archive_memory":
        await _apply_archive(payload)
    elif op_type == "merge_memory":
        _apply_merge(payload)
    elif op_type == "delete_memory":
        await _apply_delete(payload)
    elif op_type == "consolidate_tips":
        _apply_consolidate_tips(payload)
    elif op_type == "update_hotness":
        _apply_hotness(payload)
    elif op_type == "skip_store":
        pass


async def _apply_archive(payload: dict):
    """Set archived=true on all Qdrant points for each memory."""
    from archivist.lifecycle.memory_lifecycle import archive_memory_complete
    from archivist.lifecycle.cascade import PartialDeletionError

    memory_ids = payload.get("memory_ids", [])
    namespace = payload.get("namespace", "")
    if not memory_ids:
        return

    for mid in memory_ids:
        try:
            await archive_memory_complete(mid, namespace)
        except PartialDeletionError:
            raise
        except Exception as e:
            logger.warning("Failed to archive memory %s: %s", mid, e)


def _apply_merge(payload: dict):
    """Merge memory content — delegates to the existing merge module."""
    pass  # placeholder: full merge logic lives in merge.py


_last_pre_prune_snapshot: float = 0.0
_PRE_PRUNE_DEBOUNCE_SECONDS = 300


def _maybe_pre_prune_snapshot() -> None:
    """Create a backup snapshot before destructive operations (debounced)."""
    global _last_pre_prune_snapshot
    from archivist.core.config import BACKUP_PRE_PRUNE

    if not BACKUP_PRE_PRUNE:
        return

    now = time.time()
    if now - _last_pre_prune_snapshot < _PRE_PRUNE_DEBOUNCE_SECONDS:
        return

    _last_pre_prune_snapshot = now
    try:
        from archivist.storage.backup_manager import create_snapshot, prune_snapshots
        create_snapshot(label="pre-prune")
        prune_snapshots()
        logger.info("Pre-prune backup snapshot created")
    except Exception as e:
        logger.warning("Pre-prune snapshot failed (non-blocking): %s", e)


async def _apply_delete(payload: dict):
    """Delete memories and ALL derived artifacts via the lifecycle module."""
    from archivist.lifecycle.memory_lifecycle import delete_memory_complete
    from archivist.lifecycle.cascade import PartialDeletionError

    memory_ids = payload.get("memory_ids", [])
    namespace = payload.get("namespace", "")
    if not memory_ids:
        return

    _maybe_pre_prune_snapshot()

    for mid in memory_ids:
        try:
            await delete_memory_complete(mid, namespace)
        except PartialDeletionError:
            raise
        except Exception as e:
            logger.warning("Failed to delete memory %s: %s", mid, e)


def _apply_consolidate_tips(payload: dict):
    """Insert consolidated tip and archive originals."""
    consolidated = payload.get("consolidated_tip")
    original_ids = payload.get("original_tip_ids", [])

    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            if consolidated:
                tip_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO tips (id, trajectory_id, agent_id, category, tip_text, context, negative_example, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        tip_id,
                        consolidated.get("trajectory_id", "consolidated"),
                        consolidated.get("agent_id", "curator"),
                        consolidated.get("category", "strategy"),
                        consolidated.get("tip", ""),
                        consolidated.get("context", ""),
                        consolidated.get("negative_example", ""),
                        now,
                    ),
                )

            if original_ids:
                placeholders = ",".join("?" for _ in original_ids)
                conn.execute(
                    f"UPDATE tips SET archived = 1 WHERE id IN ({placeholders})",
                    original_ids,
                )
            conn.commit()
        finally:
            conn.close()


def _apply_hotness(payload: dict):
    """Update hotness scores in the memory_hotness table."""
    scores = payload.get("scores", {})
    now = datetime.now(timezone.utc).isoformat()

    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            for memory_id, score in scores.items():
                conn.execute(
                    "INSERT OR REPLACE INTO memory_hotness (memory_id, score, retrieval_count, last_accessed, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (memory_id, score, payload.get("counts", {}).get(memory_id, 0),
                     payload.get("last_access", {}).get(memory_id, now), now),
                )
            conn.commit()
        finally:
            conn.close()


def stats() -> dict:
    """Return queue statistics."""
    _ensure_schema()
    conn = get_db()
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM curator_queue GROUP BY status"
    ).fetchall()
    conn.close()

    counts = {row[0]: row[1] for row in rows}
    return {
        "pending": counts.get("pending", 0),
        "applied": counts.get("applied", 0),
        "failed": counts.get("failed", 0),
        "total": sum(counts.values()),
    }


def _update_depth_gauge():
    """Update the Prometheus gauge for queue depth."""
    try:
        s = stats()
        m.gauge_set(m.CURATOR_QUEUE_DEPTH, s["pending"])
    except Exception:
        pass
