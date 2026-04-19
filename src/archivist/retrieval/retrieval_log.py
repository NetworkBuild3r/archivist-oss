"""Retrieval trajectory logging and export — records every search pipeline execution
for debugging, analytics, and UI visualization.

Each trajectory captures the full pipeline: query → coarse → dedupe → graph →
temporal → threshold → outcome-adjust → rerank → parent → refine → synthesize,
plus timing and the final retrieval_trace dict.
"""

import json
import logging
import uuid
from datetime import UTC, datetime

from archivist.core.config import TRAJECTORY_EXPORT_ENABLED
from archivist.storage.graph import schema_guard

logger = logging.getLogger("archivist.retrieval_log")

_ensure_schema = schema_guard("""
    CREATE TABLE IF NOT EXISTS retrieval_logs (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        query TEXT NOT NULL,
        namespace TEXT DEFAULT '',
        tier TEXT DEFAULT 'l2',
        memory_type TEXT DEFAULT '',
        retrieval_trace TEXT NOT NULL,
        result_count INTEGER DEFAULT 0,
        cache_hit INTEGER DEFAULT 0,
        duration_ms INTEGER,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_rl_agent ON retrieval_logs(agent_id);
    CREATE INDEX IF NOT EXISTS idx_rl_created ON retrieval_logs(created_at);
""")


async def log_retrieval(
    agent_id: str,
    query: str,
    namespace: str,
    tier: str,
    memory_type: str,
    retrieval_trace: dict,
    result_count: int,
    cache_hit: bool = False,
    duration_ms: int | None = None,
) -> str:
    """Persist a retrieval execution record using the async pool."""
    from archivist.storage.sqlite_pool import pool

    if not TRAJECTORY_EXPORT_ENABLED:
        return ""

    _ensure_schema()
    log_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    try:
        async with pool.write() as conn:
            await conn.execute(
                """INSERT INTO retrieval_logs
                   (id, agent_id, query, namespace, tier, memory_type,
                    retrieval_trace, result_count, cache_hit, duration_ms, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    log_id,
                    agent_id,
                    query,
                    namespace,
                    tier,
                    memory_type,
                    json.dumps(retrieval_trace),
                    result_count,
                    1 if cache_hit else 0,
                    duration_ms,
                    now,
                ),
            )
    except Exception as e:
        logger.warning(
            "retrieval_log.log_retrieval failed (agent=%s): %s",
            agent_id,
            e,
            extra={"agent_id": agent_id, "duration_ms": duration_ms},
        )
        return ""

    return log_id


def get_retrieval_logs(
    agent_id: str = "",
    limit: int = 50,
    offset: int = 0,
    since: str = "",
) -> list[dict]:
    """Retrieve recent retrieval logs for debugging/export."""
    from archivist.storage.graph import get_db

    _ensure_schema()
    conn = get_db()

    conditions = []
    params: list = []
    if agent_id:
        conditions.append("agent_id = ?")
        params.append(agent_id)
    if since:
        conditions.append("created_at >= ?")
        params.append(since)

    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    params.extend([limit, offset])

    cur = conn.execute(
        f"SELECT * FROM retrieval_logs{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params,
    )
    rows = []
    for r in cur.fetchall():
        d = dict(r)
        try:
            d["retrieval_trace"] = json.loads(d["retrieval_trace"])
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(
                "retrieval_log.get_retrieval_logs: trace JSON parse failed for id=%s: %s",
                d.get("id"),
                e,
            )
        d["cache_hit"] = bool(d.get("cache_hit"))
        rows.append(d)
    conn.close()
    return rows


def get_retrieval_stats(agent_id: str = "", window_days: int = 7) -> dict:
    """Aggregate retrieval statistics for monitoring."""
    from archivist.storage.graph import get_db

    _ensure_schema()
    conn = get_db()

    agent_filter = ""
    params: list = [f"-{window_days} days"]
    if agent_id:
        agent_filter = " AND agent_id = ?"
        params.append(agent_id)

    cur = conn.execute(
        f"""SELECT
                COUNT(*) as total,
                SUM(cache_hit) as cache_hits,
                AVG(duration_ms) as avg_duration_ms,
                AVG(result_count) as avg_results,
                MIN(duration_ms) as min_duration_ms,
                MAX(duration_ms) as max_duration_ms
            FROM retrieval_logs
            WHERE created_at >= datetime('now', ?){agent_filter}""",
        params,
    )
    row = cur.fetchone()
    result = dict(row) if row else {}

    for key in ("avg_duration_ms", "avg_results"):
        if result.get(key) is not None:
            result[key] = round(result[key], 1)

    cur2 = conn.execute(
        f"""SELECT agent_id, COUNT(*) as cnt
            FROM retrieval_logs
            WHERE created_at >= datetime('now', ?){agent_filter}
            GROUP BY agent_id ORDER BY cnt DESC LIMIT 10""",
        params,
    )
    result["top_agents"] = [dict(r) for r in cur2.fetchall()]
    conn.close()

    result["window_days"] = window_days
    result["cache_hit_rate"] = (
        round(result.get("cache_hits", 0) / result["total"], 3) if result.get("total") else None
    )
    return result
