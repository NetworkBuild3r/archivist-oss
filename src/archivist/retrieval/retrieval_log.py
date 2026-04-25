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
        created_at TEXT NOT NULL,
        tokens_returned INTEGER,
        tokens_naive INTEGER,
        savings_pct REAL,
        pack_policy TEXT DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_rl_agent ON retrieval_logs(agent_id);
    CREATE INDEX IF NOT EXISTS idx_rl_created ON retrieval_logs(created_at);
    CREATE INDEX IF NOT EXISTS idx_rl_pack_policy ON retrieval_logs(pack_policy);
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
    tokens_returned: int | None = None,
    tokens_naive: int | None = None,
    savings_pct: float | None = None,
    pack_policy: str = "",
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
                    retrieval_trace, result_count, cache_hit, duration_ms, created_at,
                    tokens_returned, tokens_naive, savings_pct, pack_policy)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    tokens_returned,
                    tokens_naive,
                    savings_pct,
                    pack_policy,
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


async def get_retrieval_logs(
    agent_id: str = "",
    limit: int = 50,
    offset: int = 0,
    since: str = "",
) -> list[dict]:
    """Retrieve recent retrieval logs for debugging/export."""
    from archivist.storage.sqlite_pool import pool

    _ensure_schema()

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

    async with pool.read() as conn:
        raw_rows = await conn.fetchall(
            f"SELECT * FROM retrieval_logs{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
    rows = []
    for r in raw_rows:
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
    return rows


async def get_retrieval_stats(agent_id: str = "", window_days: int = 7) -> dict:
    """Aggregate retrieval statistics for monitoring."""
    from archivist.storage.graph import _is_postgres
    from archivist.storage.sqlite_pool import pool

    _ensure_schema()

    agent_filter = ""
    if agent_id:
        agent_filter = " AND agent_id = ?"

    if _is_postgres():
        date_expr = f"NOW() - INTERVAL '{window_days} days'"
        params_main: list = []
        params_agents: list = []
        if agent_id:
            params_main.append(agent_id)
            params_agents.append(agent_id)
    else:
        date_expr = "datetime('now', ?)"
        params_main = [f"-{window_days} days"]
        params_agents = [f"-{window_days} days"]
        if agent_id:
            params_main.append(agent_id)
            params_agents.append(agent_id)

    async with pool.read() as conn:
        row = await conn.fetchone(
            f"""SELECT
                    COUNT(*) as total,
                    SUM(cache_hit) as cache_hits,
                    AVG(duration_ms) as avg_duration_ms,
                    AVG(result_count) as avg_results,
                    MIN(duration_ms) as min_duration_ms,
                    MAX(duration_ms) as max_duration_ms
                FROM retrieval_logs
                WHERE created_at >= {date_expr}{agent_filter}""",
            params_main,
        )
        result = dict(row) if row else {}

        for key in ("avg_duration_ms", "avg_results"):
            if result.get(key) is not None:
                result[key] = round(result[key], 1)

        top_rows = await conn.fetchall(
            f"""SELECT agent_id, COUNT(*) as cnt
                FROM retrieval_logs
                WHERE created_at >= {date_expr}{agent_filter}
                GROUP BY agent_id ORDER BY cnt DESC LIMIT 10""",
            params_agents,
        )
        result["top_agents"] = [dict(r) for r in top_rows]

    result["window_days"] = window_days
    result["cache_hit_rate"] = (
        round(result.get("cache_hits", 0) / result["total"], 3) if result.get("total") else None
    )
    return result


async def get_token_savings_stats(agent_id: str = "", window_days: int = 7) -> dict:
    """Aggregate token savings and tier-distribution statistics for the dashboard.

    Returns averages, percentiles, per-policy breakdown, and daily token
    savings totals for the given window.
    """
    from archivist.storage.graph import _is_postgres
    from archivist.storage.sqlite_pool import pool

    _ensure_schema()

    agent_filter = ""
    agent_params: list = []
    if agent_id:
        agent_filter = " AND agent_id = ?"
        agent_params = [agent_id]

    if _is_postgres():
        date_expr = f"NOW() - INTERVAL '{window_days} days'"
        params_base: list = [] + agent_params
        params_policy: list = [] + agent_params
    else:
        date_expr = "datetime('now', ?)"
        params_base = [f"-{window_days} days"] + agent_params
        params_policy = [f"-{window_days} days"] + agent_params

    async with pool.read() as conn:
        row = await conn.fetchone(
            f"""SELECT
                    COUNT(*) as total,
                    AVG(savings_pct) as avg_savings_pct,
                    MIN(savings_pct) as min_savings_pct,
                    MAX(savings_pct) as max_savings_pct,
                    SUM(CASE WHEN tokens_naive > 0
                             THEN tokens_naive - tokens_returned ELSE 0 END) as total_tokens_saved,
                    SUM(tokens_returned) as total_tokens_returned,
                    SUM(tokens_naive) as total_tokens_naive,
                    AVG(tokens_returned) as avg_tokens_returned,
                    AVG(tokens_naive) as avg_tokens_naive,
                    SUM(CASE WHEN savings_pct IS NOT NULL THEN 1 ELSE 0 END) as rows_with_savings
                FROM retrieval_logs
                WHERE created_at >= {date_expr}
                  AND tokens_naive IS NOT NULL{agent_filter}""",
            params_base,
        )
        stats = dict(row) if row else {}

        policy_rows = await conn.fetchall(
            f"""SELECT
                    pack_policy,
                    COUNT(*) as cnt,
                    AVG(savings_pct) as avg_savings_pct,
                    SUM(CASE WHEN tokens_naive > 0
                             THEN tokens_naive - tokens_returned ELSE 0 END) as tokens_saved
                FROM retrieval_logs
                WHERE created_at >= {date_expr}
                  AND pack_policy IS NOT NULL
                  AND pack_policy != ''{agent_filter}
                GROUP BY pack_policy
                ORDER BY cnt DESC""",
            params_policy,
        )
        stats["per_policy"] = [dict(r) for r in policy_rows]

    for key in (
        "avg_savings_pct",
        "min_savings_pct",
        "max_savings_pct",
        "avg_tokens_returned",
        "avg_tokens_naive",
    ):
        if stats.get(key) is not None:
            stats[key] = round(float(stats[key]), 1)
    for key in ("total_tokens_saved", "total_tokens_returned", "total_tokens_naive"):
        if stats.get(key) is not None:
            stats[key] = int(stats[key])

    stats["window_days"] = window_days
    return stats
