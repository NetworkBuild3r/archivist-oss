"""Health dashboard and batch-size heuristic — aggregates operational metrics
across memory, skills, retrieval, and conflicts for a single-pane view.

The batch heuristic: when memory health degrades (high conflict rate, stale
memories, low retrieval quality), recommend smaller batches / more frequent
checkpoints.
"""

import logging
import time
from datetime import UTC, datetime

import archivist.core.health as health
from archivist.core.config import QDRANT_COLLECTION
from archivist.storage.qdrant import qdrant_client
from archivist.storage.sqlite_pool import pool

logger = logging.getLogger("archivist.dashboard")


async def build_dashboard(window_days: int = 7) -> dict:
    """Aggregate health metrics across all subsystems."""
    now_iso = datetime.now(UTC).isoformat()

    qdrant_stats = _qdrant_stats()

    async with pool.read() as conn:
        audit_stats = await _audit_stats(conn, window_days)
        retrieval = await _retrieval_stats(conn, window_days)
        skill_overview = await _skill_overview(conn, window_days)
        token_savings = await _token_savings_stats(conn, window_days)
        tier_dist = await _tier_distribution_stats(conn, window_days)

    hotness_heatmap = await _hotness_heatmap()
    stale = _stale_estimate()

    import archivist.retrieval.hot_cache as hot_cache

    cache = hot_cache.stats()

    return {
        "generated_at": now_iso,
        "window_days": window_days,
        "memories": qdrant_stats,
        "stale_estimate": stale,
        "conflicts": audit_stats,
        "retrieval": retrieval,
        "skills": skill_overview,
        "cache": {
            "enabled": cache.get("enabled"),
            "total_entries": cache.get("total_entries"),
            "agents": cache.get("agents"),
        },
        "token_savings": token_savings,
        "tier_distribution": tier_dist,
        "hotness_heatmap": hotness_heatmap,
        "subsystems": health.all_status(),
    }


async def batch_heuristic(window_days: int = 7) -> dict:
    """Recommend batch size based on memory health signals.

    Returns a recommendation dict with suggested_batch_size (1=tiny/careful,
    5=normal, 10=large/confident) and the signals used.
    """
    dashboard = await build_dashboard(window_days)

    score = 5.0
    signals = []

    conflict_rate = dashboard["conflicts"].get("conflict_rate", 0)
    if conflict_rate > 0.2:
        score -= 2
        signals.append(f"High conflict rate ({conflict_rate:.0%})")
    elif conflict_rate > 0.05:
        score -= 1
        signals.append(f"Moderate conflict rate ({conflict_rate:.0%})")

    stale_pct = dashboard["stale_estimate"].get("stale_pct", 0)
    if stale_pct > 30:
        score -= 2
        signals.append(f"High stale memory % ({stale_pct:.0f}%)")
    elif stale_pct > 10:
        score -= 1
        signals.append(f"Moderate stale memory % ({stale_pct:.0f}%)")

    cache_hit = dashboard.get("retrieval", {}).get("cache_hit_rate")
    if cache_hit is not None and cache_hit < 0.1:
        score -= 0.5
        signals.append(f"Low cache hit rate ({cache_hit:.0%})")

    skill_health = dashboard["skills"].get("degraded_count", 0)
    if skill_health > 2:
        score -= 1
        signals.append(f"{skill_health} degraded/broken skills")

    score = max(1, min(10, round(score)))

    if score <= 2:
        recommendation = "Reduce batch size. High conflict/stale rate — use single-item operations with verification."
    elif score <= 4:
        recommendation = "Use small batches (2–3 items). Check conflicts before bulk operations."
    elif score <= 7:
        recommendation = "Normal batch size. Memory health is acceptable."
    else:
        recommendation = "Large batches safe. Memory health is excellent."

    return {
        "suggested_batch_size": score,
        "recommendation": recommendation,
        "signals": signals,
        "dashboard_summary": {
            "conflict_rate": conflict_rate,
            "stale_pct": stale_pct,
            "cache_hit_rate": cache_hit,
            "degraded_skills": skill_health,
        },
    }


def _qdrant_stats() -> dict:
    try:
        client = qdrant_client()
        info = client.get_collection(QDRANT_COLLECTION)
        return {
            "total_points": info.points_count,
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
            "status": str(info.status),
        }
    except Exception as e:
        logger.warning("dashboard._qdrant_stats failed: %s", e, exc_info=True)
        return {"error": str(e)}


def _stale_estimate() -> dict:
    try:
        from qdrant_client.models import FieldCondition, Filter, Range

        client = qdrant_client()
        now_ts = int(time.time())

        info = client.get_collection(QDRANT_COLLECTION)
        total = info.points_count or 1

        stale_filter = Filter(
            must=[FieldCondition(key="ttl_expires_at", range=Range(lte=now_ts, gt=0))]
        )
        stale_count = client.count(
            collection_name=QDRANT_COLLECTION,
            count_filter=stale_filter,
            exact=False,
        ).count
        return {
            "stale_count": stale_count,
            "total": total,
            "stale_pct": round(stale_count / total * 100, 1),
        }
    except Exception as e:
        logger.warning("dashboard._stale_estimate failed: %s", e, exc_info=True)
        return {"error": str(e), "stale_pct": 0}


async def _audit_stats(conn, window_days: int) -> dict:
    try:
        from archivist.core.config import GRAPH_BACKEND

        if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
            cutoff = f"timestamp >= NOW() - INTERVAL '{window_days} days'"
            params: list = []
        else:
            cutoff = "timestamp >= datetime('now', ?)"
            params = [f"-{window_days} days"]

        rows = await conn.fetchall(
            f"SELECT action, COUNT(*) as cnt FROM audit_log WHERE {cutoff} GROUP BY action",
            params,
        )
        counts = {row["action"]: row["cnt"] for row in rows}
        total_writes = counts.get("create", 0) + counts.get("merge", 0) + counts.get("update", 0)
        conflicts = counts.get("conflict_detected", 0)
        conflict_rate = conflicts / total_writes if total_writes > 0 else 0
        return {
            "total_writes": total_writes,
            "conflicts": conflicts,
            "conflict_rate": round(conflict_rate, 3),
            "deletes": counts.get("delete", 0),
            "annotations": counts.get("annotate", 0),
            "ratings": counts.get("rate", 0),
        }
    except Exception as e:
        logger.warning("dashboard._audit_stats failed: %s", e, exc_info=True)
        return {"total_writes": 0, "conflicts": 0, "conflict_rate": 0}


async def _retrieval_stats(conn, window_days: int) -> dict:
    try:
        from archivist.core.config import GRAPH_BACKEND

        if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
            cutoff = f"created_at >= NOW() - INTERVAL '{window_days} days'"
            params: list = []
        else:
            cutoff = "created_at >= datetime('now', ?)"
            params = [f"-{window_days} days"]

        row = await conn.fetchone(
            f"""SELECT
                   COUNT(*) as total,
                   SUM(cache_hit) as cache_hits,
                   AVG(duration_ms) as avg_duration_ms
               FROM retrieval_logs
               WHERE {cutoff}""",
            params,
        )
        result = dict(row) if row else {}
        total = result.get("total", 0) or 0
        hits = result.get("cache_hits", 0) or 0
        result["cache_hit_rate"] = round(hits / total, 3) if total > 0 else None
        if result.get("avg_duration_ms"):
            result["avg_duration_ms"] = round(result["avg_duration_ms"], 1)
        return result
    except Exception as e:
        logger.warning("dashboard._retrieval_stats failed: %s", e, exc_info=True)
        return {"total": 0, "cache_hits": 0, "cache_hit_rate": None, "avg_duration_ms": None}


async def _skill_overview(conn, window_days: int) -> dict:
    try:
        from archivist.core.config import GRAPH_BACKEND

        if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
            cutoff = f"created_at >= NOW() - INTERVAL '{window_days} days'"
            params: list = []
        else:
            cutoff = "created_at >= datetime('now', ?)"
            params = [f"-{window_days} days"]

        status_rows = await conn.fetchall(
            "SELECT status, COUNT(*) as cnt FROM skills GROUP BY status"
        )
        status_counts = {row["status"]: row["cnt"] for row in status_rows}

        total_skills = sum(status_counts.values())
        degraded = status_counts.get("broken", 0) + status_counts.get("deprecated", 0)

        event_rows = await conn.fetchall(
            f"SELECT outcome, COUNT(*) as cnt FROM skill_events WHERE {cutoff} GROUP BY outcome",
            params,
        )
        event_counts = {row["outcome"]: row["cnt"] for row in event_rows}
        total_events = sum(event_counts.values())
        success_rate = (
            round(event_counts.get("success", 0) / total_events, 3) if total_events > 0 else None
        )

        return {
            "total_skills": total_skills,
            "status_breakdown": status_counts,
            "degraded_count": degraded,
            "events_in_window": total_events,
            "skill_success_rate": success_rate,
        }
    except Exception as e:
        logger.warning("dashboard._skill_overview failed: %s", e, exc_info=True)
        return {
            "total_skills": 0,
            "degraded_count": 0,
            "events_in_window": 0,
            "skill_success_rate": None,
        }


async def _token_savings_stats(conn, window_days: int) -> dict:
    """Aggregate token savings statistics from retrieval logs."""
    try:
        from archivist.core.config import GRAPH_BACKEND

        if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
            cutoff = f"created_at >= NOW() - INTERVAL '{window_days} days'"
            params: list = []
        else:
            cutoff = "created_at >= datetime('now', ?)"
            params = [f"-{window_days} days"]

        row = await conn.fetchone(
            f"""SELECT
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN tokens_naive IS NOT NULL THEN 1 ELSE 0 END) as queries_with_savings_data,
                    AVG(savings_pct) as avg_savings_pct,
                    MIN(savings_pct) as min_savings_pct,
                    MAX(savings_pct) as max_savings_pct,
                    SUM(CASE WHEN tokens_naive > 0
                             THEN tokens_naive - tokens_returned ELSE 0 END) as total_tokens_saved,
                    SUM(tokens_returned) as total_tokens_returned,
                    SUM(tokens_naive) as total_tokens_naive
                FROM retrieval_logs
                WHERE {cutoff}
                  AND tokens_naive IS NOT NULL""",
            params,
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
                WHERE {cutoff}
                  AND pack_policy IS NOT NULL
                  AND pack_policy != ''
                GROUP BY pack_policy
                ORDER BY cnt DESC""",
            params,
        )
        stats["per_policy"] = [dict(r) for r in policy_rows]

        for key in ("avg_savings_pct", "min_savings_pct", "max_savings_pct"):
            if stats.get(key) is not None:
                stats[key] = round(float(stats[key]), 1)
        for key in ("total_tokens_saved", "total_tokens_returned", "total_tokens_naive"):
            if stats.get(key) is not None:
                stats[key] = int(stats[key])

        return stats
    except Exception as e:
        logger.warning("dashboard._token_savings_stats failed: %s", e, exc_info=True)
        return {
            "total_queries": 0,
            "queries_with_savings_data": 0,
            "avg_savings_pct": None,
            "total_tokens_saved": 0,
            "per_policy": [],
        }


async def _tier_distribution_stats(conn, window_days: int) -> dict:
    """Aggregate tier-distribution breakdown from retrieval_trace context_status."""
    try:
        from archivist.core.config import GRAPH_BACKEND

        if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
            cutoff = f"created_at >= NOW() - INTERVAL '{window_days} days'"
            json_l0 = "retrieval_trace::json->'context_status'->>'tier_distribution'"
            params: list = []
        else:
            cutoff = "created_at >= datetime('now', ?)"
            json_l0 = "json_extract(retrieval_trace, '$.context_status.tier_distribution')"
            params = [f"-{window_days} days"]

        pack_rows = await conn.fetchall(
            f"""SELECT
                    pack_policy,
                    COUNT(*) as cnt,
                    AVG(result_count) as avg_results
                FROM retrieval_logs
                WHERE {cutoff}
                  AND pack_policy IS NOT NULL
                  AND pack_policy != ''
                GROUP BY pack_policy""",
            params,
        )
        return {
            "by_pack_policy": [dict(r) for r in pack_rows],
        }
    except Exception as e:
        logger.warning("dashboard._tier_distribution_stats failed: %s", e, exc_info=True)
        return {"by_pack_policy": []}


async def _hotness_heatmap(top_n: int = 50) -> list[dict]:
    """Return top-N memories by hotness score for the heatmap widget."""
    try:
        async with pool.read() as conn:
            rows = await conn.fetchall(
                """SELECT
                       mh.memory_id,
                       mh.score,
                       mh.retrieval_count,
                       mh.last_accessed,
                       mc.tier_label,
                       mc.importance,
                       mc.file_path
                   FROM memory_hotness mh
                   LEFT JOIN memory_chunks mc ON mc.qdrant_id = mh.memory_id
                   ORDER BY mh.score DESC
                   LIMIT ?""",
                (top_n,),
            )
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("dashboard._hotness_heatmap failed: %s", e, exc_info=True)
        return []
