"""Health dashboard and batch-size heuristic — aggregates operational metrics
across memory, skills, retrieval, and conflicts for a single-pane view.

The batch heuristic is inspired by the Batch Size Gravity article: when
memory health degrades (high conflict rate, stale memories, low retrieval
quality), recommend smaller batches / more frequent checkpoints.
"""

import logging
import time
from datetime import datetime, timezone

from graph import get_db
from config import QDRANT_COLLECTION
from qdrant import qdrant_client

logger = logging.getLogger("archivist.dashboard")


def build_dashboard(window_days: int = 7) -> dict:
    """Aggregate health metrics across all subsystems."""
    conn = get_db()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Memory counts from Qdrant
    qdrant_stats = _qdrant_stats()

    # Conflict rate (from audit)
    audit_stats = _audit_stats(conn, window_days)

    # Retrieval stats
    retrieval = _retrieval_stats(conn, window_days)

    # Skill health overview
    skill_overview = _skill_overview(conn, window_days)

    # Stale memory estimate (TTL-based)
    stale = _stale_estimate()

    # Cache stats
    import hot_cache
    cache = hot_cache.stats()

    conn.close()

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
    }


def batch_heuristic(window_days: int = 7) -> dict:
    """Recommend batch size based on memory health signals.

    Returns a recommendation dict with suggested_batch_size (1=tiny/careful,
    5=normal, 10=large/confident) and the signals used.
    """
    dashboard = build_dashboard(window_days)

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
        client = qdrant_client(timeout=10)
        info = client.get_collection(QDRANT_COLLECTION)
        return {
            "total_points": info.points_count,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"error": str(e)}


def _stale_estimate() -> dict:
    try:
        from qdrant_client.models import Filter, FieldCondition, Range
        client = qdrant_client(timeout=10)
        now_ts = int(time.time())

        info = client.get_collection(QDRANT_COLLECTION)
        total = info.points_count or 1

        expired, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="ttl_expires_at", range=Range(lte=now_ts, gt=0))]
            ),
            limit=0,
            with_payload=False,
        )
        stale_count = len(expired) if expired else 0
        return {"stale_count": stale_count, "total": total, "stale_pct": round(stale_count / total * 100, 1)}
    except Exception as e:
        return {"error": str(e), "stale_pct": 0}


def _audit_stats(conn, window_days: int) -> dict:
    try:
        cur = conn.execute(
            """SELECT action, COUNT(*) as cnt FROM audit_log
               WHERE timestamp >= datetime('now', ?)
               GROUP BY action""",
            (f"-{window_days} days",),
        )
        counts = {row["action"]: row["cnt"] for row in cur.fetchall()}
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
    except Exception:
        return {"total_writes": 0, "conflicts": 0, "conflict_rate": 0}


def _retrieval_stats(conn, window_days: int) -> dict:
    try:
        cur = conn.execute(
            """SELECT
                   COUNT(*) as total,
                   SUM(cache_hit) as cache_hits,
                   AVG(duration_ms) as avg_duration_ms
               FROM retrieval_logs
               WHERE created_at >= datetime('now', ?)""",
            (f"-{window_days} days",),
        )
        row = dict(cur.fetchone())
        total = row.get("total", 0) or 0
        hits = row.get("cache_hits", 0) or 0
        row["cache_hit_rate"] = round(hits / total, 3) if total > 0 else None
        if row.get("avg_duration_ms"):
            row["avg_duration_ms"] = round(row["avg_duration_ms"], 1)
        return row
    except Exception:
        return {"total": 0, "cache_hits": 0, "cache_hit_rate": None, "avg_duration_ms": None}


def _skill_overview(conn, window_days: int) -> dict:
    try:
        cur = conn.execute("SELECT status, COUNT(*) as cnt FROM skills GROUP BY status")
        status_counts = {row["status"]: row["cnt"] for row in cur.fetchall()}

        total_skills = sum(status_counts.values())
        degraded = status_counts.get("broken", 0) + status_counts.get("deprecated", 0)

        cur2 = conn.execute(
            """SELECT outcome, COUNT(*) as cnt FROM skill_events
               WHERE created_at >= datetime('now', ?) GROUP BY outcome""",
            (f"-{window_days} days",),
        )
        event_counts = {row["outcome"]: row["cnt"] for row in cur2.fetchall()}
        total_events = sum(event_counts.values())
        success_rate = (
            round(event_counts.get("success", 0) / total_events, 3)
            if total_events > 0 else None
        )

        return {
            "total_skills": total_skills,
            "status_breakdown": status_counts,
            "degraded_count": degraded,
            "events_in_window": total_events,
            "skill_success_rate": success_rate,
        }
    except Exception:
        return {"total_skills": 0, "degraded_count": 0, "events_in_window": 0, "skill_success_rate": None}
