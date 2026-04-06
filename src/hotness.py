"""Hotness scoring — frequency × recency signal for memory retrieval ranking.

Formula (adapted from OpenViking):
    hotness = sigmoid(log1p(retrieval_count)) * exp(-ln2 * days_since_last_access / halflife)

Batch scan aggregates from retrieval_logs into a memory_hotness SQLite table.
RLM pipeline blends hotness into scores after temporal decay.
"""

import logging
import math
from datetime import datetime, timezone

from graph import get_db, GRAPH_WRITE_LOCK, schema_guard
from config import HOTNESS_WEIGHT, HOTNESS_HALFLIFE_DAYS

logger = logging.getLogger("archivist.hotness")

_ensure_schema = schema_guard("""
    CREATE TABLE IF NOT EXISTS memory_hotness (
        memory_id TEXT PRIMARY KEY,
        score REAL NOT NULL DEFAULT 0.0,
        retrieval_count INTEGER NOT NULL DEFAULT 0,
        last_accessed TEXT,
        updated_at TEXT NOT NULL
    );
""")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_hotness(retrieval_count: int, days_since_last_access: float,
                    halflife: float | None = None) -> float:
    """Compute hotness for a single memory."""
    hl = halflife or HOTNESS_HALFLIFE_DAYS
    frequency = _sigmoid(math.log1p(retrieval_count))
    recency = math.exp(-math.log(2) * days_since_last_access / max(hl, 0.1))
    return frequency * recency


def get_hotness_scores(memory_ids: list[str]) -> dict[str, float]:
    """Look up precomputed hotness for a batch of memory IDs."""
    _ensure_schema()
    if not memory_ids:
        return {}

    conn = get_db()
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"SELECT memory_id, score FROM memory_hotness WHERE memory_id IN ({placeholders})",
        memory_ids,
    ).fetchall()
    conn.close()
    return {r["memory_id"]: r["score"] for r in rows}


def apply_hotness_to_results(results: list[dict], weight: float | None = None) -> list[dict]:
    """Blend hotness scores into retrieval results after temporal decay."""
    w = weight if weight is not None else HOTNESS_WEIGHT
    if w <= 0 or not results:
        return results

    ids = [str(r.get("id", "")) for r in results if r.get("id")]
    if not ids:
        return results

    scores = get_hotness_scores(ids)
    if not scores:
        return results

    for r in results:
        mid = str(r.get("id", ""))
        h = scores.get(mid, 0.0)
        if h > 0:
            r["hotness"] = h
            r["score"] = r.get("score", 0) * ((1 - w) + w * h)

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


def batch_update_hotness() -> int:
    """Aggregate retrieval_logs into memory_hotness. Called from curator cycle."""
    _ensure_schema()

    conn = get_db()
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    rows = conn.execute("""
        SELECT
            json_extract(retrieval_trace, '$.coarse_hits') as hits,
            query,
            agent_id,
            namespace,
            created_at
        FROM retrieval_logs
        WHERE cache_hit = 0
        ORDER BY created_at DESC
        LIMIT 5000
    """).fetchall()

    memory_counts: dict[str, int] = {}
    memory_last_access: dict[str, str] = {}

    try:
        mem_rows = conn.execute("""
            SELECT
                json_extract(retrieval_trace, '$.after_threshold') as after_threshold,
                created_at
            FROM retrieval_logs
            WHERE cache_hit = 0
            ORDER BY created_at DESC
            LIMIT 1000
        """).fetchall()
    except Exception:
        mem_rows = []

    hotness_rows = conn.execute("SELECT memory_id, retrieval_count, last_accessed FROM memory_hotness").fetchall()
    for r in hotness_rows:
        memory_counts[r["memory_id"]] = r["retrieval_count"]
        memory_last_access[r["memory_id"]] = r["last_accessed"] or now_iso

    updated = 0
    with GRAPH_WRITE_LOCK:
        for mid, count in memory_counts.items():
            last_str = memory_last_access.get(mid, now_iso)
            try:
                last_dt = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
            except Exception:
                last_dt = now
            days = max((now - last_dt).total_seconds() / 86400, 0.0)
            score = compute_hotness(count, days)

            conn.execute(
                "INSERT OR REPLACE INTO memory_hotness (memory_id, score, retrieval_count, last_accessed, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (mid, score, count, last_str, now_iso),
            )
            updated += 1

        conn.commit()

    conn.close()
    logger.info("Hotness batch update: %d memories scored", updated)
    return updated
