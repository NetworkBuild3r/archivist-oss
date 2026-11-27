"""Hotness scoring — frequency × recency signal for memory retrieval ranking.

Formula (adapted from OpenViking):
    hotness = sigmoid(log1p(retrieval_count)) * exp(-ln2 * days_since_last_access / halflife)

Batch scan aggregates from retrieval_logs into a memory_hotness table.
RLM pipeline blends hotness into scores after temporal decay.
"""

import logging
import math
from datetime import UTC, datetime, timedelta

from archivist.core.config import HOTNESS_HALFLIFE_DAYS, HOTNESS_WEIGHT
from archivist.storage.graph import schema_guard

logger = logging.getLogger("archivist.hotness")

_HOTNESS_WATERMARK_KEY = "hotness_last_log_ts"

# Module-level guard so SQLite DDL runs at most once per DB path instead of
# opening a synchronous sqlite3 connection on every search (MAJ-12).
_ensure_hotness_schema = schema_guard("""
    CREATE TABLE IF NOT EXISTS memory_hotness (
        memory_id TEXT PRIMARY KEY,
        score REAL NOT NULL DEFAULT 0.0,
        retrieval_count INTEGER NOT NULL DEFAULT 0,
        last_accessed TEXT,
        updated_at TEXT NOT NULL,
        importance_signal REAL NOT NULL DEFAULT 0.5
    );
""")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_hotness(
    retrieval_count: int, days_since_last_access: float, halflife: float | None = None
) -> float:
    """Compute hotness for a single memory."""
    hl = halflife or HOTNESS_HALFLIFE_DAYS
    frequency = _sigmoid(math.log1p(retrieval_count))
    recency = math.exp(-math.log(2) * days_since_last_access / max(hl, 0.1))
    return frequency * recency


async def get_hotness_scores(memory_ids: list[str]) -> dict[str, float]:
    """Look up precomputed hotness for a batch of memory IDs."""
    from archivist.storage.graph import _is_postgres
    from archivist.storage.sqlite_pool import pool

    if not memory_ids:
        return {}

    if not _is_postgres():
        _ensure_hotness_schema()

    placeholders = ",".join("?" for _ in memory_ids)
    async with pool.read() as conn:
        rows = await conn.fetchall(
            f"SELECT memory_id, score FROM memory_hotness WHERE memory_id IN ({placeholders})",
            memory_ids,
        )
    return {r["memory_id"]: r["score"] for r in rows}


async def apply_hotness_to_results(results: list[dict], weight: float | None = None) -> list[dict]:
    """Blend hotness scores into retrieval results after temporal decay."""
    w = weight if weight is not None else HOTNESS_WEIGHT
    if w <= 0 or not results:
        return results

    ids = [str(r.get("id", "")) for r in results if r.get("id")]
    if not ids:
        return results

    scores = await get_hotness_scores(ids)
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


def _retrieval_logs_query(is_postgres: bool) -> str:
    """Return the parameterized retrieval_logs aggregation query for the active backend.

    Expects a single ISO-8601 UTC cutoff timestamp parameter ``?``.
    Uses ``cache_hit = 0`` uniformly (INTEGER column on both SQLite and Postgres)
    and compares ``created_at`` (TEXT ISO-8601 UTC) against a bound string.
    """
    if is_postgres:
        return """
            SELECT
                retrieval_trace::json->>'result_ids' AS result_ids,
                created_at
            FROM retrieval_logs
            WHERE cache_hit = 0
              AND created_at > ?
            ORDER BY created_at DESC
            LIMIT 5000
        """
    return """
        SELECT
            json_extract(retrieval_trace, '$.result_ids') as result_ids,
            created_at
        FROM retrieval_logs
        WHERE cache_hit = 0
          AND created_at > ?
        ORDER BY created_at DESC
        LIMIT 5000
    """


async def batch_update_hotness() -> int:
    """Aggregate retrieval_logs into memory_hotness. Called from curator cycle.

    Uses a persisted watermark in ``curator_state`` (``hotness_last_log_ts``) so
    repeated curator cycles only aggregate newly arrived retrieval_logs rows
    rather than double-counting the 7-day window.
    """
    import json as _json

    from archivist.storage.graph import _is_postgres
    from archivist.storage.sqlite_pool import pool

    if not _is_postgres():
        _ensure_hotness_schema()

    now = datetime.now(UTC)
    now_iso = now.isoformat()
    default_cutoff = (now - timedelta(days=7)).isoformat()

    memory_counts: dict[str, int] = {}
    memory_last_access: dict[str, str] = {}
    watermark = default_cutoff

    async with pool.read() as conn:
        hotness_rows = await conn.fetchall(
            "SELECT memory_id, retrieval_count, last_accessed FROM memory_hotness"
        )
        for r in hotness_rows:
            memory_counts[r["memory_id"]] = r["retrieval_count"]
            memory_last_access[r["memory_id"]] = r["last_accessed"] or now_iso

        try:
            wm_row = await conn.fetchone(
                "SELECT value FROM curator_state WHERE key = ?",
                (_HOTNESS_WATERMARK_KEY,),
            )
            if wm_row and wm_row["value"]:
                watermark = max(str(wm_row["value"]), default_cutoff)
        except Exception as e:
            logger.debug("hotness.batch_update: watermark read skipped: %s", e)

        try:
            log_rows = await conn.fetchall(_retrieval_logs_query(_is_postgres()), (watermark,))
        except Exception as e:
            logger.warning("hotness.batch_update: retrieval_logs query failed: %s", e)
            log_rows = []

    max_seen_ts = ""
    for row in log_rows:
        row_ts = str(row["created_at"] or "")
        if row_ts > max_seen_ts:
            max_seen_ts = row_ts
        try:
            ids = _json.loads(row["result_ids"] or "[]")
        except Exception as e:
            logger.debug("hotness.batch_update: JSON parse failed for row: %s", e)
            continue
        if not isinstance(ids, list):
            continue
        for mid in ids:
            if not mid:
                continue
            mid = str(mid)
            memory_counts[mid] = memory_counts.get(mid, 0) + 1
            if mid not in memory_last_access or row_ts > memory_last_access.get(mid, ""):
                memory_last_access[mid] = row_ts

    updated = 0
    async with pool.write() as conn:
        for mid, count in memory_counts.items():
            last_str = memory_last_access.get(mid, now_iso)
            try:
                last_dt = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
            except Exception as e:
                logger.debug("hotness.batch_update: ISO parse failed for '%s': %s", last_str, e)
                last_dt = now
            days = max((now - last_dt).total_seconds() / 86400, 0.0)
            score = compute_hotness(count, days)

            await conn.execute(
                "INSERT INTO memory_hotness "
                "(memory_id, score, retrieval_count, last_accessed, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(memory_id) DO UPDATE SET score=excluded.score, "
                "retrieval_count=excluded.retrieval_count, "
                "last_accessed=excluded.last_accessed, "
                "updated_at=excluded.updated_at",
                (mid, score, count, last_str, now_iso),
            )
            updated += 1

        if max_seen_ts:
            try:
                await conn.execute(
                    "INSERT INTO curator_state (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (_HOTNESS_WATERMARK_KEY, max_seen_ts),
                )
            except Exception as e:
                logger.debug("hotness.batch_update: watermark write skipped: %s", e)

    logger.info("Hotness batch update: %d memories scored", updated)
    return updated
