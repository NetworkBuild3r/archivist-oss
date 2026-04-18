"""Prometheus-compatible metrics for Archivist.

Exposes counters, histograms, and gauges via GET /metrics in the standard
text exposition format. No external dependency required — pure-Python
implementation that any Prometheus scraper can consume.

Histograms use pre-aggregated bucket counters (not raw sample lists) so
memory is O(buckets × label-sets) regardless of request volume.

When ``METRICS_ENABLED`` is false (see ``config``), recording is a no-op and
``render()`` returns an empty string.
"""

import asyncio
import logging
import sqlite3
import threading
from collections import defaultdict
from datetime import UTC, datetime

_lock = threading.Lock()

logger = logging.getLogger("archivist.metrics")

# ── Counters ─────────────────────────────────────────────────────────────────
_counters: dict[str, float] = defaultdict(float)

# ── Histograms (pre-aggregated bucket counts) ───────────────────────────────────
# Default buckets: latency in milliseconds (existing metrics).
_HISTOGRAM_BUCKETS_MS = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, float("inf")]
# Search result counts (number of items in ``sources``).
_SEARCH_RESULT_BUCKETS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, float("inf")]

_histogram_buckets: dict[str, list[int]] = {}
# Parallel to _histogram_buckets keys: which bucket boundaries apply (ms vs result counts).
_histogram_layout: dict[str, list[float]] = {}


def _boundaries_for(name: str) -> list[float]:
    """Histogram bucket upper bounds for ``name`` (latency ms vs search result cardinality)."""
    if name == "archivist_search_results":
        return _SEARCH_RESULT_BUCKETS
    return _HISTOGRAM_BUCKETS_MS


# ── Gauges ───────────────────────────────────────────────────────────────────
_gauges: dict[str, float] = {}


def _metrics_enabled() -> bool:
    # Import inside the function avoids import cycles (config imports no metrics).
    from archivist.core.config import METRICS_ENABLED

    return METRICS_ENABLED


def inc(name: str, labels: dict | None = None, value: float = 1.0):
    """Increment a counter."""
    if not _metrics_enabled():
        return
    key = _key(name, labels)
    with _lock:
        _counters[key] += value


def observe(name: str, value: float, labels: dict | None = None):
    """Record a histogram observation (O(buckets), constant memory)."""
    if not _metrics_enabled():
        return
    boundaries = _boundaries_for(name)
    key = _key(name, labels)
    with _lock:
        buckets = _histogram_buckets.get(key)
        if buckets is None:
            buckets = [0] * len(boundaries)
            _histogram_buckets[key] = buckets
            _histogram_layout[key] = boundaries
        else:
            boundaries = _histogram_layout[key]
        # Prometheus-style: one increment per bucket where observation <= le.
        for i, boundary in enumerate(boundaries):
            if value <= boundary:
                buckets[i] += 1
    # Histogram _count / _sum companions (render emits full histogram + these).
    inc(f"{name}_count", labels)
    inc(f"{name}_sum", labels, value)


def gauge_set(name: str, value: float, labels: dict | None = None):
    """Set a gauge to an absolute value."""
    if not _metrics_enabled():
        return
    key = _key(name, labels)
    with _lock:
        _gauges[key] = value


def gauge_inc(name: str, labels: dict | None = None, value: float = 1.0):
    if not _metrics_enabled():
        return
    key = _key(name, labels)
    with _lock:
        _gauges[key] = _gauges.get(key, 0) + value


def _key(name: str, labels: dict | None) -> str:
    if not labels:
        return name
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"


def render() -> str:
    """Render all metrics in Prometheus text exposition format."""
    if not _metrics_enabled():
        return ""
    lines = []
    seen_names: set[str] = set()

    with _lock:
        for key, val in sorted(_counters.items()):
            base = key.split("{")[0]
            if base not in seen_names:
                lines.append(f"# TYPE {base} counter")
                seen_names.add(base)
            lines.append(f"{key} {val}")

        for key, val in sorted(_gauges.items()):
            base = key.split("{")[0]
            if base not in seen_names:
                lines.append(f"# TYPE {base} gauge")
                seen_names.add(base)
            lines.append(f"{key} {val}")

        for key, buckets in sorted(_histogram_buckets.items()):
            base = key.split("{")[0]
            if base not in seen_names:
                lines.append(f"# TYPE {base} histogram")
                seen_names.add(base)
            label_part = key[len(base) :]
            boundaries = _histogram_layout[key]
            cumulative = 0
            for i, boundary in enumerate(boundaries):
                cumulative += buckets[i]
                le_label = "+Inf" if boundary == float("inf") else str(boundary)
                if label_part:
                    inner = label_part[1:-1]
                    lines.append(f'{base}_bucket{{{inner},le="{le_label}"}} {cumulative}')
                else:
                    lines.append(f'{base}_bucket{{le="{le_label}"}} {cumulative}')

    lines.append("")
    return "\n".join(lines)


def collect_storage_gauges_tick() -> None:
    """Update storage and availability gauges (blocking; run via asyncio.to_thread).

    Called from ``run_storage_gauges_loop`` so sync Qdrant/SQLite work never blocks the event loop.
    """
    if not _metrics_enabled():
        return

    import os

    from archivist.core.config import SQLITE_PATH

    # SQLite: file size, SELECT 1 liveness, and per-namespace “live” memory count from audit_log.
    try:
        if os.path.isfile(SQLITE_PATH):
            gauge_set(SQLITE_SIZE_BYTES, float(os.path.getsize(SQLITE_PATH)))
    except OSError:
        pass

    try:
        from archivist.storage.graph import get_db

        conn = get_db()
        try:
            conn.execute("SELECT 1").fetchone()
            gauge_set(SQLITE_AVAILABLE, 1.0)
            try:
                # Latest audit row per memory_id; count rows whose latest action is not delete.
                cur = conn.execute(
                    """
                    WITH ranked AS (
                      SELECT memory_id, namespace, action,
                        ROW_NUMBER() OVER (PARTITION BY memory_id ORDER BY timestamp DESC) AS rn
                      FROM audit_log
                      WHERE memory_id IS NOT NULL AND TRIM(memory_id) != ''
                    )
                    SELECT COALESCE(NULLIF(TRIM(namespace), ''), '_default') AS ns, COUNT(*) AS cnt
                    FROM ranked
                    WHERE rn = 1 AND action != 'delete'
                    GROUP BY ns
                    """
                )
                for row in cur.fetchall():
                    ns = str(row[0] or "_default")
                    gauge_set(TOTAL_MEMORIES, float(row[1]), {"namespace": ns})
            except sqlite3.OperationalError as e:
                if "no such table" not in str(e).lower():
                    logger.debug("metrics audit_log query: %s", e)
        finally:
            conn.close()
    except Exception as e:
        logger.debug("metrics sqlite gauges: %s", e)
        gauge_set(SQLITE_AVAILABLE, 0.0)

    # Qdrant: points_count per collection; availability from health.register("qdrant") if present,
    # else infer 1 after a successful list collections (startup may not have registered yet).
    try:
        import archivist.core.health as health
        from archivist.storage.qdrant import qdrant_client

        st = health.all_status()
        q_entry = st.get("qdrant")
        if q_entry is not None:
            gauge_set(QDRANT_AVAILABLE, 1.0 if q_entry.get("healthy") else 0.0)
        client = qdrant_client()
        cols = client.get_collections().collections
        if q_entry is None:
            gauge_set(QDRANT_AVAILABLE, 1.0)
        for c in cols:
            try:
                info = client.get_collection(c.name)
                pts = float(info.points_count)
            except Exception:
                pts = 0.0
            gauge_set(QDRANT_VECTORS_TOTAL, pts, {"collection": c.name})
    except Exception:
        gauge_set(QDRANT_AVAILABLE, 0.0)

    # Outbox: pending count, dead count, and lag of oldest pending row.
    try:
        from archivist.storage.graph import get_db

        conn = get_db()
        try:
            cur = conn.execute(
                "SELECT status, COUNT(*) FROM outbox GROUP BY status"
            )
            status_counts: dict[str, int] = {r[0]: r[1] for r in cur.fetchall()}
            gauge_set(OUTBOX_PENDING, float(status_counts.get("pending", 0)))
            gauge_set(OUTBOX_DEAD, float(status_counts.get("dead", 0)))

            cur2 = conn.execute(
                "SELECT MIN(created_at) FROM outbox WHERE status = 'pending'"
            )
            oldest_row = cur2.fetchone()
            if oldest_row and oldest_row[0]:
                try:
                    oldest_dt = datetime.fromisoformat(oldest_row[0])
                    if oldest_dt.tzinfo is None:
                        oldest_dt = oldest_dt.replace(tzinfo=UTC)
                    lag = (datetime.now(UTC) - oldest_dt).total_seconds()
                    gauge_set(OUTBOX_LAG_SECONDS, max(0.0, lag))
                except ValueError:
                    pass
            else:
                gauge_set(OUTBOX_LAG_SECONDS, 0.0)
        finally:
            conn.close()
    except Exception as e:
        logger.debug("metrics outbox gauges: %s", e)


async def run_storage_gauges_loop(interval_seconds: float) -> None:
    """Background task: refresh storage gauges periodically (first tick runs immediately)."""
    from archivist.core.config import METRICS_ENABLED

    # Floor avoids tight loops if env is mis-set too low.
    interval = max(5.0, float(interval_seconds))
    while True:
        if METRICS_ENABLED:
            try:
                await asyncio.to_thread(collect_storage_gauges_tick)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Storage gauge collection failed: %s", e)
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise


# ── Convenience metric names ─────────────────────────────────────────────────

SEARCH_TOTAL = "archivist_search_total"
SEARCH_DURATION = "archivist_search_duration_ms"
SEARCH_RESULTS = "archivist_search_results"
STORE_TOTAL = "archivist_store_total"
STORE_CONFLICT = "archivist_store_conflict_total"
CACHE_HIT = "archivist_cache_hit_total"
CACHE_MISS = "archivist_cache_miss_total"
WEBHOOK_FIRE = "archivist_webhook_fire_total"
WEBHOOK_FAIL = "archivist_webhook_fail_total"
SKILL_EVENT = "archivist_skill_event_total"
INDEX_CHUNKS = "archivist_index_chunks_total"
LLM_CALL = "archivist_llm_call_total"
LLM_DURATION = "archivist_llm_duration_ms"
LLM_ERROR = "archivist_llm_error_total"

# ── Curator intelligence (v1.0) ────────────────────────────────────────────────
CURATOR_QUEUE_DEPTH = "archivist_curator_queue_depth"
CURATOR_DEDUP_DECISION = "archivist_curator_dedup_decisions_total"
CURATOR_TIP_CONSOLIDATIONS = "archivist_curator_tip_consolidations_total"
CURATOR_LLM_CALLS = "archivist_curator_llm_calls_total"
CURATOR_DRAIN_DURATION = "archivist_curator_drain_duration_ms"

# ── MCP tool observability (v1.10) ───────────────────────────────────────────
TOOL_DURATION = "archivist_mcp_tool_duration_ms"
TOOL_ERRORS = "archivist_mcp_tool_errors_total"
EMBED_DURATION = "archivist_embed_duration_ms"
QDRANT_QUERY_DURATION = "archivist_qdrant_query_duration_ms"
INVALIDATE_DURATION = "archivist_invalidate_duration_ms"
INVALIDATE_COUNT = "archivist_invalidate_count_total"
CURATOR_CYCLE_DURATION = "archivist_curator_cycle_duration_ms"
DELETE_COMPLETE = "archivist_delete_complete_total"
ARCHIVE_COMPLETE = "archivist_archive_complete_total"
SOFT_DELETE_INITIATED = "archivist_soft_delete_initiated_total"

# ── Needle retrieval telemetry (v1.11) ───────────────────────────────────────
NEEDLE_REGISTRY_HITS = "archivist_needle_registry_hits_total"
NEEDLE_REGISTRY_STALE = "archivist_needle_registry_stale_total"
MICRO_CHUNK_HITS = "archivist_micro_chunk_hits_total"

# ── Cascade / orphan sweeper (v1.11) ───────────────────────────────────────
ORPHAN_SWEEP = "archivist_orphan_sweep_cleaned_total"

# ── Storage / health gauges + normalized names (v1.11) ─────────────────────
TOTAL_MEMORIES = "archivist_total_memories"
SQLITE_SIZE_BYTES = "archivist_sqlite_size_bytes"
QDRANT_VECTORS_TOTAL = "archivist_qdrant_vectors_total"
QDRANT_AVAILABLE = "archivist_qdrant_available"
SQLITE_AVAILABLE = "archivist_sqlite_available"

EMBED_CACHE_HIT = "archivist_embed_cache_hit_total"
EMBED_CACHE_MISS = "archivist_embed_cache_miss_total"
HYDE_DURATION = "archivist_hyde_duration_ms"
REVERSE_HYDE_DURATION = "archivist_reverse_hyde_duration_ms"
QUERY_EXPANSION_DURATION = "archivist_query_expansion_duration_ms"

# ── SQLite async pool observability (v1.12) ──────────────────────────────────
SQLITE_POOL_ACQUIRE_MS = "archivist_sqlite_pool_acquire_ms"
SQLITE_POOL_WRITE_ERRORS = "archivist_sqlite_pool_write_errors_total"

# ── Transactional outbox observability (Phase 3 enterprise hardening) ─────────
OUTBOX_PENDING = "archivist_outbox_pending"
"""Gauge: current count of outbox rows with status='pending'."""
OUTBOX_DEAD = "archivist_outbox_dead"
"""Gauge: current count of outbox rows with status='dead'."""
OUTBOX_LAG_SECONDS = "archivist_outbox_lag_seconds"
"""Gauge: age in seconds of the oldest pending outbox row (0 when queue empty)."""
OUTBOX_DRAIN_DURATION = "archivist_outbox_drain_duration_ms"
"""Histogram: wall-clock time per drain() call in milliseconds."""
OUTBOX_APPLIED_TOTAL = "archivist_outbox_applied_total"
"""Counter: cumulative events successfully applied to Qdrant."""
OUTBOX_DEAD_LETTER_TOTAL = "archivist_outbox_dead_letter_total"
"""Counter: cumulative events moved to dead-letter status."""
OUTBOX_RECOVERY_COUNT = "archivist_outbox_recovery_total"
"""Counter: cumulative 'processing' events recovered by the orphan sweep."""
OUTBOX_PRUNED_TOTAL = "archivist_outbox_pruned_total"
"""Counter: cumulative 'applied' rows deleted by retention pruning."""
