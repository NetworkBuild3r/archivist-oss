"""Prometheus-compatible metrics for Archivist.

Exposes counters, histograms, and gauges via GET /metrics in the standard
text exposition format. No external dependency required — pure-Python
implementation that any Prometheus scraper can consume.

Histograms use pre-aggregated bucket counters (not raw sample lists) so
memory is O(buckets × label-sets) regardless of request volume.
"""

import threading
import time
from collections import defaultdict

_lock = threading.Lock()

# ── Counters ─────────────────────────────────────────────────────────────────
_counters: dict[str, float] = defaultdict(float)

# ── Histograms (pre-aggregated bucket counts) ────────────────────────────────
_HISTOGRAM_BUCKETS = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, float("inf")]
_histogram_buckets: dict[str, list[int]] = {}

# ── Gauges ───────────────────────────────────────────────────────────────────
_gauges: dict[str, float] = {}


def inc(name: str, labels: dict | None = None, value: float = 1.0):
    """Increment a counter."""
    key = _key(name, labels)
    with _lock:
        _counters[key] += value


def observe(name: str, value: float, labels: dict | None = None):
    """Record a histogram observation (O(buckets), constant memory)."""
    key = _key(name, labels)
    with _lock:
        buckets = _histogram_buckets.get(key)
        if buckets is None:
            buckets = [0] * len(_HISTOGRAM_BUCKETS)
            _histogram_buckets[key] = buckets
        for i, boundary in enumerate(_HISTOGRAM_BUCKETS):
            if value <= boundary:
                buckets[i] += 1
    inc(f"{name}_count", labels)
    inc(f"{name}_sum", labels, value)


def gauge_set(name: str, value: float, labels: dict | None = None):
    """Set a gauge to an absolute value."""
    key = _key(name, labels)
    with _lock:
        _gauges[key] = value


def gauge_inc(name: str, labels: dict | None = None, value: float = 1.0):
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
            label_part = key[len(base):]
            cumulative = 0
            for i, boundary in enumerate(_HISTOGRAM_BUCKETS):
                cumulative += buckets[i]
                le_label = '+Inf' if boundary == float("inf") else str(boundary)
                if label_part:
                    inner = label_part[1:-1]
                    lines.append(f'{base}_bucket{{{inner},le="{le_label}"}} {cumulative}')
                else:
                    lines.append(f'{base}_bucket{{le="{le_label}"}} {cumulative}')

    lines.append("")
    return "\n".join(lines)


# ── Convenience metric names ─────────────────────────────────────────────────

SEARCH_TOTAL = "archivist_search_total"
SEARCH_DURATION = "archivist_search_duration_ms"
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

# ── Curator intelligence (v1.0) ─────────────────────────────────────────────
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

# ── Needle retrieval telemetry (v1.11) ───────────────────────────────────────
NEEDLE_REGISTRY_HITS = "archivist_needle_registry_hits_total"
NEEDLE_REGISTRY_STALE = "archivist_needle_registry_stale_total"
MICRO_CHUNK_HITS = "archivist_micro_chunk_hits_total"

# ── Cascade / orphan sweeper (v1.11) ─────────────────────────────────────────
ORPHAN_SWEEP = "archivist_orphan_sweep_cleaned_total"
