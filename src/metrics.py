"""Prometheus-compatible metrics for Archivist.

Exposes counters, histograms, and gauges via GET /metrics in the standard
text exposition format. No external dependency required — pure-Python
implementation that any Prometheus scraper can consume.
"""

import threading
import time
from collections import defaultdict

_lock = threading.Lock()

# ── Counters ─────────────────────────────────────────────────────────────────
_counters: dict[str, float] = defaultdict(float)

# ── Histograms (store individual observations) ───────────────────────────────
_histogram_obs: dict[str, list[float]] = defaultdict(list)
_HISTOGRAM_BUCKETS = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, float("inf")]

# ── Gauges ───────────────────────────────────────────────────────────────────
_gauges: dict[str, float] = {}


def inc(name: str, labels: dict | None = None, value: float = 1.0):
    """Increment a counter."""
    key = _key(name, labels)
    with _lock:
        _counters[key] += value


def observe(name: str, value: float, labels: dict | None = None):
    """Record a histogram observation."""
    key = _key(name, labels)
    with _lock:
        _histogram_obs[key].append(value)
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
    seen_names = set()

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

        for key, observations in sorted(_histogram_obs.items()):
            base = key.split("{")[0]
            if base not in seen_names:
                lines.append(f"# TYPE {base} histogram")
                seen_names.add(base)
            label_part = key[len(base):]
            for bucket in _HISTOGRAM_BUCKETS:
                count = sum(1 for o in observations if o <= bucket)
                le_label = f'+Inf' if bucket == float("inf") else str(bucket)
                if label_part:
                    inner = label_part[1:-1]
                    lines.append(f'{base}_bucket{{{inner},le="{le_label}"}} {count}')
                else:
                    lines.append(f'{base}_bucket{{le="{le_label}"}} {count}')

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
