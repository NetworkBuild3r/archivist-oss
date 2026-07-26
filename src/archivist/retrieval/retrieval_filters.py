"""Pure retrieval filters — shared by rlm_retriever and tests.

INIT-003/SPEC-005 — pre-rank tenant/lifecycle filters + stable recall memories shape.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from archivist.lifecycle.visibility import filter_recall_visible, is_recall_visible

# Provenance keys safe to return on the coach recall path (no secrets).
_SAFE_PROVENANCE_KEYS = frozenset(
    {
        "source",
        "subject",
        "confidence",
        "sensitivity",
        "purpose",
        "statement_kind",
        "namespace",
        "agent_id",
        "date",
        "memory_type",
        "actor_type",
        "actor_id",
        "created_at",
        "updated_at",
        "supersedes_id",
        "file_path",
        "tier",
    }
)

_SECRET_KEY_RE = re.compile(
    r"(secret|token|password|passwd|api[_-]?key|authorization|credential|cookie|private[_-]?key)",
    re.IGNORECASE,
)


def apply_retrieval_threshold(results: list[dict], threshold: float) -> list[dict]:
    """Filter out results with vector score below threshold."""
    return [r for r in results if r["score"] >= threshold]


# ── Dynamic threshold (v1.10) ────────────────────────────────────────────────
_DYNAMIC_FLOOR = 0.25
_DYNAMIC_RELATIVE_RATIO = 0.55
_DYNAMIC_MIN_KEEP = 3


def apply_dynamic_threshold(
    results: list[dict],
    fallback_threshold: float,
    min_keep: int = _DYNAMIC_MIN_KEEP,
) -> list[dict]:
    """Score-distribution-aware threshold that adapts per query.

    Instead of a single fixed cutoff, computes:
      effective = max(floor, top_score * ratio, statistical_cutoff)
    but never returns fewer than ``min_keep`` results (so a low-scoring
    needle is never silently dropped when it is the best match).

    Falls back to the static threshold when there are too few results
    to compute meaningful statistics.
    """
    if not results:
        return []

    scores = [r.get("score", 0) or 0 for r in results]
    top_score = max(scores)

    if len(scores) < 4:
        effective = min(fallback_threshold, top_score * _DYNAMIC_RELATIVE_RATIO)
    else:
        median = statistics.median(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        stat_cutoff = median - stdev * 1.5 if stdev > 0 else median * 0.5

        effective = max(
            _DYNAMIC_FLOOR,
            top_score * _DYNAMIC_RELATIVE_RATIO,
            stat_cutoff,
        )
        effective = min(effective, fallback_threshold)

    filtered = [r for r in results if r.get("score", 0) >= effective]

    if len(filtered) < min_keep and results:
        by_score = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        filtered = by_score[:min_keep]

    return filtered


def _row_get(row: Mapping[str, Any] | Any, *keys: str) -> Any:
    if isinstance(row, Mapping):
        for key in keys:
            if key in row:
                return row[key]
        return None
    for key in keys:
        if hasattr(row, key):
            return getattr(row, key)
    return None


def _norm_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def apply_prerank_filters(
    results: Iterable[Mapping[str, Any] | Any],
    *,
    namespace: str = "",
    subject: str = "",
    purpose: str = "",
    sensitivity: str = "",
    known_superseded_ids: set[str] | frozenset[str] | None = None,
) -> list[Any]:
    """Apply lifecycle + tenant filters **before** ranking / threshold.

    Always omits suppressed and superseded losers (via SPEC-007 visibility).
    When ``namespace`` / ``subject`` are set, mismatched rows are dropped —
    callers cannot disable these gates to widen tenants.

    Optional ``purpose`` / ``sensitivity`` further narrow within the namespace;
    empty means no restriction on that axis (aud-1).
    """
    ns = _norm_str(namespace)
    subj = _norm_str(subject)
    purp = _norm_str(purpose)
    sens = _norm_str(sensitivity)

    visible = filter_recall_visible(
        results,
        known_superseded_ids=known_superseded_ids,
    )

    out: list[Any] = []
    for row in visible:
        if ns:
            row_ns = _norm_str(_row_get(row, "namespace"))
            if row_ns and row_ns != ns:
                continue
        if subj:
            row_subj = _norm_str(_row_get(row, "subject"))
            if row_subj != subj:
                continue
        if purp:
            row_purp = _norm_str(_row_get(row, "purpose"))
            if row_purp != purp:
                continue
        if sens:
            row_sens = _norm_str(_row_get(row, "sensitivity")) or "standard"
            if row_sens != sens:
                continue
        out.append(row)
    return out


def sanitize_provenance(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a provenance dict with only safe, non-secret keys."""
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None or value == "":
            continue
        key_s = str(key)
        if _SECRET_KEY_RE.search(key_s):
            continue
        if key_s not in _SAFE_PROVENANCE_KEYS:
            continue
        out[key_s] = value
    return out


def _usable_text(row: Mapping[str, Any]) -> str:
    for key in ("tier_text", "text", "l1", "l0", "extraction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_stable_memories(sources: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize hit rows into the canonical ``memories[]`` recall shape.

    Each item: ``{id, text, score, provenance}``. ``text`` is non-empty whenever
    the source row carries any usable content field.
    """
    memories: list[dict[str, Any]] = []
    for row in sources:
        mid = _norm_str(_row_get(row, "id", "qdrant_id", "memory_id"))
        text = _usable_text(row)
        if not mid and not text:
            continue
        nested = row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
        prov_src: dict[str, Any] = dict(nested) if nested else {}
        for key in _SAFE_PROVENANCE_KEYS:
            if key in row and row[key] not in (None, ""):
                prov_src.setdefault(key, row[key])
        memories.append(
            {
                "id": mid,
                "text": text,
                "score": float(row.get("score", 0.0) or 0.0),
                "provenance": sanitize_provenance(prov_src),
            }
        )
    return memories


def attach_stable_memories(result: dict[str, Any]) -> dict[str, Any]:
    """Mutate *result* to include canonical ``memories`` from ``sources``."""
    sources = result.get("sources") or []
    result["memories"] = build_stable_memories(sources)
    return result


# Re-export visibility predicate for callers/tests that import from this module.
__all__ = [
    "apply_dynamic_threshold",
    "apply_prerank_filters",
    "apply_retrieval_threshold",
    "attach_stable_memories",
    "build_stable_memories",
    "filter_recall_visible",
    "is_recall_visible",
    "sanitize_provenance",
]
