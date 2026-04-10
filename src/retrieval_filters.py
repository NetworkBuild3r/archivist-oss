"""Pure retrieval filters — shared by rlm_retriever and tests."""

import statistics


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
