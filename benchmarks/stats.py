"""Shared statistical utilities for Archivist benchmark harnesses.

All benchmark harnesses (LongMemEval, BEIR, pipeline) import from here so
that statistical methods are consistent and independently testable.
"""

from __future__ import annotations

import hashlib
import random
import shutil
import statistics
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    *,
    seed: int | None = 42,
) -> tuple[float, float, float]:
    """Estimate mean and bootstrap 95% confidence interval.

    Uses the percentile bootstrap method: resample ``values`` with replacement
    ``n_boot`` times, compute the mean of each resample, then take the
    ``alpha/2`` and ``1 - alpha/2`` percentiles of the bootstrap distribution
    as the interval endpoints.

    Args:
        values:  Sequence of per-sample scores (e.g. 0/1 correctness flags or
                 float metrics). Must contain at least 2 elements.
        n_boot:  Number of bootstrap resamples.  1000 is standard for 95% CI.
        alpha:   Significance level.  0.05 gives a 95% CI.
        seed:    RNG seed for reproducibility.  Pass ``None`` for non-determinism.

    Returns:
        ``(mean, ci_low, ci_high)`` where ``ci_low`` and ``ci_high`` are the
        lower and upper bounds of the ``(1 - alpha) * 100``% confidence interval.

    Raises:
        ValueError: If ``values`` is empty or has fewer than 2 elements.

    Notes:
        - The CI is *only reliable when n >= 30*.  Callers should check
          ``len(values) >= 30`` before surfacing CI numbers to users.
        - The percentile bootstrap can be slightly conservative for proportions
          near 0 or 1, but is simple and widely understood.
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"bootstrap_ci requires at least 2 values, got {n}")

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        resample = [rng.choice(values) for _ in range(n)]  # type: ignore[arg-type]
        means.append(statistics.mean(resample))

    means.sort()
    lo_idx = int(alpha / 2 * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return (
        statistics.mean(values),
        means[max(lo_idx, 0)],
        means[min(hi_idx, n_boot - 1)],
    )


MIN_N_FOR_CI: int = 30
"""Minimum sample size before CI is surfaced in output.

Below this threshold the bootstrap percentile CI is unreliable and should
not be reported.  The field is still computed internally but excluded from
the output dict when n < MIN_N_FOR_CI.
"""


def maybe_ci(
    values: Sequence[float],
    key: str,
    out: dict,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 42,
) -> None:
    """Compute bootstrap CI for ``values`` and inject it into ``out`` if n >= 30.

    Mutates ``out`` in-place:
    - Always sets ``out[key]`` to the mean (rounded to 4 decimal places).
    - Sets ``out[key + "_ci95"]`` to ``[ci_low, ci_high]`` only when
      ``len(values) >= MIN_N_FOR_CI``.

    Args:
        values:  Per-sample metric values.
        key:     Output dict key for the mean (e.g. ``"overall_accuracy"``).
        out:     Dict to mutate.
        n_boot:  Passed to ``bootstrap_ci``.
        alpha:   Passed to ``bootstrap_ci``.
        seed:    Passed to ``bootstrap_ci``.
    """
    if not values:
        out[key] = 0.0
        return
    if len(values) == 1:
        out[key] = round(float(values[0]), 4)
        return
    mean, ci_lo, ci_hi = bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)
    out[key] = round(mean, 4)
    if len(values) >= MIN_N_FOR_CI:
        out[key + "_ci95"] = [round(ci_lo, 4), round(ci_hi, 4)]


def percentile(values: Sequence[float], pct: float) -> float:
    """Compute linear-interpolated percentile ``pct`` in ``[0, 100]``."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    clamped = max(0.0, min(100.0, pct))
    pos = (clamped / 100.0) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def prompt_sha256(text: str) -> str:
    """Return a SHA-256 hex digest identifying a judge/answer prompt template."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _detect_git_sha() -> str:
    git_bin = shutil.which("git")
    if not git_bin:
        return "unknown"
    try:
        out = subprocess.check_output(
            [git_bin, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip()
    except Exception:
        return "unknown"


def build_run_manifest(
    *,
    benchmark: str,
    profile: str,
    data_file: str,
    evaluated_questions: int,
    total_questions: int,
    judge_model: str,
    judge_prompt_hash: str,
    answer_model: str,
    embed_model: str,
    max_tokens: int,
    judge_fallbacks: int,
    estimated_usd: float,
    metrics: dict[str, Any],
    reproduce_cmd: str = "",
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Build a canonical immutable benchmark run manifest (INIT-023/SPEC-003)."""
    return {
        "schema_version": "3.0",
        "benchmark": benchmark,
        "git_sha": git_sha or _detect_git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "profile": profile,
        "data_file": data_file,
        "total_questions": total_questions,
        "evaluated_questions": evaluated_questions,
        "models": {
            "embed_model": embed_model,
            "answer_model": answer_model,
            "judge_model": judge_model,
            "judge_prompt_sha256": judge_prompt_hash,
        },
        "max_tokens": max_tokens,
        "judge_fallbacks": judge_fallbacks,
        "estimated_usd": round(estimated_usd, 6),
        "metrics": metrics,
        "reproduce_cmd": reproduce_cmd,
    }
