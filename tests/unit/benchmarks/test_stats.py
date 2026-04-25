"""Unit tests for benchmarks/stats.py — bootstrap_ci and maybe_ci."""

from __future__ import annotations

import pytest
from benchmarks.stats import MIN_N_FOR_CI, bootstrap_ci, maybe_ci

# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_returns_mean_within_values():
    values = [0.0] * 50 + [1.0] * 50  # true mean = 0.5
    mean, lo, hi = bootstrap_ci(values, n_boot=500)
    assert abs(mean - 0.5) < 1e-9


def test_ci_contains_mean():
    values = [float(i % 2) for i in range(60)]
    mean, lo, hi = bootstrap_ci(values, n_boot=500)
    assert lo <= mean <= hi


def test_ci_interval_ordered():
    values = [0.3, 0.7, 0.5, 0.6, 0.4] * 10
    _, lo, hi = bootstrap_ci(values, n_boot=500)
    assert lo <= hi


def test_deterministic_with_same_seed():
    values = [float(i % 3) / 2 for i in range(40)]
    r1 = bootstrap_ci(values, n_boot=200, seed=7)
    r2 = bootstrap_ci(values, n_boot=200, seed=7)
    assert r1 == r2


def test_different_seeds_may_differ():
    """With only 2 points the CI is trivially constant — use a real distribution."""
    values = [float(i % 7) / 6 for i in range(50)]
    r1 = bootstrap_ci(values, n_boot=200, seed=1)
    r2 = bootstrap_ci(values, n_boot=200, seed=2)
    # Means are the same; CIs may differ
    assert r1[0] == r2[0]


def test_raises_on_empty():
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_ci([])


def test_raises_on_single_element():
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_ci([1.0])


def test_perfect_score_ci():
    """All-ones: mean=1.0, CI should be [1.0, 1.0]."""
    values = [1.0] * 50
    mean, lo, hi = bootstrap_ci(values, n_boot=200)
    assert mean == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_zero_score_ci():
    """All-zeros: mean=0.0, CI should be [0.0, 0.0]."""
    values = [0.0] * 50
    mean, lo, hi = bootstrap_ci(values, n_boot=200)
    assert mean == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_ci_narrower_with_more_data():
    """More data → narrower interval (on the same distribution)."""
    small = [float(i % 2) for i in range(20)]
    large = [float(i % 2) for i in range(200)]
    _, lo_s, hi_s = bootstrap_ci(small, n_boot=500, seed=42)
    _, lo_l, hi_l = bootstrap_ci(large, n_boot=500, seed=42)
    assert (hi_l - lo_l) <= (hi_s - lo_s)


# ---------------------------------------------------------------------------
# maybe_ci
# ---------------------------------------------------------------------------


def test_maybe_ci_sets_mean_key():
    values = [0.6] * 40
    out: dict = {}
    maybe_ci(values, "recall_at_5", out)
    assert "recall_at_5" in out
    assert abs(out["recall_at_5"] - 0.6) < 1e-4


def test_maybe_ci_adds_ci95_when_n_ge_30():
    values = [float(i % 2) for i in range(MIN_N_FOR_CI)]
    out: dict = {}
    maybe_ci(values, "recall_at_5", out)
    assert "recall_at_5_ci95" in out
    lo, hi = out["recall_at_5_ci95"]
    assert lo <= out["recall_at_5"] <= hi


def test_maybe_ci_omits_ci95_when_n_lt_30():
    values = [float(i % 2) for i in range(MIN_N_FOR_CI - 1)]
    out: dict = {}
    maybe_ci(values, "recall_at_5", out)
    assert "recall_at_5" in out
    assert "recall_at_5_ci95" not in out


def test_maybe_ci_empty_values_sets_zero():
    out: dict = {}
    maybe_ci([], "recall_at_5", out)
    assert out["recall_at_5"] == 0.0
    assert "recall_at_5_ci95" not in out


def test_maybe_ci_does_not_overwrite_other_keys():
    out = {"other_key": 99}
    maybe_ci([0.5] * 40, "accuracy", out)
    assert out["other_key"] == 99
