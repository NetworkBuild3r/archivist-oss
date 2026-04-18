"""Unit tests for retrieval threshold filtering."""

import pytest

from retrieval_filters import apply_retrieval_threshold

pytestmark = [pytest.mark.unit, pytest.mark.retrieval]


def test_threshold_filters_low_scores():
    """Results below threshold are filtered out."""
    results = [
        {"score": 0.9, "text": "high relevance"},
        {"score": 0.7, "text": "medium relevance"},
        {"score": 0.5, "text": "low relevance"},
        {"score": 0.3, "text": "noise"},
    ]
    filtered = apply_retrieval_threshold(results, threshold=0.65)
    assert len(filtered) == 2
    assert all(r["score"] >= 0.65 for r in filtered)


def test_threshold_returns_empty_on_no_match():
    """All results below threshold → empty list."""
    results = [
        {"score": 0.4, "text": "low"},
        {"score": 0.2, "text": "very low"},
    ]
    filtered = apply_retrieval_threshold(results, threshold=0.65)
    assert len(filtered) == 0


def test_threshold_zero_passes_all():
    """Threshold of 0 should pass everything."""
    results = [
        {"score": 0.1, "text": "a"},
        {"score": 0.9, "text": "b"},
    ]
    filtered = apply_retrieval_threshold(results, threshold=0.0)
    assert len(filtered) == 2


def test_threshold_one_filters_all_imperfect():
    """Threshold of 1.0 should only pass perfect scores."""
    results = [
        {"score": 0.99, "text": "almost"},
        {"score": 1.0, "text": "perfect"},
    ]
    filtered = apply_retrieval_threshold(results, threshold=1.0)
    assert len(filtered) == 1
    assert filtered[0]["text"] == "perfect"


def test_threshold_preserves_order():
    """Filtered results maintain original order."""
    results = [
        {"score": 0.8, "text": "first"},
        {"score": 0.9, "text": "second"},
        {"score": 0.7, "text": "third"},
    ]
    filtered = apply_retrieval_threshold(results, threshold=0.65)
    assert [r["text"] for r in filtered] == ["first", "second", "third"]
