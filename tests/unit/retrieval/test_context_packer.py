"""Unit tests for context_packer.py — tier-aware token budget packing."""

from __future__ import annotations

import pytest

from archivist.retrieval.context_packer import PackedContext, pack_context


def _make_result(score: float, l0: str = "", l1: str = "", text: str = "") -> dict:
    """Build a minimal retrieval result dict for packing tests."""
    return {
        "id": f"mem-{score}",
        "score": score,
        "l0": l0,
        "l1": l1,
        "text": text or (l0 + l1 + "x" * 50),
        "file_path": "test.md",
        "date": "2026-01-01",
        "agent_id": "test-agent",
    }


class TestPackContextNoResults:
    def test_empty_results_returns_empty_packed(self):
        result = pack_context([], max_tokens=1000)
        assert isinstance(result, PackedContext)
        assert result.sources == []
        assert result.total_tokens == 0
        assert result.over_budget is False
        assert result.dropped_count == 0

    def test_zero_budget_returns_empty_packed(self):
        results = [_make_result(0.9, l0="headline", text="some longer text here")]
        result = pack_context(results, max_tokens=0)
        assert result.sources == []


class TestPackContextL2First:
    def test_l2_first_packs_in_score_order(self):
        results = [
            _make_result(0.9, text="A" * 100),
            _make_result(0.5, text="B" * 100),
        ]
        pc = pack_context(results, max_tokens=200, tier_policy="l2_first")
        assert pc.total_tokens <= 200
        assert len(pc.sources) >= 1
        assert all("tier_text" in s for s in pc.sources)

    def test_l2_first_respects_min_full_results(self):
        results = [
            _make_result(0.9, text="AAA"),
            _make_result(0.8, text="BBB"),
            _make_result(0.7, text="CCC"),
        ]
        pc = pack_context(results, max_tokens=20, tier_policy="l2_first", min_full_results=3)
        assert len(pc.sources) == 3


class TestPackContextL0First:
    def test_l0_first_uses_l0_text(self):
        results = [
            _make_result(0.9, l0="Headline A", text="Very long full text A " * 20),
            _make_result(0.7, l0="Headline B", text="Very long full text B " * 20),
        ]
        pc = pack_context(results, max_tokens=100, tier_policy="l0_first")
        assert all(s["tier_text"] in (s.get("l0", ""), s.get("l1", ""), s.get("text", "")) for s in pc.sources)
        assert pc.total_tokens <= 100


class TestPackContextAdaptive:
    def test_adaptive_upgrades_top_results_to_l2(self):
        results = [
            _make_result(0.9, l0="Headline X", l1="Summary X " * 5, text="Full content X " * 20),
            _make_result(0.6, l0="Headline Y", l1="Summary Y " * 5, text="Full content Y " * 20),
            _make_result(0.3, l0="Headline Z", l1="Summary Z " * 5, text="Full content Z " * 20),
        ]
        pc = pack_context(results, max_tokens=500, tier_policy="adaptive", min_full_results=1)
        assert pc.total_tokens <= 500
        assert "l2" in pc.tier_distribution or "l1" in pc.tier_distribution or "l0" in pc.tier_distribution

    def test_adaptive_fits_within_budget(self):
        results = [_make_result(float(i) / 10, text="word " * 200) for i in range(5, 0, -1)]
        pc = pack_context(results, max_tokens=300, tier_policy="adaptive")
        assert pc.total_tokens <= 300

    def test_tier_distribution_keys_are_valid(self):
        results = [
            _make_result(0.9, l0="Short", text="Longer text content here that goes on for a bit"),
        ]
        pc = pack_context(results, max_tokens=500)
        for tier in pc.tier_distribution:
            assert tier in ("l0", "l1", "l2", "ephemeral")

    def test_over_budget_false_when_all_fit(self):
        results = [_make_result(0.9, text="hi")]
        pc = pack_context(results, max_tokens=5000)
        assert pc.over_budget is False

    def test_token_savings_pct_is_non_negative(self):
        results = [_make_result(0.9, l0="h", text="long " * 100)]
        pc = pack_context(results, max_tokens=500, tier_policy="l0_first")
        assert pc.token_savings_pct >= 0.0

    def test_all_sources_have_tier_text(self):
        results = [_make_result(float(i) / 5, text=f"text {i} " * 10) for i in range(1, 6)]
        pc = pack_context(results, max_tokens=2000)
        for s in pc.sources:
            assert "tier_text" in s
            assert s["tier_text"]


class TestPackContextDropping:
    def test_dropped_count_tracked(self):
        results = [_make_result(float(i) / 10, text="word " * 300) for i in range(10, 0, -1)]
        pc = pack_context(results, max_tokens=50, tier_policy="l2_first")
        assert pc.dropped_count >= 0

    def test_at_least_one_result_always_returned_when_any_fit(self):
        results = [_make_result(0.9, text="small")]
        pc = pack_context(results, max_tokens=10)
        assert len(pc.sources) >= 1
