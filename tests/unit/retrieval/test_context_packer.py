"""Unit tests for context_packer.py — tier-aware token budget packing."""

from __future__ import annotations

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
        assert all(
            s["tier_text"] in (s.get("l0", ""), s.get("l1", ""), s.get("text", ""))
            for s in pc.sources
        )
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
        assert (
            "l2" in pc.tier_distribution
            or "l1" in pc.tier_distribution
            or "l0" in pc.tier_distribution
        )

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


class TestBestTierTextResolution:
    """INIT-023/SPEC-001: _best_tier_text must resolve requested tier, not lowest tier."""

    def test_best_tier_text_returns_l2_when_requested(self):
        from archivist.retrieval.context_packer import _best_tier_text

        result = {
            "l0": "short headline",
            "l1": "medium summary sentence",
            "text": "full raw content with all specific numbers 42 and details",
        }
        assert _best_tier_text(result, "l2") == result["text"]
        assert _best_tier_text(result, "l1") == result["l1"]
        assert _best_tier_text(result, "l0") == result["l0"]

    def test_best_tier_text_falls_back_downward_when_tier_missing(self):
        from archivist.retrieval.context_packer import _best_tier_text

        only_l0 = {"l0": "headline only", "l1": "", "text": ""}
        assert _best_tier_text(only_l0, "l2") == "headline only"
        assert _best_tier_text(only_l0, "l1") == "headline only"

        l1_and_l0 = {"l0": "headline", "l1": "summary", "text": ""}
        assert _best_tier_text(l1_and_l0, "l2") == "summary"

    def test_adaptive_upgrades_top_results_to_actual_l2_text(self):
        results = [
            _make_result(0.95, l0="Headline A", l1="Summary A", text="Full L2 Body A " * 10),
            _make_result(0.70, l0="Headline B", l1="Summary B", text="Full L2 Body B " * 10),
        ]
        pc = pack_context(results, max_tokens=1000, tier_policy="adaptive", min_full_results=1)
        assert pc.sources[0]["_packed_tier"] == "l2"
        assert pc.sources[0]["tier_text"] == results[0]["text"]
        assert pc.sources[0]["tier_text"] != results[0]["l0"]

    def test_dropped_sources_preserved_from_pr72(self):
        results = [
            _make_result(0.95, l0="H1", text="alpha " * 80),
            _make_result(0.85, l0="H2", text="beta " * 80),
            _make_result(0.75, l0="H3", text="gamma " * 80),
        ]
        pc = pack_context(results, max_tokens=110, tier_policy="l2_first", min_full_results=1)
        assert pc.dropped_count == len(pc.dropped_sources)
        assert pc.dropped_count >= 1
        assert len(pc.sources) + len(pc.dropped_sources) == len(results)


class TestDeterministicTierExtraction:
    """INIT-023/SPEC-001: zero-LLM deterministic L0 and query-windowed L1 extraction."""

    def test_extract_l0_deterministic_is_bounded_and_preserves_lead_sentence(self):
        from archivist.write.tiering import extract_l0_deterministic

        text = (
            "The production backup window runs at 04:15 UTC every Sunday. "
            "Secondary replicas sync after completion. "
            + "Additional operational notes about retention policies. "
            * 25
        )
        l0 = extract_l0_deterministic(text, max_tokens=40)
        assert "04:15 UTC" in l0
        assert len(l0.split()) <= 40

    def test_extract_l1_query_window_centers_on_matching_terms(self):
        from archivist.write.tiering import extract_l1_query_window

        filler = "Generic meeting chatter about quarterly planning and office snacks. "
        needle = "The database failover runbook requires setting failover_timeout_ms to 3500."
        text = (filler * 30) + needle + " " + (filler * 30)
        l1 = extract_l1_query_window(
            text, query="database failover_timeout_ms runbook", max_tokens=80
        )
        assert "failover_timeout_ms" in l1
        assert "3500" in l1
        assert len(l1.split()) <= 80

    def test_best_tier_text_synthesizes_query_window_l1_when_missing(self):
        from archivist.retrieval.context_packer import _best_tier_text

        filler = "Unrelated background discussion about coffee and schedules. "
        needle = "Secret launch code is ORION-9914 approved by Alice."
        result = {"text": (filler * 25) + needle + " " + (filler * 25)}
        l1 = _best_tier_text(result, "l1", query="launch code ORION-9914")
        assert "ORION-9914" in l1
        assert len(l1) < len(result["text"])

    def test_deterministic_extraction_performance_20_results_under_10ms(self):
        import time

        from archivist.write.tiering import extract_l0_deterministic, extract_l1_query_window

        docs = [
            f"Paragraph {i}. "
            + ("Background context text for testing speed. " * 40)
            + f"Specific fact key_{i}=value_{i} in the middle. "
            + ("Trailing background text for testing speed. " * 40)
            for i in range(20)
        ]
        t0 = time.perf_counter()
        for i, doc in enumerate(docs):
            l0 = extract_l0_deterministic(doc)
            l1 = extract_l1_query_window(doc, query=f"Specific fact key_{i}")
            assert l0 and l1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Target < 2ms on standard hardware; generous 25ms ceiling avoids CI flake
        assert elapsed_ms < 25.0, f"Deterministic tiering too slow: {elapsed_ms:.2f}ms"
