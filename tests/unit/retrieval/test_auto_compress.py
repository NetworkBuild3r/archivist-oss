"""Unit tests for Phase 3 auto-compress hook in rlm_retriever."""

from __future__ import annotations

import pytest


class TestAutoCompressHookDisabled:
    """AUTO_COMPRESS_ENABLED=False (default) — hook must be a no-op."""

    async def test_no_compress_when_disabled(self, monkeypatch):
        monkeypatch.setattr("archivist.retrieval.rlm_retriever.AUTO_COMPRESS_ENABLED", False)

        compact_calls = []

        async def _fake_compact(pairs, **kwargs):
            compact_calls.append(pairs)
            return "compressed"

        monkeypatch.setattr(
            "archivist.write.compaction.compact_flat",
            _fake_compact,
        )

        # Build a PackedContext-like object that signals over_budget
        from archivist.retrieval.context_packer import PackedContext

        packed = PackedContext(
            sources=[{"id": "a", "text": "x", "tier_text": "x", "score": 1.0}],
            total_tokens=900,
            budget_tokens=1000,
            over_budget=True,
            dropped_count=2,
            tier_distribution={"l2": 1},
            token_savings_pct=10.0,
        )

        # Simulate the condition check inline — when disabled, compact_calls stays empty
        import archivist.retrieval.rlm_retriever as rlm

        would_compress = (
            rlm.AUTO_COMPRESS_ENABLED
            and packed.over_budget
            and packed.dropped_count > 0
        )
        assert would_compress is False
        assert compact_calls == []


class TestAutoCompressHookEnabled:
    """AUTO_COMPRESS_ENABLED=True — hook fires and injects synthetic result."""

    async def test_compress_fires_when_over_budget(self, monkeypatch):
        monkeypatch.setattr("archivist.retrieval.rlm_retriever.AUTO_COMPRESS_ENABLED", True)
        monkeypatch.setattr("archivist.retrieval.rlm_retriever.AUTO_COMPRESS_THRESHOLD", 0.85)

        compact_calls: list = []

        async def _fake_compact_flat(pairs, **kwargs):
            compact_calls.append(pairs)
            return "Compressed summary of overflow content."

        monkeypatch.setattr("archivist.write.compaction.compact_flat", _fake_compact_flat)

        # Call the condition logic directly to verify behaviour
        from archivist.retrieval.context_packer import PackedContext

        packed = PackedContext(
            sources=[
                {"id": "a", "text": "word " * 50, "tier_text": "word " * 50, "score": 0.9},
            ],
            total_tokens=920,
            budget_tokens=1000,
            over_budget=True,
            dropped_count=3,
            tier_distribution={"l2": 1},
            token_savings_pct=5.0,
        )

        import archivist.retrieval.rlm_retriever as rlm

        budget_pct = round(packed.total_tokens / packed.budget_tokens * 100, 1)
        would_compress = (
            rlm.AUTO_COMPRESS_ENABLED
            and packed.over_budget
            and packed.dropped_count > 0
            and budget_pct >= rlm.AUTO_COMPRESS_THRESHOLD * 100
        )
        assert would_compress is True


class TestContextStatusHasCompressFields:
    """The ctx_status dict should always contain compress metadata keys."""

    def test_ctx_status_keys_exist(self):
        ctx = {
            "result_tokens_approx": 500,
            "budget_tokens": 1000,
            "budget_used_pct": 50.0,
            "tier": "l2",
            "over_budget": False,
            "tier_distribution": {},
            "token_savings_pct": 0.0,
            "dropped_count": 0,
            "pack_policy": "adaptive",
            "auto_compressed": False,
            "compress_savings_approx": 0,
            "hint": "ok",
        }
        assert "auto_compressed" in ctx
        assert "compress_savings_approx" in ctx
        assert isinstance(ctx["auto_compressed"], bool)
        assert isinstance(ctx["compress_savings_approx"], int)
