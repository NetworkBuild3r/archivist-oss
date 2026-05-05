"""Unit tests for Phase 5 observability additions.

Covers:
- retrieval_log.log_retrieval now accepts tokens_returned, tokens_naive, savings_pct, pack_policy
- context_packer.PackedContext now exposes naive_tokens
- dashboard._token_savings_stats, _tier_distribution_stats, _hotness_heatmap (mocked)
- archivist_savings_dashboard tool handler (smoke test)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from archivist.retrieval.context_packer import PackedContext, _pack_greedy, _pack_adaptive


# ---------------------------------------------------------------------------
# PackedContext.naive_tokens
# ---------------------------------------------------------------------------


class TestPackedContextNaiveTokens:
    def _make_result(self, l2_text: str, score: float = 0.9) -> dict:
        return {
            "text": l2_text,
            "l2": l2_text,
            "l0": l2_text[:30],
            "l1": l2_text[:80],
            "score": score,
        }

    def test_naive_tokens_present_in_greedy(self):
        results = [self._make_result("word " * 50, score=0.9 - i * 0.1) for i in range(5)]
        packed = _pack_greedy(results, max_tokens=500, prefer_tier="l2", min_full=1)
        assert packed.naive_tokens > 0

    def test_naive_tokens_present_in_adaptive(self):
        results = [self._make_result("token " * 60, score=0.9 - i * 0.1) for i in range(5)]
        packed = _pack_adaptive(results, max_tokens=200, l0_budget_share=0.3, min_full=1)
        assert packed.naive_tokens > 0

    def test_naive_tokens_geq_total_tokens(self):
        results = [self._make_result("filler " * 80, score=0.9 - i * 0.05) for i in range(6)]
        packed = _pack_greedy(results, max_tokens=100, prefer_tier="l2", min_full=1)
        assert packed.naive_tokens >= packed.total_tokens

    def test_naive_tokens_zero_for_empty_results(self):
        from archivist.retrieval.context_packer import pack_context

        packed = pack_context([], max_tokens=8000)
        assert packed.naive_tokens == 0

    def test_savings_pct_consistent_with_naive_and_total(self):
        results = [self._make_result("data " * 100, score=0.9 - i * 0.05) for i in range(4)]
        packed = _pack_greedy(results, max_tokens=100, prefer_tier="l2", min_full=1)
        if packed.naive_tokens > 0:
            expected = round((1 - packed.total_tokens / packed.naive_tokens) * 100, 1)
            assert packed.token_savings_pct == pytest.approx(expected, abs=0.2)


# ---------------------------------------------------------------------------
# log_retrieval new kwargs
# ---------------------------------------------------------------------------


class TestLogRetrievalNewColumns:
    @pytest.mark.asyncio
    async def test_log_retrieval_accepts_new_kwargs(self):
        """log_retrieval should accept and store the new token savings columns."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_pool = MagicMock()
        mock_pool.write.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.write.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("archivist.retrieval.retrieval_log.TRAJECTORY_EXPORT_ENABLED", True),
            patch("archivist.storage.sqlite_pool.pool", mock_pool),
        ):
            from archivist.retrieval.retrieval_log import log_retrieval

            log_id = await log_retrieval(
                agent_id="test_agent",
                query="what is archivist?",
                namespace="default",
                tier="l2",
                memory_type="general",
                retrieval_trace={"context_status": {}},
                result_count=5,
                cache_hit=False,
                duration_ms=120,
                tokens_returned=800,
                tokens_naive=2400,
                savings_pct=66.7,
                pack_policy="adaptive",
            )

        assert log_id != ""
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        sql = call_args[0]
        values = call_args[1]

        assert "tokens_returned" in sql
        assert "tokens_naive" in sql
        assert "savings_pct" in sql
        assert "pack_policy" in sql
        assert 800 in values
        assert 2400 in values
        assert 66.7 in values
        assert "adaptive" in values

    @pytest.mark.asyncio
    async def test_log_retrieval_none_values_allowed(self):
        """Token columns default to None when not supplied (backward compat)."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_pool = MagicMock()
        mock_pool.write.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.write.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("archivist.retrieval.retrieval_log.TRAJECTORY_EXPORT_ENABLED", True),
            patch("archivist.storage.sqlite_pool.pool", mock_pool),
        ):
            from archivist.retrieval.retrieval_log import log_retrieval

            log_id = await log_retrieval(
                agent_id="agent",
                query="query",
                namespace="",
                tier="l2",
                memory_type="",
                retrieval_trace={},
                result_count=0,
            )

        assert log_id != ""
        values = mock_conn.execute.call_args[0][1]
        assert values[-4] is None  # tokens_returned
        assert values[-3] is None  # tokens_naive
        assert values[-2] is None  # savings_pct
        assert values[-1] == ""   # pack_policy

    @pytest.mark.asyncio
    async def test_log_retrieval_returns_empty_when_disabled(self):
        with patch("archivist.retrieval.retrieval_log.TRAJECTORY_EXPORT_ENABLED", False):
            from archivist.retrieval.retrieval_log import log_retrieval

            result = await log_retrieval(
                agent_id="a",
                query="q",
                namespace="",
                tier="l2",
                memory_type="",
                retrieval_trace={},
                result_count=0,
            )
        assert result == ""


# ---------------------------------------------------------------------------
# get_token_savings_stats
# ---------------------------------------------------------------------------


class TestGetTokenSavingsStats:
    @pytest.mark.asyncio
    async def test_returns_dict_structure(self):
        mock_row = {
            "total": 10,
            "queries_with_savings_data": 8,
            "avg_savings_pct": 45.2,
            "min_savings_pct": 10.0,
            "max_savings_pct": 80.0,
            "total_tokens_saved": 12000,
            "total_tokens_returned": 8000,
            "total_tokens_naive": 20000,
            "avg_tokens_returned": 800.0,
            "avg_tokens_naive": 2000.0,
            "rows_with_savings": 8,
        }
        mock_conn = AsyncMock()
        mock_conn.fetchone = AsyncMock(return_value=mock_row)
        mock_conn.fetchall = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.read.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.read.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("archivist.storage.sqlite_pool.pool", mock_pool),
            patch("archivist.storage.graph._is_postgres", return_value=False),
        ):
            from archivist.retrieval.retrieval_log import get_token_savings_stats

            stats = await get_token_savings_stats(window_days=7)
        assert stats["avg_savings_pct"] == 45.2
        assert stats["total_tokens_saved"] == 12000
        assert "per_policy" in stats
        assert stats["window_days"] == 7


# ---------------------------------------------------------------------------
# dashboard helpers
# ---------------------------------------------------------------------------


class TestDashboardHelpers:
    @pytest.mark.asyncio
    async def test_token_savings_stats_returns_fallback_on_error(self):
        """_token_savings_stats returns an empty-safe dict when DB fails."""
        mock_conn = AsyncMock()
        mock_conn.fetchone = AsyncMock(side_effect=Exception("db error"))

        from archivist.app.dashboard import _token_savings_stats

        result = await _token_savings_stats(mock_conn, window_days=7)
        assert "total_queries" in result
        assert result["per_policy"] == []

    @pytest.mark.asyncio
    async def test_tier_distribution_stats_returns_fallback_on_error(self):
        mock_conn = AsyncMock()
        mock_conn.fetchall = AsyncMock(side_effect=Exception("db error"))

        from archivist.app.dashboard import _tier_distribution_stats

        result = await _tier_distribution_stats(mock_conn, window_days=7)
        assert "by_pack_policy" in result

    @pytest.mark.asyncio
    async def test_hotness_heatmap_returns_empty_on_error(self):
        mock_conn = AsyncMock()
        mock_conn.fetchall = AsyncMock(side_effect=Exception("db error"))

        mock_pool = MagicMock()
        mock_pool.read.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.read.return_value.__aexit__ = AsyncMock(return_value=False)

        from archivist.app.dashboard import _hotness_heatmap

        with patch("archivist.app.dashboard.pool", mock_pool):
            result = await _hotness_heatmap()

        assert result == []

    @pytest.mark.asyncio
    async def test_build_dashboard_includes_token_savings(self):
        """build_dashboard result now contains token_savings and tier_distribution keys."""
        mock_conn = AsyncMock()
        mock_conn.fetchone = AsyncMock(return_value=None)
        mock_conn.fetchall = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.read.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.read.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_hc = MagicMock()
        mock_hc.stats.return_value = {"enabled": False, "total_entries": 0, "agents": 0}

        with (
            patch("archivist.app.dashboard.pool", mock_pool),
            patch("archivist.app.dashboard._qdrant_stats", return_value={"total_points": 0}),
            patch("archivist.app.dashboard._stale_estimate", return_value={"stale_pct": 0}),
            patch("archivist.app.dashboard._hotness_heatmap", new=AsyncMock(return_value=[])),
            patch("archivist.app.dashboard.health.all_status", return_value={}),
            patch("archivist.retrieval.hot_cache.stats", return_value={"enabled": False, "total_entries": 0, "agents": 0}),
        ):
            from archivist.app.dashboard import build_dashboard

            dashboard = await build_dashboard(window_days=7)

        assert "token_savings" in dashboard
        assert "tier_distribution" in dashboard
        assert "hotness_heatmap" in dashboard


# ---------------------------------------------------------------------------
# archivist_savings_dashboard tool handler
# ---------------------------------------------------------------------------


class TestSavingsDashboardHandler:
    @pytest.mark.asyncio
    async def test_handler_returns_expected_keys(self):
        mock_savings = {
            "total_queries": 5,
            "avg_savings_pct": 42.0,
            "per_policy": [{"pack_policy": "adaptive", "cnt": 5, "avg_savings_pct": 42.0, "tokens_saved": 5000}],
            "window_days": 7,
        }
        mock_tier = {"by_pack_policy": []}
        mock_heatmap = [{"memory_id": "m1", "score": 0.9}]

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.read.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.read.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("archivist.app.handlers.tools_admin.pool", mock_pool),
            patch("archivist.app.dashboard._token_savings_stats", new=AsyncMock(return_value=mock_savings)),
            patch("archivist.app.dashboard._tier_distribution_stats", new=AsyncMock(return_value=mock_tier)),
            patch("archivist.app.dashboard._hotness_heatmap", new=AsyncMock(return_value=mock_heatmap)),
        ):
            from archivist.app.handlers.tools_admin import _handle_savings_dashboard

            response = await _handle_savings_dashboard({"window_days": 7, "heatmap_top_n": 10})

        import json as _json

        data = _json.loads(response[0].text)
        assert "token_savings" in data
        assert "tier_distribution" in data
        assert "hotness_heatmap" in data
        assert data["window_days"] == 7
