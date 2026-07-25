"""Unit tests for lineage + token USD estimates (INIT-001/SPEC-011)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Rate math
# ---------------------------------------------------------------------------


class TestEstimateTokenUsd:
    def test_rate_unset_returns_null(self):
        from archivist.app.dashboard import estimate_token_usd

        assert estimate_token_usd(10_000, rate_per_1k=None) is None

    def test_basic_rate_math(self):
        from archivist.app.dashboard import estimate_token_usd

        # 5000 tokens × $0.002 / 1k = $0.01
        assert estimate_token_usd(5000, rate_per_1k=0.002) == pytest.approx(0.01)

    def test_zero_tokens(self):
        from archivist.app.dashboard import estimate_token_usd

        assert estimate_token_usd(0, rate_per_1k=0.01) == 0.0

    def test_none_tokens_returns_null(self):
        from archivist.app.dashboard import estimate_token_usd

        assert estimate_token_usd(None, rate_per_1k=0.01) is None


class TestAttachCostEstimates:
    def test_null_when_rate_unset(self):
        from archivist.app.dashboard import attach_cost_estimates

        with patch("archivist.app.dashboard.TOKEN_USD_PER_1K", None):
            out = attach_cost_estimates(
                {
                    "total_tokens_saved": 1000,
                    "total_tokens_returned": 500,
                    "total_tokens_naive": 1500,
                }
            )
        assert out["token_usd_per_1k"] is None
        assert out["estimated_usd_saved"] is None
        assert out["estimated_usd_returned"] is None
        assert out["estimated_usd_naive"] is None

    def test_usd_fields_when_rate_set(self):
        from archivist.app.dashboard import attach_cost_estimates

        out = attach_cost_estimates(
            {
                "total_tokens_saved": 2000,
                "total_tokens_returned": 1000,
                "total_tokens_naive": 3000,
            },
            rate_per_1k=0.001,
        )
        assert out["token_usd_per_1k"] == 0.001
        assert out["estimated_usd_saved"] == pytest.approx(0.002)
        assert out["estimated_usd_returned"] == pytest.approx(0.001)
        assert out["estimated_usd_naive"] == pytest.approx(0.003)


# ---------------------------------------------------------------------------
# Empty lineage + id validation
# ---------------------------------------------------------------------------


class TestLineageValidation:
    def test_reject_empty_memory_id(self):
        from archivist.app.lineage import validate_memory_id

        assert validate_memory_id("") is None
        assert validate_memory_id("  ") is None

    def test_reject_path_injection(self):
        from archivist.app.lineage import validate_memory_id

        assert validate_memory_id("../etc/passwd") is None
        assert validate_memory_id("mem;drop") is None

    def test_accept_uuid_like(self):
        from archivist.app.lineage import validate_memory_id

        assert validate_memory_id("abc123-def456") == "abc123-def456"


class TestEmptyLineage:
    @pytest.mark.asyncio
    async def test_unknown_memory_returns_empty_edges(self):
        from archivist.app.lineage import build_memory_lineage

        mock_conn = AsyncMock()
        mock_conn.fetchone = AsyncMock(return_value=None)
        mock_conn.fetchall = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.read.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.read.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("archivist.app.lineage.pool", mock_pool),
            patch(
                "archivist.app.lineage._provenance_edges",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "archivist.app.lineage.resolve_memory_namespace",
                new=AsyncMock(return_value=""),
            ),
            patch(
                "archivist.core.audit.get_audit_trail",
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = await build_memory_lineage("missing-memory-id")

        assert result["resource_id"] == "missing-memory-id"
        assert result["edge_count"] == 0
        assert result["edges"] == []
        assert result["sources"] == []


# ---------------------------------------------------------------------------
# Handler: RBAC deny + cost keys on savings dashboard
# ---------------------------------------------------------------------------


class TestMemoryLineageHandlerRbac:
    @pytest.mark.asyncio
    async def test_unauthorized_namespace_denied(self):
        from mcp.types import TextContent

        from archivist.app.handlers.tools_admin import _handle_memory_lineage

        lineage_payload = {
            "resource_type": "memory",
            "resource_id": "mem-1",
            "namespace": "secret-ns",
            "edge_count": 1,
            "edges": [{"edge_type": "audit", "from_id": "a", "to_id": "mem-1"}],
            "sources": ["audit"],
        }
        denied = [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "access_denied",
                        "reason": "Agent 'spy' cannot read namespace 'secret-ns'",
                    }
                ),
            )
        ]

        with (
            patch(
                "archivist.app.handlers.tools_admin.require_caller",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_admin.resolve_caller",
                return_value="spy",
            ),
            patch(
                "archivist.core.rbac.is_permissive_mode",
                return_value=False,
            ),
            patch(
                "archivist.app.handlers.tools_admin.require_rbac",
                return_value=denied,
            ),
            patch(
                "archivist.app.lineage.resolve_memory_namespace",
                new=AsyncMock(return_value="secret-ns"),
            ),
            patch(
                "archivist.app.lineage.build_memory_lineage",
                new=AsyncMock(return_value=lineage_payload),
            ),
        ):
            response = await _handle_memory_lineage(
                {
                    "agent_id": "spy",
                    "memory_id": "mem-1",
                    "namespace": "secret-ns",
                }
            )

        data = json.loads(response[0].text)
        assert data["error"] == "access_denied"

    @pytest.mark.asyncio
    async def test_claimed_namespace_cannot_widen_access(self):
        """SEC-012-01: RBAC must bind to the memory's owner namespace."""
        from archivist.app.handlers.tools_admin import _handle_memory_lineage

        build = AsyncMock(
            return_value={
                "resource_type": "memory",
                "resource_id": "mem-1",
                "namespace": "victim-ns",
                "edge_count": 1,
                "edges": [{"edge_type": "audit", "from_id": "a", "to_id": "mem-1"}],
                "sources": ["audit"],
            }
        )
        rbac = MagicMock(return_value=None)

        with (
            patch("archivist.app.handlers.tools_admin.require_caller", return_value=None),
            patch("archivist.app.handlers.tools_admin.resolve_caller", return_value="spy"),
            patch("archivist.core.rbac.is_permissive_mode", return_value=False),
            patch("archivist.app.handlers.tools_admin.require_rbac", rbac),
            patch(
                "archivist.app.lineage.resolve_memory_namespace",
                new=AsyncMock(return_value="victim-ns"),
            ),
            patch("archivist.app.lineage.build_memory_lineage", new=build),
        ):
            response = await _handle_memory_lineage(
                {
                    "agent_id": "spy",
                    "memory_id": "mem-1",
                    # Attacker claims a namespace they *do* have read access to.
                    "namespace": "spy-own-ns",
                }
            )

        data = json.loads(response[0].text)
        assert data["error"] == "access_denied"
        # Lineage must not even be assembled for a mismatched claim.
        build.assert_not_awaited()
        rbac.assert_not_called()

    @pytest.mark.asyncio
    async def test_rbac_evaluated_against_owner_namespace(self):
        """SEC-012-01: the resolved owner namespace is what gets authorized."""
        from archivist.app.handlers.tools_admin import _handle_memory_lineage

        rbac = MagicMock(return_value=None)
        with (
            patch("archivist.app.handlers.tools_admin.require_caller", return_value=None),
            patch("archivist.app.handlers.tools_admin.resolve_caller", return_value="agent-a"),
            patch("archivist.core.rbac.is_permissive_mode", return_value=False),
            patch("archivist.app.handlers.tools_admin.require_rbac", rbac),
            patch(
                "archivist.app.lineage.resolve_memory_namespace",
                new=AsyncMock(return_value="owner-ns"),
            ),
            patch(
                "archivist.app.lineage.build_memory_lineage",
                new=AsyncMock(
                    return_value={
                        "resource_type": "memory",
                        "resource_id": "mem-1",
                        "namespace": "owner-ns",
                        "edge_count": 0,
                        "edges": [],
                        "sources": [],
                    }
                ),
            ),
        ):
            response = await _handle_memory_lineage({"agent_id": "agent-a", "memory_id": "mem-1"})

        data = json.loads(response[0].text)
        assert data["namespace"] == "owner-ns"
        rbac.assert_called_once_with("agent-a", "read", "owner-ns")


class TestLineageNamespaceScoping:
    """SEC-012-02 / SEC-012-04: entity scoping and LIKE-pattern containment."""

    def test_like_escaping_neutralizes_wildcards(self):
        from archivist.app.lineage import _escape_like

        assert _escape_like("a_b%c") == "a\\_b\\%c"
        assert _escape_like("plain-id") == "plain-id"

    @pytest.mark.asyncio
    async def test_entity_queries_filter_by_namespace(self):
        from archivist.app.lineage import build_entity_lineage

        mock_conn = AsyncMock()
        mock_conn.fetchone = AsyncMock(return_value=None)
        mock_conn.fetchall = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.read.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.read.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("archivist.app.lineage.pool", mock_pool):
            await build_entity_lineage("Acme Corp", namespace="tenant-a")

        sql, params = mock_conn.fetchone.await_args.args
        assert "namespace = ?" in sql
        assert params[-1] == "tenant-a"

    @pytest.mark.asyncio
    async def test_retrieval_edges_escape_underscore_ids(self):
        from archivist.app.lineage import _retrieval_edges

        mock_conn = AsyncMock()
        mock_conn.fetchall = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.read.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.read.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("archivist.app.lineage.pool", mock_pool):
            await _retrieval_edges("a_______", 10)

        sql, params = mock_conn.fetchall.await_args.args
        assert "ESCAPE" in sql
        assert params[0] == "%a\\_\\_\\_\\_\\_\\_\\_%"


class TestSavingsDashboardCostKeys:
    @pytest.mark.asyncio
    async def test_savings_payload_includes_estimated_usd_keys(self):
        mock_savings = {
            "total_queries": 5,
            "total_tokens_saved": 1000,
            "total_tokens_returned": 500,
            "total_tokens_naive": 1500,
            "avg_savings_pct": 42.0,
            "per_policy": [],
            "token_usd_per_1k": 0.002,
            "estimated_usd_saved": 0.002,
            "estimated_usd_returned": 0.001,
            "estimated_usd_naive": 0.003,
        }
        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.read.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.read.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("archivist.app.handlers.tools_admin.pool", mock_pool),
            patch(
                "archivist.app.dashboard._token_savings_stats",
                new=AsyncMock(return_value=mock_savings),
            ),
            patch(
                "archivist.app.dashboard._tier_distribution_stats",
                new=AsyncMock(return_value={"by_pack_policy": []}),
            ),
            patch(
                "archivist.app.dashboard._hotness_heatmap",
                new=AsyncMock(return_value=[]),
            ),
        ):
            from archivist.app.handlers.tools_admin import _handle_savings_dashboard

            response = await _handle_savings_dashboard({"window_days": 7})

        data = json.loads(response[0].text)
        ts = data["token_savings"]
        assert "estimated_usd_saved" in ts
        assert "token_usd_per_1k" in ts
        assert ts["estimated_usd_saved"] == pytest.approx(0.002)
