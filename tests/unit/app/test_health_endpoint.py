"""Unit tests for enriched /health and /debug/config endpoints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(path: str = "/health", api_key: str = "") -> MagicMock:
    req = MagicMock()
    req.url.path = path
    req.headers.get = MagicMock(return_value=f"Bearer {api_key}" if api_key else "")
    req.query_params = {}
    return req


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


class TestHandleHealth:
    """handle_health() returns correct status and HTTP code."""

    @pytest.fixture(autouse=True)
    def _post_startup(self, monkeypatch):
        # Production marks startup complete at end of _startup(); tests only
        # patch subsystem status — simulate post-startup so /health is not stuck
        # on {"status": "starting"} with HTTP 200.
        import archivist.core.health as health

        monkeypatch.setattr(health, "_startup_complete", True)

    @pytest.mark.asyncio
    async def test_healthy_returns_200(self, monkeypatch):
        import archivist.core.health as health
        from archivist.app.main import handle_health

        # Start from a clean slate — register only healthy subsystems
        monkeypatch.setattr(
            health,
            "_status",
            {"qdrant": {"healthy": True, "detail": "", "since": "t", "latency_ms": 0.0}},
        )

        resp = await handle_health(_make_request())
        assert resp.status_code == 200
        body = json.loads(resp.body)
        assert body["status"] == "healthy"
        assert "subsystems" in body
        assert "timestamp" in body
        assert body["service"] == "archivist"

    @pytest.mark.asyncio
    async def test_degraded_returns_503(self, monkeypatch):
        import archivist.core.health as health
        from archivist.app.main import handle_health

        monkeypatch.setattr(
            health,
            "_status",
            {
                "qdrant": {"healthy": True, "detail": "", "since": "t", "latency_ms": 0.0},
                "postgres": {
                    "healthy": False,
                    "detail": "pool not init",
                    "since": "t",
                    "latency_ms": 0.0,
                },
            },
        )

        resp = await handle_health(_make_request())
        assert resp.status_code == 503
        body = json.loads(resp.body)
        assert body["status"] == "degraded"
        assert "postgres" in body["subsystems"]
        assert body["subsystems"]["postgres"]["healthy"] is False

    @pytest.mark.asyncio
    async def test_empty_subsystems_returns_200(self, monkeypatch):
        import archivist.core.health as health
        from archivist.app.main import handle_health

        monkeypatch.setattr(health, "_status", {})
        resp = await handle_health(_make_request())
        assert resp.status_code == 200
        body = json.loads(resp.body)
        assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_response_includes_version(self, monkeypatch):
        import archivist.core.health as health
        from archivist.app.main import handle_health

        monkeypatch.setattr(health, "_status", {})
        resp = await handle_health(_make_request())
        body = json.loads(resp.body)
        assert "version" in body
        assert body["version"]  # non-empty

    @pytest.mark.asyncio
    async def test_all_unhealthy_returns_503(self, monkeypatch):
        import archivist.core.health as health
        from archivist.app.main import handle_health

        monkeypatch.setattr(
            health,
            "_status",
            {
                "qdrant": {"healthy": False, "detail": "down", "since": "t", "latency_ms": 0.0},
                "fts5": {
                    "healthy": False,
                    "detail": "init failed",
                    "since": "t",
                    "latency_ms": 0.0,
                },
            },
        )
        resp = await handle_health(_make_request())
        assert resp.status_code == 503
        body = json.loads(resp.body)
        assert body["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_during_startup_returns_200_starting_even_if_degraded(self, monkeypatch):
        """Before mark_startup_complete, liveness must not get 503 from degraded subsystems."""
        import archivist.core.health as health
        from archivist.app.main import handle_health

        monkeypatch.setattr(health, "_startup_complete", False)
        monkeypatch.setattr(
            health,
            "_status",
            {
                "qdrant": {"healthy": False, "detail": "not yet", "since": "t", "latency_ms": 0.0},
            },
        )
        resp = await handle_health(_make_request())
        assert resp.status_code == 200
        body = json.loads(resp.body)
        assert body["status"] == "starting"


# ---------------------------------------------------------------------------
# /debug/config endpoint
# ---------------------------------------------------------------------------


class TestHandleDebugConfig:
    """handle_debug_config() returns expected keys and no secrets."""

    @pytest.mark.asyncio
    async def test_returns_expected_keys(self):
        from archivist.app.main import handle_debug_config

        resp = await handle_debug_config(_make_request("/debug/config"))
        assert resp.status_code == 200
        body = json.loads(resp.body)

        expected_keys = {
            "graph_backend",
            "metrics_enabled",
            "bm25_enabled",
            "outbox_enabled",
            "reranker_enabled",
            "curator_interval_minutes",
            "qdrant_collection",
            "vector_dim",
            "timestamp",
        }
        assert expected_keys.issubset(body.keys()), f"missing keys: {expected_keys - body.keys()}"

    @pytest.mark.asyncio
    async def test_does_not_expose_secrets(self):
        from archivist.app.main import handle_debug_config

        resp = await handle_debug_config(_make_request("/debug/config"))
        body = json.loads(resp.body)

        # These must never appear in the response
        forbidden_keys = {"archivist_api_key", "database_url", "api_key", "password", "secret"}
        lower_keys = {k.lower() for k in body.keys()}
        assert not forbidden_keys.intersection(lower_keys), (
            f"secret key(s) exposed: {forbidden_keys.intersection(lower_keys)}"
        )

    @pytest.mark.asyncio
    async def test_graph_backend_is_string(self):
        from archivist.app.main import handle_debug_config

        resp = await handle_debug_config(_make_request("/debug/config"))
        body = json.loads(resp.body)
        assert isinstance(body["graph_backend"], str)
        assert body["graph_backend"] in ("sqlite", "postgres")

    @pytest.mark.asyncio
    async def test_timestamp_is_iso_format(self):
        from datetime import datetime

        from archivist.app.main import handle_debug_config

        resp = await handle_debug_config(_make_request("/debug/config"))
        body = json.loads(resp.body)
        ts = body["timestamp"]
        # Should parse without error
        datetime.fromisoformat(ts)
