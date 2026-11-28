"""Live async ASGI loop smoke tests with >50ms stall watchdog (INIT-025/SPEC-011, SPEC-012).

Exercises Starlette ASGI routes directly over ``httpx.ASGITransport`` inside
the running asyncio loop with ``event_loop_watchdog`` active, verifying:
  - ``GET /health`` returns 200 healthy payload with service + version
  - ``GET /metrics`` returns Prometheus text satisfying the ``le="+Inf" == _count`` invariant
  - ``GET /debug/config`` returns non-secret runtime settings
  - ``GET /admin/invalidate`` is rejected (405 Method Not Allowed)
  - Zero event-loop stalls (>50ms) on async HTTP handlers
"""

from __future__ import annotations

import re

import httpx
import pytest

import archivist.app.main as main
import archivist.core.health as health
import archivist.core.metrics as metrics

pytestmark = [pytest.mark.integration]

_HIST_BUCKET_INF_RE = re.compile(
    r'^(?P<name>[a-zA-Z0-9_:]+)_bucket\{(?P<labels>[^}]*le="\+Inf"[^}]*)\}\s+(?P<val>[0-9.eE+-]+)$'
)
_HIST_COUNT_RE = re.compile(
    r"^(?P<name>[a-zA-Z0-9_:]+)_count(?:\{(?P<labels>[^}]*)\})?\s+(?P<val>[0-9.eE+-]+)$"
)


def _strip_le_label(labels: str) -> str:
    parts = [p.strip() for p in labels.split(",") if p.strip()]
    kept = [p for p in parts if not p.startswith("le=")]
    return ",".join(sorted(kept))


def assert_prometheus_histogram_invariant(metrics_text: str) -> None:
    """Assert that every histogram's ``le="+Inf"`` bucket equals its ``_count`` series."""
    inf_buckets: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], float] = {}

    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m_inf = _HIST_BUCKET_INF_RE.match(line)
        if m_inf:
            key = (m_inf.group("name"), _strip_le_label(m_inf.group("labels")))
            inf_buckets[key] = float(m_inf.group("val"))
            continue
        m_cnt = _HIST_COUNT_RE.match(line)
        if m_cnt:
            key = (m_cnt.group("name"), _strip_le_label(m_cnt.group("labels") or ""))
            counts[key] = float(m_cnt.group("val"))

    assert len(inf_buckets) > 0, (
        "Expected at least one Prometheus histogram with le='+Inf' bucket in /metrics "
        "(histogram invariant must not pass vacuously)"
    )
    for key, inf_val in inf_buckets.items():
        assert key in counts, f"Histogram {key} has +Inf bucket but no _count series"
        assert inf_val == pytest.approx(counts[key]), (
            f"Prometheus histogram invariant broken for {key}: +Inf={inf_val} != _count={counts[key]}"
        )


async def test_live_asgi_routes_and_metrics_invariant_without_loop_stalls(
    monkeypatch,
    event_loop_watchdog,
) -> None:
    """Core ASGI routes respond cleanly with valid Prometheus histograms and zero >50ms stalls."""
    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "", raising=False)
    monkeypatch.setattr(main, "ARCHIVIST_ADMIN_API_KEY", "", raising=False)
    monkeypatch.setattr(main, "METRICS_ENABLED", True, raising=False)

    # Record a real histogram observation so /metrics emits histogram +Inf & _count series.
    metrics.observe(metrics.INVALIDATE_DURATION, 12.5)
    for subsystem in ("qdrant", "sqlite", "postgres", "embeddings"):
        health.register(subsystem, healthy=True)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        health_body = health_resp.json()
        assert health_body["status"] == "healthy"
        assert health_body["service"] == "archivist"
        assert health_body["version"]

        metrics_resp = await client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "# TYPE" in metrics_resp.text
        assert_prometheus_histogram_invariant(metrics_resp.text)

        cfg_resp = await client.get("/debug/config")
        assert cfg_resp.status_code == 200
        cfg_body = cfg_resp.json()
        assert "graph_backend" in cfg_body
        assert "bm25_enabled" in cfg_body

        inv_get = await client.get("/admin/invalidate")
        assert inv_get.status_code == 405

    # Verify live MCP Streamable HTTP initialize handshake on the running loop.
    mgr = main._create_streamable_http_session_manager()
    monkeypatch.setattr(main, "streamable_http_session_manager", mgr)
    async with mgr.run():
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            mcp_resp = await client.post(
                "/mcp",
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "asgi-live-test", "version": "1.0"},
                    },
                },
            )
            assert mcp_resp.status_code == 200
            assert "serverInfo" in mcp_resp.text

    assert event_loop_watchdog.stalls == []
