"""Metrics hardening (v1.11): METRICS_ENABLED gating, storage gauges, naming, auth exempt."""

from starlette.testclient import TestClient


async def _noop_startup():
    """Avoid Qdrant/SQLite init when exercising HTTP routes via TestClient."""
    return None


def _clear_metrics():
    import metrics as m

    m._counters.clear()
    m._gauges.clear()
    m._histogram_buckets.clear()
    m._histogram_layout.clear()


def test_metrics_disabled_noop(monkeypatch):
    import metrics as m

    monkeypatch.setattr("config.METRICS_ENABLED", False)
    _clear_metrics()
    m.inc(m.SEARCH_TOTAL)
    m.observe(m.SEARCH_DURATION, 50.0)
    m.gauge_set(m.SQLITE_AVAILABLE, 1.0)
    assert m.render() == ""
    assert len(m._counters) == 0
    assert len(m._gauges) == 0
    assert len(m._histogram_buckets) == 0


def test_metrics_disabled_endpoint_404(monkeypatch):
    import main

    monkeypatch.setattr(main, "_startup", _noop_startup)
    monkeypatch.setattr(main, "METRICS_ENABLED", False)
    with TestClient(main.app) as client:
        r = client.get("/metrics")
    assert r.status_code == 404


def test_storage_gauge_constants_exist():
    from metrics import (
        EMBED_CACHE_HIT,
        EMBED_CACHE_MISS,
        QDRANT_AVAILABLE,
        QDRANT_VECTORS_TOTAL,
        SEARCH_RESULTS,
        SQLITE_AVAILABLE,
        SQLITE_SIZE_BYTES,
        TOTAL_MEMORIES,
    )

    for name in (
        TOTAL_MEMORIES,
        SQLITE_SIZE_BYTES,
        QDRANT_VECTORS_TOTAL,
        QDRANT_AVAILABLE,
        SQLITE_AVAILABLE,
        SEARCH_RESULTS,
        EMBED_CACHE_HIT,
        EMBED_CACHE_MISS,
    ):
        assert name.startswith("archivist_")


def test_embed_cache_metric_prefix():
    from metrics import EMBED_CACHE_HIT, EMBED_CACHE_MISS

    assert EMBED_CACHE_HIT == "archivist_embed_cache_hit_total"
    assert EMBED_CACHE_MISS == "archivist_embed_cache_miss_total"


def test_search_results_histogram():
    import metrics as m

    _clear_metrics()
    m.observe(m.SEARCH_RESULTS, 0.0, {"namespace": "ns1"})
    m.observe(m.SEARCH_RESULTS, 12.0, {"namespace": "ns1"})
    text = m.render()
    assert "archivist_search_results_bucket" in text
    assert "archivist_search_results_count" in text
    assert 'namespace="ns1"' in text
    assert "archivist_search_results_sum" in text


def test_metrics_auth_exempt_allows_unauthenticated_scrape(monkeypatch):
    import main

    monkeypatch.setattr(main, "_startup", _noop_startup)
    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "secret-key")
    monkeypatch.setattr(main, "METRICS_AUTH_EXEMPT", True)
    monkeypatch.setattr(main, "METRICS_ENABLED", True)
    with TestClient(main.app) as client:
        r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")


def test_metrics_requires_auth_when_not_exempt(monkeypatch):
    import main

    monkeypatch.setattr(main, "_startup", _noop_startup)
    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "secret-key")
    monkeypatch.setattr(main, "METRICS_AUTH_EXEMPT", False)
    monkeypatch.setattr(main, "METRICS_ENABLED", True)
    with TestClient(main.app) as client:
        r = client.get("/metrics")
    assert r.status_code == 401


def test_config_metrics_flags():
    from config import METRICS_AUTH_EXEMPT, METRICS_COLLECT_INTERVAL_SECONDS, METRICS_ENABLED

    assert isinstance(METRICS_ENABLED, bool)
    assert isinstance(METRICS_AUTH_EXEMPT, bool)
    assert isinstance(METRICS_COLLECT_INTERVAL_SECONDS, int)
    assert METRICS_COLLECT_INTERVAL_SECONDS >= 5
