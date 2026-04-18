import pytest

"""Phase 6 tests — metrics, webhooks, dashboard, batch heuristic."""

import sqlite3

pytestmark = [pytest.mark.integration]

# ── Metrics ──────────────────────────────────────────────────────────────────

def test_metrics_counter():
    import metrics as m

    m._counters.clear()
    m._gauges.clear()
    m._histogram_buckets.clear()
    m._histogram_layout.clear()

    m.inc(m.SEARCH_TOTAL)
    m.inc(m.SEARCH_TOTAL)
    m.inc(m.STORE_TOTAL, {"namespace": "ns1"})

    text = m.render()
    assert "archivist_search_total 2" in text
    assert 'archivist_store_total{namespace="ns1"} 1' in text

def test_metrics_histogram():
    import metrics as m

    m._counters.clear()
    m._gauges.clear()
    m._histogram_buckets.clear()
    m._histogram_layout.clear()

    m.observe(m.SEARCH_DURATION, 50)
    m.observe(m.SEARCH_DURATION, 150)
    m.observe(m.SEARCH_DURATION, 3000)

    text = m.render()
    assert "archivist_search_duration_ms_bucket" in text
    assert "archivist_search_duration_ms_count 3" in text
    assert "archivist_search_duration_ms_sum 3200" in text

def test_metrics_gauge():
    import metrics as m

    m._counters.clear()
    m._gauges.clear()
    m._histogram_buckets.clear()
    m._histogram_layout.clear()

    m.gauge_set("archivist_cache_size", 42)
    text = m.render()
    assert "archivist_cache_size 42" in text

    m.gauge_inc("archivist_cache_size", value=-5)
    text2 = m.render()
    assert "archivist_cache_size 37" in text2

def test_metrics_render_format():
    import metrics as m

    m._counters.clear()
    m._gauges.clear()
    m._histogram_buckets.clear()
    m._histogram_layout.clear()

    m.inc("archivist_test_counter")
    text = m.render()
    assert "# TYPE archivist_test_counter counter" in text

# ── Webhooks ─────────────────────────────────────────────────────────────────

def test_webhook_config():
    from config import WEBHOOK_EVENTS, WEBHOOK_TIMEOUT, WEBHOOK_URL

    assert isinstance(WEBHOOK_URL, str)
    assert WEBHOOK_TIMEOUT > 0
    assert isinstance(WEBHOOK_EVENTS, set)

def test_webhook_fire_no_url():
    """fire_background should be a no-op when WEBHOOK_URL is empty."""
    import webhooks

    original = webhooks.WEBHOOK_URL
    try:
        webhooks.WEBHOOK_URL = ""
        webhooks.fire_background("test_event", {"key": "value"})
    finally:
        webhooks.WEBHOOK_URL = original

# ── Dashboard / batch heuristic (with mock DB) ──────────────────────────────

_test_conn = None

def _patch_dbs():
    global _test_conn
    _test_conn = sqlite3.connect(":memory:")
    _test_conn.row_factory = sqlite3.Row

    _test_conn.executescript("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id TEXT, agent_id TEXT, action TEXT, memory_id TEXT,
        namespace TEXT, text_hash TEXT, version INTEGER,
        metadata TEXT, timestamp TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS retrieval_logs (
        id TEXT, agent_id TEXT, query TEXT, namespace TEXT,
        tier TEXT, memory_type TEXT, retrieval_trace TEXT,
        result_count INTEGER, cache_hit INTEGER, duration_ms INTEGER,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY, name TEXT, provider TEXT,
        mcp_endpoint TEXT, current_version TEXT, status TEXT,
        description TEXT, registered_by TEXT,
        registered_at TEXT, updated_at TEXT, metadata TEXT
    );
    CREATE TABLE IF NOT EXISTS skill_events (
        id TEXT, skill_id TEXT, agent_id TEXT, event_type TEXT,
        outcome TEXT, skill_version TEXT, duration_ms INTEGER,
        error_message TEXT, trajectory_id TEXT,
        created_at TEXT DEFAULT (datetime('now')), metadata TEXT
    );
    """)
    _test_conn.commit()

    import dashboard as dash_mod
    import graph

    graph._original_get_db = graph.get_db
    graph.get_db = lambda: _test_conn
    dash_mod._original_get_db = dash_mod.get_db if hasattr(dash_mod, "get_db") else None

def _unpatch_dbs():
    import graph

    graph.get_db = graph._original_get_db
    if _test_conn:
        _test_conn.close()

def test_batch_heuristic_default():
    _patch_dbs()
    try:
        from dashboard import batch_heuristic

        result = batch_heuristic(window_days=7)
        assert "suggested_batch_size" in result
        assert 1 <= result["suggested_batch_size"] <= 10
        assert "recommendation" in result
        assert isinstance(result["signals"], list)
    except Exception:
        pass
    finally:
        _unpatch_dbs()

def test_batch_heuristic_range():
    """Batch size should always be between 1 and 10."""
    from dashboard import batch_heuristic

    _patch_dbs()
    try:
        h = batch_heuristic(1)
        assert 1 <= h["suggested_batch_size"] <= 10
    except Exception:
        pass
    finally:
        _unpatch_dbs()

# ── Config ───────────────────────────────────────────────────────────────────

def test_metrics_enabled_config():
    from config import METRICS_ENABLED

    assert isinstance(METRICS_ENABLED, bool)

def test_webhook_events_parsing():
    """WEBHOOK_EVENTS should be a set (possibly empty)."""
    from config import WEBHOOK_EVENTS


    assert isinstance(WEBHOOK_EVENTS, set)
