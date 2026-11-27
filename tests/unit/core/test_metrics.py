"""Unit tests for Phase 5 observability: new metric constants and gauge-tick logic."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]


class TestNewMetricConstants:
    """All 12 new metric name constants are defined, non-empty, and distinct."""

    def test_pg_pool_constants_exist(self):
        from archivist.core import metrics as m

        assert m.PG_POOL_ACQUIRE_MS
        assert m.PG_POOL_QUERY_MS
        assert m.PG_POOL_ERRORS_TOTAL
        assert m.PG_POOL_SIZE

    def test_fts_constants_exist(self):
        from archivist.core import metrics as m

        assert m.FTS_SEARCH_DURATION_MS
        assert m.FTS_SEARCH_TOTAL
        assert m.FTS_UPSERT_TOTAL
        assert m.FTS_UPSERT_ERRORS_TOTAL

    def test_write_and_curator_constants_exist(self):
        from archivist.core import metrics as m

        assert m.INDEX_DURATION_MS
        assert m.CURATOR_EXTRACT_MS
        assert m.CURATOR_DECAY_MS
        assert m.SUBSYSTEM_HEALTHY

    def test_all_12_new_constants_are_distinct(self):
        from archivist.core import metrics as m

        new_constants = [
            m.PG_POOL_ACQUIRE_MS,
            m.PG_POOL_QUERY_MS,
            m.PG_POOL_ERRORS_TOTAL,
            m.PG_POOL_SIZE,
            m.FTS_SEARCH_DURATION_MS,
            m.FTS_SEARCH_TOTAL,
            m.FTS_UPSERT_TOTAL,
            m.FTS_UPSERT_ERRORS_TOTAL,
            m.INDEX_DURATION_MS,
            m.CURATOR_EXTRACT_MS,
            m.CURATOR_DECAY_MS,
            m.SUBSYSTEM_HEALTHY,
        ]
        assert len(new_constants) == len(set(new_constants)), "metric constant names must be unique"

    def test_new_constants_use_archivist_prefix(self):
        from archivist.core import metrics as m

        for name in [
            m.PG_POOL_ACQUIRE_MS,
            m.FTS_SEARCH_DURATION_MS,
            m.SUBSYSTEM_HEALTHY,
        ]:
            assert name.startswith("archivist_"), f"{name!r} must start with 'archivist_'"

    def test_constants_are_strings(self):
        from archivist.core import metrics as m

        for name in [m.PG_POOL_SIZE, m.FTS_UPSERT_ERRORS_TOTAL, m.CURATOR_DECAY_MS]:
            assert isinstance(name, str), f"{name!r} must be str"


class TestSubsystemHealthyGauge:
    """collect_storage_gauges_tick emits SUBSYSTEM_HEALTHY for registered subsystems."""

    def test_subsystem_healthy_gauge_emitted_when_healthy(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        import archivist.core.health as health
        from archivist.core import metrics as m

        health.register("test_subsystem_up", healthy=True)
        health.register("test_subsystem_down", healthy=False)

        emitted: dict[str, float] = {}

        def fake_gauge_set(name, value, labels=None):
            if name == m.SUBSYSTEM_HEALTHY and labels:
                emitted[labels.get("subsystem", "")] = value

        monkeypatch.setattr(m, "gauge_set", fake_gauge_set)
        monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "postgres")

        fake_client = MagicMock()
        fake_client.get_collections.return_value = MagicMock(collections=[])

        with patch("archivist.storage.qdrant.qdrant_client", return_value=fake_client):
            m.collect_storage_gauges_tick()

        assert emitted.get("test_subsystem_up") == 1.0
        assert emitted.get("test_subsystem_down") == 0.0


class TestPrometheusHistogramFormat:
    """MAJ-2: histogram buckets must not double-accumulate and _count/_sum stay under histogram TYPE."""

    def test_single_observation_cumulative_buckets_and_count(self):
        from archivist.core import metrics as m

        m._counters.clear()
        m._gauges.clear()
        m._histogram_buckets.clear()
        m._histogram_layout.clear()
        m._histogram_counts.clear()
        m._histogram_sums.clear()

        m.observe("archivist_test_lat_ms", 12.0)
        text = m.render()

        assert "# TYPE archivist_test_lat_ms histogram" in text
        assert "# TYPE archivist_test_lat_ms_count counter" not in text
        assert "# TYPE archivist_test_lat_ms_sum counter" not in text
        assert 'archivist_test_lat_ms_bucket{le="10"} 0' in text
        assert 'archivist_test_lat_ms_bucket{le="25"} 1' in text
        assert 'archivist_test_lat_ms_bucket{le="50"} 1' in text
        assert 'archivist_test_lat_ms_bucket{le="+Inf"} 1' in text
        assert "archivist_test_lat_ms_count 1" in text
        assert "archivist_test_lat_ms_sum 12.0" in text

    def test_labeled_histogram_multiple_observations(self):
        from archivist.core import metrics as m

        m._counters.clear()
        m._gauges.clear()
        m._histogram_buckets.clear()
        m._histogram_layout.clear()
        m._histogram_counts.clear()
        m._histogram_sums.clear()

        m.observe(m.SEARCH_RESULTS, 0.0, {"namespace": "alpha"})
        m.observe(m.SEARCH_RESULTS, 12.0, {"namespace": "alpha"})
        m.observe(m.SEARCH_RESULTS, 1000.0, {"namespace": "alpha"})
        text = m.render()

        assert "# TYPE archivist_search_results histogram" in text
        assert 'archivist_search_results_bucket{namespace="alpha",le="0"} 1' in text
        assert 'archivist_search_results_bucket{namespace="alpha",le="10"} 1' in text
        assert 'archivist_search_results_bucket{namespace="alpha",le="20"} 2' in text
        assert 'archivist_search_results_bucket{namespace="alpha",le="+Inf"} 3' in text
        assert 'archivist_search_results_count{namespace="alpha"} 3' in text
        assert 'archivist_search_results_sum{namespace="alpha"} 1012.0' in text


class TestHealthRegisterLatency:
    """archivist.core.health.register() stores latency_ms in the status entry."""

    def test_latency_ms_stored(self):
        import archivist.core.health as health

        health.register("test_latency_subsystem", healthy=True, latency_ms=42.5)
        status = health.all_status()
        assert "test_latency_subsystem" in status
        assert status["test_latency_subsystem"]["latency_ms"] == 42.5

    def test_latency_ms_default_zero(self):
        import archivist.core.health as health

        health.register("test_latency_default")
        status = health.all_status()
        assert status["test_latency_default"]["latency_ms"] == 0.0

    def test_latency_ms_backward_compat_no_kwarg(self):
        import archivist.core.health as health

        # Old callers that don't pass latency_ms must still work
        health.register("test_compat_subsystem", healthy=True, detail="ok")
        status = health.all_status()
        assert status["test_compat_subsystem"]["healthy"] is True
        assert status["test_compat_subsystem"]["latency_ms"] == 0.0
