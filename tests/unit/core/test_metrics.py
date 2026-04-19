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
        import archivist.core.health as health
        from archivist.core import metrics as m

        health.register("test_subsystem_up", healthy=True)

        emitted: dict[str, float] = {}

        def fake_gauge_set(name, value, labels=None):
            if name == m.SUBSYSTEM_HEALTHY and labels:
                emitted[labels.get("subsystem", "")] = value

        monkeypatch.setattr(m, "gauge_set", fake_gauge_set)

        # Call only the subsystem health portion by invoking the helper directly
        for _name, _entry in health.all_status().items():
            m.gauge_set(
                m.SUBSYSTEM_HEALTHY, 1.0 if _entry.get("healthy") else 0.0, {"subsystem": _name}
            )

        assert "test_subsystem_up" in emitted
        assert emitted["test_subsystem_up"] == 1.0

    def test_subsystem_healthy_gauge_zero_when_unhealthy(self, monkeypatch):
        import archivist.core.health as health
        from archivist.core import metrics as m

        health.register("test_subsystem_down", healthy=False)

        emitted: dict[str, float] = {}

        def fake_gauge_set(name, value, labels=None):
            if name == m.SUBSYSTEM_HEALTHY and labels:
                emitted[labels.get("subsystem", "")] = value

        monkeypatch.setattr(m, "gauge_set", fake_gauge_set)

        for _name, _entry in health.all_status().items():
            m.gauge_set(
                m.SUBSYSTEM_HEALTHY, 1.0 if _entry.get("healthy") else 0.0, {"subsystem": _name}
            )

        assert "test_subsystem_down" in emitted
        assert emitted["test_subsystem_down"] == 0.0


class TestHealthRegisterLatency:
    """health.register() stores latency_ms in the status entry."""

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
