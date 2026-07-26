"""INIT-004/SPEC-001 — compressed index rebuild timing hooks."""

from __future__ import annotations

import logging

import pytest

pytestmark = [pytest.mark.unit]


class TestCompressedIndexRebuildTiming:
    async def test_empty_rebuild_emits_log_and_metric(self, async_pool, monkeypatch, caplog):
        from archivist.core import metrics as m
        from archivist.storage.compressed_index import build_namespace_index

        monkeypatch.setattr("archivist.core.config.METRICS_ENABLED", True)
        observed: list[tuple[str, float]] = []

        def _capture(name: str, value: float, labels: dict | None = None):
            observed.append((name, value))

        monkeypatch.setattr(m, "observe", _capture)

        with caplog.at_level(logging.INFO, logger="archivist.compressed_index"):
            text = await build_namespace_index("timing-empty-ns-004")

        assert "No indexed knowledge yet" in text
        assert any(n == m.INDEX_DURATION_MS and v >= 0 for n, v in observed), observed
        rebuild_logs = [
            r for r in caplog.records if "compressed_index.rebuild_complete" in r.message
        ]
        assert rebuild_logs, "expected compressed_index.rebuild_complete log"
        rec = rebuild_logs[-1]
        assert hasattr(rec, "rebuild_ms")
        assert rec.rebuild_ms >= 0
        assert rec.namespace == "timing-empty-ns-004"

    def test_rebuild_hook_present_in_source(self):
        import inspect

        from archivist.storage import compressed_index as ci

        # Hooks live on the payload builder; build_namespace_index is a markdown wrapper.
        source = inspect.getsource(ci.build_namespace_index_payload)
        assert "compressed_index.rebuild_complete" in source
        assert "rebuild_ms" in source
        assert "INDEX_DURATION_MS" in source
