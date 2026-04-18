"""v1.10 observability: request IDs, slow-path thresholds, dispatch metrics, invalidation export."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.integration]


def test_set_request_id_from_x_request_id():
    from observability import get_request_id, reset_request_id, set_request_id_from_scope

    scope = {"headers": [(b"x-request-id", b"trace-abc-99")]}
    tok = set_request_id_from_scope(scope)
    try:
        assert get_request_id() == "trace-abc-99"
    finally:
        reset_request_id(tok)


def test_set_request_id_generates_when_missing():
    from observability import get_request_id, reset_request_id, set_request_id_from_scope

    scope = {"headers": []}
    tok = set_request_id_from_scope(scope)
    try:
        rid = get_request_id()
        assert len(rid) == 12
        assert all(c in "0123456789abcdef" for c in rid)
    finally:
        reset_request_id(tok)


def test_slow_embed_warning_when_over_threshold(monkeypatch, caplog):
    import observability as obs

    monkeypatch.setattr(obs, "SLOW_EMBED_MS", 10.0)
    tok = obs._request_id.set("req-xyz")
    try:
        with caplog.at_level(logging.WARNING, logger="archivist.observability"):
            obs.slow_embed_check(50.0)
    finally:
        obs._request_id.reset(tok)

    assert "slow_path" in caplog.text
    assert "embed" in caplog.text
    assert "req-xyz" in caplog.text


def test_slow_embed_no_warning_when_disabled(monkeypatch, caplog):
    import observability as obs

    monkeypatch.setattr(obs, "SLOW_EMBED_MS", 0.0)
    with caplog.at_level(logging.WARNING, logger="archivist.observability"):
        obs.slow_embed_check(9999.0)
    assert "slow_path" not in caplog.text


async def test_dispatch_tool_records_tool_duration():
    import metrics as m
    from handlers._registry import dispatch_tool

    m._counters.clear()
    m._gauges.clear()
    m._histogram_buckets.clear()
    m._histogram_layout.clear()

    await dispatch_tool("archivist_context_check", {})
    text = m.render()
    assert "archivist_mcp_tool_duration_ms" in text


async def test_handle_invalidate_appends_export_jsonl(tmp_path, monkeypatch):
    import audit
    import main
    import memory_lifecycle

    export_path = tmp_path / "inv.jsonl"
    monkeypatch.setattr(main, "ARCHIVIST_INVALIDATION_EXPORT_PATH", str(export_path))

    class _Pt:
        def __init__(self, pid, ns=""):
            self.id = pid
            self.payload = {"namespace": ns}

    mock_client = MagicMock()
    mock_client.scroll.return_value = ([_Pt("pid-1", "ns-a"), _Pt("pid-2", "ns-b")], None)
    mock_client.delete = MagicMock()
    monkeypatch.setattr(main, "qdrant_client", lambda **kwargs: mock_client)
    monkeypatch.setattr(audit, "log_memory_event", AsyncMock())
    # TTL invalidation calls full cascade delete — mock it so CI does not need Qdrant.
    monkeypatch.setattr(memory_lifecycle, "delete_memory_complete", AsyncMock())

    resp = await main.handle_invalidate(None)
    assert resp.status_code == 200
    body = json.loads(resp.body.decode())
    assert body["invalidated"] == 2

    lines = export_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["count"] == 2
    assert row["reason"] == "ttl_expired"
    assert "pid-1" in row["sample_ids"]
    assert "ns-a" in row["sample_namespaces"]
