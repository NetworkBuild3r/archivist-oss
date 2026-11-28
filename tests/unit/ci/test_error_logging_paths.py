"""Unit tests verifying hardened error-logging paths (formerly S110/S112) (INIT-025/SPEC-012)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit]


def test_hot_cache_invalidation_logs_inventory_errors(caplog) -> None:
    import archivist.retrieval.hot_cache as hot_cache

    with caplog.at_level(logging.DEBUG, logger="archivist.cache"):
        with (
            patch(
                "archivist.storage.namespace_inventory.invalidate",
                side_effect=RuntimeError("inv boom"),
            ),
            patch(
                "archivist.storage.namespace_inventory.invalidate_all",
                side_effect=RuntimeError("inv_all boom"),
            ),
        ):
            assert hot_cache.invalidate_namespace("ns-err") == 0
            assert hot_cache.invalidate_all() == 0

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "inv boom" in messages
    assert "inv_all boom" in messages


async def test_namespace_inventory_invalidation_and_tips_errors_logged(caplog) -> None:
    import archivist.storage.namespace_inventory as inv

    with caplog.at_level(logging.DEBUG, logger="archivist.inventory"):
        with (
            patch(
                "archivist.retrieval.query_classifier.invalidate_cache",
                side_effect=RuntimeError("qc ns fail"),
            ),
            patch(
                "archivist.retrieval.query_classifier.invalidate_all_cache",
                side_effect=RuntimeError("qc all fail"),
            ),
        ):
            inv.invalidate("ns-err")
            inv.invalidate_all()

        fake_conn = MagicMock()
        fake_conn.fetchall = AsyncMock(return_value=[])
        fake_conn.fetchone = AsyncMock(side_effect=RuntimeError("tips missing"))

        class _FakePool:
            def read(self):
                return self

            async def __aenter__(self):
                return fake_conn

            async def __aexit__(self, *args):
                return False

        with patch("archivist.storage.sqlite_pool.pool", _FakePool()):
            result = await inv._fetch_inventory("ns-err")

    assert result.has_fleet_tips is False
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "qc ns fail" in messages
    assert "qc all fail" in messages
    assert "tips missing" in messages


def test_metrics_pg_pool_size_error_logged(caplog, monkeypatch) -> None:
    import archivist.core.config as config
    import archivist.core.metrics as metrics

    monkeypatch.setattr(config, "GRAPH_BACKEND", "postgres", raising=False)
    monkeypatch.setattr(metrics, "_metrics_enabled", lambda: True)

    broken_pg_pool = MagicMock()
    broken_pg_pool.get_size.side_effect = RuntimeError("pg size boom")
    fake_backend = SimpleNamespace(_pool=broken_pg_pool)

    with caplog.at_level(logging.DEBUG, logger="archivist.metrics"):
        with (
            patch("archivist.storage.sqlite_pool.pool", fake_backend),
            patch("archivist.storage.qdrant.qdrant_client", side_effect=RuntimeError("no qdrant")),
        ):
            metrics.collect_storage_gauges_tick()

    assert any("pg size boom" in r.getMessage() for r in caplog.records)


async def test_lineage_qdrant_retrieve_errors_logged_and_continue(caplog) -> None:
    import archivist.app.lineage as lineage

    fake_conn = MagicMock()
    fake_conn.fetchone = AsyncMock(return_value=None)

    class _FakePool:
        def read(self):
            return self

        async def __aenter__(self):
            return fake_conn

        async def __aexit__(self, *args):
            return False

    fake_client = MagicMock()
    fake_client.retrieve.side_effect = RuntimeError("qdrant unavailable")

    with caplog.at_level(logging.DEBUG, logger="archivist.lineage"):
        with (
            patch("archivist.storage.sqlite_pool.pool", _FakePool()),
            patch("archivist.storage.qdrant.qdrant_client", return_value=fake_client),
            patch(
                "archivist.storage.collection_router.collections_for_query",
                return_value=["col_a", "col_b"],
            ),
        ):
            ns = await lineage.resolve_memory_namespace("mem-missing")
            edges = await lineage._provenance_edges("mem-missing")

    assert ns == ""
    assert edges == []
    assert any("qdrant unavailable" in r.getMessage() for r in caplog.records)


def test_rlm_literal_search_logs_scroll_errors_and_continues(caplog) -> None:
    import archivist.retrieval.rlm_retriever as rlm

    fake_client = MagicMock()
    fake_client.scroll.side_effect = RuntimeError("scroll timeout")

    with caplog.at_level(logging.DEBUG, logger="archivist.rlm"):
        with (
            patch("archivist.retrieval.rlm_retriever.qdrant_client", return_value=fake_client),
            patch("archivist.retrieval.rlm_retriever.collection_for", return_value="col_test"),
        ):
            hits = rlm._literal_search_sync(["10.0.0.1"], namespace="ns1")

    assert hits == []
    assert any("scroll timeout" in r.getMessage() for r in caplog.records)


def test_backup_qdrant_collection_logs_temp_delete_error(caplog, tmp_path) -> None:
    import archivist.storage.backup_manager as bm

    post_resp = MagicMock()
    post_resp.json.return_value = {"result": {"name": "snap-1.snapshot"}}
    get_resp = MagicMock()
    get_resp.content = b"snapshot-bytes"

    def _fake_http(method: str, url: str):
        if method == "post":
            return post_resp
        if method == "get":
            return get_resp
        if method == "delete":
            raise RuntimeError("delete temp failed")
        raise AssertionError(f"unexpected {method}")

    with caplog.at_level(logging.DEBUG, logger="archivist.backup"):
        with (
            patch.object(bm, "_collection_point_count", return_value=3),
            patch.object(bm, "_qdrant_http", side_effect=_fake_http),
        ):
            info = bm._backup_qdrant_collection("test_col", tmp_path)

    assert info["points"] == 3
    assert info["size_bytes"] == len(b"snapshot-bytes")
    assert any("delete temp failed" in r.getMessage() for r in caplog.records)


async def test_tools_storage_compress_logs_malformed_store_json(caplog) -> None:
    from mcp.types import TextContent

    import archivist.app.handlers.tools_storage as ts

    fake_point = SimpleNamespace(id="m1", payload={"text": "hello", "agent_id": "a1"})
    fake_client = MagicMock()
    fake_client.retrieve.return_value = [fake_point]

    with caplog.at_level(logging.WARNING, logger="archivist.handlers"):
        with (
            patch.object(ts, "require_rbac", return_value=None),
            patch.object(ts, "get_namespace_for_agent", return_value="ns1"),
            patch.object(ts, "qdrant_client", return_value=fake_client),
            patch.object(ts, "collections_for_query", return_value=["col1"]),
            patch.object(ts, "_check_point_write_rbac", return_value=None),
            patch.object(
                ts,
                "_handle_store",
                new_callable=AsyncMock,
                return_value=[TextContent(type="text", text="not-valid-json")],
            ),
        ):
            resp = await ts._handle_compress(
                {
                    "memory_ids": ["m1"],
                    "agent_id": "a1",
                    "namespace": "ns1",
                    "summary": "compressed summary text",
                }
            )

    assert "Failed to store compressed summary" in resp[0].text
    assert any("compress_memories" in r.getMessage() for r in caplog.records)


async def test_tools_trajectory_session_end_logs_malformed_store_json(caplog) -> None:
    from mcp.types import TextContent

    import archivist.app.handlers.tools_trajectory as tt

    fake_ss = MagicMock()
    fake_ss.flush.return_value = [{"key": "k1", "value": "v1", "promoted": True}]

    with caplog.at_level(logging.WARNING, logger="archivist.handlers"):
        with (
            patch.object(
                tt,
                "session_end_summary",
                new_callable=AsyncMock,
                return_value={"summary": "Great session"},
            ),
            patch.object(
                tt,
                "_handle_store",
                new_callable=AsyncMock,
                return_value=[TextContent(type="text", text="{invalid")],
            ),
            patch(
                "archivist.retrieval.session_store.get_session_store",
                return_value=fake_ss,
            ),
        ):
            resp = await tt._handle_session_end(
                {"agent_id": "a1", "session_id": "s1", "store_as_memory": True}
            )

    assert "stored_memory_id" in resp[0].text
    msgs = [r.getMessage() for r in caplog.records]
    assert any("failed to parse summary store response" in m for m in msgs)
    assert any("failed to parse ephemeral store response" in m for m in msgs)


async def test_compressed_index_wake_up_no_agent_filter_and_tips_error(caplog) -> None:
    import archivist.storage.compressed_index as ci

    fake_conn = MagicMock()

    async def _fake_fetchall(sql: str, params=None):
        if "FROM tips" in sql:
            raise RuntimeError("tips table missing")
        return []

    fake_conn.fetchall = AsyncMock(side_effect=_fake_fetchall)
    fake_conn.fetchone = AsyncMock(return_value={"c": 0, "ls": ""})

    class _FakePool:
        def read(self):
            return self

        async def __aenter__(self):
            return fake_conn

        async def __aexit__(self, *args):
            return False

    with caplog.at_level(logging.DEBUG, logger="archivist.compressed_index"):
        with (
            patch("archivist.storage.sqlite_pool.pool", _FakePool()),
            patch.object(ci, "build_namespace_index", new_callable=AsyncMock, return_value="toc"),
        ):
            payload = await ci.build_wake_up_context(namespace="ns-empty", agent_id="")

    assert payload["total_memories"] == 0
    assert any("tips table missing" in r.getMessage() for r in caplog.records)
