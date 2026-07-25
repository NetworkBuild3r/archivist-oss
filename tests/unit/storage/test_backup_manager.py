"""Regression tests for the backup/restore event-loop hardening fix.

INIT-022/SPEC-002 — the review found:

  - C2 (Critical): ``_restore_sqlite()`` assumed it always ran off the event-loop
    thread, but both real call sites (``tools_admin.py``, ``main.py``) called
    ``restore_snapshot()`` directly and synchronously from ``async def``
    handlers, so ``run_coroutine_threadsafe(...).result(timeout=120)`` blocked
    the very thread needed to run the scheduled coroutine — restore always
    timed out on the default (SQLite) backend.
  - H5 (High): ``create_snapshot()``/``restore_snapshot()`` do fully synchronous,
    multi-minute I/O directly inside ``async def`` handlers with no
    ``asyncio.to_thread`` offload, freezing the whole MCP server meanwhile.
  - M15 (Medium): ``import_agent()`` called ``asyncio.run(...)`` once per NDJSON
    line for FTS upserts instead of batching into a single call.

The fix wraps both call sites in ``asyncio.to_thread(...)`` and threads the
caller's original event loop through to ``_restore_sqlite()`` so its internal
``run_coroutine_threadsafe`` handoff runs on a loop that is actually free to
service it, and batches ``import_agent()``'s FTS upserts into one
``asyncio.run(...)`` call after the main NDJSON loop.

These tests exercise the real call path (``tools_admin._handle_backup``) rather
than calling ``backup_manager`` functions directly, per ac-1/ac-4 — a unit test
against ``backup_manager`` alone would not have caught the original deadlock,
since it only manifests when the synchronous restore work runs on the same
thread as the event loop it needs to schedule work back onto.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]


def _write_manifest(snap_dir: Path, snapshot_id: str, *, sqlite_backed_up: bool) -> None:
    snap_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": 1,
        "snapshot_id": snapshot_id,
        "label": "",
        "created_at": "2026-07-23T00:00:00+00:00",
        "vector_dim": 768,
        "graph_backend": "sqlite",
        "collections": {},
        "sqlite_backed_up": sqlite_backed_up,
        "postgres_backed_up": False,
        "files_backed_up": False,
        "errors": [],
    }
    with open(snap_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)


class TestRestoreThroughHandlerPath:
    """ac-1 (C2): restore must complete (no TimeoutError) through the real
    ``tools_admin._handle_backup`` call path — not a direct ``backup_manager``
    call — when the snapshot carries a real SQLite backup file."""

    async def test_restore_with_real_sqlite_backup_completes_via_handler(
        self, tmp_path, monkeypatch
    ):
        from archivist.app.handlers import tools_admin

        dest_db = tmp_path / "dest_graph.db"
        monkeypatch.setattr("backup_manager.BACKUP_DIR", str(tmp_path))
        monkeypatch.setattr("backup_manager.SQLITE_PATH", str(dest_db))
        monkeypatch.setattr("backup_manager.VECTOR_DIM", 768)

        snapshot_id = "20260723T191500Z_handler_test"
        snap_dir = tmp_path / snapshot_id
        _write_manifest(snap_dir, snapshot_id, sqlite_backed_up=True)

        # A real SQLite backup file — this is what forces execution into
        # _restore_sqlite() (previously the deadlocking path); a snapshot with
        # no graph.db file never reaches it at all.
        src_conn = sqlite3.connect(str(snap_dir / "graph.db"))
        src_conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        src_conn.execute("INSERT INTO t VALUES (1, 'hello')")
        src_conn.commit()
        src_conn.close()

        # Bounded well under the pre-existing 120s restore timeout: if the C2
        # deadlock regresses, this fails fast with a clear TimeoutError instead
        # of hanging for two minutes.
        result = await asyncio.wait_for(
            tools_admin._handle_backup(
                {"action": "restore", "snapshot_id": snapshot_id, "target": "sqlite"}
            ),
            timeout=10,
        )

        body = json.loads(result[0].text)
        assert body["errors"] == [], body
        assert body["sqlite"] == "restored"
        assert dest_db.is_file()

        restored_conn = sqlite3.connect(str(dest_db))
        rows = restored_conn.execute("SELECT * FROM t").fetchall()
        restored_conn.close()
        assert rows == [(1, "hello")]


class TestToThreadOffload:
    """ac-2 (H5): create_snapshot()/restore_snapshot() must run via
    asyncio.to_thread() — off the event-loop thread — so the loop stays
    responsive to other work while a backup/restore is in flight."""

    async def test_create_snapshot_runs_off_the_event_loop_thread(self, monkeypatch):
        from archivist.app.handlers import tools_admin

        main_thread = threading.current_thread()
        worker_thread_holder: dict[str, threading.Thread] = {}
        started = threading.Event()
        release = threading.Event()

        def blocking_create_snapshot(label: str = "") -> dict:
            worker_thread_holder["thread"] = threading.current_thread()
            started.set()
            # Simulate the multi-minute synchronous I/O H5 describes; bounded
            # so a regression fails the test instead of hanging indefinitely.
            release.wait(timeout=5)
            return {"snapshot_id": "fake-snapshot", "label": label}

        monkeypatch.setattr("backup_manager.create_snapshot", blocking_create_snapshot)
        monkeypatch.setattr("backup_manager.prune_snapshots", list)

        other_call_completed = []

        async def other_mcp_tool_call() -> None:
            await asyncio.sleep(0.05)
            other_call_completed.append(True)

        create_task = asyncio.create_task(
            tools_admin._handle_backup({"action": "create", "label": ""})
        )
        other_task = asyncio.create_task(other_mcp_tool_call())

        # Wait (cooperatively — without blocking this test's own event loop)
        # for the offloaded call to actually start running.
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "create_snapshot never started — to_thread offload isn't happening"

        # It must be running on a different OS thread than this coroutine.
        assert worker_thread_holder["thread"] is not main_thread

        # While create_snapshot's worker thread is still blocked (release not
        # yet set), another coroutine scheduled on the same loop must still be
        # able to run to completion — proves the loop was not blocked by the
        # backup call.
        await asyncio.wait_for(other_task, timeout=2)
        assert other_call_completed == [True]

        release.set()
        result = await asyncio.wait_for(create_task, timeout=5)
        body = json.loads(result[0].text)
        assert body["snapshot_id"] == "fake-snapshot"

    async def test_restore_snapshot_runs_off_the_event_loop_thread(self, monkeypatch):
        from archivist.app.handlers import tools_admin

        main_thread = threading.current_thread()
        worker_thread_holder: dict[str, threading.Thread] = {}
        started = threading.Event()
        release = threading.Event()

        def blocking_restore_snapshot(snapshot_id, target="all", *, loop=None) -> dict:
            worker_thread_holder["thread"] = threading.current_thread()
            started.set()
            release.wait(timeout=5)
            return {"snapshot_id": snapshot_id, "errors": []}

        monkeypatch.setattr("backup_manager.restore_snapshot", blocking_restore_snapshot)

        other_call_completed = []

        async def other_mcp_tool_call() -> None:
            await asyncio.sleep(0.05)
            other_call_completed.append(True)

        restore_task = asyncio.create_task(
            tools_admin._handle_backup(
                {"action": "restore", "snapshot_id": "some-snap", "target": "all"}
            )
        )
        other_task = asyncio.create_task(other_mcp_tool_call())

        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "restore_snapshot never started — to_thread offload isn't happening"
        assert worker_thread_holder["thread"] is not main_thread

        await asyncio.wait_for(other_task, timeout=2)
        assert other_call_completed == [True]

        release.set()
        result = await asyncio.wait_for(restore_task, timeout=5)
        body = json.loads(result[0].text)
        assert body["snapshot_id"] == "some-snap"


class TestFTSUpsertBatching:
    """ac-3 (M15): import_agent()'s FTS upserts must be batched into a single
    asyncio.run(...) call after the main NDJSON loop, not once per line."""

    def test_multi_line_import_calls_asyncio_run_exactly_once(self, tmp_path):
        ndjson_path = tmp_path / "import.ndjson"
        records = [
            {"id": "p1", "vector": [0.1, 0.2], "payload": {"agent_id": "nova", "text": "one"}},
            {"id": "p2", "vector": [0.3, 0.4], "payload": {"agent_id": "nova", "text": "two"}},
            {"id": "p3", "vector": [0.5, 0.6], "payload": {"agent_id": "nova", "text": "three"}},
        ]
        with open(ndjson_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        mock_client = MagicMock()

        def _closing_run(coro):
            # Mirrors the real asyncio.run() call being made exactly once
            # while avoiding a real event loop / DB dependency and an
            # unawaited-coroutine warning.
            coro.close()
            return None

        with (
            patch("config.BM25_ENABLED", True),
            patch("collection_router.ensure_collection", return_value="test_coll"),
            patch("qdrant.qdrant_client", return_value=mock_client),
            patch("asyncio.run", side_effect=_closing_run) as mock_run,
        ):
            from backup_manager import import_agent

            result = import_agent(str(ndjson_path))

        assert mock_run.call_count == 1, (
            f"expected asyncio.run() exactly once for the whole batch, "
            f"got {mock_run.call_count} calls"
        )
        assert result["imported"] == 3
        assert result["fts_rebuilt"] == 3

    def test_empty_ndjson_skips_asyncio_run_entirely(self, tmp_path):
        """Zero-line edge case: no FTS work queued means asyncio.run() must
        not be called at all (not even once with an empty batch)."""
        ndjson_path = tmp_path / "empty.ndjson"
        ndjson_path.write_text("")

        with (
            patch("config.BM25_ENABLED", True),
            patch("asyncio.run") as mock_run,
        ):
            from backup_manager import import_agent

            result = import_agent(str(ndjson_path))

        mock_run.assert_not_called()
        assert result["imported"] == 0
        assert result["fts_rebuilt"] == 0
