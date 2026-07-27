"""INIT-011/SPEC-004 — agentic MaP MCP profile / RBAC / path / round-trip.

Diff #4 product bar: ops can map; core cannot; import RBAC + path escape
fail closed; snapshot→export→import via MCP handlers restores markers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.system,
    pytest.mark.mcp,
    pytest.mark.agentic_memory,
]


def _map_backup_patches(monkeypatch, backup: Path) -> None:
    from archivist.storage.backup_manager import _snapshot_dir, delete_snapshot

    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(backup))
    monkeypatch.setattr("archivist.storage.memory_product._snapshot_dir", _snapshot_dir)
    monkeypatch.setattr("archivist.storage.memory_product.delete_snapshot", delete_snapshot)


async def _seed(async_pool, *, namespace: str, agent_id: str, text: str, qid: str) -> None:
    async with async_pool.write() as conn:
        await conn.execute(
            """
            INSERT INTO memory_chunks (
                qdrant_id, text, file_path, chunk_index, agent_id, namespace,
                date, memory_type, is_excluded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (qid, text, "map.md", 0, agent_id, namespace, "", "general"),
        )


class TestMapProfileAndRbac:
    def test_core_excludes_map_ops_includes(self, monkeypatch):
        """ac-3: core profile excludes archivist_map_*; ops includes them."""
        import archivist.core.config as config
        from archivist.app.handlers._registry import allowed_tool_names, get_all_tools

        monkeypatch.setattr(config, "TOOL_PROFILE", "core")
        core_names = {t.name for t in get_all_tools()}
        assert not any(n.startswith("archivist_map_") for n in core_names)
        assert not any(n.startswith("archivist_map_") for n in allowed_tool_names("core"))

        monkeypatch.setattr(config, "TOOL_PROFILE", "ops")
        ops_names = {t.name for t in get_all_tools()}
        assert "archivist_map_list" in ops_names
        assert "archivist_map_import" in ops_names
        assert "archivist_map_snapshot" in allowed_tool_names("ops")

    @pytest.mark.asyncio
    async def test_mcp_import_rbac_deny(self, async_pool, tmp_path, monkeypatch, rbac_config):
        """ac-4: write/import denied at MCP boundary when caller lacks write."""
        from archivist.app.handlers.tools_memory_product import (
            _handle_map_import,
            _handle_map_snapshot,
        )

        backup = tmp_path / "backups"
        backup.mkdir()
        _map_backup_patches(monkeypatch, backup)
        monkeypatch.setattr("archivist.core.rbac.is_permissive_mode", lambda: False)

        await _seed(
            async_pool,
            namespace="shared",
            agent_id="chief",
            text="rbac seed",
            qid="map-rbac-1",
        )
        snap = await _handle_map_snapshot(
            {"namespace": "shared", "caller_agent_id": "chief", "agent_id": "chief"}
        )
        snap_data = json.loads(snap[0].text)
        archive_id = snap_data["version"]["archive_id"]

        denied = await _handle_map_import(
            {
                "archive_id": archive_id,
                "target_namespace": "deployer",
                "caller_agent_id": "gitbob",
                "target_agent_id": "gitbob",
            }
        )
        data = json.loads(denied[0].text)
        assert data["error"] == "access_denied"
        assert "Traceback" not in denied[0].text

    @pytest.mark.asyncio
    async def test_mcp_import_path_escape(self, async_pool, tmp_path, monkeypatch, rbac_config):
        """ac-6: path-escape archive_id fails closed at MCP boundary."""
        from archivist.app.handlers.tools_memory_product import _handle_map_import

        backup = tmp_path / "backups"
        backup.mkdir()
        _map_backup_patches(monkeypatch, backup)
        monkeypatch.setattr("archivist.core.rbac.is_permissive_mode", lambda: False)

        result = await _handle_map_import(
            {
                "archive_id": "../../etc/passwd",
                "target_namespace": "pipeline",
                "caller_agent_id": "gitbob",
                "target_agent_id": "gitbob",
            }
        )
        data = json.loads(result[0].text)
        assert data["error"] == "invalid_archive_id"
        assert "Traceback" not in result[0].text


class TestMapMcpRoundTrip:
    @pytest.mark.asyncio
    async def test_mcp_snapshot_export_import_round_trip(
        self, async_pool, tmp_path, monkeypatch, rbac_config
    ):
        """ac-1: MCP handlers snapshot → export → import restore marker text."""
        from archivist.app.handlers.tools_memory_product import (
            _handle_map_export,
            _handle_map_fork,
            _handle_map_import,
            _handle_map_list,
            _handle_map_snapshot,
        )

        backup = tmp_path / "backups"
        backup.mkdir()
        _map_backup_patches(monkeypatch, backup)
        monkeypatch.setattr("archivist.core.rbac.is_permissive_mode", lambda: False)

        marker = "MAP_MCP_ROUNDTRIP_INIT011"
        await _seed(
            async_pool,
            namespace="shared",
            agent_id="chief",
            text=f"usable {marker}",
            qid="map-mcp-1",
        )

        snap = await _handle_map_snapshot(
            {
                "namespace": "shared",
                "caller_agent_id": "chief",
                "agent_id": "chief",
                "label": "mcp-snap",
            }
        )
        snap_data = json.loads(snap[0].text)
        version_id = snap_data["version"]["id"]
        archive_id = snap_data["version"]["archive_id"]

        exported = await _handle_map_export(
            {
                "namespace": "shared",
                "caller_agent_id": "chief",
                "agent_id": "chief",
                "version_id": version_id,
            }
        )
        export_data = json.loads(exported[0].text)
        assert export_data["archive_id"] == archive_id

        imported = await _handle_map_import(
            {
                "archive_id": archive_id,
                "target_namespace": "pipeline",
                "caller_agent_id": "gitbob",
                "target_agent_id": "gitbob-mcp",
                "label": "mcp-import",
            }
        )
        import_data = json.loads(imported[0].text)
        assert import_data["version"]["operation"] == "import"
        assert import_data["version"]["chunk_count"] == 1

        listed = await _handle_map_list(
            {
                "namespace": "pipeline",
                "caller_agent_id": "gitbob",
                "agent_id": "gitbob-mcp",
            }
        )
        list_data = json.loads(listed[0].text)
        assert list_data["count"] >= 1
        assert any(v["operation"] == "import" for v in list_data["versions"])

        async with async_pool.read() as conn:
            cur = await conn.execute(
                "SELECT text FROM memory_chunks WHERE namespace = ? AND agent_id = ?",
                ("pipeline", "gitbob-mcp"),
            )
            rows = [dict(r) for r in await cur.fetchall()]
        assert len(rows) == 1
        assert marker in rows[0]["text"]

        # ac-2: fork still works via MCP alongside import.
        forked = await _handle_map_fork(
            {
                "source_version_id": version_id,
                "target_namespace": "pipeline",
                "caller_agent_id": "gitbob",
                "target_agent_id": "gitbob-fork-mcp",
                "label": "mcp-fork",
            }
        )
        fork_data = json.loads(forked[0].text)
        assert fork_data["version"]["operation"] == "fork"
        assert fork_data["version"]["chunk_count"] == 1
