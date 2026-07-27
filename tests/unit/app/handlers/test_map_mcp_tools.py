"""INIT-011/SPEC-003 — archivist_map_* MCP tools (ops/full; hidden on core)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.unit]

_MAP_TOOLS = {
    "archivist_map_list",
    "archivist_map_get",
    "archivist_map_snapshot",
    "archivist_map_fork",
    "archivist_map_export",
    "archivist_map_import",
}


@dataclass
class _FakeRecord:
    id: str = "ver-1"
    source_namespace: str = "pipeline"
    source_agent_id: str = "gitbob"
    version: int = 1
    label: str = "t"
    parent_version_id: str | None = None
    chunk_count: int = 2
    point_count: int = 0
    archive_id: str = "memprod_test"
    operation: str = "snapshot"
    created_by: str = "gitbob"
    created_at: str = "2026-07-27T00:00:00Z"
    lineage: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.lineage is None:
            self.lineage = {}


class TestMapProfileGating:
    def test_core_excludes_map_tools(self, monkeypatch):
        import archivist.core.config as config
        from archivist.app.handlers._registry import CORE_TOOL_NAMES, get_all_tools

        monkeypatch.setattr(config, "TOOL_PROFILE", "core")
        names = {t.name for t in get_all_tools()}
        assert names.isdisjoint(_MAP_TOOLS)
        assert not any(n.startswith("archivist_map_") for n in CORE_TOOL_NAMES)
        assert len(CORE_TOOL_NAMES) <= 12

    def test_ops_includes_map_excludes_checkpoint(self, monkeypatch):
        import archivist.core.config as config
        from archivist.app.handlers._registry import get_all_tools

        monkeypatch.setattr(config, "TOOL_PROFILE", "ops")
        names = {t.name for t in get_all_tools()}
        assert names >= _MAP_TOOLS
        assert not any(n.startswith("archivist_checkpoint_") for n in names)

    def test_full_includes_map(self, monkeypatch):
        import archivist.core.config as config
        from archivist.app.handlers._registry import ALL_TOOLS, get_all_tools

        monkeypatch.setattr(config, "TOOL_PROFILE", "full")
        names = {t.name for t in get_all_tools()}
        assert names >= _MAP_TOOLS
        assert names == {t.name for t in ALL_TOOLS}

    @pytest.mark.asyncio
    async def test_dispatch_map_blocked_under_core(self, monkeypatch):
        import archivist.core.config as config
        from archivist.app.handlers._registry import TOOL_REGISTRY, dispatch_tool

        monkeypatch.setattr(config, "TOOL_PROFILE", "core")
        assert "archivist_map_list" in TOOL_REGISTRY
        result = await dispatch_tool("archivist_map_list", {})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "not available" in data["error"]

    @pytest.mark.asyncio
    async def test_dispatch_map_reaches_handler_under_ops(self, monkeypatch):
        import archivist.core.config as config
        from archivist.app.handlers._registry import dispatch_tool

        monkeypatch.setattr(config, "TOOL_PROFILE", "ops")
        result = await dispatch_tool(
            "archivist_map_list",
            {"namespace": "pipeline", "caller_agent_id": ""},
        )
        data = json.loads(result[0].text)
        assert "error" in data
        assert "not available" not in data["error"]


class TestMapHandlers:
    @pytest.mark.asyncio
    async def test_map_list_read_happy(self, monkeypatch):
        from archivist.app.handlers import tools_memory_product as tmp

        monkeypatch.setattr(
            "archivist.core.rbac.is_permissive_mode",
            lambda: False,
        )
        with patch(
            "archivist.app.handlers.tools_memory_product.list_scope_versions",
            new=AsyncMock(return_value=[_FakeRecord()]),
        ) as mock_list:
            result = await tmp._handle_map_list(
                {"namespace": "pipeline", "caller_agent_id": "gitbob"}
            )
        data = json.loads(result[0].text)
        assert data["count"] == 1
        assert data["versions"][0]["id"] == "ver-1"
        mock_list.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_map_snapshot_mutate_happy(self, monkeypatch):
        from archivist.app.handlers import tools_memory_product as tmp

        monkeypatch.setattr(
            "archivist.core.rbac.is_permissive_mode",
            lambda: False,
        )
        with patch(
            "archivist.app.handlers.tools_memory_product.create_scope_snapshot",
            new=AsyncMock(return_value=_FakeRecord(operation="snapshot")),
        ) as mock_snap:
            result = await tmp._handle_map_snapshot(
                {
                    "namespace": "pipeline",
                    "caller_agent_id": "gitbob",
                    "label": "v1",
                }
            )
        data = json.loads(result[0].text)
        assert data["version"]["operation"] == "snapshot"
        assert data["version"]["archive_id"] == "memprod_test"
        mock_snap.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_map_import_maps_authz_error(self, monkeypatch):
        from archivist.app.handlers import tools_memory_product as tmp
        from archivist.storage.memory_product import MemoryProductAuthzError

        monkeypatch.setattr(
            "archivist.core.rbac.is_permissive_mode",
            lambda: False,
        )
        with patch(
            "archivist.app.handlers.tools_memory_product.import_scope",
            new=AsyncMock(side_effect=MemoryProductAuthzError("no write")),
        ):
            result = await tmp._handle_map_import(
                {
                    "archive_id": "memprod_x",
                    "target_namespace": "deployer",
                    "caller_agent_id": "gitbob",
                }
            )
        data = json.loads(result[0].text)
        assert data["error"] == "access_denied"
        assert "Traceback" not in result[0].text

    @pytest.mark.asyncio
    async def test_map_import_maps_path_error(self, monkeypatch):
        from archivist.app.handlers import tools_memory_product as tmp
        from archivist.storage.backup_manager import SnapshotPathError

        monkeypatch.setattr(
            "archivist.core.rbac.is_permissive_mode",
            lambda: False,
        )
        with patch(
            "archivist.app.handlers.tools_memory_product.import_scope",
            new=AsyncMock(side_effect=SnapshotPathError("escape")),
        ):
            result = await tmp._handle_map_import(
                {
                    "archive_id": "../../etc/passwd",
                    "target_namespace": "pipeline",
                    "caller_agent_id": "gitbob",
                }
            )
        data = json.loads(result[0].text)
        assert data["error"] == "invalid_archive_id"
        assert "Traceback" not in result[0].text

    @pytest.mark.asyncio
    async def test_map_get_not_found(self, monkeypatch):
        from archivist.app.handlers import tools_memory_product as tmp

        monkeypatch.setattr(
            "archivist.core.rbac.is_permissive_mode",
            lambda: False,
        )
        with patch(
            "archivist.app.handlers.tools_memory_product.get_scope_version",
            new=AsyncMock(return_value=None),
        ):
            result = await tmp._handle_map_get(
                {"version_id": "missing", "caller_agent_id": "gitbob"}
            )
        data = json.loads(result[0].text)
        assert data["error"] == "not_found"
