"""INIT-003/SPEC-003 — MCP tool profile filtering (core|ops|full)."""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.unit]


class TestCoreProfileMembership:
    def test_core_contains_coach_contract(self):
        from archivist.app.handlers._registry import CORE_TOOL_NAMES

        required = {
            "archivist_store",
            "archivist_search",
            "archivist_get_context",
            "archivist_index",
            "archivist_delete",
        }
        assert required <= CORE_TOOL_NAMES
        assert len(CORE_TOOL_NAMES) <= 12

    def test_core_excludes_share_and_checkpoint(self, monkeypatch):
        import archivist.core.config as config
        from archivist.app.handlers._registry import get_all_tools

        monkeypatch.setattr(config, "TOOL_PROFILE", "core")
        names = {t.name for t in get_all_tools()}
        assert names == {
            "archivist_store",
            "archivist_search",
            "archivist_get_context",
            "archivist_index",
            "archivist_delete",
            "archivist_health_dashboard",
            "archivist_namespaces",
            "archivist_get_reference_docs",
        }
        assert not any(n.startswith("archivist_share_") for n in names)
        assert not any(n.startswith("archivist_checkpoint_") for n in names)

    def test_default_profile_is_core(self):
        from archivist.core.config import ArchivistSettings

        assert ArchivistSettings.model_fields["archivist_tool_profile"].default == "core"
        s = ArchivistSettings.model_construct()
        assert s.archivist_tool_profile == "core"


class TestOpsAndFullProfiles:
    def test_ops_includes_share_excludes_checkpoint(self, monkeypatch):
        """INIT-009/SPEC-002: share_* promoted to ops; checkpoint still full-only."""
        import archivist.core.config as config
        from archivist.app.handlers._registry import ALL_TOOLS, get_all_tools

        monkeypatch.setattr(config, "TOOL_PROFILE", "ops")
        names = {t.name for t in get_all_tools()}
        assert "archivist_store" in names
        assert "archivist_context_check" in names
        assert "archivist_share_propose" in names
        assert "archivist_share_attach_conflict" in names
        assert not any(n.startswith("archivist_checkpoint_") for n in names)
        # ops is a middle set — smaller than full, larger than core
        assert len(names) < len(ALL_TOOLS)
        assert len(names) > 12

    def test_full_restores_entire_registry(self, monkeypatch):
        import archivist.core.config as config
        from archivist.app.handlers._registry import ALL_TOOLS, get_all_tools

        monkeypatch.setattr(config, "TOOL_PROFILE", "full")
        names = [t.name for t in get_all_tools()]
        assert names == [t.name for t in ALL_TOOLS]
        assert "archivist_share_propose" in names
        assert "archivist_checkpoint_save" in names


class TestDispatchFailClosed:
    @pytest.mark.asyncio
    async def test_hidden_tool_blocked_under_core(self, monkeypatch):
        import archivist.core.config as config
        from archivist.app.handlers._registry import TOOL_REGISTRY, dispatch_tool

        monkeypatch.setattr(config, "TOOL_PROFILE", "core")
        assert "archivist_share_propose" in TOOL_REGISTRY
        result = await dispatch_tool("archivist_share_propose", {})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "not available" in data["error"]
        assert "core" in data["error"]

    @pytest.mark.asyncio
    async def test_share_available_under_ops(self, monkeypatch):
        """INIT-009/SPEC-002: share_* dispatchable on ops (checkpoint still blocked)."""
        import archivist.core.config as config
        from archivist.app.handlers._registry import TOOL_REGISTRY, dispatch_tool

        monkeypatch.setattr(config, "TOOL_PROFILE", "ops")
        assert "archivist_share_propose" in TOOL_REGISTRY
        # Missing required args → handler error, not profile gate
        result = await dispatch_tool(
            "archivist_share_propose",
            {"agent_id": "a", "recipient_agent_id": "b", "namespace": ""},
        )
        data = json.loads(result[0].text)
        assert "error" in data
        assert "not available" not in data["error"]

    @pytest.mark.asyncio
    async def test_checkpoint_blocked_under_ops(self, monkeypatch):
        import archivist.core.config as config
        from archivist.app.handlers._registry import TOOL_REGISTRY, dispatch_tool

        monkeypatch.setattr(config, "TOOL_PROFILE", "ops")
        assert "archivist_checkpoint_save" in TOOL_REGISTRY
        result = await dispatch_tool("archivist_checkpoint_save", {})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "not available" in data["error"]

    @pytest.mark.asyncio
    async def test_unknown_tool_still_unknown(self, monkeypatch):
        import archivist.core.config as config
        from archivist.app.handlers._registry import dispatch_tool

        monkeypatch.setattr(config, "TOOL_PROFILE", "core")
        result = await dispatch_tool("totally_fake_tool", {})
        data = json.loads(result[0].text)
        assert data["error"] == "Unknown tool: totally_fake_tool"


class TestConfigValidation:
    def test_invalid_profile_rejected(self, monkeypatch):
        from pydantic import ValidationError

        from archivist.core.config import _build_settings

        monkeypatch.setenv("ARCHIVIST_TOOL_PROFILE", "fleet")
        with pytest.raises(ValidationError):
            _build_settings()

    def test_env_override_ops(self, monkeypatch):
        from archivist.core.config import _build_settings

        monkeypatch.setenv("ARCHIVIST_TOOL_PROFILE", "OPS")
        s = _build_settings()
        assert s.archivist_tool_profile == "ops"
