"""Tests for MCP registry — tool listing and dispatch error handling."""

import json

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mcp]

pytestmark = pytest.mark.integration

class TestToolRegistry:
    def test_all_tools_registered(self):
        from handlers._registry import ALL_TOOLS, TOOL_REGISTRY

        assert len(ALL_TOOLS) >= 30
        for tool in ALL_TOOLS:
            assert tool.name in TOOL_REGISTRY, f"Tool {tool.name} has no handler"

    def test_tool_names_unique(self):
        from handlers._registry import ALL_TOOLS

        names = [t.name for t in ALL_TOOLS]
        assert len(names) == len(set(names)), (
            f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"
        )

    def test_all_tools_have_input_schema(self):
        from handlers._registry import ALL_TOOLS

        for tool in ALL_TOOLS:
            assert tool.inputSchema is not None, f"{tool.name} missing inputSchema"
            assert "type" in tool.inputSchema, f"{tool.name} schema missing 'type'"

    def test_all_tools_have_description(self):
        from handlers._registry import ALL_TOOLS

        for tool in ALL_TOOLS:
            assert tool.description, f"{tool.name} missing description"
            assert len(tool.description) > 10, f"{tool.name} description too short"

    def test_expected_tools_present(self):
        from handlers._registry import ALL_TOOLS

        names = {t.name for t in ALL_TOOLS}
        expected = {
            "archivist_search",
            "archivist_recall",
            "archivist_timeline",
            "archivist_insights",
            "archivist_deref",
            "archivist_index",
            "archivist_contradictions",
            "archivist_store",
            "archivist_merge",
            "archivist_compress",
            "archivist_log_trajectory",
            "archivist_annotate",
            "archivist_rate",
            "archivist_tips",
            "archivist_session_end",
            "archivist_register_skill",
            "archivist_skill_event",
            "archivist_skill_lesson",
            "archivist_skill_health",
            "archivist_skill_relate",
            "archivist_skill_dependencies",
            "archivist_context_check",
            "archivist_namespaces",
            "archivist_audit_trail",
            "archivist_resolve_uri",
            "archivist_retrieval_logs",
            "archivist_health_dashboard",
            "archivist_batch_heuristic",
            "archivist_cache_stats",
            "archivist_cache_invalidate",
        }
        missing = expected - names
        assert not missing, f"Missing expected tools: {missing}"

class TestDispatch:
    async def test_unknown_tool_returns_error(self):
        from handlers._registry import dispatch_tool

        result = await dispatch_tool("nonexistent_tool", {})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_context_check_no_args(self):
        from handlers._registry import dispatch_tool

        result = await dispatch_tool("archivist_context_check", {})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["hint"] == "ok"
        assert data["total_tokens"] == 0

    async def test_context_check_with_messages(self):
        from handlers._registry import dispatch_tool

        result = await dispatch_tool(
            "archivist_context_check",
            {
                "messages": [
                    {"role": "user", "content": "Hello world, this is a test message."},
                ],
                "budget_tokens": 1000,
            },
        )
        data = json.loads(result[0].text)
        assert data["total_tokens"] > 0
        assert data["hint"] == "ok"

    async def test_namespaces_returns_result(self, rbac_config):
        from handlers._registry import dispatch_tool


        result = await dispatch_tool("archivist_namespaces", {"agent_id": "chief"})
        data = json.loads(result[0].text)
        assert "accessible_namespaces" in data
