"""Archivist MCP server — thin orchestrator wiring the MCP SDK to domain tool modules.

Tool definitions and handlers live in handlers/ subpackage, split by domain:
  - tools_search.py    (search, recall, timeline, insights, deref, index, contradictions)
  - tools_storage.py   (store, merge, compress)
  - tools_trajectory.py (log_trajectory, annotate, rate, tips, session_end)
  - tools_skills.py    (register_skill, skill_event, skill_lesson, skill_health, skill_relate, skill_dependencies)
  - tools_admin.py     (namespaces, audit_trail, resolve_uri, retrieval_logs, health_dashboard, batch_heuristic)
  - tools_cache.py     (cache_stats, cache_invalidate)

The registry in handlers/_registry.py aggregates all tools and dispatches by name.
"""

from mcp.server import Server
from mcp.types import Tool, TextContent

from handlers._registry import get_all_tools, dispatch_tool

server = Server("archivist")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return get_all_tools()


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    return await dispatch_tool(name, arguments)
