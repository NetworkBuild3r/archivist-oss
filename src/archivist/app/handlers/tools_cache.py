"""MCP tool handlers — hot cache stats and invalidation."""

import json

from mcp.types import Tool, TextContent

import archivist.retrieval.hot_cache as hot_cache

from ._common import success_response

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="archivist_cache_stats",
        description="Return hot cache statistics: entries per agent, TTL, hit rate.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="archivist_cache_invalidate",
        description=(
            "Manually invalidate the hot cache. Useful after bulk writes or migrations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Invalidate entries for this namespace (optional)", "default": ""},
                "agent_id": {"type": "string", "description": "Invalidate entries for this agent (optional)", "default": ""},
                "all": {"type": "boolean", "description": "Invalidate entire cache", "default": False},
            },
            "required": [],
        },
    ),
]

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_cache_stats(arguments: dict) -> list[TextContent]:
    """Return hot cache statistics."""
    return success_response(hot_cache.stats())


async def _handle_cache_invalidate(arguments: dict) -> list[TextContent]:
    """Manually invalidate cache entries."""
    if arguments.get("all"):
        n = hot_cache.invalidate_all()
        return success_response({"invalidated": n, "scope": "all"})
    if arguments.get("agent_id"):
        n = hot_cache.invalidate_agent(arguments["agent_id"])
        return success_response({"invalidated": n, "scope": f"agent:{arguments['agent_id']}"})
    if arguments.get("namespace"):
        n = hot_cache.invalidate_namespace(arguments["namespace"])
        return success_response({"invalidated": n, "scope": f"namespace:{arguments['namespace']}"})
    return success_response({"invalidated": 0, "hint": "Specify namespace, agent_id, or all=true"})


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, object] = {
    "archivist_cache_stats": _handle_cache_stats,
    "archivist_cache_invalidate": _handle_cache_invalidate,
}
