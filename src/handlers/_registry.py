"""Central tool registry — aggregates tool definitions and handlers from domain modules."""

import logging
from typing import Callable, Awaitable

from mcp.types import Tool, TextContent

from .tools_search import TOOLS as SEARCH_TOOLS, HANDLERS as SEARCH_HANDLERS
from .tools_storage import TOOLS as STORAGE_TOOLS, HANDLERS as STORAGE_HANDLERS
from .tools_trajectory import TOOLS as TRAJECTORY_TOOLS, HANDLERS as TRAJECTORY_HANDLERS
from .tools_skills import TOOLS as SKILL_TOOLS, HANDLERS as SKILL_HANDLERS
from .tools_admin import TOOLS as ADMIN_TOOLS, HANDLERS as ADMIN_HANDLERS
from .tools_cache import TOOLS as CACHE_TOOLS, HANDLERS as CACHE_HANDLERS
from ._common import error_response

logger = logging.getLogger("archivist.mcp")

HandlerFn = Callable[[dict], Awaitable[list[TextContent]]]

TOOL_REGISTRY: dict[str, HandlerFn] = {}
for _handlers in (
    SEARCH_HANDLERS,
    STORAGE_HANDLERS,
    TRAJECTORY_HANDLERS,
    SKILL_HANDLERS,
    ADMIN_HANDLERS,
    CACHE_HANDLERS,
):
    TOOL_REGISTRY.update(_handlers)

ALL_TOOLS: list[Tool] = (
    SEARCH_TOOLS
    + STORAGE_TOOLS
    + TRAJECTORY_TOOLS
    + SKILL_TOOLS
    + ADMIN_TOOLS
    + CACHE_TOOLS
)


def get_all_tools() -> list[Tool]:
    return ALL_TOOLS


async def dispatch_tool(name: str, arguments: dict) -> list[TextContent]:
    """Look up a handler by tool name and call it, with top-level error handling."""
    try:
        handler = TOOL_REGISTRY.get(name)
        if handler:
            return await handler(arguments)
        return error_response({"error": f"Unknown tool: {name}"})
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e, exc_info=True)
        return error_response({"error": str(e)})
