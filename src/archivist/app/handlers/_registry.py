"""Central tool registry — aggregates tool definitions and handlers from domain modules."""

import logging
import time
from collections.abc import Awaitable, Callable

from mcp.types import TextContent, Tool

import archivist.core.metrics as m
from archivist.core.observability import get_request_id, tool_span

from ._common import error_response
from .tools_admin import HANDLERS as ADMIN_HANDLERS
from .tools_admin import TOOLS as ADMIN_TOOLS
from .tools_cache import HANDLERS as CACHE_HANDLERS
from .tools_cache import TOOLS as CACHE_TOOLS
from .tools_context import HANDLERS as CONTEXT_HANDLERS
from .tools_context import TOOLS as CONTEXT_TOOLS
from .tools_docs import HANDLERS as DOCS_HANDLERS
from .tools_docs import TOOLS as DOCS_TOOLS
from .tools_search import HANDLERS as SEARCH_HANDLERS
from .tools_search import TOOLS as SEARCH_TOOLS
from .tools_skills import HANDLERS as SKILL_HANDLERS
from .tools_skills import TOOLS as SKILL_TOOLS
from .tools_storage import HANDLERS as STORAGE_HANDLERS
from .tools_storage import TOOLS as STORAGE_TOOLS
from .tools_trajectory import HANDLERS as TRAJECTORY_HANDLERS
from .tools_trajectory import TOOLS as TRAJECTORY_TOOLS

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
    CONTEXT_HANDLERS,
    DOCS_HANDLERS,
):
    TOOL_REGISTRY.update(_handlers)

ALL_TOOLS: list[Tool] = (
    SEARCH_TOOLS
    + STORAGE_TOOLS
    + TRAJECTORY_TOOLS
    + SKILL_TOOLS
    + ADMIN_TOOLS
    + CACHE_TOOLS
    + CONTEXT_TOOLS
    + DOCS_TOOLS
)


def get_all_tools() -> list[Tool]:
    return ALL_TOOLS


async def dispatch_tool(name: str, arguments: dict) -> list[TextContent]:
    """Look up a handler by tool name and call it, with top-level error handling."""
    handler = TOOL_REGISTRY.get(name)
    if not handler:
        return error_response({"error": f"Unknown tool: {name}"})

    rid = get_request_id()
    caller = (arguments.get("caller_agent_id") or arguments.get("agent_id") or "")[:64]
    logger.info("tool.started tool=%s caller=%s request_id=%s", name, caller, rid)
    t0 = time.monotonic()

    try:
        with tool_span(name):
            result = await handler(arguments)
        dur = round((time.monotonic() - t0) * 1000, 1)
        logger.info(
            "tool.finished tool=%s caller=%s duration_ms=%.1f request_id=%s",
            name,
            caller,
            dur,
            rid,
        )
        m.observe(m.TOOL_DURATION, dur, {"tool": name})
        return result
    except Exception as e:
        dur = round((time.monotonic() - t0) * 1000, 1)
        logger.error(
            "tool.failed tool=%s caller=%s duration_ms=%.1f error=%s request_id=%s",
            name,
            caller,
            dur,
            e,
            rid,
            exc_info=True,
        )
        m.inc(m.TOOL_ERRORS, {"tool": name})
        m.observe(m.TOOL_DURATION, dur, {"tool": name})
        return error_response({"error": str(e)})
