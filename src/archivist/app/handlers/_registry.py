"""Central tool registry — aggregates tool definitions and handlers from domain modules.

INIT-003/SPEC-003: ``get_all_tools`` / ``dispatch_tool`` honor
``ARCHIVIST_TOOL_PROFILE`` (core|ops|full). Non-core tools stay registered;
the profile only gates discovery and invocation.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import cast

from mcp.types import TextContent, Tool

import archivist.core.config as config
import archivist.core.metrics as m
from archivist.core.observability import get_request_id, tool_span

from ._common import error_response
from .tools_admin import HANDLERS as ADMIN_HANDLERS
from .tools_admin import TOOLS as ADMIN_TOOLS
from .tools_cache import HANDLERS as CACHE_HANDLERS
from .tools_cache import TOOLS as CACHE_TOOLS
from .tools_checkpoint import HANDLERS as CHECKPOINT_HANDLERS
from .tools_checkpoint import TOOLS as CHECKPOINT_TOOLS
from .tools_context import HANDLERS as CONTEXT_HANDLERS
from .tools_context import TOOLS as CONTEXT_TOOLS
from .tools_coordination import HANDLERS as COORDINATION_HANDLERS
from .tools_coordination import TOOLS as COORDINATION_TOOLS
from .tools_docs import HANDLERS as DOCS_HANDLERS
from .tools_docs import TOOLS as DOCS_TOOLS
from .tools_memory_product import HANDLERS as MAP_HANDLERS
from .tools_memory_product import TOOLS as MAP_TOOLS
from .tools_search import HANDLERS as SEARCH_HANDLERS
from .tools_search import TOOLS as SEARCH_TOOLS
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
    ADMIN_HANDLERS,
    CACHE_HANDLERS,
    CONTEXT_HANDLERS,
    CHECKPOINT_HANDLERS,
    COORDINATION_HANDLERS,
    DOCS_HANDLERS,
    MAP_HANDLERS,
):
    # Domain modules type HANDLERS as dict[str, object]; values are HandlerFn.
    TOOL_REGISTRY.update(cast(dict[str, HandlerFn], _handlers))

ALL_TOOLS: list[Tool] = (
    SEARCH_TOOLS
    + STORAGE_TOOLS
    + TRAJECTORY_TOOLS
    + ADMIN_TOOLS
    + CACHE_TOOLS
    + CONTEXT_TOOLS
    + CHECKPOINT_TOOLS
    + COORDINATION_TOOLS
    + DOCS_TOOLS
    + MAP_TOOLS
)

# Coach-core surface (ADR-003 five-tool contract + small read helpers). ≤12.
# Forget path: ADR names ``archivist_forget``; core tool is ``archivist_delete``
# with ``mode=delete|suppress`` (INIT-003/SPEC-006). No separate forget tool —
# keeps core ≤12 (GR-PROD-002).
# INIT-003/SPEC-003
CORE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "archivist_store",
        "archivist_search",
        "archivist_get_context",
        "archivist_index",
        "archivist_delete",
        "archivist_health_dashboard",
        "archivist_namespaces",
        "archivist_get_reference_docs",
    }
)

# Ops hides unfinished checkpoint wedge only. ``archivist_share_*`` is promoted
# to ops under INIT-009/SPEC-002 (ADR-009 GR-PROD-002 — still off core).
# ``archivist_map_*`` is promoted to ops under INIT-011/SPEC-003 (ADR-011) —
# do **not** add ``archivist_map_`` here.
_OPS_HIDDEN_PREFIXES: tuple[str, ...] = ("archivist_checkpoint_",)


def allowed_tool_names(profile: str | None = None) -> frozenset[str]:
    """Return tool names visible/callable for ``profile`` (default: config)."""
    resolved = (profile if profile is not None else config.TOOL_PROFILE).strip().lower()
    if resolved == "full":
        return frozenset(t.name for t in ALL_TOOLS)
    if resolved == "ops":
        return frozenset(
            t.name for t in ALL_TOOLS if not any(t.name.startswith(p) for p in _OPS_HIDDEN_PREFIXES)
        )
    # Default / unknown → core (fail closed to smallest surface)
    return CORE_TOOL_NAMES


def get_all_tools() -> list[Tool]:
    """Tools exposed via MCP ``list_tools`` for the active profile."""
    allowed = allowed_tool_names()
    return [t for t in ALL_TOOLS if t.name in allowed]


async def dispatch_tool(name: str, arguments: dict) -> list[TextContent]:
    """Look up a handler by tool name and call it, with top-level error handling.

    Hidden-by-profile tools fail closed with a clear error (INIT-003/SPEC-003).
    """
    allowed = allowed_tool_names()
    if name not in allowed:
        if name in TOOL_REGISTRY:
            return error_response(
                {
                    "error": (
                        f"Tool '{name}' is not available in tool profile "
                        f"'{config.TOOL_PROFILE}'. Set ARCHIVIST_TOOL_PROFILE=full "
                        "(or ops) to enable."
                    )
                }
            )
        return error_response({"error": f"Unknown tool: {name}"})

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
