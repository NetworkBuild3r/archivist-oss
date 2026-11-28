"""Central tool registry — aggregates tool definitions and handlers from domain modules.

INIT-024/SPEC-008: ``get_all_tools`` / ``dispatch_tool`` honor
``ARCHIVIST_TOOL_PROFILE`` (coding|core|legacy|ops|full). Default ``coding``
exposes the pruned 5-tool v3 coding surface (<900 schema tokens) with **lazy**
loading of legacy v2.5 modules so default wheel / ``uvx archivist-mcp`` installs
never import optional ``qdrant_client`` or fleet dependencies. Non-allowed tools
fail closed on dispatch.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

from mcp.types import TextContent, Tool

import archivist.core.config as config
import archivist.core.metrics as m
from archivist.core.observability import get_request_id, tool_span

from ._common import error_response
from .v3_coding import CODING_HANDLERS, CODING_TOOL_NAMES, CODING_TOOLS

logger = logging.getLogger("archivist.mcp")

HandlerFn = Callable[[dict], Awaitable[list[TextContent]]]

# Legacy v2.5 coach-core surface (INIT-003/SPEC-003, retained under core/legacy).
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

# Empty hide list: ops currently equals full until the next unfinished wedge.
_OPS_HIDDEN_PREFIXES: tuple[str, ...] = ()

_legacy_loaded = False
_full_registry: dict[str, HandlerFn] = dict(cast(dict[str, HandlerFn], CODING_HANDLERS))
_full_tools: list[Tool] = list(CODING_TOOLS)


def _ensure_legacy_loaded() -> None:
    """Lazily import legacy v2.5 handlers only when a legacy profile or full table is requested."""
    global _legacy_loaded
    if _legacy_loaded:
        return

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

    for handlers in (
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
        _full_registry.update(cast(dict[str, HandlerFn], handlers))

    _full_tools.extend(
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
    _legacy_loaded = True


def __getattr__(name: str) -> Any:
    """Provide lazy backwards-compatible ``ALL_TOOLS`` and ``TOOL_REGISTRY`` exports."""
    if name == "ALL_TOOLS":
        _ensure_legacy_loaded()
        globals()["ALL_TOOLS"] = _full_tools
        return _full_tools
    if name == "TOOL_REGISTRY":
        _ensure_legacy_loaded()
        globals()["TOOL_REGISTRY"] = _full_registry
        return _full_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def allowed_tool_names(profile: str | None = None) -> frozenset[str]:
    """Return tool names visible/callable for ``profile`` (default: config)."""
    resolved = (profile if profile is not None else config.TOOL_PROFILE).strip().lower()
    if resolved == "full":
        _ensure_legacy_loaded()
        return frozenset(t.name for t in _full_tools)
    if resolved == "ops":
        _ensure_legacy_loaded()
        return frozenset(
            t.name
            for t in _full_tools
            if not any(t.name.startswith(p) for p in _OPS_HIDDEN_PREFIXES)
        )
    if resolved in ("core", "legacy"):
        return CORE_TOOL_NAMES
    # Default / unknown -> coding (fail closed to smallest 5-tool surface, <900 tok)
    return CODING_TOOL_NAMES


def get_all_tools(profile: str | None = None) -> list[Tool]:
    """Tools exposed via MCP ``list_tools`` for the active profile."""
    resolved = (profile if profile is not None else config.TOOL_PROFILE).strip().lower()
    if resolved not in ("full", "ops", "core", "legacy"):
        return list(CODING_TOOLS)
    _ensure_legacy_loaded()
    allowed = allowed_tool_names(resolved)
    return [t for t in _full_tools if t.name in allowed]


def _is_known_hidden_tool(name: str) -> bool:
    """Return True if *name* is a known tool outside the active profile without forcing imports."""
    if name in _full_registry or name in CORE_TOOL_NAMES or name in CODING_TOOL_NAMES:
        return True
    return name.startswith("archivist_")


async def dispatch_tool(name: str, arguments: dict) -> list[TextContent]:
    """Look up a handler by tool name and call it, with top-level error handling.

    Hidden-by-profile tools fail closed with a clear error (INIT-024/SPEC-008).
    """
    allowed = allowed_tool_names()
    if name not in allowed:
        if _is_known_hidden_tool(name):
            return error_response(
                {
                    "error": (
                        f"Tool '{name}' is not available in tool profile "
                        f"'{config.TOOL_PROFILE}'. Set ARCHIVIST_TOOL_PROFILE=core|ops|full "
                        "to enable."
                    )
                }
            )
        return error_response({"error": f"Unknown tool: {name}"})

    if name not in _full_registry:
        _ensure_legacy_loaded()

    handler = _full_registry.get(name)
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
