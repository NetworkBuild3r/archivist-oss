"""Archivist MCP server — transport-agnostic orchestrator (INIT-024/SPEC-007, SPEC-008).

Wires the MCP SDK ``Server`` for ``stdio``, ``archivistd`` Unix domain socket,
and Streamable HTTP transports:
- Tools: default 5-tool ``coding`` profile (<900 schema tokens) via ``_registry.py``
- Resources: 5 zero-schema-token ``archivist://`` resources & templates
- Prompts: 3 zero-schema-token workflow prompts
- Session context: extracts ``clientInfo.name`` and caches ``roots/list`` per
  session (invalidated via ``notifications/roots/list_changed``)
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Any

from mcp.server import Server
from mcp.types import (
    GetPromptResult,
    Prompt,
    Resource,
    ResourceTemplate,
    RootsListChangedNotification,
    TextContent,
    Tool,
)
from pydantic import AnyUrl

from archivist.app.handlers._registry import dispatch_tool, get_all_tools
from archivist.app.handlers.scope_resolver import (
    get_request_context,
    set_request_context,
    set_session_roots,
)
from archivist.app.handlers.v3_resources import (
    PROMPTS,
    get_prompt_by_name,
    get_resource_templates,
    get_static_resources,
    read_resource_uri,
)
from archivist.security.trust_tiers import RequestContext

logger = logging.getLogger("archivist.mcp")

# Per-session roots/list cache (weak-keyed so closed sessions garbage-collect).
_SESSION_ROOTS_CACHE: weakref.WeakKeyDictionary[Any, tuple[str, ...]] = weakref.WeakKeyDictionary()
_SESSION_ROOTS_FALLBACK: dict[int, tuple[str, ...]] = {}


def _get_cached_roots(session: Any) -> tuple[str, ...] | None:
    try:
        return _SESSION_ROOTS_CACHE.get(session)
    except TypeError:
        return _SESSION_ROOTS_FALLBACK.get(id(session))


def _set_cached_roots(session: Any, roots: tuple[str, ...]) -> None:
    try:
        _SESSION_ROOTS_CACHE[session] = roots
    except TypeError:
        _SESSION_ROOTS_FALLBACK[id(session)] = roots


def invalidate_session_roots(session: Any) -> None:
    """Clear cached ``roots/list`` entries for *session*."""
    try:
        _SESSION_ROOTS_CACHE.pop(session, None)
    except TypeError:
        _SESSION_ROOTS_FALLBACK.pop(id(session), None)


async def _sync_session_metadata(srv: Server) -> None:
    """Capture MCP ``clientInfo.name`` and cached ``roots/list`` into task-local contextvars."""
    try:
        req_ctx = srv.request_context
    except LookupError:
        return

    session = getattr(req_ctx, "session", None)
    if session is None:
        return

    # 1. Capture clientInfo.name from initialize params if available.
    client_params = getattr(session, "client_params", None)
    client_info = getattr(client_params, "clientInfo", None) if client_params is not None else None
    client_name = getattr(client_info, "name", None) if client_info is not None else None
    if client_name:
        current = get_request_context()
        if current.source_client != client_name:
            set_request_context(
                RequestContext(
                    principal_id=current.principal_id,
                    actor_type=current.actor_type,
                    tenant_id=current.tenant_id,
                    source_client=str(client_name),
                    channel=current.channel,
                    uid=current.uid,
                )
            )

    # 2. Query MCP roots/list once per session when client advertised roots capability,
    #    refreshing on notifications/roots/list_changed (HIGH-7).
    capabilities = (
        getattr(client_params, "capabilities", None) if client_params is not None else None
    )
    if capabilities is not None and getattr(capabilities, "roots", None) is not None:
        cached = _get_cached_roots(session)
        if cached is not None:
            if cached:
                set_session_roots(cached)
            return
        try:
            roots_resp = await session.list_roots()
            uris = tuple(str(r.uri) for r in getattr(roots_resp, "roots", []) if str(r.uri).strip())
            _set_cached_roots(session, uris)
            if uris:
                set_session_roots(uris)
        except Exception as exc:
            logger.debug("roots/list unavailable for session: %s", exc)


def create_mcp_server(name: str = "archivist") -> Server:
    """Build a transport-agnostic MCP ``Server`` wired to tools, resources, and prompts."""
    srv = Server(name)

    async def _on_roots_list_changed(_notification: RootsListChangedNotification) -> None:
        try:
            req_ctx = srv.request_context
            session = getattr(req_ctx, "session", None)
            if session is not None:
                invalidate_session_roots(session)
        except LookupError:
            pass

    srv.notification_handlers[RootsListChangedNotification] = _on_roots_list_changed

    @srv.list_tools()
    async def list_tools() -> list[Tool]:
        return get_all_tools()

    @srv.call_tool()
    async def call_tool(tool_name: str, arguments: dict[str, Any]) -> list[TextContent]:
        await _sync_session_metadata(srv)
        return await dispatch_tool(tool_name, arguments)

    @srv.list_resources()
    async def list_resources() -> list[Resource]:
        return get_static_resources()

    @srv.list_resource_templates()
    async def list_resource_templates() -> list[ResourceTemplate]:
        return get_resource_templates()

    @srv.read_resource()
    async def read_resource(uri: AnyUrl) -> str:
        await _sync_session_metadata(srv)
        return await asyncio.to_thread(read_resource_uri, uri)

    @srv.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return list(PROMPTS)

    @srv.get_prompt()
    async def get_prompt(
        prompt_name: str, arguments: dict[str, str] | None = None
    ) -> GetPromptResult:
        await _sync_session_metadata(srv)
        return await asyncio.to_thread(get_prompt_by_name, prompt_name, arguments)

    return srv


server = create_mcp_server("archivist")
