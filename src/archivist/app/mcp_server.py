"""Archivist MCP server — transport-agnostic orchestrator (INIT-024/SPEC-007, SPEC-008).

Wires the MCP SDK ``Server`` for ``stdio``, ``archivistd`` Unix domain socket,
and Streamable HTTP transports:
- Tools: default 5-tool ``coding`` profile (<900 schema tokens) via ``_registry.py``
- Resources: 5 zero-schema-token ``archivist://`` resources & templates
- Prompts: 3 zero-schema-token workflow prompts
- Session context: extracts ``roots/list`` and ``clientInfo.name`` when supported
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server import Server
from mcp.types import (
    GetPromptResult,
    Prompt,
    Resource,
    ResourceTemplate,
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


async def _sync_session_metadata(srv: Server) -> None:
    """Capture MCP ``clientInfo.name`` and ``roots/list`` into task-local contextvars."""
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

    # 2. Query MCP roots/list when client advertised roots capability.
    capabilities = (
        getattr(client_params, "capabilities", None) if client_params is not None else None
    )
    if capabilities is not None and getattr(capabilities, "roots", None) is not None:
        try:
            roots_resp = await session.list_roots()
            uris = [str(r.uri) for r in getattr(roots_resp, "roots", [])]
            if uris:
                set_session_roots(uris)
        except Exception as exc:
            logger.debug("roots/list unavailable for session: %s", exc)


def create_mcp_server(name: str = "archivist") -> Server:
    """Build a transport-agnostic MCP ``Server`` wired to tools, resources, and prompts."""
    srv = Server(name)

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
        return read_resource_uri(uri)

    @srv.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return list(PROMPTS)

    @srv.get_prompt()
    async def get_prompt(
        prompt_name: str, arguments: dict[str, str] | None = None
    ) -> GetPromptResult:
        await _sync_session_metadata(srv)
        return get_prompt_by_name(prompt_name, arguments)

    return srv


server = create_mcp_server("archivist")
