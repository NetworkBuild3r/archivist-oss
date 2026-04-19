"""Archivist MCP server — thin orchestrator wiring the MCP SDK to domain tool modules.

Tool definitions and handlers live in handlers/ subpackage, split by domain:
  - tools_search.py    (search, recall, timeline, insights, deref, index, contradictions)
  - tools_storage.py   (store, merge, compress)
  - tools_trajectory.py (log_trajectory, annotate, rate, tips, session_end)
  - tools_skills.py    (register_skill, skill_event, skill_lesson, skill_health, skill_relate, skill_dependencies)
  - tools_admin.py     (namespaces, audit_trail, resolve_uri, retrieval_logs, health_dashboard, batch_heuristic)
  - tools_cache.py     (cache_stats, cache_invalidate)

The registry in handlers/_registry.py aggregates all tools and dispatches by name.

MCP Resources
-------------
Two resource families are exposed so compliant hosts can pre-load them before
the agent issues its first tool call:

  archivist://memory-index/{agent_id}
      The compressed memory index for the agent's default namespace — identical
      to calling archivist_index(agent_id=…).  Agents should save this as
      ``memory_index.md`` in their project root on first connection.

  archivist://namespaces/{agent_id}
      A JSON summary of every namespace the agent can read or write — identical
      to calling archivist_namespaces(agent_id=…).
"""

import json

from mcp.server import Server
from mcp.types import Resource, ResourceTemplate, TextContent, TextResourceContents, Tool
from pydantic import AnyUrl

from handlers._registry import dispatch_tool, get_all_tools

server = Server("archivist")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[Tool]:
    return get_all_tools()


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    return await dispatch_tool(name, arguments)


# ---------------------------------------------------------------------------
# Resources — static list + URI-template read
# ---------------------------------------------------------------------------

_INDEX_TEMPLATE = "archivist://memory-index/{agent_id}"
_NS_TEMPLATE = "archivist://namespaces/{agent_id}"


@server.list_resources()
async def list_resources() -> list[Resource]:
    """Return the two well-known resource URIs agents should fetch on connect."""
    return [
        Resource(
            uri=AnyUrl("archivist://memory-index/"),  # type: ignore[arg-type]
            name="Memory Index",
            description=(
                "Compressed navigational index of your agent's memory namespace. "
                "Fetch as archivist://memory-index/{your_agent_id} and save as "
                "memory_index.md in your project root so you always know what "
                "knowledge exists before searching or storing."
            ),
            mimeType="text/markdown",
        ),
        Resource(
            uri=AnyUrl("archivist://namespaces/"),  # type: ignore[arg-type]
            name="Namespace Access List",
            description=(
                "JSON list of every namespace you can read or write. "
                "Fetch as archivist://namespaces/{your_agent_id}."
            ),
            mimeType="application/json",
        ),
    ]


@server.list_resource_templates()
async def list_resource_templates() -> list[ResourceTemplate]:
    return [
        ResourceTemplate(
            uriTemplate=_INDEX_TEMPLATE,
            name="Memory Index (per agent)",
            description=(
                "Compressed memory index for agent_id's default namespace. "
                "Same content as archivist_index(agent_id=…). "
                "Save as memory_index.md in your project root."
            ),
            mimeType="text/markdown",
        ),
        ResourceTemplate(
            uriTemplate=_NS_TEMPLATE,
            name="Namespace Access List (per agent)",
            description=(
                "JSON namespace access summary for agent_id. "
                "Same content as archivist_namespaces(agent_id=…)."
            ),
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def read_resource(uri: AnyUrl) -> list[TextResourceContents]:
    """Serve memory-index and namespace resources by URI.

    URI patterns:
      archivist://memory-index/{agent_id}  → markdown memory index
      archivist://namespaces/{agent_id}    → JSON namespace list
    """
    uri_str = str(uri)

    # ---- memory index ---------------------------------------------------
    if uri_str.startswith("archivist://memory-index/"):
        agent_id = uri_str.removeprefix("archivist://memory-index/").strip("/")
        from archivist.core.rbac import get_namespace_for_agent
        from archivist.storage.compressed_index import build_namespace_index

        namespace = get_namespace_for_agent(agent_id) if agent_id else ""
        index_text = build_namespace_index(namespace, agent_ids=[agent_id] if agent_id else None)

        # Prepend a save-as hint so the agent knows what to do with this content.
        header = (
            f"<!-- archivist memory index for agent={agent_id!r} namespace={namespace!r} -->\n"
            "<!-- Save this file as memory_index.md in your project root. -->\n\n"
        )
        return [
            TextResourceContents(
                uri=uri,
                mimeType="text/markdown",
                text=header + index_text,
            )
        ]

    # ---- namespace list --------------------------------------------------
    if uri_str.startswith("archivist://namespaces/"):
        agent_id = uri_str.removeprefix("archivist://namespaces/").strip("/")
        from archivist.core.rbac import list_accessible_namespaces

        namespaces = list_accessible_namespaces(agent_id)
        return [
            TextResourceContents(
                uri=uri,
                mimeType="application/json",
                text=json.dumps(
                    {
                        "agent_id": agent_id,
                        "namespaces": namespaces,
                        "hint": (
                            "Save archivist://memory-index/{agent_id} as memory_index.md "
                            "in your project root for a full navigational index."
                        ),
                    },
                    indent=2,
                ),
            )
        ]

    # ---- unknown URI ----------------------------------------------------
    return [
        TextResourceContents(
            uri=uri,
            mimeType="application/json",
            text=json.dumps(
                {
                    "error": "unknown_resource",
                    "uri": uri_str,
                    "available_templates": [_INDEX_TEMPLATE, _NS_TEMPLATE],
                    "hint": (
                        "Fetch archivist://memory-index/{agent_id} for your memory index "
                        "or archivist://namespaces/{agent_id} for your namespace access list."
                    ),
                }
            ),
        )
    ]
