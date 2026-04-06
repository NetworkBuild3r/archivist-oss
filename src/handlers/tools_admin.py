"""MCP tool handlers — admin, observability, namespaces, audit, URI resolution."""

import logging

from mcp.types import Tool, TextContent

from rbac import get_namespace_for_agent, list_accessible_namespaces
from skills import find_skill, get_skill_health, get_lessons
from retrieval_log import get_retrieval_logs, get_retrieval_stats
from dashboard import build_dashboard, batch_heuristic

from ._common import _rbac_gate, error_response, success_response

logger = logging.getLogger("archivist.mcp")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="archivist_context_check",
        description=(
            "Check a set of messages or memory texts against a token budget. "
            "Returns token count, budget usage percentage, and a hint "
            "(ok / compress / critical). Use before reasoning to decide if "
            "context should be compacted."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                    "description": "Chat messages to count tokens for (alternative to memory_texts).",
                },
                "memory_texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Raw memory texts to count tokens for (alternative to messages).",
                },
                "budget_tokens": {
                    "type": "integer",
                    "description": "Target token budget (e.g. 128000 for GPT-4o). Defaults to DEFAULT_CONTEXT_BUDGET env.",
                },
                "reserve_from_tail": {
                    "type": "integer",
                    "description": "Tokens to reserve for recent messages when splitting (default 2000). Only used with messages.",
                    "default": 2000,
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="archivist_namespaces",
        description="List memory namespaces accessible to the calling agent.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The calling agent's ID"},
            },
            "required": ["agent_id"],
        },
    ),
    Tool(
        name="archivist_audit_trail",
        description="View audit log for a specific memory or agent activity.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Calling agent"},
                "memory_id": {"type": "string", "description": "Specific memory ID to audit (optional)", "default": ""},
                "target_agent": {"type": "string", "description": "Agent whose activity to view (optional)", "default": ""},
                "limit": {"type": "integer", "description": "Max entries to return", "default": 50},
            },
            "required": ["agent_id"],
        },
    ),
    Tool(
        name="archivist_resolve_uri",
        description=(
            "Resolve an archivist:// URI to its underlying resource. "
            "Supports memory, entity, namespace, and skill URIs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "uri": {"type": "string", "description": "An archivist:// URI to resolve"},
                "agent_id": {"type": "string", "description": "Calling agent for RBAC", "default": ""},
            },
            "required": ["uri"],
        },
    ),
    Tool(
        name="archivist_retrieval_logs",
        description=(
            "Export recent retrieval trajectory logs for debugging and analytics. "
            "Shows full pipeline traces: query, cache hit, duration, stage counts."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Filter by agent (optional)", "default": ""},
                "limit": {"type": "integer", "description": "Max entries", "default": 20},
                "since": {"type": "string", "description": "ISO datetime lower bound (optional)", "default": ""},
                "stats_only": {
                    "type": "boolean",
                    "description": "Return aggregate stats instead of individual logs",
                    "default": False,
                },
                "window_days": {"type": "integer", "description": "Stats aggregation window (days)", "default": 7},
            },
            "required": [],
        },
    ),
    Tool(
        name="archivist_health_dashboard",
        description=(
            "Get a comprehensive health dashboard: memory counts, stale %, conflict rate, "
            "retrieval stats, skill health, cache status — all in one view."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "window_days": {"type": "integer", "description": "Analysis window in days", "default": 7},
            },
            "required": [],
        },
    ),
    Tool(
        name="archivist_batch_heuristic",
        description=(
            "Recommend a safe batch size based on memory health signals. "
            "Considers conflict rate, stale memory %, cache hit rate, and degraded skills. "
            "Inspired by Batch Size Gravity — when health degrades, use smaller batches."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "window_days": {"type": "integer", "description": "Analysis window in days", "default": 7},
            },
            "required": [],
        },
    ),
]

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_context_check(arguments: dict) -> list[TextContent]:
    """Token-count messages or memory texts against a budget."""
    from config import DEFAULT_CONTEXT_BUDGET
    from context_manager import check_context, check_memories_budget

    budget = arguments.get("budget_tokens", DEFAULT_CONTEXT_BUDGET)
    messages = arguments.get("messages")
    memory_texts = arguments.get("memory_texts")

    if messages:
        reserve = arguments.get("reserve_from_tail", 2000)
        result = check_context(messages, budget, reserve_from_tail=reserve)
    elif memory_texts:
        result = check_memories_budget(memory_texts, budget)
    else:
        result = {
            "total_tokens": 0,
            "budget_tokens": budget,
            "over_budget": False,
            "budget_used_pct": 0.0,
            "hint": "ok",
            "note": "Supply messages or memory_texts to count tokens.",
        }

    return success_response(result)


async def _handle_namespaces(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]
    namespaces = list_accessible_namespaces(agent_id)
    return success_response({
        "agent_id": agent_id,
        "default_namespace": get_namespace_for_agent(agent_id),
        "accessible_namespaces": namespaces,
    })


async def _handle_audit_trail(arguments: dict) -> list[TextContent]:
    from audit import get_audit_trail, get_agent_activity

    memory_id = arguments.get("memory_id", "")
    target_agent = arguments.get("target_agent", "")
    limit = arguments.get("limit", 50)

    if memory_id:
        entries = get_audit_trail(memory_id, limit=limit)
    elif target_agent:
        entries = get_agent_activity(target_agent, limit=limit)
    else:
        entries = get_agent_activity("", limit=limit)

    return success_response({"entries": entries}, default=str)


async def _handle_resolve_uri(arguments: dict) -> list[TextContent]:
    """Resolve an archivist:// URI to the underlying resource."""
    from archivist_uri import parse_uri

    uri = parse_uri(arguments["uri"])
    if not uri:
        return error_response({
            "error": "invalid_uri", "uri": arguments["uri"],
            "hint": "Format: archivist://{namespace}/{memory|entity|namespace|skill}/{id}",
        })

    agent_id = arguments.get("agent_id", "")

    if uri.is_memory:
        from .tools_search import _handle_deref
        return await _handle_deref({"memory_id": uri.resource_id, "agent_id": agent_id})

    if uri.is_entity:
        from .tools_search import _handle_recall
        return await _handle_recall({"entity": uri.resource_id, "agent_id": agent_id, "namespace": uri.namespace})

    if uri.is_namespace:
        from .tools_search import _handle_index
        return await _handle_index({"agent_id": agent_id, "namespace": uri.namespace})

    if uri.is_skill:
        skill = find_skill(uri.resource_id)
        if skill:
            health = get_skill_health(skill["id"])
            health["recent_lessons"] = get_lessons(skill["id"], limit=5)
            return success_response(health)
        return error_response({"error": "skill_not_found", "name": uri.resource_id})

    return error_response({"error": "unsupported_resource_type", "type": uri.resource_type})


async def _handle_retrieval_logs(arguments: dict) -> list[TextContent]:
    """Export retrieval logs or aggregate stats."""
    if arguments.get("stats_only"):
        stats = get_retrieval_stats(
            agent_id=arguments.get("agent_id", ""),
            window_days=arguments.get("window_days", 7),
        )
        return success_response(stats)

    logs = get_retrieval_logs(
        agent_id=arguments.get("agent_id", ""),
        limit=arguments.get("limit", 20),
        since=arguments.get("since", ""),
    )
    return success_response({"logs": logs, "count": len(logs)})


async def _handle_health_dashboard(arguments: dict) -> list[TextContent]:
    """Comprehensive health dashboard across all subsystems."""
    result = build_dashboard(window_days=arguments.get("window_days", 7))
    return success_response(result, default=str)


async def _handle_batch_heuristic(arguments: dict) -> list[TextContent]:
    """Recommend batch size from memory health signals."""
    result = batch_heuristic(window_days=arguments.get("window_days", 7))
    return success_response(result)


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, object] = {
    "archivist_context_check": _handle_context_check,
    "archivist_namespaces": _handle_namespaces,
    "archivist_audit_trail": _handle_audit_trail,
    "archivist_resolve_uri": _handle_resolve_uri,
    "archivist_retrieval_logs": _handle_retrieval_logs,
    "archivist_health_dashboard": _handle_health_dashboard,
    "archivist_batch_heuristic": _handle_batch_heuristic,
}
