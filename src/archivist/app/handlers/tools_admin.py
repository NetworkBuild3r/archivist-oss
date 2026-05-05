"""MCP tool handlers — admin, observability, namespaces, audit, URI resolution."""

import logging

from mcp.types import TextContent, Tool

from archivist.app.dashboard import batch_heuristic, build_dashboard
from archivist.core.rbac import get_namespace_for_agent, list_accessible_namespaces
from archivist.features.skills import find_skill, get_lessons, get_skill_health
from archivist.retrieval.retrieval_log import get_retrieval_logs, get_retrieval_stats
from archivist.storage.sqlite_pool import pool

from ._common import error_response, success_response

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
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                            },
                            {"type": "string"},
                        ],
                    },
                    "description": "Chat messages to count tokens for (alternative to memory_texts). Each item can be a {role, content} object or a plain string.",
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
                "memory_id": {
                    "type": "string",
                    "description": "Specific memory ID to audit (optional)",
                    "default": "",
                },
                "target_agent": {
                    "type": "string",
                    "description": "Agent whose activity to view (optional)",
                    "default": "",
                },
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
                "uri": {
                    "type": "string",
                    "description": (
                        "An archivist:// URI to resolve. "
                        "Format: archivist://{namespace}/{resource_type}/{id} "
                        "where resource_type is one of: memory, entity, namespace, skill. "
                        "Examples: archivist://agents-nova/memory/abc123, "
                        "archivist://shared/entity/42, archivist://agents-nova/skill/web_search"
                    ),
                },
                "agent_id": {
                    "type": "string",
                    "description": "Calling agent for RBAC",
                    "default": "",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Original caller agent (overrides agent_id for RBAC)",
                    "default": "",
                },
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
                "agent_id": {
                    "type": "string",
                    "description": "Filter by agent (optional)",
                    "default": "",
                },
                "limit": {"type": "integer", "description": "Max entries", "default": 20},
                "since": {
                    "type": "string",
                    "description": "ISO datetime lower bound (optional)",
                    "default": "",
                },
                "stats_only": {
                    "type": "boolean",
                    "description": "Return aggregate stats instead of individual logs",
                    "default": False,
                },
                "window_days": {
                    "type": "integer",
                    "description": "Stats aggregation window (days)",
                    "default": 7,
                },
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
                "window_days": {
                    "type": "integer",
                    "description": "Analysis window in days",
                    "default": 7,
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="archivist_batch_heuristic",
        description=(
            "Recommend a safe batch size based on memory health signals. "
            "Considers conflict rate, stale memory %, cache hit rate, and degraded skills. "
            "When health degrades, use smaller batches."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "window_days": {
                    "type": "integer",
                    "description": "Analysis window in days",
                    "default": 7,
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="archivist_savings_dashboard",
        description=(
            "Token savings and tier-distribution observability dashboard. "
            "Shows avg/min/max savings %, total tokens saved vs naive full-L2 baseline, "
            "per-policy breakdown (adaptive/l0_first/l2_first), and a hotness heatmap "
            "of top-N most frequently retrieved memories. "
            "Use to measure the efficiency of the Answer Finder pipeline over time."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "window_days": {
                    "type": "integer",
                    "description": "Analysis window in days (default 7).",
                    "default": 7,
                },
                "heatmap_top_n": {
                    "type": "integer",
                    "description": "Number of top memories to include in hotness heatmap (default 50).",
                    "default": 50,
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="archivist_backup",
        description=(
            "Create, list, or restore memory snapshots. Snapshots include Qdrant vectors "
            "and SQLite graph data for disaster recovery and agent migration."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "restore", "delete", "export_agent", "import_agent"],
                    "description": "Action to perform",
                },
                "snapshot_id": {
                    "type": "string",
                    "description": "Snapshot ID (required for restore/delete)",
                    "default": "",
                },
                "label": {
                    "type": "string",
                    "description": "Optional label for the snapshot (used with create)",
                    "default": "",
                },
                "target": {
                    "type": "string",
                    "enum": ["all", "qdrant", "sqlite"],
                    "description": "What to restore (default: all)",
                    "default": "all",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Agent ID for export_agent action",
                    "default": "",
                },
                "file": {
                    "type": "string",
                    "description": "NDJSON file path for import_agent action",
                    "default": "",
                },
            },
            "required": ["action"],
        },
    ),
]

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_context_check(arguments: dict) -> list[TextContent]:
    """Token-count messages or memory texts against a budget."""
    from archivist.core.config import DEFAULT_CONTEXT_BUDGET
    from archivist.utils.context_manager import check_context, check_memories_budget

    budget = arguments.get("budget_tokens", DEFAULT_CONTEXT_BUDGET)
    messages = arguments.get("messages")
    memory_texts = arguments.get("memory_texts")

    if messages:
        messages = [
            m if isinstance(m, dict) else {"role": "user", "content": str(m)} for m in messages
        ]
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
    return success_response(
        {
            "agent_id": agent_id,
            "default_namespace": get_namespace_for_agent(agent_id),
            "accessible_namespaces": namespaces,
        }
    )


async def _handle_audit_trail(arguments: dict) -> list[TextContent]:
    from archivist.core.audit import get_agent_activity, get_audit_trail

    memory_id = arguments.get("memory_id", "")
    target_agent = arguments.get("target_agent", "")
    limit = arguments.get("limit", 50)

    if memory_id:
        entries = await get_audit_trail(memory_id, limit=limit)
    elif target_agent:
        entries = await get_agent_activity(target_agent, limit=limit)
    else:
        entries = await get_agent_activity("", limit=limit)

    return success_response({"entries": entries}, default=str)


async def _handle_resolve_uri(arguments: dict) -> list[TextContent]:
    """Resolve an archivist:// URI to the underlying resource."""
    from archivist.core.archivist_uri import parse_uri

    raw_uri = arguments["uri"]
    uri = parse_uri(raw_uri)
    if not uri:
        parts = raw_uri.split("/")
        diag = (
            "URI must start with 'archivist://'"
            if not raw_uri.startswith("archivist://")
            else (
                f"Could not parse resource_type from URI (got {len(parts)} segments, expected at least 5)"
            )
        )
        return error_response(
            {
                "error": "invalid_uri",
                "uri": raw_uri,
                "diagnostic": diag,
                "expected_format": "archivist://{namespace}/{resource_type}/{resource_id}",
                "valid_resource_types": ["memory", "entity", "namespace", "skill"],
                "examples": [
                    "archivist://agents-nova/memory/abc123-def456",
                    "archivist://shared/entity/42",
                    "archivist://agents-nova/skill/web_search",
                ],
            }
        )

    agent_id = arguments.get("agent_id", "")
    caller_agent_id = arguments.get("caller_agent_id", "")

    if uri.is_memory:
        from .tools_search import _handle_deref

        return await _handle_deref(
            {"memory_id": uri.resource_id, "agent_id": agent_id, "caller_agent_id": caller_agent_id}
        )

    if uri.is_entity:
        from .tools_search import _handle_recall

        return await _handle_recall(
            {
                "entity": uri.resource_id,
                "agent_id": agent_id,
                "caller_agent_id": caller_agent_id,
                "namespace": uri.namespace,
            }
        )

    if uri.is_namespace:
        from .tools_search import _handle_index

        return await _handle_index(
            {"agent_id": agent_id, "caller_agent_id": caller_agent_id, "namespace": uri.namespace}
        )

    if uri.is_skill:
        skill = await find_skill(uri.resource_id)
        if skill:
            health = await get_skill_health(skill["id"])
            health["recent_lessons"] = await get_lessons(skill["id"], limit=5)
            return success_response(health)
        return error_response({"error": "skill_not_found", "name": uri.resource_id})

    return error_response({"error": "unsupported_resource_type", "type": uri.resource_type})


async def _handle_retrieval_logs(arguments: dict) -> list[TextContent]:
    """Export retrieval logs or aggregate stats."""
    if arguments.get("stats_only"):
        stats = await get_retrieval_stats(
            agent_id=arguments.get("agent_id", ""),
            window_days=arguments.get("window_days", 7),
        )
        return success_response(stats)

    logs = await get_retrieval_logs(
        agent_id=arguments.get("agent_id", ""),
        limit=arguments.get("limit", 20),
        since=arguments.get("since", ""),
    )
    return success_response({"logs": logs, "count": len(logs)})


async def _handle_health_dashboard(arguments: dict) -> list[TextContent]:
    """Comprehensive health dashboard across all subsystems."""
    result = await build_dashboard(window_days=arguments.get("window_days", 7))
    return success_response(result, default=str)


async def _handle_savings_dashboard(arguments: dict) -> list[TextContent]:
    """Token savings + tier distribution + hotness heatmap dashboard."""
    from archivist.app.dashboard import (
        _hotness_heatmap,
        _tier_distribution_stats,
        _token_savings_stats,
    )

    window_days = int(arguments.get("window_days") or 7)
    top_n = int(arguments.get("heatmap_top_n") or 50)

    async with pool.read() as conn:
        savings = await _token_savings_stats(conn, window_days)
        tier_dist = await _tier_distribution_stats(conn, window_days)

    heatmap = await _hotness_heatmap(top_n=top_n)

    return success_response(
        {
            "window_days": window_days,
            "token_savings": savings,
            "tier_distribution": tier_dist,
            "hotness_heatmap": heatmap,
            "heatmap_count": len(heatmap),
        },
        default=str,
    )


async def _handle_batch_heuristic(arguments: dict) -> list[TextContent]:
    """Recommend batch size from memory health signals."""
    result = await batch_heuristic(window_days=arguments.get("window_days", 7))
    return success_response(result)


async def _handle_backup(arguments: dict) -> list[TextContent]:
    """Create, list, restore, or delete memory snapshots."""
    from archivist.storage.backup_manager import (
        create_snapshot,
        delete_snapshot,
        export_agent,
        import_agent,
        list_snapshots,
        prune_snapshots,
        restore_snapshot,
    )

    action = arguments.get("action", "")

    if action == "create":
        label = arguments.get("label", "")
        result = create_snapshot(label=label)
        prune_snapshots()
        return success_response(result)

    if action == "list":
        snapshots = list_snapshots()
        return success_response({"snapshots": snapshots, "count": len(snapshots)})

    if action == "restore":
        snapshot_id = arguments.get("snapshot_id", "").strip()
        if not snapshot_id:
            return error_response({"error": "snapshot_id is required for restore"})
        target = arguments.get("target", "all")
        try:
            result = restore_snapshot(snapshot_id, target=target)
            return success_response(result)
        except (FileNotFoundError, ValueError) as e:
            return error_response({"error": str(e)})

    if action == "delete":
        snapshot_id = arguments.get("snapshot_id", "").strip()
        if not snapshot_id:
            return error_response({"error": "snapshot_id is required for delete"})
        if delete_snapshot(snapshot_id):
            return success_response({"deleted": snapshot_id})
        return error_response({"error": "snapshot not found"})

    if action == "export_agent":
        agent_id = arguments.get("agent_id", "").strip()
        if not agent_id:
            return error_response({"error": "agent_id is required for export_agent"})
        result = export_agent(agent_id)
        return success_response(result)

    if action == "import_agent":
        file_path = arguments.get("file", "").strip()
        if not file_path:
            return error_response({"error": "file path is required for import_agent"})
        try:
            result = import_agent(file_path)
            return success_response(result)
        except FileNotFoundError as e:
            return error_response({"error": str(e)})

    return error_response({"error": f"Unknown action: {action}"})


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
    "archivist_savings_dashboard": _handle_savings_dashboard,
    "archivist_backup": _handle_backup,
}
