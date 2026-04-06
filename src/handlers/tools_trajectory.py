"""MCP tool handlers — trajectory logging, annotations, ratings, tips, session end."""

import json
import logging

from mcp.types import Tool, TextContent

from trajectory import (
    log_trajectory, attribute_decisions, extract_tips, search_tips,
    add_annotation, get_annotations, add_rating, get_rating_summary,
    session_end_summary,
)

from ._common import error_response, success_response
from .tools_storage import _handle_store

logger = logging.getLogger("archivist.mcp")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="archivist_log_trajectory",
        description=(
            "Log an execution trajectory (task + actions + outcome). "
            "Optionally include memory_ids_used so outcome-aware retrieval can boost/warn linked memories. "
            "Auto-extracts actionable tips via LLM post-mortem."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent that executed the trajectory"},
                "task_description": {"type": "string", "description": "What the agent was trying to accomplish"},
                "actions": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Ordered list of actions taken, e.g. [{\"action\": \"search\", \"result\": \"found X\"}]",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "partial", "failure", "unknown"],
                    "description": "Overall outcome of the task",
                },
                "outcome_score": {"type": "number", "description": "Optional 0.0-1.0 quality score"},
                "memory_ids_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory point IDs that informed decisions during this trajectory",
                },
                "session_id": {"type": "string", "description": "Session grouping key (optional)", "default": ""},
            },
            "required": ["agent_id", "task_description", "actions", "outcome"],
        },
    ),
    Tool(
        name="archivist_annotate",
        description=(
            "Add a quality annotation or note to a memory point. "
            "Use to record observations about memory accuracy, relevance, or staleness."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Qdrant point ID to annotate"},
                "agent_id": {"type": "string", "description": "Annotating agent"},
                "content": {"type": "string", "description": "Annotation text"},
                "annotation_type": {
                    "type": "string",
                    "enum": ["note", "correction", "stale", "verified", "quality"],
                    "description": "Type of annotation",
                    "default": "note",
                },
                "quality_score": {"type": "number", "description": "Optional 0.0-1.0 quality assessment"},
            },
            "required": ["memory_id", "agent_id", "content"],
        },
    ),
    Tool(
        name="archivist_rate",
        description=(
            "Rate a memory as helpful (+1) or unhelpful (-1). "
            "Ratings feed into analytics and can inform future retrieval ranking."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Qdrant point ID to rate"},
                "agent_id": {"type": "string", "description": "Rating agent"},
                "rating": {"type": "integer", "enum": [-1, 1], "description": "+1 (helpful) or -1 (unhelpful)"},
                "context": {"type": "string", "description": "Optional context for the rating", "default": ""},
            },
            "required": ["memory_id", "agent_id", "rating"],
        },
    ),
    Tool(
        name="archivist_tips",
        description=(
            "Retrieve actionable tips extracted from past trajectories. "
            "Tips are categorized as strategy, recovery, or optimization."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent whose tips to retrieve"},
                "category": {
                    "type": "string",
                    "enum": ["strategy", "recovery", "optimization", ""],
                    "description": "Filter by tip category (optional)",
                    "default": "",
                },
                "limit": {"type": "integer", "description": "Max tips to return", "default": 10},
            },
            "required": ["agent_id"],
        },
    ),
    Tool(
        name="archivist_session_end",
        description=(
            "Summarize a work session into a durable memory. "
            "Aggregates all trajectories logged under a session_id and produces a concise summary."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent whose session to summarize"},
                "session_id": {"type": "string", "description": "Session identifier to summarize"},
                "store_as_memory": {
                    "type": "boolean",
                    "description": "If true, also store the summary as a durable memory via archivist_store",
                    "default": True,
                },
            },
            "required": ["agent_id", "session_id"],
        },
    ),
]

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_log_trajectory(arguments: dict) -> list[TextContent]:
    """Log a trajectory, run attribution + tip extraction."""
    result = await log_trajectory(
        agent_id=arguments["agent_id"],
        task_description=arguments["task_description"],
        actions=arguments["actions"],
        outcome=arguments["outcome"],
        outcome_score=arguments.get("outcome_score"),
        memory_ids_used=arguments.get("memory_ids_used"),
        session_id=arguments.get("session_id", ""),
        metadata=arguments.get("metadata"),
    )
    traj_id = result["trajectory_id"]

    mem_ids = arguments.get("memory_ids_used")
    attributions = []
    if mem_ids:
        try:
            attributions = await attribute_decisions(traj_id)
        except Exception as e:
            logger.warning("Attribution failed for trajectory %s: %s", traj_id, e)

    tips = []
    try:
        tips = await extract_tips(traj_id)
    except Exception as e:
        logger.warning("Tip extraction failed for trajectory %s: %s", traj_id, e)

    from audit import log_memory_event
    await log_memory_event(
        agent_id=arguments["agent_id"],
        action="log_trajectory",
        memory_id=traj_id,
        namespace="",
        text_hash="",
        metadata={"outcome": arguments["outcome"], "tips_extracted": len(tips)},
    )

    return success_response({**result, "attributions": attributions, "tips": tips}, default=str)


async def _handle_annotate(arguments: dict) -> list[TextContent]:
    """Add a quality annotation to a memory point."""
    ann_id = add_annotation(
        memory_id=arguments["memory_id"],
        agent_id=arguments["agent_id"],
        content=arguments["content"],
        annotation_type=arguments.get("annotation_type", "note"),
        quality_score=arguments.get("quality_score"),
    )

    from audit import log_memory_event
    await log_memory_event(
        agent_id=arguments["agent_id"],
        action="annotate",
        memory_id=arguments["memory_id"],
        namespace="",
        text_hash="",
        metadata={"annotation_id": ann_id, "type": arguments.get("annotation_type", "note")},
    )

    return success_response({
        "annotation_id": ann_id,
        "memory_id": arguments["memory_id"],
        "annotations": get_annotations(arguments["memory_id"]),
    }, default=str)


async def _handle_rate(arguments: dict) -> list[TextContent]:
    """Rate a memory as helpful or unhelpful."""
    rating_id = add_rating(
        memory_id=arguments["memory_id"],
        agent_id=arguments["agent_id"],
        rating=arguments["rating"],
        context=arguments.get("context", ""),
    )

    from audit import log_memory_event
    await log_memory_event(
        agent_id=arguments["agent_id"],
        action="rate",
        memory_id=arguments["memory_id"],
        namespace="",
        text_hash="",
        metadata={"rating_id": rating_id, "rating": arguments["rating"]},
    )

    summary = get_rating_summary(arguments["memory_id"])
    return success_response({"rating_id": rating_id, **summary})


async def _handle_tips(arguments: dict) -> list[TextContent]:
    """Retrieve tips from past trajectories."""
    tips = search_tips(
        agent_id=arguments["agent_id"],
        category=arguments.get("category", ""),
        limit=arguments.get("limit", 10),
    )
    return success_response({
        "agent_id": arguments["agent_id"],
        "tips": tips,
        "count": len(tips),
    }, default=str)


async def _handle_session_end(arguments: dict) -> list[TextContent]:
    """Summarize a session and optionally store as durable memory."""
    result = await session_end_summary(
        agent_id=arguments["agent_id"],
        session_id=arguments["session_id"],
    )

    if result.get("error"):
        return error_response(result)

    store = arguments.get("store_as_memory", True)
    stored_id = None
    if store and result.get("summary"):
        store_result = await _handle_store({
            "text": f"[Session summary — {arguments['session_id']}]\n{result['summary']}",
            "agent_id": arguments["agent_id"],
            "importance_score": 0.8,
        })
        try:
            stored_data = json.loads(store_result[0].text)
            stored_id = stored_data.get("memory_id")
        except Exception:
            pass

    result["stored_memory_id"] = stored_id
    return success_response(result)


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, object] = {
    "archivist_log_trajectory": _handle_log_trajectory,
    "archivist_annotate": _handle_annotate,
    "archivist_rate": _handle_rate,
    "archivist_tips": _handle_tips,
    "archivist_session_end": _handle_session_end,
}
