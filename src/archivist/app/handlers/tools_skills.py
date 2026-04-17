"""MCP tool handlers — skill registry, events, lessons, health, relations."""

import logging

from mcp.types import Tool, TextContent

from archivist.features.skills import (
    register_skill, record_version, add_lesson, get_lessons,
    log_skill_event, get_skill_health, find_skill,
    add_skill_relation, get_skill_relations, get_skill_substitutes,
)
import archivist.core.metrics as m
import archivist.features.webhooks as webhooks

from ._common import error_response, success_response

logger = logging.getLogger("archivist.mcp")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="archivist_register_skill",
        description=(
            "Register a new skill (MCP tool) or update an existing one. "
            "Tracks provider, endpoint, version. Use when discovering a new tool or detecting a version change."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill/tool name"},
                "provider": {"type": "string", "description": "Provider or manufacturer (e.g. 'openai', 'internal')", "default": ""},
                "mcp_endpoint": {"type": "string", "description": "MCP server endpoint URL (optional)", "default": ""},
                "version": {"type": "string", "description": "Current version string", "default": "0.0.0"},
                "description": {"type": "string", "description": "What this skill does", "default": ""},
                "agent_id": {"type": "string", "description": "Agent registering this skill"},
                "changelog": {"type": "string", "description": "What changed in this version (optional)", "default": ""},
                "breaking_changes": {"type": "string", "description": "Known breaking changes (optional)", "default": ""},
            },
            "required": ["name", "agent_id"],
        },
    ),
    Tool(
        name="archivist_skill_event",
        description=(
            "Log a skill usage event — invocation, success, failure. "
            "Feeds into skill health scoring and connects to trajectories."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Skill name (looked up in registry)"},
                "provider": {"type": "string", "description": "Provider to disambiguate (optional)", "default": ""},
                "agent_id": {"type": "string", "description": "Agent that used the skill"},
                "outcome": {
                    "type": "string",
                    "enum": ["success", "partial", "failure"],
                    "description": "Outcome of the invocation",
                },
                "skill_version": {"type": "string", "description": "Version used (optional, defaults to current)", "default": ""},
                "duration_ms": {"type": "integer", "description": "Execution time in milliseconds (optional)"},
                "error_message": {"type": "string", "description": "Error details if failed", "default": ""},
                "trajectory_id": {"type": "string", "description": "Link to a trajectory (optional)", "default": ""},
            },
            "required": ["skill_name", "agent_id", "outcome"],
        },
    ),
    Tool(
        name="archivist_skill_lesson",
        description=(
            "Add a lesson learned to a skill — failure modes, workarounds, best usage patterns. "
            "Each skill accumulates operational knowledge over time."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Skill name"},
                "provider": {"type": "string", "description": "Provider to disambiguate (optional)", "default": ""},
                "title": {"type": "string", "description": "Short title for the lesson"},
                "content": {"type": "string", "description": "Full lesson content — failure mode, workaround, or usage tip"},
                "lesson_type": {
                    "type": "string",
                    "enum": ["failure_mode", "workaround", "best_practice", "breaking_change", "general"],
                    "description": "Category of lesson",
                    "default": "general",
                },
                "skill_version": {"type": "string", "description": "Version this lesson applies to (optional)", "default": ""},
                "agent_id": {"type": "string", "description": "Agent contributing this lesson"},
            },
            "required": ["skill_name", "title", "content", "agent_id"],
        },
    ),
    Tool(
        name="archivist_skill_health",
        description=(
            "Get health and operational status for a skill — success rate, recent failures, "
            "lessons learned count, version history. Use before invoking an unreliable skill."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Skill name to check"},
                "provider": {"type": "string", "description": "Provider to disambiguate (optional)", "default": ""},
                "window_days": {"type": "integer", "description": "How many days of history to analyze", "default": 30},
                "include_lessons": {"type": "boolean", "description": "Include recent lessons in response", "default": True},
            },
            "required": ["skill_name"],
        },
    ),
    Tool(
        name="archivist_skill_relate",
        description=(
            "Create or update a relation between two skills. "
            "Relation types: similar_to, depend_on, compose_with, replaced_by."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "skill_a": {"type": "string", "description": "First skill name"},
                "skill_b": {"type": "string", "description": "Second skill name"},
                "relation_type": {
                    "type": "string",
                    "enum": ["similar_to", "depend_on", "compose_with", "replaced_by"],
                    "description": "Type of relationship",
                },
                "confidence": {"type": "number", "description": "Confidence 0-1", "default": 1.0},
                "evidence": {"type": "string", "description": "Why this relation exists", "default": ""},
                "agent_id": {"type": "string", "description": "Agent creating this relation"},
                "provider_a": {"type": "string", "description": "Provider for skill_a (optional)", "default": ""},
                "provider_b": {"type": "string", "description": "Provider for skill_b (optional)", "default": ""},
            },
            "required": ["skill_a", "skill_b", "relation_type", "agent_id"],
        },
    ),
    Tool(
        name="archivist_skill_dependencies",
        description=(
            "Return the dependency/relation graph for a skill. "
            "Shows substitutes, dependencies, and composable skills."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Skill to get relations for"},
                "provider": {"type": "string", "description": "Provider to disambiguate (optional)", "default": ""},
                "depth": {"type": "integer", "description": "Graph traversal depth (1=direct)", "default": 1},
            },
            "required": ["skill_name"],
        },
    ),
]

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _resolve_skill(name: str, provider: str = "") -> dict | None:
    return find_skill(name, provider)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_register_skill(arguments: dict) -> list[TextContent]:
    name = arguments["name"]
    agent_id = arguments["agent_id"]
    version = arguments.get("version", "0.0.0")
    changelog = arguments.get("changelog", "")
    breaking_changes = arguments.get("breaking_changes", "")

    result = register_skill(
        name=name,
        provider=arguments.get("provider", ""),
        mcp_endpoint=arguments.get("mcp_endpoint", ""),
        version=version,
        description=arguments.get("description", ""),
        registered_by=agent_id,
    )

    if changelog or breaking_changes:
        record_version(
            skill_id=result["skill_id"],
            version=version,
            changelog=changelog,
            breaking_changes=breaking_changes,
            reported_by=agent_id,
        )

    return success_response(result)


async def _handle_skill_event(arguments: dict) -> list[TextContent]:
    skill = _resolve_skill(arguments["skill_name"], arguments.get("provider", ""))
    if not skill:
        return error_response({
            "error": "skill_not_found",
            "skill_name": arguments["skill_name"],
            "hint": "Register the skill first with archivist_register_skill",
        })

    event_id = log_skill_event(
        skill_id=skill["id"],
        agent_id=arguments["agent_id"],
        outcome=arguments["outcome"],
        skill_version=arguments.get("skill_version", ""),
        duration_ms=arguments.get("duration_ms"),
        error_message=arguments.get("error_message", ""),
        trajectory_id=arguments.get("trajectory_id", ""),
    )

    m.inc(m.SKILL_EVENT, {"outcome": arguments["outcome"]})
    webhooks.fire_background("skill_event", {
        "skill_name": skill["name"], "outcome": arguments["outcome"],
        "agent_id": arguments["agent_id"],
    })

    return success_response({
        "event_id": event_id,
        "skill_id": skill["id"],
        "skill_name": skill["name"],
        "outcome": arguments["outcome"],
    })


async def _handle_skill_lesson(arguments: dict) -> list[TextContent]:
    skill = _resolve_skill(arguments["skill_name"], arguments.get("provider", ""))
    if not skill:
        return error_response({
            "error": "skill_not_found",
            "skill_name": arguments["skill_name"],
            "hint": "Register the skill first with archivist_register_skill",
        })

    lesson_id = add_lesson(
        skill_id=skill["id"],
        title=arguments["title"],
        content=arguments["content"],
        lesson_type=arguments.get("lesson_type", "general"),
        skill_version=arguments.get("skill_version", ""),
        agent_id=arguments["agent_id"],
    )

    return success_response({
        "lesson_id": lesson_id,
        "skill_id": skill["id"],
        "skill_name": skill["name"],
        "title": arguments["title"],
    })


async def _handle_skill_health(arguments: dict) -> list[TextContent]:
    skill = _resolve_skill(arguments["skill_name"], arguments.get("provider", ""))
    if not skill:
        return error_response({
            "error": "skill_not_found",
            "skill_name": arguments["skill_name"],
        })

    health = get_skill_health(
        skill_id=skill["id"],
        window_days=arguments.get("window_days", 30),
    )

    if arguments.get("include_lessons", True):
        health["recent_lessons"] = get_lessons(skill["id"], limit=10)

    health["related_skills"] = [
        {"name": s["name"], "relation": s["relation_type"], "confidence": s["confidence"]}
        for s in get_skill_substitutes(skill["id"])
    ]

    return success_response(health)


async def _handle_skill_relate(arguments: dict) -> list[TextContent]:
    """Create or update a relation between two skills."""
    skill_a = _resolve_skill(arguments["skill_a"], arguments.get("provider_a", ""))
    skill_b = _resolve_skill(arguments["skill_b"], arguments.get("provider_b", ""))

    if not skill_a:
        return error_response({
            "error": "skill_not_found",
            "skill_name": arguments["skill_a"],
            "hint": f"Skill '{arguments['skill_a']}' is not registered. Use archivist_register_skill to create it first.",
        })
    if not skill_b:
        return error_response({
            "error": "skill_not_found",
            "skill_name": arguments["skill_b"],
            "hint": f"Skill '{arguments['skill_b']}' is not registered. Use archivist_register_skill to create it first.",
        })

    rel_id = add_skill_relation(
        skill_a_id=skill_a["id"],
        skill_b_id=skill_b["id"],
        relation_type=arguments["relation_type"],
        confidence=arguments.get("confidence", 1.0),
        evidence=arguments.get("evidence", ""),
        created_by=arguments["agent_id"],
    )

    return success_response({
        "relation_id": rel_id,
        "skill_a": skill_a["name"],
        "skill_b": skill_b["name"],
        "relation_type": arguments["relation_type"],
    })


async def _handle_skill_dependencies(arguments: dict) -> list[TextContent]:
    """Return the dependency/relation graph for a skill."""
    skill = _resolve_skill(arguments["skill_name"], arguments.get("provider", ""))
    if not skill:
        return error_response({
            "error": "skill_not_found",
            "skill_name": arguments["skill_name"],
            "hint": f"Skill '{arguments['skill_name']}' is not registered. Use archivist_register_skill to create it first.",
        })

    depth = arguments.get("depth", 1)
    relations = get_skill_relations(skill["id"], depth=depth)
    substitutes = get_skill_substitutes(skill["id"])

    return success_response({
        "skill": skill["name"],
        "skill_id": skill["id"],
        "relations": relations,
        "substitutes": [{"name": s["name"], "relation": s["relation_type"], "confidence": s["confidence"]} for s in substitutes],
        "depth": depth,
    }, default=str)


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, object] = {
    "archivist_register_skill": _handle_register_skill,
    "archivist_skill_event": _handle_skill_event,
    "archivist_skill_lesson": _handle_skill_lesson,
    "archivist_skill_health": _handle_skill_health,
    "archivist_skill_relate": _handle_skill_relate,
    "archivist_skill_dependencies": _handle_skill_dependencies,
}
