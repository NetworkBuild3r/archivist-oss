"""Archivist MCP server — exposes memory tools via HTTP SSE (Model Context Protocol)."""

import asyncio
import logging
import json
import hashlib
import uuid
from datetime import datetime, timezone

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rlm_retriever import recursive_retrieve, search_vectors
from graph import (
    search_entities, get_entity_facts, get_entity_relationships,
    upsert_entity, add_fact, init_schema, get_db,
)
from embeddings import embed_text
from config import (
    QDRANT_URL, QDRANT_COLLECTION, TEAM_MAP,
    CONFLICT_CHECK_ON_STORE, CONFLICT_BLOCK_ON_STORE,
)
from conflict_detection import check_for_conflicts, llm_adjudicated_dedup
from rbac import (
    check_access, get_namespace_for_agent, list_accessible_namespaces,
    get_namespace_config, can_read_agent_memory, is_permissive_mode,
    filter_agents_for_read,
)
from compressed_index import build_namespace_index
from graph_retrieval import detect_contradictions
from trajectory import (
    log_trajectory, attribute_decisions, extract_tips, search_tips,
    add_annotation, get_annotations, add_rating, get_rating_summary,
    session_end_summary,
)
from skills import (
    register_skill, record_version, add_lesson, get_lessons,
    log_skill_event, get_skill_health, find_skill, list_skills,
    update_skill_status, add_skill_relation, get_skill_relations,
    get_skill_substitutes,
)
import hot_cache
from archivist_uri import memory_uri, entity_uri, namespace_uri
from retrieval_log import get_retrieval_logs, get_retrieval_stats
import metrics as m
import webhooks
from dashboard import build_dashboard, batch_heuristic
import curator_queue

logger = logging.getLogger("archivist.mcp")

server = Server("archivist")


def _rbac_gate(agent_id: str, action: str, namespace: str) -> str | None:
    """Return error JSON string if access denied, None if allowed."""
    policy = check_access(agent_id, action, namespace)
    if not policy.allowed:
        return json.dumps({"error": "access_denied", "reason": policy.reason})
    return None


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="archivist_search",
            description=(
                "Semantic search across agent memories with RLM recursive retrieval. "
                "Supports fleet-wide search or a list of agent_ids (multi-agent memory). "
                "Set caller_agent_id when reading other agents' memories so RBAC can apply. "
                "Returns synthesized answer with source citations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "agent_id": {"type": "string", "description": "Filter to one agent's memories (optional)", "default": ""},
                    "agent_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search only these agents' memories (OR). Omit for fleet-wide or use agent_id for a single agent.",
                    },
                    "caller_agent_id": {
                        "type": "string",
                        "description": "Identity of the invoking agent — used for RBAC when reading others' namespaces. Defaults to agent_id if set.",
                    },
                    "namespace": {"type": "string", "description": "Memory namespace to search (optional, auto-detect from agent_id)", "default": ""},
                    "team": {"type": "string", "description": "Filter by team (optional)", "default": ""},
                    "refine": {"type": "boolean", "description": "Use LLM refinement for higher quality (slower). Default true.", "default": True},
                    "limit": {"type": "integer", "description": "Max chunks to refine/synthesize after retrieval", "default": 20},
                    "min_score": {
                        "type": "number",
                        "description": "Minimum vector similarity (0–1). Overrides RETRIEVAL_THRESHOLD for this call; omit to use env default. Use 0 to disable filtering.",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["l0", "l1", "l2"],
                        "description": "Context tier: l0 (abstract ~100 tokens), l1 (overview ~500 tokens), l2 (full detail). Default l2.",
                        "default": "l2",
                    },
                    "date_from": {"type": "string", "description": "ISO date lower bound (inclusive), e.g. 2026-01-01", "default": ""},
                    "date_to": {"type": "string", "description": "ISO date upper bound (inclusive)", "default": ""},
                    "max_tokens": {
                        "type": "integer",
                        "description": "Approximate token budget for context returned (caps chunk count). Omit for unlimited.",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["experience", "skill", "general", ""],
                        "description": "Filter by memory type: experience, skill, or general. Omit for all types.",
                        "default": "",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="archivist_recall",
            description=(
                "Graph-based multi-hop recall. Finds entities and their relationships/facts. "
                "Use for questions like 'What do we know about X?' or 'How does X relate to Y?'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity name to look up"},
                    "related_to": {"type": "string", "description": "Optional second entity to find connections", "default": ""},
                    "agent_id": {"type": "string", "description": "Calling agent for RBAC (optional)", "default": ""},
                    "caller_agent_id": {
                        "type": "string",
                        "description": "Identity for read access checks (defaults to agent_id). Required when RBAC namespaces are configured.",
                    },
                    "namespace": {"type": "string", "description": "Memory namespace scope (optional)", "default": ""},
                },
                "required": ["entity"],
            },
        ),
        Tool(
            name="archivist_store",
            description=(
                "Explicitly store a memory/fact with entity extraction. "
                "Use when you want to ensure something specific is remembered."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The memory or fact to store"},
                    "agent_id": {"type": "string", "description": "Which agent is storing this"},
                    "namespace": {"type": "string", "description": "Target namespace (default: auto-detect from agent_id)", "default": ""},
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity names mentioned (optional, will auto-extract if empty)",
                        "default": [],
                    },
                    "importance_score": {"type": "number", "description": "0.0-1.0 importance score (higher = longer retention)", "default": 0.5},
                    "memory_type": {
                        "type": "string",
                        "enum": ["experience", "skill", "general"],
                        "description": "Tag this memory as an experience (I did X), skill (how to do X), or general. Default general.",
                        "default": "general",
                    },
                    "force_skip_conflict_check": {
                        "type": "boolean",
                        "description": "If true, skip vector similarity conflict check against other agents' memories (use sparingly).",
                        "default": False,
                    },
                },
                "required": ["text", "agent_id"],
            },
        ),
        Tool(
            name="archivist_timeline",
            description=(
                "Get a chronological timeline of memories about a topic. "
                "Use for questions like 'What happened with X over the last 2 weeks?'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to build timeline for"},
                    "agent_id": {"type": "string", "description": "Filter to specific agent (optional)", "default": ""},
                    "caller_agent_id": {
                        "type": "string",
                        "description": "Invoker identity for RBAC (defaults to agent_id when set).",
                    },
                    "namespace": {"type": "string", "description": "Memory namespace to search (optional)", "default": ""},
                    "days": {"type": "integer", "description": "How many days back to search", "default": 14},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="archivist_insights",
            description=(
                "Get curated cross-agent insights for a topic. "
                "Searches across all accessible namespaces to find shared knowledge and patterns."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to get insights on"},
                    "agent_id": {"type": "string", "description": "Calling agent for RBAC (optional)", "default": ""},
                    "caller_agent_id": {
                        "type": "string",
                        "description": "Invoker identity for RBAC when reading cross-agent insights (defaults to agent_id).",
                    },
                    "namespace": {"type": "string", "description": "Namespace scope (optional)", "default": ""},
                    "limit": {"type": "integer", "description": "Max insights to return", "default": 10},
                },
                "required": ["topic"],
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
            name="archivist_merge",
            description=(
                "Merge conflicting memory entries using a specified strategy. "
                "Strategies: latest (keep newest), concat (join all), semantic (LLM synthesis), manual (flag for review)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Calling agent"},
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Memory point IDs to merge",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["latest", "concat", "semantic", "manual"],
                        "description": "Merge strategy",
                    },
                    "namespace": {"type": "string", "description": "Namespace for the merged result", "default": ""},
                },
                "required": ["agent_id", "memory_ids", "strategy"],
            },
        ),
        Tool(
            name="archivist_deref",
            description=(
                "Dereference a memory point by ID. Returns the full L2 text and metadata for a specific memory. "
                "Use after an L0/L1 search to get full detail on a relevant hit."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Qdrant point ID to retrieve"},
                    "agent_id": {"type": "string", "description": "Calling agent for RBAC", "default": ""},
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="archivist_index",
            description=(
                "Get a compressed navigational index of what knowledge exists in a namespace. "
                "Returns a short text (~500 tokens) listing entity categories and top topics — "
                "useful for cross-domain bridging and deciding what to search for."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Calling agent for RBAC and default namespace resolution"},
                    "namespace": {"type": "string", "description": "Namespace to index (default: auto-detect)", "default": ""},
                },
                "required": ["agent_id"],
            },
        ),
        Tool(
            name="archivist_contradictions",
            description=(
                "Surface contradicting facts about an entity from different agents. "
                "Uses heuristic keyword detection on the knowledge graph."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity name to check for contradictions"},
                    "agent_id": {"type": "string", "description": "Calling agent for RBAC", "default": ""},
                },
                "required": ["entity"],
            },
        ),
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
            name="archivist_cache_stats",
            description="Return hot cache statistics: entries per agent, TTL, hit rate.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="archivist_cache_invalidate",
            description=(
                "Manually invalidate the hot cache. Useful after bulk writes or migrations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Invalidate entries for this namespace (optional)", "default": ""},
                    "agent_id": {"type": "string", "description": "Invalidate entries for this agent (optional)", "default": ""},
                    "all": {"type": "boolean", "description": "Invalidate entire cache", "default": False},
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
        Tool(
            name="archivist_compress",
            description=(
                "Archive memory blocks and return a compact summary. "
                "Agents call this mid-session to manage context budget. "
                "Originals are archived (kept but excluded from default search)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Agent requesting compression"},
                    "namespace": {"type": "string", "description": "Target namespace"},
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Memory point IDs to compress",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Optional agent-provided summary. If omitted, LLM generates one.",
                        "default": "",
                    },
                },
                "required": ["agent_id", "namespace", "memory_ids"],
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


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "archivist_search":
            return await _handle_search(arguments)
        elif name == "archivist_recall":
            return await _handle_recall(arguments)
        elif name == "archivist_store":
            return await _handle_store(arguments)
        elif name == "archivist_timeline":
            return await _handle_timeline(arguments)
        elif name == "archivist_insights":
            return await _handle_insights(arguments)
        elif name == "archivist_namespaces":
            return await _handle_namespaces(arguments)
        elif name == "archivist_audit_trail":
            return await _handle_audit_trail(arguments)
        elif name == "archivist_merge":
            return await _handle_merge(arguments)
        elif name == "archivist_deref":
            return await _handle_deref(arguments)
        elif name == "archivist_index":
            return await _handle_index(arguments)
        elif name == "archivist_contradictions":
            return await _handle_contradictions(arguments)
        elif name == "archivist_log_trajectory":
            return await _handle_log_trajectory(arguments)
        elif name == "archivist_annotate":
            return await _handle_annotate(arguments)
        elif name == "archivist_rate":
            return await _handle_rate(arguments)
        elif name == "archivist_tips":
            return await _handle_tips(arguments)
        elif name == "archivist_session_end":
            return await _handle_session_end(arguments)
        elif name == "archivist_register_skill":
            return await _handle_register_skill(arguments)
        elif name == "archivist_skill_event":
            return await _handle_skill_event(arguments)
        elif name == "archivist_skill_lesson":
            return await _handle_skill_lesson(arguments)
        elif name == "archivist_skill_health":
            return await _handle_skill_health(arguments)
        elif name == "archivist_resolve_uri":
            return await _handle_resolve_uri(arguments)
        elif name == "archivist_retrieval_logs":
            return await _handle_retrieval_logs(arguments)
        elif name == "archivist_cache_stats":
            return await _handle_cache_stats(arguments)
        elif name == "archivist_cache_invalidate":
            return await _handle_cache_invalidate(arguments)
        elif name == "archivist_health_dashboard":
            return await _handle_health_dashboard(arguments)
        elif name == "archivist_batch_heuristic":
            return await _handle_batch_heuristic(arguments)
        elif name == "archivist_compress":
            return await _handle_compress(arguments)
        elif name == "archivist_skill_relate":
            return await _handle_skill_relate(arguments)
        elif name == "archivist_skill_dependencies":
            return await _handle_skill_dependencies(arguments)
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e, exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_search(arguments: dict) -> list[TextContent]:
    agent_id = arguments.get("agent_id", "")
    namespace = arguments.get("namespace", "")

    raw_ids = arguments.get("agent_ids")
    agent_ids: list[str] | None = None
    if raw_ids is not None:
        if isinstance(raw_ids, str):
            agent_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
        else:
            agent_ids = [str(x).strip() for x in raw_ids if str(x).strip()]

    caller = (arguments.get("caller_agent_id") or "").strip() or agent_id

    if namespace and caller:
        denied = _rbac_gate(caller, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    if agent_ids:
        allowed, denied_list = filter_agents_for_read(caller, agent_ids)
        if not allowed:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "access_denied",
                    "reason": "Caller cannot read any of the requested agents' namespaces",
                    "denied_agents": denied_list,
                    "caller_agent_id": caller,
                }),
            )]
        agent_ids = allowed
    elif agent_id:
        allowed, denied_list = filter_agents_for_read(caller, [agent_id])
        if not allowed:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "access_denied",
                    "reason": f"Cannot read memories for agent '{agent_id}'",
                    "denied_agents": denied_list,
                    "caller_agent_id": caller,
                }),
            )]

    min_score = arguments.get("min_score")
    threshold = float(min_score) if min_score is not None else None

    result = await recursive_retrieve(
        query=arguments["query"],
        agent_id="" if agent_ids else agent_id,
        agent_ids=agent_ids,
        team=arguments.get("team", ""),
        namespace=namespace,
        limit=arguments.get("limit", 20),
        refine=arguments.get("refine", True),
        threshold=threshold,
        tier=arguments.get("tier", "l2"),
        date_from=arguments.get("date_from", ""),
        date_to=arguments.get("date_to", ""),
        max_tokens=arguments.get("max_tokens"),
        memory_type=arguments.get("memory_type", ""),
    )
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


def _filter_facts_rels_for_caller(caller: str, facts: list[dict], rels: list[dict]):
    """Drop graph rows the caller is not allowed to read (fact/rel agent_id)."""
    ff = [f for f in facts if can_read_agent_memory(caller, (f.get("agent_id") or "").strip())]
    rr = [r for r in rels if can_read_agent_memory(caller, (r.get("agent_id") or "").strip())]
    return ff, rr


async def _handle_recall(arguments: dict) -> list[TextContent]:
    entity_name = arguments["entity"]
    related_name = arguments.get("related_to", "")
    namespace = arguments.get("namespace", "")

    agent_id = (arguments.get("agent_id") or "").strip()
    caller = (arguments.get("caller_agent_id") or "").strip() or agent_id

    if not is_permissive_mode() and not caller:
        return [TextContent(
            type="text",
            text=json.dumps({"error": "caller_agent_id_or_agent_id_required", "reason": "RBAC requires caller identity for graph recall"}),
        )]

    if namespace and caller:
        denied = _rbac_gate(caller, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    entities = search_entities(entity_name)
    if not entities:
        return [TextContent(type="text", text=json.dumps({"error": f"Entity '{entity_name}' not found in knowledge graph"}))]

    eid = entities[0]["id"]
    facts = get_entity_facts(eid)
    rels = get_entity_relationships(eid)

    facts, rels = _filter_facts_rels_for_caller(caller, facts, rels)

    result = {
        "entity": entities[0],
        "facts": facts[:20],
        "relationships": rels[:20],
    }

    if related_name:
        rel_entities = search_entities(related_name)
        if rel_entities:
            rel_eid = rel_entities[0]["id"]
            rel_facts = get_entity_facts(rel_eid)
            rel_facts, _ = _filter_facts_rels_for_caller(caller, rel_facts, [])
            result["related_entity"] = rel_entities[0]
            result["related_facts"] = rel_facts[:10]
            shared = [
                r for r in rels
                if r["target_entity_id"] == rel_eid or r["source_entity_id"] == rel_eid
            ]
            result["shared_relationships"] = shared

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _handle_store(arguments: dict) -> list[TextContent]:
    text = arguments["text"]
    agent_id = arguments["agent_id"]
    namespace = arguments.get("namespace", "") or get_namespace_for_agent(agent_id)
    entity_names = arguments.get("entities", [])
    importance = arguments.get("importance_score", 0.5)
    force_skip = bool(arguments.get("force_skip_conflict_check", False))

    denied = _rbac_gate(agent_id, "write", namespace)
    if denied:
        return [TextContent(type="text", text=denied)]

    if CONFLICT_CHECK_ON_STORE and not force_skip:
        cr = await check_for_conflicts(text, namespace, agent_id)
        if cr.has_conflict and CONFLICT_BLOCK_ON_STORE:
            m.inc(m.STORE_CONFLICT, {"namespace": namespace})
            webhooks.fire_background("memory_conflict", {
                "agent_id": agent_id, "namespace": namespace,
                "max_similarity": cr.max_similarity,
                "conflicting_ids": cr.conflicting_ids,
            })
            return [TextContent(type="text", text=json.dumps({
                "stored": False,
                "conflict": True,
                "max_similarity": cr.max_similarity,
                "conflicting_ids": cr.conflicting_ids,
                "recommendation": cr.recommendation,
                "hint": "Set force_skip_conflict_check true to store anyway, or merge with conflicting memories.",
            }))]

    if not force_skip:
        dedup = await llm_adjudicated_dedup(text, namespace, agent_id)
        if dedup and dedup.action == "skip":
            return [TextContent(type="text", text=json.dumps({
                "stored": False,
                "dedup_action": "skip",
                "reason": "LLM determined this memory is a duplicate",
                "existing_ids": dedup.existing_ids,
                "decisions": dedup.decisions,
            }))]
        if dedup and dedup.action == "merge":
            curator_queue.enqueue("merge_memory", {
                "new_text": text, "agent_id": agent_id, "namespace": namespace,
                "existing_ids": dedup.existing_ids, "decisions": dedup.decisions,
            })
        if dedup and dedup.action == "delete_old":
            for d in dedup.decisions:
                if d.get("decision") == "delete":
                    curator_queue.enqueue("archive_memory", {
                        "memory_ids": [d.get("existing_id", "")],
                        "reason": "superseded",
                    })

    ns_config = get_namespace_config(namespace)
    consistency = ns_config.consistency if ns_config else "eventual"

    for ename in entity_names:
        eid = upsert_entity(ename.strip())
        add_fact(eid, text, f"explicit/{agent_id}", agent_id)

    if not entity_names:
        eid = upsert_entity(agent_id, "agent")
        add_fact(eid, text, f"explicit/{agent_id}", agent_id)

    vec = await embed_text(text)
    client = QdrantClient(url=QDRANT_URL, timeout=30)
    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    checksum = hashlib.sha256(f"{text}:{agent_id}:{namespace}".encode()).hexdigest()

    ttl_expires_at = None
    if ns_config and ns_config.ttl_days is not None:
        if importance < 0.9:
            from datetime import timedelta
            ttl_expires_at = int((now + timedelta(days=ns_config.ttl_days)).timestamp())

    payload = {
        "agent_id": agent_id,
        "text": text,
        "file_path": f"explicit/{agent_id}",
        "file_type": "explicit",
        "date": now.strftime("%Y-%m-%d"),
        "team": TEAM_MAP.get(agent_id, "unknown"),
        "chunk_index": 0,
        "namespace": namespace,
        "version": 1,
        "consistency_level": consistency,
        "checksum": checksum,
        "importance_score": importance,
        "memory_type": arguments.get("memory_type", "general"),
    }
    if ttl_expires_at is not None:
        payload["ttl_expires_at"] = ttl_expires_at

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[PointStruct(id=pid, vector=vec, payload=payload)],
    )

    from audit import log_memory_event
    await log_memory_event(
        agent_id=agent_id,
        action="create",
        memory_id=pid,
        namespace=namespace,
        text_hash=checksum,
        version=1,
        metadata={"trigger": "api", "importance_score": importance},
    )

    hot_cache.invalidate_namespace(namespace)

    m.inc(m.STORE_TOTAL, {"namespace": namespace})
    webhooks.fire_background("memory_store", {
        "memory_id": pid, "agent_id": agent_id, "namespace": namespace,
    })

    return [TextContent(type="text", text=json.dumps({
        "stored": True,
        "memory_id": pid,
        "uri": memory_uri(namespace, pid),
        "namespace": namespace,
        "entities": entity_names or [agent_id],
        "version": 1,
    }))]


async def _handle_timeline(arguments: dict) -> list[TextContent]:
    query = arguments["query"]
    agent_id = arguments.get("agent_id", "")
    namespace = arguments.get("namespace", "")
    caller = (arguments.get("caller_agent_id") or "").strip() or agent_id

    if not is_permissive_mode() and not caller:
        return [TextContent(
            type="text",
            text=json.dumps({"error": "caller_agent_id_or_agent_id_required", "reason": "RBAC requires caller identity for timeline"}),
        )]

    if namespace and caller:
        denied = _rbac_gate(caller, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    if agent_id:
        allowed, denied_list = filter_agents_for_read(caller, [agent_id])
        if not allowed:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "access_denied",
                    "reason": "Caller cannot read this agent's memories",
                    "denied_agents": denied_list,
                }),
            )]

    results = await search_vectors(query, agent_id=agent_id, namespace=namespace, limit=50)

    results = [r for r in results if can_read_agent_memory(caller, r.get("agent_id", ""))]

    results.sort(key=lambda x: x.get("date", ""))

    timeline = []
    for r in results:
        timeline.append({
            "date": r["date"],
            "agent": r["agent_id"],
            "source": r["file_path"],
            "namespace": r.get("namespace", ""),
            "text": r["text"][:500],
            "score": r["score"],
        })

    return [TextContent(type="text", text=json.dumps({"query": query, "timeline": timeline[:30]}, indent=2))]


async def _handle_insights(arguments: dict) -> list[TextContent]:
    topic = arguments["topic"]
    limit = arguments.get("limit", 10)
    namespace = arguments.get("namespace", "")
    agent_id = arguments.get("agent_id", "")
    caller = (arguments.get("caller_agent_id") or "").strip() or agent_id

    if not is_permissive_mode() and not caller:
        return [TextContent(
            type="text",
            text=json.dumps({"error": "caller_agent_id_or_agent_id_required", "reason": "RBAC requires caller identity for insights"}),
        )]

    if namespace and caller:
        denied = _rbac_gate(caller, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    results = await search_vectors(topic, namespace=namespace, limit=limit * 3)

    results = [r for r in results if can_read_agent_memory(caller, r.get("agent_id", ""))]

    agents_seen = set()
    insights = []
    teams_seen = set()
    for r in results:
        key = f"{r['agent_id']}:{r['text'][:100]}"
        if key not in agents_seen:
            agents_seen.add(key)
            teams_seen.add(r["team"])
            insights.append({
                "agent": r["agent_id"],
                "team": r["team"],
                "date": r["date"],
                "namespace": r.get("namespace", ""),
                "text": r["text"][:500],
                "source": r["file_path"],
                "score": r["score"],
            })
        if len(insights) >= limit:
            break

    return [TextContent(type="text", text=json.dumps({
        "topic": topic,
        "teams_represented": list(teams_seen),
        "insights": insights,
    }, indent=2))]


async def _handle_namespaces(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]
    namespaces = list_accessible_namespaces(agent_id)
    return [TextContent(type="text", text=json.dumps({
        "agent_id": agent_id,
        "default_namespace": get_namespace_for_agent(agent_id),
        "accessible_namespaces": namespaces,
    }, indent=2))]


async def _handle_audit_trail(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]

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

    return [TextContent(type="text", text=json.dumps({"entries": entries}, indent=2, default=str))]


async def _handle_merge(arguments: dict) -> list[TextContent]:
    agent_id = arguments["agent_id"]
    memory_ids = arguments["memory_ids"]
    strategy = arguments["strategy"]
    namespace = arguments.get("namespace", "")

    from merge import merge_memories
    result = await merge_memories(memory_ids, strategy, agent_id, namespace)
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _handle_deref(arguments: dict) -> list[TextContent]:
    """Dereference a single memory point by ID — returns full L2 text + metadata."""
    memory_id = arguments["memory_id"]
    agent_id = arguments.get("agent_id", "")

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    try:
        points = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[memory_id],
            with_payload=True,
        )
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    if not points:
        return [TextContent(type="text", text=json.dumps({"error": "not_found", "memory_id": memory_id}))]

    p = points[0]
    payload = p.payload or {}

    if agent_id and not is_permissive_mode():
        ns = payload.get("namespace", "")
        if ns:
            denied = _rbac_gate(agent_id, "read", ns)
            if denied:
                return [TextContent(type="text", text=denied)]

    return [TextContent(type="text", text=json.dumps({
        "memory_id": str(p.id),
        "uri": memory_uri(payload.get("namespace", ""), str(p.id)),
        "text": payload.get("text", ""),
        "l0": payload.get("l0", ""),
        "l1": payload.get("l1", ""),
        "agent_id": payload.get("agent_id", ""),
        "namespace": payload.get("namespace", ""),
        "file_path": payload.get("file_path", ""),
        "file_type": payload.get("file_type", ""),
        "date": payload.get("date", ""),
        "parent_id": payload.get("parent_id"),
        "is_parent": payload.get("is_parent", False),
        "importance_score": payload.get("importance_score", 0.5),
        "version": payload.get("version", 1),
    }, indent=2))]


async def _handle_index(arguments: dict) -> list[TextContent]:
    """Return a compressed navigational index for a namespace."""
    agent_id = arguments.get("agent_id", "")
    namespace = arguments.get("namespace", "")

    if not namespace and agent_id:
        namespace = get_namespace_for_agent(agent_id)

    if namespace and agent_id and not is_permissive_mode():
        denied = _rbac_gate(agent_id, "read", namespace)
        if denied:
            return [TextContent(type="text", text=denied)]

    index_text = build_namespace_index(namespace, agent_ids=[agent_id] if agent_id else None)
    return [TextContent(type="text", text=index_text)]


async def _handle_contradictions(arguments: dict) -> list[TextContent]:
    """Surface contradicting facts about an entity from different agents."""
    entity_name = arguments["entity"]
    agent_id = arguments.get("agent_id", "")

    entities = search_entities(entity_name, limit=1)
    if not entities:
        return [TextContent(type="text", text=json.dumps({
            "entity": entity_name,
            "contradictions": [],
            "note": "Entity not found in knowledge graph",
        }))]

    eid = entities[0]["id"]
    contradictions = detect_contradictions(eid)

    return [TextContent(type="text", text=json.dumps({
        "entity": entity_name,
        "entity_id": eid,
        "contradictions": contradictions,
        "count": len(contradictions),
    }, indent=2))]


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

    return [TextContent(type="text", text=json.dumps({
        **result,
        "attributions": attributions,
        "tips": tips,
    }, indent=2, default=str))]


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

    return [TextContent(type="text", text=json.dumps({
        "annotation_id": ann_id,
        "memory_id": arguments["memory_id"],
        "annotations": get_annotations(arguments["memory_id"]),
    }, indent=2, default=str))]


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
    return [TextContent(type="text", text=json.dumps({
        "rating_id": rating_id,
        **summary,
    }, indent=2))]


async def _handle_tips(arguments: dict) -> list[TextContent]:
    """Retrieve tips from past trajectories."""
    tips = search_tips(
        agent_id=arguments["agent_id"],
        category=arguments.get("category", ""),
        limit=arguments.get("limit", 10),
    )
    return [TextContent(type="text", text=json.dumps({
        "agent_id": arguments["agent_id"],
        "tips": tips,
        "count": len(tips),
    }, indent=2, default=str))]


async def _handle_session_end(arguments: dict) -> list[TextContent]:
    """Summarize a session and optionally store as durable memory."""
    result = await session_end_summary(
        agent_id=arguments["agent_id"],
        session_id=arguments["session_id"],
    )

    if result.get("error"):
        return [TextContent(type="text", text=json.dumps(result))]

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
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# ── Skill tools (v0.7) ────────────────────────────────────────────────────────

def _resolve_skill(name: str, provider: str = "") -> dict | None:
    return find_skill(name, provider)


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

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_skill_event(arguments: dict) -> list[TextContent]:
    skill = _resolve_skill(arguments["skill_name"], arguments.get("provider", ""))
    if not skill:
        return [TextContent(type="text", text=json.dumps({
            "error": "skill_not_found",
            "skill_name": arguments["skill_name"],
            "hint": "Register the skill first with archivist_register_skill",
        }))]

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

    return [TextContent(type="text", text=json.dumps({
        "event_id": event_id,
        "skill_id": skill["id"],
        "skill_name": skill["name"],
        "outcome": arguments["outcome"],
    }, indent=2))]


async def _handle_skill_lesson(arguments: dict) -> list[TextContent]:
    skill = _resolve_skill(arguments["skill_name"], arguments.get("provider", ""))
    if not skill:
        return [TextContent(type="text", text=json.dumps({
            "error": "skill_not_found",
            "skill_name": arguments["skill_name"],
            "hint": "Register the skill first with archivist_register_skill",
        }))]

    lesson_id = add_lesson(
        skill_id=skill["id"],
        title=arguments["title"],
        content=arguments["content"],
        lesson_type=arguments.get("lesson_type", "general"),
        skill_version=arguments.get("skill_version", ""),
        agent_id=arguments["agent_id"],
    )

    return [TextContent(type="text", text=json.dumps({
        "lesson_id": lesson_id,
        "skill_id": skill["id"],
        "skill_name": skill["name"],
        "title": arguments["title"],
    }, indent=2))]


async def _handle_skill_health(arguments: dict) -> list[TextContent]:
    skill = _resolve_skill(arguments["skill_name"], arguments.get("provider", ""))
    if not skill:
        return [TextContent(type="text", text=json.dumps({
            "error": "skill_not_found",
            "skill_name": arguments["skill_name"],
        }))]

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

    return [TextContent(type="text", text=json.dumps(health, indent=2))]


# ── Memory hierarchy, URI, trajectory export (v0.8) ──────────────────────────

async def _handle_resolve_uri(arguments: dict) -> list[TextContent]:
    """Resolve an archivist:// URI to the underlying resource."""
    from archivist_uri import parse_uri
    uri = parse_uri(arguments["uri"])
    if not uri:
        return [TextContent(type="text", text=json.dumps({
            "error": "invalid_uri", "uri": arguments["uri"],
            "hint": "Format: archivist://{namespace}/{memory|entity|namespace|skill}/{id}",
        }))]

    agent_id = arguments.get("agent_id", "")

    if uri.is_memory:
        return await _handle_deref({"memory_id": uri.resource_id, "agent_id": agent_id})

    if uri.is_entity:
        return await _handle_recall({"entity": uri.resource_id, "agent_id": agent_id, "namespace": uri.namespace})

    if uri.is_namespace:
        return await _handle_index({"agent_id": agent_id, "namespace": uri.namespace})

    if uri.is_skill:
        skill = find_skill(uri.resource_id)
        if skill:
            health = get_skill_health(skill["id"])
            health["recent_lessons"] = get_lessons(skill["id"], limit=5)
            return [TextContent(type="text", text=json.dumps(health, indent=2))]
        return [TextContent(type="text", text=json.dumps({"error": "skill_not_found", "name": uri.resource_id}))]

    return [TextContent(type="text", text=json.dumps({"error": "unsupported_resource_type", "type": uri.resource_type}))]


async def _handle_retrieval_logs(arguments: dict) -> list[TextContent]:
    """Export retrieval logs or aggregate stats."""
    if arguments.get("stats_only"):
        stats = get_retrieval_stats(
            agent_id=arguments.get("agent_id", ""),
            window_days=arguments.get("window_days", 7),
        )
        return [TextContent(type="text", text=json.dumps(stats, indent=2))]

    logs = get_retrieval_logs(
        agent_id=arguments.get("agent_id", ""),
        limit=arguments.get("limit", 20),
        since=arguments.get("since", ""),
    )
    return [TextContent(type="text", text=json.dumps({
        "logs": logs,
        "count": len(logs),
    }, indent=2))]


async def _handle_cache_stats(arguments: dict) -> list[TextContent]:
    """Return hot cache statistics."""
    return [TextContent(type="text", text=json.dumps(hot_cache.stats(), indent=2))]


async def _handle_cache_invalidate(arguments: dict) -> list[TextContent]:
    """Manually invalidate cache entries."""
    if arguments.get("all"):
        n = hot_cache.invalidate_all()
        return [TextContent(type="text", text=json.dumps({"invalidated": n, "scope": "all"}))]
    if arguments.get("agent_id"):
        n = hot_cache.invalidate_agent(arguments["agent_id"])
        return [TextContent(type="text", text=json.dumps({"invalidated": n, "scope": f"agent:{arguments['agent_id']}"}))]
    if arguments.get("namespace"):
        n = hot_cache.invalidate_namespace(arguments["namespace"])
        return [TextContent(type="text", text=json.dumps({"invalidated": n, "scope": f"namespace:{arguments['namespace']}"}))]
    return [TextContent(type="text", text=json.dumps({"invalidated": 0, "hint": "Specify namespace, agent_id, or all=true"}))]


# ── Observability & ops (v0.9) ───────────────────────────────────────────────

async def _handle_health_dashboard(arguments: dict) -> list[TextContent]:
    """Comprehensive health dashboard across all subsystems."""
    result = build_dashboard(window_days=arguments.get("window_days", 7))
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _handle_batch_heuristic(arguments: dict) -> list[TextContent]:
    """Recommend batch size from memory health signals."""
    result = batch_heuristic(window_days=arguments.get("window_days", 7))
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# ── Curator intelligence (v1.0) ──────────────────────────────────────────────

async def _handle_compress(arguments: dict) -> list[TextContent]:
    """Archive memory blocks and return a compact summary."""
    from llm import llm_query

    agent_id = arguments["agent_id"]
    namespace = arguments["namespace"]
    memory_ids = arguments["memory_ids"]
    user_summary = arguments.get("summary", "")

    denied = _rbac_gate(agent_id, "write", namespace)
    if denied:
        return [TextContent(type="text", text=denied)]

    if not memory_ids:
        return [TextContent(type="text", text=json.dumps({"error": "memory_ids required"}))]

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    texts = []
    for mid in memory_ids:
        try:
            points = client.retrieve(
                collection_name=QDRANT_COLLECTION, ids=[mid], with_payload=True,
            )
            if points:
                texts.append((str(points[0].id), (points[0].payload or {}).get("text", "")))
        except Exception as e:
            logger.warning("Compress: failed to retrieve %s: %s", mid, e)

    if not texts:
        return [TextContent(type="text", text=json.dumps({"error": "no memories found for given IDs"}))]

    summary = user_summary
    if not summary:
        combined = "\n\n---\n\n".join(f"[{mid}] {t[:400]}" for mid, t in texts)
        try:
            summary = await llm_query(
                combined,
                system=(
                    "Summarize these memory entries into a single concise summary (200 tokens max). "
                    "Preserve key facts, entities, and actionable insights."
                ),
                max_tokens=300,
            )
            summary = summary.strip()
        except Exception as e:
            logger.warning("Compress LLM summary failed: %s", e)
            summary = f"Compressed {len(texts)} memories from namespace {namespace}."

    store_result = await _handle_store({
        "text": f"[Compressed summary]\n{summary}",
        "agent_id": agent_id,
        "namespace": namespace,
        "importance_score": 0.8,
        "memory_type": "general",
        "force_skip_conflict_check": True,
    })

    stored_data = {}
    try:
        stored_data = json.loads(store_result[0].text)
    except Exception:
        pass

    curator_queue.enqueue("archive_memory", {
        "memory_ids": memory_ids,
        "reason": "compressed",
        "compressed_into": stored_data.get("memory_id", ""),
    })

    hot_cache.invalidate_namespace(namespace)

    return [TextContent(type="text", text=json.dumps({
        "compressed": True,
        "compressed_memory_id": stored_data.get("memory_id"),
        "uri": stored_data.get("uri"),
        "summary_l0": summary[:200],
        "archived_count": len(memory_ids),
        "archived_ids": memory_ids,
    }, indent=2))]


async def _handle_skill_relate(arguments: dict) -> list[TextContent]:
    """Create or update a relation between two skills."""
    skill_a = _resolve_skill(arguments["skill_a"], arguments.get("provider_a", ""))
    skill_b = _resolve_skill(arguments["skill_b"], arguments.get("provider_b", ""))

    if not skill_a:
        return [TextContent(type="text", text=json.dumps({"error": "skill_not_found", "skill_name": arguments["skill_a"]}))]
    if not skill_b:
        return [TextContent(type="text", text=json.dumps({"error": "skill_not_found", "skill_name": arguments["skill_b"]}))]

    rel_id = add_skill_relation(
        skill_a_id=skill_a["id"],
        skill_b_id=skill_b["id"],
        relation_type=arguments["relation_type"],
        confidence=arguments.get("confidence", 1.0),
        evidence=arguments.get("evidence", ""),
        created_by=arguments["agent_id"],
    )

    return [TextContent(type="text", text=json.dumps({
        "relation_id": rel_id,
        "skill_a": skill_a["name"],
        "skill_b": skill_b["name"],
        "relation_type": arguments["relation_type"],
    }, indent=2))]


async def _handle_skill_dependencies(arguments: dict) -> list[TextContent]:
    """Return the dependency/relation graph for a skill."""
    skill = _resolve_skill(arguments["skill_name"], arguments.get("provider", ""))
    if not skill:
        return [TextContent(type="text", text=json.dumps({"error": "skill_not_found", "skill_name": arguments["skill_name"]}))]

    depth = arguments.get("depth", 1)
    relations = get_skill_relations(skill["id"], depth=depth)
    substitutes = get_skill_substitutes(skill["id"])

    return [TextContent(type="text", text=json.dumps({
        "skill": skill["name"],
        "skill_id": skill["id"],
        "relations": relations,
        "substitutes": [{"name": s["name"], "relation": s["relation_type"], "confidence": s["confidence"]} for s in substitutes],
        "depth": depth,
    }, indent=2, default=str))]



