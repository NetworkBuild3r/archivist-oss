"""MCP tool handlers — high-level context assembly and agent handoff (Phase 4)."""

from __future__ import annotations

import dataclasses
import logging

from mcp.types import TextContent, Tool

from ._common import error_response, require_caller, success_response

logger = logging.getLogger("archivist.mcp")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="archivist_get_context",
        description=(
            "Assemble token-budgeted, tier-aware context for an agent task in a single call. "
            "Replaces the pattern of calling archivist_search + archivist_tips + archivist_context_check "
            "separately. Returns a synthesized answer (when available), ranked memory chunks packed "
            "to fit max_tokens, relevant graph facts, and procedural tips. "
            "Metadata includes token savings, tier distribution, and provenance IDs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The agent requesting context (required).",
                },
                "task_description": {
                    "type": "string",
                    "description": "The agent's current task or query.",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Hard token budget for packed context (default 8000).",
                    "default": 8000,
                },
                "namespace": {
                    "type": "string",
                    "description": "Restrict retrieval to this namespace (empty = all).",
                    "default": "",
                },
                "tier_policy": {
                    "type": "string",
                    "description": "'adaptive' (default), 'l0_first', or 'l2_first'.",
                    "default": "adaptive",
                    "enum": ["adaptive", "l0_first", "l2_first"],
                },
                "include_graph": {
                    "type": "boolean",
                    "description": "Include entity facts from the knowledge graph (default true).",
                    "default": True,
                },
                "include_tips": {
                    "type": "boolean",
                    "description": "Include procedural tips for this agent (default true).",
                    "default": True,
                },
                "extra_memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Pin additional memory IDs into the context window (e.g. from a handoff packet).",
                },
                "format_for_prompt": {
                    "type": "boolean",
                    "description": "When true, also return a pre-formatted prompt string (default false).",
                    "default": False,
                },
            },
            "required": ["agent_id", "task_description"],
        },
    ),
    Tool(
        name="archivist_handoff",
        description=(
            "Package an agent session's goals, progress, key memories, and ephemeral notes "
            "into a HandoffPacket for transfer to another agent. "
            "Call this at the end of a session when handing off to a peer or successor agent. "
            "The receiving agent should call archivist_receive_handoff with the returned packet."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The agent initiating the handoff (sender).",
                },
                "session_id": {
                    "type": "string",
                    "description": "The session ID being handed off.",
                },
                "receiving_agent_id": {
                    "type": "string",
                    "description": "The agent ID that will receive this handoff.",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Token budget for the handoff packet (default 4000).",
                    "default": 4000,
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace scope for knowledge snapshot (empty = global).",
                    "default": "",
                },
            },
            "required": ["agent_id", "session_id", "receiving_agent_id"],
        },
    ),
    Tool(
        name="archivist_receive_handoff",
        description=(
            "Inject a HandoffPacket from a peer agent into the current session's ephemeral memory. "
            "Call this at the start of a session when picking up work from another agent. "
            "Active goals, open questions, and ephemeral notes are injected as session-scoped "
            "memory so they are immediately available to archivist_get_context. "
            "Returns the list of injected keys and pinned memory IDs to pass to archivist_get_context."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The receiving agent's ID.",
                },
                "session_id": {
                    "type": "string",
                    "description": "The new session ID for the receiving agent.",
                },
                "handoff_packet": {
                    "type": "object",
                    "description": "The raw JSON object returned by archivist_handoff.",
                },
            },
            "required": ["agent_id", "session_id", "handoff_packet"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_get_context(arguments: dict) -> list[TextContent]:
    """Assemble token-budgeted context for one agent query."""
    from archivist.retrieval.context_api import (
        RelevantContext,
        format_context_for_prompt,
        get_relevant_context,
    )

    agent_id = (arguments.get("agent_id") or "").strip()
    task = (arguments.get("task_description") or "").strip()

    if err := require_caller(agent_id):
        return err
    if not task:
        return error_response({"error": "task_description is required"})

    try:
        ctx: RelevantContext = await get_relevant_context(
            agent_id=agent_id,
            task_description=task,
            max_tokens=int(arguments.get("max_tokens") or 8000),
            namespace=(arguments.get("namespace") or ""),
            tier_policy=(arguments.get("tier_policy") or "adaptive"),
            include_graph=bool(arguments.get("include_graph", True)),
            include_tips=bool(arguments.get("include_tips", True)),
            extra_memory_ids=arguments.get("extra_memory_ids") or [],
        )
    except Exception as exc:
        logger.exception("archivist_get_context failed")
        return error_response({"error": str(exc)})

    payload: dict = {
        "agent_id": agent_id,
        "task_description": task,
        "answer": ctx.answer,
        "sources": [
            {
                "memory_id": c.memory_id,
                "text": c.text,
                "score": round(c.score, 4),
                "tier": c.tier,
                "file_path": c.file_path,
                "date": c.date,
                "agent_id": c.agent_id,
            }
            for c in ctx.sources
        ],
        "graph_facts": ctx.graph_facts,
        "tips": ctx.tips,
        "provenance": ctx.provenance,
        "context_status": {
            "total_tokens": ctx.total_tokens,
            "budget_tokens": ctx.budget_tokens,
            "over_budget": ctx.over_budget,
            "tier_distribution": ctx.tier_distribution,
            "token_savings_pct": round(ctx.token_savings_pct, 1),
            "pack_policy": ctx.pack_policy,
        },
    }

    if arguments.get("format_for_prompt"):
        payload["prompt_text"] = format_context_for_prompt(ctx)

    return success_response(payload, default=str)


async def _handle_handoff(arguments: dict) -> list[TextContent]:
    """Create a HandoffPacket for agent-to-agent task transfer."""
    from archivist.retrieval.context_api import HandoffPacket, create_handoff_packet

    agent_id = (arguments.get("agent_id") or "").strip()
    session_id = (arguments.get("session_id") or "").strip()
    receiving = (arguments.get("receiving_agent_id") or "").strip()

    if err := require_caller(agent_id):
        return err
    if not session_id:
        return error_response({"error": "session_id is required"})
    if not receiving:
        return error_response({"error": "receiving_agent_id is required"})

    try:
        packet: HandoffPacket = await create_handoff_packet(
            agent_id=agent_id,
            session_id=session_id,
            receiving_agent_id=receiving,
            max_tokens=int(arguments.get("max_tokens") or 4000),
            namespace=(arguments.get("namespace") or ""),
        )
    except Exception as exc:
        logger.exception("archivist_handoff failed")
        return error_response({"error": str(exc)})

    return success_response(dataclasses.asdict(packet), default=str)


async def _handle_receive_handoff(arguments: dict) -> list[TextContent]:
    """Inject a HandoffPacket into the receiving agent's ephemeral session memory."""
    from archivist.retrieval.context_api import HandoffPacket, receive_handoff_packet

    agent_id = (arguments.get("agent_id") or "").strip()
    session_id = (arguments.get("session_id") or "").strip()
    raw_packet = arguments.get("handoff_packet") or {}

    if err := require_caller(agent_id):
        return err
    if not session_id:
        return error_response({"error": "session_id is required"})
    if not raw_packet:
        return error_response({"error": "handoff_packet is required"})

    try:
        packet = HandoffPacket(
            from_agent=raw_packet.get("from_agent", ""),
            to_agent=raw_packet.get("to_agent", agent_id),
            session_summary=raw_packet.get("session_summary", ""),
            active_goals=raw_packet.get("active_goals", []),
            open_questions=raw_packet.get("open_questions", []),
            key_memory_ids=raw_packet.get("key_memory_ids", []),
            knowledge_snapshot=raw_packet.get("knowledge_snapshot", {}),
            token_count=raw_packet.get("token_count", 0),
            created_at=raw_packet.get("created_at", ""),
            ephemeral_notes=raw_packet.get("ephemeral_notes", []),
        )
        result = await receive_handoff_packet(
            packet=packet,
            receiving_agent_id=agent_id,
            session_id=session_id,
        )
    except Exception as exc:
        logger.exception("archivist_receive_handoff failed")
        return error_response({"error": str(exc)})

    return success_response(result, default=str)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, object] = {
    "archivist_get_context": _handle_get_context,
    "archivist_handoff": _handle_handoff,
    "archivist_receive_handoff": _handle_receive_handoff,
}
