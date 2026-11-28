"""Pruned 5-tool ``coding`` MCP surface (INIT-024/SPEC-008).

Hard CI schema-token budget: ``<900`` total tokens across all 5 tools
(measured on serialized ``list_tools()`` JSON with ``tiktoken`` ``cl100k_base``
and ``o200k_base``).

Tools:
- ``memory_context`` — single-call hybrid retrieval + Answer Finder packing + framing
- ``memory_save``    — <30ms zero-LLM ACID write through <5ms re2 Memory Firewall
- ``memory_search``  — ranked search returning compact hit summaries + IDs/URIs
- ``memory_forget``  — atomic suppress or hard delete across SQLite + FTS5 + vectors
- ``memory_inspect`` — inspect scope inventory, provenance, trust tiers, or quarantines
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from mcp.types import TextContent, Tool

from archivist.app.handlers.scope_resolver import (
    get_request_context,
    resolve_read_scopes,
    resolve_write_scope,
)
from archivist.storage.local_store import LocalMemoryStore

logger = logging.getLogger("archivist.mcp.coding")

_LOCAL_STORE: LocalMemoryStore | None = None


def set_local_store(store: LocalMemoryStore | None) -> None:
    """Inject or clear the process-wide :class:`LocalMemoryStore`."""
    global _LOCAL_STORE
    _LOCAL_STORE = store


def get_local_store() -> LocalMemoryStore:
    """Return active :class:`LocalMemoryStore`, initializing lazily on first call."""
    global _LOCAL_STORE
    if _LOCAL_STORE is None:
        db_override = os.environ.get("ARCHIVIST_LOCAL_DB", "").strip() or None
        backend = os.environ.get("ARCHIVIST_LOCAL_VECTOR_BACKEND", "sqlite-vec").strip()
        offline = os.environ.get("ARCHIVIST_LOCAL_EMBED_OFFLINE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        _LOCAL_STORE = LocalMemoryStore(
            db_path=db_override,
            vector_backend=backend,
            offline_deterministic=offline,
        )
    return _LOCAL_STORE


# ---------------------------------------------------------------------------
# 5 Core `coding` Tool Schemas (kept crisp & flat — target 650–800 tokens total)
# ---------------------------------------------------------------------------

CODING_TOOLS: list[Tool] = [
    Tool(
        name="memory_context",
        description="Retrieve token-budgeted framed memory context for a query.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_tokens": {
                    "type": "integer",
                    "default": 4000,
                    "description": "Token budget.",
                },
                "scope": {"type": "string", "description": "Optional scope override."},
                "as_of": {"type": "string", "description": "ISO-8601 point-in-time filter."},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_save",
        description="Save a memory through the secret/injection firewall.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory text to save."},
                "kind": {
                    "type": "string",
                    "default": "fact",
                    "description": "Kind: fact, decision, preference, tip.",
                },
                "scope": {"type": "string", "description": "Optional scope override."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags.",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="memory_search",
        description="Search memories and return ranked summaries and URIs.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "limit": {"type": "integer", "default": 10, "description": "Max hits."},
                "scope": {"type": "string", "description": "Optional scope override."},
                "kind": {"type": "string", "description": "Optional kind filter."},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_forget",
        description="Suppress or delete a memory atomically across all indexes.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to forget."},
                "mode": {
                    "type": "string",
                    "enum": ["suppress", "delete"],
                    "default": "suppress",
                    "description": "Suppress or hard delete.",
                },
                "reason": {"type": "string", "description": "Optional reason."},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="memory_inspect",
        description="Inspect scope inventory, single memory provenance, or quarantines.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Optional scope filter."},
                "memory_id": {"type": "string", "description": "Optional memory ID."},
                "limit": {"type": "integer", "default": 20, "description": "Max items."},
            },
            "required": [],
        },
    ),
]

CODING_TOOL_NAMES: frozenset[str] = frozenset(t.name for t in CODING_TOOLS)


def _json_text(payload: dict[str, Any]) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_memory_context(arguments: dict[str, Any]) -> list[TextContent]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        return _json_text({"error": "query is required"})
    max_tokens = int(arguments.get("max_tokens") or 4000)
    if max_tokens <= 0:
        return _json_text({"error": "max_tokens must be positive"})
    explicit_scope = arguments.get("scope")
    as_of = arguments.get("as_of")
    scopes = resolve_read_scopes(str(explicit_scope) if explicit_scope else None)
    store = get_local_store()
    result = store.build_context(
        query,
        scopes=scopes,
        max_tokens=max_tokens,
        as_of=str(as_of) if as_of else None,
    )
    return _json_text(result)


async def _handle_memory_save(arguments: dict[str, Any]) -> list[TextContent]:
    content = str(arguments.get("content") or "").strip()
    if not content:
        return _json_text({"error": "content is required"})
    kind = str(arguments.get("kind") or "fact").strip() or "fact"
    explicit_scope = arguments.get("scope")
    tags_raw = arguments.get("tags")
    tags = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []

    scope = resolve_write_scope(str(explicit_scope) if explicit_scope else None)
    ctx = get_request_context()
    store = get_local_store()
    saved = store.save_memory(
        content,
        scope=scope,
        kind=kind,
        tags=tags,
        ctx=ctx,
    )
    return _json_text(
        {
            "memory_id": saved.memory_id,
            "uri": f"archivist://{saved.scope}/memory/{saved.memory_id}",
            "scope": saved.scope,
            "kind": saved.kind,
            "trust_tier": saved.trust_tier,
            "quarantined": saved.quarantined,
            "redacted_count": saved.redacted_count,
            "enrichment_status": saved.enrichment_status,
            "deduplicated": saved.deduplicated,
            "quarantine_reasons": list(saved.quarantine_reasons),
        }
    )


async def _handle_memory_search(arguments: dict[str, Any]) -> list[TextContent]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        return _json_text({"error": "query is required"})
    limit = int(arguments.get("limit") or 10)
    explicit_scope = arguments.get("scope")
    kind = arguments.get("kind")
    scopes = resolve_read_scopes(str(explicit_scope) if explicit_scope else None)
    store = get_local_store()
    hits = store.search_memories(
        query,
        scopes=scopes,
        limit=limit,
        kind=str(kind) if kind else None,
        include_quarantined=False,
    )
    return _json_text(
        {
            "query": query,
            "scopes": scopes,
            "count": len(hits),
            "results": [
                {
                    "id": h["id"],
                    "uri": f"archivist://{h['scope']}/memory/{h['id']}",
                    "scope": h["scope"],
                    "kind": h["kind"],
                    "trust_tier": h["trust_tier"],
                    "score": h.get("score", 0.0),
                    "summary": h.get("l0") or h["text"][:160],
                    "created_at": h["created_at"],
                }
                for h in hits
            ],
        }
    )


async def _handle_memory_forget(arguments: dict[str, Any]) -> list[TextContent]:
    memory_id = str(arguments.get("memory_id") or "").strip()
    if not memory_id:
        return _json_text({"error": "memory_id is required"})
    mode = str(arguments.get("mode") or "suppress").strip().lower()
    reason = arguments.get("reason")
    store = get_local_store()
    result = store.forget_memory(
        memory_id,
        mode=mode,
        reason=str(reason) if reason else None,
    )
    return _json_text(result)


async def _handle_memory_inspect(arguments: dict[str, Any]) -> list[TextContent]:
    explicit_scope = arguments.get("scope")
    memory_id = arguments.get("memory_id")
    limit = int(arguments.get("limit") or 20)
    store = get_local_store()
    scope_filter: str | None = str(explicit_scope).strip() if explicit_scope else None
    if scope_filter is None and not memory_id:
        scope_filter = resolve_write_scope(None)
    result = store.inspect_memories(
        scope=scope_filter,
        memory_id=str(memory_id).strip() if memory_id else None,
        limit=limit,
    )
    return _json_text(result)


CODING_HANDLERS: dict[str, object] = {
    "memory_context": _handle_memory_context,
    "memory_save": _handle_memory_save,
    "memory_search": _handle_memory_search,
    "memory_forget": _handle_memory_forget,
    "memory_inspect": _handle_memory_inspect,
}
