"""Zero-schema-token MCP Resources & Prompts (INIT-024/SPEC-008).

Resources (5):
- ``archivist://index/{scope}``
- ``archivist://{scope}/memory/{id}``
- ``archivist://health``
- ``archivist://stats/savings``
- ``archivist://docs/{topic}``

Prompts (3):
- ``bootstrap-session``
- ``save-session-lessons``
- ``audit-contradictions``
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote

from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    ResourceTemplate,
    TextContent,
)
from pydantic import AnyUrl

from archivist import __version__
from archivist.app.handlers.scope_resolver import resolve_read_scopes, resolve_write_scope
from archivist.app.handlers.tools_docs import list_bundled_doc_topics, read_bundled_doc
from archivist.app.handlers.v3_coding import get_local_store

# ---------------------------------------------------------------------------
# Resource & ResourceTemplate definitions
# ---------------------------------------------------------------------------


def get_static_resources() -> list[Resource]:
    """Static concrete resources advertised via ``resources/list``."""
    return [
        Resource(
            uri=AnyUrl("archivist://health"),
            name="Archivist Health",
            description="Local store health, model metadata, and scope summary.",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("archivist://stats/savings"),
            name="Archivist Token Savings",
            description="Stored memory volume and token-budget efficiency statistics.",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("archivist://docs/skill"),
            name="Archivist Reference Docs",
            description="Bundled Archivist MCP skill documentation.",
            mimeType="text/markdown",
        ),
        Resource(
            uri=AnyUrl("archivist://index/all"),
            name="Archivist Scope Index",
            description="Inventory of recent memories across all scopes.",
            mimeType="application/json",
        ),
    ]


def get_resource_templates() -> list[ResourceTemplate]:
    """URI templates advertised via ``resources/templates/list``."""
    return [
        ResourceTemplate(
            uriTemplate="archivist://index/{scope}",
            name="Scope Memory Index",
            description="List memories in a specific scope (or 'all').",
            mimeType="application/json",
        ),
        ResourceTemplate(
            uriTemplate="archivist://{scope}/memory/{id}",
            name="Single Memory Record",
            description="Read full content, provenance, and trust tier for one memory.",
            mimeType="application/json",
        ),
        ResourceTemplate(
            uriTemplate="archivist://docs/{topic}",
            name="Bundled Reference Documentation",
            description="Read bundled documentation by topic (skill, reference, overview, or section).",
            mimeType="text/markdown",
        ),
    ]


def read_resource_uri(uri: str | AnyUrl) -> str:
    """Resolve any ``archivist://`` resource URI and return UTF-8 text payload."""
    raw = str(uri).strip()
    if not raw.startswith("archivist://"):
        raise ValueError(f"Unsupported resource URI scheme: {raw!r}")

    remainder = raw[len("archivist://") :]

    # 1. archivist://health
    if remainder == "health":
        store = get_local_store()
        payload = store.stats()
        payload["version"] = __version__
        return json.dumps(payload, indent=2, ensure_ascii=False)

    # 2. archivist://stats/savings
    if remainder == "stats/savings":
        store = get_local_store()
        stats = store.stats()
        active = int(stats.get("active_memories", 0))
        stored_tokens = int(stats.get("total_stored_tokens", 0))
        estimated_l0_tokens = active * 25
        savings_pct = (
            round((1.0 - (estimated_l0_tokens / stored_tokens)) * 100.0, 1)
            if stored_tokens > 0
            else 0.0
        )
        return json.dumps(
            {
                "active_memories": active,
                "total_stored_tokens": stored_tokens,
                "estimated_l0_index_tokens": estimated_l0_tokens,
                "estimated_tier_savings_pct": max(0.0, savings_pct),
                "schema_profile": "coding",
                "schema_tools": 5,
            },
            indent=2,
            ensure_ascii=False,
        )

    # 3. archivist://docs/{topic}
    if remainder.startswith("docs/"):
        topic = unquote(remainder[len("docs/") :]).strip() or "skill"
        doc_text = read_bundled_doc(topic)
        if doc_text is None:
            return json.dumps(
                {
                    "error": "doc_topic_not_found",
                    "topic": topic,
                    "available_topics": list_bundled_doc_topics(),
                }
            )
        return doc_text

    # 4. archivist://index/{scope}
    if remainder.startswith("index/"):
        scope_part = unquote(remainder[len("index/") :]).strip()
        scope_arg = None if scope_part in ("", "all", "*") else scope_part
        store = get_local_store()
        info = store.inspect_memories(scope=scope_arg, limit=50)
        return json.dumps(info, indent=2, ensure_ascii=False)

    # 5. archivist://{scope}/memory/{id}
    marker = "/memory/"
    if marker in remainder:
        scope_part, mem_id = remainder.rsplit(marker, 1)
        scope_clean = unquote(scope_part).strip()
        mem_id_clean = unquote(mem_id).strip()
        store = get_local_store()
        record = store.get_memory(mem_id_clean)
        if record is None or (
            scope_clean and scope_clean != "*" and record["scope"] != scope_clean
        ):
            return json.dumps(
                {
                    "error": "memory_not_found",
                    "scope": scope_clean,
                    "memory_id": mem_id_clean,
                }
            )
        return json.dumps(record, indent=2, ensure_ascii=False)

    raise ValueError(f"Unknown Archivist resource URI: {raw!r}")


# ---------------------------------------------------------------------------
# Prompt definitions (3 zero-schema-token prompts)
# ---------------------------------------------------------------------------

PROMPTS: list[Prompt] = [
    Prompt(
        name="bootstrap-session",
        description="Load compact project context and key conventions at session start.",
        arguments=[
            PromptArgument(
                name="scope",
                description="Optional repository scope (auto-detected if omitted).",
                required=False,
            ),
            PromptArgument(
                name="task",
                description="Optional current task or question to focus retrieval.",
                required=False,
            ),
        ],
    ),
    Prompt(
        name="save-session-lessons",
        description="Distill durable decisions, pitfalls, and fixes from this session into memory.",
        arguments=[
            PromptArgument(
                name="scope",
                description="Optional repository scope (auto-detected if omitted).",
                required=False,
            ),
        ],
    ),
    Prompt(
        name="audit-contradictions",
        description="Audit stored memories for stale facts, contradictions, or quarantined items.",
        arguments=[
            PromptArgument(
                name="scope",
                description="Optional repository scope (auto-detected if omitted).",
                required=False,
            ),
        ],
    ),
]


def get_prompt_by_name(name: str, arguments: dict[str, Any] | None = None) -> GetPromptResult:
    """Render one of the 3 zero-schema-token MCP Prompts."""
    args = arguments or {}
    scope = resolve_write_scope(str(args.get("scope")) if args.get("scope") else None)

    if name == "bootstrap-session":
        task = str(args.get("task") or "project conventions, architecture decisions, and gotchas")
        store = get_local_store()
        read_scopes = resolve_read_scopes(scope)
        ctx = store.build_context(task, scopes=read_scopes, max_tokens=1500, limit=8)
        body = (
            f"Session bootstrap for scope `{scope}`.\n"
            f"Call `memory_context` as needed during work.\n\n"
            f"{ctx['context'] or '(No stored memories yet for this scope.)'}"
        )
        return GetPromptResult(
            description=f"Bootstrap context for {scope}",
            messages=[
                PromptMessage(role="user", content=TextContent(type="text", text=body)),
            ],
        )

    if name == "save-session-lessons":
        body = (
            f"Review this coding session for `{scope}` and persist only durable, reusable "
            "knowledge via `memory_save` (kind='decision', 'fact', or 'tip'):\n"
            "1. Architectural decisions and why alternatives were rejected.\n"
            "2. Non-obvious bug root causes, invariants, or build/test gotchas.\n"
            "3. User preferences that should persist across future sessions.\n"
            "Never save secrets, tokens, or transient one-off debugging noise."
        )
        return GetPromptResult(
            description=f"Save durable session lessons for {scope}",
            messages=[
                PromptMessage(role="user", content=TextContent(type="text", text=body)),
            ],
        )

    if name == "audit-contradictions":
        store = get_local_store()
        inv = store.inspect_memories(scope=scope, limit=30)
        body = (
            f"Audit stored memories for `{scope}` (active shown: {inv['total_shown']}, "
            f"quarantined: {inv['quarantined_count']}).\n"
            "Use `memory_inspect` to review items, flag superseded or conflicting facts, "
            "and call `memory_forget(memory_id=..., mode='suppress', reason=...)` on stale entries."
        )
        return GetPromptResult(
            description=f"Audit contradictions and quarantines in {scope}",
            messages=[
                PromptMessage(role="user", content=TextContent(type="text", text=body)),
            ],
        )

    raise ValueError(f"Unknown prompt: {name!r}")
