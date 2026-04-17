"""MCP tool handler — reference documentation for agents.

Exposes archivist_get_reference_docs so any connected agent can retrieve the
full tool skill guide (or a named section of it) without requiring out-of-band
documentation.
"""

import json
import logging
from pathlib import Path

from mcp.types import TextContent, Tool

logger = logging.getLogger("archivist.mcp")

# Primary reference document bundled with the repo.
# Fall back to REFERENCE.md if the skill doc is missing (shouldn't happen in
# normal deployments, but makes the handler robust during development).
_SKILL_DOC = Path(__file__).parent.parent.parent / "docs" / "CURSOR_SKILL.md"
_FALLBACK_DOC = Path(__file__).parent.parent.parent / "docs" / "REFERENCE.md"


def _doc_path() -> Path:
    return _SKILL_DOC if _SKILL_DOC.exists() else _FALLBACK_DOC


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="archivist_get_reference_docs",
        description=(
            "Return the Archivist agent skill reference — complete documentation "
            "of every available MCP tool, its parameters, and usage examples. "
            "Call this on first connection or whenever you are unsure how to use "
            "a tool. "
            "Optionally pass `section` to return only the matching heading block "
            "(e.g. 'search', 'storage', 'trajectory', 'skills', 'admin', 'tips')."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": (
                        "Optional heading keyword to filter. "
                        "Matches the first top-level section whose heading "
                        "contains this string (case-insensitive). "
                        "Omit to return the full reference."
                    ),
                },
            },
            "required": [],
        },
    ),
]

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_text, block_text) pairs on '## ' boundaries.

    The block includes the heading line itself and all content until the next
    same-level heading.  Content before the first '## ' heading is returned as
    a single block with an empty heading string so the preamble is accessible.
    """
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            sections.append((current_heading, "".join(current_lines)))
            current_heading = line[3:].rstrip()
            current_lines = [line]
        else:
            current_lines.append(line)

    sections.append((current_heading, "".join(current_lines)))
    return sections


async def _handle_get_reference_docs(arguments: dict) -> list[TextContent]:
    """Return the agent skill reference, optionally filtered to one section."""
    doc = _doc_path()
    if not doc.exists():
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "reference_docs_not_found", "path": str(doc)}),
            )
        ]

    try:
        text = doc.read_text(encoding="utf-8")
    except OSError as exc:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "read_failed", "detail": str(exc)}),
            )
        ]

    section_filter = (arguments.get("section") or "").strip().lower()
    if not section_filter:
        return [TextContent(type="text", text=text)]

    sections = _split_sections(text)
    for heading, block in sections:
        if section_filter in heading.lower():
            return [TextContent(type="text", text=block)]

    available = [h for h, _ in sections if h]
    return [
        TextContent(
            type="text",
            text=json.dumps(
                {
                    "error": "section_not_found",
                    "section": section_filter,
                    "available_sections": available,
                }
            ),
        )
    ]


HANDLERS: dict[str, object] = {
    "archivist_get_reference_docs": _handle_get_reference_docs,
}
