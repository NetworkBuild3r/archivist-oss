"""MCP tool handler — reference documentation for agents (INIT-024/SPEC-005).

Resolves bundled reference docs from ``archivist.app.docs_data`` via
``importlib.resources`` so installed wheels (including ``uvx``) never depend on
repository-relative paths outside the package.
"""

from __future__ import annotations

import json
import logging
from importlib import resources
from pathlib import Path

from mcp.types import TextContent, Tool

logger = logging.getLogger("archivist.mcp")


def _bundled_docs_dir() -> Path:
    """Return filesystem path to bundled ``archivist.app.docs_data`` directory."""
    return Path(str(resources.files("archivist.app.docs_data")))


_BUNDLED_DIR = _bundled_docs_dir()
_SKILL_DOC = _BUNDLED_DIR / "CURSOR_SKILL.md"
_FALLBACK_DOC = _BUNDLED_DIR / "REFERENCE.md"

_DOC_TOPICS: dict[str, str] = {
    "skill": "CURSOR_SKILL.md",
    "cursor": "CURSOR_SKILL.md",
    "reference": "REFERENCE.md",
    "tools": "CURSOR_SKILL.md",
    "overview": "REFERENCE.md",
}


def _doc_path() -> Path:
    return _SKILL_DOC if _SKILL_DOC.exists() else _FALLBACK_DOC


def list_bundled_doc_topics() -> list[str]:
    """Return available topic slugs for ``archivist://docs/{topic}``."""
    return sorted(_DOC_TOPICS.keys())


def read_bundled_doc(topic: str = "skill") -> str | None:
    """Read a bundled reference document (or heading section) via ``importlib.resources``."""
    norm = (topic or "skill").strip().lower()
    docs_root = resources.files("archivist.app.docs_data")

    filename = _DOC_TOPICS.get(norm)
    if filename:
        res = docs_root.joinpath(filename)
        if res.is_file():
            return res.read_text(encoding="utf-8")

    # Also allow matching a section heading inside the primary skill doc.
    primary = _doc_path()
    if primary.exists():
        full_text = primary.read_text(encoding="utf-8")
        for heading, block in _split_sections(full_text):
            if norm and heading and norm in heading.lower():
                return block
    return None


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
            "(e.g. 'search', 'storage', 'trajectory', 'admin', 'tips')."
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
                text=json.dumps(
                    {
                        "error": "reference_docs_not_found",
                        "tried": [str(_SKILL_DOC), str(_FALLBACK_DOC)],
                        "hint": "Neither CURSOR_SKILL.md nor REFERENCE.md was found in archivist.app.docs_data.",
                    }
                ),
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
