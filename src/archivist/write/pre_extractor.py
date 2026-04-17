"""Deterministic pre-extraction from markdown — no LLM calls.

Uses a two-pass approach: deterministic regex/AST extraction first, then LLM
as a second pass.  Runs before the LLM curator to provide structured hints,
reducing hallucination and token spend.
"""

from __future__ import annotations

import re
from collections import Counter

from archivist.utils.chunking import NEEDLE_PATTERNS as _CHUNKING_NEEDLE_PATTERNS

# ---------------------------------------------------------------------------
# Thought-type taxonomy (mid-granularity semantic classification)
# ---------------------------------------------------------------------------

THOUGHT_TYPES = (
    "decision",
    "lesson",
    "constraint",
    "insight",
    "preference",
    "milestone",
    "correction",
    "general",
)

_THOUGHT_HEADER_MAP: dict[str, str] = {
    "decision": "decision",
    "decisions": "decision",
    "decided": "decision",
    "lesson": "lesson",
    "lessons": "lesson",
    "learned": "lesson",
    "takeaway": "lesson",
    "takeaways": "lesson",
    "constraint": "constraint",
    "constraints": "constraint",
    "limitation": "constraint",
    "limitations": "constraint",
    "insight": "insight",
    "insights": "insight",
    "observation": "insight",
    "observations": "insight",
    "finding": "insight",
    "findings": "insight",
    "preference": "preference",
    "preferences": "preference",
    "milestone": "milestone",
    "milestones": "milestone",
    "achievement": "milestone",
    "achievements": "milestone",
    "correction": "correction",
    "corrections": "correction",
    "fix": "correction",
    "fixes": "correction",
    "bugfix": "correction",
    "incident": "lesson",
    "postmortem": "lesson",
    "outage": "lesson",
}

_THOUGHT_BODY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:we decided|decision was|chose to|opted for|agreed to)\b", re.I), "decision"),
    (re.compile(r"\b(?:lesson learned|takeaway|in hindsight|next time)\b", re.I), "lesson"),
    (re.compile(r"\b(?:must not|cannot|blocked by|constraint|limitation)\b", re.I), "constraint"),
    (re.compile(r"\b(?:discovered|realized|noticed|found that|turns out)\b", re.I), "insight"),
    (re.compile(r"\b(?:prefer|always use|never use|default to)\b", re.I), "preference"),
    (
        re.compile(r"\b(?:completed|shipped|deployed|launched|released|milestone)\b", re.I),
        "milestone",
    ),
    (re.compile(r"\b(?:fixed|corrected|was wrong|actually|correction)\b", re.I), "correction"),
]

# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------

_BOLD_ENTITY_RE = re.compile(r"\*\*([A-Z][A-Za-z0-9_\-./]{1,60})\*\*")
_BACKTICK_ENTITY_RE = re.compile(r"`([A-Za-z][A-Za-z0-9_\-./]{2,60})`")
_HEADER_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

_ENTITY_TYPE_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(?:https?://|www\.)", re.I), "url"),
    (re.compile(r"\.\d+\.\d+"), "version"),
    (re.compile(r"(?:\.yaml|\.yml|\.json|\.toml|\.py|\.go|\.rs|\.ts|\.js)$", re.I), "file"),
    (re.compile(r"(?:prod|staging|dev|test)[-_]", re.I), "host"),
    (re.compile(r"(?:db|rds|redis|postgres|mysql)", re.I), "database"),
    (re.compile(r"(?:k8s|kube|cluster|node|pod)", re.I), "kubernetes"),
]

_STOP_ENTITIES = frozenset(
    {
        "the",
        "this",
        "that",
        "true",
        "false",
        "none",
        "null",
        "todo",
        "note",
        "example",
        "yes",
        "no",
        "ok",
        "done",
        "wip",
    }
)

_NEEDLE_ENTITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_CHUNKING_NEEDLE_PATTERNS[0], "ip_address"),  # IP / CIDR
    (_CHUNKING_NEEDLE_PATTERNS[5], "uuid"),  # UUID
    (_CHUNKING_NEEDLE_PATTERNS[3], "ticket_id"),  # employee / ticket IDs
    (
        re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+){2,}\b"),
        "hostname",
    ),  # filtered in extract_needle_entities
]


def _looks_like_hostname(text: str) -> bool:
    """Require at least one segment to contain a digit, avoiding common English."""
    return any(c.isdigit() for c in text)


def _classify_entity(name: str) -> str:
    """Guess entity type from name patterns."""
    for pat, etype in _ENTITY_TYPE_HINTS:
        if pat.search(name):
            return etype
    return "unknown"


def extract_needle_entities(text: str) -> list[dict]:
    """Extract high-specificity tokens (IPs, UUIDs, ticket IDs, hostnames) as entities."""
    results: list[dict] = []
    seen: set[str] = set()
    for pat, etype in _NEEDLE_ENTITY_PATTERNS:
        for mt in pat.finditer(text):
            val = mt.group().strip()
            if val and val not in seen and len(val) >= 3:
                if etype == "hostname" and not _looks_like_hostname(val):
                    continue
                seen.add(val)
                results.append({"name": val, "type": etype, "confidence": "needle_pattern"})
    return results


def pre_extract(text: str, source_file: str = "") -> dict:
    """Deterministic pre-extraction from markdown. No LLM calls.

    Returns:
        {
            "entities": [{"name": str, "type": str, "confidence": "extracted"}],
            "dates": ["2025-02-15", ...],
            "thought_type": str,
            "sections": [str, ...],
            "provenance": "deterministic",
        }
    """
    entities: list[dict] = []
    seen_names: set[str] = set()

    for m in _BOLD_ENTITY_RE.finditer(text):
        name = m.group(1).strip()
        lower = name.lower()
        if lower not in _STOP_ENTITIES and lower not in seen_names:
            seen_names.add(lower)
            entities.append(
                {
                    "name": name,
                    "type": _classify_entity(name),
                    "confidence": "extracted",
                }
            )

    for m in _BACKTICK_ENTITY_RE.finditer(text):
        name = m.group(1).strip()
        lower = name.lower()
        if (
            lower not in _STOP_ENTITIES
            and lower not in seen_names
            and not name.startswith(("--", "//", "#"))
        ):
            seen_names.add(lower)
            entities.append(
                {
                    "name": name,
                    "type": _classify_entity(name),
                    "confidence": "extracted",
                }
            )

    dates = _DATE_RE.findall(text)
    if source_file:
        dates.extend(_DATE_RE.findall(source_file))
    dates = list(dict.fromkeys(dates))

    sections = [m.group(1).strip() for m in _HEADER_RE.finditer(text)]

    thought_type = _detect_thought_type(text, sections)

    return {
        "entities": entities[:30],
        "dates": dates,
        "thought_type": thought_type,
        "sections": sections,
        "provenance": "deterministic",
    }


def _detect_thought_type(text: str, sections: list[str]) -> str:
    """Classify the dominant thought type from headers and body patterns."""
    votes: Counter = Counter()

    for section in sections:
        words = section.lower().split()
        for w in words:
            tt = _THOUGHT_HEADER_MAP.get(w)
            if tt:
                votes[tt] += 3

    for pat, tt in _THOUGHT_BODY_PATTERNS:
        hits = len(pat.findall(text))
        if hits:
            votes[tt] += hits

    if not votes:
        return "general"

    best, count = votes.most_common(1)[0]
    return best if count >= 2 else "general"
