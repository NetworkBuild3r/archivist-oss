"""Lightweight query temporal intent classifier (v1.9).

Classifies whether a query is looking for recent information (apply normal decay),
historical/specific-date information (skip or soften decay), or is temporally
neutral (apply gentle decay).

No LLM calls — pure regex heuristics for zero-latency classification.
"""

from __future__ import annotations

import re

_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_YEAR_RE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+20\d{2}\b", re.IGNORECASE)

_RECENCY_WORDS = frozenset({
    "recent", "recently", "latest", "newest", "current", "today",
    "this week", "this month", "right now", "just now", "last hour",
})

_HISTORICAL_WORDS = frozenset({
    "when did", "when was", "what happened", "history", "historical",
    "previously", "past", "back then", "timeline", "chronolog",
    "incident on", "outage on", "event on", "on the date",
})

_RECENCY_PATTERNS = [
    re.compile(r"\b(?:latest|newest|most recent|current|up.to.date)\b", re.IGNORECASE),
    re.compile(r"\blast\s+(?:week|day|hour|few)\b", re.IGNORECASE),
]

_HISTORICAL_PATTERNS = [
    re.compile(r"\bwhen\s+(?:did|was|were|is)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+happened\b", re.IGNORECASE),
    re.compile(r"\b(?:incident|outage|event|issue)\s+(?:on|from|in|during)\b", re.IGNORECASE),
    re.compile(r"\b(?:on|from|during|before|after)\s+\d{4}-\d{2}", re.IGNORECASE),
    re.compile(r"\bhistor(?:y|ical)\b", re.IGNORECASE),
]


def classify_temporal_intent(query: str) -> str:
    """Classify temporal intent of a query.

    Returns:
        "recency"    — wants the latest/most-current information
        "historical" — looking for specific past events or date-anchored facts
        "neutral"    — no strong temporal signal
    """
    q = query.strip()
    if not q:
        return "neutral"

    has_iso_date = bool(_ISO_DATE_RE.search(q))
    has_month_year = bool(_MONTH_YEAR_RE.search(q))
    has_year = bool(_YEAR_RE.search(q))

    historical_score = 0
    recency_score = 0

    if has_iso_date:
        historical_score += 3
    if has_month_year:
        historical_score += 2
    if has_year and not has_iso_date:
        historical_score += 1

    for pattern in _HISTORICAL_PATTERNS:
        if pattern.search(q):
            historical_score += 2

    for pattern in _RECENCY_PATTERNS:
        if pattern.search(q):
            recency_score += 2

    q_lower = q.lower()
    for word in _RECENCY_WORDS:
        if word in q_lower:
            recency_score += 1

    for word in _HISTORICAL_WORDS:
        if word in q_lower:
            historical_score += 1

    if historical_score >= 3 and historical_score > recency_score:
        return "historical"
    if recency_score >= 2 and recency_score > historical_score:
        return "recency"

    return "neutral"


def extract_query_dates(query: str) -> list[str]:
    """Extract ISO dates or year-month patterns from the query text."""
    dates = _ISO_DATE_RE.findall(query)
    return dates
