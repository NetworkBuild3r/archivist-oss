"""HyDE — Hypothetical Document Embedding for factoid/needle queries.

When a user asks a question, the embedding of the question sits in a
different region of vector space than the embedding of the answer.
HyDE bridges this gap by generating a short hypothetical answer and
embedding that alongside the original query.

Only activated for queries that look like factoid/needle requests
(configurable heuristic) or when the primary search scores low.
"""

import hashlib
import logging
import re
import time
import threading
from collections import OrderedDict

from config import LLM_REFINE_MODEL, LLM_MODEL, REVERSE_HYDE_ENABLED, REVERSE_HYDE_QUESTIONS_PER_CHUNK
from llm import llm_query
import metrics as m

logger = logging.getLogger("archivist.hyde")

_HYDE_SYSTEM = (
    "Answer the following question in one concise sentence as if you know the exact answer. "
    "Do not hedge or say you are unsure. Just state the answer directly."
)

_NEEDLE_PATTERNS = [
    re.compile(r"\bexact\b", re.I),
    re.compile(r"\bspecific\b", re.I),
    re.compile(r"\bwhat is the\b", re.I),
    re.compile(r"\bwhat was the\b", re.I),
    re.compile(r"\bwhat are the\b", re.I),
    re.compile(r"\bfind the\b", re.I),
    re.compile(r"\bwho is\b", re.I),
    re.compile(r"\bwhen is\b", re.I),
    re.compile(r"\bwhen was\b", re.I),
    re.compile(r"\bwhere is\b", re.I),
    re.compile(r"\bIP\s+address\b", re.I),
    re.compile(r"\bport\b", re.I),
    re.compile(r"\bcron\b", re.I),
    re.compile(r"\bpassword\b", re.I),
    re.compile(r"\bbudget\b", re.I),
    re.compile(r"\blaunch date\b", re.I),
    re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"),  # IP literal
]

# Cache hypothetical documents to avoid redundant LLM calls.
_HYDE_CACHE_MAX = 256
_HYDE_CACHE_TTL = 600  # 10 minutes
_hyde_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
_hyde_lock = threading.Lock()


def is_needle_query(query: str, entity_count: int = -1) -> bool:
    """Heuristic: does this query look like it's seeking a specific fact?

    Tier 1: Fast regex check against known needle patterns.
    Tier 2: Short queries (< 15 tokens) with no known entity match
    are assumed to be potential needles — activates HyDE as a safety net.
    Pass ``entity_count`` from graph retrieval when available; -1 skips tier 2.
    """
    for pat in _NEEDLE_PATTERNS:
        if pat.search(query):
            return True
    if entity_count >= 0:
        tokens = query.strip().split()
        if len(tokens) < 15 and entity_count == 0:
            return True
    return False


def _cache_get(query: str) -> str | None:
    with _hyde_lock:
        entry = _hyde_cache.get(query)
        if entry is None:
            return None
        ts, doc = entry
        if time.monotonic() - ts > _HYDE_CACHE_TTL:
            _hyde_cache.pop(query, None)
            return None
        _hyde_cache.move_to_end(query)
        return doc


def _cache_put(query: str, doc: str) -> None:
    with _hyde_lock:
        _hyde_cache[query] = (time.monotonic(), doc)
        _hyde_cache.move_to_end(query)
        while len(_hyde_cache) > _HYDE_CACHE_MAX:
            _hyde_cache.popitem(last=False)


async def generate_hypothetical_document(
    query: str,
    model: str = "",
) -> str | None:
    """Generate a short hypothetical answer to ``query``.

    Returns the hypothetical document text, or None on failure.
    Cached in-process to avoid repeated LLM calls for the same query.
    """
    cached = _cache_get(query)
    if cached is not None:
        return cached

    hyde_model = model or LLM_REFINE_MODEL or LLM_MODEL
    t0 = time.monotonic()
    try:
        doc = await llm_query(
            query,
            system=_HYDE_SYSTEM,
            max_tokens=128,
            model=hyde_model,
            stage="hyde",
        )
        dur_ms = round((time.monotonic() - t0) * 1000, 1)
        m.observe(m.HYDE_DURATION, dur_ms)
        doc = doc.strip()
        if doc:
            _cache_put(query, doc)
            logger.debug("hyde dur_ms=%.1f doc=%r", dur_ms, doc[:100])
            return doc
        return None
    except Exception as e:
        logger.warning("HyDE generation failed: %s", e)
        return None


# ── Reverse HyDE: write-time hypothetical question generation (v2.0) ─────────

_REVERSE_HYDE_SYSTEM = (
    "You are a search query generator. Given a piece of factual content, "
    "generate {count} diverse questions that someone might ask to find this information. "
    "Include both natural language questions and keyword-style queries. "
    "Focus on specific facts, names, numbers, identifiers, and dates in the content. "
    "Output one question per line. No numbering, no bullets."
)

_REVERSE_HYDE_CACHE_MAX = 512
_REVERSE_HYDE_CACHE_TTL = 3600
_reverse_hyde_cache: OrderedDict[str, tuple[float, list[str]]] = OrderedDict()
_reverse_hyde_lock = threading.Lock()


def _reverse_cache_get(text_key: str) -> list[str] | None:
    with _reverse_hyde_lock:
        entry = _reverse_hyde_cache.get(text_key)
        if entry is None:
            return None
        ts, questions = entry
        if time.monotonic() - ts > _REVERSE_HYDE_CACHE_TTL:
            _reverse_hyde_cache.pop(text_key, None)
            return None
        _reverse_hyde_cache.move_to_end(text_key)
        return questions


def _reverse_cache_put(text_key: str, questions: list[str]) -> None:
    with _reverse_hyde_lock:
        _reverse_hyde_cache[text_key] = (time.monotonic(), questions)
        _reverse_hyde_cache.move_to_end(text_key)
        while len(_reverse_hyde_cache) > _REVERSE_HYDE_CACHE_MAX:
            _reverse_hyde_cache.popitem(last=False)


async def generate_reverse_hyde_questions(
    text: str,
    count: int = 0,
    model: str = "",
) -> list[str]:
    """Generate hypothetical questions that ``text`` would answer.

    Used at write/index time to create question-embedding vectors
    that live in the same semantic space as future user queries.
    Returns up to ``count`` questions, or empty list on failure.
    """
    if not REVERSE_HYDE_ENABLED:
        return []
    if not count:
        count = REVERSE_HYDE_QUESTIONS_PER_CHUNK

    cache_key = hashlib.md5(text.encode()).hexdigest()
    cached = _reverse_cache_get(cache_key)
    if cached is not None:
        return cached[:count]

    hyde_model = model or LLM_REFINE_MODEL or LLM_MODEL
    system = _REVERSE_HYDE_SYSTEM.replace("{count}", str(count))
    t0 = time.monotonic()
    try:
        raw = await llm_query(
            text[:1500],
            system=system,
            max_tokens=256,
            model=hyde_model,
            stage="reverse_hyde",
        )
        dur_ms = round((time.monotonic() - t0) * 1000, 1)
        m.observe(m.REVERSE_HYDE_DURATION, dur_ms)

        questions = [
            q.strip().lstrip("0123456789.-) ")
            for q in raw.strip().splitlines()
            if q.strip() and len(q.strip()) > 5
        ][:count]

        if questions:
            _reverse_cache_put(cache_key, questions)
            logger.debug("reverse_hyde dur_ms=%.1f count=%d", dur_ms, len(questions))
        return questions
    except Exception as e:
        logger.warning("Reverse HyDE generation failed: %s", e)
        return []
