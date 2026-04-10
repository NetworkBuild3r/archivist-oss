"""HyDE — Hypothetical Document Embedding for factoid/needle queries.

When a user asks a question, the embedding of the question sits in a
different region of vector space than the embedding of the answer.
HyDE bridges this gap by generating a short hypothetical answer and
embedding that alongside the original query.

Only activated for queries that look like factoid/needle requests
(configurable heuristic) or when the primary search scores low.
"""

import logging
import re
import time
import threading
from collections import OrderedDict

from config import LLM_REFINE_MODEL, LLM_MODEL
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


def is_needle_query(query: str) -> bool:
    """Heuristic: does this query look like it's seeking a specific fact?"""
    for pat in _NEEDLE_PATTERNS:
        if pat.search(query):
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
        m.observe("hyde_duration_ms", dur_ms)
        doc = doc.strip()
        if doc:
            _cache_put(query, doc)
            logger.debug("hyde dur_ms=%.1f doc=%r", dur_ms, doc[:100])
            return doc
        return None
    except Exception as e:
        logger.warning("HyDE generation failed: %s", e)
        return None
