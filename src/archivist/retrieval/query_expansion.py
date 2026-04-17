"""Multi-query expansion — generate alternative search queries via LLM.

Bridges the vocabulary gap between how users phrase questions and how
facts are stored.  Each expanded variant is embedded and searched
independently; results are merged with RRF in the retrieval pipeline.

The expansion prompt is deliberately compact to keep latency low
(typically 50-150ms on a fast model).
"""

import logging
import time
import threading
from collections import OrderedDict

from archivist.core.config import (
    LLM_REFINE_MODEL,
    LLM_MODEL,
)
from archivist.features.llm import llm_query
import archivist.core.metrics as m

logger = logging.getLogger("archivist.query_expansion")

QUERY_EXPANSION_SYSTEM = (
    "You are a search query expander. Given a user query, generate {count} alternative "
    "search queries that would find the same information using different phrasing. "
    "Include both natural language questions and keyword-style queries. "
    "Return ONLY the queries, one per line. No numbering, no explanations."
)

# In-process expansion cache (keyed by query text, TTL-bounded).
_EXPANSION_CACHE_MAX = 512
_EXPANSION_CACHE_TTL = 600  # 10 minutes
_expansion_cache: OrderedDict[str, tuple[float, list[str]]] = OrderedDict()
_expansion_lock = threading.Lock()


def _cache_get(query: str) -> list[str] | None:
    with _expansion_lock:
        entry = _expansion_cache.get(query)
        if entry is None:
            return None
        ts, variants = entry
        if time.monotonic() - ts > _EXPANSION_CACHE_TTL:
            _expansion_cache.pop(query, None)
            return None
        _expansion_cache.move_to_end(query)
        return list(variants)


def _cache_put(query: str, variants: list[str]) -> None:
    with _expansion_lock:
        _expansion_cache[query] = (time.monotonic(), list(variants))
        _expansion_cache.move_to_end(query)
        while len(_expansion_cache) > _EXPANSION_CACHE_MAX:
            _expansion_cache.popitem(last=False)


async def expand_query(
    query: str,
    count: int = 3,
    model: str = "",
) -> list[str]:
    """Generate ``count`` alternative query phrasings via LLM.

    Returns the original query plus the expansions (so the caller always
    gets at least 1 query back even if expansion fails).

    Results are cached in-process to avoid redundant LLM calls for
    repeated queries.
    """
    cached = _cache_get(query)
    if cached is not None:
        return cached

    expansion_model = model or LLM_REFINE_MODEL or LLM_MODEL
    system = QUERY_EXPANSION_SYSTEM.format(count=count)

    t0 = time.monotonic()
    try:
        raw = await llm_query(
            query,
            system=system,
            max_tokens=256,
            model=expansion_model,
            stage="query_expansion",
        )
        dur_ms = round((time.monotonic() - t0) * 1000, 1)
        m.observe(m.QUERY_EXPANSION_DURATION, dur_ms)

        lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        # Remove numbering if the model adds it ("1. ...", "- ...")
        cleaned = []
        for line in lines:
            if line and line[0].isdigit() and (". " in line[:4] or ") " in line[:4]):
                line = line.split(". ", 1)[-1] if ". " in line[:4] else line.split(") ", 1)[-1]
            elif line.startswith("- "):
                line = line[2:]
            line = line.strip().strip('"').strip("'")
            if line and line.lower() != query.lower():
                cleaned.append(line)

        variants = [query] + cleaned[:count]
        _cache_put(query, variants)
        logger.debug(
            "query_expansion count=%d dur_ms=%.1f original=%r variants=%r",
            len(variants) - 1, dur_ms, query, cleaned[:count],
        )
        return variants

    except Exception as e:
        logger.warning("Query expansion failed: %s — using original query only", e)
        return [query]
