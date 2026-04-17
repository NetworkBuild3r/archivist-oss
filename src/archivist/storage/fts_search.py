"""BM25/FTS5 hybrid search — keyword retrieval fused with vector results.

v1.10: Dual-mode BM25 (AND + OR), stopword filtering, RRF fusion.
v1.11: Non-stemmed exact table for identifier/IP token matching.
"""

import logging
import re

import archivist.core.health as health
from archivist.core.config import BM25_ENABLED, BM25_WEIGHT
from archivist.retrieval.rank_fusion import rrf_merge
from archivist.storage.graph import search_fts, search_fts_exact

logger = logging.getLogger("archivist.fts")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "into",
        "about",
        "between",
        "through",
        "after",
        "before",
        "and",
        "or",
        "but",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "all",
        "each",
        "every",
        "any",
        "some",
    }
)

_EXACT_TOKEN_RE = re.compile(
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"  # IP addresses
    r"|[A-Z]{2,}-\d{4,}"  # ticket/employee IDs
    r"|[0-9a-f]{8}-[0-9a-f]{4}"  # UUID prefix
    r"|:\d{2,5}\b"  # port numbers
    r"|[\d,]+\s*(?:MiB|GiB|KiB|ms|KB|MB|GB)",  # numeric with units
    re.I,
)


def _clean_token(t: str) -> str:
    return t.strip("\"'()[]{}*:!?,.")


def _fts5_safe_query(raw_query: str) -> str:
    """OR-mode: each content token quoted, joined with OR. High recall."""
    tokens = raw_query.strip().split()
    if not tokens:
        return ""
    safe = []
    for t in tokens:
        cleaned = _clean_token(t)
        if cleaned and len(cleaned) >= 1:
            safe.append(f'"{cleaned}"')
    return " OR ".join(safe) if safe else ""


def _fts5_and_query(raw_query: str) -> str:
    """AND-mode: stopwords removed, remaining tokens ANDed. High precision."""
    tokens = raw_query.strip().split()
    if not tokens:
        return ""
    content_tokens = []
    for t in tokens:
        cleaned = _clean_token(t).lower()
        if cleaned and len(cleaned) >= 2 and cleaned not in _STOPWORDS:
            content_tokens.append(f'"{cleaned}"')
    if len(content_tokens) < 2:
        return ""
    return " AND ".join(content_tokens)


def _fts5_phrase_query(raw_query: str) -> str:
    """Phrase-mode: entire query as a phrase match. Highest precision."""
    tokens = raw_query.strip().split()
    if not tokens:
        return ""
    content_tokens = []
    for t in tokens:
        cleaned = _clean_token(t).lower()
        if cleaned and len(cleaned) >= 2 and cleaned not in _STOPWORDS:
            content_tokens.append(cleaned)
    if len(content_tokens) < 2:
        return ""
    return '"' + " ".join(content_tokens) + '"'


async def search_bm25(
    query: str,
    namespace: str = "",
    agent_id: str = "",
    memory_type: str = "",
    limit: int = 30,
    actor_type: str = "",
) -> list[dict]:
    """Run BM25 keyword search via FTS5 in dual AND+OR mode.

    Returns results from both AND (high precision) and OR (high recall)
    queries, merged via RRF to balance precision and recall.
    """
    if not BM25_ENABLED or not health.is_healthy("fts5"):
        return []

    rankings: list[list[dict]] = []

    or_q = _fts5_safe_query(query)
    or_hits: list[dict] = []
    if or_q:
        or_hits = await search_fts(
            query=or_q,
            namespace=namespace,
            agent_id=agent_id,
            memory_type=memory_type,
            limit=limit,
            actor_type=actor_type,
        )
        if or_hits:
            rankings.append(or_hits)

    # On small indices OR already covers most chunks; extra modes add noise.
    _run_extra_modes = len(or_hits) >= 200

    if _run_extra_modes:
        and_q = _fts5_and_query(query)
        if and_q:
            try:
                and_hits = await search_fts(
                    query=and_q,
                    namespace=namespace,
                    agent_id=agent_id,
                    memory_type=memory_type,
                    limit=limit,
                    actor_type=actor_type,
                )
                if and_hits:
                    rankings.append(and_hits)
            except Exception:
                pass

        phrase_q = _fts5_phrase_query(query)
        if phrase_q and phrase_q != and_q:
            try:
                phrase_hits = await search_fts(
                    query=phrase_q,
                    namespace=namespace,
                    agent_id=agent_id,
                    memory_type=memory_type,
                    limit=limit,
                    actor_type=actor_type,
                )
                if phrase_hits:
                    rankings.append(phrase_hits)
            except Exception:
                pass

        if _EXACT_TOKEN_RE.search(query):
            exact_q = _fts5_safe_query(query)
            if exact_q:
                try:
                    exact_hits = await search_fts_exact(
                        query=exact_q,
                        namespace=namespace,
                        agent_id=agent_id,
                        memory_type=memory_type,
                        limit=limit,
                        actor_type=actor_type,
                    )
                    if exact_hits:
                        rankings.append(exact_hits)
                except Exception:
                    pass
    else:
        # Always run exact-token search even on small indices (needle recall)
        if _EXACT_TOKEN_RE.search(query):
            exact_q = _fts5_safe_query(query)
            if exact_q:
                try:
                    exact_hits = await search_fts_exact(
                        query=exact_q,
                        namespace=namespace,
                        agent_id=agent_id,
                        memory_type=memory_type,
                        limit=limit,
                        actor_type=actor_type,
                    )
                    if exact_hits:
                        rankings.append(exact_hits)
                except Exception:
                    pass

    if not rankings:
        return []

    merged = rrf_merge(rankings, k=20, id_key="qdrant_id", limit=limit)
    for r in merged:
        if "bm25_score" not in r:
            r["bm25_score"] = r.get("rrf_score", 0)
    return merged


def merge_vector_and_bm25(
    vector_results: list[dict],
    bm25_results: list[dict],
) -> list[dict]:
    """Fuse vector and BM25 results using RRF with vector-priority rescue.

    Preserves original vector scores through fusion so downstream stages
    operate on meaningful signal instead of flat RRF noise.
    """
    if not bm25_results:
        return vector_results[:20]
    if not vector_results:
        out = []
        for r in bm25_results:
            entry = dict(r)
            entry["score"] = entry.get("bm25_score", 0) * BM25_WEIGHT
            out.append(entry)
        return sorted(out, key=lambda x: x["score"], reverse=True)

    vector_id_scores: dict[str, float] = {}
    for r in vector_results:
        rid = r.get("id") or r.get("qdrant_id", "")
        if rid:
            vector_id_scores[rid] = r.get("score", 0)

    rankings = [vector_results, bm25_results]
    merged = rrf_merge(rankings, k=20)

    vector_top_ids = {(r.get("id") or r.get("qdrant_id", "")) for r in vector_results[:8]}
    for r in merged:
        rid = r.get("id") or r.get("qdrant_id", "")
        r["vector_score"] = vector_id_scores.get(rid, 0)
        r["score"] = r["rrf_score"]
        if rid in vector_top_ids:
            r["score"] = max(r["score"], r["vector_score"] + 0.5)
        if "bm25_score" not in r:
            r["bm25_score"] = 0

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:20]
