"""BM25/FTS5 hybrid search — keyword retrieval fused with vector results.

v1.10: Dual-mode BM25 (AND + OR), stopword filtering, RRF fusion.
"""

import logging
from config import BM25_ENABLED, BM25_WEIGHT, VECTOR_WEIGHT
from graph import search_fts
from rank_fusion import rrf_merge
import health

logger = logging.getLogger("archivist.fts")

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "between", "through", "after", "before",
    "and", "or", "but", "not", "no", "nor", "so", "if", "then",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "any", "some",
})


def _clean_token(t: str) -> str:
    return t.strip('"\'()[]{}*:!?,.')


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


def search_bm25(
    query: str,
    namespace: str = "",
    agent_id: str = "",
    memory_type: str = "",
    limit: int = 30,
) -> list[dict]:
    """Run BM25 keyword search via FTS5 in dual AND+OR mode.

    Returns results from both AND (high precision) and OR (high recall)
    queries, merged via RRF to balance precision and recall.
    """
    if not BM25_ENABLED or not health.is_healthy("fts5"):
        return []

    rankings: list[list[dict]] = []

    or_q = _fts5_safe_query(query)
    if or_q:
        or_hits = search_fts(
            query=or_q, namespace=namespace, agent_id=agent_id,
            memory_type=memory_type, limit=limit,
        )
        if or_hits:
            rankings.append(or_hits)

    and_q = _fts5_and_query(query)
    if and_q:
        try:
            and_hits = search_fts(
                query=and_q, namespace=namespace, agent_id=agent_id,
                memory_type=memory_type, limit=limit,
            )
            if and_hits:
                rankings.append(and_hits)
        except Exception:
            pass

    phrase_q = _fts5_phrase_query(query)
    if phrase_q and phrase_q != and_q:
        try:
            phrase_hits = search_fts(
                query=phrase_q, namespace=namespace, agent_id=agent_id,
                memory_type=memory_type, limit=limit,
            )
            if phrase_hits:
                rankings.append(phrase_hits)
        except Exception:
            pass

    if not rankings:
        return []

    merged = rrf_merge(rankings, k=60, id_key="qdrant_id", limit=limit)
    for r in merged:
        if "bm25_score" not in r:
            r["bm25_score"] = r.get("rrf_score", 0)
    return merged


def merge_vector_and_bm25(
    vector_results: list[dict],
    bm25_results: list[dict],
) -> list[dict]:
    """Fuse vector and BM25 results using RRF.

    Replaces the old linear 0.7v + 0.3b fusion with rank-based fusion
    which is more robust when score distributions differ.
    """
    if not bm25_results:
        return vector_results
    if not vector_results:
        out = []
        for r in bm25_results:
            entry = dict(r)
            entry["score"] = entry.get("bm25_score", 0) * BM25_WEIGHT
            out.append(entry)
        return sorted(out, key=lambda x: x["score"], reverse=True)

    rankings = [vector_results, bm25_results]
    merged = rrf_merge(rankings, k=60)

    for r in merged:
        r["score"] = r["rrf_score"]
        if "bm25_score" not in r:
            r["bm25_score"] = 0

    return merged
