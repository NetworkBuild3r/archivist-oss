"""BM25/FTS5 hybrid search — keyword retrieval fused with vector results.

Fusion formula: fused = VECTOR_WEIGHT * norm(vector) + BM25_WEIGHT * norm(bm25)
"""

import logging
from config import BM25_ENABLED, BM25_WEIGHT, VECTOR_WEIGHT
from graph import search_fts

logger = logging.getLogger("archivist.fts")


def _fts5_safe_query(raw_query: str) -> str:
    """Sanitize a natural language query for FTS5 MATCH syntax.

    Wraps each token in double quotes to prevent FTS5 syntax errors from
    special characters (AND, OR, NOT, parentheses, colons, asterisks).
    """
    tokens = raw_query.strip().split()
    if not tokens:
        return ""
    safe = []
    for t in tokens:
        cleaned = t.strip('"\'()[]{}*:')
        if cleaned and len(cleaned) >= 2:
            safe.append(f'"{cleaned}"')
    return " OR ".join(safe) if safe else ""


def search_bm25(
    query: str,
    namespace: str = "",
    agent_id: str = "",
    memory_type: str = "",
    limit: int = 30,
) -> list[dict]:
    """Run a BM25 keyword search via FTS5. Returns [] if BM25 disabled."""
    if not BM25_ENABLED:
        return []

    safe_q = _fts5_safe_query(query)
    if not safe_q:
        return []

    return search_fts(
        query=safe_q,
        namespace=namespace,
        agent_id=agent_id,
        memory_type=memory_type,
        limit=limit,
    )


def merge_vector_and_bm25(
    vector_results: list[dict],
    bm25_results: list[dict],
) -> list[dict]:
    """Fuse vector and BM25 results using weighted score normalization.

    Both inputs should have a "score" field (vector) or "bm25_score" (keyword).
    Deduplicates on qdrant_id, preferring the fused score.
    """
    if not bm25_results:
        return vector_results
    if not vector_results:
        for r in bm25_results:
            r["score"] = r.get("bm25_score", 0) * BM25_WEIGHT
        return sorted(bm25_results, key=lambda x: x["score"], reverse=True)

    v_max = max((r.get("score", 0) for r in vector_results), default=1.0) or 1.0
    b_max = max((r.get("bm25_score", 0) for r in bm25_results), default=1.0) or 1.0

    merged: dict[str, dict] = {}

    for r in vector_results:
        key = str(r.get("id", r.get("qdrant_id", id(r))))
        norm_v = r.get("score", 0) / v_max
        merged[key] = {**r, "score": VECTOR_WEIGHT * norm_v, "vector_score": r.get("score", 0)}

    for r in bm25_results:
        key = str(r.get("qdrant_id", id(r)))
        norm_b = r.get("bm25_score", 0) / b_max
        bm25_contrib = BM25_WEIGHT * norm_b

        if key in merged:
            merged[key]["score"] += bm25_contrib
            merged[key]["bm25_score"] = r.get("bm25_score", 0)
        else:
            merged[key] = {
                "id": r.get("qdrant_id", ""),
                "score": bm25_contrib,
                "bm25_score": r.get("bm25_score", 0),
                "text": r.get("text", ""),
                "l0": "",
                "l1": "",
                "agent_id": r.get("agent_id", ""),
                "file_path": r.get("file_path", ""),
                "file_type": "",
                "date": r.get("date", ""),
                "team": "",
                "namespace": r.get("namespace", ""),
                "chunk_index": r.get("chunk_index", 0),
                "parent_id": None,
                "is_parent": False,
            }

    results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return results
