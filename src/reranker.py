"""Cross-encoder reranker for second-stage retrieval refinement.

Requires `sentence-transformers` (optional dependency).  When the model
is unavailable the reranker degrades gracefully — results pass through
sorted by their original vector score.
"""

import logging
from typing import Optional

logger = logging.getLogger("archivist.reranker")

_model: Optional[object] = None
_load_failed = False


def _get_model(model_name: str):
    """Lazy-load the CrossEncoder model once."""
    global _model, _load_failed
    if _load_failed:
        return None
    if _model is not None:
        return _model
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(model_name)
        logger.info("Loaded reranker model: %s", model_name)
        return _model
    except Exception as e:
        logger.warning("Failed to load reranker model '%s': %s — reranking disabled", model_name, e)
        _load_failed = True
        return None


def rerank_results(
    query: str,
    results: list[dict],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    limit: int = 10,
) -> list[dict]:
    """Re-score results with a cross-encoder and return the top `limit`.

    Each result dict must have a "text" key. A "rerank_score" key is added.
    If the model is unavailable, results are returned sorted by original "score".
    """
    if not results:
        return results

    model = _get_model(model_name)
    if model is None:
        # Graceful fallback: return top-limit by original vector score
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:limit]

    pairs = [[query, r["text"]] for r in results]
    scores = model.predict(pairs)

    for i, r in enumerate(results):
        r["rerank_score"] = float(scores[i])

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:limit]
