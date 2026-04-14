"""Cross-encoder reranker — Phase 2 of the v2 retrieval architecture.

Replaces the multi-stage scoring pipeline (temporal decay, hotness, threshold,
rescue) with a single model that takes (query, passage) and returns a calibrated
relevance score.

Requires ``sentence-transformers`` (optional dependency).  When the model is
unavailable the reranker degrades gracefully — candidates pass through sorted
by their original vector score.

Input construction per candidate:
    Query: {query}

    Context: {chunk_text}

    Parent context: {parent_text or ''}

The cross-encoder sees both the chunk and its stored parent text so it can judge
relevance without a separate parent-enrichment stage.
"""

import asyncio
import logging
import time
from typing import Optional

import metrics as m

logger = logging.getLogger("archivist.reranker")

_model: Optional[object] = None
_model_name: str = ""
_load_failed = False


def _get_model(model_name: str):
    """Lazy-load the CrossEncoder model once.  Thread-safe via GIL."""
    global _model, _model_name, _load_failed
    if _load_failed:
        return None
    if _model is not None and _model_name == model_name:
        return _model
    try:
        from sentence_transformers import CrossEncoder
        t0 = time.monotonic()
        _model = CrossEncoder(model_name)
        _model_name = model_name
        dur = round((time.monotonic() - t0) * 1000)
        logger.info("Loaded reranker model %s in %d ms", model_name, dur)
        return _model
    except Exception as e:
        logger.warning(
            "Failed to load reranker model '%s': %s — reranking disabled",
            model_name, e,
        )
        _load_failed = True
        return None


def _build_pair(query: str, candidate: dict) -> str:
    """Construct the passage side of a (query, passage) pair for the cross-encoder."""
    chunk_text = candidate.get("text", "")
    parent_ctx = candidate.get("parent_text", "") or candidate.get("parent_context", "")
    parts = [f"Context: {chunk_text}"]
    if parent_ctx:
        parts.append(f"Parent context: {parent_ctx}")
    return "\n\n".join(parts)


async def rerank_candidates(
    query: str,
    candidates: list[dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 20,
) -> list[dict]:
    """Score candidates with a cross-encoder and return the top ``top_k``.

    Each candidate dict must have a ``text`` key.  ``parent_text`` (stored at
    index time) is used when present.  A ``reranker_score`` key (float 0-1) is
    added to every candidate that was scored.

    If the model is unavailable, candidates are returned sorted by original
    ``score`` (graceful degradation).  The CPU-bound ``model.predict()`` is
    offloaded to a thread so the event loop stays responsive.
    """
    if not candidates:
        return candidates

    model = _get_model(model_name)
    if model is None:
        return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    pairs = [[f"Query: {query}", _build_pair(query, c)] for c in candidates]

    t0 = time.monotonic()
    raw_scores = await asyncio.to_thread(model.predict, pairs)
    dur_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.debug("reranker.predict n=%d dur_ms=%.1f", len(candidates), dur_ms)

    for i, c in enumerate(candidates):
        score = float(raw_scores[i])
        c["reranker_score"] = score

    candidates.sort(key=lambda x: x["reranker_score"], reverse=True)
    return candidates[:top_k]


# Legacy API — kept for backward compat with existing RERANK_ENABLED code path
async def rerank_results(
    query: str,
    results: list[dict],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    limit: int = 10,
) -> list[dict]:
    """Re-score results with a cross-encoder and return the top ``limit``.

    This is the legacy API used by the old RERANK_ENABLED code path.
    Each result dict must have a ``text`` key.  A ``rerank_score`` key is added.
    """
    if not results:
        return results

    model = _get_model(model_name)
    if model is None:
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:limit]

    pairs = [[query, r["text"]] for r in results]
    scores = await asyncio.to_thread(model.predict, pairs)

    for i, r in enumerate(results):
        r["rerank_score"] = float(scores[i])

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:limit]
