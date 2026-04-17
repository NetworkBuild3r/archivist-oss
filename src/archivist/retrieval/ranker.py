"""Learning-to-Rank inference — lightweight feature extraction + model scoring.

Replaces the hand-tuned multiplicative blending of hotness, importance,
temporal decay, and outcome adjustments with a single learned model that
predicts relevance from ~12 features already computed in the pipeline.

Gracefully falls back to the hand-tuned pipeline when no trained model
is available.
"""

import logging
import os
import time
from datetime import UTC, datetime

logger = logging.getLogger("archivist.ranker")

_MODEL_PATH = os.getenv("ARCHIVIST_RANKER_MODEL", "models/ranker.xgb")
_model = None
_model_loaded = False


def _load_model():
    """Attempt to load a trained XGBoost model. No-op if file doesn't exist."""
    global _model, _model_loaded
    _model_loaded = True
    if not os.path.isfile(_MODEL_PATH):
        logger.info("No LTR model at %s — using hand-tuned pipeline", _MODEL_PATH)
        return
    try:
        import xgboost as xgb

        _model = xgb.Booster()
        _model.load_model(_MODEL_PATH)
        logger.info("Loaded LTR model from %s", _MODEL_PATH)
    except ImportError:
        logger.info("xgboost not installed — LTR disabled")
    except Exception as e:
        logger.warning("Failed to load LTR model: %s", e)


def ltr_available() -> bool:
    """Return True if a trained LTR model is loaded and ready."""
    if not _model_loaded:
        _load_model()
    return _model is not None


# ── Feature names (order matters — must match training) ──────────────────────
FEATURE_NAMES = [
    "vector_score",
    "bm25_score",
    "rrf_score",
    "hotness",
    "importance_score",
    "temporal_decay",
    "outcome_adjustment",
    "graph_hop",
    "mention_count",
    "rerank_score",
    "chunk_length",
    "recency_days",
]


def extract_features(result: dict, query_meta: dict | None = None) -> list[float]:
    """Pull the 12-feature vector from a single retrieval result dict.

    All features are already computed by earlier pipeline stages and
    stored on the result dict.  Missing values default to sensible
    neutral values.
    """
    date_str = result.get("date", "")
    recency_days = 0.0
    if date_str:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            recency_days = max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400)
        except (ValueError, TypeError):
            pass

    text = result.get("text", "")
    chunk_length = float(len(text)) if text else 0.0

    return [
        float(result.get("vector_score", result.get("score", 0))),
        float(result.get("bm25_score", 0)),
        float(result.get("rrf_score", 0)),
        float(result.get("hotness", 0)),
        float(
            result.get(
                "importance_score",
                result.get("payload", {}).get("importance_score", 0.5)
                if isinstance(result.get("payload"), dict)
                else 0.5,
            )
        ),
        float(result.get("temporal_decay", 1.0)),
        float(result.get("outcome_adjustment", 0)),
        float(result.get("graph_hop", 0)),
        float(result.get("mention_count", 0)),
        float(result.get("rerank_score", 0)),
        chunk_length,
        recency_days,
    ]


def rank_results(results: list[dict], query_meta: dict | None = None) -> list[dict]:
    """Score and re-sort results using the LTR model.

    If no model is loaded, returns the input unchanged (the caller
    continues with the hand-tuned pipeline).

    Each result gets an ``ltr_score`` field with the model's prediction.
    """
    if not ltr_available() or not results:
        return results

    import numpy as np
    import xgboost as xgb

    features = [extract_features(r, query_meta) for r in results]
    dmatrix = xgb.DMatrix(np.array(features, dtype=np.float32), feature_names=FEATURE_NAMES)

    t0 = time.monotonic()
    scores = _model.predict(dmatrix)
    dur_us = round((time.monotonic() - t0) * 1_000_000, 1)
    logger.debug("LTR inference: %d candidates in %.1fµs", len(results), dur_us)

    for r, s in zip(results, scores):
        r["ltr_score"] = float(s)
        r["score"] = float(s)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
