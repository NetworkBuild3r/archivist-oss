"""Offline LTR training script — trains a ranking model from feedback data.

Reads from:
  - ``ratings`` table (explicit user feedback)
  - ``memory_outcomes`` table (trajectory success/failure)
  - ``retrieval_logs`` table (implicit signals — duration, cache hits)

Produces a lightweight XGBoost LambdaMART model at models/ranker.xgb.

Usage:
    python -m ranker_train [--output models/ranker.xgb] [--min-samples 50]

Can also be invoked from the curator loop or an admin tool.
"""

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger("archivist.ranker_train")

# Feature list must match ranker.py exactly.
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


def _load_training_data(db_path: str) -> tuple[list[list[float]], list[float], list[int]]:
    """Build (features, labels, groups) from stored feedback.

    Returns:
        features: list of 12-element feature vectors
        labels:   relevance grades (0=irrelevant, 1=relevant, 2=highly relevant)
        groups:   group sizes for LambdaMART (each group = one query's candidates)
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        rated: dict[str, int] = {}
        try:
            for row in conn.execute("SELECT memory_id, rating FROM ratings"):
                mid = row["memory_id"]
                grade = 2 if row["rating"] > 0 else 0
                rated[mid] = max(rated.get(mid, 0), grade)
        except Exception:
            pass

        try:
            for row in conn.execute("SELECT memory_id, outcome FROM memory_outcomes"):
                mid = row["memory_id"]
                grade = 2 if row["outcome"] == "success" else 0
                rated[mid] = max(rated.get(mid, 0), grade)
        except Exception:
            pass

        if not rated:
            return [], [], []

        features: list[list[float]] = []
        labels: list[float] = []
        groups: list[int] = []

        try:
            rows = conn.execute(
                "SELECT query, retrieval_trace FROM retrieval_logs "
                "ORDER BY created_at DESC LIMIT 5000"
            ).fetchall()
        except Exception:
            rows = []

        for row in rows:
            try:
                trace = json.loads(row["retrieval_trace"]) if isinstance(row["retrieval_trace"], str) else row["retrieval_trace"]
            except (json.JSONDecodeError, TypeError):
                continue

            candidates = trace.get("candidates", [])
            if not candidates:
                continue

            group_features = []
            group_labels = []
            for cand in candidates:
                mid = cand.get("id", "")
                fvec = [
                    float(cand.get("vector_score", cand.get("score", 0))),
                    float(cand.get("bm25_score", 0)),
                    float(cand.get("rrf_score", 0)),
                    float(cand.get("hotness", 0)),
                    float(cand.get("importance_score", 0.5)),
                    float(cand.get("temporal_decay", 1.0)),
                    float(cand.get("outcome_adjustment", 0)),
                    float(cand.get("graph_hop", 0)),
                    float(cand.get("mention_count", 0)),
                    float(cand.get("rerank_score", 0)),
                    float(len(cand.get("text", ""))),
                    float(cand.get("recency_days", 0)),
                ]
                grade = float(rated.get(mid, 1))
                group_features.append(fvec)
                group_labels.append(grade)

            if len(group_features) >= 2:
                features.extend(group_features)
                labels.extend(group_labels)
                groups.append(len(group_features))

        return features, labels, groups
    finally:
        conn.close()


def train(db_path: str, output_path: str, min_samples: int = 50) -> bool:
    """Train a LambdaMART model and save it.

    Returns True if training succeeded, False otherwise.
    """
    try:
        import xgboost as xgb
        import numpy as np
    except ImportError:
        logger.error("xgboost and numpy are required for training: pip install xgboost numpy")
        return False

    features, labels, groups = _load_training_data(db_path)

    if len(features) < min_samples:
        logger.info(
            "Not enough training data (%d samples, need %d) — skipping training",
            len(features), min_samples,
        )
        return False

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    dtrain.set_group(groups)

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg",
        "eta": 0.1,
        "max_depth": 4,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.1,
        "nthread": 2,
        "verbosity": 0,
    }

    logger.info("Training LTR model: %d samples, %d groups", len(features), len(groups))
    model = xgb.train(params, dtrain, num_boost_round=100)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save_model(output_path)
    logger.info("Saved LTR model to %s (%.1fKB)", output_path, os.path.getsize(output_path) / 1024)
    return True


def main():
    parser = argparse.ArgumentParser(description="Train Archivist LTR ranking model")
    parser.add_argument("--db", default=os.getenv("SQLITE_PATH", "/data/archivist/graph.db"),
                        help="Path to SQLite database")
    parser.add_argument("--output", default="models/ranker.xgb",
                        help="Output model path")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="Minimum training samples required")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    success = train(args.db, args.output, args.min_samples)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
