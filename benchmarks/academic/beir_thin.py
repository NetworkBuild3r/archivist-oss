"""Thin BEIR dense baseline (NFCorpus by default).

This is **not** the full Archivist RLM pipeline — it runs the standard BEIR
bi-encoder retrieval + nDCG/Recall using ``beir.retrieval.models.SentenceBERT``.
Use it for a compact, comparable IR number (same class of metric as BEIR papers).

Requires: ``pip install -r requirements-benchmark.txt``
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger("archivist.benchmark.beir_thin")

# Official BEIR download URLs (UKP)
BEIR_DATASETS = {
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
}


def _run(
    dataset: str,
    data_root: str,
    model_name: str,
    limit_queries: int,
    batch_size: int,
    split: str,
) -> dict:
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval import models
        from beir.retrieval.evaluation import EvaluateRetrieval
        from beir.retrieval.search.dense import DenseRetrievalExactSearch
    except ImportError as e:
        return {
            "error": "beir not installed",
            "hint": "pip install -r requirements-benchmark.txt",
            "detail": str(e),
        }

    if dataset not in BEIR_DATASETS:
        return {"error": f"unknown dataset {dataset}", "valid": list(BEIR_DATASETS.keys())}

    ds_path = os.path.join(data_root, dataset)
    os.makedirs(ds_path, exist_ok=True)
    zip_url = BEIR_DATASETS[dataset]
    data_path = util.download_and_unzip(zip_url, ds_path)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

    if limit_queries > 0:
        q_items = list(queries.items())[:limit_queries]
        q_ids = {k for k, _ in q_items}
        queries = dict(q_items)
        qrels = {k: v for k, v in qrels.items() if k in q_ids}

    model = models.SentenceBERT(model_name)
    dense = DenseRetrievalExactSearch(model, batch_size=batch_size)
    k_values = [1, 3, 5, 10]
    evaluator = EvaluateRetrieval(dense, k_values=k_values)

    results = evaluator.retrieve(corpus, queries)
    # BEIR defaults to ignore_identical_ids=True and logs INFO every time; NFCorpus
    # uses disjoint q/doc ids so False matches typical BEIR reporting and silences that line.
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels, results, k_values, ignore_identical_ids=False
    )

    summary = {
        "benchmark": "BEIR",
        "mode": "dense_bi_encoder",
        "dataset": dataset,
        "split": split,
        "model": model_name,
        "corpus_size": len(corpus),
        "queries_evaluated": len(queries),
        "note": "Dense baseline only — not Archivist hybrid RLM; compare embedding families.",
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
    }
    return {"summary": summary}


def main() -> None:
    p = argparse.ArgumentParser(description="Thin BEIR dense baseline (NFCorpus / SciFact)")
    p.add_argument("--dataset", default="nfcorpus", choices=list(BEIR_DATASETS.keys()))
    p.add_argument(
        "--data-root",
        default="data/beir",
        help="Directory for downloaded corpora (created if missing)",
    )
    p.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model id (384-dim; override to match your embedder)",
    )
    p.add_argument("--limit-queries", type=int, default=50, help="0 = all queries in split")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--split", default="test", help="BEIR split (usually test)")
    p.add_argument(
        "--output",
        default=".benchmarks/beir_thin.json",
        help="JSON output path",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = _run(
        dataset=args.dataset,
        data_root=args.data_root,
        model_name=args.model,
        limit_queries=args.limit_queries,
        batch_size=args.batch_size,
        split=args.split,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if "error" in out:
        logger.error("%s", out)
        sys.exit(1)

    s = out["summary"]
    print()
    print("=" * 60)
    print(f"  BEIR thin — {s['dataset']}  ({s['queries_evaluated']} queries)")
    print(f"  Model: {s['model']}")
    print("=" * 60)
    for k, v in s.get("ndcg", {}).items():
        print(f"  {k}: {v:.4f}")
    print("=" * 60)
    print(f"  Written: {args.output}")
    print()


if __name__ == "__main__":
    main()
