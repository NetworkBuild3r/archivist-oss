"""Pipeline ablation harness — runs the Archivist retrieval pipeline under
various configuration variants and computes Recall@K, MRR, latency, and token cost.

Prerequisites:
    - Qdrant running (docker compose up -d qdrant)
    - Embedding API reachable (LLM_URL / EMBED_URL configured)
    - Seed corpus indexed: python -m benchmarks.pipeline.evaluate --index-only

Usage:
    # Index corpus then run all ablation variants:
    python -m benchmarks.pipeline.evaluate

    # Run a single variant:
    python -m benchmarks.pipeline.evaluate --variant vector_only

    # Skip indexing (already done):
    python -m benchmarks.pipeline.evaluate --skip-index

    # Output to file:
    python -m benchmarks.pipeline.evaluate --output results.json
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
CORPUS_DIR = os.path.join(FIXTURES_DIR, "corpus")
QUESTIONS_PATH = os.path.join(FIXTURES_DIR, "questions.json")
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

logger = logging.getLogger("archivist.benchmark.pipeline")

VARIANTS = {
    "vector_only": {
        "BM25_ENABLED": "false",
        "GRAPH_RETRIEVAL_ENABLED": "false",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "0",
    },
    "plus_bm25": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "false",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "0",
    },
    "plus_graph": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "0",
    },
    "plus_temporal": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "30",
    },
    "plus_hotness": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "30",
    },
    "plus_rerank": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "true",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "30",
    },
    "full_pipeline": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "true",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "30",
    },
}


def _apply_variant(variant_name: str):
    """Apply config overrides for a variant by setting env vars and reloading config."""
    env_overrides = VARIANTS[variant_name]
    for key, value in env_overrides.items():
        os.environ[key] = value

    import importlib
    import config
    importlib.reload(config)

    for attr, val in env_overrides.items():
        if hasattr(config, attr):
            config_val = val
            if val in ("true", "false"):
                config_val = val == "true"
            elif val.replace(".", "").replace("-", "").isdigit():
                config_val = float(val) if "." in val else int(val)
            setattr(config, attr, config_val)


def _keyword_recall(result_text: str, expected_keywords: list[str]) -> float:
    """Compute keyword recall: fraction of expected keywords found in result."""
    if not expected_keywords:
        return 0.0
    text_lower = result_text.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return found / len(expected_keywords)


def _reciprocal_rank(result_sources: list[dict], expected_answer: str) -> float:
    """Compute reciprocal rank: 1/rank of first relevant source."""
    if not expected_answer:
        return 0.0
    answer_words = set(expected_answer.lower().split())
    for rank, source in enumerate(result_sources, 1):
        source_text = str(source)
        source_words = set(source_text.lower().split())
        overlap = len(answer_words & source_words) / max(len(answer_words), 1)
        if overlap > 0.3:
            return 1.0 / rank
    return 0.0


async def index_corpus():
    """Index the seed corpus files into Qdrant and FTS5."""
    import config
    config.MEMORY_ROOT = CORPUS_DIR

    from indexer import full_index
    logger.info("Indexing corpus from %s ...", CORPUS_DIR)
    count = await full_index(hierarchical=True)
    logger.info("Indexed %d chunks from corpus", count)
    return count


async def run_variant(variant_name: str, questions: list[dict], refine: bool) -> dict:
    """Run all questions against a pipeline variant and collect metrics."""
    _apply_variant(variant_name)

    import hot_cache
    hot_cache.invalidate_all()

    from rlm_retriever import recursive_retrieve
    from tokenizer import count_tokens

    results = []
    latencies = []
    token_costs = []
    recall_at_5 = []
    recall_at_10 = []
    mrr_scores = []

    for q in questions:
        t0 = time.monotonic()
        try:
            result = await recursive_retrieve(
                query=q["query"],
                namespace="",
                limit=10,
                refine=refine,
                tier="l2",
            )
        except Exception as e:
            logger.warning("Query %d failed: %s", q["id"], e)
            result = {"answer": "", "sources": [], "retrieval_trace": {}}
        elapsed_ms = (time.monotonic() - t0) * 1000
        latencies.append(elapsed_ms)

        answer_text = result.get("answer", "")
        sources = result.get("sources", [])
        trace = result.get("retrieval_trace", {})

        token_cost = count_tokens(answer_text) if answer_text else 0
        for s in sources:
            token_cost += count_tokens(str(s))
        token_costs.append(token_cost)

        kw_recall = _keyword_recall(
            answer_text + " " + " ".join(str(s) for s in sources),
            q.get("expected_keywords", []),
        )
        recall_at_5.append(min(kw_recall, 1.0))

        full_text = answer_text + " " + " ".join(str(s) for s in sources[:10])
        kw_recall_10 = _keyword_recall(full_text, q.get("expected_keywords", []))
        recall_at_10.append(min(kw_recall_10, 1.0))

        rr = _reciprocal_rank(sources, q.get("expected_answer", ""))
        mrr_scores.append(rr)

        results.append({
            "question_id": q["id"],
            "query": q["query"],
            "query_type": q.get("query_type", ""),
            "latency_ms": round(elapsed_ms, 1),
            "token_cost": token_cost,
            "keyword_recall": round(kw_recall, 3),
            "reciprocal_rank": round(rr, 3),
            "sources_count": len(sources),
            "trace": trace,
        })

    summary = {
        "variant": variant_name,
        "refine": refine,
        "total_queries": len(questions),
        "recall_at_5": round(statistics.mean(recall_at_5), 4) if recall_at_5 else 0,
        "recall_at_10": round(statistics.mean(recall_at_10), 4) if recall_at_10 else 0,
        "mrr": round(statistics.mean(mrr_scores), 4) if mrr_scores else 0,
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1),
        "latency_mean_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "total_tokens": sum(token_costs),
        "avg_tokens_per_query": round(statistics.mean(token_costs), 0) if token_costs else 0,
    }

    by_type = {}
    for r in results:
        qt = r["query_type"]
        if qt not in by_type:
            by_type[qt] = {"recall": [], "mrr": [], "latency": []}
        by_type[qt]["recall"].append(r["keyword_recall"])
        by_type[qt]["mrr"].append(r["reciprocal_rank"])
        by_type[qt]["latency"].append(r["latency_ms"])

    summary["by_query_type"] = {
        qt: {
            "count": len(vals["recall"]),
            "recall": round(statistics.mean(vals["recall"]), 4),
            "mrr": round(statistics.mean(vals["mrr"]), 4),
            "latency_p50": round(statistics.median(vals["latency"]), 1),
        }
        for qt, vals in by_type.items()
    }

    return {"summary": summary, "results": results}


def format_table(all_summaries: list[dict]) -> str:
    """Format summaries as a markdown comparison table."""
    lines = [
        "| Variant | Recall@5 | Recall@10 | MRR | p50 Latency (ms) | p95 Latency (ms) | Tokens/Query |",
        "|---------|----------|-----------|-----|-------------------|-------------------|--------------|",
    ]
    for s in all_summaries:
        lines.append(
            f"| {s['variant']} | {s['recall_at_5']:.4f} | {s['recall_at_10']:.4f} | "
            f"{s['mrr']:.4f} | {s['latency_p50_ms']:.0f} | {s['latency_p95_ms']:.0f} | "
            f"{s['avg_tokens_per_query']:.0f} |"
        )
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Archivist pipeline ablation benchmark")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), help="Run a single variant")
    parser.add_argument("--skip-index", action="store_true", help="Skip corpus indexing")
    parser.add_argument("--index-only", action="store_true", help="Only index corpus, don't run queries")
    parser.add_argument("--no-refine", action="store_true", help="Skip LLM refinement stages (faster)")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.skip_index:
        await index_corpus()

    if args.index_only:
        return

    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)
    if args.limit > 0:
        questions = questions[:args.limit]

    logger.info("Running %d questions", len(questions))

    variant_names = [args.variant] if args.variant else list(VARIANTS.keys())
    refine = not args.no_refine

    all_results = {}
    all_summaries = []

    for variant in variant_names:
        logger.info("=== Variant: %s ===", variant)
        data = await run_variant(variant, questions, refine=refine)
        all_results[variant] = data
        all_summaries.append(data["summary"])
        logger.info(
            "  Recall@5=%.4f  MRR=%.4f  p50=%.0fms  tokens/q=%.0f",
            data["summary"]["recall_at_5"],
            data["summary"]["mrr"],
            data["summary"]["latency_p50_ms"],
            data["summary"]["avg_tokens_per_query"],
        )

    print("\n" + format_table(all_summaries))

    if args.output:
        output_data = {
            "summaries": all_summaries,
            "comparison_table": format_table(all_summaries),
            "full_results": {k: v["results"] for k, v in all_results.items()},
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
