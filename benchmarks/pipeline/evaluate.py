"""Pipeline ablation harness — runs the Archivist retrieval pipeline under
various configuration variants and computes Recall@K, MRR, latency, and token cost.

Prerequisites:
    - Qdrant running (docker compose up -d qdrant)
    - Embedding API reachable (LLM_URL / EMBED_URL configured)
    - Seed corpus: benchmarks/fixtures/corpus/ (legacy) or generate presets:
          python benchmarks/fixtures/generate_corpus.py --preset small
          python benchmarks/fixtures/generate_corpus.py --write-questions

Usage:
    # Legacy corpus, index then all variants:
    python -m benchmarks.pipeline.evaluate

    # Dual-track preset (corpus under fixtures/corpus_small/):
    python -m benchmarks.pipeline.evaluate --memory-scale small --output results.json

    # After indexing, populate SQLite graph (LLM cost — optional):
    python -m benchmarks.pipeline.evaluate --memory-scale medium --run-curator

    # Compare vector_only vs full_pipeline across three corpus sizes:
    python -m benchmarks.pipeline.evaluate --scale-sweep --variants vector_only,full_pipeline

    # Single variant:
    python -m benchmarks.pipeline.evaluate --variant vector_only
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

env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(env_path):
    with open(env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
DEFAULT_CORPUS_DIR = os.path.join(FIXTURES_DIR, "corpus")
QUESTIONS_PATH = os.path.join(FIXTURES_DIR, "questions.json")
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

CORPUS_BY_PRESET = {
    "small": os.path.join(FIXTURES_DIR, "corpus_small"),
    "medium": os.path.join(FIXTURES_DIR, "corpus_medium"),
    "large": os.path.join(FIXTURES_DIR, "corpus_large"),
}

HYPOTHESES = {
    "H1_large": (
        "As indexed chunk count rises (fixed embedding model), the gap between vector_only "
        "and full_pipeline on needle and multi_hop questions widens (BM25 + graph help more in a deep haystack)."
    ),
    "H2_small": (
        "On a small corpus, headline recall may stay flat; value shows on contradiction, temporal, "
        "and agent_scoped slices and on MRR, not global Recall@5 alone."
    ),
    "H3_graph": (
        "If the knowledge graph is empty, graph-dependent questions cannot beat vector-only — "
        "curator extraction (or offline seeding) is a prerequisite at every scale."
    ),
}

logger = logging.getLogger("archivist.benchmark.pipeline")

VARIANTS = {
    "vector_only": {
        "BM25_ENABLED": "false",
        "GRAPH_RETRIEVAL_ENABLED": "false",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "0",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "plus_bm25": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "false",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "0",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "plus_graph": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "0",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "plus_temporal": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "plus_hotness": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "plus_rerank": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "full_pipeline": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "RETRIEVAL_THRESHOLD": "0.2",
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

    propagation_targets = [
        "rlm_retriever", "graph_retrieval", "retrieval_filters",
        "memory_fusion", "hot_cache", "hotness",
    ]
    for mod_name in propagation_targets:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])


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


def resolve_corpus_dir(memory_scale: str | None, corpus_dir_override: str | None) -> str:
    if corpus_dir_override:
        return os.path.abspath(corpus_dir_override)
    if memory_scale and memory_scale in CORPUS_BY_PRESET:
        return CORPUS_BY_PRESET[memory_scale]
    return DEFAULT_CORPUS_DIR


def filter_questions_for_scale(questions: list[dict], memory_scale: str | None) -> list[dict]:
    """Drop questions whose `scales` list excludes this preset (Phase 2)."""
    if not memory_scale:
        return questions
    out = []
    for q in questions:
        scales = q.get("scales")
        if scales is None:
            out.append(q)
            continue
        if "all" in scales or memory_scale in scales:
            out.append(q)
    return out


async def index_corpus(corpus_dir: str):
    """Index the seed corpus files into Qdrant and FTS5."""
    import config
    config.MEMORY_ROOT = corpus_dir

    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

    client = QdrantClient(url=config.QDRANT_URL, timeout=30)
    collections = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=config.VECTOR_DIM, distance=Distance.COSINE),
        )
        for field, schema in [
            ("agent_id", PayloadSchemaType.KEYWORD),
            ("namespace", PayloadSchemaType.KEYWORD),
            ("date", PayloadSchemaType.KEYWORD),
            ("memory_type", PayloadSchemaType.KEYWORD),
            ("is_parent", PayloadSchemaType.BOOL),
            ("parent_id", PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(
                collection_name=config.QDRANT_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        logger.info("Created benchmark collection '%s' (%d-dim)", config.QDRANT_COLLECTION, config.VECTOR_DIM)
    else:
        info = client.get_collection(config.QDRANT_COLLECTION)
        if info.points_count > 0:
            client.delete_collection(config.QDRANT_COLLECTION)
            client.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=config.VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info("Recreated benchmark collection '%s' for clean run", config.QDRANT_COLLECTION)

    from indexer import full_index
    from graph import init_schema
    init_schema()
    logger.info("Indexing corpus from %s ...", corpus_dir)
    count = await full_index(hierarchical=True)
    logger.info("Indexed %d chunks from corpus", count)
    return count


async def run_variant(variant_name: str, questions: list[dict], refine: bool) -> dict:
    """Run all questions against a pipeline variant and collect metrics."""
    _apply_variant(variant_name)

    import importlib
    import hot_cache
    if "hot_cache" in sys.modules:
        importlib.reload(sys.modules["hot_cache"])
        hot_cache = sys.modules["hot_cache"]
    hot_cache.invalidate_all()

    if "rlm_retriever" in sys.modules:
        importlib.reload(sys.modules["rlm_retriever"])
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


async def _run_benchmark_session(
    *,
    corpus_dir: str,
    questions: list[dict],
    variant_names: list[str],
    refine: bool,
    skip_index: bool,
    run_curator: bool,
    memory_scale: str | None,
) -> tuple[dict, list[dict]]:
    """Index (optional), curator (optional), run variants; return (all_results, all_summaries)."""
    import importlib
    import config

    corpus_dir = os.path.abspath(corpus_dir)
    os.environ["MEMORY_ROOT"] = corpus_dir
    importlib.reload(config)

    if not skip_index:
        await index_corpus(corpus_dir)

    if run_curator:
        import curator

        importlib.reload(config)
        importlib.reload(curator)
        logger.info("Running curator extraction over %s (LLM calls)...", corpus_dir)
        n = await curator.extract_all_agent_memories()
        logger.info("Curator finished: %d files extracted", n)

    all_results = {}
    all_summaries = []

    for variant in variant_names:
        logger.info("=== Variant: %s ===", variant)
        data = await run_variant(variant, questions, refine=refine)
        all_results[variant] = data
        summary = dict(data["summary"])
        if memory_scale:
            summary["memory_scale"] = memory_scale
        summary["corpus_dir"] = corpus_dir
        all_summaries.append(summary)
        logger.info(
            "  Recall@5=%.4f  MRR=%.4f  p50=%.0fms  tokens/q=%.0f",
            data["summary"]["recall_at_5"],
            data["summary"]["mrr"],
            data["summary"]["latency_p50_ms"],
            data["summary"]["avg_tokens_per_query"],
        )

    return all_results, all_summaries


async def main():
    parser = argparse.ArgumentParser(description="Archivist pipeline ablation benchmark")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), help="Run a single variant")
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Comma-separated variants (used with --scale-sweep); default all variants",
    )
    parser.add_argument("--skip-index", action="store_true", help="Skip corpus indexing")
    parser.add_argument("--index-only", action="store_true", help="Only index corpus, don't run queries")
    parser.add_argument("--no-refine", action="store_true", help="Skip LLM refinement stages (faster)")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument(
        "--memory-scale",
        choices=["small", "medium", "large"],
        default=None,
        help="Use benchmarks/fixtures/corpus_<scale>/ (generate with generate_corpus.py --preset)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Override corpus root (agents/ tree). Overrides --memory-scale.",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Path to questions JSON (default: fixtures/questions.json)",
    )
    parser.add_argument(
        "--run-curator",
        action="store_true",
        help="After indexing, run LLM graph extraction on all agent markdown (costs tokens)",
    )
    parser.add_argument(
        "--scale-sweep",
        action="store_true",
        help="Run the same variants for small, medium, large presets (re-indexes each time)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    questions_path = args.questions or QUESTIONS_PATH
    with open(questions_path, "r", encoding="utf-8") as f:
        questions_all = json.load(f)
    if args.limit > 0:
        questions_all = questions_all[: args.limit]

    variant_names = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants
        else ([args.variant] if args.variant else list(VARIANTS.keys()))
    )
    refine = not args.no_refine

    if args.scale_sweep:
        by_scale = {}
        combined_summaries = []
        for scale in ("small", "medium", "large"):
            corpus_dir = resolve_corpus_dir(scale, args.corpus_dir)
            if not os.path.isdir(os.path.join(corpus_dir, "agents")):
                logger.warning(
                    "Skipping scale=%s — missing corpus at %s (run generate_corpus.py --preset %s)",
                    scale,
                    corpus_dir,
                    scale,
                )
                continue
            qs = filter_questions_for_scale(questions_all, scale)
            logger.info("Scale %s: %d questions, corpus %s", scale, len(qs), corpus_dir)
            all_results, summaries = await _run_benchmark_session(
                corpus_dir=corpus_dir,
                questions=qs,
                variant_names=variant_names,
                refine=refine,
                skip_index=args.skip_index,
                run_curator=args.run_curator,
                memory_scale=scale,
            )
            by_scale[scale] = {
                "summaries": summaries,
                "full_results": {k: v["results"] for k, v in all_results.items()},
            }
            combined_summaries.extend(summaries)

        print("\n" + format_table(combined_summaries))

        if args.output:
            output_data = {
                "benchmark_meta": {
                    "hypotheses": HYPOTHESES,
                    "scale_sweep": True,
                    "variants": variant_names,
                },
                "by_scale": by_scale,
                "summaries": combined_summaries,
                "comparison_table": format_table(combined_summaries),
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            logger.info("Results written to %s", args.output)
        return

    corpus_dir = resolve_corpus_dir(args.memory_scale, args.corpus_dir)
    questions = filter_questions_for_scale(questions_all, args.memory_scale)

    if args.index_only:
        if not args.skip_index:
            await index_corpus(corpus_dir)
        return

    logger.info("Corpus: %s | Questions: %d", corpus_dir, len(questions))

    all_results, all_summaries = await _run_benchmark_session(
        corpus_dir=corpus_dir,
        questions=questions,
        variant_names=variant_names,
        refine=refine,
        skip_index=args.skip_index,
        run_curator=args.run_curator,
        memory_scale=args.memory_scale,
    )

    print("\n" + format_table(all_summaries))

    if args.output:
        output_data = {
            "benchmark_meta": {
                "hypotheses": HYPOTHESES,
                "memory_scale": args.memory_scale,
                "corpus_dir": corpus_dir,
                "questions_path": questions_path,
                "variants": variant_names,
            },
            "summaries": all_summaries,
            "comparison_table": format_table(all_summaries),
            "full_results": {k: v["results"] for k, v in all_results.items()},
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
