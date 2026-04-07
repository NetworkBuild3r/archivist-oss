"""LongMemEval benchmark adapter — evaluates Archivist against the LongMemEval
benchmark (ICLR 2025, xiaowu0162/LongMemEval).

LongMemEval tests 5 core long-term memory abilities across 500 questions:
  - Information Extraction (single-session-user, single-session-assistant, single-session-preference)
  - Multi-Session Reasoning (multi-session)
  - Knowledge Updates (knowledge-update)
  - Temporal Reasoning (temporal-reasoning)
  - Abstention (question_id ends with '_abs')

Setup:
    1. Download the dataset:
       mkdir -p data/longmemeval && cd data/longmemeval
       wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
       wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json

    2. Run single variant:
       python -m benchmarks.academic.longmemeval.adapter \\
           --data-file data/longmemeval/longmemeval_s_cleaned.json \\
           --variant full_pipeline \\
           --output .benchmarks/longmemeval_results.json

    3. Run competitive ablation (all 4 variants):
       python -m benchmarks.academic.longmemeval.adapter \\
           --data-file data/longmemeval/longmemeval_s_cleaned.json \\
           --ablation --output .benchmarks/longmemeval_ablation.json

    Quick test:
       python -m benchmarks.academic.longmemeval.adapter \\
           --data-file data/longmemeval/longmemeval_s_cleaned.json \\
           --limit 10 --no-refine --output .benchmarks/longmemeval_quick.json

"""

import argparse
import asyncio
import collections
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

logger = logging.getLogger("archivist.benchmark.longmemeval")

# ── Ablation variants (mirrors evaluate.py pattern) ──────────────────────────
VARIANTS = {
    "vector_only": {
        "BM25_ENABLED": "false",
        "GRAPH_RETRIEVAL_ENABLED": "false",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "0",
        "TEMPORAL_INTENT_ENABLED": "false",
        "BM25_RESCUE_ENABLED": "false",
        "ADAPTIVE_VECTOR_LIMIT_ENABLED": "false",
        "TOPIC_ROUTING_ENABLED": "false",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "full_pipeline": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "TEMPORAL_INTENT_ENABLED": "true",
        "BM25_RESCUE_ENABLED": "true",
        "ADAPTIVE_VECTOR_LIMIT_ENABLED": "true",
        "TOPIC_ROUTING_ENABLED": "false",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "full_plus_topic": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "TEMPORAL_INTENT_ENABLED": "true",
        "BM25_RESCUE_ENABLED": "true",
        "ADAPTIVE_VECTOR_LIMIT_ENABLED": "true",
        "TOPIC_ROUTING_ENABLED": "true",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "full_plus_rerank": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "true",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "TEMPORAL_INTENT_ENABLED": "true",
        "BM25_RESCUE_ENABLED": "true",
        "ADAPTIVE_VECTOR_LIMIT_ENABLED": "true",
        "TOPIC_ROUTING_ENABLED": "true",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
}


def _apply_variant(variant_name: str):
    """Apply config overrides for a variant by setting env vars and reloading config."""
    import importlib
    env_overrides = VARIANTS[variant_name]
    for key, value in env_overrides.items():
        os.environ[key] = value
    import config
    importlib.reload(config)


# Category mapping from LongMemEval question_type to readable names
_CATEGORY_MAP = {
    "single-session-user": "information_extraction",
    "single-session-assistant": "information_extraction",
    "single-session-preference": "information_extraction",
    "multi-session": "multi_session_reasoning",
    "knowledge-update": "knowledge_updates",
    "temporal-reasoning": "temporal_reasoning",
}


def _classify_question(item: dict) -> str:
    """Map a LongMemEval question to a high-level category."""
    if str(item.get("question_id", "")).endswith("_abs"):
        return "abstention"
    qt = item.get("question_type", "")
    return _CATEGORY_MAP.get(qt, qt or "unknown")


def _compute_keyword_recall(answer: str, ground_truth: str) -> float:
    """Fraction of ground-truth words found in the answer (case-insensitive)."""
    gt_words = set(ground_truth.lower().split())
    if not gt_words:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for w in gt_words if w in answer_lower)
    return found / len(gt_words)


def _compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = pred_tokens & truth_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _load_data(data_file: str) -> list[dict]:
    """Load a LongMemEval JSON file (longmemeval_s_cleaned.json etc.)."""
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        logger.error("Expected list of items in %s, got %s", data_file, type(data).__name__)
        return []
    logger.info("Loaded %d items from %s", len(data), data_file)
    return data


def _sessions_to_markdown(
    sessions: list[list[dict]],
    dates: list[str],
    question_id: str,
) -> list[tuple[str, str]]:
    """Convert LongMemEval sessions to markdown files (one per session).

    Each session is a list of turns: [{"role": "user/assistant", "content": "..."}].
    Returns list of (filename, markdown_content) pairs.
    """
    files = []
    for si, session in enumerate(sessions):
        date_str = dates[si] if si < len(dates) else f"session-{si:04d}"
        safe_date = date_str.replace("/", "-").replace(" ", "_")[:20]
        lines = [f"# Session {si + 1} — {date_str}\n"]
        for turn in session:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"**{role}:** {content}\n")
        filename = f"longmemeval-{question_id}/session-{si + 1:04d}-{safe_date}.md"
        files.append((filename, "\n".join(lines)))
    return files


async def _run_curator_on_files(mem_root: str) -> None:
    """Run curator entity extraction on all indexed files."""
    from curator import extract_knowledge, process_extraction
    from graph import init_schema
    init_schema()
    md_files = sorted(Path(mem_root).rglob("*.md"))
    for fp in md_files:
        try:
            text = fp.read_text(encoding="utf-8")
            rel = str(fp.relative_to(mem_root))
            knowledge = await extract_knowledge(text, agent_id="longmemeval", source_file=rel)
            if knowledge:
                await process_extraction(knowledge, agent_id="longmemeval", source_file=rel)
        except Exception as e:
            logger.debug("Curator extraction failed for %s: %s", fp, e)


async def run_longmemeval_benchmark(
    data_file: str,
    limit: int = 0,
    refine: bool = True,
    run_curator: bool = False,
    search_limit: int = 10,
    variant: str = "",
) -> dict:
    """Run the LongMemEval benchmark against Archivist.

    For each question:
      1. Write its haystack sessions as markdown files
      2. Index them into Qdrant + FTS5
      3. Optionally run curator for KG population
      4. Call recursive_retrieve
      5. Measure keyword_recall (R@k proxy), F1, and latency

    If *variant* is set, applies env-var overrides before retrieval.
    Returns summary + per-question results.
    """
    if variant:
        if variant not in VARIANTS:
            return {"error": f"Unknown variant: {variant}", "valid": list(VARIANTS.keys())}
        _apply_variant(variant)
    items = _load_data(data_file)
    if not items:
        return {"error": "No items loaded", "data_file": data_file}
    if limit > 0:
        items = items[:limit]

    work_dir = tempfile.mkdtemp(prefix="longmemeval_bench_")
    mem_root = os.path.join(work_dir, "memories")
    os.makedirs(mem_root, exist_ok=True)

    import config
    original_memory_root = config.MEMORY_ROOT
    original_sqlite = config.SQLITE_PATH

    try:
        config.MEMORY_ROOT = mem_root
        os.environ["MEMORY_ROOT"] = mem_root
        sqlite_path = os.path.join(work_dir, "graph.db")
        config.SQLITE_PATH = sqlite_path
        os.environ["SQLITE_PATH"] = sqlite_path

        from indexer import full_index
        from rlm_retriever import recursive_retrieve
        from graph import init_schema
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance

        qclient = QdrantClient(url=config.QDRANT_URL, timeout=30)
        init_schema()

        all_results: list[dict] = []
        results_by_category: dict[str, list[dict]] = collections.defaultdict(list)

        for qi, item in enumerate(items):
            qid = str(item.get("question_id", qi))
            question = item.get("question", "")
            answer_gt = item.get("answer", "")
            category = _classify_question(item)
            sessions = item.get("haystack_sessions", [])
            dates = item.get("haystack_dates", [])

            if not question or not sessions:
                logger.warning("Skipping item %s: no question or sessions", qid)
                continue

            # Write sessions to disk
            md_files = _sessions_to_markdown(sessions, dates, qid)
            for filename, content in md_files:
                filepath = os.path.join(mem_root, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            # Fresh collection per question to avoid cross-contamination
            try:
                qclient.delete_collection(config.QDRANT_COLLECTION)
            except Exception:
                pass
            qclient.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=config.VECTOR_DIM, distance=Distance.COSINE,
                ),
            )

            chunk_count = await full_index(hierarchical=True)

            if run_curator:
                await _run_curator_on_files(mem_root)

            t0 = time.monotonic()
            try:
                result = await recursive_retrieve(
                    query=question, namespace="", limit=search_limit,
                    refine=refine, tier="l2",
                )
            except Exception as e:
                logger.warning("Query %s failed: %s", qid, e)
                result = {"answer": "", "sources": []}
            elapsed_ms = (time.monotonic() - t0) * 1000

            answer = result.get("answer", "")
            sources_text = " ".join(
                s.get("text", "")[:500] for s in result.get("sources", [])
            )
            combined = f"{answer} {sources_text}"

            recall = _compute_keyword_recall(combined, answer_gt) if answer_gt else 0.0
            f1 = _compute_f1(combined, answer_gt) if answer_gt else 0.0

            # Session-level recall: did we retrieve from evidence sessions?
            evidence_ids = set(item.get("answer_session_ids", []))
            retrieved_files = {s.get("file_path", "") for s in result.get("sources", [])}
            session_hits = 0
            for eid in evidence_ids:
                for rf in retrieved_files:
                    if f"session-{eid + 1:04d}" in rf or f"session-{eid:04d}" in rf:
                        session_hits += 1
                        break
            session_recall = session_hits / max(len(evidence_ids), 1)

            entry = {
                "question_id": qid,
                "category": category,
                "question_type": item.get("question_type", ""),
                "query": question[:200],
                "ground_truth": answer_gt[:200],
                "answer": answer[:200],
                "keyword_recall": round(recall, 4),
                "f1": round(f1, 4),
                "session_recall": round(session_recall, 4),
                "latency_ms": round(elapsed_ms, 1),
                "sources_count": len(result.get("sources", [])),
                "chunks_indexed": chunk_count,
                "sessions_count": len(sessions),
            }
            all_results.append(entry)
            results_by_category[category].append(entry)

            # Clean up for next question
            for filename, _ in md_files:
                fp = os.path.join(mem_root, filename)
                if os.path.exists(fp):
                    os.remove(fp)
            parent = os.path.join(mem_root, f"longmemeval-{qid}")
            if os.path.isdir(parent):
                shutil.rmtree(parent, ignore_errors=True)

            if (qi + 1) % 10 == 0:
                logger.info(
                    "Progress: %d/%d  |  avg keyword_recall=%.3f  session_recall=%.3f",
                    qi + 1, len(items),
                    _mean([r["keyword_recall"] for r in all_results]),
                    _mean([r["session_recall"] for r in all_results]),
                )

        # Build summary
        by_category = {}
        for cat, cat_results in results_by_category.items():
            latencies = sorted(r["latency_ms"] for r in cat_results)
            by_category[cat] = {
                "count": len(cat_results),
                "keyword_recall": _mean([r["keyword_recall"] for r in cat_results]),
                "f1": _mean([r["f1"] for r in cat_results]),
                "session_recall": _mean([r["session_recall"] for r in cat_results]),
                "latency_p50": round(latencies[len(latencies) // 2], 1) if latencies else 0,
            }

        summary = {
            "benchmark": "LongMemEval",
            "variant": variant or "default",
            "data_file": os.path.basename(data_file),
            "total_questions": len(items),
            "evaluated_questions": len(all_results),
            "overall_keyword_recall": round(_mean([r["keyword_recall"] for r in all_results]), 4),
            "overall_f1": round(_mean([r["f1"] for r in all_results]), 4),
            "overall_session_recall": round(_mean([r["session_recall"] for r in all_results]), 4),
            "by_category": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                for kk, vv in v.items()}
                           for k, v in by_category.items()},
            "search_limit": search_limit,
            "refine": refine,
            "curator": run_curator,
        }

        return {"summary": summary, "results": all_results}

    finally:
        config.MEMORY_ROOT = original_memory_root
        config.SQLITE_PATH = original_sqlite
        os.environ["MEMORY_ROOT"] = original_memory_root
        os.environ["SQLITE_PATH"] = original_sqlite
        shutil.rmtree(work_dir, ignore_errors=True)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_summary(s: dict) -> None:
    """Print a single variant's summary to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  LongMemEval — variant: {s.get('variant', 'default')}")
    print(f"{'=' * 60}")
    print(f"  Dataset:    {s['data_file']}")
    print(f"  Questions:  {s['evaluated_questions']}/{s['total_questions']}")
    print(f"  Curator:    {'yes' if s['curator'] else 'no'}")
    print(f"  Refine:     {'yes' if s['refine'] else 'no'}")
    print()
    print(f"  Keyword Recall:  {s['overall_keyword_recall']:.4f}")
    print(f"  Session Recall:  {s['overall_session_recall']:.4f}")
    print(f"  Token F1:        {s['overall_f1']:.4f}")
    print()
    print(f"  By category:")
    for cat, vals in s["by_category"].items():
        print(f"    {cat:25s}  recall={vals['keyword_recall']:.4f}"
              f"  session={vals['session_recall']:.4f}"
              f"  f1={vals['f1']:.4f}  (n={vals['count']})")
    print(f"{'=' * 60}")


def _print_ablation_comparison(ablation_data: dict) -> None:
    """Print a side-by-side comparison table across variants."""
    variants = ablation_data.get("variants", {})
    if not variants:
        return

    print(f"\n{'=' * 78}")
    print(f"  LongMemEval Ablation Results")
    print(f"{'=' * 78}")
    print(f"  {'Variant':<25s}  {'Keyword R':>10s}  {'Session R':>10s}  {'F1':>10s}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for vname, vdata in variants.items():
        s = vdata.get("summary", {})
        kr = s.get("overall_keyword_recall", 0)
        sr = s.get("overall_session_recall", 0)
        f1 = s.get("overall_f1", 0)
        print(f"  {vname:<25s}  {kr:>10.4f}  {sr:>10.4f}  {f1:>10.4f}")

    print(f"{'=' * 78}")


async def main():
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark adapter for Archivist",
    )
    parser.add_argument(
        "--data-file", required=True,
        help="Path to longmemeval_s_cleaned.json or longmemeval_m_cleaned.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")
    parser.add_argument("--no-refine", action="store_true", help="Skip LLM refinement")
    parser.add_argument("--run-curator", action="store_true", help="Run curator KG extraction")
    parser.add_argument("--search-limit", type=int, default=10, help="Top-k retrieval limit")
    parser.add_argument(
        "--variant", choices=list(VARIANTS.keys()),
        help="Run a single pipeline variant (default: no overrides)",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run all variants sequentially and produce comparison table",
    )
    parser.add_argument(
        "--variants", type=str, default="",
        help="Comma-separated variant names for --ablation (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default=".benchmarks/longmemeval_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.ablation:
        variant_names = (
            [v.strip() for v in args.variants.split(",") if v.strip()]
            if args.variants else list(VARIANTS.keys())
        )
        ablation_results: dict = {
            "benchmark": "LongMemEval",
            "mode": "ablation",
            "variants": {},
        }
        for vname in variant_names:
            logger.info("=== Running variant: %s ===", vname)
            data = await run_longmemeval_benchmark(
                args.data_file,
                limit=args.limit,
                refine=not args.no_refine,
                run_curator=args.run_curator,
                search_limit=args.search_limit,
                variant=vname,
            )
            ablation_results["variants"][vname] = data
            if "summary" in data:
                _print_summary(data["summary"])

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, indent=2, ensure_ascii=False)

        _print_ablation_comparison(ablation_results)
        logger.info("Ablation results written to %s", args.output)
    else:
        data = await run_longmemeval_benchmark(
            args.data_file,
            limit=args.limit,
            refine=not args.no_refine,
            run_curator=args.run_curator,
            search_limit=args.search_limit,
            variant=args.variant or "",
        )

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if "summary" in data:
            _print_summary(data["summary"])
        else:
            print(json.dumps(data, indent=2))

        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
