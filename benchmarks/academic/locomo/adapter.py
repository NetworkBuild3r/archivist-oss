"""LoCoMo benchmark adapter — evaluates Archivist against the Long Conversation
Memory benchmark (snap-research/locomo).

LoCoMo tests memory and reasoning over very long multi-session dialogues with
5 QA types: single-hop, multi-hop, temporal, commonsense, and adversarial.

Setup:
    1. Clone the LoCoMo dataset:
       git clone https://github.com/snap-research/locomo.git data/locomo

    2. Install evaluation deps:
       pip install rouge-score nltk

    3. Run:
       python -m benchmarks.academic.locomo.adapter --data-dir data/locomo

Published competitor scores (2026):
    - Letta/MemGPT: ~83.2%
    - Zep (Graphiti): ~85%
    - Mem0: ~58-66%
"""

import argparse
import asyncio
import collections
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

logger = logging.getLogger("archivist.benchmark.locomo")


def _compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
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


def _compute_bleu(prediction: str, ground_truth: str) -> float:
    """Compute BLEU-1 score (unigram overlap)."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = ground_truth.lower().split()
        pred_tokens = prediction.lower().split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        return sentence_bleu(
            [ref_tokens], pred_tokens,
            weights=(1.0, 0, 0, 0),
            smoothing_function=SmoothingFunction().method1,
        )
    except ImportError:
        logger.warning("nltk not installed, using F1 as BLEU proxy")
        return _compute_f1(prediction, ground_truth)


def _compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L F1 score."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(ground_truth, prediction)
        return scores["rougeL"].fmeasure
    except ImportError:
        logger.warning("rouge-score not installed, using F1 as ROUGE-L proxy")
        return _compute_f1(prediction, ground_truth)


def _load_locomo_data(data_dir: str) -> list[dict]:
    """Load LoCoMo conversation + QA data from the dataset directory.

    Supports multiple directory structures: the raw JSON files or the
    processed conversation format. Returns a list of dialogue records,
    each containing 'sessions' (conversation turns) and 'questions'.
    """
    dialogues = []

    json_files = list(Path(data_dir).rglob("*.json"))
    if not json_files:
        logger.error("No JSON files found in %s", data_dir)
        return []

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and ("conversation" in item or "sessions" in item or "questions" in item):
                    dialogues.append(item)
        elif isinstance(data, dict):
            if "conversation" in data or "sessions" in data or "questions" in data:
                dialogues.append(data)
            elif "data" in data and isinstance(data["data"], list):
                dialogues.extend(data["data"])

    logger.info("Loaded %d dialogue records from %s", len(dialogues), data_dir)
    return dialogues


def _dialogue_to_markdown(dialogue: dict, dialogue_idx: int) -> list[tuple[str, str]]:
    """Convert a LoCoMo dialogue into markdown files (one per session).

    Returns list of (filename, content) pairs.
    """
    files = []
    sessions = dialogue.get("sessions", dialogue.get("conversation", []))

    if isinstance(sessions, list) and sessions:
        if isinstance(sessions[0], dict) and "turns" in sessions[0]:
            for si, session in enumerate(sessions):
                lines = [f"# Session {si + 1}\n"]
                for turn in session.get("turns", []):
                    speaker = turn.get("speaker", turn.get("role", "unknown"))
                    text = turn.get("text", turn.get("content", ""))
                    lines.append(f"**{speaker}:** {text}\n")
                filename = f"dialogue-{dialogue_idx:03d}/session-{si + 1:03d}.md"
                files.append((filename, "\n".join(lines)))
        elif isinstance(sessions[0], dict) and ("speaker" in sessions[0] or "role" in sessions[0]):
            chunk_size = 20
            for ci in range(0, len(sessions), chunk_size):
                chunk = sessions[ci:ci + chunk_size]
                lines = [f"# Session chunk {ci // chunk_size + 1}\n"]
                for turn in chunk:
                    speaker = turn.get("speaker", turn.get("role", "unknown"))
                    text = turn.get("text", turn.get("content", ""))
                    lines.append(f"**{speaker}:** {text}\n")
                filename = f"dialogue-{dialogue_idx:03d}/chunk-{ci // chunk_size + 1:03d}.md"
                files.append((filename, "\n".join(lines)))
        elif isinstance(sessions[0], str):
            for si, session_text in enumerate(sessions):
                filename = f"dialogue-{dialogue_idx:03d}/session-{si + 1:03d}.md"
                files.append((filename, f"# Session {si + 1}\n\n{session_text}\n"))

    if not files:
        raw = json.dumps(dialogue, indent=2, ensure_ascii=False)
        for ci in range(0, len(raw), 4000):
            chunk = raw[ci:ci + 4000]
            filename = f"dialogue-{dialogue_idx:03d}/raw-{ci // 4000 + 1:03d}.md"
            files.append((filename, f"# Dialogue {dialogue_idx} (raw)\n\n```\n{chunk}\n```\n"))

    return files


def _extract_questions(dialogue: dict) -> list[dict]:
    """Extract QA pairs from a LoCoMo dialogue record."""
    questions = dialogue.get("questions", dialogue.get("qa_pairs", []))
    if not questions:
        for key in ("single_hop", "multi_hop", "temporal", "commonsense", "adversarial"):
            if key in dialogue:
                items = dialogue[key]
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            item.setdefault("category", key)
                            questions.append(item)
    return questions


async def run_locomo_benchmark(data_dir: str, limit: int = 0, refine: bool = True) -> dict:
    """Run the full LoCoMo benchmark against Archivist.

    Steps:
        1. Load LoCoMo dialogue data
        2. Convert each dialogue to markdown files in a temp MEMORY_ROOT
        3. Index all files
        4. For each QA pair, call archivist_search
        5. Evaluate with F1, BLEU, ROUGE-L
    """
    dialogues = _load_locomo_data(data_dir)
    if not dialogues:
        return {"error": "No dialogues loaded", "data_dir": data_dir}

    if limit > 0:
        dialogues = dialogues[:limit]

    work_dir = tempfile.mkdtemp(prefix="locomo_bench_")
    mem_root = os.path.join(work_dir, "memories")
    os.makedirs(mem_root, exist_ok=True)

    try:
        import config
        config.MEMORY_ROOT = mem_root
        os.environ["MEMORY_ROOT"] = mem_root

        total_files = 0
        all_questions = []

        for di, dialogue in enumerate(dialogues):
            md_files = _dialogue_to_markdown(dialogue, di)
            for filename, content in md_files:
                filepath = os.path.join(mem_root, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                total_files += 1

            qa_pairs = _extract_questions(dialogue)
            for qa in qa_pairs:
                qa["dialogue_idx"] = di
            all_questions.extend(qa_pairs)

        logger.info("Created %d files from %d dialogues, %d QA pairs",
                     total_files, len(dialogues), len(all_questions))

        from indexer import full_index
        chunk_count = await full_index(hierarchical=True)
        logger.info("Indexed %d chunks", chunk_count)

        from rlm_retriever import recursive_retrieve

        results_by_category = collections.defaultdict(list)
        all_results = []

        for qi, qa in enumerate(all_questions):
            query = qa.get("question", qa.get("query", qa.get("text", "")))
            ground_truth = qa.get("answer", qa.get("ground_truth", qa.get("expected", "")))
            category = qa.get("category", qa.get("type", "unknown"))

            if not query:
                continue

            t0 = time.monotonic()
            try:
                result = await recursive_retrieve(
                    query=query, namespace="", limit=10,
                    refine=refine, tier="l2",
                )
            except Exception as e:
                logger.warning("Query %d failed: %s", qi, e)
                result = {"answer": "", "sources": []}
            elapsed_ms = (time.monotonic() - t0) * 1000

            answer = result.get("answer", "")

            f1 = _compute_f1(answer, ground_truth) if ground_truth else 0.0
            bleu = _compute_bleu(answer, ground_truth) if ground_truth else 0.0
            rouge_l = _compute_rouge_l(answer, ground_truth) if ground_truth else 0.0

            entry = {
                "question_idx": qi,
                "category": category,
                "query": query,
                "ground_truth": ground_truth[:200],
                "answer": answer[:200],
                "f1": round(f1, 4),
                "bleu": round(bleu, 4),
                "rouge_l": round(rouge_l, 4),
                "latency_ms": round(elapsed_ms, 1),
                "sources_count": len(result.get("sources", [])),
            }
            all_results.append(entry)
            results_by_category[category].append(entry)

        overall_f1 = _mean([r["f1"] for r in all_results])
        overall_bleu = _mean([r["bleu"] for r in all_results])
        overall_rouge_l = _mean([r["rouge_l"] for r in all_results])

        by_category = {}
        for cat, cat_results in results_by_category.items():
            by_category[cat] = {
                "count": len(cat_results),
                "f1": _mean([r["f1"] for r in cat_results]),
                "bleu": _mean([r["bleu"] for r in cat_results]),
                "rouge_l": _mean([r["rouge_l"] for r in cat_results]),
                "latency_p50": round(
                    sorted(r["latency_ms"] for r in cat_results)[len(cat_results) // 2], 1
                ) if cat_results else 0,
            }

        summary = {
            "benchmark": "LoCoMo",
            "dialogues": len(dialogues),
            "total_files": total_files,
            "total_chunks": chunk_count,
            "total_questions": len(all_questions),
            "evaluated_questions": len(all_results),
            "overall_f1": round(overall_f1, 4),
            "overall_bleu": round(overall_bleu, 4),
            "overall_rouge_l": round(overall_rouge_l, 4),
            "by_category": by_category,
            "competitor_scores": {
                "letta": 0.832,
                "zep": 0.85,
                "mem0": 0.58,
            },
        }

        return {"summary": summary, "results": all_results}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def main():
    parser = argparse.ArgumentParser(description="LoCoMo benchmark adapter for Archivist")
    parser.add_argument("--data-dir", required=True, help="Path to LoCoMo dataset directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit dialogues (0=all)")
    parser.add_argument("--no-refine", action="store_true", help="Skip LLM refinement")
    parser.add_argument("--output", type=str, default="locomo_results.json", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    data = await run_locomo_benchmark(args.data_dir, limit=args.limit, refine=not args.no_refine)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if "summary" in data:
        s = data["summary"]
        print(f"\n=== LoCoMo Benchmark Results ===")
        print(f"Dialogues: {s['dialogues']}, Questions: {s['evaluated_questions']}")
        print(f"Overall F1:      {s['overall_f1']:.4f}")
        print(f"Overall BLEU:    {s['overall_bleu']:.4f}")
        print(f"Overall ROUGE-L: {s['overall_rouge_l']:.4f}")
        print(f"\nBy category:")
        for cat, vals in s["by_category"].items():
            print(f"  {cat:15s}: F1={vals['f1']:.4f}  BLEU={vals['bleu']:.4f}  ROUGE-L={vals['rouge_l']:.4f}  (n={vals['count']})")
        print(f"\nCompetitor scores (LoCoMo QA):")
        for name, score in s["competitor_scores"].items():
            print(f"  {name:15s}: {score:.1%}")
    else:
        print(json.dumps(data, indent=2))

    logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
