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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from benchmarks.env_loader import load_repo_env

load_repo_env()
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

logger = logging.getLogger("archivist.benchmark.locomo")


def _normalize_answer(s: str) -> str:
    """Official LoCoMo answer normalization (from task_eval/evaluation.py)."""
    import string
    import unicodedata
    s = unicodedata.normalize("NFD", s)
    s = s.replace(",", "")
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    import re
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = " ".join(s.split())
    return s


def _stem_tokens(text: str) -> list[str]:
    """Stem tokens using Porter stemmer (matches official LoCoMo eval)."""
    try:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        return [ps.stem(w) for w in _normalize_answer(text).split()]
    except ImportError:
        return _normalize_answer(text).split()


def _compute_f1(prediction: str, ground_truth: str) -> float:
    """Official LoCoMo stemmed F1 score (from task_eval/evaluation.py).

    Uses answer normalization + Porter stemming for token-level F1, matching
    what the paper reports.
    """
    from collections import Counter
    pred_tokens = _stem_tokens(prediction)
    truth_tokens = _stem_tokens(ground_truth)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _compute_f1_multi(prediction: str, ground_truth: str) -> float:
    """F1 for multi-hop with comma-separated sub-answers (official protocol).

    For multi-hop answers like 'Alice, Bob', computes F1 for each sub-answer
    against each sub-prediction and takes the mean of best matches.
    """
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    if not ground_truths:
        return 0.0
    import numpy as np
    scores = []
    for gt in ground_truths:
        best = max(_compute_f1(pred, gt) for pred in predictions) if predictions else 0.0
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


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


def _keyword_recall(text: str, ground_truth: str) -> float:
    """Fraction of ground-truth words found in text (case-insensitive)."""
    gt_words = set(ground_truth.lower().split())
    if not gt_words:
        return 0.0
    text_lower = text.lower()
    found = sum(1 for w in gt_words if w in text_lower)
    return found / len(gt_words)


async def _run_curator_extraction(mem_root: str) -> None:
    """Run curator entity extraction over indexed files for KG population."""
    from curator import extract_knowledge, process_extraction
    from graph import init_schema
    init_schema()
    for fp in sorted(Path(mem_root).rglob("*.md")):
        try:
            text = fp.read_text(encoding="utf-8")
            rel = str(fp.relative_to(mem_root))
            knowledge = await extract_knowledge(text, agent_id="locomo", source_file=rel)
            if knowledge:
                await process_extraction(knowledge, agent_id="locomo", source_file=rel)
        except Exception as e:
            logger.debug("Curator extraction failed for %s: %s", fp, e)


async def run_locomo_benchmark(
    data_dir: str,
    limit: int = 0,
    refine: bool = True,
    run_curator: bool = False,
) -> dict:
    """Run the full LoCoMo benchmark against Archivist.

    Steps:
        1. Load LoCoMo dialogue data
        2. Convert each dialogue to markdown files in a temp MEMORY_ROOT
        3. Index all files
        4. (Optional) Run curator for KG population
        5. For each QA pair, call archivist_search
        6. Evaluate with keyword_recall, F1, BLEU, ROUGE-L
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
        from graph import init_schema
        init_schema()
        chunk_count = await full_index(hierarchical=True)
        logger.info("Indexed %d chunks", chunk_count)

        if run_curator:
            logger.info("Running curator extraction over %d files...", total_files)
            await _run_curator_extraction(mem_root)

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
            sources_text = " ".join(
                s.get("text", "")[:500] for s in result.get("sources", [])
            )
            combined = f"{answer} {sources_text}"

            # Official LoCoMo category-specific scoring
            # Categories: 1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop, 5=adversarial
            cat_num = qa.get("category", 0) if isinstance(qa.get("category"), int) else 0
            if cat_num == 1 or category == "multi_hop":
                f1 = _compute_f1_multi(answer, ground_truth) if ground_truth else 0.0
            elif cat_num == 5 or category == "adversarial":
                f1 = 1.0 if ("no information available" in answer.lower()
                             or "not mentioned" in answer.lower()) else 0.0
            else:
                f1 = _compute_f1(answer, ground_truth) if ground_truth else 0.0

            bleu = _compute_bleu(answer, ground_truth) if ground_truth else 0.0
            rouge_l = _compute_rouge_l(answer, ground_truth) if ground_truth else 0.0

            entry = {
                "question_idx": qi,
                "category": category,
                "query": query,
                "ground_truth": ground_truth[:200],
                "prediction": answer[:500],
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
                "f1": round(_mean([r["f1"] for r in cat_results]), 4),
                "bleu": round(_mean([r["bleu"] for r in cat_results]), 4),
                "rouge_l": round(_mean([r["rouge_l"] for r in cat_results]), 4),
                "latency_p50": round(
                    sorted(r["latency_ms"] for r in cat_results)[len(cat_results) // 2], 1
                ) if cat_results else 0,
            }

        summary = {
            "benchmark": "LoCoMo",
            "eval_protocol": "official (stemmed F1, category-specific scoring)",
            "dialogues": len(dialogues),
            "total_files": total_files,
            "total_chunks": chunk_count,
            "total_questions": len(all_questions),
            "evaluated_questions": len(all_results),
            "overall_f1": round(overall_f1, 4),
            "overall_bleu": round(overall_bleu, 4),
            "overall_rouge_l": round(overall_rouge_l, 4),
            "by_category": by_category,
            "curator": run_curator,
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
    parser.add_argument("--run-curator", action="store_true", help="Run curator KG extraction")
    parser.add_argument("--output", type=str, default="locomo_results.json", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    data = await run_locomo_benchmark(
        args.data_dir,
        limit=args.limit,
        refine=not args.no_refine,
        run_curator=args.run_curator,
    )

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if "summary" in data:
        s = data["summary"]
        print(f"\n{'=' * 72}")
        print(f"  LoCoMo Benchmark — Official Stemmed F1 Evaluation")
        print(f"{'=' * 72}")
        print(f"  Dialogues:  {s['dialogues']}   Questions: {s['evaluated_questions']}")
        print(f"  Curator:    {'yes' if s.get('curator') else 'no'}")
        print(f"  Protocol:   {s.get('eval_protocol', 'official')}")
        print()
        print(f"  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  Overall F1:      {s['overall_f1']:.4f}                              │")
        print(f"  │  Overall BLEU-1:  {s['overall_bleu']:.4f}                              │")
        print(f"  │  Overall ROUGE-L: {s['overall_rouge_l']:.4f}                              │")
        print(f"  └─────────────────────────────────────────────────────────┘")
        print()
        print(f"  {'Category':<18s}  {'F1':>6s}  {'BLEU':>6s}  {'ROUGE-L':>7s}  {'n':>4s}")
        print(f"  {'-' * 18}  {'-' * 6}  {'-' * 6}  {'-' * 7}  {'-' * 4}")
        for cat, vals in s["by_category"].items():
            print(f"  {cat:<18s}  {vals['f1']:.4f}  {vals['bleu']:.4f}  {vals['rouge_l']:.4f}   {vals['count']:>4d}")
        print(f"{'=' * 72}")
    else:
        print(json.dumps(data, indent=2))

    logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
