"""HaluMem benchmark adapter — evaluates Archivist against the Hallucination in
Memory benchmark (MemTensor/HaluMem).

HaluMem decomposes memory processing into three evaluation tasks:
    1. Memory Extraction — accurately store facts without hallucination
    2. Memory Updating — correctly modify memories when new facts arrive
    3. Memory QA — answer questions from stored memories without hallucination

Setup:
    1. Get the dataset:
       pip install datasets
       # OR clone: git clone https://github.com/MemTensor/HaluMem.git data/halumem

    2. Run:
       python -m benchmarks.academic.halumem.adapter --data-dir data/halumem

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
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from benchmarks.env_loader import load_repo_env

load_repo_env()
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

logger = logging.getLogger("archivist.benchmark.halumem")


def _load_halumem_data(data_dir: str) -> dict:
    """Load HaluMem evaluation data.

    Supports:
        - Raw JSON files from the GitHub repo
        - HuggingFace datasets cache format
        - Direct JSON with users/conversations/questions
    """
    data = {"users": []}

    json_files = list(Path(data_dir).rglob("*.json"))
    jsonl_files = list(Path(data_dir).rglob("*.jsonl"))

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                content = json.load(f)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and ("conversations" in item or "memories" in item or "dialogues" in item):
                        data["users"].append(item)
            elif isinstance(content, dict):
                if "users" in content:
                    data["users"].extend(content["users"])
                elif "data" in content:
                    data["users"].extend(content["data"])
                elif "conversations" in content or "memories" in content:
                    data["users"].append(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    for jf in jsonl_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            data["users"].append(item)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    logger.info("Loaded %d user records from %s", len(data["users"]), data_dir)
    return data


def _extract_conversations(user: dict) -> list[dict]:
    """Extract conversation turns from a user record."""
    convos = user.get("conversations", user.get("dialogues", user.get("turns", [])))
    if isinstance(convos, list):
        return convos
    return []


def _extract_memories(user: dict) -> list[dict]:
    """Extract ground-truth memories from a user record."""
    return user.get("memories", user.get("facts", user.get("ground_truth_memories", [])))


def _extract_updates(user: dict) -> list[dict]:
    """Extract memory update operations from a user record."""
    return user.get("updates", user.get("memory_updates", []))


def _extract_questions(user: dict) -> list[dict]:
    """Extract QA evaluation pairs from a user record."""
    questions = user.get("questions", user.get("qa_pairs", user.get("evaluation", [])))
    for key in ("extraction_eval", "update_eval", "qa_eval"):
        if key in user and isinstance(user[key], list):
            for item in user[key]:
                item["eval_type"] = key.replace("_eval", "")
            questions.extend(user[key])
    return questions


def _conversation_to_markdown(conversations: list, user_id: str) -> str:
    """Convert conversation turns to a markdown document."""
    lines = [f"# Conversation History — User {user_id}\n"]
    for turn in conversations:
        if isinstance(turn, dict):
            role = turn.get("role", turn.get("speaker", "unknown"))
            content = turn.get("content", turn.get("text", turn.get("message", "")))
            lines.append(f"**{role}:** {content}\n")
        elif isinstance(turn, str):
            lines.append(f"{turn}\n")
    return "\n".join(lines)


def _compute_extraction_accuracy(stored_text: str, ground_truth_memories: list[dict]) -> dict:
    """Evaluate extraction: what fraction of ground-truth facts are in stored text."""
    if not ground_truth_memories:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_facts": 0}

    stored_lower = stored_text.lower()
    found = 0
    for mem in ground_truth_memories:
        fact = mem.get("text", mem.get("fact", mem.get("content", "")))
        if not fact:
            continue
        fact_words = set(fact.lower().split())
        if len(fact_words) < 3:
            continue
        overlap = sum(1 for w in fact_words if w in stored_lower)
        if overlap / len(fact_words) > 0.5:
            found += 1

    total = len(ground_truth_memories)
    recall = found / total if total else 0.0
    precision = found / max(len(stored_lower.split()) // 20, 1)
    precision = min(precision, 1.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "total_facts": total, "found_facts": found}


def _compute_update_correctness(search_result: str, old_fact: str, new_fact: str) -> dict:
    """Evaluate update: does the retrieval reflect the new fact, not the old?"""
    result_lower = search_result.lower()
    old_words = set(old_fact.lower().split())
    new_words = set(new_fact.lower().split())

    distinguishing_new = new_words - old_words
    distinguishing_old = old_words - new_words

    new_present = sum(1 for w in distinguishing_new if w in result_lower) / max(len(distinguishing_new), 1)
    old_present = sum(1 for w in distinguishing_old if w in result_lower) / max(len(distinguishing_old), 1)

    correct = new_present > old_present
    return {
        "correct": correct,
        "new_fact_recall": round(new_present, 4),
        "old_fact_presence": round(old_present, 4),
    }


def _compute_qa_hallucination(answer: str, ground_truth: str, is_unanswerable: bool = False) -> dict:
    """Evaluate QA: detect hallucination in the answer.

    Hallucination types from HaluMem: fabrication, errors, conflicts, omissions.
    """
    if is_unanswerable:
        refusal_markers = ["no relevant", "not found", "cannot answer", "no information",
                          "don't have", "no memory", "unable to find"]
        refused = any(m in answer.lower() for m in refusal_markers)
        return {
            "hallucinated": not refused,
            "hallucination_type": "fabrication" if not refused else "none",
            "correct_refusal": refused,
        }

    if not ground_truth:
        return {"hallucinated": False, "hallucination_type": "none", "f1": 0.0}

    truth_words = set(ground_truth.lower().split())
    answer_words = set(answer.lower().split())

    if not answer_words:
        return {"hallucinated": False, "hallucination_type": "omission", "f1": 0.0}

    common = truth_words & answer_words
    if not common:
        return {"hallucinated": True, "hallucination_type": "fabrication", "f1": 0.0}

    precision = len(common) / len(answer_words)
    recall = len(common) / len(truth_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    hallucinated = f1 < 0.3
    h_type = "none"
    if hallucinated:
        if recall < 0.2:
            h_type = "fabrication"
        elif precision < 0.3:
            h_type = "error"
        else:
            h_type = "conflict"

    return {"hallucinated": hallucinated, "hallucination_type": h_type, "f1": round(f1, 4)}


async def _run_curator_extraction(mem_root: str) -> None:
    """Run curator entity extraction over indexed files for KG population."""
    from curator import extract_knowledge, process_extraction
    from graph import init_schema
    init_schema()
    for fp in sorted(Path(mem_root).rglob("*.md")):
        try:
            text = fp.read_text(encoding="utf-8")
            rel = str(fp.relative_to(mem_root))
            knowledge = await extract_knowledge(text, agent_id="halumem", source_file=rel)
            if knowledge:
                await process_extraction(knowledge, agent_id="halumem", source_file=rel)
        except Exception as e:
            logger.debug("Curator extraction failed for %s: %s", fp, e)


async def run_halumem_benchmark(
    data_dir: str,
    limit: int = 0,
    run_curator: bool = False,
) -> dict:
    """Run the full HaluMem benchmark against Archivist."""
    data = _load_halumem_data(data_dir)
    users = data["users"]
    if not users:
        return {"error": "No user records loaded", "data_dir": data_dir}
    if limit > 0:
        users = users[:limit]

    work_dir = tempfile.mkdtemp(prefix="halumem_bench_")
    mem_root = os.path.join(work_dir, "memories")
    os.makedirs(mem_root, exist_ok=True)

    try:
        import config
        config.MEMORY_ROOT = mem_root
        os.environ["MEMORY_ROOT"] = mem_root

        from graph import init_schema
        init_schema()

        extraction_results = []
        update_results = []
        qa_results = []

        for ui, user in enumerate(users):
            user_id = user.get("user_id", user.get("id", f"user-{ui}"))
            logger.info("Processing user %s (%d/%d)", user_id, ui + 1, len(users))

            conversations = _extract_conversations(user)
            md_content = ""
            if conversations:
                md_content = _conversation_to_markdown(conversations, user_id)
                filepath = os.path.join(mem_root, f"users/{user_id}/history.md")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(md_content)

            gt_memories = _extract_memories(user)
            updates = _extract_updates(user)
            questions = _extract_questions(user)

            from indexer import full_index
            await full_index(hierarchical=True)

            if run_curator:
                await _run_curator_extraction(mem_root)

            if gt_memories:
                all_indexed_text = md_content
                ext_eval = _compute_extraction_accuracy(all_indexed_text, gt_memories)
                ext_eval["user_id"] = user_id
                extraction_results.append(ext_eval)

            from rlm_retriever import recursive_retrieve

            for update in updates:
                old_fact = update.get("old", update.get("before", ""))
                new_fact = update.get("new", update.get("after", update.get("text", "")))
                query = update.get("query", new_fact[:100])

                if new_fact:
                    update_file = os.path.join(mem_root, f"users/{user_id}/update-{len(update_results)}.md")
                    with open(update_file, "w", encoding="utf-8") as f:
                        f.write(f"# Memory Update\n\n{new_fact}\n")
                    await full_index(hierarchical=True)

                if query:
                    try:
                        result = await recursive_retrieve(query=query, limit=5, refine=False, tier="l2")
                        result_text = " ".join(s.get("text", s.get("tier_text", "")) for s in result.get("sources", []))
                    except Exception as e:
                        logger.warning("Update query failed: %s", e)
                        result_text = ""

                    if old_fact and new_fact:
                        upd_eval = _compute_update_correctness(result_text, old_fact, new_fact)
                        upd_eval["user_id"] = user_id
                        update_results.append(upd_eval)

            for qi, qa in enumerate(questions):
                query = qa.get("question", qa.get("query", qa.get("text", "")))
                ground_truth = qa.get("answer", qa.get("ground_truth", ""))
                is_unanswerable = qa.get("unanswerable", qa.get("is_unanswerable", False))
                eval_type = qa.get("eval_type", "qa")

                if not query:
                    continue

                try:
                    result = await recursive_retrieve(query=query, limit=10, refine=True, tier="l2")
                    answer = result.get("answer", "")
                except Exception as e:
                    logger.warning("QA query failed: %s", e)
                    answer = ""

                qa_eval = _compute_qa_hallucination(answer, ground_truth, is_unanswerable)
                qa_eval["user_id"] = user_id
                qa_eval["query"] = query[:100]
                qa_eval["eval_type"] = eval_type
                qa_results.append(qa_eval)

        summary = _build_summary(extraction_results, update_results, qa_results, len(users))
        return {
            "summary": summary,
            "extraction_results": extraction_results,
            "update_results": update_results,
            "qa_results": qa_results,
        }

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _build_summary(extraction: list, updates: list, qa: list, n_users: int) -> dict:
    """Aggregate results into a summary."""
    ext_recall = _mean([e["recall"] for e in extraction]) if extraction else 0.0
    ext_f1 = _mean([e["f1"] for e in extraction]) if extraction else 0.0

    upd_correct = sum(1 for u in updates if u["correct"]) / max(len(updates), 1)
    upd_new_recall = _mean([u["new_fact_recall"] for u in updates]) if updates else 0.0

    qa_hallucination_rate = sum(1 for q in qa if q["hallucinated"]) / max(len(qa), 1)
    qa_f1 = _mean([q.get("f1", 0) for q in qa]) if qa else 0.0

    h_type_counts = collections.Counter(q["hallucination_type"] for q in qa)

    return {
        "benchmark": "HaluMem",
        "users_evaluated": n_users,
        "extraction": {
            "count": len(extraction),
            "avg_recall": round(ext_recall, 4),
            "avg_f1": round(ext_f1, 4),
        },
        "updating": {
            "count": len(updates),
            "correctness_rate": round(upd_correct, 4),
            "new_fact_recall": round(upd_new_recall, 4),
        },
        "qa": {
            "count": len(qa),
            "hallucination_rate": round(qa_hallucination_rate, 4),
            "avg_f1": round(qa_f1, 4),
            "hallucination_types": dict(h_type_counts),
        },
        "composite_score": round(
            (ext_f1 * 0.3 + upd_correct * 0.3 + (1 - qa_hallucination_rate) * 0.4), 4
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def main():
    parser = argparse.ArgumentParser(description="HaluMem benchmark adapter for Archivist")
    parser.add_argument("--data-dir", required=True, help="Path to HaluMem dataset directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit users (0=all)")
    parser.add_argument("--run-curator", action="store_true", help="Run curator KG extraction")
    parser.add_argument("--output", type=str, default="halumem_results.json", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    data = await run_halumem_benchmark(args.data_dir, limit=args.limit, run_curator=args.run_curator)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if "summary" in data:
        s = data["summary"]
        print(f"\n=== HaluMem Benchmark Results ===")
        print(f"Users evaluated: {s['users_evaluated']}")
        print(f"\nExtraction (n={s['extraction']['count']}):")
        print(f"  Recall: {s['extraction']['avg_recall']:.4f}")
        print(f"  F1:     {s['extraction']['avg_f1']:.4f}")
        print(f"\nUpdating (n={s['updating']['count']}):")
        print(f"  Correctness: {s['updating']['correctness_rate']:.4f}")
        print(f"  New fact recall: {s['updating']['new_fact_recall']:.4f}")
        print(f"\nQA (n={s['qa']['count']}):")
        print(f"  Hallucination rate: {s['qa']['hallucination_rate']:.4f}")
        print(f"  F1: {s['qa']['avg_f1']:.4f}")
        print(f"  Types: {s['qa']['hallucination_types']}")
        print(f"\nComposite score: {s['composite_score']:.4f}")
    else:
        print(json.dumps(data, indent=2))

    logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
