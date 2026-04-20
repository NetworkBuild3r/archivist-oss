"""LongMemEval benchmark adapter — evaluates Archivist against the LongMemEval
benchmark (ICLR 2025, xiaowu0162/LongMemEval).

Uses the **official evaluation protocol**:
  - QA Accuracy via LLM-as-judge (task-specific prompts from evaluate_qa.py)
  - Retrieval: Recall@5, Recall@10, NDCG@5, NDCG@10 using ground-truth evidence sessions
  - Per-category breakdown matching the paper's 6 question types

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

    Optional dedicated judge LLM (fast yes/no; does not change retrieval LLM):

       BENCHMARK_JUDGE_LLM_URL=http://192.0.2.10:11435
       BENCHMARK_JUDGE_LLM_MODEL=qwen3.6-35b-a3b
       BENCHMARK_JUDGE_LLM_API_KEY=

       NOTE: Do NOT include /v1 in the URL. The client appends /v1/chat/completions
       automatically. Setting http://host:port/v1 produces .../v1/v1/... → 404.

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

    Scripted thin runs: see ``benchmarks/README.md``.

"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import math
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


# ── Official LLM-as-judge prompts (from xiaowu0162/LongMemEval) ──────────────
# These task-specific prompts are from src/evaluation/evaluate_qa.py in the
# official repo (MIT license).  They produce a yes/no verdict with >97% human
# agreement per the paper.


def _get_judge_prompt(
    task: str, question: str, answer: str, response: str, abstention: bool = False
) -> str:
    """Build the official LongMemEval answer-check prompt for the LLM judge."""
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response "
            "from a model. Please answer yes if the model correctly identifies the question "
            "as unanswerable. The model could say that the information is incomplete, or "
            "some other information is given but the asked information is not.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )

    templates = {
        "single-session-user": (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no."
        ),
        "single-session-assistant": None,  # same as user
        "multi-session": None,  # same as user
        "temporal-reasoning": (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. "
            "In addition, do not penalize off-by-one errors for the number of days. "
            "If the question asks for the number of days/weeks/months, etc., and the model "
            "makes off-by-one errors (e.g., predicting 19 days when the answer is 18), "
            "the model's response is still correct."
        ),
        "knowledge-update": (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is "
            "the required answer."
        ),
        "single-session-preference": (
            "I will give you a question, a rubric for desired personalized response, and a "
            "response from a model. Please answer yes if the response satisfies the desired "
            "response. Otherwise, answer no. The model does not need to reflect all the points "
            "in the rubric. The response is correct as long as it recalls and utilizes the "
            "user's personal information correctly."
        ),
    }

    base = templates.get(task)
    if base is None:
        base = templates["single-session-user"]

    label_a = "Correct Answer" if task != "single-session-preference" else "Rubric"
    return (
        f"{base}\n\n"
        f"Question: {question}\n\n{label_a}: {answer}\n\n"
        f"Model Response: {response}\n\n"
        "Is the model response correct? Answer yes or no only."
    )


# ── Official retrieval metrics (from src/retrieval/eval_utils.py) ─────────────


def _dcg(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    rel = relevances[:k]
    if not rel:
        return 0.0
    result = float(rel[0])
    for i in range(1, len(rel)):
        result += rel[i] / math.log2(i + 2)
    return result


def _ndcg_at_k(retrieved_session_ids: list[int], evidence_ids: set[int], k: int) -> float:
    """NDCG@k — did we rank evidence sessions highly?"""
    relevances = [1 if sid in evidence_ids else 0 for sid in retrieved_session_ids]
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = _dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(relevances[:k], k) / ideal_dcg


def _evidence_session_indices(item: dict) -> set[int]:
    """Map LongMemEval ``answer_session_ids`` to 0-based haystack session indices.

    The cleaned dataset stores string IDs (e.g. ``answer_280352e9``) in both
    ``answer_session_ids`` and ``haystack_session_ids``; the adapter filenames
    use ``session-{1-based index}`` order matching ``haystack_sessions``.
    """
    haystack_ids = item.get("haystack_session_ids") or []
    answer_ids = set(item.get("answer_session_ids") or [])
    if not haystack_ids or not answer_ids:
        return set()
    return {i for i, sid in enumerate(haystack_ids) if sid in answer_ids}


def _recall_at_k(retrieved_session_ids: list[int], evidence_ids: set[int], k: int) -> float:
    """Recall@k — fraction of evidence sessions found in top-k retrieved."""
    if not evidence_ids:
        return 0.0
    top_k = set(retrieved_session_ids[:k])
    return len(top_k & evidence_ids) / len(evidence_ids)


# ── Data loading and helpers ──────────────────────────────────────────────────


def _load_data(data_file: str) -> list[dict]:
    """Load a LongMemEval JSON file (longmemeval_s_cleaned.json etc.)."""
    with open(data_file, encoding="utf-8") as f:
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
    """Convert LongMemEval sessions to markdown files (one per session)."""
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


async def _llm_judge(
    question: str, ground_truth: str, hypothesis: str, question_type: str, is_abstention: bool
) -> bool:
    """Call the LLM to judge whether the hypothesis answers the question correctly.

    Uses the same task-specific prompts as the official LongMemEval evaluation.
    Returns True if the judge says 'yes'.

    Optional **dedicated judge endpoint** (fast yes/no, e.g. Ollama on a GPU box):
    ``BENCHMARK_JUDGE_LLM_URL``, ``BENCHMARK_JUDGE_LLM_MODEL``,
    optional ``BENCHMARK_JUDGE_LLM_API_KEY`` (omit to reuse ``LLM_API_KEY``;
    set to empty for no ``Authorization`` header). Base URL must not include ``/v1``.
    """
    import config as cfg
    from llm import llm_query

    prompt = _get_judge_prompt(
        task=question_type,
        question=question,
        answer=ground_truth,
        response=hypothesis,
        abstention=is_abstention,
    )

    judge_url = (cfg.BENCHMARK_JUDGE_LLM_URL or "").strip()
    if judge_url:
        j_model = cfg.BENCHMARK_JUDGE_LLM_MODEL or cfg.LLM_MODEL
        # None → llm_query uses LLM_API_KEY; set to "" or ollama for a dedicated header.
        j_key = cfg.BENCHMARK_JUDGE_LLM_API_KEY
    else:
        j_model = cfg.LLM_MODEL
        j_key = None
        judge_url = ""

    try:
        response = await llm_query(
            prompt,
            max_tokens=10,
            model=j_model,
            url=judge_url,
            api_key=j_key,
            stage="longmemeval_judge",
        )
        return "yes" in response.lower()
    except Exception as e:
        logger.warning("LLM judge failed: %s — defaulting to keyword fallback", e)
        gt_words = set(ground_truth.lower().split())
        if not gt_words:
            return False
        hyp_lower = hypothesis.lower()
        found = sum(1 for w in gt_words if w in hyp_lower)
        return (found / len(gt_words)) > 0.5


# ── Main benchmark runner ────────────────────────────────────────────────────


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
      5. **LLM-as-judge** for QA accuracy (official protocol)
      6. **Recall@k** and **NDCG@k** for retrieval quality (official protocol)
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
    _rlm_restore: dict[str, object] | None = None
    _sqlite_pool_started = False

    try:
        config.MEMORY_ROOT = mem_root
        os.environ["MEMORY_ROOT"] = mem_root
        sqlite_path = os.path.join(work_dir, "graph.db")
        config.SQLITE_PATH = sqlite_path
        os.environ["SQLITE_PATH"] = sqlite_path

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        import rlm_retriever as rlm_mod
        from archivist.storage.sqlite_pool import initialize_pool
        from graph import init_schema
        from indexer import full_index
        from rlm_retriever import recursive_retrieve

        qclient = QdrantClient(url=config.QDRANT_URL, timeout=30)
        init_schema()
        # Indexer / retrieval use the async SQLite pool (FTS, graph); temp DB requires this.
        await initialize_pool()
        _sqlite_pool_started = True

        # Isolate harness from a production .env that enables the v2 cross-encoder
        # path or a high static retrieval threshold — both can yield **zero**
        # sources on LongMemEval despite successful indexing (below-threshold
        # filter or empty rerank pool).  Mutate module globals so existing
        # ``recursive_retrieve`` bytecode sees the overrides at runtime.
        _rlm_restore = {
            "RERANKER_ENABLED": rlm_mod.RERANKER_ENABLED,
            "RETRIEVAL_THRESHOLD": rlm_mod.RETRIEVAL_THRESHOLD,
        }
        rlm_mod.RERANKER_ENABLED = False
        rlm_mod.RETRIEVAL_THRESHOLD = 0.0

        all_results: list[dict] = []
        # Track by the 6 official question_type categories
        results_by_qtype: dict[str, list[dict]] = collections.defaultdict(list)

        for qi, item in enumerate(items):
            qid = str(item.get("question_id", qi))
            question = item.get("question", "")
            answer_gt = item.get("answer", "")
            question_type = item.get("question_type", "")
            is_abstention = str(qid).endswith("_abs")
            sessions = item.get("haystack_sessions", [])
            dates = item.get("haystack_dates", [])

            if not question or not sessions:
                logger.warning("Skipping item %s: no question or sessions", qid)
                continue

            md_files = _sessions_to_markdown(sessions, dates, qid)
            for filename, content in md_files:
                filepath = os.path.join(mem_root, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            try:
                qclient.delete_collection(config.QDRANT_COLLECTION)
            except Exception:
                pass
            qclient.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=config.VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )

            chunk_count = await full_index(hierarchical=True, root_dir=mem_root)

            if run_curator:
                await _run_curator_on_files(mem_root)

            t0 = time.monotonic()
            try:
                result = await recursive_retrieve(
                    query=question,
                    namespace="",
                    limit=search_limit,
                    refine=refine,
                    tier="l2",
                    threshold=0.0,
                )
            except Exception as e:
                logger.warning("Query %s failed: %s", qid, e)
                result = {"answer": "", "sources": []}
            elapsed_ms = (time.monotonic() - t0) * 1000

            # JSON null → Python None: .get("answer", "") still returns None; coerce to str.
            answer = result.get("answer") or ""
            # Thin runs use ``--no-refine`` (speed): the pipeline returns empty
            # synthesis but still attaches ``sources``.  Feed retrieved text to
            # the judge so QA accuracy is not trivially 0 %.
            if not str(answer).strip() and not refine:
                srcs = result.get("sources") or []
                if srcs:
                    parts: list[str] = []
                    for s in srcs[: min(10, search_limit)]:
                        t = (s.get("tier_text") or s.get("text") or "").strip()
                        if t:
                            parts.append(t[:2000])
                    answer = "\n\n---\n\n".join(parts)

            # ── Official metric 1: LLM-as-judge QA accuracy ──────────────
            correct = await _llm_judge(
                question=question,
                ground_truth=answer_gt,
                hypothesis=answer,
                question_type=question_type,
                is_abstention=is_abstention,
            )

            # ── Official metric 2: Session-level Recall@k and NDCG@k ────
            evidence_ids = _evidence_session_indices(item)
            retrieved_files = [s.get("file_path", "") for s in result.get("sources", [])]
            retrieved_session_ids: list[int] = []
            seen: set[int] = set()
            for rf in retrieved_files:
                for si in range(len(sessions)):
                    if si not in seen and (
                        f"session-{si + 1:04d}" in rf or f"session-{si:04d}" in rf
                    ):
                        retrieved_session_ids.append(si)
                        seen.add(si)
                        break

            retrieval_metrics = {}
            if evidence_ids and not is_abstention:
                for k in (5, 10):
                    retrieval_metrics[f"recall@{k}"] = round(
                        _recall_at_k(retrieved_session_ids, evidence_ids, k), 4
                    )
                    retrieval_metrics[f"ndcg@{k}"] = round(
                        _ndcg_at_k(retrieved_session_ids, evidence_ids, k), 4
                    )

            entry = {
                "question_id": qid,
                "question_type": question_type,
                "is_abstention": is_abstention,
                "query": question[:200],
                "ground_truth": answer_gt[:200],
                "hypothesis": answer[:500],
                "correct": correct,
                "retrieval": retrieval_metrics,
                "latency_ms": round(elapsed_ms, 1),
                "sources_count": len(result.get("sources", [])),
                "chunks_indexed": chunk_count,
                "sessions_count": len(sessions),
            }
            all_results.append(entry)
            results_by_qtype[question_type].append(entry)

            # Clean up for next question
            for filename, _ in md_files:
                fp = os.path.join(mem_root, filename)
                if os.path.exists(fp):
                    os.remove(fp)
            parent = os.path.join(mem_root, f"longmemeval-{qid}")
            if os.path.isdir(parent):
                shutil.rmtree(parent, ignore_errors=True)

            status = "✓" if correct else "✗"
            if (qi + 1) % 5 == 0 or qi == 0:
                running_acc = _mean([float(r["correct"]) for r in all_results])
                logger.info(
                    "[%d/%d] %s q=%s acc=%.1f%%",
                    qi + 1,
                    len(items),
                    status,
                    qid,
                    running_acc * 100,
                )

        # ── Build official-format summary ─────────────────────────────────
        non_abs = [r for r in all_results if not r["is_abstention"]]
        abs_only = [r for r in all_results if r["is_abstention"]]

        by_qtype = {}
        task_accs = []
        for qt, qt_results in results_by_qtype.items():
            acc = _mean([float(r["correct"]) for r in qt_results])
            task_accs.append(acc)
            retrieval_keys = ["recall@5", "ndcg@5", "recall@10", "ndcg@10"]
            qt_retrieval = {}
            for rk in retrieval_keys:
                vals = [r["retrieval"].get(rk, 0) for r in qt_results if r["retrieval"]]
                if vals:
                    qt_retrieval[rk] = round(_mean(vals), 4)
            by_qtype[qt] = {
                "count": len(qt_results),
                "accuracy": round(acc, 4),
                **qt_retrieval,
            }

        overall_acc = _mean([float(r["correct"]) for r in all_results])
        task_avg_acc = _mean(task_accs) if task_accs else 0.0
        abs_acc = _mean([float(r["correct"]) for r in abs_only]) if abs_only else None

        retrieval_summary = {}
        for rk in ["recall@5", "ndcg@5", "recall@10", "ndcg@10"]:
            vals = [r["retrieval"].get(rk, 0) for r in non_abs if r["retrieval"]]
            if vals:
                retrieval_summary[rk] = round(_mean(vals), 4)

        _ju = getattr(config, "BENCHMARK_JUDGE_LLM_URL", "") or ""
        _jm = getattr(config, "BENCHMARK_JUDGE_LLM_MODEL", "") or ""
        summary = {
            "benchmark": "LongMemEval",
            "variant": variant or "default",
            "eval_protocol": "llm-as-judge (official, per-task prompts)",
            "judge_model": (
                f"{_jm or config.LLM_MODEL} @ {_ju}"
                if _ju
                else os.environ.get("LLM_MODEL", "unknown")
            ),
            "judge_llm_url": _ju or config.LLM_URL,
            "data_file": os.path.basename(data_file),
            "total_questions": len(items),
            "evaluated_questions": len(all_results),
            "overall_accuracy": round(overall_acc, 4),
            "task_averaged_accuracy": round(task_avg_acc, 4),
            "abstention_accuracy": round(abs_acc, 4) if abs_acc is not None else None,
            "retrieval": retrieval_summary,
            "by_question_type": by_qtype,
            "search_limit": search_limit,
            "refine": refine,
            "curator": run_curator,
        }

        return {"summary": summary, "results": all_results}

    finally:
        if _sqlite_pool_started:
            try:
                from archivist.storage.sqlite_pool import close_pool

                await close_pool()
            except Exception:
                logger.debug("LongMemEval: close_pool failed", exc_info=True)
        if _rlm_restore is not None:
            import rlm_retriever as _rlm

            _rlm.RERANKER_ENABLED = _rlm_restore["RERANKER_ENABLED"]  # type: ignore[assignment]
            _rlm.RETRIEVAL_THRESHOLD = _rlm_restore["RETRIEVAL_THRESHOLD"]  # type: ignore[assignment]
        config.MEMORY_ROOT = original_memory_root
        config.SQLITE_PATH = original_sqlite
        os.environ["MEMORY_ROOT"] = original_memory_root
        os.environ["SQLITE_PATH"] = original_sqlite
        shutil.rmtree(work_dir, ignore_errors=True)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_summary(s: dict) -> None:
    """Print a single variant's summary in a format comparable to published results."""
    print(f"\n{'=' * 72}")
    print(f"  LongMemEval — {s.get('variant', 'default')}  (judge: {s.get('judge_model', '?')})")
    print(f"{'=' * 72}")
    print(f"  Dataset:    {s['data_file']}")
    print(f"  Questions:  {s['evaluated_questions']}/{s['total_questions']}")
    print(f"  Protocol:   {s.get('eval_protocol', 'llm-as-judge')}")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  Overall Accuracy:        {s['overall_accuracy']:>6.1%}                         │")
    print(
        f"  │  Task-Averaged Accuracy:  {s['task_averaged_accuracy']:>6.1%}                         │"
    )
    if s.get("abstention_accuracy") is not None:
        print(
            f"  │  Abstention Accuracy:     {s['abstention_accuracy']:>6.1%}                         │"
        )
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    retr = s.get("retrieval", {})
    if retr:
        print(
            f"  Retrieval:  Recall@5={retr.get('recall@5', 0):.4f}  "
            f"NDCG@5={retr.get('ndcg@5', 0):.4f}  "
            f"Recall@10={retr.get('recall@10', 0):.4f}  "
            f"NDCG@10={retr.get('ndcg@10', 0):.4f}"
        )
        print()

    print(f"  {'Question Type':<28s}  {'Acc':>6s}  {'R@5':>6s}  {'NDCG@5':>6s}  {'n':>4s}")
    print(f"  {'-' * 28}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 4}")
    for qt, vals in s.get("by_question_type", {}).items():
        acc = f"{vals['accuracy']:.1%}"
        r5 = f"{vals.get('recall@5', 0):.4f}" if vals.get("recall@5") else "  —"
        n5 = f"{vals.get('ndcg@5', 0):.4f}" if vals.get("ndcg@5") else "  —"
        print(f"  {qt:<28s}  {acc:>6s}  {r5:>6s}  {n5:>6s}  {vals['count']:>4d}")

    print(f"{'=' * 72}")


def _print_ablation_comparison(ablation_data: dict) -> None:
    """Print a side-by-side comparison table across variants."""
    variants = ablation_data.get("variants", {})
    if not variants:
        return

    print(f"\n{'=' * 80}")
    print("  LongMemEval Ablation — QA Accuracy (Official LLM-as-Judge)")
    print(f"{'=' * 80}")
    print(f"  {'Variant':<25s}  {'Accuracy':>10s}  {'Task-Avg':>10s}  {'R@5':>8s}  {'NDCG@5':>8s}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 8}")

    for vname, vdata in variants.items():
        s = vdata.get("summary", {})
        acc = s.get("overall_accuracy", 0)
        tavg = s.get("task_averaged_accuracy", 0)
        r5 = s.get("retrieval", {}).get("recall@5", 0)
        n5 = s.get("retrieval", {}).get("ndcg@5", 0)
        print(f"  {vname:<25s}  {acc:>9.1%}  {tavg:>9.1%}  {r5:>8.4f}  {n5:>8.4f}")

    print(f"{'=' * 80}")


async def main():
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark adapter for Archivist (official eval protocol)",
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to longmemeval_s_cleaned.json or longmemeval_m_cleaned.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")
    parser.add_argument("--no-refine", action="store_true", help="Skip LLM refinement")
    parser.add_argument("--run-curator", action="store_true", help="Run curator KG extraction")
    parser.add_argument("--search-limit", type=int, default=10, help="Top-k retrieval limit")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        help="Run a single pipeline variant (default: no overrides)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run all variants sequentially and produce comparison table",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Comma-separated variant names for --ablation (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".benchmarks/longmemeval_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.ablation:
        variant_names = (
            [v.strip() for v in args.variants.split(",") if v.strip()]
            if args.variants
            else list(VARIANTS.keys())
        )
        ablation_results: dict = {
            "benchmark": "LongMemEval",
            "mode": "ablation",
            "eval_protocol": "llm-as-judge (official, per-task prompts)",
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
