"""Context stuffing baseline — the naive "load everything into the prompt" approach.

This module simulates how an agent like OpenClaw would consume memory files without
a retrieval system: concatenate every .md file in the corpus into one prompt and
send it to the LLM. This is the break-even comparison against Archivist retrieval.

The break-even question:
    At what corpus size does context stuffing overflow the context window (or degrade
    recall so badly) that Archivist retrieval is clearly superior?

Key characteristics of context stuffing:
    - Token cost = entire corpus size, per query (no batching benefit across queries)
    - Recall is theoretically perfect when all files fit in context
    - "Lost in the middle" degrades recall even within the context window
    - Past the context limit, files must be truncated — older/less-salient facts are lost
    - Latency scales with corpus token count (larger prompt = slower completion)
"""

import asyncio
import logging
import os
import statistics
import time

logger = logging.getLogger("archivist.benchmark.stuffing")

_STUFF_SYSTEM = (
    "You are a memory assistant. The following are memory files from an AI agent fleet. "
    "Answer the query using only information found in these memories. "
    "Be concise and specific. If the answer is not present, say so."
)


def load_all_memories(corpus_dir: str) -> list[dict]:
    """Walk corpus_dir/agents/, read every .md file, return sorted list of {path, text}."""
    agents_dir = os.path.join(corpus_dir, "agents")
    memories = []
    if not os.path.isdir(agents_dir):
        return memories
    for root, _dirs, files in os.walk(agents_dir):
        for fname in sorted(files):
            if fname.endswith(".md"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                    memories.append({"path": fpath, "text": text})
                except Exception as e:
                    logger.warning("Could not read %s: %s", fpath, e)
    # Sort for determinism: by path so same files always appear in the same order.
    memories.sort(key=lambda m: m["path"])
    return memories


def count_corpus_tokens(memories: list[dict]) -> int:
    """Sum token counts over all loaded memory texts."""
    sys.path.insert(0, _src_path())
    from tokenizer import count_tokens
    return sum(count_tokens(m["text"]) for m in memories)


def _src_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _build_stuffed_prompt(query: str, memories: list[dict], context_budget: int) -> tuple[str, int, bool, int]:
    """Build a context-stuffed user prompt.

    Concatenates memory files in order until the context budget is reached (leaving
    ~2000 tokens of headroom for the system prompt + completion).

    Returns:
        prompt (str), tokens_used (int), overflow (bool), files_included (int)
    """
    import sys as _sys
    _sys.path.insert(0, _src_path())
    from tokenizer import count_tokens

    # Reserve tokens for system prompt + query + completion buffer.
    system_tokens = count_tokens(_STUFF_SYSTEM)
    query_tokens = count_tokens(query)
    # Generous headroom: system + query + ~1024 completion + 200 overhead.
    available = context_budget - system_tokens - query_tokens - 1224

    parts = []
    tokens_used = 0
    overflow = False
    files_included = 0

    for mem in memories:
        chunk_tokens = count_tokens(mem["text"])
        if tokens_used + chunk_tokens > available:
            overflow = True
            break
        parts.append(mem["text"])
        tokens_used += chunk_tokens
        files_included += 1

    memories_block = "\n\n---\n\n".join(parts)
    prompt = f"Memory files:\n\n{memories_block}\n\n---\n\nQuery: {query}"
    return prompt, tokens_used, overflow, files_included


async def stuff_and_query(
    query: str,
    memories: list[dict],
    context_budget: int,
    model: str | None = None,
) -> dict:
    """Run a single query using the context stuffing approach.

    Returns a result dict compatible with the pipeline benchmark format:
        answer, tokens_used, overflow, files_included, files_total, latency_ms
    """
    import sys as _sys
    _sys.path.insert(0, _src_path())
    from llm import llm_query
    from config import LLM_MODEL

    _model = model or LLM_MODEL
    prompt, tokens_used, overflow, files_included = _build_stuffed_prompt(
        query, memories, context_budget
    )

    t0 = time.monotonic()
    try:
        answer = await llm_query(prompt, system=_STUFF_SYSTEM, model=_model, max_tokens=512)
    except Exception as e:
        logger.warning("Stuffing query failed: %s", e)
        answer = ""
    elapsed_ms = (time.monotonic() - t0) * 1000

    return {
        "answer": answer,
        "tokens_used": tokens_used,
        "overflow": overflow,
        "files_included": files_included,
        "files_total": len(memories),
        "latency_ms": round(elapsed_ms, 1),
    }


def _eval_stuffing_result(result: dict, question: dict) -> dict:
    """Compute keyword recall and MRR for a stuffing result against a question."""
    import sys as _sys
    _sys.path.insert(0, _src_path())
    from tokenizer import count_tokens

    answer_text = result.get("answer", "")
    expected_keywords = question.get("expected_keywords", [])
    expected_answer = question.get("expected_answer", "")

    # Keyword recall over the answer text only (no sources list for stuffing).
    if expected_keywords:
        text_lower = answer_text.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
        keyword_recall = found / len(expected_keywords)
    else:
        keyword_recall = 0.0

    # MRR: treat the whole answer as a single "source."
    mrr = 0.0
    if expected_answer:
        answer_words = set(expected_answer.lower().split())
        resp_words = set(answer_text.lower().split())
        overlap = len(answer_words & resp_words) / max(len(answer_words), 1)
        if overlap > 0.3:
            mrr = 1.0  # rank 1 — it's the only "source"

    return {
        "keyword_recall": round(keyword_recall, 3),
        "mrr": round(mrr, 3),
        "answer_tokens": count_tokens(answer_text),
    }


async def run_stuffing_baseline(
    corpus_dir: str,
    questions: list[dict],
    context_budget: int,
    model: str | None = None,
    call_llm: bool = True,
    *,
    progress_pct_step: int = 10,
    checkpoint_path: str | None = None,
    memory_scale: str | None = None,
    use_progress_bar: bool = True,
) -> dict:
    """Run all questions through the context stuffing baseline.

    Args:
        corpus_dir: root containing agents/ tree.
        questions: same question list as the pipeline benchmark.
        context_budget: token budget (default DEFAULT_CONTEXT_BUDGET = 128000).
        model: LLM model name; None → use LLM_MODEL from config.
        call_llm: if False, skip actual LLM calls (token-count-only mode for fast overflow detection).

    Returns a dict with summary metrics and per-question results.

    Optional ``progress_pct_step`` / ``checkpoint_path`` enable milestone logs and
    atomic ``*.run_state.json`` checkpoints (same contract as ``evaluate.run_variant``).
    """
    from benchmarks.pipeline.run_progress import ProgressTracker

    memories = load_all_memories(corpus_dir)
    corpus_tokens = count_corpus_tokens(memories)
    file_count = len(memories)

    # Determine overflow for the corpus as a whole (before per-query headroom).
    _, _, corpus_overflow, _ = _build_stuffed_prompt("test", memories, context_budget)

    logger.info(
        "Stuffing baseline: %d files, ~%d tokens, overflow=%s",
        file_count,
        corpus_tokens,
        corpus_overflow,
    )

    results = []
    latencies = []
    recall_scores = []
    mrr_scores = []
    overflow_count = 0

    progress = ProgressTracker(
        total=len(questions),
        phase="stuffing",
        memory_scale=memory_scale,
        pct_step=progress_pct_step,
        checkpoint_path=checkpoint_path,
        use_progress_bar=use_progress_bar,
    )

    try:
        for q in questions:
            if call_llm:
                result = await stuff_and_query(q["query"], memories, context_budget, model=model)
            else:
                # Token-count-only: no LLM call — use overflow flag and empty answer.
                _, tokens_used, overflow, files_included = _build_stuffed_prompt(
                    q["query"], memories, context_budget
                )
                result = {
                    "answer": "",
                    "tokens_used": tokens_used,
                    "overflow": overflow,
                    "files_included": files_included,
                    "files_total": file_count,
                    "latency_ms": 0.0,
                }

            if result["overflow"]:
                overflow_count += 1

            evals = _eval_stuffing_result(result, q)
            recall_scores.append(evals["keyword_recall"])
            mrr_scores.append(evals["mrr"])
            if result["latency_ms"] > 0:
                latencies.append(result["latency_ms"])

            results.append({
                "question_id": q["id"],
                "query": q["query"],
                "query_type": q.get("query_type", ""),
                "latency_ms": result["latency_ms"],
                "tokens_used": result["tokens_used"],
                "overflow": result["overflow"],
                "files_included": result["files_included"],
                "files_total": file_count,
                "keyword_recall": evals["keyword_recall"],
                "mrr": evals["mrr"],
                "answer_tokens": evals["answer_tokens"],
                "llm_called": call_llm,
            })

            progress.step(
                len(results),
                results=results,
                rolling_recall=statistics.mean(recall_scores) if recall_scores else 0.0,
                rolling_mrr=statistics.mean(mrr_scores) if mrr_scores else 0.0,
            )
    finally:
        progress.close()

    # Aggregate per-query-type metrics (same pattern as run_variant in evaluate.py)
    by_type: dict = {}
    for r in results:
        qt = r.get("query_type", "")
        if qt not in by_type:
            by_type[qt] = {"recall": [], "mrr": [], "overflow": [], "latency": []}
        by_type[qt]["recall"].append(r["keyword_recall"])
        by_type[qt]["mrr"].append(r["mrr"])
        by_type[qt]["overflow"].append(1 if r.get("overflow") else 0)
        if r["latency_ms"] > 0:
            by_type[qt]["latency"].append(r["latency_ms"])

    by_query_type = {
        qt: {
            "count": len(vals["recall"]),
            "recall": round(statistics.mean(vals["recall"]), 4),
            "mrr": round(statistics.mean(vals["mrr"]), 4),
            "overflow_pct": round(statistics.mean(vals["overflow"]) * 100, 1),
            "latency_p50": round(statistics.median(vals["latency"]), 1) if vals["latency"] else None,
        }
        for qt, vals in by_type.items()
    }

    summary = {
        "mode": "context_stuffing",
        "corpus_dir": os.path.abspath(corpus_dir),
        "file_count": file_count,
        "corpus_tokens": corpus_tokens,
        "corpus_overflow": corpus_overflow,
        "context_budget": context_budget,
        "total_queries": len(questions),
        "overflow_count": overflow_count,
        "overflow_pct": round(overflow_count / max(len(questions), 1) * 100, 1),
        "recall": round(statistics.mean(recall_scores), 4) if recall_scores else 0.0,
        "mrr": round(statistics.mean(mrr_scores), 4) if mrr_scores else 0.0,
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "latency_p95_ms": round(
            sorted(latencies)[int(len(latencies) * 0.95)], 1
        ) if len(latencies) >= 2 else (latencies[0] if latencies else None),
        "llm_called": call_llm,
        "by_query_type": by_query_type,
    }

    return {"summary": summary, "results": results}


import sys  # noqa: E402 — needed at module level for _src_path references above
