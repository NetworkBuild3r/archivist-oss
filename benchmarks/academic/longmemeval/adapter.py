"""LongMemEval benchmark adapter — evaluates Archivist against LongMemEval
(ICLR 2025, xiaowu0162/LongMemEval) under the official evaluation protocol.

Implements all ``INIT-023/SPEC-003`` adapter guarantees (B1–B7):
  - **B1**: Separates retrieval context assembly (``get_relevant_context``) from
    the pinned Answer LLM step. Retrieval-only mode (``--no-refine`` or
    ``--mode retrieval-only``) computes ``Recall@k``/``NDCG@k`` only and never
    passes an empty string to ``_llm_judge``.
  - **B2**: Complete per-question isolation — fresh SQLite DB, fresh graph pool,
    ``hot_cache.clear()``, and fresh vector collection per question.
  - **B3**: Maps string ``haystack_session_ids[si]`` directly against
    ``set(item["answer_session_ids"])`` (fixing the int-vs-str zero-recall bug).
  - **B4**: Threads ``item["question_date"]`` as ``reference_time`` / ``as_of``
    into retrieval and answer generation.
  - **B5**: ``--profile default`` runs unmodified production ``ArchivistSettings``
    with zero hidden ``BENCHMARK_FAST`` or threshold overrides.
  - **B6**: Pinned ``gpt-4o-2024-08-06`` judge (``temperature=0``), zero silent
    keyword fallback (retries + ``JudgeEvaluationError``), atomic resumable
    ``results.jsonl``, SHA-256 disk caching, and 95% bootstrap CIs.
  - **B7**: Exercises production ``get_relevant_context(..., max_tokens=...)``
    with ``--max-tokens`` (default ``4000``) / ``--max-tokens-sweep`` and logs
    ``latency_ms`` (p50/p95), ``context_tokens`` (median/p95), ``ingest_time_ms``,
    ``ingest_llm_calls``, ``judge_fallbacks`` (always ``0``), and ``estimated_usd``.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from benchmarks.cache import BudgetExceededError, DiskCache, SpendTracker
from benchmarks.env_loader import load_repo_env
from benchmarks.stats import (
    MIN_N_FOR_CI,
    build_run_manifest,
    maybe_ci,
    percentile,
    prompt_sha256,
)

load_repo_env()
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

logger = logging.getLogger("archivist.benchmark.longmemeval")

DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"
DEFAULT_ANSWER_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 4000
DEFAULT_CACHE_DIR = ".benchmarks/cache"

_SESSION_INDEX_RE = re.compile(r"session-(\d{4})")


class JudgeEvaluationError(RuntimeError):
    """Raised when the LLM judge fails after exhausting retries (zero silent fallback)."""


# ── Named pipeline variants (optional — `--profile default` applies NO overrides) ──
VARIANTS: dict[str, dict[str, str]] = {
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
        "RETRIEVAL_THRESHOLD": "0.25",
    },
    "full_pipeline": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "0",
        "TEMPORAL_INTENT_ENABLED": "true",
        "BM25_RESCUE_ENABLED": "true",
        "ADAPTIVE_VECTOR_LIMIT_ENABLED": "true",
        "TOPIC_ROUTING_ENABLED": "false",
        "RETRIEVAL_THRESHOLD": "0.25",
    },
    "full_plus_topic": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "false",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "0",
        "TEMPORAL_INTENT_ENABLED": "true",
        "BM25_RESCUE_ENABLED": "true",
        "ADAPTIVE_VECTOR_LIMIT_ENABLED": "true",
        "TOPIC_ROUTING_ENABLED": "true",
        "RETRIEVAL_THRESHOLD": "0.25",
    },
    "full_plus_rerank": {
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "true",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "0",
        "TEMPORAL_INTENT_ENABLED": "true",
        "BM25_RESCUE_ENABLED": "true",
        "ADAPTIVE_VECTOR_LIMIT_ENABLED": "true",
        "TOPIC_ROUTING_ENABLED": "true",
        "RETRIEVAL_THRESHOLD": "0.25",
    },
}


def _apply_variant(variant_name: str) -> None:
    """Apply explicit config overrides for a named variant and refresh bound modules."""
    import importlib

    env_overrides = VARIANTS[variant_name]
    for key, value in env_overrides.items():
        os.environ[key] = value
    import archivist.core.config as config

    importlib.reload(config)

    # Refresh dependent modules that bind config flags by value at import time
    # so sequential --ablation runs actually observe the new variant's settings.
    for mod_name in (
        "archivist.storage.graph_retrieval",
        "archivist.retrieval.rlm_retriever",
        "archivist.write.indexer",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            importlib.reload(mod)


# ── Official LLM-as-judge prompts (from xiaowu0162/LongMemEval) ──────────────

_ANSWER_SYSTEM_PROMPT = (
    "You are a concise, accurate personal memory assistant. Answer the user's "
    "question using ONLY the retrieved memory context provided. If the context "
    "does not contain enough information to answer the question, explicitly "
    "state that the information is not available in your memory."
)


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
        "single-session-assistant": None,
        "multi-session": None,
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


def _judge_prompt_template_hash() -> str:
    canonical = "\n---\n".join(
        _get_judge_prompt(t, "{q}", "{a}", "{r}", abstention=(t == "_abs"))
        for t in (
            "single-session-user",
            "temporal-reasoning",
            "knowledge-update",
            "single-session-preference",
            "_abs",
        )
    )
    return prompt_sha256(canonical)


# ── Official retrieval metrics (string session IDs — B3) ─────────────────────


def _dcg(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    rel = relevances[:k]
    if not rel:
        return 0.0
    result = float(rel[0])
    for i in range(1, len(rel)):
        result += rel[i] / math.log2(i + 2)
    return result


def _ndcg_at_k(retrieved_session_ids: list[str], evidence_ids: set[str], k: int) -> float:
    """NDCG@k against the true ideal ranking of ``min(len(evidence_ids), k)`` relevant docs."""
    if not evidence_ids or k <= 0:
        return 0.0
    relevances = [1 if sid in evidence_ids else 0 for sid in retrieved_session_ids[:k]]
    ideal = [1] * min(len(evidence_ids), k)
    ideal_dcg = _dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(relevances, k) / ideal_dcg


def _recall_at_k(retrieved_session_ids: list[str], evidence_ids: set[str], k: int) -> float:
    """Recall@k — fraction of ground-truth evidence sessions found in top-k retrieved."""
    if not evidence_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_session_ids[:k])
    return len(top_k & evidence_ids) / len(evidence_ids)


def _normalize_source(src: Any) -> dict[str, Any]:
    """Normalize a retrieval source (dict or ``ContextChunk`` dataclass) into a plain dict."""
    if isinstance(src, dict):
        return src
    return {
        "id": getattr(src, "memory_id", getattr(src, "id", "")),
        "text": getattr(src, "text", ""),
        "tier_text": getattr(src, "tier_text", getattr(src, "text", "")),
        "tier": getattr(src, "tier", ""),
        "file_path": getattr(src, "file_path", ""),
        "date": getattr(src, "date", ""),
        "content_date": getattr(src, "content_date", getattr(src, "date", "")),
        "score": getattr(src, "score", 0.0),
        "agent_id": getattr(src, "agent_id", ""),
        "session_id": getattr(src, "session_id", ""),
    }


def map_sources_to_session_ids(
    sources: list[Any],
    session_ids: list[str],
    filename_to_session_id: dict[str, str],
) -> list[str]:
    """Map retrieved source dicts or ``ContextChunk`` objects to string session IDs (B3)."""
    mapped: list[str] = []
    seen: set[str] = set()
    for raw_src in sources:
        src = _normalize_source(raw_src)
        sid = str(src.get("session_id") or "").strip()
        if not sid:
            rf = str(src.get("file_path") or "")
            sid = filename_to_session_id.get(rf, "")
            if not sid:
                # Match suffix (relative vs absolute path)
                for fname, candidate_sid in filename_to_session_id.items():
                    if rf.endswith(fname) or fname.endswith(rf):
                        sid = candidate_sid
                        break
            if not sid:
                m = _SESSION_INDEX_RE.search(rf)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(session_ids):
                        sid = session_ids[idx]
        if sid and sid not in seen:
            seen.add(sid)
            mapped.append(sid)
    return mapped


# ── Data loading, markdown formatting & atomic JSONL checkpoints ─────────────


def _load_data(data_file: str) -> list[dict[str, Any]]:
    """Load a LongMemEval JSON file (``longmemeval_s_cleaned.json`` etc.)."""
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
    sessions: list[list[dict[str, Any]]],
    dates: list[str],
    question_id: str,
    session_ids: list[str] | None = None,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Convert LongMemEval sessions to markdown files + filename→session_id map."""
    files: list[tuple[str, str]] = []
    fname_to_sid: dict[str, str] = {}
    safe_qid = re.sub(r"[^A-Za-z0-9._-]", "_", str(question_id))
    for si, session in enumerate(sessions):
        sid = (
            str(session_ids[si])
            if session_ids and si < len(session_ids)
            else f"session-{si + 1:04d}"
        )
        date_str = dates[si] if si < len(dates) else f"session-{si:04d}"
        safe_date = date_str.replace("/", "-").replace(" ", "_")[:20]
        lines = [f"# Session {si + 1} ({sid}) — {date_str}\n"]
        for turn in session:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"**{role}:** {content}\n")
        filename = f"longmemeval-{safe_qid}/session-{si + 1:04d}-{safe_date}.md"
        files.append((filename, "\n".join(lines)))
        fname_to_sid[filename] = sid
    return files, fname_to_sid


def _load_checkpoint_jsonl(checkpoint_path: Path | None) -> dict[str, dict[str, Any]]:
    """Load previously completed question results from an atomic JSONL checkpoint."""
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint_path is None or not checkpoint_path.is_file():
        return completed
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
                qid = str(rec.get("question_id", ""))
                if qid:
                    completed[qid] = rec
            except ValueError:
                logger.warning("Skipping corrupt checkpoint line in %s", checkpoint_path)
    return completed


def _append_checkpoint_jsonl(checkpoint_path: Path | None, entry: dict[str, Any]) -> None:
    """Atomically append one completed question entry to JSONL and fsync."""
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


async def _run_curator_on_files(mem_root: str) -> int:
    """Run curator entity extraction on all indexed files. Returns LLM call count."""
    from archivist.lifecycle.curator import extract_knowledge, process_extraction
    from archivist.storage.graph import init_schema

    init_schema()
    llm_calls = 0
    md_files = sorted(Path(mem_root).rglob("*.md"))
    for fp in md_files:
        try:
            text = fp.read_text(encoding="utf-8")
            rel = str(fp.relative_to(mem_root))
            llm_calls += 1
            knowledge = await extract_knowledge(text, agent_id="longmemeval", source_file=rel)
            if knowledge:
                await process_extraction(knowledge, agent_id="longmemeval", source_file=rel)
        except Exception as e:
            logger.debug("Curator extraction failed for %s: %s", fp, e)
    return llm_calls


# ── Answer generation & pinned LLM-as-judge (B1, B4, B6) ─────────────────────


def _format_packed_context(sources: list[Any]) -> str:
    """Format packed retrieval sources (dicts or ``ContextChunk``s) for the Answer LLM."""
    parts: list[str] = []
    for idx, raw_src in enumerate(sources, 1):
        src = _normalize_source(raw_src)
        body = (src.get("tier_text") or src.get("text") or "").strip()
        if not body:
            continue
        date_str = src.get("date") or src.get("content_date") or ""
        header = f"[Memory {idx}" + (f" | date={date_str}" if date_str else "") + "]"
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


async def _generate_answer(
    *,
    question: str,
    context_text: str,
    question_date: str = "",
    model: str = "",
    cache: DiskCache | None = None,
    spend_tracker: SpendTracker | None = None,
) -> str:
    """Generate a final QA hypothesis from packed context using a pinned Answer LLM (B1, B4)."""
    import archivist.core.config as cfg
    from archivist.features.llm import llm_query
    from archivist.utils.tokenizer import count_tokens

    answer_model = model or cfg.LLM_MODEL or DEFAULT_ANSWER_MODEL
    date_header = f"Current date: {question_date}\n\n" if question_date else ""
    user_prompt = (
        f"{date_header}"
        f"Retrieved memories:\n{context_text or '(no memories retrieved)'}\n\n"
        f"Question: {question}\n\nAnswer concisely:"
    )

    cache_payload = {
        "model": answer_model,
        "system": _ANSWER_SYSTEM_PROMPT,
        "prompt": user_prompt,
        "temperature": 0.0,
    }
    if cache is not None:
        found, cached = cache.get("longmemeval_answer", cache_payload)
        if found:
            return str(cached)

    if spend_tracker is not None:
        spend_tracker.check_and_reserve(
            stage="answer",
            model=answer_model,
            input_tokens=count_tokens(_ANSWER_SYSTEM_PROMPT) + count_tokens(user_prompt),
            output_tokens=256,
        )

    response = await llm_query(
        user_prompt,
        system=_ANSWER_SYSTEM_PROMPT,
        max_tokens=256,
        model=answer_model,
        stage="longmemeval_answer",
        temperature=0.0,
    )
    ans = (response or "").strip()
    if cache is not None:
        cache.set("longmemeval_answer", cache_payload, ans)
    return ans


async def _llm_judge(
    question: str,
    ground_truth: str,
    hypothesis: str,
    question_type: str,
    is_abstention: bool,
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    cache: DiskCache | None = None,
    spend_tracker: SpendTracker | None = None,
    max_retries: int = 3,
    retry_base_delay_s: float = 1.0,
) -> bool:
    """Call the pinned LLM judge (``gpt-4o-2024-08-06``, ``temperature=0``).

    Zero silent fallback (B6): retries transient errors with exponential backoff
    and raises :class:`JudgeEvaluationError` if the judge still fails.
    """
    import archivist.core.config as cfg
    from archivist.features.llm import llm_query
    from archivist.utils.tokenizer import count_tokens

    prompt = _get_judge_prompt(
        task=question_type,
        question=question,
        answer=ground_truth,
        response=hypothesis,
        abstention=is_abstention,
    )

    judge_url = (cfg.BENCHMARK_JUDGE_LLM_URL or "").strip()
    if judge_url:
        j_model = cfg.BENCHMARK_JUDGE_LLM_MODEL or judge_model or DEFAULT_JUDGE_MODEL
        j_key = cfg.BENCHMARK_JUDGE_LLM_API_KEY
    else:
        j_model = judge_model or cfg.BENCHMARK_JUDGE_LLM_MODEL or DEFAULT_JUDGE_MODEL
        j_key = None
        judge_url = ""

    cache_payload = {
        "model": j_model,
        "prompt": prompt,
        "temperature": 0.0,
    }
    if cache is not None:
        found, cached = cache.get("longmemeval_judge", cache_payload)
        if found:
            return bool(cached)

    last_err: Exception | None = None
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            if spend_tracker is not None:
                spend_tracker.check_and_reserve(
                    stage="judge",
                    model=j_model,
                    input_tokens=count_tokens(prompt),
                    output_tokens=10,
                )
            response = await llm_query(
                prompt,
                max_tokens=10,
                model=j_model,
                url=judge_url,
                api_key=j_key,
                stage="longmemeval_judge",
                temperature=0.0,
            )
            verdict = "yes" in (response or "").strip().lower()
            if cache is not None:
                cache.set("longmemeval_judge", cache_payload, verdict)
            return verdict
        except BudgetExceededError:
            # Never retry or wrap hard spend-cap aborts into JudgeEvaluationError.
            raise
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                delay = retry_base_delay_s * (2 ** (attempt - 1))
                logger.warning(
                    "LLM judge attempt %d/%d failed (%s) — retrying in %.1fs",
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

    raise JudgeEvaluationError(
        f"LLM judge failed after {max_retries} attempts (no silent fallback permitted): {last_err}"
    ) from last_err


# ── Per-question state isolation helpers (B2) ────────────────────────────────


async def _reset_question_storage(sqlite_path: str, mem_root: str) -> None:
    """Re-bind SQLite pool, config/indexer paths, and clear hot_cache for a fresh question (B2)."""
    import archivist.core.config as config
    import archivist.retrieval.hot_cache as hot_cache
    import archivist.storage.graph as graph_mod
    import archivist.storage.sqlite_pool as pool_mod
    import archivist.write.indexer as indexer_mod
    from archivist.storage.sqlite_pool import SQLiteGraphBackend

    # Clear hot cache completely so no cached answers leak across questions.
    hot_cache.invalidate_all()

    # Close any existing pool bound to a previous question's DB.
    old_pool = getattr(pool_mod, "pool", None)
    if old_pool is not None and hasattr(old_pool, "close"):
        try:
            await old_pool.close()
        except Exception as exc:
            logger.debug("Ignoring error closing prior question pool: %s", exc)

    config.MEMORY_ROOT = mem_root
    config.SQLITE_PATH = sqlite_path
    os.environ["MEMORY_ROOT"] = mem_root
    os.environ["SQLITE_PATH"] = sqlite_path
    graph_mod.SQLITE_PATH = sqlite_path
    indexer_mod.MEMORY_ROOT = mem_root

    fresh_backend = SQLiteGraphBackend()
    await fresh_backend.initialize(sqlite_path, wal_autocheckpoint=config.SQLITE_WAL_AUTOCHECKPOINT)
    pool_mod.pool = fresh_backend
    await graph_mod.init_schema_async()


# ── Main benchmark runner ────────────────────────────────────────────────────


async def run_longmemeval_benchmark(
    data_file: str,
    limit: int = 0,
    refine: bool = True,
    run_curator: bool = False,
    search_limit: int = 10,
    variant: str = "",
    *,
    profile: str = "default",
    mode: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    answer_model: str = "",
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    checkpoint_path: str | Path | None = None,
    max_budget_usd: float | None = None,
    question_ids: set[str] | None = None,
    # Injectable hooks for deterministic offline unit tests (zero network calls)
    index_fn: Callable[[str, str, list[tuple[str, str]]], Awaitable[int]] | None = None,
    retrieve_fn: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    answer_fn: Callable[..., Awaitable[str]] | None = None,
    judge_fn: Callable[..., Awaitable[bool]] | None = None,
    cache: DiskCache | None = None,
    spend_tracker: SpendTracker | None = None,
) -> dict[str, Any]:
    """Run LongMemEval against Archivist with full B1–B7 correctness guarantees.

    Modes:
      - ``retrieval-only`` (also selected when ``refine=False`` and ``mode`` is omitted):
        Runs retrieval + context packing and computes ``Recall@k``/``NDCG@k`` ONLY.
        Never calls the Answer LLM or ``_llm_judge`` (B1).
      - ``full`` (default when ``refine=True``):
        Runs retrieval + context packing, generates a hypothesis with the pinned
        Answer LLM, and scores it with the pinned ``gpt-4o-2024-08-06`` judge.
    """
    effective_mode = (mode or ("full" if refine else "retrieval-only")).strip().lower()
    if effective_mode not in ("retrieval-only", "full"):
        return {"error": f"Unknown mode: {effective_mode}", "valid": ["retrieval-only", "full"]}

    if variant:
        if variant not in VARIANTS:
            return {"error": f"Unknown variant: {variant}", "valid": list(VARIANTS.keys())}
        _apply_variant(variant)

    items = _load_data(data_file)
    if not items:
        return {"error": "No items loaded", "data_file": data_file}

    if question_ids is not None:
        items = [it for it in items if str(it.get("question_id", "")) in question_ids]
    if limit > 0:
        items = items[:limit]

    active_cache = cache if cache is not None else (DiskCache(cache_dir) if cache_dir else None)
    active_spend = (
        spend_tracker if spend_tracker is not None else SpendTracker(max_budget_usd=max_budget_usd)
    )
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    completed_by_qid = _load_checkpoint_jsonl(ckpt_path)

    work_dir = tempfile.mkdtemp(prefix="longmemeval_bench_")

    import archivist.core.config as config
    import archivist.storage.graph as graph_mod
    import archivist.storage.sqlite_pool as pool_mod
    import archivist.write.indexer as indexer_mod

    original_memory_root = config.MEMORY_ROOT
    original_indexer_memory_root = indexer_mod.MEMORY_ROOT
    original_sqlite = config.SQLITE_PATH
    original_graph_sqlite = graph_mod.SQLITE_PATH
    original_pool = pool_mod.pool

    try:
        from archivist.retrieval.context_api import get_relevant_context
        from archivist.utils.tokenizer import count_tokens

        qclient = None
        if index_fn is None:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            qclient = QdrantClient(url=config.QDRANT_URL, timeout=30)

        all_results: list[dict[str, Any]] = []
        results_by_qtype: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        total_ingest_llm_calls = 0

        for qi, item in enumerate(items):
            qid = str(item.get("question_id", qi))
            question = str(item.get("question", ""))
            answer_gt = str(item.get("answer", ""))
            question_type = str(item.get("question_type", ""))
            question_date = str(item.get("question_date") or "").strip()
            is_abstention = qid.endswith("_abs")
            sessions = item.get("haystack_sessions", [])
            dates = item.get("haystack_dates", [])
            raw_session_ids = item.get("haystack_session_ids") or []
            session_ids = (
                [str(s) for s in raw_session_ids]
                if len(raw_session_ids) == len(sessions)
                else [f"session-{si + 1:04d}" for si in range(len(sessions))]
            )

            if not question or not sessions:
                logger.warning("Skipping item %s: no question or sessions", qid)
                continue

            if qid in completed_by_qid:
                cached_entry = completed_by_qid[qid]
                all_results.append(cached_entry)
                results_by_qtype[question_type].append(cached_entry)
                continue

            # B2: Fresh per-question directory, SQLite DB, graph pool, and hot cache.
            safe_qid = re.sub(r"[^A-Za-z0-9._-]", "_", qid)
            q_dir = os.path.join(work_dir, f"q_{qi:04d}_{safe_qid}")
            q_mem_root = os.path.join(q_dir, "memories")
            q_sqlite_path = os.path.join(q_dir, "graph.db")
            os.makedirs(q_mem_root, exist_ok=True)

            try:
                await _reset_question_storage(q_sqlite_path, q_mem_root)

                md_files, fname_to_sid = _sessions_to_markdown(
                    sessions, dates, qid, session_ids=session_ids
                )
                for filename, content in md_files:
                    filepath = os.path.join(q_mem_root, filename)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)

                # Index into fresh per-question collection / store.
                t_ingest = time.monotonic()
                q_ingest_llm_calls = 0
                if index_fn is not None:
                    chunk_count = await index_fn(q_mem_root, q_sqlite_path, md_files)
                else:
                    assert qclient is not None
                    try:
                        qclient.delete_collection(config.QDRANT_COLLECTION)
                    except Exception as exc:
                        logger.debug(
                            "Collection %s delete before recreate skipped: %s",
                            config.QDRANT_COLLECTION,
                            exc,
                        )
                    qclient.create_collection(
                        collection_name=config.QDRANT_COLLECTION,
                        vectors_config=VectorParams(
                            size=config.VECTOR_DIM,
                            distance=Distance.COSINE,
                        ),
                    )
                    from archivist.write.indexer import full_index

                    chunk_count = await full_index(hierarchical=True)

                if run_curator:
                    q_ingest_llm_calls += await _run_curator_on_files(q_mem_root)
                ingest_time_ms = round((time.monotonic() - t_ingest) * 1000, 1)
                total_ingest_llm_calls += q_ingest_llm_calls

                # B4 + B7: Exercise production get_relevant_context(..., max_tokens=..., reference_time=...)
                t_ret = time.monotonic()
                if retrieve_fn is not None:
                    ret_out = await retrieve_fn(
                        query=question,
                        max_tokens=max_tokens,
                        limit=search_limit,
                        reference_time=question_date,
                        as_of=question_date,
                        sqlite_path=q_sqlite_path,
                    )
                    sources = list(ret_out.get("sources") or [])
                    context_tokens = int(
                        ret_out.get("total_tokens")
                        or sum(
                            count_tokens(s.get("tier_text") or s.get("text", "")) for s in sources
                        )
                    )
                else:
                    ctx = await get_relevant_context(
                        agent_id="longmemeval",
                        task_description=question,
                        max_tokens=max_tokens,
                        namespace="",
                        include_tips=False,
                        reference_time=question_date,
                        as_of=question_date,
                        limit=search_limit,
                    )
                    sources = [_normalize_source(s) for s in ctx.sources]
                    context_tokens = int(ctx.total_tokens)
                elapsed_ms = round((time.monotonic() - t_ret) * 1000, 1)

                # B3: Map retrieved sources to string haystack_session_ids.
                evidence_ids = {str(e) for e in item.get("answer_session_ids", [])}
                retrieved_session_ids = map_sources_to_session_ids(
                    sources, session_ids, fname_to_sid
                )

                retrieval_metrics: dict[str, float] = {}
                if evidence_ids and not is_abstention:
                    for k in (5, 10):
                        retrieval_metrics[f"recall@{k}"] = round(
                            _recall_at_k(retrieved_session_ids, evidence_ids, k), 4
                        )
                        retrieval_metrics[f"ndcg@{k}"] = round(
                            _ndcg_at_k(retrieved_session_ids, evidence_ids, k), 4
                        )

                # B1 + B6: Retrieval-only NEVER calls Answer LLM or _llm_judge.
                hypothesis = ""
                correct: bool | None = None
                if effective_mode == "full":
                    context_text = _format_packed_context(sources)
                    if answer_fn is not None:
                        hypothesis = await answer_fn(
                            question=question,
                            context_text=context_text,
                            question_date=question_date,
                            cache=active_cache,
                            spend_tracker=active_spend,
                        )
                    else:
                        hypothesis = await _generate_answer(
                            question=question,
                            context_text=context_text,
                            question_date=question_date,
                            model=answer_model,
                            cache=active_cache,
                            spend_tracker=active_spend,
                        )

                    if judge_fn is not None:
                        correct = await judge_fn(
                            question=question,
                            ground_truth=answer_gt,
                            hypothesis=hypothesis,
                            question_type=question_type,
                            is_abstention=is_abstention,
                            judge_model=judge_model,
                            cache=active_cache,
                            spend_tracker=active_spend,
                        )
                    else:
                        correct = await _llm_judge(
                            question=question,
                            ground_truth=answer_gt,
                            hypothesis=hypothesis,
                            question_type=question_type,
                            is_abstention=is_abstention,
                            judge_model=judge_model,
                            cache=active_cache,
                            spend_tracker=active_spend,
                        )

                entry: dict[str, Any] = {
                    "question_id": qid,
                    "question_type": question_type,
                    "question_date": question_date,
                    "is_abstention": is_abstention,
                    "query": question[:200],
                    "ground_truth": answer_gt[:200],
                    "hypothesis": hypothesis[:500],
                    "correct": correct,
                    "retrieval": retrieval_metrics,
                    "retrieved_session_ids": retrieved_session_ids[:10],
                    "evidence_session_ids": sorted(evidence_ids),
                    "latency_ms": elapsed_ms,
                    "context_tokens": context_tokens,
                    "ingest_time_ms": ingest_time_ms,
                    "ingest_llm_calls": q_ingest_llm_calls,
                    "sources_count": len(sources),
                    "chunks_indexed": chunk_count,
                    "sessions_count": len(sessions),
                }
                all_results.append(entry)
                results_by_qtype[question_type].append(entry)
                _append_checkpoint_jsonl(ckpt_path, entry)

                if (qi + 1) % 5 == 0 or qi == 0:
                    if effective_mode == "full":
                        scored = [
                            float(r["correct"]) for r in all_results if r["correct"] is not None
                        ]
                        logger.info(
                            "[%d/%d] q=%s acc=%.1f%% lat=%.1fms tok=%d",
                            qi + 1,
                            len(items),
                            qid,
                            _mean(scored) * 100,
                            elapsed_ms,
                            context_tokens,
                        )
                    else:
                        r5_vals = [
                            r["retrieval"]["recall@5"]
                            for r in all_results
                            if "recall@5" in r.get("retrieval", {})
                        ]
                        logger.info(
                            "[%d/%d] q=%s R@5=%.4f lat=%.1fms tok=%d",
                            qi + 1,
                            len(items),
                            qid,
                            _mean(r5_vals),
                            elapsed_ms,
                            context_tokens,
                        )
            finally:
                shutil.rmtree(q_dir, ignore_errors=True)

        # ── Build official-format summary + 95% bootstrap CIs (B6, B7) ────
        non_abs = [r for r in all_results if not r["is_abstention"]]
        abs_only = [r for r in all_results if r["is_abstention"]]

        by_qtype: dict[str, dict[str, Any]] = {}
        task_accs: list[float] = []
        for qt, qt_results in results_by_qtype.items():
            qt_entry: dict[str, Any] = {"count": len(qt_results)}
            if effective_mode == "full":
                qt_scores = [float(r["correct"]) for r in qt_results if r["correct"] is not None]
                acc = _mean(qt_scores)
                task_accs.append(acc)
                qt_entry["accuracy"] = round(acc, 4)
                if len(qt_scores) >= MIN_N_FOR_CI:
                    maybe_ci(qt_scores, "accuracy", qt_entry)
            else:
                qt_entry["accuracy"] = None
            for rk in ("recall@5", "ndcg@5", "recall@10", "ndcg@10"):
                vals = [r["retrieval"][rk] for r in qt_results if rk in r.get("retrieval", {})]
                if vals:
                    qt_entry[rk] = round(_mean(vals), 4)
            by_qtype[qt] = qt_entry

        overall_acc: float | None = None
        overall_acc_ci95: list[float] | None = None
        task_avg_acc: float | None = None
        abs_acc: float | None = None
        if effective_mode == "full":
            all_scores = [float(r["correct"]) for r in all_results if r["correct"] is not None]
            acc_bucket: dict[str, Any] = {}
            maybe_ci(all_scores, "overall_accuracy", acc_bucket)
            overall_acc = acc_bucket["overall_accuracy"]
            overall_acc_ci95 = acc_bucket.get("overall_accuracy_ci95")
            task_avg_acc = round(_mean(task_accs), 4) if task_accs else 0.0
            abs_scores = [float(r["correct"]) for r in abs_only if r["correct"] is not None]
            abs_acc = round(_mean(abs_scores), 4) if abs_scores else None

        retrieval_summary: dict[str, Any] = {}
        for rk in ("recall@5", "ndcg@5", "recall@10", "ndcg@10"):
            vals = [r["retrieval"][rk] for r in non_abs if rk in r.get("retrieval", {})]
            if vals:
                maybe_ci(vals, rk, retrieval_summary)

        latencies = [float(r["latency_ms"]) for r in all_results]
        ctx_tokens = [float(r["context_tokens"]) for r in all_results]
        ingest_times = [float(r["ingest_time_ms"]) for r in all_results]

        effective_judge = (
            getattr(config, "BENCHMARK_JUDGE_LLM_MODEL", "") or judge_model or DEFAULT_JUDGE_MODEL
        )
        effective_answer = answer_model or config.LLM_MODEL or DEFAULT_ANSWER_MODEL

        summary: dict[str, Any] = {
            "benchmark": "LongMemEval",
            "profile": variant or profile,
            "variant": variant or profile,
            "mode": effective_mode,
            "eval_protocol": (
                "retrieval-only (Recall@k, NDCG@k — no LLM judge)"
                if effective_mode == "retrieval-only"
                else "llm-as-judge (official, per-task prompts, temperature=0)"
            ),
            "judge_model": effective_judge if effective_mode == "full" else None,
            "answer_model": effective_answer if effective_mode == "full" else None,
            "data_file": os.path.basename(data_file),
            "total_questions": len(items),
            "evaluated_questions": len(all_results),
            "overall_accuracy": overall_acc,
            "overall_accuracy_ci95": overall_acc_ci95,
            "task_averaged_accuracy": task_avg_acc,
            "abstention_accuracy": abs_acc,
            "retrieval": retrieval_summary,
            "by_question_type": by_qtype,
            "max_tokens": max_tokens,
            "search_limit": search_limit,
            "refine": effective_mode == "full",
            "curator": run_curator,
            "latency_ms": {
                "p50": round(percentile(latencies, 50), 1),
                "p95": round(percentile(latencies, 95), 1),
            },
            "context_tokens": {
                "median": round(percentile(ctx_tokens, 50), 1),
                "p95": round(percentile(ctx_tokens, 95), 1),
            },
            "ingest_time_ms": {
                "median": round(percentile(ingest_times, 50), 1),
                "total": round(sum(ingest_times), 1),
            },
            "ingest_llm_calls": total_ingest_llm_calls,
            "judge_fallbacks": 0,
            "estimated_usd": round(active_spend.spent_usd, 6),
            "spend": active_spend.summary(),
            "cache": active_cache.stats() if active_cache is not None else None,
        }

        manifest = build_run_manifest(
            benchmark="LongMemEval_S",
            profile=variant or profile,
            data_file=os.path.basename(data_file),
            evaluated_questions=len(all_results),
            total_questions=len(items),
            judge_model=effective_judge if effective_mode == "full" else "none",
            judge_prompt_hash=_judge_prompt_template_hash(),
            answer_model=effective_answer if effective_mode == "full" else "none",
            embed_model=config.EMBED_MODEL,
            max_tokens=max_tokens,
            judge_fallbacks=0,
            estimated_usd=active_spend.spent_usd,
            metrics={
                "overall_accuracy": overall_acc,
                "overall_accuracy_ci95": overall_acc_ci95,
                "task_averaged_accuracy": task_avg_acc,
                "abstention_accuracy": abs_acc,
                "retrieval": retrieval_summary,
                "latency_ms": summary["latency_ms"],
                "context_tokens": summary["context_tokens"],
            },
        )

        return {"manifest": manifest, "summary": summary, "results": all_results}

    finally:
        # Close last per-question pool and restore original global state.
        cur_pool = getattr(pool_mod, "pool", None)
        if cur_pool is not None and cur_pool is not original_pool and hasattr(cur_pool, "close"):
            try:
                await cur_pool.close()
            except Exception as exc:
                logger.debug("Per-question pool close failed during teardown: %s", exc)
        pool_mod.pool = original_pool
        config.MEMORY_ROOT = original_memory_root
        indexer_mod.MEMORY_ROOT = original_indexer_memory_root
        config.SQLITE_PATH = original_sqlite
        graph_mod.SQLITE_PATH = original_graph_sqlite
        os.environ["MEMORY_ROOT"] = original_memory_root
        os.environ["SQLITE_PATH"] = original_sqlite
        shutil.rmtree(work_dir, ignore_errors=True)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_summary(s: dict[str, Any]) -> None:
    """Print a single variant's summary in a format comparable to published results."""
    print(f"\n{'=' * 72}")
    print(
        f"  LongMemEval — {s.get('profile', 'default')} "
        f"[{s.get('mode', 'full')}] (judge: {s.get('judge_model') or 'none'})"
    )
    print(f"{'=' * 72}")
    print(f"  Dataset:    {s['data_file']}")
    print(f"  Questions:  {s['evaluated_questions']}/{s['total_questions']}")
    print(f"  Protocol:   {s.get('eval_protocol', '')}")
    print(f"  Max Tokens: {s.get('max_tokens', DEFAULT_MAX_TOKENS)}")
    print()
    if s.get("overall_accuracy") is not None:
        ci = s.get("overall_accuracy_ci95")
        ci_str = f" (95% CI [{ci[0]:.1%}, {ci[1]:.1%}])" if ci else ""
        print(f"  Overall Accuracy:        {s['overall_accuracy']:>6.1%}{ci_str}")
        if s.get("task_averaged_accuracy") is not None:
            print(f"  Task-Averaged Accuracy:  {s['task_averaged_accuracy']:>6.1%}")
        if s.get("abstention_accuracy") is not None:
            print(f"  Abstention Accuracy:     {s['abstention_accuracy']:>6.1%}")
        print()

    retr = s.get("retrieval", {})
    if retr:
        print(
            f"  Retrieval:  Recall@5={retr.get('recall@5', 0):.4f}  "
            f"NDCG@5={retr.get('ndcg@5', 0):.4f}  "
            f"Recall@10={retr.get('recall@10', 0):.4f}  "
            f"NDCG@10={retr.get('ndcg@10', 0):.4f}"
        )
    lat = s.get("latency_ms", {})
    tok = s.get("context_tokens", {})
    print(
        f"  Latency:    p50={lat.get('p50', 0):.1f}ms  p95={lat.get('p95', 0):.1f}ms  |  "
        f"Context Tokens: median={tok.get('median', 0):.0f}  p95={tok.get('p95', 0):.0f}"
    )
    print(
        f"  Cost/Guard: estimated_usd=${s.get('estimated_usd', 0):.4f}  "
        f"ingest_llm_calls={s.get('ingest_llm_calls', 0)}  "
        f"judge_fallbacks={s.get('judge_fallbacks', 0)}"
    )
    print()

    print(f"  {'Question Type':<28s}  {'Acc':>6s}  {'R@5':>6s}  {'NDCG@5':>6s}  {'n':>4s}")
    print(f"  {'-' * 28}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 4}")
    for qt, vals in s.get("by_question_type", {}).items():
        acc = f"{vals['accuracy']:.1%}" if vals.get("accuracy") is not None else "   —"
        r5 = f"{vals.get('recall@5', 0):.4f}" if vals.get("recall@5") is not None else "  —"
        n5 = f"{vals.get('ndcg@5', 0):.4f}" if vals.get("ndcg@5") is not None else "  —"
        print(f"  {qt:<28s}  {acc:>6s}  {r5:>6s}  {n5:>6s}  {vals['count']:>4d}")

    print(f"{'=' * 72}")


def _print_ablation_comparison(ablation_data: dict[str, Any]) -> None:
    """Print a side-by-side comparison table across variants."""
    variants = ablation_data.get("variants", {})
    if not variants:
        return

    print(f"\n{'=' * 80}")
    print("  LongMemEval Ablation")
    print(f"{'=' * 80}")
    print(f"  {'Variant':<25s}  {'Accuracy':>10s}  {'Task-Avg':>10s}  {'R@5':>8s}  {'NDCG@5':>8s}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 8}")

    for vname, vdata in variants.items():
        s = vdata.get("summary", {})
        acc_v = s.get("overall_accuracy")
        tavg_v = s.get("task_averaged_accuracy")
        acc = f"{acc_v:>9.1%}" if isinstance(acc_v, (int, float)) else f"{'—':>10s}"
        tavg = f"{tavg_v:>9.1%}" if isinstance(tavg_v, (int, float)) else f"{'—':>10s}"
        r5 = s.get("retrieval", {}).get("recall@5", 0)
        n5 = s.get("retrieval", {}).get("ndcg@5", 0)
        print(f"  {vname:<25s}  {acc}  {tavg}  {r5:>8.4f}  {n5:>8.4f}")

    print(f"{'=' * 80}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark adapter for Archivist (official eval protocol)",
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to longmemeval_s_cleaned.json or longmemeval_m_cleaned.json",
    )
    parser.add_argument(
        "--profile",
        default="default",
        choices=["default"],
        help="Production settings profile (default: unmodified ArchivistSettings)",
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval-only", "full"],
        default=None,
        help="Evaluation mode (default: retrieval-only if --no-refine else full)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Retrieval-only mode: compute Recall@k/NDCG@k without Answer LLM or Judge",
    )
    parser.add_argument("--run-curator", action="store_true", help="Run curator KG extraction")
    parser.add_argument("--search-limit", type=int, default=10, help="Top-k retrieval limit")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Context packing token budget (default: 4000; supports 2000/4000/8000)",
    )
    parser.add_argument(
        "--max-tokens-sweep",
        type=str,
        default="",
        help="Optional comma-separated token budgets to sweep (e.g. 2000,4000,8000)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Pinned LLM judge model (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--answer-model",
        type=str,
        default="",
        help="Optional Answer LLM model override (default: config.LLM_MODEL)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="SHA-256 disk cache directory (empty string disables disk cache)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional atomic JSONL checkpoint path for resumable runs",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=None,
        help="Optional hard USD spend ceiling (omit for unlimited; 0.0 rejects paid calls)",
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        help="Run a single named pipeline variant instead of --profile default",
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
    effective_mode = args.mode or ("retrieval-only" if args.no_refine else "full")
    budgets = (
        [int(x.strip()) for x in args.max_tokens_sweep.split(",") if x.strip()]
        if args.max_tokens_sweep
        else [args.max_tokens]
    )

    if args.ablation:
        variant_names = (
            [v.strip() for v in args.variants.split(",") if v.strip()]
            if args.variants
            else list(VARIANTS.keys())
        )
        ablation_results: dict[str, Any] = {
            "benchmark": "LongMemEval",
            "mode": effective_mode,
            "variants": {},
        }
        for vname in variant_names:
            logger.info("=== Running variant: %s ===", vname)
            ckpt = args.checkpoint or str(Path(args.output).with_suffix(f".{vname}.results.jsonl"))
            data = await run_longmemeval_benchmark(
                args.data_file,
                limit=args.limit,
                refine=effective_mode == "full",
                run_curator=args.run_curator,
                search_limit=args.search_limit,
                variant=vname,
                profile=args.profile,
                mode=effective_mode,
                max_tokens=budgets[0],
                judge_model=args.judge_model,
                answer_model=args.answer_model,
                cache_dir=args.cache_dir or None,
                checkpoint_path=ckpt,
                max_budget_usd=args.max_budget_usd,
            )
            ablation_results["variants"][vname] = data
            if "summary" in data:
                _print_summary(data["summary"])

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, indent=2, ensure_ascii=False)

        _print_ablation_comparison(ablation_results)
        logger.info("Ablation results written to %s", args.output)
    elif len(budgets) > 1:
        sweep_results: dict[str, Any] = {
            "benchmark": "LongMemEval",
            "mode": effective_mode,
            "profile": args.variant or args.profile,
            "budgets": {},
        }
        for tok_budget in budgets:
            ckpt = str(Path(args.output).with_suffix(f".tok{tok_budget}.results.jsonl"))
            data = await run_longmemeval_benchmark(
                args.data_file,
                limit=args.limit,
                refine=effective_mode == "full",
                run_curator=args.run_curator,
                search_limit=args.search_limit,
                variant=args.variant or "",
                profile=args.profile,
                mode=effective_mode,
                max_tokens=tok_budget,
                judge_model=args.judge_model,
                answer_model=args.answer_model,
                cache_dir=args.cache_dir or None,
                checkpoint_path=ckpt,
                max_budget_usd=args.max_budget_usd,
            )
            sweep_results["budgets"][str(tok_budget)] = data
            if "summary" in data:
                _print_summary(data["summary"])

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2, ensure_ascii=False)
        logger.info("Budget sweep results written to %s", args.output)
    else:
        ckpt = args.checkpoint or str(Path(args.output).with_suffix(".results.jsonl"))
        data = await run_longmemeval_benchmark(
            args.data_file,
            limit=args.limit,
            refine=effective_mode == "full",
            run_curator=args.run_curator,
            search_limit=args.search_limit,
            variant=args.variant or "",
            profile=args.profile,
            mode=effective_mode,
            max_tokens=budgets[0],
            judge_model=args.judge_model,
            answer_model=args.answer_model,
            cache_dir=args.cache_dir or None,
            checkpoint_path=ckpt,
            max_budget_usd=args.max_budget_usd,
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
