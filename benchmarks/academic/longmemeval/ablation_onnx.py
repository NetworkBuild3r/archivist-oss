"""Week-1 <=$5.00 Empirical 100-Q ONNX vs Cloud Embedding Ablation (INIT-023/SPEC-004).

Evaluates local ONNX INT8 embedder + reranker pairs against the cloud reference
baseline (``text-embedding-3-small``) on a deterministic stratified 100-question
slice of ``LongMemEval_S`` (``benchmarks/splits/longmemeval_s_dev100_ids.json``).

Candidates:
  - Local ONNX INT8 embedders:
      1. ``BAAI/bge-small-en-v1.5`` (384d)
      2. ``nomic-embed-text-v1.5`` (512d Matryoshka)
      3. ``snowflake-arctic-embed-m-v2.0`` (768d)
  - Local ONNX INT8 rerankers:
      1. ``ms-marco-MiniLM-L-6-v2`` (~23MB)
      2. ``bge-reranker-base-int8``
  - Cloud reference baseline:
      - ``text-embedding-3-small`` (1536d)

Modes:
  - ``dry-run``: deterministic offline execution (zero external keys, zero network),
    verifies cache hits, spend-cap gating, metrics, 95% CIs, and manifest schema.
  - ``local-onnx``: runs real local ONNX INT8 retrieval & CPU timing/RAM checks
    without requiring cloud API keys (requires ``fastembed`` / ``onnxruntime``).
  - ``live``: full ONNX vs cloud ``text-embedding-3-small`` + pinned
    ``gpt-4o-2024-08-06`` judge within the strict ``<= $5.00`` USD spend cap.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.academic.longmemeval.adapter import (
    DEFAULT_ANSWER_MODEL,
    DEFAULT_JUDGE_MODEL,
    _generate_answer,
    _judge_prompt_template_hash,
    _llm_judge,
    _load_data,
    _ndcg_at_k,
    _recall_at_k,
)
from benchmarks.cache import BudgetExceededError, DiskCache, SpendTracker
from benchmarks.env_loader import load_repo_env
from benchmarks.stats import build_run_manifest, maybe_ci, percentile

load_repo_env()
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

DEFAULT_SPLIT_PATH = Path(_REPO_ROOT) / "benchmarks" / "splits" / "longmemeval_s_dev100_ids.json"
DEFAULT_MAX_BUDGET_USD = 5.0
HARD_WAVE1_BUDGET_CAP_USD = 5.0


@dataclass(frozen=True)
class EmbedderSpec:
    id: str
    dim: int
    backend: str  # "onnx" | "cloud" | "dry_run"
    fastembed_name: str = ""
    approx_download_mb: float = 0.0
    matryoshka_dim: int | None = None


@dataclass(frozen=True)
class RerankerSpec:
    id: str
    backend: str  # "onnx" | "none" | "dry_run"
    fastembed_name: str = ""
    approx_download_mb: float = 0.0


LOCAL_ONNX_EMBEDDERS: tuple[EmbedderSpec, ...] = (
    EmbedderSpec(
        id="BAAI/bge-small-en-v1.5",
        dim=384,
        backend="onnx",
        fastembed_name="BAAI/bge-small-en-v1.5",
        approx_download_mb=33.0,
    ),
    EmbedderSpec(
        id="nomic-embed-text-v1.5",
        dim=512,
        backend="onnx",
        fastembed_name="nomic-ai/nomic-embed-text-v1.5-Q",
        approx_download_mb=137.0,
        matryoshka_dim=512,
    ),
    EmbedderSpec(
        id="snowflake-arctic-embed-m-v2.0",
        dim=768,
        backend="onnx",
        fastembed_name="snowflake/snowflake-arctic-embed-m",
        approx_download_mb=110.0,
    ),
)

CLOUD_BASELINE_EMBEDDER = EmbedderSpec(
    id="text-embedding-3-small",
    dim=1536,
    backend="cloud",
    approx_download_mb=0.0,
)

LOCAL_ONNX_RERANKERS: tuple[RerankerSpec, ...] = (
    RerankerSpec(
        id="ms-marco-MiniLM-L-6-v2",
        backend="onnx",
        fastembed_name="Xenova/ms-marco-MiniLM-L-6-v2",
        approx_download_mb=23.0,
    ),
    RerankerSpec(
        id="bge-reranker-base-int8",
        backend="onnx",
        fastembed_name="BAAI/bge-reranker-base",
        approx_download_mb=278.0,
    ),
)

NO_RERANKER = RerankerSpec(id="none", backend="none", approx_download_mb=0.0)


# ── Deterministic stratified 100-Q split generator ───────────────────────────


def select_stratified_dev100(
    items: Sequence[dict[str, Any]],
    *,
    target_n: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Deterministically select *target_n* stratified question IDs across all 6 types."""
    if not items:
        return {
            "schema_version": "1.0",
            "dataset": "longmemeval_s_cleaned.json",
            "seed": seed,
            "target_n": target_n,
            "total_selected": 0,
            "strata_counts": {},
            "abstention_count": 0,
            "question_ids": [],
        }

    rng = random.Random(seed)
    by_stratum: dict[tuple[str, bool], list[str]] = collections.defaultdict(list)
    for item in items:
        qid = str(item.get("question_id", ""))
        qtype = str(item.get("question_type", "unknown"))
        is_abs = qid.endswith("_abs")
        if qid:
            by_stratum[(qtype, is_abs)].append(qid)

    total = sum(len(v) for v in by_stratum.values())
    n_pick = min(target_n, total)

    # Largest-remainder proportional allocation across (question_type, is_abstention) strata.
    exact_quotas: dict[tuple[str, bool], float] = {
        k: (len(v) / total) * n_pick for k, v in by_stratum.items()
    }
    alloc: dict[tuple[str, bool], int] = {k: math.floor(q) for k, q in exact_quotas.items()}
    # Guarantee at least 1 sample from every non-empty stratum when possible.
    for k in sorted(by_stratum.keys()):
        if alloc[k] == 0 and len(by_stratum[k]) > 0 and sum(alloc.values()) < n_pick:
            alloc[k] = 1

    remainder = n_pick - sum(alloc.values())
    if remainder > 0:
        ranked = sorted(
            by_stratum.keys(),
            key=lambda k: (exact_quotas[k] - alloc[k], k[0], k[1]),
            reverse=True,
        )
        for k in ranked:
            if remainder <= 0:
                break
            if alloc[k] < len(by_stratum[k]):
                alloc[k] += 1
                remainder -= 1
    elif remainder < 0:
        ranked = sorted(
            [k for k, cnt in alloc.items() if cnt > 1],
            key=lambda k: (exact_quotas[k] - alloc[k], k[0], k[1]),
        )
        for k in ranked:
            if remainder >= 0:
                break
            alloc[k] -= 1
            remainder += 1

    selected_ids: list[str] = []
    strata_counts: dict[str, int] = collections.defaultdict(int)
    abs_count = 0
    for k in sorted(by_stratum.keys()):
        pool = sorted(by_stratum[k])
        rng.shuffle(pool)
        chosen = sorted(pool[: alloc[k]])
        selected_ids.extend(chosen)
        strata_counts[k[0]] += len(chosen)
        if k[1]:
            abs_count += len(chosen)

    return {
        "schema_version": "1.0",
        "dataset": "longmemeval_s_cleaned.json",
        "seed": seed,
        "target_n": target_n,
        "total_selected": len(selected_ids),
        "strata_counts": dict(sorted(strata_counts.items())),
        "abstention_count": abs_count,
        "question_ids": selected_ids,
    }


def load_dev100_ids(split_path: str | Path = DEFAULT_SPLIT_PATH) -> list[str]:
    """Load the pinned 100-Q dev split IDs from JSON."""
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    return [str(qid) for qid in payload.get("question_ids", [])]


# ── Embedding & reranking helpers (dry-run, local ONNX, cloud) ───────────────


def _current_rss_mb() -> float:
    """Return current process max RSS in MB (Linux ru_maxrss is in KB)."""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss_kb / 1024.0, 2)


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _dry_run_embed(texts: Sequence[str], *, model_id: str, dim: int) -> list[list[float]]:
    """Deterministic bag-of-hashed-tokens dense projection for offline dry-run verification."""
    out: list[list[float]] = []
    # Use a compact fixed projection dimension internally for speed, padded/sliced to `dim`.
    eff_dim = min(dim, 128)
    for text in texts:
        vec = [0.0] * eff_dim
        words = [w.strip(".,!?;:\"'()[]{}").lower() for w in (text or "").split()]
        for w in words:
            if len(w) < 2:
                continue
            digest = hashlib.sha256(f"{model_id}:{w}".encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % eff_dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


def _dry_run_rerank(query: str, docs: Sequence[str], *, model_id: str) -> list[float]:
    """Deterministic lexical overlap proxy for offline reranker verification."""
    q_terms = {w.strip(".,!?;:\"'()[]{}").lower() for w in query.split() if len(w) >= 2}
    if not q_terms:
        return [0.0 for _ in docs]
    bias = (int(hashlib.sha256(model_id.encode()).hexdigest()[:4], 16) % 10) * 1e-4
    scores: list[float] = []
    for doc in docs:
        d_terms = {w.strip(".,!?;:\"'()[]{}").lower() for w in doc.split() if len(w) >= 2}
        overlap = len(q_terms & d_terms) / max(len(q_terms), 1)
        scores.append(round(overlap + bias, 6))
    return scores


class _EmbedderRuntime:
    """Thin cached wrapper over dry-run, local ONNX (fastembed), or cloud OpenAI embeddings."""

    def __init__(
        self,
        spec: EmbedderSpec,
        *,
        mode: str,
        cache: DiskCache | None,
        spend_tracker: SpendTracker,
    ) -> None:
        self.spec = spec
        self.mode = mode
        self.cache = cache
        self.spend_tracker = spend_tracker
        self._onnx_model: Any = None
        self.rss_delta_mb: float = 0.0

    def _ensure_onnx(self) -> Any:
        if self._onnx_model is not None:
            return self._onnx_model
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "local-onnx mode requires `fastembed` (`pip install fastembed`). "
                "Use `--mode dry-run` to verify the harness offline."
            ) from exc
        rss_before = _current_rss_mb()
        self._onnx_model = TextEmbedding(model_name=self.spec.fastembed_name)
        self.rss_delta_mb = round(max(0.0, _current_rss_mb() - rss_before), 2)
        return self._onnx_model

    async def embed_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], float]:
        """Embed *texts* using SHA-256 disk cache; return ``(vectors, elapsed_ms)``."""
        t0 = time.monotonic()
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for idx, txt in enumerate(texts):
            if self.cache is not None:
                found, vec = self.cache.get(
                    "ablation_embed",
                    {"mode": self.mode, "model": self.spec.id, "dim": self.spec.dim, "text": txt},
                )
                if found:
                    results[idx] = vec
                    continue
            uncached_indices.append(idx)
            uncached_texts.append(txt)

        if uncached_texts:
            if self.mode == "dry-run":
                computed = _dry_run_embed(uncached_texts, model_id=self.spec.id, dim=self.spec.dim)
            elif self.spec.backend == "onnx":
                model = self._ensure_onnx()
                raw_vecs = await asyncio.to_thread(lambda: list(model.embed(uncached_texts)))
                computed = []
                for vec in raw_vecs:
                    arr = [float(x) for x in vec]
                    if self.spec.matryoshka_dim and len(arr) > self.spec.matryoshka_dim:
                        arr = arr[: self.spec.matryoshka_dim]
                        norm = math.sqrt(sum(v * v for v in arr)) or 1.0
                        arr = [v / norm for v in arr]
                    computed.append(arr)
            else:
                computed = await self._embed_cloud_openai(uncached_texts)

            for idx, txt, vec in zip(uncached_indices, uncached_texts, computed):
                results[idx] = vec
                if self.cache is not None:
                    self.cache.set(
                        "ablation_embed",
                        {
                            "mode": self.mode,
                            "model": self.spec.id,
                            "dim": self.spec.dim,
                            "text": txt,
                        },
                        vec,
                    )

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return [r or [] for r in results], elapsed_ms

    async def _embed_cloud_openai(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx

        api_key = (
            os.getenv("EMBED_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("LLM_API_KEY")
            or ""
        ).strip()
        if not api_key:
            raise RuntimeError(
                "Missing cloud embedding API key (set OPENAI_API_KEY or EMBED_API_KEY)"
            )
        base_url = (
            os.getenv("OPENAI_BASE_URL") or os.getenv("EMBED_URL") or "https://api.openai.com"
        ).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        approx_tokens = sum(max(1, len(t) // 4) for t in texts)
        self.spend_tracker.check_and_reserve(
            stage=f"embed:{self.spec.id}",
            model=self.spec.id,
            input_tokens=approx_tokens,
        )

        vectors: list[list[float]] = []
        batch_size = 64
        async with httpx.AsyncClient(timeout=60.0) as client:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                resp = await client.post(
                    f"{base_url}/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": self.spec.id, "input": batch},
                )
                resp.raise_for_status()
                payload = resp.json()
                data = sorted(payload["data"], key=lambda item: item["index"])
                vectors.extend([list(item["embedding"]) for item in data])
        return vectors


class _RerankerRuntime:
    """Thin wrapper over dry-run or local ONNX cross-encoder rerankers."""

    def __init__(self, spec: RerankerSpec, *, mode: str) -> None:
        self.spec = spec
        self.mode = mode
        self._onnx_reranker: Any = None
        self.rss_delta_mb: float = 0.0

    def _ensure_onnx(self) -> Any:
        if self._onnx_reranker is not None:
            return self._onnx_reranker
        try:
            from fastembed.rerank.cross_encoder import (  # type: ignore[import-not-found]
                TextCrossEncoder,
            )
        except ImportError as exc:
            raise RuntimeError(
                "local-onnx reranker requires `fastembed` (`pip install fastembed`)."
            ) from exc
        rss_before = _current_rss_mb()
        self._onnx_reranker = TextCrossEncoder(model_name=self.spec.fastembed_name)
        self.rss_delta_mb = round(max(0.0, _current_rss_mb() - rss_before), 2)
        return self._onnx_reranker

    async def rerank(self, query: str, docs: Sequence[str]) -> tuple[list[float], float]:
        if self.spec.backend == "none" or not docs:
            return [0.0 for _ in docs], 0.0
        t0 = time.monotonic()
        if self.mode == "dry-run":
            scores = _dry_run_rerank(query, docs, model_id=self.spec.id)
        else:
            model = self._ensure_onnx()
            raw = await asyncio.to_thread(lambda: list(model.rerank(query, list(docs))))
            scores = [float(s) for s in raw]
        return scores, (time.monotonic() - t0) * 1000.0


# ── Decision gate evaluation ─────────────────────────────────────────────────


def evaluate_selection_gate(
    pair_results: Sequence[dict[str, Any]],
    *,
    cloud_baseline_key: str = "text-embedding-3-small + none",
    max_delta_pp: float = 3.0,
    max_combined_p50_ms: float = 40.0,
) -> dict[str, Any]:
    """Select the winning ONNX embedder + reranker meeting SPEC-004 gates.

    Gate criteria:
      - Accuracy delta (or Recall@5 delta when QA accuracy is unavailable in offline mode)
        <= ``max_delta_pp`` percentage points vs ``text-embedding-3-small``.
      - Combined CPU embed + rerank p50 latency < ``max_combined_p50_ms`` ms.
    """
    baseline = next((p for p in pair_results if p.get("pair_key") == cloud_baseline_key), None)
    local_candidates = [p for p in pair_results if p.get("embedder_backend") == "onnx"]
    if not local_candidates:
        return {"status": "no_local_candidates", "winner": None, "passing": []}

    use_qa = baseline is not None and baseline.get("qa_accuracy") is not None
    metric_name = "qa_accuracy" if use_qa else "recall@5"
    baseline_val = float(baseline.get(metric_name) or 0.0) if baseline is not None else None

    passing: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []
    for cand in local_candidates:
        cand_val = float(cand.get(metric_name) or 0.0)
        delta_pp = round((baseline_val - cand_val) * 100.0, 2) if baseline_val is not None else None
        p50_ms = float(cand.get("combined_cpu_p50_ms") or 0.0)
        meets_quality = delta_pp is None or delta_pp <= max_delta_pp
        meets_latency = p50_ms < max_combined_p50_ms
        rec = {
            "pair_key": cand["pair_key"],
            "embedder": cand["embedder"],
            "embedder_dim": cand["embedder_dim"],
            "reranker": cand["reranker"],
            "metric": metric_name,
            "score": round(cand_val, 4),
            "delta_vs_cloud_pp": delta_pp,
            "combined_cpu_p50_ms": round(p50_ms, 2),
            "meets_quality_gate": meets_quality,
            "meets_latency_gate": meets_latency,
            "passes_all": meets_quality and meets_latency,
        }
        evaluated.append(rec)
        if rec["passes_all"]:
            passing.append(rec)

    passing.sort(key=lambda r: (-r["score"], r["combined_cpu_p50_ms"]))
    if baseline is None:
        status = "partial_offline"
    elif passing:
        status = "locked" if use_qa else "provisional_retrieval_only"
    else:
        status = "escalate"

    return {
        "status": status,
        "metric_used": metric_name,
        "baseline_score": round(baseline_val, 4) if baseline_val is not None else None,
        "max_delta_pp": max_delta_pp,
        "max_combined_p50_ms": max_combined_p50_ms,
        "winner": passing[0] if passing else None,
        "passing": passing,
        "evaluated": evaluated,
    }


# ── Core ablation runner ─────────────────────────────────────────────────────


def _session_text(session: list[dict[str, Any]], date_str: str) -> str:
    lines = [f"[Date: {date_str}]"] if date_str else []
    for turn in session:
        role = turn.get("role", "user")
        content = str(turn.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def _evaluate_pair(
    *,
    embedder: EmbedderSpec,
    reranker: RerankerSpec,
    items: Sequence[dict[str, Any]],
    mode: str,
    cache: DiskCache | None,
    spend_tracker: SpendTracker,
    top_k_retrieve: int = 25,
    top_k_final: int = 10,
    run_qa: bool = False,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict[str, Any]:
    embed_rt = _EmbedderRuntime(embedder, mode=mode, cache=cache, spend_tracker=spend_tracker)
    rerank_rt = _RerankerRuntime(reranker, mode=mode)

    query_embed_latencies: list[float] = []
    rerank_latencies: list[float] = []
    combined_latencies: list[float] = []
    r5_vals: list[float] = []
    n5_vals: list[float] = []
    r10_vals: list[float] = []
    n10_vals: list[float] = []
    qa_scores: list[float] = []

    for item in items:
        qid = str(item.get("question_id", ""))
        question = str(item.get("question", ""))
        question_type = str(item.get("question_type", ""))
        question_date = str(item.get("question_date") or "").strip()
        answer_gt = str(item.get("answer", ""))
        is_abs = qid.endswith("_abs")
        sessions = item.get("haystack_sessions", [])
        dates = item.get("haystack_dates", [])
        raw_sids = item.get("haystack_session_ids") or []
        session_ids = (
            [str(s) for s in raw_sids]
            if len(raw_sids) == len(sessions)
            else [f"session-{i + 1:04d}" for i in range(len(sessions))]
        )
        evidence_ids = {str(e) for e in item.get("answer_session_ids", [])}

        if not question or not sessions:
            continue

        docs = [
            _session_text(sess, dates[si] if si < len(dates) else "")
            for si, sess in enumerate(sessions)
        ]

        # Embed haystack sessions (cached by SHA-256 across pairs sharing the same embedder).
        doc_vecs, _ = await embed_rt.embed_texts(docs)

        # Measure per-query embed + optional rerank latency.
        q_vecs, q_embed_ms = await embed_rt.embed_texts([question])
        q_vec = q_vecs[0]

        scored = [(si, _cosine_sim(q_vec, dvec)) for si, dvec in enumerate(doc_vecs)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        candidates = scored[: min(top_k_retrieve, len(scored))]

        if reranker.backend != "none":
            cand_docs = [docs[si] for si, _ in candidates]
            rr_scores, rr_ms = await rerank_rt.rerank(question, cand_docs)
            reranked = sorted(
                zip([si for si, _ in candidates], rr_scores),
                key=lambda pair: pair[1],
                reverse=True,
            )
            ranked_indices = [si for si, _ in reranked[:top_k_final]]
        else:
            rr_ms = 0.0
            ranked_indices = [si for si, _ in candidates[:top_k_final]]

        query_embed_latencies.append(q_embed_ms)
        rerank_latencies.append(rr_ms)
        combined_latencies.append(q_embed_ms + rr_ms)

        retrieved_sids = [session_ids[si] for si in ranked_indices]
        if evidence_ids and not is_abs:
            r5_vals.append(_recall_at_k(retrieved_sids, evidence_ids, 5))
            n5_vals.append(_ndcg_at_k(retrieved_sids, evidence_ids, 5))
            r10_vals.append(_recall_at_k(retrieved_sids, evidence_ids, 10))
            n10_vals.append(_ndcg_at_k(retrieved_sids, evidence_ids, 10))

        if run_qa and mode == "live":
            ctx_text = "\n\n".join(docs[si][:1500] for si in ranked_indices[:5])
            hypothesis = await _generate_answer(
                question=question,
                context_text=ctx_text,
                question_date=question_date,
                model=answer_model,
                cache=cache,
                spend_tracker=spend_tracker,
            )
            verdict = await _llm_judge(
                question=question,
                ground_truth=answer_gt,
                hypothesis=hypothesis,
                question_type=question_type,
                is_abstention=is_abs,
                judge_model=judge_model,
                cache=cache,
                spend_tracker=spend_tracker,
            )
            qa_scores.append(1.0 if verdict else 0.0)

    metrics: dict[str, Any] = {}
    maybe_ci(r5_vals, "recall@5", metrics)
    maybe_ci(n5_vals, "ndcg@5", metrics)
    maybe_ci(r10_vals, "recall@10", metrics)
    maybe_ci(n10_vals, "ndcg@10", metrics)
    if qa_scores:
        maybe_ci(qa_scores, "qa_accuracy", metrics)

    return {
        "pair_key": f"{embedder.id} + {reranker.id}",
        "embedder": embedder.id,
        "embedder_dim": embedder.dim,
        "embedder_backend": embedder.backend,
        "reranker": reranker.id,
        "reranker_backend": reranker.backend,
        "evaluated_questions": len(items),
        "recall@5": metrics.get("recall@5", 0.0),
        "recall@5_ci95": metrics.get("recall@5_ci95"),
        "ndcg@5": metrics.get("ndcg@5", 0.0),
        "ndcg@5_ci95": metrics.get("ndcg@5_ci95"),
        "recall@10": metrics.get("recall@10", 0.0),
        "ndcg@10": metrics.get("ndcg@10", 0.0),
        "qa_accuracy": metrics.get("qa_accuracy"),
        "qa_accuracy_ci95": metrics.get("qa_accuracy_ci95"),
        "embed_cpu_p50_ms": round(percentile(query_embed_latencies, 50), 2),
        "embed_cpu_p95_ms": round(percentile(query_embed_latencies, 95), 2),
        "rerank_cpu_p50_ms": round(percentile(rerank_latencies, 50), 2),
        "rerank_cpu_p95_ms": round(percentile(rerank_latencies, 95), 2),
        "combined_cpu_p50_ms": round(percentile(combined_latencies, 50), 2),
        "combined_cpu_p95_ms": round(percentile(combined_latencies, 95), 2),
        "approx_download_mb": round(embedder.approx_download_mb + reranker.approx_download_mb, 1),
        "rss_delta_mb": round(embed_rt.rss_delta_mb + rerank_rt.rss_delta_mb, 2),
        "judge_fallbacks": 0,
    }


def _has_live_api_credentials() -> bool:
    return bool(
        (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("EMBED_API_KEY") or "").strip()
        or (os.getenv("LLM_API_KEY") or "").strip()
    )


async def run_onnx_ablation(
    *,
    data_file: str | Path,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    mode: str = "dry-run",
    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
    cache_dir: str | Path | None = ".benchmarks/cache/ablation_onnx",
    limit: int = 0,
    embedder_ids: Sequence[str] | None = None,
    reranker_ids: Sequence[str] | None = None,
    run_qa: bool = True,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict[str, Any]:
    """Run the 100-Q ONNX vs cloud embedding ablation within ``max_budget_usd``."""
    if mode not in ("dry-run", "local-onnx", "live"):
        raise ValueError(f"Unsupported mode {mode!r}; expected dry-run|local-onnx|live")
    if max_budget_usd <= 0 or max_budget_usd > HARD_WAVE1_BUDGET_CAP_USD:
        raise ValueError(
            f"Wave-1 ablation max_budget_usd must be in (0, {HARD_WAVE1_BUDGET_CAP_USD:.2f}], "
            f"got {max_budget_usd}"
        )
    if mode == "live" and not _has_live_api_credentials():
        raise RuntimeError(
            "Live mode requires OPENAI_API_KEY (or EMBED_API_KEY + LLM_API_KEY). "
            "No credentials found in environment."
        )

    all_items = _load_data(str(data_file))
    if not all_items:
        raise ValueError(f"No items loaded from {data_file}")

    split_file = Path(split_path)
    if split_file.is_file():
        selected_set = set(load_dev100_ids(split_file))
        items = [it for it in all_items if str(it.get("question_id", "")) in selected_set]
        if not items:
            # Fallback for synthetic unit-test datasets whose IDs are not in dev100 file.
            items = list(all_items)
    else:
        split_data = select_stratified_dev100(all_items, target_n=100, seed=42)
        selected_set = set(split_data["question_ids"])
        items = [it for it in all_items if str(it.get("question_id", "")) in selected_set]

    if limit > 0:
        items = items[:limit]

    cache = DiskCache(cache_dir) if cache_dir else None
    spend_tracker = SpendTracker(max_budget_usd=max_budget_usd)

    embedders: list[EmbedderSpec] = list(LOCAL_ONNX_EMBEDDERS)
    if mode in ("live", "dry-run"):
        embedders.append(CLOUD_BASELINE_EMBEDDER)
    if embedder_ids:
        wanted_e = set(embedder_ids)
        embedders = [e for e in embedders if e.id in wanted_e]

    rerankers: list[RerankerSpec] = list(LOCAL_ONNX_RERANKERS)
    if reranker_ids:
        wanted_r = set(reranker_ids)
        rerankers = [r for r in rerankers if r.id in wanted_r]

    pair_results: list[dict[str, Any]] = []
    for emb in embedders:
        pair_rerankers = [NO_RERANKER] if emb.backend == "cloud" else rerankers
        for rr in pair_rerankers:
            res = await _evaluate_pair(
                embedder=emb,
                reranker=rr,
                items=items,
                mode=mode,
                cache=cache,
                spend_tracker=spend_tracker,
                run_qa=(run_qa and mode == "live"),
                answer_model=answer_model,
                judge_model=judge_model,
            )
            pair_results.append(res)

    gate = evaluate_selection_gate(pair_results)
    manifest = build_run_manifest(
        benchmark="LongMemEval_S_dev100_onnx_ablation",
        profile=f"ablation-{mode}",
        data_file=os.path.basename(str(data_file)),
        evaluated_questions=len(items),
        total_questions=len(items),
        judge_model=judge_model if (run_qa and mode == "live") else "none",
        judge_prompt_hash=_judge_prompt_template_hash(),
        answer_model=answer_model if (run_qa and mode == "live") else "none",
        embed_model=",".join(e.id for e in embedders),
        max_tokens=4000,
        judge_fallbacks=0,
        estimated_usd=spend_tracker.spent_usd,
        metrics={"pairs": pair_results, "selection_gate": gate},
    )

    return {
        "manifest": manifest,
        "mode": mode,
        "evaluated_questions": len(items),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "candidates": {
            "embedders": [asdict(e) for e in embedders],
            "rerankers": [asdict(r) for r in rerankers],
        },
        "pairs": pair_results,
        "selection_gate": gate,
        "spend": spend_tracker.summary(),
        "cache": cache.stats() if cache is not None else None,
        "judge_fallbacks": 0,
    }


def _print_table(report: dict[str, Any]) -> None:
    print(f"\n{'=' * 96}")
    print(
        f"  INIT-023/SPEC-004 ONNX vs Cloud Ablation "
        f"[mode={report['mode']} | n={report['evaluated_questions']} | "
        f"usd=${report['spend']['estimated_usd']:.4f}/${report['spend']['max_budget_usd']:.2f}]"
    )
    print(f"{'=' * 96}")
    print(f"  {'Pair':<52s} {'R@5':>6s} {'N@5':>6s} {'QA':>6s} {'p50ms':>7s} {'MB':>6s}")
    print(f"  {'-' * 52} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 6}")
    for p in report.get("pairs", []):
        qa_str = f"{p['qa_accuracy']:.1%}" if p.get("qa_accuracy") is not None else "  —  "
        print(
            f"  {p['pair_key']:<52s} "
            f"{p['recall@5']:>6.4f} {p['ndcg@5']:>6.4f} {qa_str:>6s} "
            f"{p['combined_cpu_p50_ms']:>7.2f} {p['approx_download_mb']:>6.1f}"
        )
    gate = report.get("selection_gate", {})
    print(f"{'=' * 96}")
    print(
        f"  Gate status: {gate.get('status')} | Winner: {(gate.get('winner') or {}).get('pair_key')}"
    )
    print(f"{'=' * 96}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="INIT-023/SPEC-004 100-Q LongMemEval_S ONNX vs Cloud Embedding Ablation",
    )
    parser.add_argument("--data-file", required=True, help="Path to longmemeval_s_cleaned.json")
    parser.add_argument(
        "--split-file",
        default=str(DEFAULT_SPLIT_PATH),
        help="Path to stratified 100-Q IDs JSON",
    )
    parser.add_argument(
        "--generate-split",
        action="store_true",
        help="Write stratified 100-Q IDs JSON to --split-file and exit",
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "local-onnx", "live"],
        default="dry-run",
        help="Execution mode (default: dry-run)",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=DEFAULT_MAX_BUDGET_USD,
        help="Hard USD spend cap (<= $5.00 for Wave 1)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".benchmarks/cache/ablation_onnx",
        help="SHA-256 disk cache directory",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional question cap for smoke runs")
    parser.add_argument("--no-qa", action="store_true", help="Skip Answer+Judge in live mode")
    parser.add_argument(
        "--output",
        default=".benchmarks/wave1_onnx_ablation.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    if args.generate_split:
        items = _load_data(args.data_file)
        split = select_stratified_dev100(items, target_n=100, seed=42)
        out_path = Path(args.split_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {split['total_selected']} stratified IDs to {out_path}")
        return

    try:
        report = await run_onnx_ablation(
            data_file=args.data_file,
            split_path=args.split_file,
            mode=args.mode,
            max_budget_usd=args.max_budget_usd,
            cache_dir=args.cache_dir or None,
            limit=args.limit,
            run_qa=not args.no_qa,
        )
    except BudgetExceededError as exc:
        print(f"ABORTED (budget cap): {exc}", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_table(report)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
