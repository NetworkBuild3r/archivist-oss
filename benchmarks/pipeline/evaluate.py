"""Archivist retrieval benchmark harness — declarative variant definitions,
trustworthy metrics, and clean A/B comparison output.

Metrics: Recall@1, Recall@5, Recall@10, NDCG@5, NDCG@10, latency, token cost.
Query expansion is disabled by default across all variants (see Phase 1 notes).

Prerequisites:
    - Qdrant running (docker compose up -d qdrant)
    - Embedding API reachable (LLM_URL / EMBED_URL configured)
    - Seed corpus: benchmarks/fixtures/corpus/ (legacy) or generate presets:
          python benchmarks/fixtures/generate_corpus.py --preset small
          python benchmarks/fixtures/generate_corpus.py --write-questions

Usage:
    # Standard run: small corpus, three clean-path variants
    python -m benchmarks.pipeline.evaluate --memory-scale small \\
        --variants vector_only,vector_plus_synth,vector_plus_synth_plus_reranker \\
        --no-refine --print-slices --output .benchmarks/phase1.json

    # One-time expansion A/B proof (use --skip-index if already indexed)
    python -m benchmarks.pipeline.evaluate --memory-scale small \\
        --variants expansion_off,expansion_on \\
        --no-refine --print-slices --output .benchmarks/expansion_ab.json

    # Scale sweep across small/medium/large
    python -m benchmarks.pipeline.evaluate --scale-sweep \\
        --variants vector_only,vector_plus_synth_plus_reranker \\
        --no-refine --print-slices
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time

env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(env_path):
    with open(env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# Host runs: default SQLITE_PATH is /data/... (Docker). Use a writable repo-local DB unless set in .env.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault(
    "SQLITE_PATH",
    os.path.join(_repo_root, ".benchmarks", "sqlite", "graph.db"),
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
DEFAULT_CORPUS_DIR = os.path.join(FIXTURES_DIR, "corpus")
QUESTIONS_PATH = os.path.join(FIXTURES_DIR, "questions.json")
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

CORPUS_BY_PRESET = {
    "small": os.path.join(FIXTURES_DIR, "corpus_small"),
    "medium": os.path.join(FIXTURES_DIR, "corpus_medium"),
    "large": os.path.join(FIXTURES_DIR, "corpus_large"),
    "stress": os.path.join(FIXTURES_DIR, "corpus_stress"),
}

HYPOTHESES = {
    "H0_expansion_ab": (
        "Query expansion adds negligible recall (<1pp) while doubling p50 latency "
        "due to 3 extra LLM calls per query.  Disabling it is a strict improvement."
    ),
    "H1_synth_questions": (
        "Index-time synthetic question embeddings improve recall by creating semantic "
        "bridges between queries and content — value should show on needle queries."
    ),
    "H2_reranker": (
        "A cross-encoder reranker applied after coarse nomination should improve "
        "ranking quality (NDCG) without changing recall."
    ),
    "H3_scale": (
        "As corpus grows, simple vector search degrades on needle/temporal queries; "
        "the gap between vector_only and enriched variants widens."
    ),
}

logger = logging.getLogger("archivist.benchmark.pipeline")

# ── Variant definitions ──────────────────────────────────────────────────────
# Each variant declares the full set of flags it cares about.  Every variant
# explicitly sets QUERY_EXPANSION_ENABLED=false because expansion was measured
# at +0.4pp recall for +6s latency and is not worth the cost.
#
# Two temporary variants (expansion_off / expansion_on) exist solely for the
# one-time A/B proof run.  Delete them after the comparison is recorded.

_COMMON_DISABLED = {
    "QUERY_EXPANSION_ENABLED": "false",
    "RERANK_ENABLED": "false",
    "RERANKER_ENABLED": "false",
    "GRAPH_RETRIEVAL_ENABLED": "false",
    "HOTNESS_WEIGHT": "0",
    "TEMPORAL_DECAY_HALFLIFE_DAYS": "0",
}

VARIANTS = {
    # ── v2 clean-path variants (no legacy pipeline stages) ───────────────────
    "vector_only": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "false",
        "RETRIEVAL_THRESHOLD": "0.2",
        "SYNTHETIC_QUESTIONS_ENABLED": "false",
    },
    "vector_plus_synth": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "false",
        "RETRIEVAL_THRESHOLD": "0.2",
        "SYNTHETIC_QUESTIONS_ENABLED": "true",
    },
    "vector_plus_synth_plus_reranker": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "false",
        "RERANKER_ENABLED": "true",
        "RETRIEVAL_THRESHOLD": "0.0",
        "SYNTHETIC_QUESTIONS_ENABLED": "true",
    },
    # ── v2 clean nominate-then-rerank (Phase 3) ──────────────────────────────
    # Full nomination pool (vector + synth + BM25 + needle registry + graph)
    # → ID-dedupe → cross-encoder rerank → top-K.  No RRF, no threshold, no
    # temporal decay, no hotness, no rescue.  The cross-encoder is the single
    # ranking authority.
    "clean_reranker": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANKER_ENABLED": "true",
        "RETRIEVAL_THRESHOLD": "0.0",
        "SYNTHETIC_QUESTIONS_ENABLED": "true",
    },
    # ── One-time expansion A/B (delete after proof run) ──────────────────────
    "expansion_off": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "false",
        "RETRIEVAL_THRESHOLD": "0.2",
        "SYNTHETIC_QUESTIONS_ENABLED": "false",
        "QUERY_EXPANSION_ENABLED": "false",
    },
    "expansion_on": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "false",
        "RETRIEVAL_THRESHOLD": "0.2",
        "SYNTHETIC_QUESTIONS_ENABLED": "false",
        "QUERY_EXPANSION_ENABLED": "true",
    },
    # ── Legacy additive variants (kept for historical comparison) ────────────
    "plus_bm25": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "true",
        "RETRIEVAL_THRESHOLD": "0.2",
        "SYNTHETIC_QUESTIONS_ENABLED": "false",
    },
    "plus_graph": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "plus_temporal": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "plus_hotness": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "plus_rerank": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "true",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "full_pipeline": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "RETRIEVAL_THRESHOLD": "0.2",
    },
    "full_pipeline_rerank": {
        **_COMMON_DISABLED,
        "BM25_ENABLED": "true",
        "GRAPH_RETRIEVAL_ENABLED": "true",
        "RERANK_ENABLED": "true",
        "HOTNESS_WEIGHT": "0.15",
        "TEMPORAL_DECAY_HALFLIFE_DAYS": "365",
        "RETRIEVAL_THRESHOLD": "0.2",
        "TEMPORAL_INTENT_ENABLED": "true",
        "BM25_RESCUE_ENABLED": "true",
        "ADAPTIVE_VECTOR_LIMIT_ENABLED": "true",
    },
}


def _apply_variant(variant_name: str):
    """Apply config overrides for a variant by setting env vars and reloading config.

    Logs the key flags being toggled so benchmark output is self-documenting.
    """
    env_overrides = VARIANTS[variant_name]
    for key, value in env_overrides.items():
        os.environ[key] = value

    import importlib

    import config

    importlib.reload(config)

    bench_coll = os.environ.get("QDRANT_COLLECTION", "")
    if bench_coll and bench_coll.startswith("archivist_benchmark_"):
        config.QDRANT_COLLECTION = bench_coll

    for attr, val in env_overrides.items():
        if hasattr(config, attr):
            config_val = val
            if val in ("true", "false"):
                config_val = val == "true"
            elif val.replace(".", "").replace("-", "").isdigit():
                config_val = float(val) if "." in val else int(val)
            setattr(config, attr, config_val)

    propagation_targets = [
        "rlm_retriever",
        "graph_retrieval",
        "retrieval_filters",
        "memory_fusion",
        "hot_cache",
        "hotness",
        "synthetic_questions",
        "reranker",
        "query_expansion",
        "fts_search",
        "rank_fusion",
    ]
    for mod_name in propagation_targets:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    enabled_flags = [k for k, v in env_overrides.items() if v in ("true", "1")]
    disabled_flags = [k for k, v in env_overrides.items() if v in ("false", "0")]
    logger.info(
        "  variant=%s  ON=[%s]  OFF=[%s]",
        variant_name,
        ", ".join(enabled_flags) or "none",
        ", ".join(disabled_flags) or "none",
    )


def _extract_source_text(source: dict) -> str:
    """Extract the actual content from a source dict.

    With refine=False the source is the raw Qdrant payload dict containing 'text'
    and 'tier_text'.  With refine=True the source is a slim metadata dict and the
    relevant content lives in the synthesized 'answer' — not in the source at all.
    We never fall back to str(dict) because matching keywords against metadata
    keys and score floats produces false positives.
    """
    for field in ("tier_text", "text", "l1", "l0"):
        val = source.get(field, "")
        if val:
            return str(val)
    return ""


def _keyword_recall_at_k(
    answer_text: str,
    sources: list[dict],
    expected_keywords: list[str],
    k: int,
) -> float:
    """Fraction of expected keywords found in answer_text + top-k source texts.

    keyword matching operates on content fields only — never on str(dict).
    """
    if not expected_keywords:
        return 0.0
    parts = [answer_text] if answer_text else []
    for s in sources[:k]:
        parts.append(_extract_source_text(s))
    text_lower = " ".join(parts).lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return found / len(expected_keywords)


def _ndcg_at_k(
    sources: list[dict],
    expected_keywords: list[str],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain using keyword overlap as graded relevance.

    Each source gets a relevance grade = fraction of expected keywords it contains.
    NDCG@k compares the actual ordering against the ideal (best possible) ordering
    within the top-k positions.  Returns 0.0 when no source contains any keyword.
    """
    import math

    if not expected_keywords or not sources:
        return 0.0

    kw_lower = [kw.lower() for kw in expected_keywords]

    def _relevance(src: dict) -> float:
        text = _extract_source_text(src).lower()
        if not text:
            return 0.0
        return sum(1 for kw in kw_lower if kw in text) / len(kw_lower)

    gains = [_relevance(s) for s in sources[:k]]

    def _dcg(g: list[float]) -> float:
        return sum(g_i / math.log2(i + 2) for i, g_i in enumerate(g))

    dcg = _dcg(gains)
    ideal_gains = sorted(gains, reverse=True)
    idcg = _dcg(ideal_gains)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def resolve_corpus_dir(memory_scale: str | None, corpus_dir_override: str | None) -> str:
    if corpus_dir_override:
        return os.path.abspath(corpus_dir_override)
    if memory_scale and memory_scale in CORPUS_BY_PRESET:
        return CORPUS_BY_PRESET[memory_scale]
    return DEFAULT_CORPUS_DIR


def filter_questions_for_scale(questions: list[dict], memory_scale: str | None) -> list[dict]:
    """Keep only questions whose ``scales`` list includes this corpus preset.

    Questions without a ``scales`` field are treated as valid for all sizes.
    The ``stress`` corpus uses the same question mix as ``large``.
    """
    if not memory_scale:
        return questions
    effective_scale = "large" if memory_scale == "stress" else memory_scale
    out = []
    for q in questions:
        scales = q.get("scales")
        if scales is None or "all" in scales or effective_scale in scales:
            out.append(q)
    n_dropped = len(questions) - len(out)
    if n_dropped:
        logger.info(
            "Filtered %d/%d questions not applicable to scale=%s",
            n_dropped,
            len(questions),
            memory_scale,
        )
    return out


def _benchmark_collection_name(memory_scale: str | None) -> str:
    """Return a benchmark-specific Qdrant collection name to avoid stomping production data."""
    scale = memory_scale or "default"
    return f"archivist_benchmark_{scale}"


async def index_corpus(corpus_dir: str, memory_scale: str | None = None):
    """Index the seed corpus files into Qdrant and FTS5."""
    import config

    config.MEMORY_ROOT = corpus_dir

    # Use a benchmark-specific collection so production data is never touched
    bench_coll = _benchmark_collection_name(memory_scale)
    os.environ["QDRANT_COLLECTION"] = bench_coll
    config.QDRANT_COLLECTION = bench_coll

    # Disable write-time LLM enrichment features (contextual augmentation, reverse HyDE) for fast indexing.
    # These improve recall at write time but make benchmark indexing take 10-50x longer with a slow LLM.
    os.environ["CONTEXTUAL_AUGMENTATION_ENABLED"] = "false"
    os.environ["REVERSE_HYDE_ENABLED"] = "false"
    os.environ["TIERED_CONTEXT_ENABLED"] = "false"
    config.CONTEXTUAL_AUGMENTATION_ENABLED = False
    config.REVERSE_HYDE_ENABLED = False
    config.TIERED_CONTEXT_ENABLED = False

    # Synthetic questions are opt-in for benchmarks (set env or pass variant list with vector_plus_synth)
    _synth_q = os.environ.get("SYNTHETIC_QUESTIONS_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    config.SYNTHETIC_QUESTIONS_ENABLED = _synth_q
    if _synth_q:
        logger.info("Synthetic question generation ENABLED for this indexing run")

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

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
            ("representation_type", PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(
                collection_name=config.QDRANT_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        logger.info(
            "Created benchmark collection '%s' (%d-dim)",
            config.QDRANT_COLLECTION,
            config.VECTOR_DIM,
        )
    else:
        info = client.get_collection(config.QDRANT_COLLECTION)
        if info.points_count > 0:
            client.delete_collection(config.QDRANT_COLLECTION)
            client.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=config.VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info(
                "Recreated benchmark collection '%s' for clean run", config.QDRANT_COLLECTION
            )

    # Always start from a clean SQLite DB so schema migrations never run against
    # a stale database from a previous benchmark run.
    _db_path = os.environ.get("SQLITE_PATH", "")
    if _db_path and os.path.exists(_db_path):
        os.remove(_db_path)
        logger.info("Removed stale benchmark DB: %s", _db_path)
    if _db_path:
        os.makedirs(os.path.dirname(_db_path), exist_ok=True)

    from graph import init_schema
    from indexer import full_index

    init_schema()
    from archivist.storage.sqlite_pool import initialize_pool
    await initialize_pool()
    logger.info("Indexing corpus from %s ...", corpus_dir)
    count = await full_index(hierarchical=True)
    logger.info("Indexed %d chunks from corpus", count)
    return count


async def run_variant(
    variant_name: str,
    questions: list[dict],
    refine: bool,
    *,
    progress_pct_step: int = 10,
    checkpoint_path: str | None = None,
    memory_scale: str | None = None,
    use_progress_bar: bool = True,
) -> dict:
    """Run all questions against a pipeline variant and collect metrics."""
    from benchmarks.pipeline.run_progress import ProgressTracker

    _apply_variant(variant_name)

    # Force-clear the hot cache AFTER variant config is applied.
    # invalidate_all() is a no-op when HOT_CACHE_ENABLED=false, but the PREVIOUS
    # variant may have populated the cache while it was enabled.  We must clear
    # the raw data structures to prevent cross-variant leakage.
    import hot_cache as _hc

    _hc.force_invalidate_all()

    # Re-import recursive_retrieve from the freshly-reloaded rlm_retriever
    # (_apply_variant already reloaded it with the new config).
    from rlm_retriever import recursive_retrieve
    from tokenizer import count_tokens

    results = []
    latencies = []
    token_costs = []
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    ndcg_at_5 = []
    ndcg_at_10 = []

    progress = ProgressTracker(
        total=len(questions),
        phase=variant_name,
        memory_scale=memory_scale,
        pct_step=progress_pct_step,
        checkpoint_path=checkpoint_path,
        use_progress_bar=use_progress_bar,
    )

    try:
        for q in questions:
            t0 = time.monotonic()
            q_namespace = q.get("namespace", "")
            q_agent_id = q.get("caller_agent_id", "")
            q_date_from = q.get("date_from", "")
            q_date_to = q.get("date_to", "")
            try:
                result = await recursive_retrieve(
                    query=q["query"],
                    namespace=q_namespace,
                    agent_id=q_agent_id,
                    limit=10,
                    refine=refine,
                    tier="l2",
                    date_from=q_date_from,
                    date_to=q_date_to,
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
                token_cost += count_tokens(_extract_source_text(s) or "")
            token_costs.append(token_cost)

            expected_kw = q.get("expected_keywords", [])

            r1 = _keyword_recall_at_k(answer_text, sources, expected_kw, k=1)
            recall_at_1.append(min(r1, 1.0))

            r5 = _keyword_recall_at_k(answer_text, sources, expected_kw, k=5)
            recall_at_5.append(min(r5, 1.0))

            r10 = _keyword_recall_at_k(answer_text, sources, expected_kw, k=10)
            recall_at_10.append(min(r10, 1.0))

            n5 = _ndcg_at_k(sources, expected_kw, k=5)
            ndcg_at_5.append(n5)

            n10 = _ndcg_at_k(sources, expected_kw, k=10)
            ndcg_at_10.append(n10)

            results.append(
                {
                    "question_id": q["id"],
                    "query": q["query"],
                    "query_type": q.get("query_type", ""),
                    "latency_ms": round(elapsed_ms, 1),
                    "token_cost": token_cost,
                    "recall_at_1": round(r1, 4),
                    "recall_at_5": round(r5, 4),
                    "recall_at_10": round(r10, 4),
                    "ndcg_at_5": round(n5, 4),
                    "ndcg_at_10": round(n10, 4),
                    "sources_count": len(sources),
                    "synthetic_hits": trace.get("synthetic_hits", 0),
                    "nomination_pool_size": trace.get("nomination_pool_size", 0),
                    "trace": trace,
                }
            )

            progress.step(
                len(results),
                results=results,
                rolling_recall=statistics.mean(recall_at_5),
                rolling_mrr=statistics.mean(ndcg_at_5),
            )
    finally:
        progress.close()

    summary = {
        "variant": variant_name,
        "refine": refine,
        "total_queries": len(questions),
        "recall_at_1": round(statistics.mean(recall_at_1), 4) if recall_at_1 else 0,
        "recall_at_5": round(statistics.mean(recall_at_5), 4) if recall_at_5 else 0,
        "recall_at_10": round(statistics.mean(recall_at_10), 4) if recall_at_10 else 0,
        "ndcg_at_5": round(statistics.mean(ndcg_at_5), 4) if ndcg_at_5 else 0,
        "ndcg_at_10": round(statistics.mean(ndcg_at_10), 4) if ndcg_at_10 else 0,
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "latency_p95_ms": round(
            sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1
        ),
        "latency_mean_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "total_tokens": sum(token_costs),
        "avg_tokens_per_query": round(statistics.mean(token_costs), 0) if token_costs else 0,
        "total_synthetic_hits": sum(r.get("synthetic_hits", 0) for r in results),
        "queries_with_synthetic_hits": sum(1 for r in results if r.get("synthetic_hits", 0) > 0),
        "avg_nomination_pool_size": round(
            statistics.mean(
                [
                    r.get("nomination_pool_size", 0)
                    for r in results
                    if r.get("nomination_pool_size", 0) > 0
                ]
            ),
            1,
        )
        if any(r.get("nomination_pool_size", 0) > 0 for r in results)
        else 0,
    }

    by_type = {}
    for r in results:
        qt = r["query_type"]
        if qt not in by_type:
            by_type[qt] = {
                "recall_1": [],
                "recall_5": [],
                "recall_10": [],
                "ndcg_5": [],
                "ndcg_10": [],
                "latency": [],
            }
        by_type[qt]["recall_1"].append(r["recall_at_1"])
        by_type[qt]["recall_5"].append(r["recall_at_5"])
        by_type[qt]["recall_10"].append(r["recall_at_10"])
        by_type[qt]["ndcg_5"].append(r["ndcg_at_5"])
        by_type[qt]["ndcg_10"].append(r["ndcg_at_10"])
        by_type[qt]["latency"].append(r["latency_ms"])

    summary["by_query_type"] = {
        qt: {
            "count": len(vals["recall_5"]),
            "recall_at_1": round(statistics.mean(vals["recall_1"]), 4),
            "recall_at_5": round(statistics.mean(vals["recall_5"]), 4),
            "recall_at_10": round(statistics.mean(vals["recall_10"]), 4),
            "ndcg_at_5": round(statistics.mean(vals["ndcg_5"]), 4),
            "ndcg_at_10": round(statistics.mean(vals["ndcg_10"]), 4),
            "latency_p50": round(statistics.median(vals["latency"]), 1),
        }
        for qt, vals in by_type.items()
    }

    return {"summary": summary, "results": results}


def format_table(all_summaries: list[dict]) -> str:
    """Format summaries as a markdown comparison table with optional wall time, synth hits, and pool size."""
    has_wall = any(s.get("wall_time_s") for s in all_summaries)
    has_synth = any(s.get("total_synthetic_hits", 0) > 0 for s in all_summaries)
    has_pool = any(s.get("avg_nomination_pool_size", 0) > 0 for s in all_summaries)
    hdr = "| Variant | R@1 | R@5 | R@10 | NDCG@5 | NDCG@10 | p50 (ms) | p95 (ms) | Tok/Q |"
    sep = "|---------|-----|-----|------|--------|---------|----------|----------|-------|"
    if has_synth:
        hdr += " Synth Hits |"
        sep += "------------|"
    if has_pool:
        hdr += " Pool |"
        sep += "------|"
    if has_wall:
        hdr += " Wall (s) |"
        sep += "----------|"
    lines = [hdr, sep]
    for s in all_summaries:
        row = (
            f"| {s['variant']} "
            f"| {s['recall_at_1']:.4f} "
            f"| {s['recall_at_5']:.4f} "
            f"| {s['recall_at_10']:.4f} "
            f"| {s['ndcg_at_5']:.4f} "
            f"| {s['ndcg_at_10']:.4f} "
            f"| {s['latency_p50_ms']:.0f} "
            f"| {s['latency_p95_ms']:.0f} "
            f"| {s['avg_tokens_per_query']:.0f} |"
        )
        if has_synth:
            total_sh = s.get("total_synthetic_hits", 0)
            q_sh = s.get("queries_with_synthetic_hits", 0)
            row += f" {total_sh} ({q_sh}q) |"
        if has_pool:
            row += f" {s.get('avg_nomination_pool_size', 0):.0f} |"
        if has_wall:
            row += f" {s.get('wall_time_s', 0):.1f} |"
        lines.append(row)
    return "\n".join(lines)


def format_retention_slices_table(
    summaries: list[dict],
    slice_types: tuple[str, ...] = ("temporal", "needle", "multi_hop", "single_hop"),
) -> str:
    """Markdown table: per-variant recall@5 and NDCG@5 on key query type slices."""
    header = "| Variant | " + " | ".join(f"{t} (R@5 / NDCG@5)" for t in slice_types) + " |"
    sep = "|---------|" + "|".join("--------|" for _ in slice_types)
    lines = [header, sep]
    for s in summaries:
        by = s.get("by_query_type") or {}
        v = s.get("variant", "")
        ms = s.get("memory_scale")
        label = f"{ms}/{v}" if ms else v
        cells = [label]
        for t in slice_types:
            row = by.get(t) or {}
            r5 = row.get("recall_at_5")
            ndcg = row.get("ndcg_at_5")
            n = row.get("count", 0)
            if isinstance(r5, int | float) and isinstance(ndcg, int | float):
                cells.append(f"{r5:.4f} / {ndcg:.4f} (n={n})")
            else:
                cells.append("---")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_comparison_table(
    stuffing_summaries: list[dict],
    archivist_summaries: list[dict],
) -> str:
    """Side-by-side markdown table: context stuffing baseline vs best Archivist variant per scale."""
    archivist_by_scale: dict[str, list[dict]] = {}
    for s in archivist_summaries:
        scale = s.get("memory_scale") or "default"
        archivist_by_scale.setdefault(scale, []).append(s)

    lines = [
        "## Break-Even: Context Stuffing vs Archivist Retrieval",
        "",
        "| Scale | Files | Corpus Tokens | Fits in Context? | Stuffing Recall | Stuffing p50 (ms) "
        "| Best Archivist Variant | Archivist Recall | Archivist p50 (ms) | Archivist Tok/Q |",
        "|-------|-------|---------------|-----------------|-----------------|------------------"
        "|------------------------|------------------|---------------------|-----------------|",
    ]

    for ss in stuffing_summaries:
        scale = ss.get("memory_scale") or "default"
        file_count = ss.get("file_count", "?")
        corpus_tokens = ss.get("corpus_tokens", 0)
        corpus_overflow = ss.get("corpus_overflow", False)
        fits = "NO — OVERFLOW" if corpus_overflow else "YES"

        stuffing_recall = f"{ss.get('recall', 0):.4f}" if ss.get("llm_called") else "—(no LLM)"
        stuffing_p50 = (
            f"{ss['latency_p50_ms']:.0f}" if ss.get("latency_p50_ms") is not None else "—"
        )

        arch_variants = archivist_by_scale.get(scale, [])
        if arch_variants:
            best_arch = max(arch_variants, key=lambda v: v.get("recall_at_5", 0))
            arch_variant = best_arch.get("variant", "?")
            arch_recall = f"{best_arch.get('recall_at_5', 0):.4f}"
            arch_p50 = f"{best_arch.get('latency_p50_ms', 0):.0f}"
            arch_tok = f"{best_arch.get('avg_tokens_per_query', 0):.0f}"
        else:
            arch_variant = arch_recall = arch_p50 = arch_tok = "—"

        lines.append(
            f"| {scale} | {file_count} | {corpus_tokens:,} | {fits} | {stuffing_recall} "
            f"| {stuffing_p50} | {arch_variant} | {arch_recall} | {arch_p50} | {arch_tok} |"
        )

    lines += [
        "",
        "> **Reading the table:** When 'Fits in Context?' = YES, stuffing has all facts available "
        "but pays full corpus token cost per query and suffers 'lost in the middle' degradation. "
        "When OVERFLOW, older or less-salient memories are silently truncated — facts are lost. "
        "Archivist Tok/Q stays fixed regardless of corpus size.",
    ]
    return "\n".join(lines)


def format_full_comparison_table(
    stuffing_summaries: list[dict],
    archivist_summaries: list[dict],
) -> str:
    """Per-query-type side-by-side: context stuffing (with real LLM) vs best Archivist variant."""
    query_types = [
        "single_hop",
        "multi_hop",
        "temporal",
        "adversarial",
        "agent_scoped",
        "broad",
        "contradiction",
        "needle",
    ]

    stuffing_by_scale: dict[str, dict] = {
        ss.get("memory_scale", "default"): ss for ss in stuffing_summaries
    }
    archivist_by_scale: dict[str, list[dict]] = {}
    for s in archivist_summaries:
        scale = s.get("memory_scale", "default")
        archivist_by_scale.setdefault(scale, []).append(s)

    scales = list(
        dict.fromkeys(
            [ss.get("memory_scale", "default") for ss in stuffing_summaries]
            + list(archivist_by_scale.keys())
        )
    )

    lines = [
        "## Full Comparison: Context Stuffing vs Archivist — Per Query Type",
        "",
        "> Stuffing recall measured with real LLM calls. "
        "Archivist recall = best variant (full_pipeline preferred) at each scale.",
        "",
    ]

    for scale in scales:
        ss = stuffing_by_scale.get(scale)
        arch_variants = archivist_by_scale.get(scale, [])
        best_arch = next(
            (v for v in arch_variants if v.get("variant") == "full_pipeline"), None
        ) or (max(arch_variants, key=lambda v: v.get("recall_at_5", 0)) if arch_variants else None)

        overflow_label = ""
        if ss:
            if ss.get("corpus_overflow"):
                overflow_label = (
                    f" **OVERFLOW** ({ss['corpus_tokens']:,} tok > {ss['context_budget']:,} budget)"
                )
            else:
                pct = round(ss.get("corpus_tokens", 0) / max(ss.get("context_budget", 1), 1) * 100)
                overflow_label = f" ({ss['corpus_tokens']:,} tok, {pct}% of window)"

        arch_label = f"Archivist/{best_arch['variant']}" if best_arch else "Archivist"
        lines.append(f"### Scale: {scale}{overflow_label}")
        lines.append("")
        lines.append(
            f"| Query Type | Count | Stuffing Recall | Stuffing Overflow% | {arch_label} Recall | Delta |"
        )
        lines.append(
            "|------------|-------|-----------------|--------------------|---------------------|-------|"
        )

        all_types_here: list[str] = []
        stuff_by_type = (ss or {}).get("by_query_type", {})
        arch_by_type = (best_arch or {}).get("by_query_type", {})
        for qt in query_types:
            if qt in stuff_by_type or qt in arch_by_type:
                all_types_here.append(qt)

        for qt in all_types_here:
            sr = stuff_by_type.get(qt, {})
            ar = arch_by_type.get(qt, {})
            n = sr.get("count") or ar.get("count") or 0
            s_recall = sr.get("recall")
            a_recall = ar.get("recall_at_5")
            s_overflow = sr.get("overflow_pct")
            if s_recall is not None and a_recall is not None:
                delta = a_recall - s_recall
                delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            else:
                delta_str = "—"
            lines.append(
                f"| {qt} | {n} "
                f"| {f'{s_recall:.3f}' if s_recall is not None else '—'} "
                f"| {f'{s_overflow:.0f}%' if s_overflow is not None else '—'} "
                f"| {f'{a_recall:.3f}' if a_recall is not None else '—'} "
                f"| {delta_str} |"
            )

        s_overall = (ss or {}).get("recall")
        a_overall = best_arch.get("recall_at_5") if best_arch else None
        if s_overall is not None and a_overall is not None:
            delta = a_overall - s_overall
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            lines.append(
                f"| **Overall** | — "
                f"| **{s_overall:.3f}** "
                f"| {(ss or {}).get('overflow_pct', 0):.0f}% "
                f"| **{a_overall:.3f}** "
                f"| **{delta_str}** |"
            )
        lines.append("")

    return "\n".join(lines)


_FIXTURE_GLOSSARY = {
    "kubernetes": {"type": "technology", "aliases": ["k8s"]},
    "prometheus": {"type": "tool"},
    "grafana": {"type": "tool"},
    "argocd": {"type": "tool", "aliases": ["argo cd"]},
    "gitlab ci": {"type": "tool", "aliases": ["gitlab"]},
    "postgresql": {"type": "database", "aliases": ["postgres"]},
    "redis": {"type": "database"},
    "vault": {"type": "tool", "aliases": ["hashicorp vault"]},
    "istio": {"type": "technology"},
    "rabbitmq": {"type": "tool"},
    "pagerduty": {"type": "tool"},
    "karpenter": {"type": "tool"},
    "flyway": {"type": "tool"},
    "pgbouncer": {"type": "tool"},
    "playwright": {"type": "tool"},
    "thanos": {"type": "tool"},
    "etcd": {"type": "database"},
    "chief": {"type": "agent"},
    "gitbob": {"type": "agent"},
    "grafgreg": {"type": "agent"},
    "argo": {"type": "agent"},
    "kubekate": {"type": "agent"},
    "securitysam": {"type": "agent"},
    "needleagent": {"type": "agent"},
}


def _seed_graph_from_fixtures(corpus_dir: str):
    """Deterministic graph population from a known glossary — no LLM calls."""
    from graph import add_fact, init_schema, upsert_entity

    init_schema()

    for name, meta in _FIXTURE_GLOSSARY.items():
        eid = upsert_entity(name, meta["type"])
        add_fact(
            eid,
            f"{name} is a {meta['type']} in the benchmark corpus",
            "benchmark/fixtures",
            "benchmark",
        )
        for alias in meta.get("aliases", []):
            upsert_entity(alias, meta["type"])

    import glob as _glob

    md_files = _glob.glob(os.path.join(corpus_dir, "**", "*.md"), recursive=True)
    for fpath in md_files[:50]:
        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                text = f.read(2000).lower()
        except Exception:
            continue
        rel = os.path.relpath(fpath, corpus_dir)
        for name, meta in _FIXTURE_GLOSSARY.items():
            if name in text:
                eid = upsert_entity(name, meta["type"])
                snippet = text[:200].replace("\n", " ").strip()
                add_fact(eid, f"Referenced in {rel}: {snippet[:120]}", rel, "benchmark")


async def _run_benchmark_session(
    *,
    corpus_dir: str,
    questions: list[dict],
    variant_names: list[str],
    refine: bool,
    skip_index: bool,
    run_curator: bool,
    memory_scale: str | None,
    **kwargs,
) -> tuple[dict, list[dict]]:
    """Index (optional), curator (optional), run variants; return (all_results, all_summaries).

    Prints a clear session header/footer with timing data for easy copy-paste.
    """
    import importlib

    import config

    corpus_dir = os.path.abspath(corpus_dir)
    os.environ["MEMORY_ROOT"] = corpus_dir

    bench_coll = _benchmark_collection_name(memory_scale)
    os.environ["QDRANT_COLLECTION"] = bench_coll
    importlib.reload(config)
    config.QDRANT_COLLECTION = bench_coll

    session_start = time.monotonic()
    scale_label = memory_scale or "default"
    logger.info(
        "━━━ Session: scale=%s  variants=%s  questions=%d  refine=%s ━━━",
        scale_label,
        ",".join(variant_names),
        len(questions),
        refine,
    )

    if not skip_index:
        if any(
            v in variant_names
            for v in (
                "vector_plus_synth",
                "vector_plus_synth_plus_reranker",
                "clean_reranker",
            )
        ):
            os.environ["SYNTHETIC_QUESTIONS_ENABLED"] = "true"
        await index_corpus(corpus_dir, memory_scale=memory_scale)

    if run_curator:
        import curator

        importlib.reload(config)
        importlib.reload(curator)
        logger.info("Running curator extraction over %s (LLM calls)...", corpus_dir)
        n = await curator.extract_all_agent_memories()
        logger.info("Curator finished: %d files extracted", n)

    warm_graph = kwargs.get("warm_graph", False)
    if warm_graph and not run_curator:
        _seed_graph_from_fixtures(corpus_dir)
        logger.info("Warm graph seeded from fixture glossary")

    all_results = {}
    all_summaries = []

    progress_pct_step = int(kwargs.get("progress_pct_step", 10))
    checkpoint_path = kwargs.get("checkpoint_path")
    use_progress_bar = kwargs.get("use_progress_bar", True)
    for i, variant in enumerate(variant_names, 1):
        logger.info("─── [%d/%d] Variant: %s ───", i, len(variant_names), variant)
        variant_start = time.monotonic()
        data = await run_variant(
            variant,
            questions,
            refine=refine,
            progress_pct_step=progress_pct_step,
            checkpoint_path=checkpoint_path,
            memory_scale=memory_scale,
            use_progress_bar=use_progress_bar,
        )
        variant_elapsed = time.monotonic() - variant_start
        all_results[variant] = data
        summary = dict(data["summary"])
        if memory_scale:
            summary["memory_scale"] = memory_scale
        summary["corpus_dir"] = corpus_dir
        summary["wall_time_s"] = round(variant_elapsed, 1)
        all_summaries.append(summary)
        logger.info(
            "  R@1=%.4f  R@5=%.4f  R@10=%.4f  NDCG@5=%.4f  p50=%4.0fms  tok/q=%4.0f  wall=%.1fs",
            data["summary"]["recall_at_1"],
            data["summary"]["recall_at_5"],
            data["summary"]["recall_at_10"],
            data["summary"]["ndcg_at_5"],
            data["summary"]["latency_p50_ms"],
            data["summary"]["avg_tokens_per_query"],
            variant_elapsed,
        )

    session_elapsed = time.monotonic() - session_start
    logger.info(
        "━━━ Session complete: scale=%s  %d variants  %.1fs total ━━━",
        scale_label,
        len(variant_names),
        session_elapsed,
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
    parser.add_argument(
        "--index-only", action="store_true", help="Only index corpus, don't run queries"
    )
    parser.add_argument(
        "--no-refine", action="store_true", help="Skip LLM refinement stages (faster)"
    )
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument(
        "--memory-scale",
        choices=["small", "medium", "large", "stress"],
        default=None,
        help="Use benchmarks/fixtures/corpus_<scale>/; stress = very large haystack",
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
        "--warm-graph",
        action="store_true",
        help="Seed the KG from a deterministic fixture glossary (no LLM cost; cheaper than --run-curator)",
    )
    parser.add_argument(
        "--scale-sweep",
        action="store_true",
        help="Run the same variants for small, medium, large presets (re-indexes each time)",
    )
    parser.add_argument(
        "--print-slices",
        action="store_true",
        help="After the main table, print per-query-type recall (temporal, needle, multi_hop, single_hop)",
    )
    parser.add_argument(
        "--compare-stuffing",
        action="store_true",
        help=(
            "Run the context stuffing baseline alongside Archivist and print a break-even comparison table. "
            "By default uses token-count-only mode (no LLM calls); add --stuffing-call-llm to actually query."
        ),
    )
    parser.add_argument(
        "--stuffing-call-llm",
        action="store_true",
        help="When --compare-stuffing is set, actually call the LLM for each stuffing query (expensive).",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=128000,
        help="Token budget for context stuffing overflow detection (default: 128000).",
    )
    parser.add_argument(
        "--progress-pct",
        type=int,
        default=10,
        metavar="N",
        help="With --no-progress-bar: log every N%% (10, 20, …) plus ETA; default 10",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Atomic JSON checkpoint path for partial per-query results",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint file even when --output is set",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bar; use milestone PROGRESS log lines instead",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    questions_path = args.questions or QUESTIONS_PATH
    with open(questions_path, encoding="utf-8") as f:
        questions_all = json.load(f)
    if args.limit > 0:
        questions_all = questions_all[: args.limit]

    variant_names = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants
        else ([args.variant] if args.variant else list(VARIANTS.keys()))
    )
    refine = not args.no_refine

    checkpoint_path: str | None = None
    if not args.no_checkpoint:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        elif args.output:
            from benchmarks.pipeline.run_progress import default_checkpoint_path

            checkpoint_path = default_checkpoint_path(args.output)

    progress_kwargs = {
        "progress_pct_step": max(1, args.progress_pct),
        "checkpoint_path": checkpoint_path,
        "use_progress_bar": not args.no_progress_bar,
    }

    if args.scale_sweep:
        by_scale = {}
        combined_summaries = []
        stuffing_summaries: list[dict] = []

        for scale in ("small", "medium", "large"):
            corpus_dir = resolve_corpus_dir(scale, args.corpus_dir)
            if not os.path.isdir(os.path.join(corpus_dir, "agents")):
                logger.warning(
                    "Skipping scale=%s — missing corpus at %s",
                    scale,
                    corpus_dir,
                )
                continue
            qs = filter_questions_for_scale(questions_all, scale)
            logger.info("Scale %s: %d questions, corpus %s", scale, len(qs), corpus_dir)

            if args.compare_stuffing:
                from benchmarks.pipeline.context_stuffing_baseline import run_stuffing_baseline

                logger.info("Running context stuffing baseline for scale=%s ...", scale)
                stuffing_data = await run_stuffing_baseline(
                    corpus_dir=corpus_dir,
                    questions=qs,
                    context_budget=args.context_budget,
                    call_llm=args.stuffing_call_llm,
                    memory_scale=scale,
                    **progress_kwargs,
                )
                stuffing_summary = stuffing_data["summary"]
                stuffing_summary["memory_scale"] = scale
                stuffing_summaries.append(stuffing_summary)

            all_results, summaries = await _run_benchmark_session(
                corpus_dir=corpus_dir,
                questions=qs,
                variant_names=variant_names,
                refine=refine,
                skip_index=args.skip_index,
                run_curator=args.run_curator,
                memory_scale=scale,
                warm_graph=args.warm_graph,
                **progress_kwargs,
            )
            by_scale[scale] = {
                "summaries": summaries,
                "full_results": {k: v["results"] for k, v in all_results.items()},
            }
            combined_summaries.extend(summaries)

        print("\n" + format_table(combined_summaries))
        if args.print_slices:
            print("\n### Per-query-type slices (memory / time / haystack)\n")
            print(format_retention_slices_table(combined_summaries))
        if args.compare_stuffing and stuffing_summaries:
            print("\n")
            print(format_comparison_table(stuffing_summaries, combined_summaries))
            if args.stuffing_call_llm:
                print("\n")
                print(format_full_comparison_table(stuffing_summaries, combined_summaries))

        if args.output:
            output_data = {
                "benchmark_meta": {
                    "hypotheses": HYPOTHESES,
                    "scale_sweep": True,
                    "variants": variant_names,
                    "context_budget": args.context_budget if args.compare_stuffing else None,
                },
                "by_scale": by_scale,
                "summaries": combined_summaries,
                "comparison_table": format_table(combined_summaries),
            }
            if args.print_slices:
                output_data["retention_slices_table"] = format_retention_slices_table(
                    combined_summaries
                )
            if args.compare_stuffing and stuffing_summaries:
                output_data["stuffing_summaries"] = stuffing_summaries
                output_data["breakeven_table"] = format_comparison_table(
                    stuffing_summaries, combined_summaries
                )
                if args.stuffing_call_llm:
                    output_data["full_comparison_table"] = format_full_comparison_table(
                        stuffing_summaries, combined_summaries
                    )
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

    single_stuffing_summary: dict | None = None
    if args.compare_stuffing:
        from benchmarks.pipeline.context_stuffing_baseline import run_stuffing_baseline

        logger.info("Running context stuffing baseline ...")
        stuffing_data = await run_stuffing_baseline(
            corpus_dir=corpus_dir,
            questions=questions,
            context_budget=args.context_budget,
            call_llm=args.stuffing_call_llm,
            memory_scale=args.memory_scale,
            **progress_kwargs,
        )
        single_stuffing_summary = stuffing_data["summary"]
        if args.memory_scale:
            single_stuffing_summary["memory_scale"] = args.memory_scale

    all_results, all_summaries = await _run_benchmark_session(
        corpus_dir=corpus_dir,
        questions=questions,
        variant_names=variant_names,
        refine=refine,
        skip_index=args.skip_index,
        run_curator=args.run_curator,
        warm_graph=args.warm_graph,
        memory_scale=args.memory_scale,
        **progress_kwargs,
    )

    print("\n" + format_table(all_summaries))
    if args.print_slices:
        print("\n### Per-query-type slices (memory / time / haystack)\n")
        print(format_retention_slices_table(all_summaries))
    if args.compare_stuffing and single_stuffing_summary:
        print("\n")
        print(format_comparison_table([single_stuffing_summary], all_summaries))
        if args.stuffing_call_llm:
            print("\n")
            print(format_full_comparison_table([single_stuffing_summary], all_summaries))

    if args.output:
        output_data = {
            "benchmark_meta": {
                "hypotheses": HYPOTHESES,
                "memory_scale": args.memory_scale,
                "corpus_dir": corpus_dir,
                "questions_path": questions_path,
                "variants": variant_names,
                "context_budget": args.context_budget if args.compare_stuffing else None,
            },
            "summaries": all_summaries,
            "comparison_table": format_table(all_summaries),
            "full_results": {k: v["results"] for k, v in all_results.items()},
        }
        if args.print_slices:
            output_data["retention_slices_table"] = format_retention_slices_table(all_summaries)
        if args.compare_stuffing and single_stuffing_summary:
            output_data["stuffing_summary"] = single_stuffing_summary
            output_data["stuffing_summaries"] = [single_stuffing_summary]
            output_data["breakeven_table"] = format_comparison_table(
                [single_stuffing_summary], all_summaries
            )
            if args.stuffing_call_llm:
                output_data["full_comparison_table"] = format_full_comparison_table(
                    [single_stuffing_summary], all_summaries
                )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
