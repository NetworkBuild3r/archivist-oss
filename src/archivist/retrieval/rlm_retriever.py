"""RLM-inspired recursive retriever — coarse search, threshold filter, rerank, LLM refinement, synthesis.

Phase 1: threshold, rerank pipeline.
Phase 2 (v0.5): graph-augmented retrieval, temporal decay, tiered context, multi-hop.
Phase 3 (v0.6): outcome-aware scoring from trajectory history.
Phase 4 (v0.7): memory_type routing.
Phase 5 (v0.8): hot cache, retrieval trajectory logging.
"""

import asyncio
import logging
import time

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchText,
    MatchValue,
    SearchParams,
)

import archivist.core.health as health
import archivist.core.metrics as m
import archivist.retrieval.hot_cache as hot_cache
import archivist.retrieval.retrieval_log as retrieval_log
from archivist.core.config import (
    ADAPTIVE_VECTOR_LIMIT_ENABLED,
    ADAPTIVE_VECTOR_LIMIT_MULTIPLIER,
    ADAPTIVE_VECTOR_MIN_RESULTS,
    BM25_ENABLED,
    BM25_RESCUE_ENABLED,
    BM25_RESCUE_MAX_SLOTS,
    BM25_RESCUE_MIN_SCORE_RATIO,
    CROSS_AGENT_MAX_SHARE,
    DYNAMIC_THRESHOLD_ENABLED,
    GRAPH_RETRIEVAL_ENABLED,
    IMPORTANCE_WEIGHT,
    LLM_MODEL,
    LLM_REFINE_CONCURRENCY,
    LLM_REFINE_MODEL,
    LLM_SYNTH_MODEL,
    MULTI_HOP_DEPTH,
    QDRANT_SEARCH_EF,
    QUERY_CLASSIFICATION_ENABLED,
    QUERY_EXPANSION_COUNT,
    QUERY_EXPANSION_ENABLED,
    QUERY_EXPANSION_MODEL,
    REFINE_SKIP_THRESHOLD,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_TOP_K,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_TOP_K,
    RETRIEVAL_THRESHOLD,
    SYNTHETIC_QUESTIONS_ENABLED,
    TEMPORAL_DECAY_HALFLIFE_DAYS,
    TEMPORAL_INTENT_ENABLED,
    TOPIC_ROUTING_ENABLED,
    VECTOR_SEARCH_LIMIT,
)
from archivist.core.hotness import apply_hotness_to_results
from archivist.core.latency_budget import LatencyBudget, budget_for_query_type
from archivist.core.observability import slow_qdrant_check
from archivist.core.result_types import ResultCandidate
from archivist.core.trajectory import get_outcome_adjustments
from archivist.features.embeddings import embed_batch, embed_text
from archivist.features.llm import llm_query
from archivist.retrieval.memory_fusion import dedupe_vector_hits
from archivist.retrieval.query_classifier import SUBCATEGORY_TO_TOPIC, classify_query_full
from archivist.retrieval.query_expansion import expand_query
from archivist.retrieval.query_intent import classify_temporal_intent
from archivist.retrieval.rank_fusion import rrf_merge
from archivist.retrieval.ranker import ltr_available
from archivist.retrieval.ranker import rank_results as ltr_rank_results
from archivist.retrieval.retrieval_filters import apply_dynamic_threshold, apply_retrieval_threshold
from archivist.retrieval.topic_detector import detect_query_topic
from archivist.storage.collection_router import collection_for
from archivist.storage.fts_search import merge_vector_and_bm25, search_bm25
from archivist.storage.graph import lookup_needle_tokens
from archivist.storage.graph_retrieval import (
    apply_temporal_decay,
    build_entity_fact_results,
    extract_entity_mentions,
    graph_context_for_entities,
    merge_graph_context_into_results,
)
from archivist.storage.namespace_inventory import NamespaceInventory, get_inventory
from archivist.storage.qdrant import qdrant_client
from archivist.utils.chunking import NEEDLE_PATTERNS as LITERAL_NEEDLE_PATTERNS
from archivist.utils.tokenizer import count_tokens
from archivist.write.hyde import generate_hypothetical_document, is_needle_query
from archivist.write.tiering import select_tier

logger = logging.getLogger("archivist.rlm")


def _extract_literal_tokens(query: str) -> list[str]:
    """Extract high-specificity tokens from query for literal substring search."""
    tokens: list[str] = []
    for pat in LITERAL_NEEDLE_PATTERNS:
        for match in pat.finditer(query):
            tok = match.group().strip()
            if tok and len(tok) >= 3:
                tokens.append(tok)
    return tokens


def _literal_search_sync(
    tokens: list[str],
    namespace: str = "",
    agent_id: str = "",
    agent_ids: list[str] | None = None,
    memory_type: str = "",
    limit: int = 10,
) -> list[dict]:
    """Exact substring search on Qdrant ``text`` payload via MatchText filter.

    Only triggered when the query contains high-specificity tokens (IPs, cron, UUIDs, etc.).
    Uses a scroll (no vector) so results are unscored — they get a synthetic high score.
    """
    if not tokens:
        return []

    client = qdrant_client()
    _coll = collection_for(namespace)
    all_hits: list[dict] = []
    seen_ids: set[str] = set()

    for token in tokens[:3]:
        must_filters = [FieldCondition(key="text", match=MatchText(text=token))]
        if agent_ids:
            ids = [a for a in agent_ids if a]
            if ids:
                must_filters.append(FieldCondition(key="agent_id", match=MatchAny(any=ids)))
        elif agent_id:
            must_filters.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
        if namespace:
            must_filters.append(FieldCondition(key="namespace", match=MatchValue(value=namespace)))
        if memory_type:
            must_filters.append(
                FieldCondition(key="memory_type", match=MatchValue(value=memory_type))
            )

        literal_must_not = [
            FieldCondition(key="archived", match=MatchValue(value=True)),
            FieldCondition(key="deleted", match=MatchValue(value=True)),
        ]
        try:
            scrolled, _ = client.scroll(
                collection_name=_coll,
                scroll_filter=Filter(must=must_filters, must_not=literal_must_not),
                limit=limit,
                with_payload=True,
            )
        except Exception:
            continue

        for hit in scrolled:
            hid = str(getattr(hit, "id", ""))
            if hid in seen_ids:
                continue
            seen_ids.add(hid)
            p = hit.payload or {}
            all_hits.append(
                {
                    "id": hid,
                    "score": 0.0,
                    "text": p.get("text", ""),
                    "l0": p.get("l0", ""),
                    "l1": p.get("l1", ""),
                    "agent_id": p.get("agent_id", ""),
                    "file_path": p.get("file_path", ""),
                    "file_type": p.get("file_type", ""),
                    "date": p.get("date", ""),
                    "content_date": p.get("content_date", ""),
                    "indexed_at": p.get("indexed_at", ""),
                    "team": p.get("team", ""),
                    "namespace": p.get("namespace", ""),
                    "chunk_index": p.get("chunk_index", 0),
                    "parent_id": p.get("parent_id"),
                    "is_parent": p.get("is_parent", False),
                    "parent_text": p.get("parent_text", ""),
                    "importance_score": p.get("importance_score", 0.5),
                    "retention_class": p.get("retention_class", "standard"),
                    "topic": p.get("topic", ""),
                    "thought_type": p.get("thought_type", ""),
                    "literal_match": True,
                }
            )
    return all_hits[:limit]


def _retrieval_trace(
    *,
    vector_limit: int,
    coarse_count: int,
    deduped_count: int,
    threshold: float,
    after_threshold_count: int,
    after_rerank_count: int,
    parent_enriched: bool,
    refinement_chunks: int,
    graph_entities_found: int = 0,
    graph_context_items: int = 0,
    temporal_decay_applied: bool = False,
    tier: str = "l2",
    outcome_adjustments: int = 0,
    context_status: dict | None = None,
    bm25_hits: int = 0,
    entity_facts_injected: int = 0,
    bm25_rescue_count: int = 0,
    adaptive_widened: bool = False,
    cross_agent_capped: bool = False,
    **extra,
) -> dict:
    """Build a retrieval trace dict for observability.

    Core pipeline fields are explicit parameters for type safety.
    Additional fields (auto_subcategory, detected_topic, etc.) flow
    through **extra so callers can add new trace keys without modifying
    this signature.
    """
    trace = {
        "vector_search_limit": vector_limit,
        "coarse_hits": coarse_count,
        "bm25_enabled": BM25_ENABLED,
        "bm25_hits": bm25_hits,
        "after_dedupe": deduped_count,
        "threshold": threshold,
        "after_threshold": after_threshold_count,
        "rerank_enabled": RERANK_ENABLED,
        "rerank_model": RERANK_MODEL if RERANK_ENABLED else None,
        "after_rerank": after_rerank_count,
        "parent_enrichment": parent_enriched,
        "chunks_sent_to_refinement": refinement_chunks,
        "rerank_top_k": RERANK_TOP_K if RERANK_ENABLED else None,
        "graph_retrieval_enabled": GRAPH_RETRIEVAL_ENABLED,
        "graph_entities_found": graph_entities_found,
        "graph_context_items": graph_context_items,
        "entity_facts_injected": entity_facts_injected,
        "temporal_decay_applied": temporal_decay_applied,
        "tier": tier,
        "outcome_adjustments": outcome_adjustments,
    }
    if extra.get("stage_timings"):
        trace["stage_timings"] = extra.pop("stage_timings")
    if context_status:
        trace["context_status"] = context_status
    if bm25_rescue_count:
        trace["bm25_rescue_count"] = bm25_rescue_count
    if adaptive_widened:
        trace["adaptive_widened"] = True
    if cross_agent_capped:
        trace["cross_agent_capped"] = True
    # Merge caller/_trace_kw() extensions; omit empty placeholders so the trace stays compact.
    # Preserve explicit False for observability booleans (otherwise refine_skipped=False is dropped).
    _trace_keep_false = frozenset({"refine_skipped", "refine_parallel"})
    for k, v in extra.items():
        if v is None or v == "":
            continue
        if v is False and k not in _trace_keep_false:
            continue
        trace[k] = v
    return trace


def _memory_awareness_payload(
    inventory: NamespaceInventory,
    auto_type: str,
    user_memory_type: str,
) -> dict:
    searched_as = auto_type if auto_type else (user_memory_type if user_memory_type else "all")
    hint = ""
    by = inventory.by_type
    if searched_as == "skill" and by.get("experience", 0) > 0:
        hint = (
            f"Also {by['experience']} experience memories in this namespace — "
            "try memory_type=experience for 'what happened' queries."
        )
    elif searched_as == "experience" and by.get("skill", 0) > 0:
        hint = f"Also {by['skill']} skill memories — try memory_type=skill for how-to questions."
    elif searched_as in ("skill", "experience") and by.get("general", 0) > 0:
        hint = f"Also {by['general']} general memories available."
    out = {
        "searched_as": searched_as,
        "namespace_inventory": dict(by),
        "total_memories": inventory.total_memories,
        "top_entities": list(inventory.top_entities[:10]),
        "has_fleet_tips": inventory.has_fleet_tips,
    }
    if hint:
        out["hint"] = hint
    return out


def _attach_stage0(
    result: dict,
    inventory: NamespaceInventory | None,
    auto_type: str,
    user_memory_type: str,
) -> None:
    if inventory is None:
        return
    result["memory_awareness"] = _memory_awareness_payload(inventory, auto_type, user_memory_type)
    rt = result.get("retrieval_trace")
    if isinstance(rt, dict):
        if auto_type:
            rt["auto_classified_type"] = auto_type
        rt["inventory_total"] = inventory.total_memories


REFINE_SYSTEM = (
    "You are a memory retrieval assistant. Given a search query and a memory chunk, "
    "extract ONLY the parts that are directly relevant to the query. "
    "If the chunk contains nothing relevant, respond with exactly: IRRELEVANT\n"
    "Be concise. Return just the relevant facts or insights, no commentary."
)

SYNTHESIZE_SYSTEM = (
    "You are a memory synthesis assistant. Given multiple extracted facts from an agent's "
    "memory files, synthesize a coherent, deduplicated answer. "
    "Preserve specific details (dates, names, numbers). "
    "If facts contradict each other, note the contradiction with dates. "
    "Be concise and factual."
)

SYNTHESIZE_MULTI_AGENT = (
    SYNTHESIZE_SYSTEM
    + " When several agents contributed memories, attribute each major claim to the agent "
    "(by name or id) whose memory it came from."
)

_refine_sem = asyncio.Semaphore(LLM_REFINE_CONCURRENCY)


async def _refine_one_chunk(
    hit: dict,
    query: str,
    tier: str,
    refine_model: str,
) -> dict | None:
    """Run LLM refinement for one chunk. Concurrency is capped by ``_refine_sem``."""
    async with _refine_sem:
        context = select_tier(hit, tier)
        parent_ctx = hit.get("parent_text", "")
        if parent_ctx and tier == "l2":
            context = f"[Parent context]\n{parent_ctx}\n\n[Matched chunk]\n{context}"

        graph_extra = ""
        if hit.get("graph_context"):
            graph_extra = "\n[Graph context] " + " | ".join(hit["graph_context"][:3])

        who = hit.get("agent_id") or "unknown"
        prompt = f"Query: {query}\n\nMemory chunk (agent={who}, file={hit['file_path']}, date={hit['date']}):\n{context}{graph_extra}"
        try:
            extraction = await llm_query(
                prompt,
                system=REFINE_SYSTEM,
                max_tokens=512,
                model=refine_model,
                stage="refine",
            )
            if extraction.strip().upper() != "IRRELEVANT":
                return {
                    "extraction": extraction.strip(),
                    "source": hit["file_path"],
                    "date": hit["date"],
                    "agent_id": hit["agent_id"],
                    "score": hit["score"],
                    "rerank_score": hit.get("rerank_score"),
                }
        except Exception as e:
            logger.warning("LLM refinement failed for chunk: %s", e)
        return None


def _merge_into_results(
    filtered: list[dict],
    new_items: list[dict],
    *,
    min_score: float = 0.0,
    preserve_top_n: int = 0,
    tag: str = "",
) -> tuple[list[dict], int]:
    """Merge new_items into filtered results with dedup, floor scoring, and top-N preservation.

    Centralises the merge logic used by BM25 rescue, entity fact injection,
    and adaptive widen so each caller doesn't re-implement (and mis-implement)
    the same pattern.

    Args:
        filtered: Existing ranked results (mutated in place for efficiency).
        new_items: Items to merge in.
        min_score: Floor score for new items that lack a valid score.
        preserve_top_n: If > 0, the top N non-injected items from filtered
                        are guaranteed to remain in the top N slots after merge.
        tag: If set, added as a metadata key (e.g. "bm25_rescue", "entity_fact").

    Returns:
        (merged_results, count_added)
    """
    if not new_items:
        return filtered, 0

    winners = []
    if preserve_top_n > 0:
        winners = [r for r in filtered if r.get("file_type") != "entity_fact"][:preserve_top_n]

    existing_ids = {r.get("id") for r in filtered if r.get("id")}
    existing_texts = {r.get("text", "")[:100] for r in filtered}
    added = 0

    for item in new_items:
        item_id = item.get("id")
        item_text = item.get("text", "")[:100]

        if (item_id and item_id in existing_ids) or item_text in existing_texts:
            continue

        if item.get("score", 0) < min_score:
            item["score"] = min_score

        if tag:
            item[tag] = True

        filtered.append(item)
        if item_id:
            existing_ids.add(item_id)
        existing_texts.add(item_text)
        added += 1

    if added:
        filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Keep pre-merge top vector rows first (by object identity), then the rest — avoids a second sort undoing preservation.
    if winners and added:
        winner_ids = {id(w) for w in winners}
        rest = [r for r in filtered if id(r) not in winner_ids]
        filtered = winners + rest

    return filtered, added


async def search_vectors(
    query: str,
    agent_id: str = "",
    agent_ids: list[str] | None = None,
    team: str = "",
    namespace: str = "",
    file_type: str = "",
    date_from: str = "",
    date_to: str = "",
    memory_type: str = "",
    topic: str = "",
    thought_type: str = "",
    limit: int = 20,
    _query_vec: list[float] | None = None,
    actor_type: str = "",
) -> list[dict]:
    """Stage 1: coarse vector search in Qdrant with optional filters.

    Pass ``_query_vec`` to reuse a pre-computed embedding and avoid
    redundant calls to the embedding API (topic fallback, adaptive widen).
    """
    if _query_vec is not None:
        query_vec = _query_vec
    else:
        try:
            query_vec = await embed_text(query)
        except Exception as e:
            logger.warning("Embedding failed, skipping vector search: %s", e)
            return []
    client = qdrant_client()

    must_filters = []
    if agent_ids:
        ids = [a for a in agent_ids if a]
        if ids:
            must_filters.append(FieldCondition(key="agent_id", match=MatchAny(any=ids)))
    elif agent_id:
        must_filters.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
    if team:
        must_filters.append(FieldCondition(key="team", match=MatchValue(value=team)))
    if namespace:
        must_filters.append(FieldCondition(key="namespace", match=MatchValue(value=namespace)))
    if file_type:
        must_filters.append(FieldCondition(key="file_type", match=MatchValue(value=file_type)))
    if date_from and date_from == date_to:
        must_filters.append(FieldCondition(key="date", match=MatchValue(value=date_from)))
    if memory_type:
        must_filters.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))
    if topic:
        must_filters.append(FieldCondition(key="topic", match=MatchValue(value=topic)))
    if thought_type:
        must_filters.append(
            FieldCondition(key="thought_type", match=MatchValue(value=thought_type))
        )
    if actor_type:
        must_filters.append(FieldCondition(key="actor_type", match=MatchValue(value=actor_type)))

    _date_range_active = bool(date_from or date_to) and date_from != date_to
    must_not_filters = [
        FieldCondition(key="archived", match=MatchValue(value=True)),
        FieldCondition(key="deleted", match=MatchValue(value=True)),
    ]
    if not SYNTHETIC_QUESTIONS_ENABLED:
        must_not_filters.append(
            FieldCondition(key="representation_type", match=MatchValue(value="synthetic_question")),
        )
    search_filter = Filter(must=must_filters, must_not=must_not_filters)
    fetch_limit = limit * 3 if _date_range_active else limit

    _target_collection = collection_for(namespace)

    # Degrade to [] on Qdrant/network errors so retrieval can fall back to BM25-only paths.
    t_q = time.monotonic()
    try:
        results = client.query_points(
            collection_name=_target_collection,
            query=query_vec,
            query_filter=search_filter,
            limit=fetch_limit,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=QDRANT_SEARCH_EF),
        ).points
    except Exception as e:
        qdur_ms = (time.monotonic() - t_q) * 1000
        m.observe(m.QDRANT_QUERY_DURATION, qdur_ms)
        slow_qdrant_check(qdur_ms)
        health.register("qdrant", healthy=False, detail=str(e))
        return []

    qdur_ms = (time.monotonic() - t_q) * 1000
    m.observe(m.QDRANT_QUERY_DURATION, qdur_ms)
    slow_qdrant_check(qdur_ms)

    # After a prior failure, mark Qdrant healthy again on first successful query.
    if not health.is_healthy("qdrant"):
        health.register("qdrant", healthy=True)

    rows = [
        {
            "id": getattr(hit, "id", ""),
            "score": hit.score,
            "text": hit.payload.get("text", ""),
            "l0": hit.payload.get("l0", ""),
            "l1": hit.payload.get("l1", ""),
            "agent_id": hit.payload.get("agent_id", ""),
            "file_path": hit.payload.get("file_path", ""),
            "file_type": hit.payload.get("file_type", ""),
            "date": hit.payload.get("date", ""),
            "content_date": hit.payload.get("content_date", ""),
            "indexed_at": hit.payload.get("indexed_at", ""),
            "team": hit.payload.get("team", ""),
            "namespace": hit.payload.get("namespace", ""),
            "chunk_index": hit.payload.get("chunk_index", 0),
            "parent_id": hit.payload.get("parent_id"),
            "is_parent": hit.payload.get("is_parent", False),
            "parent_text": hit.payload.get("parent_text", ""),
            "importance_score": hit.payload.get("importance_score", 0.5),
            "retention_class": hit.payload.get("retention_class", "standard"),
            "topic": hit.payload.get("topic", ""),
            "thought_type": hit.payload.get("thought_type", ""),
            "representation_type": hit.payload.get("representation_type", "chunk"),
            "synthetic_question": hit.payload.get("synthetic_question", ""),
            "source_memory_id": hit.payload.get("source_memory_id", ""),
        }
        for hit in results
    ]

    if _date_range_active:
        if date_from:
            rows = [r for r in rows if r["date"] >= date_from]
        if date_to:
            rows = [r for r in rows if r["date"] <= date_to]
        rows = rows[:limit]

    return rows


def apply_importance_to_results(results: list[dict], weight: float | None = None) -> list[dict]:
    """Boost/penalize retrieval results based on importance_score.

    Normalised around the default (0.5) so standard memories are unaffected:
      importance 0.0 → score × 0.90  (de-prioritised)
      importance 0.5 → score × 1.00  (neutral default)
      importance 1.0 → score × 1.10  (boosted — pinned/critical)
    """
    w = weight if weight is not None else IMPORTANCE_WEIGHT
    if w <= 0 or not results:
        return results

    for r in results:
        imp = r.get("importance_score", 0.5)
        if imp != 0.5:
            boost = (imp - 0.5) * 2
            r["score"] = r.get("score", 0) * (1.0 + w * boost)

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


async def _apply_rerank(query: str, results: list[dict]) -> list[dict]:
    """Apply cross-encoder reranking if enabled."""
    if not RERANK_ENABLED:
        return results
    try:
        from archivist.retrieval.reranker import rerank_results

        return await rerank_results(query, results, model_name=RERANK_MODEL, limit=RERANK_TOP_K)
    except Exception as e:
        logger.warning("Reranking failed, using original order: %s", e)
        return results


def _observe_search_results(namespace: str, sources: list | None) -> None:
    """Record ``archivist_search_results`` (histogram of len(sources); label namespace)."""
    m.observe(m.SEARCH_RESULTS, float(len(sources or [])), {"namespace": namespace or "_default"})


async def recursive_retrieve(
    query: str,
    agent_id: str = "",
    agent_ids: list[str] | None = None,
    team: str = "",
    namespace: str = "",
    limit: int = 20,
    refine: bool = True,
    threshold: float | None = None,
    tier: str = "l2",
    date_from: str = "",
    date_to: str = "",
    max_tokens: int | None = None,
    memory_type: str = "",
    _is_retry: bool = False,
    actor_type: str = "",
) -> dict:
    """Full RLM pipeline: coarse → dedupe → graph augment → temporal decay → threshold → rerank → parent → refine → synthesize."""
    t0 = time.monotonic()
    user_memory_type = memory_type
    stage0_inventory = None
    auto_type = ""
    auto_subcategory = ""
    # Only user-specified memory_type is used as a hard filter on
    # vector/BM25 search.  Auto-classified type is recorded for tracing
    # and scoring but must NOT narrow the search or it silently drops
    # memories stored under a different type (the "general" recall bug).
    effective_memory_type = memory_type
    _initial_budget_type = "needle" if is_needle_query(query) else "default"
    _budget = LatencyBudget(max_ms=budget_for_query_type(_initial_budget_type))

    if QUERY_CLASSIFICATION_ENABLED and not memory_type:
        if namespace:
            stage0_inventory = get_inventory(namespace)
        auto_type, auto_subcategory = await classify_query_full(
            query,
            inventory=stage0_inventory,
        )
        _classified_budget_type = auto_type if auto_type else _initial_budget_type
        _new_budget_ms = budget_for_query_type(_classified_budget_type)
        if _new_budget_ms != _budget._max_ms:
            _budget._max_ms = _new_budget_ms

    temporal_intent = ""
    if TEMPORAL_INTENT_ENABLED:
        temporal_intent = classify_temporal_intent(query)

    detected_topic = ""
    topic_fallback_used = False
    if TOPIC_ROUTING_ENABLED:
        detected_topic = detect_query_topic(query)
        if not detected_topic and auto_subcategory:
            detected_topic = SUBCATEGORY_TO_TOPIC.get(auto_subcategory, "")

    effective_threshold = threshold if threshold is not None else RETRIEVAL_THRESHOLD
    _needle_query_detected = is_needle_query(query)
    if _needle_query_detected and threshold is None:
        effective_threshold = max(0.25, effective_threshold - 0.15)
    vector_limit = max(VECTOR_SEARCH_LIMIT, limit)
    n_graph_entities = 0
    n_graph_items = 0
    n_bm25_rescue = 0
    n_synthetic_hits = 0
    was_adaptive_widened = False
    was_cross_agent_capped = False

    def _trace_kw() -> dict:
        return {
            "auto_classified_type": auto_type,
            "auto_subcategory": auto_subcategory,
            "inventory_total": stage0_inventory.total_memories if stage0_inventory else None,
            "temporal_intent": temporal_intent,
            "detected_topic": detected_topic,
            "topic_fallback_used": topic_fallback_used,
            "bm25_rescue_count": n_bm25_rescue,
            "adaptive_widened": was_adaptive_widened,
            "cross_agent_capped": was_cross_agent_capped,
            "query_expansion_variants": n_expansion_variants,
            "dynamic_threshold_enabled": DYNAMIC_THRESHOLD_ENABLED,
            "ltr_used": _ltr_used,
            "hyde_used": _hyde_used,
            "needle_query_detected": _needle_query_detected,
            "needle_registry_hits": len(_registry_hits),
            "synthetic_hits": n_synthetic_hits,
            "iterative_retry": _is_retry,
            "latency_budget_ms": _budget.remaining_ms(),
        }

    cache_extra = (
        f"{agent_id}|{','.join(agent_ids or [])}|{team}|{date_from}|{date_to}|{limit}|{refine}"
    )
    cached = hot_cache.get(
        agent_id or "fleet",
        query,
        namespace=namespace,
        tier=tier,
        memory_type=effective_memory_type,
        extra=cache_extra,
    )
    if cached is not None:
        elapsed = int((time.monotonic() - t0) * 1000)
        out = dict(cached)
        out["retrieval_trace"] = dict(cached.get("retrieval_trace", {}))
        _attach_stage0(
            out,
            stage0_inventory,
            out["retrieval_trace"].get("auto_classified_type") or auto_type,
            user_memory_type,
        )
        retrieval_log.log_retrieval(
            agent_id=agent_id or "fleet",
            query=query,
            namespace=namespace,
            tier=tier,
            memory_type=effective_memory_type,
            retrieval_trace=out.get("retrieval_trace", {}),
            result_count=len(out.get("sources", [])),
            cache_hit=True,
            duration_ms=elapsed,
        )
        out["cache_hit"] = True
        m.inc(m.CACHE_HIT)
        m.inc(m.SEARCH_TOTAL)
        m.observe(m.SEARCH_DURATION, elapsed)
        _observe_search_results(namespace, out.get("sources"))
        return out

    # ── Stage timings (ms) for retrieval trace ──
    _stage_timings: dict[str, float] = {}
    n_expansion_variants = 0
    _ltr_used = False
    _hyde_used = False

    # ── Deterministic needle registry lookup (v2.0 — O(1) exact match) ──
    _registry_hits: list[dict] = []
    try:
        _raw_registry: list[dict] = []
        if agent_ids:
            _seen_mem = set()
            for aid in agent_ids:
                for rh in lookup_needle_tokens(query, namespace=namespace, agent_id=aid):
                    if rh["memory_id"] not in _seen_mem:
                        _seen_mem.add(rh["memory_id"])
                        _raw_registry.append(rh)
        else:
            _raw_registry = lookup_needle_tokens(query, namespace=namespace, agent_id=agent_id)

        if _raw_registry:
            m.inc(m.NEEDLE_REGISTRY_HITS, {"namespace": namespace}, value=len(_raw_registry))
            _reg_candidates = [ResultCandidate.from_registry_hit(rh) for rh in _raw_registry]

            _reg_ids = [c.id for c in _reg_candidates if c.id]
            if _reg_ids:
                _reg_coll = collection_for(namespace)
                try:
                    _reg_points = qdrant_client().retrieve(
                        collection_name=_reg_coll,
                        ids=_reg_ids,
                        with_payload=True,
                    )
                    _reg_payload_map = {str(p.id): p.payload for p in _reg_points}
                except Exception as e:
                    logger.warning("Registry payload refresh failed: %s", e)
                    _reg_payload_map = {}

                _live_candidates = []
                for c in _reg_candidates:
                    payload = _reg_payload_map.get(c.id)
                    if payload:
                        if payload.get("archived") or payload.get("deleted"):
                            m.inc(m.NEEDLE_REGISTRY_STALE, {"namespace": namespace})
                            logger.debug(
                                "registry.excluded: memory_id=%s is archived/deleted — dropping",
                                c.id,
                            )
                            continue
                        c.update_from_payload(payload)
                        _live_candidates.append(c)
                    else:
                        m.inc(m.NEEDLE_REGISTRY_STALE, {"namespace": namespace})
                        logger.warning(
                            "registry.stale_entry: memory_id=%s not found in Qdrant — dropping",
                            c.id,
                        )
                _reg_candidates = _live_candidates

            _registry_hits = [c.to_dict() for c in _reg_candidates]
            for rh in _registry_hits:
                rh["needle_registry_hit"] = True
    except Exception as e:
        logger.debug("Needle registry lookup failed: %s", e)

    # ── Multi-query expansion + embedding (v1.10) ──
    # Budget-gated: expansion requires ~150ms LLM call
    _t_embed = time.monotonic()
    query_variants: list[str] = [query]
    if QUERY_EXPANSION_ENABLED and _budget.can_afford(150):
        try:
            query_variants = await expand_query(
                query,
                count=QUERY_EXPANSION_COUNT,
                model=QUERY_EXPANSION_MODEL,
            )
            n_expansion_variants = len(query_variants) - 1
        except Exception as e:
            logger.warning("Query expansion failed: %s — using original query", e)

    # ── HyDE: hypothetical document embedding for needle queries (v1.10) ──
    # Budget-gated: HyDE requires ~100ms LLM call
    hyde_doc: str | None = None
    if not _is_retry and is_needle_query(query) and _budget.can_afford(100):
        try:
            hyde_doc = await generate_hypothetical_document(query)
            if hyde_doc:
                query_variants.append(hyde_doc)
                _hyde_used = True
        except Exception as e:
            logger.warning("HyDE failed: %s", e)

    # Embed all query variants in parallel (cache deduplicates repeat embeds)
    query_vecs: list[list[float] | None] = []
    try:
        query_vecs = await embed_batch(query_variants)
    except Exception as e:
        logger.warning("Embedding failed: %s — skipping vector search", e)
        query_vecs = [None] * len(query_variants)
    _cached_query_vec = query_vecs[0] if query_vecs else None
    _stage_timings["embed_ms"] = round((time.monotonic() - _t_embed) * 1000, 1)

    # Stage 1: Parallel retrieval — vector + BM25 run concurrently (v1.10 12c)
    _t_vec = time.monotonic()
    _vec_common = dict(
        agent_id=agent_id,
        agent_ids=agent_ids,
        team=team,
        namespace=namespace,
        date_from=date_from,
        date_to=date_to,
        memory_type=effective_memory_type,
        limit=vector_limit,
        actor_type=actor_type,
    )

    async def _search_one(q: str, vec: list[float] | None, topic: str = "") -> list[dict]:
        if vec is None:
            return []
        return await search_vectors(q, **_vec_common, topic=topic, _query_vec=vec)

    async def _bm25_async() -> list[dict]:
        """Wrap synchronous BM25/SQLite search for concurrent execution."""
        if not BM25_ENABLED:
            return []
        return await asyncio.to_thread(
            search_bm25,
            query,
            namespace=namespace,
            agent_id=agent_id if not agent_ids else "",
            memory_type=effective_memory_type,
            limit=vector_limit,
            actor_type=actor_type,
        )

    # Launch ALL search paths concurrently: vector variants + BM25 + literal
    vec_tasks = [_search_one(q, v, detected_topic) for q, v in zip(query_variants, query_vecs)]

    async def _literal_async() -> list[dict]:
        literal_tokens = _extract_literal_tokens(query)
        if not literal_tokens:
            return []
        return await asyncio.to_thread(
            _literal_search_sync,
            literal_tokens,
            namespace=namespace,
            agent_id=agent_id if not agent_ids else "",
            agent_ids=agent_ids,
            memory_type=effective_memory_type,
            limit=vector_limit,
        )

    all_tasks = vec_tasks + [_bm25_async(), _literal_async()]
    all_results = await asyncio.gather(*all_tasks)

    vec_results = list(all_results[: len(vec_tasks)])
    bm25_hits = all_results[len(vec_tasks)]
    literal_hits = all_results[len(vec_tasks) + 1]

    # If topic-filtered primary returned too few, retry without topic filter
    if detected_topic and len(vec_results[0]) < ADAPTIVE_VECTOR_MIN_RESULTS and _cached_query_vec:
        topic_fallback_used = True
        vec_results[0] = await search_vectors(
            query,
            **_vec_common,
            topic="",
            _query_vec=_cached_query_vec,
        )

    # Merge all vector variant results + literal + registry via RRF
    # v2 path (RERANKER_ENABLED) does its own ID-based dedup from raw sources
    # so we only do RRF merge for the legacy path; both paths still compute n_coarse.
    non_empty_rankings = [r for r in vec_results if r]
    if literal_hits:
        non_empty_rankings.append(literal_hits)
    if _registry_hits:
        non_empty_rankings.append(_registry_hits)

    if RERANKER_ENABLED:
        coarse = [item for sublist in non_empty_rankings for item in sublist]
    elif len(non_empty_rankings) > 1:
        coarse = rrf_merge(non_empty_rankings, k=20)
        for r in coarse:
            if "score" not in r or r.get("rrf_score", 0) > r.get("score", 0):
                r["score"] = r["rrf_score"]
    elif non_empty_rankings:
        coarse = non_empty_rankings[0]
    else:
        coarse = []

    _stage_timings["vector_ms"] = round((time.monotonic() - _t_vec) * 1000, 1)
    n_coarse = len(coarse)

    n_bm25 = len(bm25_hits)
    if bm25_hits and not RERANKER_ENABLED:
        coarse = merge_vector_and_bm25(coarse, bm25_hits)
    _stage_timings["bm25_ms"] = round((time.monotonic() - _t_vec) * 1000, 1)

    logger.debug(
        "POST-FUSION coarse=%d bm25=%d top5_scores=%s top3_paths=%s",
        len(coarse),
        n_bm25,
        [round(r.get("score", 0), 4) for r in coarse[:5]],
        [r.get("file_path", "?") for r in coarse[:3]],
    )

    # Stage 1a: Graph augmentation (v0.5)
    detected_entities: list[dict] = []
    _t_graph = time.monotonic()
    if GRAPH_RETRIEVAL_ENABLED:
        entities = extract_entity_mentions(query, namespace=namespace)
        detected_entities = entities
        n_graph_entities = len(entities)
        if entities:
            graph_items = graph_context_for_entities(
                [e["id"] for e in entities],
                depth=MULTI_HOP_DEPTH,
            )
            n_graph_items = len(graph_items)
            if graph_items and not RERANKER_ENABLED:
                coarse = merge_graph_context_into_results(coarse, graph_items)
    _stage_timings["graph_ms"] = round((time.monotonic() - _t_graph) * 1000, 1)

    # Late HyDE pass — legacy path only; v2 path relies on the nomination pool
    if (
        not RERANKER_ENABLED
        and not _is_retry
        and not _hyde_used
        and is_needle_query(query, entity_count=n_graph_entities)
        and _budget.can_afford(150)
        and _cached_query_vec is not None
    ):
        try:
            late_hyde_doc = await generate_hypothetical_document(query)
            if late_hyde_doc:
                late_vec = (await embed_batch([late_hyde_doc]))[0]
                if late_vec:
                    late_hits = await search_vectors(
                        query, **_vec_common, topic="", _query_vec=late_vec
                    )
                    if late_hits:
                        coarse, _ = _merge_into_results(coarse, late_hits)
                    _hyde_used = True
        except Exception as e:
            logger.debug("Late HyDE pass failed: %s", e)

    if not coarse:
        empty = {
            "answer": "No relevant memories found.",
            "sources": [],
            "chunks_searched": 0,
            "agents_scoped": agent_ids or ([agent_id] if agent_id else []),
            "retrieval_trace": _retrieval_trace(
                vector_limit=vector_limit,
                coarse_count=0,
                deduped_count=0,
                threshold=effective_threshold,
                after_threshold_count=0,
                after_rerank_count=0,
                parent_enriched=False,
                refinement_chunks=0,
                graph_entities_found=n_graph_entities,
                graph_context_items=n_graph_items,
                tier=tier,
                bm25_hits=n_bm25,
                stage_timings=_stage_timings,
                **_trace_kw(),
            ),
        }
        _attach_stage0(empty, stage0_inventory, auto_type, user_memory_type)
        _observe_search_results(namespace, [])
        return empty

    # ── Post-retrieval processing ──
    _t_post = time.monotonic()

    if RERANKER_ENABLED:
        # ══════════════════════════════════════════════════════════════════
        # v2 CLEAN PATH: nominate → ID-dedupe → parent enrich → rerank → top-K
        # The cross-encoder is the single source of truth for ranking.
        # No RRF, no threshold, no temporal decay, no hotness, no rescue.
        # ══════════════════════════════════════════════════════════════════

        # Nomination: collect ALL candidates into a pool, dedupe by Qdrant point ID.
        # Sources were already gathered in parallel above (vec_results, bm25_hits,
        # literal_hits, _registry_hits).  We flatten everything into one list
        # and keep the best score per unique point ID.
        candidate_pool: dict[str, dict] = {}
        _all_nomination_sources = []
        for vr in vec_results:
            _all_nomination_sources.extend(vr)
        if bm25_hits:
            _all_nomination_sources.extend(bm25_hits)
        if literal_hits:
            _all_nomination_sources.extend(literal_hits)
        if _registry_hits:
            _all_nomination_sources.extend(_registry_hits)

        for r in _all_nomination_sources:
            rid = str(r.get("id") or r.get("qdrant_id") or "")
            if not rid:
                fp = r.get("file_path", "")
                ci = r.get("chunk_index", 0)
                rid = f"{fp}:{ci}"
            existing = candidate_pool.get(rid)
            if existing is None or r.get("score", 0) > existing.get("score", 0):
                candidate_pool[rid] = dict(r)
            if r.get("representation_type") == "synthetic_question":
                candidate_pool[rid]["synthetic_match"] = True
            if r.get("needle_registry_hit"):
                candidate_pool[rid]["needle_registry_hit"] = True

        pool = list(candidate_pool.values())
        n_dedupe = len(pool)
        n_synthetic_hits = sum(1 for r in pool if r.get("synthetic_match"))

        # Graph entity facts: inject as additional candidates (not score modifiers)
        if GRAPH_RETRIEVAL_ENABLED and detected_entities:
            entity_facts = build_entity_fact_results(detected_entities, min_score=0.0)
            for ef in entity_facts:
                efid = str(ef.get("id", ""))
                if efid and efid not in candidate_pool:
                    candidate_pool[efid] = ef
                    pool.append(ef)

        if not pool:
            _stage_timings["postprocess_ms"] = round((time.monotonic() - _t_post) * 1000, 1)
            empty_pool = {
                "answer": "No relevant memories found.",
                "sources": [],
                "chunks_searched": n_coarse,
                "agents_scoped": agent_ids or ([agent_id] if agent_id else []),
                "retrieval_trace": _retrieval_trace(
                    vector_limit=vector_limit,
                    coarse_count=n_coarse,
                    deduped_count=0,
                    threshold=0.0,
                    after_threshold_count=0,
                    after_rerank_count=0,
                    parent_enriched=False,
                    refinement_chunks=0,
                    graph_entities_found=n_graph_entities,
                    graph_context_items=n_graph_items,
                    tier=tier,
                    bm25_hits=n_bm25,
                    stage_timings=_stage_timings,
                    reranker_enabled=True,
                    reranker_model=RERANKER_MODEL,
                    **_trace_kw(),
                ),
            }
            _attach_stage0(empty_pool, stage0_inventory, auto_type, user_memory_type)
            _observe_search_results(namespace, [])
            return empty_pool

        # Cross-encoder rerank: the sole ranking authority
        # Parent text is already stored in the payload at index time — no runtime fetch needed
        from archivist.retrieval.reranker import rerank_candidates

        reranked = await rerank_candidates(
            query,
            pool,
            model_name=RERANKER_MODEL,
            top_k=RERANKER_TOP_K,
        )
        n_rerank = len(reranked)
        filtered = reranked

        _stage_timings["postprocess_ms"] = round((time.monotonic() - _t_post) * 1000, 1)
        _common_trace = dict(
            vector_limit=vector_limit,
            coarse_count=n_coarse,
            deduped_count=n_dedupe,
            threshold=0.0,
            after_threshold_count=len(filtered),
            after_rerank_count=n_rerank,
            parent_enriched=any(r.get("parent_text") for r in filtered),
            refinement_chunks=0,
            graph_entities_found=n_graph_entities,
            graph_context_items=n_graph_items,
            entity_facts_injected=0,
            temporal_decay_applied=False,
            tier=tier,
            outcome_adjustments=0,
            context_status=None,
            bm25_hits=n_bm25,
            stage_timings=_stage_timings,
            reranker_enabled=True,
            reranker_model=RERANKER_MODEL,
            nomination_pool_size=len(pool),
            **_trace_kw(),
        )
    else:
        # ══════════════════════════════════════════════════════════════════
        # LEGACY PATH (RERANKER_ENABLED=False)
        # Kept intact for shadow comparison during migration.
        # ══════════════════════════════════════════════════════════════════

        # Legacy merging: RRF + BM25 merge (already done above in coarse)
        coarse = dedupe_vector_hits(coarse)
        n_dedupe = len(coarse)
        n_synthetic_hits = sum(1 for r in coarse if r.get("synthetic_match"))

        # Stage 1c: Temporal decay (v0.5, intent-aware v1.9)
        temporal_applied = False
        if TEMPORAL_DECAY_HALFLIFE_DAYS > 0:
            coarse = apply_temporal_decay(
                coarse,
                TEMPORAL_DECAY_HALFLIFE_DAYS,
                temporal_intent=temporal_intent,
            )
            temporal_applied = True

        # Stage 1d–1e: Scoring — LTR model (v1.10) or hand-tuned pipeline
        _ltr_used = False
        if ltr_available():
            coarse = ltr_rank_results(coarse)
            _ltr_used = True
        else:
            coarse = apply_hotness_to_results(coarse)
            coarse = apply_importance_to_results(coarse)

        # Stage 2: Threshold filter (Phase 1, dynamic v1.10)
        if DYNAMIC_THRESHOLD_ENABLED:
            filtered = apply_dynamic_threshold(coarse, effective_threshold)
        else:
            filtered = apply_retrieval_threshold(coarse, effective_threshold)

        # Stage 2-rescue: Adaptive vector limit (v1.9)
        if (
            ADAPTIVE_VECTOR_LIMIT_ENABLED
            and len(filtered) < ADAPTIVE_VECTOR_MIN_RESULTS
            and n_coarse >= vector_limit
        ):
            wider_limit = int(vector_limit * ADAPTIVE_VECTOR_LIMIT_MULTIPLIER)
            wider_coarse = await search_vectors(
                query,
                agent_id=agent_id,
                agent_ids=agent_ids,
                team=team,
                namespace=namespace,
                date_from=date_from,
                date_to=date_to,
                memory_type=effective_memory_type,
                limit=wider_limit,
                _query_vec=_cached_query_vec,
            )
            if len(wider_coarse) > n_coarse:
                was_adaptive_widened = True
                wider_coarse = dedupe_vector_hits(wider_coarse)
                if TEMPORAL_DECAY_HALFLIFE_DAYS > 0:
                    wider_coarse = apply_temporal_decay(
                        wider_coarse,
                        TEMPORAL_DECAY_HALFLIFE_DAYS,
                        temporal_intent=temporal_intent,
                    )
                wider_coarse = apply_hotness_to_results(wider_coarse)
                wider_coarse = apply_importance_to_results(wider_coarse)
                wider_filtered = apply_retrieval_threshold(wider_coarse, effective_threshold)
                filtered, _ = _merge_into_results(filtered, wider_filtered)

        # Stage 2-bm25-rescue: BM25 rescue slots (v1.9, needle-boosted v1.11)
        if BM25_RESCUE_ENABLED and BM25_ENABLED and n_bm25 > 0:
            bm25_max = max(
                (r.get("bm25_score", 0) for r in coarse if r.get("bm25_score")), default=0
            )
            if bm25_max > 0:
                rescue_threshold = bm25_max * BM25_RESCUE_MIN_SCORE_RATIO
                rescue_slots = 7 if _needle_query_detected else BM25_RESCUE_MAX_SLOTS
                rescue_candidates = [
                    r for r in coarse if r.get("bm25_score", 0) >= rescue_threshold
                ][:rescue_slots]
                filtered, n_bm25_rescue = _merge_into_results(
                    filtered,
                    rescue_candidates,
                    min_score=effective_threshold,
                    tag="bm25_rescue",
                )

        # Stage 2-xagent: Cross-agent rank guards (v1.9)
        if agent_ids and len(agent_ids) > 1 and CROSS_AGENT_MAX_SHARE < 1.0 and filtered:
            max_per_agent = max(1, int(len(filtered) * CROSS_AGENT_MAX_SHARE))
            agent_counts: dict[str, int] = {}
            guarded: list[dict] = []
            overflow: list[dict] = []
            for r in filtered:
                aid = r.get("agent_id", "")
                cnt = agent_counts.get(aid, 0)
                if cnt < max_per_agent:
                    guarded.append(r)
                    agent_counts[aid] = cnt + 1
                else:
                    overflow.append(r)
                    was_cross_agent_capped = True
            guarded.extend(overflow)
            filtered = guarded

        # Stage 2a: Entity fact injection (v1.7) — guaranteed recall for known entities
        n_entity_facts_injected = 0
        if GRAPH_RETRIEVAL_ENABLED and detected_entities:
            entity_facts = build_entity_fact_results(
                detected_entities,
                min_score=effective_threshold + 0.05,
                as_of=date_from,
            )
            if entity_facts:
                filtered, n_entity_facts_injected = _merge_into_results(
                    filtered,
                    entity_facts,
                    preserve_top_n=min(5, max(limit // 2, 1)),
                )

        if not filtered:
            _stage_timings["postprocess_ms"] = round((time.monotonic() - _t_post) * 1000, 1)
            below = {
                "status": "below_threshold",
                "answer": "",
                "sources": [],
                "chunks_searched": n_coarse,
                "threshold": effective_threshold,
                "best_score": max(r["score"] for r in coarse) if coarse else 0,
                "agents_scoped": agent_ids or ([agent_id] if agent_id else []),
                "retrieval_trace": _retrieval_trace(
                    vector_limit=vector_limit,
                    coarse_count=n_coarse,
                    deduped_count=n_dedupe,
                    threshold=effective_threshold,
                    after_threshold_count=0,
                    after_rerank_count=0,
                    parent_enriched=False,
                    refinement_chunks=0,
                    graph_entities_found=n_graph_entities,
                    graph_context_items=n_graph_items,
                    temporal_decay_applied=temporal_applied,
                    tier=tier,
                    bm25_hits=n_bm25,
                    stage_timings=_stage_timings,
                    **_trace_kw(),
                ),
            }
            _attach_stage0(below, stage0_inventory, auto_type, user_memory_type)
            _observe_search_results(namespace, [])
            return below

        # Stage 2b: Outcome-aware scoring (v0.6)
        n_outcome_adj = 0
        filtered_ids = [str(r.get("id", "")) for r in filtered if r.get("id")]
        if filtered_ids:
            try:
                adjustments = get_outcome_adjustments(filtered_ids)
                for r in filtered:
                    adj = adjustments.get(str(r.get("id", "")), 0.0)
                    if adj != 0.0:
                        r["outcome_adjustment"] = adj
                        r["score"] = r.get("score", 0) + adj
                        n_outcome_adj += 1
                if n_outcome_adj:
                    filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
            except Exception as e:
                logger.debug("Outcome adjustments skipped: %s", e)

        # Stage 3: Rerank (legacy RERANK_ENABLED path — separate from RERANKER_ENABLED)
        reranked = await _apply_rerank(query, filtered)
        n_rerank = len(reranked)

        # Stage 3a: Iterative retrieval — auto-reformulate on low-confidence (v1.10)
        _ITERATIVE_THRESHOLD = 0.45
        if (
            not _is_retry
            and reranked
            and max(r.get("score", 0) for r in reranked) < _ITERATIVE_THRESHOLD
            and _budget.can_afford(200)
        ):
            try:
                snippets = " | ".join((r.get("text", "") or "")[:80] for r in reranked[:3])
                reformulated = await llm_query(
                    f"The search query '{query}' returned low-relevance results: [{snippets}]. "
                    "Suggest a single, better search query to find the answer. Return ONLY the query.",
                    system="You are a search query optimizer. Return only the improved query, nothing else.",
                    max_tokens=64,
                    model=LLM_REFINE_MODEL or LLM_MODEL,
                    stage="iterative_retrieval",
                )
                reformulated = reformulated.strip().strip('"').strip("'")
                if reformulated and reformulated.lower() != query.lower():
                    logger.debug("Iterative retrieval: reformulated %r -> %r", query, reformulated)
                    _stage_timings["postprocess_ms"] = round((time.monotonic() - _t_post) * 1000, 1)
                    return await recursive_retrieve(
                        reformulated,
                        agent_id=agent_id,
                        agent_ids=agent_ids,
                        team=team,
                        namespace=namespace,
                        limit=limit,
                        refine=refine,
                        threshold=threshold,
                        tier=tier,
                        date_from=date_from,
                        date_to=date_to,
                        max_tokens=max_tokens,
                        memory_type=memory_type,
                        _is_retry=True,
                    )
            except Exception as e:
                logger.debug("Iterative retrieval reformulation failed: %s", e)

        # Stage 4: Parent text is now stored at index time; no runtime enrichment needed
        enriched = reranked

        _stage_timings["postprocess_ms"] = round((time.monotonic() - _t_post) * 1000, 1)
        _common_trace = dict(
            vector_limit=vector_limit,
            coarse_count=n_coarse,
            deduped_count=n_dedupe,
            threshold=effective_threshold,
            after_threshold_count=len(filtered),
            after_rerank_count=n_rerank,
            parent_enriched=any(r.get("parent_text") for r in enriched),
            refinement_chunks=0,
            graph_entities_found=n_graph_entities,
            graph_context_items=n_graph_items,
            entity_facts_injected=n_entity_facts_injected,
            temporal_decay_applied=temporal_applied,
            tier=tier,
            outcome_adjustments=n_outcome_adj,
            context_status=None,
            bm25_hits=n_bm25,
            stage_timings=_stage_timings,
            **_trace_kw(),
        )

    # ── Common tail: cap → refine → synthesize ──
    if RERANKER_ENABLED:
        enriched = filtered
    # else: enriched was set in the legacy branch above

    # Cap how many chunks we refine (per-request limit)
    enriched = enriched[:limit]

    # If max_tokens specified, further cap by token budget
    if max_tokens and max_tokens > 0:
        budget = 0
        capped = []
        for r in enriched:
            chunk_toks = count_tokens(select_tier(r, tier))
            if budget + chunk_toks > max_tokens:
                break
            budget += chunk_toks
            capped.append(r)
        enriched = capped if capped else enriched[:1]

    n_refine = len(enriched)

    _micro_hits = sum(1 for r in enriched if r.get("parent_id") and not r.get("is_parent"))
    if _micro_hits:
        m.inc(m.MICRO_CHUNK_HITS, {"namespace": namespace}, value=_micro_hits)

    # Context-status signaling (v1.0, upgraded v1.1 with tokenizer)
    result_tokens_approx = sum(count_tokens(select_tier(r, tier)) for r in enriched)
    budget_tokens = max_tokens if max_tokens and max_tokens > 0 else None
    if budget_tokens:
        budget_used_pct = round(result_tokens_approx / budget_tokens * 100, 1)
    else:
        budget_used_pct = 0.0
    _ctx_status = {
        "result_tokens_approx": result_tokens_approx,
        "budget_tokens": budget_tokens or "unlimited",
        "budget_used_pct": budget_used_pct,
        "tier": tier,
        "hint": "compress" if budget_tokens and budget_used_pct > 80 else "ok",
    }

    _stage_timings["postprocess_ms"] = round((time.monotonic() - _t_post) * 1000, 1)

    _common_trace.update(
        refinement_chunks=n_refine,
        parent_enriched=any(r.get("parent_text") for r in enriched),
        context_status=_ctx_status,
    )

    if not refine:
        # Return tier-appropriate text in sources
        for r in enriched:
            r["tier_text"] = select_tier(r, tier)
        no_refine = {
            "answer": "",
            "sources": enriched[: min(10, limit)],
            "chunks_searched": n_coarse,
            "chunks_after_threshold": len(filtered),
            "agents_scoped": agent_ids or ([agent_id] if agent_id else []),
            "retrieval_trace": _retrieval_trace(**_common_trace),
        }
        _attach_stage0(no_refine, stage0_inventory, auto_type, user_memory_type)
        _observe_search_results(namespace, no_refine.get("sources"))
        return no_refine

    # Stage 5: LLM refinement (parallel + optional high-confidence skip)
    multi_agent = len({h.get("agent_id") for h in enriched if h.get("agent_id")}) > 1
    refine_model = LLM_REFINE_MODEL or LLM_MODEL
    synth_model = LLM_SYNTH_MODEL or LLM_MODEL

    top_score = float(enriched[0].get("score", 0)) if enriched else 0.0
    if REFINE_SKIP_THRESHOLD > 0 and top_score >= REFINE_SKIP_THRESHOLD:
        refined = [
            {
                "extraction": select_tier(hit, tier),
                "source": hit["file_path"],
                "date": hit["date"],
                "agent_id": hit["agent_id"],
                "score": hit["score"],
                "rerank_score": hit.get("rerank_score"),
            }
            for hit in enriched
        ]
        _common_trace["refine_skipped"] = True
        _common_trace["refine_wall_ms"] = 0
        _common_trace["refine_parallel"] = False
        _common_trace["refine_model"] = refine_model
    else:
        t_ref0 = time.monotonic()
        tasks = [_refine_one_chunk(hit, query, tier, refine_model) for hit in enriched]
        results = await asyncio.gather(*tasks)
        refined = [r for r in results if r is not None]
        _common_trace["refine_wall_ms"] = int((time.monotonic() - t_ref0) * 1000)
        _common_trace["refine_skipped"] = False
        _common_trace["refine_parallel"] = True
        _common_trace["refine_model"] = refine_model

    if not refined:
        no_rel = {
            "answer": "Found chunks but none were relevant after refinement.",
            "sources": [],
            "chunks_searched": n_coarse,
            "chunks_after_threshold": len(filtered),
            "agents_scoped": agent_ids or ([agent_id] if agent_id else []),
            "retrieval_trace": _retrieval_trace(**_common_trace),
        }
        _attach_stage0(no_rel, stage0_inventory, auto_type, user_memory_type)
        _observe_search_results(namespace, [])
        return no_rel

    # Stage 6: Synthesis
    extractions_text = "\n\n".join(
        (
            f"[agent={r['agent_id']}] [{r['source']} ({r['date']})] {r['extraction']}"
            if multi_agent
            else f"[{r['source']} ({r['date']})] {r['extraction']}"
        )
        for r in refined
    )
    synth_prompt = f"Query: {query}\n\nExtracted memories:\n{extractions_text}"
    synth_system = SYNTHESIZE_MULTI_AGENT if multi_agent else SYNTHESIZE_SYSTEM

    t_synth0 = time.monotonic()
    try:
        answer = await llm_query(
            synth_prompt,
            system=synth_system,
            max_tokens=1024,
            model=synth_model,
            stage="synth",
        )
        if answer is None:
            answer = ""
        else:
            answer = str(answer)
        if not answer.strip():
            logger.warning(
                "Synthesis returned empty text; falling back to raw extractions (truncated)"
            )
            answer = extractions_text[:24000]
            _common_trace["synthesis_degraded"] = True
    except Exception as e:
        logger.error("Synthesis failed: %s", e)
        answer = extractions_text
        # Signal to callers/operators that the answer is raw extractions, not a synthesised response.
        _common_trace["synthesis_degraded"] = True
    finally:
        _common_trace["synth_wall_ms"] = int((time.monotonic() - t_synth0) * 1000)
        _common_trace["synth_model"] = synth_model

    final_result = {
        "answer": answer,
        "sources": [
            {
                "file": r["source"],
                "date": r["date"],
                "agent": r["agent_id"],
                "score": r["score"],
                "rerank_score": r.get("rerank_score"),
            }
            for r in refined
        ],
        "chunks_searched": n_coarse,
        "chunks_after_threshold": len(filtered),
        "chunks_relevant": len(refined),
        "multi_agent": multi_agent,
        "agents_scoped": agent_ids or ([agent_id] if agent_id else []),
        "retrieval_trace": _retrieval_trace(**_common_trace),
    }
    _attach_stage0(final_result, stage0_inventory, auto_type, user_memory_type)
    _observe_search_results(namespace, final_result.get("sources"))

    elapsed = int((time.monotonic() - t0) * 1000)
    final_result["_cache_namespace"] = namespace
    hot_cache.put(
        agent_id or "fleet",
        query,
        final_result,
        namespace=namespace,
        tier=tier,
        memory_type=effective_memory_type,
        extra=cache_extra,
    )
    retrieval_log.log_retrieval(
        agent_id=agent_id or "fleet",
        query=query,
        namespace=namespace,
        tier=tier,
        memory_type=effective_memory_type,
        retrieval_trace=final_result.get("retrieval_trace", {}),
        result_count=len(refined),
        cache_hit=False,
        duration_ms=elapsed,
    )
    m.inc(m.SEARCH_TOTAL)
    m.inc(m.CACHE_MISS)
    m.observe(m.SEARCH_DURATION, elapsed)

    logger.info(
        "retrieval_pipeline.complete",
        extra={
            "query_length": len(query),
            "namespace": namespace,
            "agent_id": agent_id or "fleet",
            "registry_hits": len(_registry_hits),
            "vector_results": n_coarse,
            "bm25_results": n_bm25,
            "graph_entities": n_graph_entities,
            "graph_items": n_graph_items,
            "deduped": n_dedupe,
            "post_threshold": len(filtered),
            "final_count": len(refined),
            "expansion_variants": n_expansion_variants,
            "hyde_used": _hyde_used,
            "ltr_used": _ltr_used,
            "bm25_rescue": n_bm25_rescue,
            "synthetic_hits": n_synthetic_hits,
            "duration_ms": elapsed,
        },
    )

    return final_result
