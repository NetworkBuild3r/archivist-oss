"""RLM-inspired recursive retriever — coarse search, threshold filter, rerank, LLM refinement, synthesis.

Phase 1: threshold, rerank pipeline.
Phase 2 (v0.5): graph-augmented retrieval, temporal decay, tiered context, multi-hop.
Phase 3 (v0.6): outcome-aware scoring from trajectory history.
Phase 4 (v0.7): memory_type routing.
Phase 5 (v0.8): hot cache, retrieval trajectory logging.
"""

import logging
import time
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

from config import (
    QDRANT_COLLECTION,
    RETRIEVAL_THRESHOLD, RERANK_ENABLED, RERANK_MODEL, RERANK_TOP_K,
    VECTOR_SEARCH_LIMIT,
    GRAPH_RETRIEVAL_ENABLED, MULTI_HOP_DEPTH, TEMPORAL_DECAY_HALFLIFE_DAYS,
    HOT_CACHE_ENABLED,
    BM25_ENABLED,
    QUERY_CLASSIFICATION_ENABLED,
    IMPORTANCE_WEIGHT,
    TEMPORAL_INTENT_ENABLED,
    BM25_RESCUE_ENABLED, BM25_RESCUE_MIN_SCORE_RATIO, BM25_RESCUE_MAX_SLOTS,
    ADAPTIVE_VECTOR_LIMIT_ENABLED, ADAPTIVE_VECTOR_MIN_RESULTS, ADAPTIVE_VECTOR_LIMIT_MULTIPLIER,
    CROSS_AGENT_MAX_SHARE,
    TOPIC_ROUTING_ENABLED,
)
from embeddings import embed_text
from llm import llm_query
from memory_fusion import dedupe_vector_hits
from retrieval_filters import apply_retrieval_threshold
from graph_retrieval import (
    extract_entity_mentions,
    graph_context_for_entities,
    apply_temporal_decay,
    merge_graph_context_into_results,
    build_entity_fact_results,
)
from fts_search import search_bm25, merge_vector_and_bm25
from qdrant import qdrant_client
from tiering import select_tier
from tokenizer import count_tokens
from trajectory import get_outcome_adjustments
from hotness import apply_hotness_to_results
from query_intent import classify_temporal_intent
from topic_detector import detect_query_topic
import hot_cache
import retrieval_log
import metrics as m
from namespace_inventory import NamespaceInventory, get_inventory
from query_classifier import classify_query_full, SUBCATEGORY_TO_TOPIC

logger = logging.getLogger("archivist.rlm")


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
    auto_classified_type: str = "",
    inventory_total: int | None = None,
    entity_facts_injected: int = 0,
    temporal_intent: str = "",
    bm25_rescue_count: int = 0,
    adaptive_widened: bool = False,
    cross_agent_capped: bool = False,
) -> dict:
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
    if context_status:
        trace["context_status"] = context_status
    if auto_classified_type:
        trace["auto_classified_type"] = auto_classified_type
    if inventory_total is not None:
        trace["inventory_total"] = inventory_total
    if temporal_intent:
        trace["temporal_intent"] = temporal_intent
    if bm25_rescue_count:
        trace["bm25_rescue_count"] = bm25_rescue_count
    if adaptive_widened:
        trace["adaptive_widened"] = True
    if cross_agent_capped:
        trace["cross_agent_capped"] = True
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
        hint = (
            f"Also {by['skill']} skill memories — try memory_type=skill for how-to questions."
        )
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
    result["memory_awareness"] = _memory_awareness_payload(
        inventory, auto_type, user_memory_type
    )
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
) -> list[dict]:
    """Stage 1: coarse vector search in Qdrant with optional filters."""
    query_vec = await embed_text(query)
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
    if date_from:
        must_filters.append(FieldCondition(key="date", match=MatchValue(value=date_from)) if date_from == date_to else FieldCondition(key="date", range=Range(gte=date_from)))
    if date_to and date_to != date_from:
        must_filters.append(FieldCondition(key="date", range=Range(lte=date_to)))
    if memory_type:
        must_filters.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))
    if topic:
        must_filters.append(FieldCondition(key="topic", match=MatchValue(value=topic)))
    if thought_type:
        must_filters.append(FieldCondition(key="thought_type", match=MatchValue(value=thought_type)))

    search_filter = Filter(must=must_filters) if must_filters else None

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vec,
        query_filter=search_filter,
        limit=limit,
        with_payload=True,
    ).points

    return [
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
            "importance_score": hit.payload.get("importance_score", 0.5),
            "retention_class": hit.payload.get("retention_class", "standard"),
            "topic": hit.payload.get("topic", ""),
            "thought_type": hit.payload.get("thought_type", ""),
        }
        for hit in results
    ]


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


def _apply_rerank(query: str, results: list[dict]) -> list[dict]:
    """Apply cross-encoder reranking if enabled."""
    if not RERANK_ENABLED:
        return results
    try:
        from reranker import rerank_results
        return rerank_results(query, results, model_name=RERANK_MODEL, limit=RERANK_TOP_K)
    except Exception as e:
        logger.warning("Reranking failed, using original order: %s", e)
        return results


async def enrich_with_parent(results: list[dict]) -> list[dict]:
    """For child chunks, fetch and attach the parent chunk text for richer context."""
    parent_ids = {r["parent_id"] for r in results if r.get("parent_id") and not r.get("is_parent")}
    if not parent_ids:
        return results

    client = qdrant_client()
    try:
        parents = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=list(parent_ids),
            with_payload=True,
        )
        parent_map = {str(p.id): p.payload.get("text", "") for p in parents}
    except Exception as e:
        logger.warning("Parent enrichment failed: %s", e)
        return results

    for r in results:
        pid = r.get("parent_id")
        if pid and pid in parent_map:
            r["parent_context"] = parent_map[pid]

    return results


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
) -> dict:
    """Full RLM pipeline: coarse → dedupe → graph augment → temporal decay → threshold → rerank → parent → refine → synthesize."""
    t0 = time.monotonic()
    user_memory_type = memory_type
    stage0_inventory = None
    auto_type = ""
    auto_subcategory = ""
    effective_memory_type = memory_type
    if QUERY_CLASSIFICATION_ENABLED and not memory_type:
        if namespace:
            stage0_inventory = get_inventory(namespace)
        auto_type, auto_subcategory = await classify_query_full(
            query, inventory=stage0_inventory,
        )
        if auto_type:
            effective_memory_type = auto_type

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
    vector_limit = max(VECTOR_SEARCH_LIMIT, limit)
    n_graph_entities = 0
    n_graph_items = 0
    n_bm25_rescue = 0
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
        }

    cache_extra = f"{agent_id}|{','.join(agent_ids or [])}|{team}|{date_from}|{date_to}|{limit}|{refine}"
    cached = hot_cache.get(agent_id or "fleet", query, namespace=namespace,
                           tier=tier, memory_type=effective_memory_type, extra=cache_extra)
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
            agent_id=agent_id or "fleet", query=query, namespace=namespace,
            tier=tier, memory_type=effective_memory_type,
            retrieval_trace=out.get("retrieval_trace", {}),
            result_count=len(out.get("sources", [])),
            cache_hit=True, duration_ms=elapsed,
        )
        out["cache_hit"] = True
        m.inc(m.CACHE_HIT)
        m.inc(m.SEARCH_TOTAL)
        m.observe(m.SEARCH_DURATION, elapsed)
        return out

    # Stage 1: Coarse vector search (wide recall)
    # Topic routing: try topic-filtered search first, fall back if too few results
    coarse = await search_vectors(
        query,
        agent_id=agent_id,
        agent_ids=agent_ids,
        team=team,
        namespace=namespace,
        date_from=date_from,
        date_to=date_to,
        memory_type=effective_memory_type,
        topic=detected_topic,
        limit=vector_limit,
    )
    if detected_topic and len(coarse) < ADAPTIVE_VECTOR_MIN_RESULTS:
        topic_fallback_used = True
        coarse = await search_vectors(
            query,
            agent_id=agent_id,
            agent_ids=agent_ids,
            team=team,
            namespace=namespace,
            date_from=date_from,
            date_to=date_to,
            memory_type=effective_memory_type,
            topic="",
            limit=vector_limit,
        )
    n_coarse = len(coarse)

    # Stage 1-bm25: BM25 keyword search + fusion (v1.2)
    n_bm25 = 0
    if BM25_ENABLED:
        bm25_hits = search_bm25(
            query,
            namespace=namespace,
            agent_id=agent_id if not agent_ids else "",
            memory_type=effective_memory_type,
            limit=vector_limit,
        )
        n_bm25 = len(bm25_hits)
        if bm25_hits:
            coarse = merge_vector_and_bm25(coarse, bm25_hits)

    # Stage 1a: Graph augmentation (v0.5)
    detected_entities: list[dict] = []
    if GRAPH_RETRIEVAL_ENABLED:
        entities = extract_entity_mentions(query)
        detected_entities = entities
        n_graph_entities = len(entities)
        if entities:
            graph_items = graph_context_for_entities(
                [e["id"] for e in entities],
                depth=MULTI_HOP_DEPTH,
            )
            n_graph_items = len(graph_items)
            if graph_items:
                coarse = merge_graph_context_into_results(coarse, graph_items)

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
                **_trace_kw(),
            ),
        }
        _attach_stage0(empty, stage0_inventory, auto_type, user_memory_type)
        return empty

    # Stage 1b: Dedupe (important when merging multi-agent results)
    coarse = dedupe_vector_hits(coarse)
    n_dedupe = len(coarse)

    # Stage 1c: Temporal decay (v0.5, intent-aware v1.9)
    temporal_applied = False
    if TEMPORAL_DECAY_HALFLIFE_DAYS > 0:
        coarse = apply_temporal_decay(
            coarse, TEMPORAL_DECAY_HALFLIFE_DAYS, temporal_intent=temporal_intent,
        )
        temporal_applied = True

    # Stage 1d: Hotness scoring (v1.0)
    coarse = apply_hotness_to_results(coarse)

    # Stage 1e: Importance scoring (v1.7)
    coarse = apply_importance_to_results(coarse)

    # Stage 2: Threshold filter (Phase 1)
    filtered = apply_retrieval_threshold(coarse, effective_threshold)

    # Stage 2-rescue: Adaptive vector limit (v1.9)
    if (ADAPTIVE_VECTOR_LIMIT_ENABLED
            and len(filtered) < ADAPTIVE_VECTOR_MIN_RESULTS
            and n_coarse >= vector_limit):
        wider_limit = int(vector_limit * ADAPTIVE_VECTOR_LIMIT_MULTIPLIER)
        wider_coarse = await search_vectors(
            query, agent_id=agent_id, agent_ids=agent_ids, team=team,
            namespace=namespace, date_from=date_from, date_to=date_to,
            memory_type=effective_memory_type, limit=wider_limit,
        )
        if len(wider_coarse) > n_coarse:
            was_adaptive_widened = True
            wider_coarse = dedupe_vector_hits(wider_coarse)
            if TEMPORAL_DECAY_HALFLIFE_DAYS > 0:
                wider_coarse = apply_temporal_decay(
                    wider_coarse, TEMPORAL_DECAY_HALFLIFE_DAYS,
                    temporal_intent=temporal_intent,
                )
            wider_coarse = apply_hotness_to_results(wider_coarse)
            wider_coarse = apply_importance_to_results(wider_coarse)
            wider_filtered = apply_retrieval_threshold(wider_coarse, effective_threshold)
            existing_ids = {r.get("id") for r in filtered if r.get("id")}
            for r in wider_filtered:
                if r.get("id") and r["id"] not in existing_ids:
                    filtered.append(r)
                    existing_ids.add(r["id"])
            filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Stage 2-bm25-rescue: BM25 rescue slots (v1.9)
    if BM25_RESCUE_ENABLED and BM25_ENABLED and n_bm25 > 0:
        bm25_max = max((r.get("bm25_score", 0) for r in coarse if r.get("bm25_score")), default=0)
        if bm25_max > 0:
            rescue_threshold = bm25_max * BM25_RESCUE_MIN_SCORE_RATIO
            existing_ids = {r.get("id") for r in filtered if r.get("id")}
            rescued = 0
            for r in coarse:
                if rescued >= BM25_RESCUE_MAX_SLOTS:
                    break
                bm25_s = r.get("bm25_score", 0)
                if bm25_s >= rescue_threshold and r.get("id") and r["id"] not in existing_ids:
                    r["bm25_rescue"] = True
                    filtered.append(r)
                    existing_ids.add(r["id"])
                    rescued += 1
            if rescued:
                n_bm25_rescue = rescued
                filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

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
    # Vector-first preservation (v1.8): snapshot top vector results so they
    # are never displaced by injected entity facts.
    n_entity_facts_injected = 0
    if GRAPH_RETRIEVAL_ENABLED and detected_entities:
        vector_preserve_n = min(5, max(limit // 2, 1))
        vector_winners = [r for r in filtered if r.get("file_type") != "entity_fact"][:vector_preserve_n]

        entity_facts = build_entity_fact_results(
            detected_entities, min_score=effective_threshold + 0.05,
            as_of=date_from,
        )
        if entity_facts:
            existing_texts = {r.get("text", "")[:100] for r in filtered}
            for ef in entity_facts:
                if ef["text"][:100] not in existing_texts:
                    filtered.append(ef)
                    existing_texts.add(ef["text"][:100])
                    n_entity_facts_injected += 1
            if n_entity_facts_injected:
                filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Re-insert vector winners that got pushed out by injections
        if vector_winners:
            present_texts = {r.get("text", "")[:100] for r in filtered[:limit]}
            for vw in vector_winners:
                if vw.get("text", "")[:100] not in present_texts:
                    filtered.insert(0, vw)
            filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

    if not filtered:
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
                **_trace_kw(),
            ),
        }
        _attach_stage0(below, stage0_inventory, auto_type, user_memory_type)
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

    # Stage 3: Rerank (Phase 1)
    reranked = _apply_rerank(query, filtered)
    n_rerank = len(reranked)

    # Stage 4: Parent-child enrichment (Phase 1)
    enriched = await enrich_with_parent(reranked)

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

    _common_trace = dict(
        vector_limit=vector_limit,
        coarse_count=n_coarse,
        deduped_count=n_dedupe,
        threshold=effective_threshold,
        after_threshold_count=len(filtered),
        after_rerank_count=n_rerank,
        parent_enriched=any(r.get("parent_context") for r in enriched),
        refinement_chunks=n_refine,
        graph_entities_found=n_graph_entities,
        graph_context_items=n_graph_items,
        entity_facts_injected=n_entity_facts_injected,
        temporal_decay_applied=temporal_applied,
        tier=tier,
        outcome_adjustments=n_outcome_adj,
        context_status=_ctx_status,
        bm25_hits=n_bm25,
        **_trace_kw(),
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
        return no_refine

    # Stage 5: LLM refinement
    refined = []
    multi_agent = len({h.get("agent_id") for h in enriched if h.get("agent_id")}) > 1
    for hit in enriched:
        context = select_tier(hit, tier)
        if hit.get("parent_context") and tier == "l2":
            context = f"[Parent context]\n{hit['parent_context']}\n\n[Matched chunk]\n{context}"

        graph_extra = ""
        if hit.get("graph_context"):
            graph_extra = "\n[Graph context] " + " | ".join(hit["graph_context"][:3])

        who = hit.get("agent_id") or "unknown"
        prompt = (
            f"Query: {query}\n\nMemory chunk (agent={who}, file={hit['file_path']}, date={hit['date']}):\n{context}{graph_extra}"
        )
        try:
            extraction = await llm_query(prompt, system=REFINE_SYSTEM, max_tokens=512)
            if extraction.strip().upper() != "IRRELEVANT":
                refined.append({
                    "extraction": extraction.strip(),
                    "source": hit["file_path"],
                    "date": hit["date"],
                    "agent_id": hit["agent_id"],
                    "score": hit["score"],
                    "rerank_score": hit.get("rerank_score"),
                })
        except Exception as e:
            logger.warning("LLM refinement failed for chunk: %s", e)

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

    try:
        answer = await llm_query(synth_prompt, system=synth_system, max_tokens=1024)
    except Exception as e:
        logger.error("Synthesis failed: %s", e)
        answer = extractions_text

    final_result = {
        "answer": answer,
        "sources": [
            {"file": r["source"], "date": r["date"], "agent": r["agent_id"],
             "score": r["score"], "rerank_score": r.get("rerank_score")}
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

    elapsed = int((time.monotonic() - t0) * 1000)
    final_result["_cache_namespace"] = namespace
    hot_cache.put(agent_id or "fleet", query, final_result, namespace=namespace,
                  tier=tier, memory_type=effective_memory_type, extra=cache_extra)
    retrieval_log.log_retrieval(
        agent_id=agent_id or "fleet", query=query, namespace=namespace,
        tier=tier, memory_type=effective_memory_type,
        retrieval_trace=final_result.get("retrieval_trace", {}),
        result_count=len(refined),
        cache_hit=False, duration_ms=elapsed,
    )
    m.inc(m.SEARCH_TOTAL)
    m.inc(m.CACHE_MISS)
    m.observe(m.SEARCH_DURATION, elapsed)

    return final_result
