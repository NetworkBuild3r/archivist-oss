"""RLM-inspired recursive retriever — coarse search, threshold filter, rerank, LLM refinement, synthesis.

Phase 1 additions:
  - Retrieval threshold: discard results below RETRIEVAL_THRESHOLD
  - Rerank pipeline: optional cross-encoder reranking before LLM refinement
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from config import (
    QDRANT_URL, QDRANT_COLLECTION,
    RETRIEVAL_THRESHOLD, RERANK_ENABLED, RERANK_MODEL, RERANK_TOP_K,
    VECTOR_SEARCH_LIMIT,
)
from embeddings import embed_text
from llm import llm_query
from memory_fusion import dedupe_vector_hits
from retrieval_filters import apply_retrieval_threshold

logger = logging.getLogger("archivist.rlm")

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


def _client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=30)


async def search_vectors(
    query: str,
    agent_id: str = "",
    agent_ids: list[str] | None = None,
    team: str = "",
    namespace: str = "",
    file_type: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 20,
) -> list[dict]:
    """Stage 1: coarse vector search in Qdrant with optional filters."""
    query_vec = await embed_text(query)
    client = _client()

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
            "agent_id": hit.payload.get("agent_id", ""),
            "file_path": hit.payload.get("file_path", ""),
            "file_type": hit.payload.get("file_type", ""),
            "date": hit.payload.get("date", ""),
            "team": hit.payload.get("team", ""),
            "namespace": hit.payload.get("namespace", ""),
            "chunk_index": hit.payload.get("chunk_index", 0),
            "parent_id": hit.payload.get("parent_id"),
            "is_parent": hit.payload.get("is_parent", False),
        }
        for hit in results
    ]


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

    client = _client()
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
) -> dict:
    """Full RLM pipeline: coarse search → dedupe → threshold → rerank → parent enrichment → LLM refinement → synthesis."""
    effective_threshold = threshold if threshold is not None else RETRIEVAL_THRESHOLD
    vector_limit = max(VECTOR_SEARCH_LIMIT, limit)

    # Stage 1: Coarse vector search (wide recall)
    coarse = await search_vectors(
        query,
        agent_id=agent_id,
        agent_ids=agent_ids,
        team=team,
        namespace=namespace,
        limit=vector_limit,
    )

    if not coarse:
        return {
            "answer": "No relevant memories found.",
            "sources": [],
            "chunks_searched": 0,
            "agents_scoped": agent_ids or ([agent_id] if agent_id else []),
        }

    # Stage 1b: Dedupe (important when merging multi-agent results)
    coarse = dedupe_vector_hits(coarse)

    # Stage 2: Threshold filter (Phase 1)
    filtered = apply_retrieval_threshold(coarse, effective_threshold)
    if not filtered:
        return {
            "status": "below_threshold",
            "answer": "",
            "sources": [],
            "chunks_searched": len(coarse),
            "threshold": effective_threshold,
            "best_score": max(r["score"] for r in coarse) if coarse else 0,
            "agents_scoped": agent_ids or ([agent_id] if agent_id else []),
        }

    # Stage 3: Rerank (Phase 1)
    reranked = _apply_rerank(query, filtered)

    # Stage 4: Parent-child enrichment (Phase 1)
    enriched = await enrich_with_parent(reranked)

    # Cap how many chunks we refine (per-request limit)
    enriched = enriched[:limit]

    if not refine:
        return {
            "answer": "",
            "sources": enriched[: min(10, limit)],
            "chunks_searched": len(coarse),
            "chunks_after_threshold": len(filtered),
            "agents_scoped": agent_ids or ([agent_id] if agent_id else []),
        }

    # Stage 5: LLM refinement
    refined = []
    multi_agent = len({h.get("agent_id") for h in enriched if h.get("agent_id")}) > 1
    for hit in enriched:
        # Include parent context in the refinement prompt if available
        context = hit["text"]
        if hit.get("parent_context"):
            context = f"[Parent context]\n{hit['parent_context']}\n\n[Matched chunk]\n{hit['text']}"

        who = hit.get("agent_id") or "unknown"
        prompt = (
            f"Query: {query}\n\nMemory chunk (agent={who}, file={hit['file_path']}, date={hit['date']}):\n{context}"
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
        return {
            "answer": "Found chunks but none were relevant after refinement.",
            "sources": [],
            "chunks_searched": len(coarse),
            "chunks_after_threshold": len(filtered),
            "agents_scoped": agent_ids or ([agent_id] if agent_id else []),
        }

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

    return {
        "answer": answer,
        "sources": [
            {"file": r["source"], "date": r["date"], "agent": r["agent_id"],
             "score": r["score"], "rerank_score": r.get("rerank_score")}
            for r in refined
        ],
        "chunks_searched": len(coarse),
        "chunks_after_threshold": len(filtered),
        "chunks_relevant": len(refined),
        "multi_agent": multi_agent,
        "agents_scoped": agent_ids or ([agent_id] if agent_id else []),
    }
