"""Conflict detection — checks for similar memories before writes.

v1.0: adds LLM-adjudicated dedup for high-similarity matches.
v1.11: consolidated query — single embed + single Qdrant search for both conflict + dedup.
"""

import json
import logging
from dataclasses import dataclass

from qdrant_client.models import FieldCondition, Filter, MatchValue

from archivist.core.config import (
    CURATOR_LLM_API_KEY,
    CURATOR_LLM_MODEL,
    CURATOR_LLM_URL,
    DEDUP_LLM_ENABLED,
    DEDUP_LLM_THRESHOLD,
    LLM_MODEL,
    LLM_URL,
)
from archivist.features.embeddings import embed_text
from archivist.storage.collection_router import collection_for

_CURATOR_MODEL = CURATOR_LLM_MODEL or LLM_MODEL
_CURATOR_URL = CURATOR_LLM_URL or LLM_URL
_CURATOR_KEY = CURATOR_LLM_API_KEY
import archivist.core.metrics as m
from archivist.features.llm import llm_query
from archivist.storage.qdrant import qdrant_client

logger = logging.getLogger("archivist.conflict")

SIMILARITY_THRESHOLD = 0.85
NEAR_DUPLICATE_THRESHOLD = 0.95

_DEDUP_SYSTEM = (
    "You are a memory deduplication judge. Given a NEW memory and up to 3 EXISTING memories "
    "from the same namespace, decide how to handle each pair.\n\n"
    "For each existing memory, output one decision:\n"
    '- "skip": The new memory is a duplicate of this existing one — do not store it.\n'
    '- "create": The new memory is distinct — store it alongside the existing one.\n'
    '- "merge": The two should be combined into one richer entry.\n'
    '- "delete": The new memory supersedes this existing one — archive the old one.\n\n'
    "Return a JSON array of objects: "
    '[{"existing_id": "...", "decision": "skip|create|merge|delete", "reasoning": "..."}]. '
    "Return ONLY the JSON array."
)


@dataclass
class ConflictResult:
    has_conflict: bool
    conflicting_ids: list[str]
    max_similarity: float
    recommendation: str  # "keep_both", "merge", "manual_review"


@dataclass
class DedupResult:
    action: str  # "store", "skip", "merge", "delete_old"
    existing_ids: list[str]
    decisions: list[dict]
    max_similarity: float


async def _query_similar(
    text: str,
    namespace: str,
    limit: int = 10,
    vec: list[float] | None = None,
) -> tuple[list[float], list]:
    """Single Qdrant similarity query for both conflict + dedup paths.

    Returns (embedding_vector, qdrant_scored_points).
    """
    if vec is None:
        vec = await embed_text(text)
    client = qdrant_client()
    _coll = collection_for(namespace)

    must_filters = [
        FieldCondition(key="namespace", match=MatchValue(value=namespace)),
    ]
    try:
        results = client.query_points(
            collection_name=_coll,
            query=vec,
            query_filter=Filter(must=must_filters),
            limit=limit,
            with_payload=True,
        ).points
    except Exception as e:
        logger.warning("Similarity query failed: %s", e)
        results = []

    return vec, results


async def check_for_conflicts(
    text: str,
    namespace: str,
    agent_id: str,
    threshold: float = SIMILARITY_THRESHOLD,
    *,
    _shared_vec: list[float] | None = None,
    _shared_results: list | None = None,
) -> ConflictResult:
    """Check if a new memory conflicts with existing ones in the same namespace."""
    if _shared_results is not None:
        results = _shared_results
    else:
        _, results = await _query_similar(text, namespace, vec=_shared_vec)

    cross_agent = [r for r in results if agent_id and (r.payload or {}).get("agent_id") != agent_id]

    conflicts = [r for r in cross_agent if r.score >= threshold]

    if not conflicts:
        return ConflictResult(
            has_conflict=False,
            conflicting_ids=[],
            max_similarity=max((r.score for r in cross_agent), default=0.0),
            recommendation="keep_both",
        )

    max_sim = max(c.score for c in conflicts)
    conflict_ids = [str(c.id) for c in conflicts]

    if max_sim >= NEAR_DUPLICATE_THRESHOLD:
        recommendation = "manual_review"
    else:
        recommendation = "merge"

    return ConflictResult(
        has_conflict=True,
        conflicting_ids=conflict_ids,
        max_similarity=max_sim,
        recommendation=recommendation,
    )


async def llm_adjudicated_dedup(
    text: str,
    namespace: str,
    agent_id: str,
    *,
    _shared_results: list | None = None,
) -> DedupResult | None:
    """Run LLM dedup on high-similarity matches. Returns None if dedup is disabled or no matches."""
    if not DEDUP_LLM_ENABLED:
        return None

    if _shared_results is not None:
        results = _shared_results
    else:
        _, results = await _query_similar(text, namespace)

    high_sim = [r for r in results if r.score >= DEDUP_LLM_THRESHOLD]
    if not high_sim:
        return None

    max_sim = max(r.score for r in high_sim)
    top_matches = high_sim[:3]

    existing_texts = []
    for r in top_matches:
        payload = r.payload or {}
        existing_texts.append(
            {
                "id": str(r.id),
                "text": (payload.get("text") or "")[:500],
                "agent_id": payload.get("agent_id", ""),
                "date": payload.get("date", ""),
                "score": r.score,
            }
        )

    prompt = f"NEW MEMORY (from agent '{agent_id}'):\n{text[:500]}\n\nEXISTING MEMORIES:\n"
    for i, ex in enumerate(existing_texts, 1):
        prompt += f"\n{i}. [ID: {ex['id']}, agent: {ex['agent_id']}, similarity: {ex['score']:.3f}]\n{ex['text']}\n"

    try:
        m.inc(m.CURATOR_LLM_CALLS)
        raw = await llm_query(
            prompt,
            system=_DEDUP_SYSTEM,
            max_tokens=512,
            json_mode=True,
            model=_CURATOR_MODEL,
            url=_CURATOR_URL,
            api_key=_CURATOR_KEY,
            stage="curator_dedup",
        )
        decisions = json.loads(raw.strip().strip("`").strip())
        if not isinstance(decisions, list):
            decisions = [decisions]
    except Exception as e:
        logger.warning("LLM dedup decision failed: %s", e)
        return None

    for d in decisions:
        decision = d.get("decision", "create")
        m.inc(m.CURATOR_DEDUP_DECISION, {"decision": decision})

    skip_decisions = [d for d in decisions if d.get("decision") == "skip"]
    merge_decisions = [d for d in decisions if d.get("decision") == "merge"]
    delete_decisions = [d for d in decisions if d.get("decision") == "delete"]

    if skip_decisions:
        action = "skip"
    elif merge_decisions:
        action = "merge"
    elif delete_decisions:
        action = "delete_old"
    else:
        action = "store"

    return DedupResult(
        action=action,
        existing_ids=[str(r.id) for r in top_matches],
        decisions=decisions,
        max_similarity=max_sim,
    )
