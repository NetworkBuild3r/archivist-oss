"""Conflict detection — checks for similar memories before writes."""

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchExcept

from config import QDRANT_URL, QDRANT_COLLECTION
from embeddings import embed_text

logger = logging.getLogger("archivist.conflict")

SIMILARITY_THRESHOLD = 0.85
NEAR_DUPLICATE_THRESHOLD = 0.95


@dataclass
class ConflictResult:
    has_conflict: bool
    conflicting_ids: list[str]
    max_similarity: float
    recommendation: str  # "keep_both", "merge", "manual_review"


async def check_for_conflicts(
    text: str,
    namespace: str,
    agent_id: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> ConflictResult:
    """Check if a new memory conflicts with existing ones in the same namespace."""
    vec = await embed_text(text)
    client = QdrantClient(url=QDRANT_URL, timeout=30)

    must_filters = [
        FieldCondition(key="namespace", match=MatchValue(value=namespace)),
    ]
    if agent_id:
        must_filters.append(
            FieldCondition(key="agent_id", match=MatchExcept(except_=agent_id))
        )

    try:
        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec,
            query_filter=Filter(must=must_filters),
            limit=10,
            with_payload=True,
        ).points
    except Exception as e:
        logger.warning("Conflict check failed: %s", e)
        return ConflictResult(
            has_conflict=False,
            conflicting_ids=[],
            max_similarity=0.0,
            recommendation="keep_both",
        )

    conflicts = [r for r in results if r.score >= threshold]

    if not conflicts:
        return ConflictResult(
            has_conflict=False,
            conflicting_ids=[],
            max_similarity=max((r.score for r in results), default=0.0),
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
