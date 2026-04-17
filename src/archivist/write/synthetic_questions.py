"""Synthetic question generation — index-time multi-representation enrichment.

For each chunk, generates questions that the chunk would perfectly answer.
These questions are embedded separately and stored as additional Qdrant points
with ``representation_type: "synthetic_question"``, so vector ANN search
matches queries in the question's semantic space — not just the passage's.

The LLM call is cached by content hash to avoid redundant work on re-index.
"""

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict

from qdrant_client.models import PointStruct

from archivist.core.config import (
    SYNTHETIC_QUESTIONS_ENABLED,
    SYNTHETIC_QUESTIONS_COUNT,
    LLM_REFINE_MODEL,
    LLM_MODEL,
    CURATOR_LLM_URL,
    CURATOR_LLM_MODEL,
    CURATOR_LLM_API_KEY,
)
from archivist.features.llm import llm_query
from archivist.features.embeddings import embed_batch
import archivist.core.metrics as m

logger = logging.getLogger("archivist.synthetic_questions")

_SYSTEM_PROMPT = (
    "You generate search queries. Given a passage of text, produce exactly {count} "
    "specific questions that this passage would be the ideal answer to. "
    "Cover different aspects: factual lookups, how-to queries, and keyword searches. "
    "Return ONLY a JSON array of strings. No explanation, no markdown fences."
)

_CACHE_MAX = 1024
_CACHE_TTL = 7200  # 2 hours
_cache: OrderedDict[str, tuple[float, list[str]]] = OrderedDict()
_cache_lock = threading.Lock()

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _cache_get(key: str) -> list[str] | None:
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        ts, questions = entry
        if time.monotonic() - ts > _CACHE_TTL:
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return questions


def _cache_put(key: str, questions: list[str]) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic(), questions)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


async def generate_synthetic_questions(
    text: str,
    count: int = 0,
) -> list[str]:
    """Generate synthetic questions for a chunk. Returns [] if disabled or on failure."""
    if not SYNTHETIC_QUESTIONS_ENABLED:
        import archivist.core.config as _cfg
        if not _cfg.SYNTHETIC_QUESTIONS_ENABLED:
            return []
    if not count:
        count = SYNTHETIC_QUESTIONS_COUNT

    key = _cache_key(text)
    cached = _cache_get(key)
    if cached is not None:
        return cached[:count]

    model = CURATOR_LLM_MODEL or LLM_REFINE_MODEL or LLM_MODEL
    url = CURATOR_LLM_URL or ""
    api_key: str | None = CURATOR_LLM_API_KEY if CURATOR_LLM_URL else None

    system = _SYSTEM_PROMPT.replace("{count}", str(count))
    t0 = time.monotonic()
    try:
        raw = await llm_query(
            text[:1500],
            system=system,
            max_tokens=768,
            model=model,
            json_mode=True,
            stage="synthetic_questions",
            url=url,
            api_key=api_key,
        )
        dur_ms = round((time.monotonic() - t0) * 1000, 1)
        logger.debug("synthetic_questions dur_ms=%.1f raw_len=%d", dur_ms, len(raw))

        questions = _parse_questions(raw, count)
        if questions:
            _cache_put(key, questions)
        return questions
    except Exception as e:
        logger.warning("Synthetic question generation failed: %s", e)
        return []


def _parse_questions(raw: str, count: int) -> list[str]:
    """Extract a list of question strings from LLM output."""
    raw = raw.strip()
    match = _JSON_ARRAY_RE.search(raw)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()][:count]
        except json.JSONDecodeError:
            pass
    # Fallback: one question per line
    lines = [q.strip().lstrip("0123456789.-) ") for q in raw.splitlines() if q.strip() and len(q.strip()) > 5]
    return lines[:count]


async def generate_and_embed_synthetic_points(
    chunk_point_id: str,
    chunk_text: str,
    base_payload: dict,
    count: int = 0,
) -> list[PointStruct]:
    """Generate synthetic questions, embed them, return Qdrant PointStructs.

    Each returned point carries the original chunk's text in ``text``,
    the synthetic question in ``synthetic_question``, and
    ``representation_type: "synthetic_question"`` to distinguish it
    from the original ``"chunk"`` point during retrieval filtering.
    """
    questions = await generate_synthetic_questions(chunk_text, count=count)
    if not questions:
        return []

    vecs = await embed_batch(questions)
    points = []
    for q, qv in zip(questions, vecs):
        qid = str(uuid.uuid4())
        payload = {
            **base_payload,
            "text": chunk_text,
            "synthetic_question": q,
            "representation_type": "synthetic_question",
            "source_memory_id": chunk_point_id,
            "is_parent": False,
            "parent_id": chunk_point_id,
        }
        points.append(PointStruct(id=qid, vector=qv, payload=payload))

    logger.debug(
        "synthetic_questions.generated point=%s count=%d",
        chunk_point_id, len(points),
    )
    return points
