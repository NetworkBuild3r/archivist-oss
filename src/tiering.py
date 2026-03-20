"""Tiered context abstraction — L0 (abstract), L1 (overview), L2 (full text).

On ingest, auto-generates compact L0 (~100 tokens) and L1 (~500 tokens) summaries
stored alongside the full L2 text.  Retrieval can return the appropriate tier,
saving tokens when full detail is unnecessary.
"""

import logging
from config import TIERED_CONTEXT_ENABLED, L0_MAX_TOKENS, L1_MAX_TOKENS
from llm import llm_query

logger = logging.getLogger("archivist.tiering")

_L0_SYSTEM = (
    "Summarize the following text in one short sentence (~20 words). "
    "Capture the single most important idea. Return ONLY the sentence, no preamble."
)

_L1_SYSTEM = (
    "Produce a concise overview of the following text in 2-4 sentences. "
    "Include key entities, dates, and actionable facts. "
    "Return ONLY the overview, no preamble or markdown."
)


async def generate_tiers(text: str) -> dict:
    """Return {"l0": str, "l1": str, "l2": str} for a chunk of text.

    When tiering is disabled or LLM fails, l0 and l1 fall back to truncated text.
    """
    l2 = text

    if not TIERED_CONTEXT_ENABLED or len(text.strip()) < 80:
        return {
            "l0": text[:L0_MAX_TOKENS * 4].strip(),
            "l1": text[:L1_MAX_TOKENS * 4].strip(),
            "l2": l2,
        }

    try:
        l0 = await llm_query(text[:3000], system=_L0_SYSTEM, max_tokens=60)
        l0 = l0.strip()
    except Exception as e:
        logger.warning("L0 generation failed, falling back to truncation: %s", e)
        l0 = text[:L0_MAX_TOKENS * 4].strip()

    try:
        l1 = await llm_query(text[:4000], system=_L1_SYSTEM, max_tokens=200)
        l1 = l1.strip()
    except Exception as e:
        logger.warning("L1 generation failed, falling back to truncation: %s", e)
        l1 = text[:L1_MAX_TOKENS * 4].strip()

    return {"l0": l0, "l1": l1, "l2": l2}


def select_tier(hit: dict, tier: str = "l2") -> str:
    """Extract the requested tier text from a search hit's payload.

    Falls through to full text if the requested tier isn't populated.
    """
    if tier == "l0":
        return hit.get("l0") or hit.get("text", "")[:400]
    if tier == "l1":
        return hit.get("l1") or hit.get("text", "")[:2000]
    return hit.get("text", "")
