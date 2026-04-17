"""Structured conversation compaction — produces Goal/Progress/Decisions/Next Steps summaries."""

import json
import logging

import llm as llm_mod
from config import LLM_MODEL, LLM_URL, CURATOR_LLM_MODEL, CURATOR_LLM_URL, CURATOR_LLM_API_KEY
from text_utils import strip_fences

logger = logging.getLogger("archivist.compaction")

_CURATOR_MODEL = CURATOR_LLM_MODEL or LLM_MODEL
_CURATOR_URL = CURATOR_LLM_URL or LLM_URL
_CURATOR_KEY = CURATOR_LLM_API_KEY

STRUCTURED_COMPACT_SYSTEM = (
    "You are a memory compaction assistant. Given a set of memory entries, "
    "produce a structured summary as a JSON object with these fields:\n"
    "- goal: What the agent is trying to achieve (1-2 sentences)\n"
    "- progress: Key accomplishments so far (2-4 bullet points)\n"
    "- decisions: Important choices made (list of strings)\n"
    "- next_steps: Recommended next actions (list of strings)\n"
    "- critical_context: Facts that must be preserved — file paths, entity names, "
    "error messages, config values (1-3 sentences)\n\n"
    "Return ONLY valid JSON. Be concise. Preserve specific details (names, paths, versions)."
)

STRUCTURED_COMPACT_MULTI_AGENT = (
    STRUCTURED_COMPACT_SYSTEM
    + "\n\nThese memories come from multiple agents. Preserve shared factual knowledge and "
    "task-invariant insights. Discard agent-specific framing, tool preferences, or reasoning style."
)

FLAT_COMPACT_SYSTEM = (
    "Summarize these memory entries into a single concise summary (200 tokens max). "
    "Preserve key facts, entities, and actionable insights."
)

FLAT_COMPACT_MULTI_AGENT = (
    FLAT_COMPACT_SYSTEM
    + " Sources are from multiple agents — keep only cross-cutting facts, not idiosyncratic style."
)


async def compact_structured(
    texts: list[tuple[str, str]],
    previous_summary: str = "",
    multi_agent: bool = False,
) -> dict:
    """Produce a structured compaction of memory texts.

    Args:
        texts: List of (memory_id, text) tuples to compact.
        previous_summary: Optional prior structured summary JSON to merge with.
        multi_agent: When True, debias toward shared facts across agents.

    Returns:
        dict with goal, progress, decisions, next_steps, critical_context.
    """
    parts = []
    for mid, text in texts:
        parts.append(f"[{mid}] {text[:800]}")

    combined = "\n\n---\n\n".join(parts)

    if previous_summary:
        combined = f"[Previous summary]\n{previous_summary}\n\n---\n\n[New memories]\n{combined}"

    system = STRUCTURED_COMPACT_MULTI_AGENT if multi_agent else STRUCTURED_COMPACT_SYSTEM
    try:
        raw = await llm_mod.llm_query(
            combined,
            system=system,
            max_tokens=600,
            json_mode=True,
            model=_CURATOR_MODEL,
            url=_CURATOR_URL,
            api_key=_CURATOR_KEY,
            stage="curator_compact_structured",
        )
        result = json.loads(strip_fences(raw))
        for key in ("goal", "progress", "decisions", "next_steps", "critical_context"):
            if key not in result:
                result[key] = "" if key in ("goal", "critical_context") else []
        return result
    except Exception as e:
        logger.warning("Structured compaction failed, falling back to flat: %s", e)
        return await _fallback_flat(combined, multi_agent=multi_agent)


async def compact_flat(texts: list[tuple[str, str]], multi_agent: bool = False) -> str:
    """Produce a flat single-paragraph summary of memory texts."""
    combined = "\n\n---\n\n".join(f"[{mid}] {t[:400]}" for mid, t in texts)
    system = FLAT_COMPACT_MULTI_AGENT if multi_agent else FLAT_COMPACT_SYSTEM
    try:
        summary = await llm_mod.llm_query(
            combined, system=system, max_tokens=300,
            model=_CURATOR_MODEL, url=_CURATOR_URL, api_key=_CURATOR_KEY,
            stage="curator_compact_flat",
        )
        return summary.strip()
    except Exception as e:
        logger.warning("Flat compaction failed: %s", e)
        return f"Compressed {len(texts)} memories."


async def _fallback_flat(combined: str, multi_agent: bool = False) -> dict:
    """Fallback when structured parsing fails — wrap flat summary in structured shell."""
    system = FLAT_COMPACT_MULTI_AGENT if multi_agent else FLAT_COMPACT_SYSTEM
    try:
        summary = await llm_mod.llm_query(
            combined, system=system, max_tokens=300,
            model=_CURATOR_MODEL, url=_CURATOR_URL, api_key=_CURATOR_KEY,
            stage="curator_compact_fallback",
        )
        return {
            "goal": "",
            "progress": [summary.strip()],
            "decisions": [],
            "next_steps": [],
            "critical_context": "",
        }
    except Exception:
        return {
            "goal": "Unable to extract",
            "progress": [],
            "decisions": [],
            "next_steps": [],
            "critical_context": "",
        }


def format_structured_summary(data: dict) -> str:
    """Render a structured compaction dict as readable markdown for storage."""
    lines = []
    if data.get("goal"):
        lines.append(f"## Goal\n{data['goal']}")
    if data.get("progress"):
        items = data["progress"]
        if isinstance(items, list):
            lines.append("## Progress\n" + "\n".join(f"- {p}" for p in items))
        else:
            lines.append(f"## Progress\n{items}")
    if data.get("decisions"):
        lines.append("## Key Decisions\n" + "\n".join(f"- {d}" for d in data["decisions"]))
    if data.get("next_steps"):
        lines.append("## Next Steps\n" + "\n".join(f"- {s}" for s in data["next_steps"]))
    if data.get("critical_context"):
        lines.append(f"## Critical Context\n{data['critical_context']}")
    return "\n\n".join(lines)
