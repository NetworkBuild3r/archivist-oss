"""Structured conversation compaction — produces Goal/Progress/Decisions/Next Steps summaries.

Inspired by ReMe's compact_memory() structured output format.
"""

import json
import logging
import re

from llm import llm_query

logger = logging.getLogger("archivist.compaction")

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

FLAT_COMPACT_SYSTEM = (
    "Summarize these memory entries into a single concise summary (200 tokens max). "
    "Preserve key facts, entities, and actionable insights."
)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


async def compact_structured(
    texts: list[tuple[str, str]],
    previous_summary: str = "",
) -> dict:
    """Produce a structured compaction of memory texts.

    Args:
        texts: List of (memory_id, text) tuples to compact.
        previous_summary: Optional prior structured summary JSON to merge with.

    Returns:
        dict with goal, progress, decisions, next_steps, critical_context.
    """
    parts = []
    for mid, text in texts:
        parts.append(f"[{mid}] {text[:800]}")

    combined = "\n\n---\n\n".join(parts)

    if previous_summary:
        combined = f"[Previous summary]\n{previous_summary}\n\n---\n\n[New memories]\n{combined}"

    try:
        raw = await llm_query(
            combined,
            system=STRUCTURED_COMPACT_SYSTEM,
            max_tokens=600,
            json_mode=True,
        )
        result = json.loads(_strip_fences(raw))
        for key in ("goal", "progress", "decisions", "next_steps", "critical_context"):
            if key not in result:
                result[key] = "" if key in ("goal", "critical_context") else []
        return result
    except Exception as e:
        logger.warning("Structured compaction failed, falling back to flat: %s", e)
        return await _fallback_flat(combined)


async def compact_flat(texts: list[tuple[str, str]]) -> str:
    """Produce a flat single-paragraph summary of memory texts."""
    combined = "\n\n---\n\n".join(f"[{mid}] {t[:400]}" for mid, t in texts)
    try:
        summary = await llm_query(combined, system=FLAT_COMPACT_SYSTEM, max_tokens=300)
        return summary.strip()
    except Exception as e:
        logger.warning("Flat compaction failed: %s", e)
        return f"Compressed {len(texts)} memories."


async def _fallback_flat(combined: str) -> dict:
    """Fallback when structured parsing fails — wrap flat summary in structured shell."""
    try:
        summary = await llm_query(combined, system=FLAT_COMPACT_SYSTEM, max_tokens=300)
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
