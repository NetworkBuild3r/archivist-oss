"""Autonomous curator — consolidates daily notes, extracts entities, detects contradictions."""

import os
import re
import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config import (
    MEMORY_ROOT, CURATOR_INTERVAL_MINUTES, TEAM_MAP, QDRANT_URL, QDRANT_COLLECTION,
    CURATOR_EXTRACT_PREFIXES, CURATOR_EXTRACT_SKIP_SEGMENTS,
)
from llm import llm_query
from graph import (
    upsert_entity, add_fact, add_relationship,
    get_curator_state, set_curator_state, get_db,
)
from indexer import index_file

logger = logging.getLogger("archivist.curator")

EXTRACT_SYSTEM = (
    "You are a knowledge extraction assistant. Given a daily memory note from an AI agent, "
    "extract structured information as JSON with these fields:\n"
    "- entities: [{name, type}] (people, systems, tools, concepts, places)\n"
    "- facts: [{entity_name, fact}] (durable facts about entities — paraphrase in short plain text; "
    "NEVER paste raw code, JSON, YAML, or markdown from the source)\n"
    "- relationships: [{source, target, type, evidence}] (connections between entities)\n"
    "- decisions: [text] (decisions made)\n"
    "- lessons: [text] (lessons learned)\n"
    "Return ONLY valid JSON, no markdown fences. All string values must be short "
    "plain-English summaries — do not embed code blocks or raw config snippets."
)

_JSON_REPAIR_SYSTEM = (
    "The previous response was invalid JSON. Return ONLY the corrected version as a "
    "single valid JSON object with no markdown fences, no explanation."
)


def should_extract_knowledge(rel_path: str) -> bool:
    """Return True if the file is agent-memory content eligible for graph extraction."""
    normalised = rel_path.replace("\\", "/")
    for seg in CURATOR_EXTRACT_SKIP_SEGMENTS:
        if seg in normalised.split("/"):
            return False
    return any(normalised.startswith(prefix) for prefix in CURATOR_EXTRACT_PREFIXES)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


async def extract_knowledge(text: str, agent_id: str, source_file: str) -> dict | None:
    """Use LLM to extract structured knowledge from a memory note."""
    if len(text.strip()) < 50:
        return None

    prompt = f"Agent: {agent_id}\nSource: {source_file}\n\nMemory note:\n{text[:3000]}"
    try:
        raw = await llm_query(
            prompt, system=EXTRACT_SYSTEM, max_tokens=1024, json_mode=True,
        )
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        try:
            repair_prompt = f"Fix this invalid JSON:\n{raw[:2000]}"
            fixed = await llm_query(
                repair_prompt, system=_JSON_REPAIR_SYSTEM, max_tokens=1024, json_mode=True,
            )
            return json.loads(_strip_fences(fixed))
        except (json.JSONDecodeError, Exception) as e2:
            logger.warning("Knowledge extraction failed for %s (after retry): %s", source_file, e2)
            return None
    except Exception as e:
        logger.warning("Knowledge extraction failed for %s: %s", source_file, e)
        return None


async def process_extraction(data: dict, agent_id: str, source_file: str):
    """Store extracted knowledge in the graph."""
    for ent in data.get("entities", []):
        name = ent.get("name", "").strip()
        if name:
            upsert_entity(name, ent.get("type", "unknown"), agent_id)

    for fact in data.get("facts", []):
        ename = fact.get("entity_name", "").strip()
        ftext = fact.get("fact", "").strip()
        if ename and ftext:
            eid = upsert_entity(ename)
            add_fact(eid, ftext, source_file, agent_id)

    for rel in data.get("relationships", []):
        src = rel.get("source", "").strip()
        tgt = rel.get("target", "").strip()
        rtype = rel.get("type", "related_to").strip()
        evidence = rel.get("evidence", "").strip()
        if src and tgt:
            sid = upsert_entity(src)
            tid = upsert_entity(tgt)
            add_relationship(sid, tid, rtype, evidence, agent_id)


async def curate_cycle():
    """Run one curation cycle: scan new files, extract knowledge, update graph."""
    last_run = get_curator_state("last_curate_time")
    if last_run:
        cutoff = datetime.fromisoformat(last_run)
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

    now = datetime.now(timezone.utc)
    processed = 0

    for root, _dirs, files in os.walk(MEMORY_ROOT):
        for fname in files:
            if not fname.endswith(".md"):
                continue
            filepath = os.path.join(root, fname)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath), tz=timezone.utc)
                if mtime <= cutoff:
                    continue

                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()

                if len(text.strip()) < 50:
                    continue

                rel = os.path.relpath(filepath, MEMORY_ROOT)
                parts = Path(rel).parts
                agent_id = ""
                if "agents" in parts:
                    idx = list(parts).index("agents")
                    if idx + 1 < len(parts):
                        agent_id = parts[idx + 1]
                elif "memories" in parts:
                    idx = list(parts).index("memories")
                    if idx + 1 < len(parts):
                        agent_id = parts[idx + 1]

                data = await extract_knowledge(text, agent_id, rel) if should_extract_knowledge(rel) else None
                if data:
                    await process_extraction(data, agent_id, rel)
                    processed += 1

                await index_file(filepath)

            except Exception as e:
                logger.error("Curator failed on %s: %s", filepath, e)

    set_curator_state("last_curate_time", now.isoformat())
    logger.info("Curator cycle complete: %d files processed", processed)
    return processed


async def decay_old_entries():
    """Soft-delete graph entries not referenced in 90+ days."""
    conn = get_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    conn.execute(
        "UPDATE facts SET is_active=0 WHERE is_active=1 AND created_at < ? AND superseded_by IS NULL",
        (cutoff,),
    )
    affected = conn.total_changes
    conn.commit()
    conn.close()
    if affected:
        logger.info("Decayed %d old facts", affected)


async def curator_loop():
    """Background loop running curation cycles."""
    logger.info("Curator loop started (interval: %d min)", CURATOR_INTERVAL_MINUTES)
    while True:
        try:
            await curate_cycle()
            await decay_old_entries()
        except Exception as e:
            logger.error("Curator cycle error: %s", e)
        await asyncio.sleep(CURATOR_INTERVAL_MINUTES * 60)
