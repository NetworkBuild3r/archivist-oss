"""Autonomous curator — consolidates daily notes, extracts entities, detects contradictions."""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import UTC, datetime, timedelta

import archivist.core.metrics as m
from archivist.core.config import (
    CURATOR_EXTRACT_PREFIXES,
    CURATOR_EXTRACT_SKIP_SEGMENTS,
    CURATOR_INTERVAL_MINUTES,
    CURATOR_LLM_API_KEY,
    CURATOR_LLM_MODEL,
    CURATOR_LLM_URL,
    CURATOR_MAX_PARALLEL,
    CURATOR_TIP_BUDGET,
    DURABLE_ENTITY_TYPES,
    LLM_MODEL,
    LLM_URL,
    MEMORY_ROOT,
    ORPHAN_SWEEP_ENABLED,
    ORPHAN_SWEEP_EVERY_N_CYCLES,
)
from archivist.features.llm import llm_query

# Resolve effective curator LLM settings once at import time.
_CURATOR_MODEL = CURATOR_LLM_MODEL or LLM_MODEL
_CURATOR_URL = CURATOR_LLM_URL or LLM_URL
_CURATOR_KEY = CURATOR_LLM_API_KEY
from archivist.core.hotness import batch_update_hotness
from archivist.core.trajectory import consolidate_tips
from archivist.storage.compressed_index import cache_wake_up
from archivist.storage.graph import (
    add_fact,
    add_relationship,
    get_curator_state,
    set_curator_state,
    upsert_entity,
)
from archivist.utils.text_utils import extract_agent_id_from_path, strip_fences
from archivist.write.indexer import index_file
from archivist.write.pre_extractor import pre_extract

logger = logging.getLogger("archivist.curator")

EXTRACT_SYSTEM = (
    "You are a knowledge extraction assistant. Given a daily memory note from an AI agent, "
    "extract structured information as JSON with these fields:\n"
    "- entities: [{name, type}] (people, systems, tools, concepts, places). "
    "Use specific types: person, host, server, service, credential, organization, cluster, "
    "database, network, user, tool, concept, place, project.\n"
    "- facts: [{entity_name, fact, valid_from, valid_until}] (durable facts about entities — "
    "paraphrase in short plain text; NEVER paste raw code, JSON, YAML, or markdown from the source). "
    "Include valid_from (ISO date YYYY-MM-DD) when the text indicates when the fact became true "
    '(e.g. "as of", "starting from", "since", "deployed on"). '
    "Include valid_until when the text indicates the fact is no longer true "
    '(e.g. "until", "ended", "deprecated", "replaced by"). '
    "Omit these fields (or use empty string) when no temporal marker is present.\n"
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


async def extract_knowledge(text: str, agent_id: str, source_file: str) -> dict | None:
    """Use LLM to extract structured knowledge from a memory note.

    Runs a deterministic pre-extraction pass first to provide hints to the LLM,
    reducing hallucination and token spend (two-pass approach).
    """
    if len(text.strip()) < 50:
        return None

    hints = pre_extract(text, source_file)
    hint_lines = []
    if hints["entities"]:
        hint_lines.append(f"Pre-extracted entities: {json.dumps(hints['entities'][:20])}")
    if hints["dates"]:
        hint_lines.append(f"Detected dates: {hints['dates']}")
    if hints["thought_type"] != "general":
        hint_lines.append(f"Detected thought type: {hints['thought_type']}")
    hint_block = "\n".join(hint_lines)

    prompt = (
        f"Agent: {agent_id}\nSource: {source_file}\n\n"
        + (f"{hint_block}\n\n" if hint_block else "")
        + f"Memory note:\n{text[:3000]}"
    )
    try:
        raw = await llm_query(
            prompt,
            system=EXTRACT_SYSTEM,
            max_tokens=1024,
            json_mode=True,
            model=_CURATOR_MODEL,
            url=_CURATOR_URL,
            api_key=_CURATOR_KEY,
            stage="curator_extract",
        )
        return json.loads(strip_fences(raw))
    except json.JSONDecodeError:
        try:
            repair_prompt = f"Fix this invalid JSON:\n{raw[:2000]}"
            fixed = await llm_query(
                repair_prompt,
                system=_JSON_REPAIR_SYSTEM,
                max_tokens=1024,
                json_mode=True,
                model=_CURATOR_MODEL,
                url=_CURATOR_URL,
                api_key=_CURATOR_KEY,
                stage="curator_json_repair",
            )
            return json.loads(strip_fences(fixed))
        except (json.JSONDecodeError, Exception) as e2:
            logger.warning("Knowledge extraction failed for %s (after retry): %s", source_file, e2)
            return None
    except Exception as e:
        logger.warning("Knowledge extraction failed for %s: %s", source_file, e)
        return None


def _retention_for_entity_type(entity_type: str) -> str:
    """Map entity type to a default retention class."""
    if entity_type.lower() in DURABLE_ENTITY_TYPES:
        return "durable"
    return "standard"


async def process_extraction(data: dict, agent_id: str, source_file: str):
    """Store extracted knowledge in the graph."""
    from archivist.core.rbac import get_namespace_for_agent

    _ns = get_namespace_for_agent(agent_id) if agent_id else "global"

    entity_retention: dict[str, str] = {}
    for ent in data.get("entities", []):
        name = ent.get("name", "").strip()
        if name:
            etype = ent.get("type", "unknown")
            rc = _retention_for_entity_type(etype)
            entity_retention[name.lower()] = rc
            await upsert_entity(name, etype, agent_id, retention_class=rc, namespace=_ns)

    file_date = ""
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", source_file)
    if m:
        file_date = m.group(1)

    for fact in data.get("facts", []):
        ename = fact.get("entity_name", "").strip()
        ftext = fact.get("fact", "").strip()
        if ename and ftext:
            rc = entity_retention.get(ename.lower(), "standard")
            eid = await upsert_entity(ename, retention_class=rc, namespace=_ns)
            vf = (fact.get("valid_from") or "").strip()
            vu = (fact.get("valid_until") or "").strip()
            if not vf and file_date:
                vf = file_date
            await add_fact(
                eid,
                ftext,
                source_file,
                agent_id,
                retention_class=rc,
                valid_from=vf,
                valid_until=vu,
                namespace=_ns,
            )

    for rel in data.get("relationships", []):
        src = rel.get("source", "").strip()
        tgt = rel.get("target", "").strip()
        rtype = rel.get("type", "related_to").strip()
        evidence = rel.get("evidence", "").strip()
        provenance = rel.get("provenance", "inferred").strip()
        if src and tgt:
            sid = await upsert_entity(src, namespace=_ns)
            tid = await upsert_entity(tgt, namespace=_ns)
            await add_relationship(
                sid, tid, rtype, evidence, agent_id, provenance=provenance, namespace=_ns
            )


def _file_checksum(text: str) -> str:
    """SHA-256 of file content for change detection."""
    return hashlib.sha256(text.encode()).hexdigest()


async def curate_cycle():
    """Run one curation cycle: scan new files, extract knowledge, update graph.

    Uses mtime as a fast first pass, then content checksum to skip files
    whose content hasn't actually changed (e.g. touch, metadata-only update).

    Files are processed concurrently (up to CURATOR_MAX_PARALLEL) to overlap
    LLM extraction latency.
    """
    last_run = await get_curator_state("last_curate_time")
    if last_run:
        cutoff = datetime.fromisoformat(last_run)
    else:
        cutoff = datetime.now(UTC) - timedelta(days=7)

    now = datetime.now(UTC)
    processed = 0
    skipped_unchanged = 0

    candidates: list[tuple[str, str, str]] = []  # (filepath, rel, text)
    for root, _dirs, files in os.walk(MEMORY_ROOT):
        for fname in files:
            if not fname.endswith(".md"):
                continue
            filepath = os.path.join(root, fname)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath), tz=UTC)
                if mtime <= cutoff:
                    continue

                with open(filepath, encoding="utf-8", errors="replace") as f:
                    text = f.read()

                if len(text.strip()) < 50:
                    continue

                rel = os.path.relpath(filepath, MEMORY_ROOT)

                current_checksum = _file_checksum(text)
                stored_checksum = await get_curator_state(f"checksum:{rel}")
                if stored_checksum == current_checksum:
                    skipped_unchanged += 1
                    continue

                candidates.append((filepath, rel, text))
            except Exception as e:
                logger.error("Curator scan failed on %s: %s", filepath, e)

    sem = asyncio.Semaphore(max(1, CURATOR_MAX_PARALLEL))

    async def _process_one(filepath: str, rel: str, text: str) -> bool:
        async with sem:
            try:
                agent_id = extract_agent_id_from_path(rel)
                data = (
                    await extract_knowledge(text, agent_id, rel)
                    if should_extract_knowledge(rel)
                    else None
                )
                if data:
                    await process_extraction(data, agent_id, rel)
                await index_file(filepath)
                await set_curator_state(f"checksum:{rel}", _file_checksum(text))
                return data is not None
            except Exception as e:
                logger.error("Curator failed on %s: %s", filepath, e)
                return False

    results = await asyncio.gather(
        *(_process_one(fp, rel, txt) for fp, rel, txt in candidates),
        return_exceptions=True,
    )
    for r in results:
        if r is True:
            processed += 1

    await set_curator_state("last_curate_time", now.isoformat())
    return {"processed": processed, "skipped": skipped_unchanged}


async def extract_all_agent_memories() -> int:
    """Run knowledge extraction on every eligible agent markdown file under MEMORY_ROOT.

    Ignores mtime/checksum (for benchmarks and backfills). Does not run decay.
    Processes files concurrently up to CURATOR_MAX_PARALLEL.
    """
    candidates: list[tuple[str, str, str]] = []
    for root, _dirs, files in os.walk(MEMORY_ROOT):
        for fname in files:
            if not fname.endswith(".md"):
                continue
            filepath = os.path.join(root, fname)
            try:
                with open(filepath, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                if len(text.strip()) < 50:
                    continue
                rel = os.path.relpath(filepath, MEMORY_ROOT)
                if not should_extract_knowledge(rel):
                    continue
                candidates.append((filepath, rel, text))
            except Exception as e:
                logger.error("Bench curator scan failed on %s: %s", filepath, e)

    sem = asyncio.Semaphore(max(1, CURATOR_MAX_PARALLEL))

    async def _extract_one(filepath: str, rel: str, text: str) -> bool:
        async with sem:
            try:
                agent_id = extract_agent_id_from_path(rel)
                data = await extract_knowledge(text, agent_id, rel)
                if data:
                    await process_extraction(data, agent_id, rel)
                    return True
            except Exception as e:
                logger.error("Bench curator failed on %s: %s", filepath, e)
            return False

    results = await asyncio.gather(
        *(_extract_one(fp, rel, txt) for fp, rel, txt in candidates),
        return_exceptions=True,
    )
    processed = sum(1 for r in results if r is True)
    logger.info("extract_all_agent_memories: processed %d files", processed)
    return processed


async def reinforce_durable_entities():
    """Touch last_seen on durable/permanent entities so they stay fresh in temporal decay.

    Also boosts confidence on relationships involving these entities.
    This runs every curator cycle and is cheap (pure SQL, no LLM).
    """
    from archivist.storage.sqlite_pool import pool

    now = datetime.now(UTC).isoformat()
    async with pool.write() as conn:
        cur = await conn.execute(
            "UPDATE entities SET last_seen=? WHERE retention_class IN ('durable', 'permanent')",
            (now,),
        )
        reinforced = cur.rowcount

        cur2 = await conn.execute(
            "UPDATE relationships SET updated_at=?, confidence=MIN(confidence+0.01, 1.0) "
            "WHERE source_entity_id IN (SELECT id FROM entities WHERE retention_class IN ('durable','permanent')) "
            "OR target_entity_id IN (SELECT id FROM entities WHERE retention_class IN ('durable','permanent'))",
            (now,),
        )
        rels_boosted = cur2.rowcount

    if reinforced:
        logger.info(
            "Reinforced %d durable/permanent entities, %d relationships", reinforced, rels_boosted
        )


async def decay_old_entries() -> dict[str, int]:
    """Soft-delete graph entries based on age and superseded status.

    Rules:
    - durable/permanent facts: never decayed
    - superseded standard/ephemeral facts: decay after 30 days
    - non-superseded standard facts: decay after 90 days
    - ephemeral facts: decay after 30 days regardless

    Returns counts for observability (curator.cycle summary).
    """
    from archivist.storage.sqlite_pool import pool

    async with pool.write() as conn:
        now = datetime.now(UTC)

        cutoff_90 = (now - timedelta(days=90)).isoformat()
        cur1 = await conn.execute(
            "UPDATE facts SET is_active=0 "
            "WHERE is_active=1 AND created_at < ? AND superseded_by IS NULL "
            "AND retention_class NOT IN ('durable', 'permanent')",
            (cutoff_90,),
        )
        aged_out = cur1.rowcount

        cutoff_30 = (now - timedelta(days=30)).isoformat()
        cur2 = await conn.execute(
            "UPDATE facts SET is_active=0 "
            "WHERE is_active=1 AND created_at < ? "
            "AND (superseded_by IS NOT NULL OR retention_class = 'ephemeral') "
            "AND retention_class NOT IN ('durable', 'permanent')",
            (cutoff_30,),
        )
        superseded_out = cur2.rowcount

    total = aged_out + superseded_out
    if total:
        logger.info(
            "Decayed %d facts (%d aged 90d, %d superseded/ephemeral 30d; durable/permanent preserved)",
            total, aged_out, superseded_out,
        )
    return {"aged_out": aged_out, "superseded_out": superseded_out, "total": total}


def _refresh_wake_up_caches() -> int:
    """Re-build wake-up context for each active namespace.

    Scans memory_chunks for distinct (namespace, agent_id) pairs and
    pre-computes the wake-up payload so the MCP tool returns instantly.

    Returns the number of namespace/agent pairs refreshed.
    """
    from archivist.storage.graph import get_db

    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT DISTINCT namespace, agent_id FROM memory_chunks WHERE namespace != '' LIMIT 500"
        ).fetchall()
    finally:
        conn.close()

    refreshed = 0
    seen: set[str] = set()
    for row in rows:
        ns = row["namespace"]
        aid = row["agent_id"]
        key = f"{ns}:{aid}"
        if key in seen:
            continue
        seen.add(key)
        try:
            cache_wake_up(ns, agent_id=aid)
            refreshed += 1
        except Exception as e:
            logger.debug("Wake-up cache failed for %s/%s: %s", ns, aid, e)

    if refreshed:
        logger.info("Refreshed wake-up context for %d namespace/agent pairs", refreshed)
    return refreshed


async def curator_loop():
    """Background loop running curation cycles with exponential backoff on failure."""
    base_interval = CURATOR_INTERVAL_MINUTES * 60
    backoff_sec = base_interval
    max_backoff = 3600
    _sweep_counter = 0
    logger.info("Curator loop started (interval: %d min)", CURATOR_INTERVAL_MINUTES)
    while True:
        try:
            t_iter = time.monotonic()
            cc = await curate_cycle()
            files_processed = cc["processed"]
            files_skipped = cc["skipped"]
            await reinforce_durable_entities()
            decay = await decay_old_entries()
            facts_decayed = decay["total"]

            n_hot = 0
            try:
                n_hot = batch_update_hotness()
                if n_hot:
                    logger.info("Hotness update: %d memories scored", n_hot)
            except Exception as e:
                logger.warning("Hotness update failed (non-fatal): %s", e)

            tips_merged = 0
            try:
                tip_result = await consolidate_tips(budget=CURATOR_TIP_BUDGET)
                tips_merged = int(tip_result.get("consolidated", 0) or 0)
                if tips_merged:
                    logger.info("Tip consolidation: %d clusters merged", tips_merged)
            except Exception as e:
                logger.warning("Tip consolidation failed (non-fatal): %s", e)

            wake_pairs = 0
            try:
                wake_pairs = _refresh_wake_up_caches()
            except Exception as e:
                logger.warning("Wake-up cache refresh failed (non-fatal): %s", e)

            orphans_cleaned = 0
            _sweep_counter += 1
            if ORPHAN_SWEEP_ENABLED and _sweep_counter >= ORPHAN_SWEEP_EVERY_N_CYCLES:
                _sweep_counter = 0
                try:
                    from archivist.lifecycle.cascade import sweep_orphans

                    sr = await sweep_orphans()
                    orphans_cleaned = sr.get("fts_cleaned", 0) + sr.get("needle_cleaned", 0)
                    if orphans_cleaned:
                        logger.info("Orphan sweep cleaned %d rows", orphans_cleaned)
                except Exception as e:
                    logger.warning("Orphan sweep failed (non-fatal): %s", e)

            dur_ms = round((time.monotonic() - t_iter) * 1000, 1)
            m.observe(m.CURATOR_CYCLE_DURATION, dur_ms)
            logger.info(
                "curator.cycle complete=1 files_processed=%d skipped=%d facts_decayed=%d "
                "hotness_updated=%d tips_merged=%d wake_pairs=%d duration_ms=%.1f",
                files_processed,
                files_skipped,
                facts_decayed,
                n_hot,
                tips_merged,
                wake_pairs,
                dur_ms,
            )

            backoff_sec = base_interval
        except Exception as e:
            logger.error("Curator cycle error (next retry in %ds): %s", backoff_sec, e)
            backoff_sec = min(backoff_sec * 2, max_backoff)
        await asyncio.sleep(backoff_sec)
