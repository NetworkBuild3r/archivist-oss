"""File chunker and indexer — watches directories, embeds, upserts to Qdrant.

Phase 1 addition: hierarchical parent-child chunking for richer retrieval context.
"""

import asyncio
import os
import re
import uuid
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from config import (
    QDRANT_COLLECTION, MEMORY_ROOT,
    TEAM_MAP,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    TIERED_CONTEXT_ENABLED,
    BM25_ENABLED, TOPIC_ROUTING_ENABLED,
    CONTEXTUAL_AUGMENTATION_ENABLED,
)
from chunking import chunk_text, chunk_text_hierarchical
from embeddings import embed_batch
from graph import upsert_fts_chunk, delete_fts_chunks_by_file, upsert_entity, add_fact, register_needle_tokens
from qdrant import qdrant_client
from rbac import get_namespace_for_agent, get_namespace_config
from text_utils import extract_agent_id_from_path, compute_memory_checksum
from tiering import generate_tiers
from topic_detector import detect_topics
from pre_extractor import pre_extract, extract_needle_entities
from collection_router import ensure_collection, collections_for_query
from contextual_augment import augment_chunk
import metrics as _metrics

logger = logging.getLogger("archivist.indexer")

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


# ── Metadata extraction ───────────────────────────────────────────────────────

def _extract_metadata(filepath: str) -> dict:
    """Extract metadata from file path: agent_id, date, file_type, team, namespace."""
    agent_id = ""
    file_type = "unknown"
    date_str = ""
    team = "unknown"

    rel = os.path.relpath(filepath, MEMORY_ROOT)
    rel_parts = Path(rel).parts

    agent_id = extract_agent_id_from_path(rel)
    if agent_id:
        team = TEAM_MAP.get(agent_id, "unknown")

    if "memories" in rel_parts:
        file_type = "legacy"

    fname = Path(filepath).stem
    m = _DATE_RE.search(fname)
    if m:
        date_str = m.group(1)
        file_type = "daily"
    elif fname.upper() == "MEMORY":
        file_type = "durable"
    elif "weekly" in str(filepath).lower() or fname.startswith("20") and "-W" in fname:
        file_type = "weekly"
    elif fname.upper() in ("IDENTITY", "SOUL", "TOOLS", "USER", "HEARTBEAT", "AGENTS"):
        file_type = "system"

    namespace = get_namespace_for_agent(agent_id) if agent_id else "default"
    indexed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return {
        "agent_id": agent_id,
        "content_date": date_str,
        "indexed_at": indexed_at,
        "date": date_str or indexed_at,
        "file_type": file_type,
        "team": team,
        "namespace": namespace,
        "file_path": rel,
    }


# ── Point ID generation ──────────────────────────────────────────────────────

def _point_id(filepath: str, chunk_idx: int) -> str:
    """Deterministic UUID from file path + chunk index."""
    h = hashlib.md5(f"{filepath}:{chunk_idx}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def compute_ttl(namespace: str, importance: float = 0.5) -> int | None:
    """Compute TTL expiration timestamp based on namespace config."""
    if importance >= 0.9:
        return None
    ns_config = get_namespace_config(namespace)
    if ns_config and ns_config.ttl_days is not None:
        expires = datetime.now(timezone.utc) + timedelta(days=ns_config.ttl_days)
        return int(expires.timestamp())
    return None


# ── Indexing ──────────────────────────────────────────────────────────────────

async def index_file(filepath: str, hierarchical: bool = True) -> int:
    """Index a single .md file into Qdrant. Returns number of points upserted.

    Args:
        filepath: Path to the markdown file.
        hierarchical: If True (default), use parent-child chunking (Phase 1).
                      If False, use flat chunking.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        logger.warning("Failed to read %s: %s", filepath, e)
        return 0

    if len(text.strip()) < 20:
        return 0

    await delete_file_points(filepath)

    meta = _extract_metadata(filepath)
    ns_config = get_namespace_config(meta["namespace"])
    consistency = ns_config.consistency if ns_config else "eventual"
    ttl_expires_at = compute_ttl(meta["namespace"])
    client = qdrant_client()

    if hierarchical:
        hier_chunks = chunk_text_hierarchical(
            text,
            filepath,
            parent_size=PARENT_CHUNK_SIZE,
            parent_overlap=PARENT_CHUNK_OVERLAP,
            child_size=CHILD_CHUNK_SIZE,
            child_overlap=CHILD_CHUNK_OVERLAP,
        )
        if not hier_chunks:
            return 0

        # Generate tiered summaries for parent chunks
        tier_map: dict[str, dict] = {}
        if TIERED_CONTEXT_ENABLED:
            for c in hier_chunks:
                if c["is_parent"]:
                    tier_map[c["id"]] = await generate_tiers(c["content"])

        contents = [c["content"] for c in hier_chunks]
        augmented_contents = contents
        if CONTEXTUAL_AUGMENTATION_ENABLED:
            _hints_for_augment = [pre_extract(c["content"]) for c in hier_chunks]
            augmented_contents = [
                augment_chunk(
                    c["content"],
                    agent_id=meta.get("agent_id", ""),
                    file_path=meta.get("file_path", ""),
                    date=meta.get("date", ""),
                    topic="",
                    hints=h,
                )
                for c, h in zip(hier_chunks, _hints_for_augment)
            ]
        vectors = await embed_batch(augmented_contents)

        points = []
        _hints_by_id: dict[str, dict] = {}
        for i, (chunk_meta, vec) in enumerate(zip(hier_chunks, vectors)):
            pid = chunk_meta["id"] if chunk_meta["id"] else _point_id(filepath, i)
            checksum = compute_memory_checksum(chunk_meta["content"], meta["agent_id"], meta["namespace"])

            tiers = tier_map.get(chunk_meta["id"], {})
            # Children inherit their parent's L0/L1 as context hint
            if not chunk_meta["is_parent"] and chunk_meta["parent_id"]:
                tiers = tier_map.get(chunk_meta["parent_id"], {})

            topics = detect_topics(chunk_meta["content"]) if TOPIC_ROUTING_ENABLED else []
            hints = _hints_for_augment[i] if CONTEXTUAL_AUGMENTATION_ENABLED else pre_extract(chunk_meta["content"])
            _hints_by_id[pid] = hints
            payload = {
                **meta,
                "chunk_index": i,
                "text": chunk_meta["content"],
                "text_augmented": augmented_contents[i] if CONTEXTUAL_AUGMENTATION_ENABLED else "",
                "l0": tiers.get("l0", ""),
                "l1": tiers.get("l1", ""),
                "parent_id": chunk_meta["parent_id"],
                "is_parent": chunk_meta["is_parent"],
                "topic": topics[0] if topics else "",
                "thought_type": hints.get("thought_type", "general"),
                "version": 1,
                "consistency_level": consistency,
                "checksum": checksum,
                "importance_score": 0.5,
                "retention_class": "standard",
            }
            if ttl_expires_at is not None:
                payload["ttl_expires_at"] = ttl_expires_at

            points.append(PointStruct(id=pid, vector=vec, payload=payload))

    else:
        chunks = chunk_text(text)
        if not chunks:
            return 0

        embed_texts = chunks
        augmented_flat: list[str] = []
        if CONTEXTUAL_AUGMENTATION_ENABLED:
            _hints_flat = [pre_extract(c) for c in chunks]
            augmented_flat = [
                augment_chunk(
                    c,
                    agent_id=meta.get("agent_id", ""),
                    file_path=meta.get("file_path", ""),
                    date=meta.get("date", ""),
                    hints=h,
                )
                for c, h in zip(chunks, _hints_flat)
            ]
            embed_texts = augmented_flat
        vectors = await embed_batch(embed_texts)

        points = []
        _hints_by_id: dict[str, dict] = {}
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = _point_id(filepath, i)
            checksum = compute_memory_checksum(chunk, meta["agent_id"], meta["namespace"])

            topics = detect_topics(chunk) if TOPIC_ROUTING_ENABLED else []
            hints = _hints_flat[i] if CONTEXTUAL_AUGMENTATION_ENABLED else pre_extract(chunk)
            _hints_by_id[pid] = hints
            payload = {
                **meta,
                "chunk_index": i,
                "text": chunk,
                "text_augmented": augmented_flat[i] if augmented_flat else "",
                "topic": topics[0] if topics else "",
                "thought_type": hints.get("thought_type", "general"),
                "version": 1,
                "consistency_level": consistency,
                "checksum": checksum,
                "importance_score": 0.5,
                "retention_class": "standard",
            }
            if ttl_expires_at is not None:
                payload["ttl_expires_at"] = ttl_expires_at

            points.append(PointStruct(id=pid, vector=vec, payload=payload))

    if points:
        _coll = ensure_collection(meta.get("namespace", ""))
        client.upsert(collection_name=_coll, points=points)
        _metrics.inc(_metrics.INDEX_CHUNKS, value=len(points))

        if BM25_ENABLED:
            for p in points:
                upsert_fts_chunk(
                    qdrant_id=str(p.id),
                    text=p.payload.get("text", ""),
                    file_path=p.payload.get("file_path", ""),
                    chunk_index=p.payload.get("chunk_index", 0),
                    agent_id=p.payload.get("agent_id", ""),
                    namespace=p.payload.get("namespace", ""),
                    date=p.payload.get("date", ""),
                    memory_type=p.payload.get("memory_type", "general"),
                )

        _agent = meta.get("agent_id", "")
        _src_file = meta.get("file_path", "")
        _seen_entity_names: set[str] = set()
        for p in points:
            if not p.payload.get("is_parent", False):
                continue
            _hints = _hints_by_id.get(str(p.id), {})
            _needle_ents = extract_needle_entities(p.payload.get("text", ""))
            for ent in _hints.get("entities", []) + _needle_ents:
                ename = ent["name"].strip()
                if ename and ename not in _seen_entity_names:
                    _seen_entity_names.add(ename)
                    etype = ent.get("type", "unknown")
                    _eid = upsert_entity(ename, etype)
                    add_fact(_eid, p.payload.get("text", "")[:200], _src_file, _agent)

        for p in points:
            register_needle_tokens(
                str(p.id), p.payload.get("text", ""),
                namespace=meta.get("namespace", ""),
                agent_id=_agent,
            )

        # Reverse HyDE: generate hypothetical questions for parent chunks (parallel)
        from config import REVERSE_HYDE_ENABLED
        if REVERSE_HYDE_ENABLED:
            from hyde import generate_reverse_hyde_questions
            _rh_semaphore = asyncio.Semaphore(3)
            _parent_points = [p for p in points if p.payload.get("is_parent", False)]

            async def _gen_rh_for_point(p):
                async with _rh_semaphore:
                    try:
                        _rh_qs = await generate_reverse_hyde_questions(p.payload.get("text", ""))
                        if not _rh_qs:
                            return []
                        _rh_vecs = await embed_batch(_rh_qs)
                        result = []
                        for qi, (q, qv) in enumerate(zip(_rh_qs, _rh_vecs)):
                            _q_id = str(uuid.uuid4())
                            result.append(PointStruct(
                                id=_q_id,
                                vector=qv,
                                payload={
                                    **meta,
                                    "text": p.payload.get("text", ""),
                                    "chunk_index": 0,
                                    "file_type": "reverse_hyde",
                                    "source_memory_id": str(p.id),
                                    "is_reverse_hyde": True,
                                    "reverse_hyde_question": q,
                                    "parent_id": str(p.id),
                                    "is_parent": False,
                                    "importance_score": 0.5,
                                    "retention_class": "standard",
                                    "version": 1,
                                },
                            ))
                        return result
                    except Exception as e:
                        logger.warning("Reverse HyDE failed for chunk %s: %s", p.id, e)
                        return []

            _rh_batches = await asyncio.gather(
                *[_gen_rh_for_point(p) for p in _parent_points],
                return_exceptions=True,
            )
            _rh_points = []
            for batch in _rh_batches:
                if isinstance(batch, BaseException):
                    logger.warning("Reverse HyDE gather error: %s", batch)
                    continue
                _rh_points.extend(batch)
            if _rh_points:
                client.upsert(collection_name=_coll, points=_rh_points)
                _metrics.inc(_metrics.INDEX_CHUNKS, value=len(_rh_points))

        logger.info("Indexed %s: %d chunks (ns=%s, hierarchical=%s, fts=%s)",
                     meta["file_path"], len(points), meta["namespace"], hierarchical, BM25_ENABLED)

    return len(points)


async def delete_file_points(filepath: str):
    """Remove all points for a given file path from Qdrant and FTS5."""
    rel = os.path.relpath(filepath, MEMORY_ROOT)
    client = qdrant_client()
    for _coll in collections_for_query(""):
        try:
            client.delete(
                collection_name=_coll,
                points_selector=Filter(
                    must=[FieldCondition(key="file_path", match=MatchValue(value=rel))]
                ),
            )
        except Exception:
            pass
    if BM25_ENABLED:
        delete_fts_chunks_by_file(rel)


_SKIP_DIRS = {"node_modules", ".git", ".cache", "__pycache__", ".pnpm", "dist", "build"}


async def full_index(hierarchical: bool = True) -> int:
    """Walk all .md files under MEMORY_ROOT and index them."""
    total = 0
    for root, dirs, files in os.walk(MEMORY_ROOT):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            if not fname.endswith(".md"):
                continue
            filepath = os.path.join(root, fname)
            try:
                count = await index_file(filepath, hierarchical=hierarchical)
                total += count
            except Exception as e:
                logger.error("Failed to index %s: %s", filepath, e)
    logger.info("Full index complete: %d total chunks", total)
    return total
