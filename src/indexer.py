"""File chunker and indexer — watches directories, embeds, upserts to Qdrant.

Phase 1 addition: hierarchical parent-child chunking for richer retrieval context.
"""

import os
import re
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from config import (
    QDRANT_URL, QDRANT_COLLECTION, MEMORY_ROOT,
    CHUNK_SIZE, CHUNK_OVERLAP, TEAM_MAP,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    TIERED_CONTEXT_ENABLED,
)
from chunking import chunk_text, chunk_text_hierarchical
from config import BM25_ENABLED
from embeddings import embed_batch
from graph import upsert_fts_chunk, delete_fts_chunks_by_file
from rbac import get_namespace_for_agent, get_namespace_config
from tiering import generate_tiers

logger = logging.getLogger("archivist.indexer")

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=30)


# ── Metadata extraction ───────────────────────────────────────────────────────

def _extract_metadata(filepath: str) -> dict:
    """Extract metadata from file path: agent_id, date, file_type, team, namespace."""
    agent_id = ""
    file_type = "unknown"
    date_str = ""
    team = "unknown"

    rel = os.path.relpath(filepath, MEMORY_ROOT)
    rel_parts = Path(rel).parts

    if "agents" in rel_parts:
        idx = list(rel_parts).index("agents")
        if idx + 1 < len(rel_parts):
            agent_id = rel_parts[idx + 1]
            team = TEAM_MAP.get(agent_id, "unknown")

    if "memories" in rel_parts:
        idx = list(rel_parts).index("memories")
        if idx + 1 < len(rel_parts):
            agent_id = agent_id or rel_parts[idx + 1]
            team = TEAM_MAP.get(agent_id, "unknown")
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

    return {
        "agent_id": agent_id,
        "date": date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
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


def _compute_ttl(namespace: str, importance: float = 0.5) -> int | None:
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
    ttl_expires_at = _compute_ttl(meta["namespace"])
    client = _client()

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
        vectors = await embed_batch(contents)

        points = []
        for i, (chunk_meta, vec) in enumerate(zip(hier_chunks, vectors)):
            pid = chunk_meta["id"] if chunk_meta["id"] else _point_id(filepath, i)
            checksum = hashlib.sha256(
                f"{chunk_meta['content']}:{meta['agent_id']}:{meta['namespace']}".encode()
            ).hexdigest()

            tiers = tier_map.get(chunk_meta["id"], {})
            # Children inherit their parent's L0/L1 as context hint
            if not chunk_meta["is_parent"] and chunk_meta["parent_id"]:
                tiers = tier_map.get(chunk_meta["parent_id"], {})

            payload = {
                **meta,
                "chunk_index": i,
                "text": chunk_meta["content"],
                "l0": tiers.get("l0", ""),
                "l1": tiers.get("l1", ""),
                "parent_id": chunk_meta["parent_id"],
                "is_parent": chunk_meta["is_parent"],
                "version": 1,
                "consistency_level": consistency,
                "checksum": checksum,
                "importance_score": 0.5,
            }
            if ttl_expires_at is not None:
                payload["ttl_expires_at"] = ttl_expires_at

            points.append(PointStruct(id=pid, vector=vec, payload=payload))

    else:
        chunks = chunk_text(text)
        if not chunks:
            return 0

        vectors = await embed_batch(chunks)

        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = _point_id(filepath, i)
            checksum = hashlib.sha256(
                f"{chunk}:{meta['agent_id']}:{meta['namespace']}".encode()
            ).hexdigest()

            payload = {
                **meta,
                "chunk_index": i,
                "text": chunk,
                "version": 1,
                "consistency_level": consistency,
                "checksum": checksum,
                "importance_score": 0.5,
            }
            if ttl_expires_at is not None:
                payload["ttl_expires_at"] = ttl_expires_at

            points.append(PointStruct(id=pid, vector=vec, payload=payload))

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

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

        logger.info("Indexed %s: %d chunks (ns=%s, hierarchical=%s, fts=%s)",
                     meta["file_path"], len(points), meta["namespace"], hierarchical, BM25_ENABLED)

    return len(points)


async def delete_file_points(filepath: str):
    """Remove all points for a given file path from Qdrant and FTS5."""
    rel = os.path.relpath(filepath, MEMORY_ROOT)
    client = _client()
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="file_path", match=MatchValue(value=rel))]
        ),
    )
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
