"""File chunker and indexer — watches directories, embeds, upserts to Qdrant.

Phase 1 addition: hierarchical parent-child chunking for richer retrieval context.
"""

import asyncio
import hashlib
import logging
import os
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

import archivist.core.metrics as _metrics
from archivist.core.config import (
    BM25_ENABLED,
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHUNKING_STRATEGY,
    CONTEXTUAL_AUGMENTATION_ENABLED,
    MEMORY_ROOT,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    TEAM_MAP,
    TIERED_CONTEXT_ENABLED,
    TOPIC_ROUTING_ENABLED,
)
from archivist.core.rbac import get_namespace_config, get_namespace_for_agent
from archivist.features.embeddings import embed_batch
from archivist.retrieval.topic_detector import detect_topics
from archivist.storage.collection_router import collections_for_query, ensure_collection
from archivist.storage.graph import (
    add_fact,
    delete_fts_chunks_by_file,
    register_memory_points_batch,
    upsert_entity,
)
from archivist.storage.qdrant import qdrant_client
from archivist.utils.chunking import chunk_text, chunk_text_hierarchical
from archivist.utils.text_utils import compute_memory_checksum, extract_agent_id_from_path
from archivist.write.contextual_augment import augment_chunk
from archivist.write.pre_extractor import extract_needle_entities, pre_extract
from archivist.write.tiering import generate_tiers

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from archivist.storage.transaction import MemoryTransaction

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
    elif "weekly" in str(filepath).lower() or (fname.startswith("20") and "-W" in fname):
        file_type = "weekly"
    elif fname.upper() in ("IDENTITY", "SOUL", "TOOLS", "USER", "HEARTBEAT", "AGENTS"):
        file_type = "system"

    namespace = get_namespace_for_agent(agent_id) if agent_id else "default"
    indexed_at = datetime.now(UTC).strftime("%Y-%m-%d")

    return {
        "agent_id": agent_id,
        "content_date": date_str,
        "indexed_at": indexed_at,
        "date": date_str or indexed_at,
        "file_type": file_type,
        "team": team,
        "namespace": namespace,
        "file_path": rel,
        "actor_id": "file_indexer",
        "actor_type": "system",
    }


# ── Point ID generation ──────────────────────────────────────────────────────


def _point_id(filepath: str, chunk_idx: int) -> str:
    """Deterministic UUID from file path + chunk index."""
    h = hashlib.md5(f"{filepath}:{chunk_idx}".encode(), usedforsecurity=False).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _tier_label_for(payload: dict) -> str:
    """Derive the tier_label from a chunk payload.

    A chunk whose Qdrant payload already has ``l0`` text is classified as 'l0',
    one with ``l1`` text is 'l1', otherwise defaults to 'l2'.  The labels map
    directly to the Phase-1 tier taxonomy.
    """
    if payload.get("l0"):
        return "l0"
    if payload.get("l1"):
        return "l1"
    return "l2"


def compute_ttl(namespace: str, importance: float = 0.5) -> int | None:
    """Compute TTL expiration timestamp based on namespace config."""
    if importance >= 0.9:
        return None
    ns_config = get_namespace_config(namespace)
    if ns_config and ns_config.ttl_days is not None:
        expires = datetime.now(UTC) + timedelta(days=ns_config.ttl_days)
        return int(expires.timestamp())
    return None


# ── Shared persistence helper (INIT-022/SPEC-001, M13 dedup) ─────────────────


async def _persist_points(
    coll: str,
    points: list[PointStruct],
    mp_records: list[dict],
    memory_id: str,
    *,
    force_transaction: bool = False,
    extra_txn_work: "Callable[[MemoryTransaction], Awaitable[None]] | None" = None,
) -> None:
    """Persist ``points`` + their ``memory_points`` rows for one write-path call site.

    Unifies the "insert memory_points rows, then enqueue-or-apply the Qdrant upsert"
    pattern previously duplicated across the primary-points, reverse-HyDE, and
    synthetic-questions write paths (INIT-022/SPEC-001, M13).

    Args:
        coll: Target Qdrant collection name.
        points: Points to upsert.
        mp_records: Rows to insert into ``memory_points``, each a dict with
            ``memory_id``, ``qdrant_id``, and ``point_type`` keys.
        memory_id: Correlation ID recorded on the enqueued outbox event.
        force_transaction: If ``True``, always run the ``MemoryTransaction`` (so
            ``extra_txn_work`` and the ``memory_points`` insert always execute),
            applying the Qdrant write inline afterward when ``OUTBOX_ENABLED`` is
            ``False`` — matches the primary-points call site's existing branching.
            If ``False`` (default), the whole persist step branches on
            ``OUTBOX_ENABLED`` — matches the reverse-HyDE / synthetic-questions
            call sites' existing branching.
        extra_txn_work: Optional coroutine run inside the open transaction before
            the ``memory_points`` insert (e.g. FTS5 + needle-token registration
            for the primary-points call site).

    Raises:
        Exception: Re-raises any failure from the transaction commit or the
            legacy (non-transactional) ``register_memory_points_batch`` call, so
            a write-path failure is surfaced to the caller instead of being
            silently reported as success (INIT-022/SPEC-001, C1 fix).
    """
    from archivist.core.config import OUTBOX_ENABLED
    from archivist.storage.transaction import MemoryTransaction

    now_iso = datetime.now(UTC).isoformat()
    mp_rows = [(r["memory_id"], r["qdrant_id"], r["point_type"], now_iso) for r in mp_records]

    async def _run_transaction() -> None:
        async with MemoryTransaction() as txn:
            if extra_txn_work is not None:
                await extra_txn_work(txn)
            # memory_points and the outbox event join the same transaction —
            # all commit atomically or none do.
            await txn.executemany(
                """INSERT OR IGNORE INTO memory_points
                       (memory_id, qdrant_id, point_type, created_at)
                   VALUES (?, ?, ?, ?)""",
                mp_rows,
            )
            txn.enqueue_qdrant_upsert(coll, points, memory_id=memory_id)

    if force_transaction:
        try:
            await _run_transaction()
        except Exception:
            logger.error(
                "indexer._persist_points: transaction failed",
                exc_info=True,
                extra={"collection": coll, "memory_id": memory_id, "point_count": len(points)},
            )
            raise
        if not OUTBOX_ENABLED:
            qdrant_client().upsert(collection_name=coll, points=points)
        return

    if OUTBOX_ENABLED:
        try:
            await _run_transaction()
        except Exception:
            logger.error(
                "indexer._persist_points: transaction failed",
                exc_info=True,
                extra={"collection": coll, "memory_id": memory_id, "point_count": len(points)},
            )
            raise
    else:
        qdrant_client().upsert(collection_name=coll, points=points)
        try:
            await register_memory_points_batch(mp_records)
        except Exception:
            logger.error(
                "indexer._persist_points: register_memory_points_batch failed",
                exc_info=True,
                extra={"collection": coll, "memory_id": memory_id, "point_count": len(points)},
            )
            raise


# ── Indexing ──────────────────────────────────────────────────────────────────


async def index_file(filepath: str, hierarchical: bool = True) -> int:
    """Index a single .md file into Qdrant. Returns number of points upserted.

    Args:
        filepath: Path to the markdown file.
        hierarchical: If True (default), use parent-child chunking (Phase 1).
                      If False, use flat chunking.
    """
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        logger.warning("Failed to read %s: %s", filepath, e)
        return 0

    if len(text.strip()) < 20:
        return 0

    await delete_file_points(filepath)

    meta = _extract_metadata(filepath)
    from archivist.core.provenance import SourceTrace, default_confidence

    _indexer_confidence = default_confidence("system")
    _indexer_trace = SourceTrace(tool="file_indexer", upstream_source=filepath).to_dict()
    ns_config = get_namespace_config(meta["namespace"])
    consistency = ns_config.consistency if ns_config else "eventual"
    ttl_expires_at = compute_ttl(meta["namespace"])

    if hierarchical:
        hier_chunks = chunk_text_hierarchical(
            text,
            filepath,
            parent_size=PARENT_CHUNK_SIZE,
            parent_overlap=PARENT_CHUNK_OVERLAP,
            child_size=CHILD_CHUNK_SIZE,
            child_overlap=CHILD_CHUNK_OVERLAP,
            strategy=CHUNKING_STRATEGY,
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

        _chunk_hints = [pre_extract(c["content"]) for c in hier_chunks]
        _chunk_topics = []
        if TOPIC_ROUTING_ENABLED:
            _chunk_topics = [detect_topics(c["content"]) for c in hier_chunks]

        augmented_contents = contents
        if CONTEXTUAL_AUGMENTATION_ENABLED:
            augmented_contents = [
                augment_chunk(
                    c["content"],
                    agent_id=meta.get("agent_id", ""),
                    file_path=meta.get("file_path", ""),
                    date=meta.get("date", ""),
                    topic=(_chunk_topics[i][0] if _chunk_topics and _chunk_topics[i] else ""),
                    thought_type=_chunk_hints[i].get("thought_type", "general"),
                )
                for i, c in enumerate(hier_chunks)
            ]
        vectors = await embed_batch(augmented_contents)

        # Build parent_text lookup so children carry their parent's text at index time
        _parent_text_map: dict[str, str] = {
            c["id"]: c["content"] for c in hier_chunks if c["is_parent"]
        }

        points = []
        _hints_by_id: dict[str, dict] = {}
        for i, (chunk_meta, vec) in enumerate(zip(hier_chunks, vectors)):
            pid = chunk_meta["id"] if chunk_meta["id"] else _point_id(filepath, i)
            checksum = compute_memory_checksum(
                chunk_meta["content"], meta["agent_id"], meta["namespace"]
            )

            tiers = tier_map.get(chunk_meta["id"], {})
            # Children inherit their parent's L0/L1 as context hint
            if not chunk_meta["is_parent"] and chunk_meta["parent_id"]:
                tiers = tier_map.get(chunk_meta["parent_id"], {})

            parent_text = ""
            if not chunk_meta["is_parent"] and chunk_meta["parent_id"]:
                parent_text = _parent_text_map.get(chunk_meta["parent_id"], "")

            topics = (
                _chunk_topics[i]
                if _chunk_topics
                else (detect_topics(chunk_meta["content"]) if TOPIC_ROUTING_ENABLED else [])
            )
            hints = _chunk_hints[i]
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
                "parent_text": parent_text,
                "topic": topics[0] if topics else "",
                "thought_type": hints.get("thought_type", "general"),
                "representation_type": "chunk",
                "version": 1,
                "consistency_level": consistency,
                "checksum": checksum,
                "importance_score": 0.5,
                "retention_class": "standard",
                "confidence": _indexer_confidence,
                "source_trace": _indexer_trace,
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
            augmented_flat = [
                augment_chunk(
                    c,
                    agent_id=meta.get("agent_id", ""),
                    file_path=meta.get("file_path", ""),
                    date=meta.get("date", ""),
                )
                for c in chunks
            ]
            embed_texts = augmented_flat
        vectors = await embed_batch(embed_texts)

        points = []
        _hints_by_id: dict[str, dict] = {}
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = _point_id(filepath, i)
            checksum = compute_memory_checksum(chunk, meta["agent_id"], meta["namespace"])

            topics = detect_topics(chunk) if TOPIC_ROUTING_ENABLED else []
            hints = pre_extract(chunk)
            _hints_by_id[pid] = hints
            payload = {
                **meta,
                "chunk_index": i,
                "text": chunk,
                "text_augmented": augmented_flat[i] if augmented_flat else "",
                "topic": topics[0] if topics else "",
                "thought_type": hints.get("thought_type", "general"),
                "parent_text": "",
                "representation_type": "chunk",
                "version": 1,
                "consistency_level": consistency,
                "checksum": checksum,
                "importance_score": 0.5,
                "retention_class": "standard",
                "confidence": _indexer_confidence,
                "source_trace": _indexer_trace,
            }
            if ttl_expires_at is not None:
                payload["ttl_expires_at"] = ttl_expires_at

            points.append(PointStruct(id=pid, vector=vec, payload=payload))

    if points:
        _coll = ensure_collection(meta.get("namespace", ""))

        _mp_records = []
        for p in points:
            pid_str = str(p.id)
            parent = p.payload.get("parent_id")
            if parent:
                _mp_records.append(
                    {"memory_id": parent, "qdrant_id": pid_str, "point_type": "micro_chunk"}
                )
            else:
                _mp_records.append(
                    {"memory_id": pid_str, "qdrant_id": pid_str, "point_type": "primary"}
                )

        async def _primary_extra_txn_work(txn: "MemoryTransaction") -> None:
            """Register FTS5 chunks (if enabled) and needle tokens for all points."""
            if BM25_ENABLED:
                for p in points:
                    await txn.upsert_fts_chunk(
                        qdrant_id=str(p.id),
                        text=p.payload.get("text", ""),
                        file_path=p.payload.get("file_path", ""),
                        chunk_index=p.payload.get("chunk_index", 0),
                        agent_id=p.payload.get("agent_id", ""),
                        namespace=p.payload.get("namespace", ""),
                        date=p.payload.get("date", ""),
                        memory_type=p.payload.get("memory_type", "general"),
                        actor_id=p.payload.get("actor_id", ""),
                        actor_type=p.payload.get("actor_type", ""),
                        importance=float(p.payload.get("importance_score", 0.5)),
                        tier_label=_tier_label_for(p.payload),
                    )
            for p in points:
                await txn.register_needle_tokens(
                    str(p.id),
                    p.payload.get("text", ""),
                    namespace=meta.get("namespace", ""),
                    agent_id=meta.get("agent_id", ""),
                    actor_id=meta.get("actor_id", ""),
                    actor_type=meta.get("actor_type", ""),
                )

        await _persist_points(
            _coll,
            points,
            _mp_records,
            meta.get("file_path", ""),
            force_transaction=True,
            extra_txn_work=_primary_extra_txn_work,
        )

        _metrics.inc(_metrics.INDEX_CHUNKS, value=len(points))

        _agent = meta.get("agent_id", "")
        _src_file = meta.get("file_path", "")
        _ns = meta.get("namespace", "")
        _actor_id = meta.get("actor_id", "")
        _actor_type = meta.get("actor_type", "")

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
                    _eid = await upsert_entity(
                        ename,
                        etype,
                        namespace=_ns or "global",
                        actor_id=_actor_id,
                        actor_type=_actor_type,
                    )
                    await add_fact(
                        _eid,
                        p.payload.get("text", "")[:200],
                        _src_file,
                        _agent,
                        namespace=_ns or "global",
                        memory_id=str(p.id),
                        confidence=_indexer_confidence,
                        provenance="file_indexer",
                        actor_id=_actor_id,
                    )

        # Reverse HyDE: generate hypothetical questions for parent chunks (parallel)
        from archivist.core.config import REVERSE_HYDE_ENABLED

        if REVERSE_HYDE_ENABLED:
            from archivist.write.hyde import generate_reverse_hyde_questions

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
                            result.append(
                                PointStruct(
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
                                )
                            )
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
                _rh_mp_records = [
                    {
                        "memory_id": rp.payload.get("source_memory_id", str(rp.id)),
                        "qdrant_id": str(rp.id),
                        "point_type": "reverse_hyde",
                    }
                    for rp in _rh_points
                ]
                await _persist_points(_coll, _rh_points, _rh_mp_records, meta.get("file_path", ""))
                _metrics.inc(_metrics.INDEX_CHUNKS, value=len(_rh_points))

        # Synthetic question generation: multi-representation indexing
        import archivist.core.config as _cfg

        if _cfg.SYNTHETIC_QUESTIONS_ENABLED:
            from archivist.write.synthetic_questions import generate_and_embed_synthetic_points

            _sq_semaphore = asyncio.Semaphore(3)
            _parent_points_sq = [p for p in points if p.payload.get("is_parent", False)]

            async def _gen_sq_for_point(p):
                async with _sq_semaphore:
                    try:
                        base_payload = {
                            k: v
                            for k, v in p.payload.items()
                            if k not in ("text_augmented", "l0", "l1", "checksum")
                        }
                        return await generate_and_embed_synthetic_points(
                            chunk_point_id=str(p.id),
                            chunk_text=p.payload.get("text", ""),
                            base_payload=base_payload,
                        )
                    except Exception as e:
                        logger.warning("Synthetic questions failed for chunk %s: %s", p.id, e)
                        return []

            _sq_batches = await asyncio.gather(
                *[_gen_sq_for_point(p) for p in _parent_points_sq],
                return_exceptions=True,
            )
            _sq_points: list[PointStruct] = []
            for batch in _sq_batches:
                if isinstance(batch, BaseException):
                    logger.warning("Synthetic questions gather error: %s", batch)
                    continue
                _sq_points.extend(batch)
            if _sq_points:
                _sq_mp_records = [
                    {
                        "memory_id": sp.payload.get("source_memory_id", str(sp.id)),
                        "qdrant_id": str(sp.id),
                        "point_type": "synthetic_question",
                    }
                    for sp in _sq_points
                ]
                await _persist_points(_coll, _sq_points, _sq_mp_records, meta.get("file_path", ""))
                _metrics.inc(_metrics.INDEX_CHUNKS, value=len(_sq_points))
                logger.info(
                    "Indexed %d synthetic question points for %s",
                    len(_sq_points),
                    meta["file_path"],
                )

        logger.info(
            "Indexed %s: %d chunks (ns=%s, hierarchical=%s, fts=%s)",
            meta["file_path"],
            len(points),
            meta["namespace"],
            hierarchical,
            BM25_ENABLED,
        )

    return len(points)


async def delete_file_points(filepath: str) -> None:
    """Remove all points for a given file path from Qdrant and FTS5."""
    rel = os.path.relpath(filepath, MEMORY_ROOT)
    client = qdrant_client()
    from archivist.core.config import OUTBOX_ENABLED

    for _coll in collections_for_query(""):
        if OUTBOX_ENABLED:
            try:
                from archivist.storage.transaction import MemoryTransaction

                _filt = Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=rel))])
                async with MemoryTransaction() as txn:
                    txn.enqueue_qdrant_delete_filter(
                        _coll,
                        _filt.model_dump(mode="json"),
                    )
            except Exception as _e:
                # Stale points may remain in `_coll` until the next delete/reindex —
                # log loudly so this is diagnosable rather than silently swallowed
                # (INIT-022/SPEC-001, H8 fix).
                logger.warning(
                    "indexer.delete_file_points: Qdrant delete-by-filter failed (outbox)",
                    exc_info=True,
                    extra={"collection": _coll, "file_path": rel, "error": str(_e)},
                )
        else:
            try:
                client.delete(
                    collection_name=_coll,
                    points_selector=Filter(
                        must=[FieldCondition(key="file_path", match=MatchValue(value=rel))]
                    ),
                )
            except Exception as _e:
                logger.warning(
                    "indexer.delete_file_points: Qdrant delete-by-filter failed (legacy)",
                    exc_info=True,
                    extra={"collection": _coll, "file_path": rel, "error": str(_e)},
                )
    if BM25_ENABLED:
        await delete_fts_chunks_by_file(rel)


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
