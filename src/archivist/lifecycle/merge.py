"""CRDT-style merge strategies for conflicting memories."""

import hashlib
import logging
import os
from datetime import UTC, datetime

from qdrant_client.models import PointStruct

from archivist.core.audit import log_memory_event
from archivist.core.config import MEMORY_ROOT, QDRANT_COLLECTION
from archivist.features.embeddings import embed_text
from archivist.features.llm import llm_query
from archivist.storage.collection_router import collection_for
from archivist.storage.qdrant import qdrant_client
from archivist.storage.versioning import record_version
from archivist.utils.text_utils import compute_memory_checksum

logger = logging.getLogger("archivist.merge")

MERGE_SYSTEM = (
    "You are a memory merge assistant. Given multiple versions of a memory, "
    "produce a single coherent merged version that preserves all unique information. "
    "If versions contradict each other, note both perspectives with dates. "
    "Be concise and factual. Output ONLY the merged text."
)


async def merge_memories(
    memory_ids: list[str],
    strategy: str,
    agent_id: str,
    namespace: str = "",
) -> dict:
    """Merge multiple memory points using the specified strategy."""
    client = qdrant_client()

    points = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=memory_ids,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        return {"error": "No points found for the given memory IDs"}

    payloads = [(p.id, p.payload) for p in points]
    payloads.sort(key=lambda x: x[1].get("date", ""))

    if strategy == "latest":
        result = _merge_latest(payloads)
    elif strategy == "concat":
        result = _merge_concat(payloads)
    elif strategy == "semantic":
        result = await _merge_semantic(payloads)
    elif strategy == "manual":
        return await _merge_manual(payloads, namespace)
    else:
        return {"error": f"Unknown merge strategy: {strategy}"}

    merged_text = result["merged_text"]
    vec = await embed_text(merged_text)
    merged_id = hashlib.md5(f"merge:{':'.join(memory_ids)}".encode()).hexdigest()
    checksum = compute_memory_checksum(merged_text, agent_id, namespace)

    ns = namespace or payloads[0][1].get("namespace", "default")
    merged_payload = {
        "agent_id": agent_id,
        "text": merged_text,
        "file_path": f"merge/{agent_id}",
        "file_type": "merged",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "team": payloads[0][1].get("team", "unknown"),
        "namespace": ns,
        "version": result["version"],
        "consistency_level": payloads[0][1].get("consistency_level", "eventual"),
        "checksum": checksum,
        "parent_id": memory_ids[0],
        "importance_score": max(p.get("importance_score", 0.5) for _, p in payloads),
        "chunk_index": 0,
    }

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[PointStruct(id=merged_id, vector=vec, payload=merged_payload)],
    )

    from archivist.lifecycle.memory_lifecycle import delete_memory_complete

    for mid in memory_ids:
        await delete_memory_complete(mid, ns, collection=collection_for(ns))

    version = record_version(
        merged_id,
        agent_id,
        checksum,
        "merge",
        parent_versions=[],
    )

    await log_memory_event(
        agent_id=agent_id,
        action="merge",
        memory_id=merged_id,
        namespace=ns,
        text_hash=checksum,
        version=version,
        metadata={
            "strategy": strategy,
            "merged_ids": memory_ids,
            "conflicts_resolved": len(memory_ids) - 1,
        },
    )

    return {
        "merged_id": merged_id,
        "strategy": strategy,
        "merged_count": len(memory_ids),
        "version": version,
    }


def _merge_latest(payloads: list[tuple[str, dict]]) -> dict:
    latest = payloads[-1]
    return {
        "merged_text": latest[1]["text"],
        "version": latest[1].get("version", 1) + 1,
    }


def _merge_concat(payloads: list[tuple[str, dict]]) -> dict:
    texts = []
    for pid, p in payloads:
        header = f"[{p.get('date', '?')} — {p.get('agent_id', '?')}]"
        texts.append(f"{header}\n{p['text']}")
    merged = "\n\n---\n\n".join(texts)
    return {
        "merged_text": merged,
        "version": max(p.get("version", 1) for _, p in payloads) + 1,
    }


async def _merge_semantic(payloads: list[tuple[str, dict]]) -> dict:
    versions_text = "\n\n".join(
        f"Version (date={p.get('date', '?')}, agent={p.get('agent_id', '?')}):\n{p['text']}"
        for _, p in payloads
    )
    prompt = f"Merge the following memory versions into a single coherent fact:\n\n{versions_text}\n\nMerged version:"

    try:
        merged = await llm_query(prompt, system=MERGE_SYSTEM, max_tokens=1024)
    except Exception as e:
        logger.error("Semantic merge LLM call failed: %s", e)
        merged = _merge_concat(payloads)["merged_text"]

    return {
        "merged_text": merged,
        "version": max(p.get("version", 1) for _, p in payloads) + 1,
    }


async def _merge_manual(payloads: list[tuple[str, dict]], namespace: str) -> dict:
    """Flag for human review — write conflict details to MEMORY_ROOT."""
    conflict_dir = os.path.join(MEMORY_ROOT, "memory-conflicts")
    os.makedirs(conflict_dir, exist_ok=True)

    conflict_id = hashlib.md5(":".join(str(pid) for pid, _ in payloads).encode()).hexdigest()[:12]
    filepath = os.path.join(conflict_dir, f"conflict-{conflict_id}.md")

    lines = [f"# Memory Conflict — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}", ""]
    lines.append(f"**Namespace:** {namespace}")
    lines.append(f"**Memory IDs:** {', '.join(str(pid) for pid, _ in payloads)}")
    lines.append("")
    for pid, p in payloads:
        lines.append(f"## Version: {p.get('agent_id', '?')} @ {p.get('date', '?')}")
        lines.append(f"```\n{p['text']}\n```")
        lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return {
        "manual_review_required": True,
        "conflict_file": filepath,
        "memory_ids": [str(pid) for pid, _ in payloads],
    }
