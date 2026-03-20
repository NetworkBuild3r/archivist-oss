"""Merge and deduplicate hits from multi-agent vector search."""

import hashlib


def dedupe_vector_hits(hits: list[dict]) -> list[dict]:
    """Drop near-duplicate chunks (same file + chunk index, or same text prefix)."""
    seen: set[str] = set()
    out: list[dict] = []
    for h in hits:
        fp = str(h.get("file_path", ""))
        idx = h.get("chunk_index", 0)
        text = str(h.get("text", ""))[:240]
        key = hashlib.sha256(f"{fp}\0{idx}\0{text}".encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out
