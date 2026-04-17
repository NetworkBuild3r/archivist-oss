"""Merge and deduplicate hits from multi-agent vector search."""

import hashlib


def dedupe_vector_hits(hits: list[dict]) -> list[dict]:
    """Drop near-duplicate chunks (same file + chunk index, or same text prefix).

    When a synthetic question point and its source chunk both match, they share
    the same (file_path, chunk_index, text[:240]) key.  We keep the copy with
    the higher score so the best semantic match wins regardless of which
    representation produced it.  The winner is tagged with
    ``synthetic_match=True`` if either copy was a synthetic question hit.
    """
    best: dict[str, dict] = {}
    for h in hits:
        fp = str(h.get("file_path", ""))
        idx = h.get("chunk_index", 0)
        text = str(h.get("text", ""))[:240]
        key = hashlib.sha256(f"{fp}\0{idx}\0{text}".encode()).hexdigest()

        is_synth = h.get("representation_type") == "synthetic_question"

        existing = best.get(key)
        if existing is None:
            entry = dict(h)
            if is_synth:
                entry["synthetic_match"] = True
            best[key] = entry
        else:
            if is_synth:
                existing["synthetic_match"] = True
            if h.get("score", 0) > existing.get("score", 0):
                new_entry = dict(h)
                new_entry["synthetic_match"] = existing.get("synthetic_match", False) or is_synth
                best[key] = new_entry

    return list(best.values())
