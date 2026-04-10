"""Reciprocal Rank Fusion (RRF) — merge heterogeneous ranked lists.

RRF is provably better than linear score combination for merging signals
with different score distributions (e.g. vector cosine vs BM25 rank).

Formula: RRF_score(d) = sum(1 / (k + rank_i(d))) for each ranking list i.

Used by:
  - Vector + BM25 fusion (replaces old linear 0.7v + 0.3b)
  - Multi-query expansion merge
  - BM25 AND + OR mode merge
"""


def rrf_merge(
    rankings: list[list[dict]],
    k: int = 60,
    id_key: str = "",
    limit: int = 0,
) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each ranking is a list of result dicts sorted by that signal's score
    (best first).  Results are identified by ``id_key`` if given, else
    by a compound key of (``id``, ``file_path``, ``text[:80]``).

    Returns a single merged list sorted by RRF score (descending).
    Each result dict keeps its original fields plus ``rrf_score``.
    """
    if not rankings:
        return []
    if len(rankings) == 1:
        out = []
        for r in rankings[0]:
            entry = dict(r)
            entry["rrf_score"] = 1.0 / (k + 1)
            out.append(entry)
        return out[:limit] if limit else out

    scores: dict[str, float] = {}
    best_copy: dict[str, dict] = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            rid = _result_id(result, id_key)
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
            if rid not in best_copy or result.get("score", 0) > best_copy[rid].get("score", 0):
                best_copy[rid] = result

    merged = []
    for rid, rrf_score in scores.items():
        entry = dict(best_copy[rid])
        entry["rrf_score"] = round(rrf_score, 6)
        merged.append(entry)

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged[:limit] if limit else merged


def _result_id(result: dict, id_key: str) -> str:
    if id_key and result.get(id_key):
        return str(result[id_key])
    rid = result.get("id") or result.get("qdrant_id") or ""
    if rid:
        return str(rid)
    fp = result.get("file_path", "")
    ci = result.get("chunk_index", 0)
    text_prefix = (result.get("text") or "")[:80]
    return f"{fp}:{ci}:{text_prefix}"
