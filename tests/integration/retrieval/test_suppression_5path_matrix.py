"""5-path suppression exclusion matrix (INIT-025/SPEC-011).

Verifies that suppressed memories (``is_suppressed = 1`` / ``True``) are
excluded across all 5 Archivist retrieval and read paths, while active memories
in the same namespace remain visible and explicit ``unsuppress_memory()``
restores visibility:

  1. Vector search (``rlm_retriever.search_vectors``)
  2. BM25 / FTS search (``search_fts`` + ``search_fts_exact``)
  3. Graph / Entity facts (``get_entity_facts`` + ``get_entity_facts_bulk``)
  4. Context API (``get_relevant_context`` normal & bootstrap + ``create_handoff_packet``)
  5. Timeline (``tools_search._handle_timeline``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

pytestmark = [pytest.mark.integration]

_NS = "suppress_matrix_ns"
_AGENT = "matrix-agent"
_ACTIVE_ID = "matrix-active-001"
_SUPPRESSED_ID = "matrix-suppressed-002"
_QUERY_TOKEN = "quasar_telemetry_beacon"
_ACTIVE_TEXT = f"Active {_QUERY_TOKEN} primary operational runbook"
_SUPPRESSED_TEXT = f"Suppressed {_QUERY_TOKEN} retracted obsolete note"


@dataclass
class _FakePoint:
    id: str
    score: float
    payload: dict[str, Any]


def _matches_condition(payload: dict[str, Any], cond: Any) -> bool:
    key = getattr(cond, "key", None)
    match = getattr(cond, "match", None)
    if key is None or match is None:
        return True
    val = payload.get(key)
    if hasattr(match, "value") and match.value is not None:
        return val == match.value
    if hasattr(match, "any") and match.any is not None:
        return val in match.any
    if hasattr(match, "text") and match.text is not None:
        return str(match.text).lower() in str(val or "").lower()
    return True


def _matches_filter(payload: dict[str, Any], qfilter: Any) -> bool:
    if qfilter is None:
        return True
    for cond in getattr(qfilter, "must", None) or []:
        if not _matches_condition(payload, cond):
            return False
    for cond in getattr(qfilter, "must_not", None) or []:
        if _matches_condition(payload, cond):
            return False
    return True


class _FilterEvaluatingQdrantStub:
    """In-memory Qdrant stub that faithfully evaluates ``must`` / ``must_not`` filters."""

    def __init__(self, points: list[_FakePoint]) -> None:
        self.points = {p.id: p for p in points}
        self.last_query_filter: Any = None

    def _filtered(self, qfilter: Any, limit: int) -> list[_FakePoint]:
        return [p for p in self.points.values() if _matches_filter(p.payload, qfilter)][:limit]

    def query_points(
        self,
        collection_name: str,
        query: Any = None,
        query_filter: Any = None,
        limit: int = 10,
        with_payload: bool = True,
        **kwargs: Any,
    ) -> Any:
        self.last_query_filter = query_filter
        matched = self._filtered(query_filter, limit)

        class _Resp:
            points = matched

        return _Resp()

    def search(
        self,
        collection_name: str,
        query_vector: Any = None,
        query_filter: Any = None,
        limit: int = 10,
        with_payload: bool = True,
        **kwargs: Any,
    ) -> list[_FakePoint]:
        self.last_query_filter = query_filter
        return self._filtered(query_filter, limit)

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 10,
        with_payload: bool = True,
        **kwargs: Any,
    ) -> tuple[list[_FakePoint], None]:
        return self._filtered(scroll_filter, limit), None

    def set_payload(
        self,
        collection_name: str,
        payload: dict[str, Any],
        points: Any,
        **kwargs: Any,
    ) -> None:
        if isinstance(points, list):
            for pid in points:
                if pid in self.points:
                    self.points[pid].payload.update(payload)
        else:
            for p in self._filtered(points, limit=1000):
                p.payload.update(payload)


@pytest.fixture
async def seeded_5path_env(async_pool, monkeypatch):
    """Seed one active and one suppressed memory across SQLite + Qdrant stub."""
    from archivist.lifecycle.correct import suppress_memory
    from archivist.storage.graph import add_fact, upsert_entity, upsert_fts_chunk

    today = datetime.now(UTC).strftime("%Y-%m-%d")

    await upsert_fts_chunk(
        qdrant_id=_ACTIVE_ID,
        text=_ACTIVE_TEXT,
        file_path=f"agents/{_AGENT}/active.md",
        chunk_index=0,
        agent_id=_AGENT,
        namespace=_NS,
        date=today,
        memory_type="fact",
        importance=0.9,
        tier_label="l1",
    )
    await upsert_fts_chunk(
        qdrant_id=_SUPPRESSED_ID,
        text=_SUPPRESSED_TEXT,
        file_path=f"agents/{_AGENT}/suppressed.md",
        chunk_index=0,
        agent_id=_AGENT,
        namespace=_NS,
        date=today,
        memory_type="fact",
        importance=0.9,
        tier_label="l1",
    )

    entity_id = await upsert_entity("QuasarBeaconService", "service", namespace=_NS)
    active_fact_id = await add_fact(
        entity_id,
        "active_status",
        f"Active {_QUERY_TOKEN} operational",
        namespace=_NS,
        memory_id=_ACTIVE_ID,
    )
    suppressed_fact_id = await add_fact(
        entity_id,
        "suppressed_status",
        f"Suppressed {_QUERY_TOKEN} retracted",
        namespace=_NS,
        memory_id=_SUPPRESSED_ID,
    )

    fake_qdrant = _FilterEvaluatingQdrantStub(
        [
            _FakePoint(
                id=_ACTIVE_ID,
                score=0.95,
                payload={
                    "text": _ACTIVE_TEXT,
                    "l0": _ACTIVE_TEXT,
                    "l1": _ACTIVE_TEXT,
                    "agent_id": _AGENT,
                    "namespace": _NS,
                    "file_path": f"agents/{_AGENT}/active.md",
                    "date": today,
                    "memory_type": "fact",
                    "is_suppressed": False,
                    "archived": False,
                    "deleted": False,
                },
            ),
            _FakePoint(
                id=_SUPPRESSED_ID,
                score=0.99,
                payload={
                    "text": _SUPPRESSED_TEXT,
                    "l0": _SUPPRESSED_TEXT,
                    "l1": _SUPPRESSED_TEXT,
                    "agent_id": _AGENT,
                    "namespace": _NS,
                    "file_path": f"agents/{_AGENT}/suppressed.md",
                    "date": today,
                    "memory_type": "fact",
                    "is_suppressed": False,
                    "archived": False,
                    "deleted": False,
                },
            ),
        ]
    )

    async def _fake_embed(*_args: Any, **_kwargs: Any) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    async def _fake_llm(*_args: Any, **_kwargs: Any) -> str:
        return "Mocked context summary."

    monkeypatch.setattr("archivist.retrieval.rlm_retriever.qdrant_client", lambda: fake_qdrant)
    monkeypatch.setattr("archivist.retrieval.rlm_retriever.embed_text", _fake_embed)
    monkeypatch.setattr("archivist.features.embeddings.embed_text", _fake_embed)
    monkeypatch.setattr("archivist.features.llm.llm_query", _fake_llm)
    monkeypatch.setattr("archivist.retrieval.rlm_retriever.llm_query", _fake_llm, raising=False)
    monkeypatch.setattr("archivist.lifecycle.correct.qdrant_client", lambda: fake_qdrant)
    monkeypatch.setattr("archivist.lifecycle.cascade.qdrant_client", lambda: fake_qdrant)
    monkeypatch.setattr("archivist.core.config.BM25_ENABLED", True)

    # Seed hotness rows for both memories so handoff packet selection must filter suppressed rows.
    now_iso = datetime.now(UTC).isoformat()
    async with async_pool.write() as conn:
        for mid, score in ((_ACTIVE_ID, 0.9), (_SUPPRESSED_ID, 0.99)):
            await conn.execute(
                "INSERT INTO memory_hotness (memory_id, score, retrieval_count, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (mid, score, 5, now_iso),
            )

    # Suppress via canonical lifecycle API (sets memory_chunks.is_suppressed=1 + Qdrant payload)
    # and mark the corresponding entity fact suppressed.
    res = await suppress_memory(_SUPPRESSED_ID, _NS, reason="matrix test suppression")
    assert res["status"] == "suppressed"
    async with async_pool.write() as conn:
        await conn.execute(
            "UPDATE facts SET is_suppressed = 1 WHERE id = ?",
            (suppressed_fact_id,),
        )

    return {
        "pool": async_pool,
        "qdrant": fake_qdrant,
        "entity_id": entity_id,
        "active_fact_id": active_fact_id,
        "suppressed_fact_id": suppressed_fact_id,
    }


async def test_path1_vector_search_excludes_suppressed(seeded_5path_env) -> None:
    """Path 1 (Vector): search_vectors excludes suppressed points via must_not filter."""
    from archivist.retrieval.rlm_retriever import search_vectors

    hits = await search_vectors(_QUERY_TOKEN, namespace=_NS, limit=10)
    ids = {h["id"] for h in hits}
    assert _ACTIVE_ID in ids
    assert _SUPPRESSED_ID not in ids

    # Also verify the Qdrant filter explicitly carried is_suppressed=True in must_not.
    qfilter = seeded_5path_env["qdrant"].last_query_filter
    must_not = getattr(qfilter, "must_not", None) or []
    assert any(
        getattr(c, "key", None) == "is_suppressed"
        and getattr(getattr(c, "match", None), "value", None) is True
        for c in must_not
    )


async def test_path2_bm25_stemmed_and_exact_exclude_suppressed(seeded_5path_env) -> None:
    """Path 2 (BM25): search_fts and search_fts_exact hide suppressed chunks until unsuppressed."""
    from archivist.lifecycle.correct import unsuppress_memory
    from archivist.storage.graph import search_fts, search_fts_exact

    stemmed_ids = {r["qdrant_id"] for r in await search_fts(_QUERY_TOKEN, namespace=_NS)}
    exact_ids = {r["qdrant_id"] for r in await search_fts_exact(_QUERY_TOKEN, namespace=_NS)}
    assert _ACTIVE_ID in stemmed_ids
    assert _SUPPRESSED_ID not in stemmed_ids
    assert _ACTIVE_ID in exact_ids
    assert _SUPPRESSED_ID not in exact_ids

    # Explicit unsuppress restores recall on BM25.
    restored = await unsuppress_memory(_SUPPRESSED_ID, _NS, reason="matrix restore check")
    assert restored["status"] == "unsuppressed"
    after_ids = {r["qdrant_id"] for r in await search_fts(_QUERY_TOKEN, namespace=_NS)}
    assert _SUPPRESSED_ID in after_ids


async def test_path3_graph_entity_facts_exclude_suppressed(seeded_5path_env) -> None:
    """Path 3 (Graph): get_entity_facts and get_entity_facts_bulk omit suppressed facts."""
    from archivist.storage.graph import get_entity_facts, get_entity_facts_bulk

    eid = seeded_5path_env["entity_id"]
    facts = await get_entity_facts(eid)
    fact_ids = {f["id"] for f in facts}
    assert seeded_5path_env["active_fact_id"] in fact_ids
    assert seeded_5path_env["suppressed_fact_id"] not in fact_ids

    bulk = await get_entity_facts_bulk([eid])
    bulk_ids = {f["id"] for f in bulk[eid]}
    assert seeded_5path_env["active_fact_id"] in bulk_ids
    assert seeded_5path_env["suppressed_fact_id"] not in bulk_ids


async def test_path4_context_api_and_handoff_exclude_suppressed(seeded_5path_env) -> None:
    """Path 4 (Context API): get_relevant_context (normal + bootstrap) and handoff omit suppressed."""
    from archivist.retrieval.context_api import create_handoff_packet, get_relevant_context

    normal_ctx = await get_relevant_context(
        agent_id=_AGENT,
        task_description=_QUERY_TOKEN,
        namespace=_NS,
        include_graph=True,
        include_tips=False,
        include_hotness=False,
        mode="normal",
    )
    normal_texts = " ".join(
        str(m.get("text", "")) + " " + str(m.get("id", "")) for m in normal_ctx.memories
    )
    assert _SUPPRESSED_ID not in normal_texts
    assert _SUPPRESSED_ID not in normal_ctx.provenance
    assert "retracted obsolete note" not in normal_texts
    assert all("retracted" not in f for f in normal_ctx.graph_facts)

    bootstrap_ctx = await get_relevant_context(
        agent_id=_AGENT,
        task_description=_QUERY_TOKEN,
        namespace=_NS,
        include_graph=True,
        include_tips=False,
        include_hotness=False,
        mode="bootstrap",
    )
    bootstrap_ids = {str(m.get("id", "")) for m in bootstrap_ctx.memories}
    assert _SUPPRESSED_ID not in bootstrap_ids
    assert _SUPPRESSED_ID not in bootstrap_ctx.provenance
    assert all("retracted" not in f for f in bootstrap_ctx.graph_facts)

    packet = await create_handoff_packet(
        agent_id=_AGENT,
        session_id="sess-matrix",
        receiving_agent_id="peer-agent",
        namespace=_NS,
    )
    assert _ACTIVE_ID in packet.key_memory_ids
    assert _SUPPRESSED_ID not in packet.key_memory_ids
    snapshot_text = json.dumps(packet.knowledge_snapshot)
    assert "active_status" in snapshot_text
    assert "retracted" not in snapshot_text


async def test_path5_timeline_excludes_suppressed(seeded_5path_env, monkeypatch) -> None:
    """Path 5 (Timeline): _handle_timeline omits suppressed memories while returning active ones."""
    from archivist.app.handlers.tools_search import _handle_timeline

    monkeypatch.setattr("archivist.app.handlers.tools_search.require_rbac", lambda *a, **kw: None)
    monkeypatch.setattr(
        "archivist.app.handlers.tools_search.can_read_agent_memory", lambda *a, **kw: True
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_search.filter_agents_for_read",
        lambda caller, agents: (list(agents), []),
    )

    resp = await _handle_timeline(
        {
            "query": _QUERY_TOKEN,
            "namespace": _NS,
            "agent_id": _AGENT,
            "caller_agent": _AGENT,
            "days": 14,
        }
    )
    payload = json.loads(resp[0].text)
    entries = payload["timeline"]
    assert len(entries) >= 1
    combined = " ".join(e["text"] for e in entries)
    assert "primary operational runbook" in combined
    assert "retracted obsolete note" not in combined
