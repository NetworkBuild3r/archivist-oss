"""INIT-003/SPEC-008 — coach-path CI eval scenarios (marker: coach_core).

Scenarios (SQLite CI path; no live Qdrant / no myaifitness harness):
1. store → index → search/get_context round-trip with provenance + stable memories[]
2. store under dead/hanging Qdrant fails fast / acks (composes SPEC-004 helpers)
3. two-namespace isolation (cross-tenant no leakage)

INIT-004/SPEC-001: store ack exposes duration_ms; index rebuild timing covered by
unit hooks (compressed_index.rebuild_complete / archivist_index_duration_ms).

INIT-004/SPEC-006: CE evals — TOC token ceiling (~500), no Key Facts prose
(GR-CE-001), get_context(mode=bootstrap) session start (SM-002).
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from tests.integration.storage.test_store_ack_slo import (
    _count_outbox_pending,
    _store_patches,
)

pytestmark = [
    pytest.mark.system,
    pytest.mark.mcp,
    pytest.mark.coach_core,
]

# docs/REFERENCE.md + ADR-004: progressive-disclosure map ~500-token intent.
# Audited CE ceiling for coach_core — markdown map must stay at or under this.
INDEX_MARKDOWN_TOKEN_CEILING = 500


def _configure_coach_store_flags(monkeypatch) -> None:
    monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
    monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", False)
    monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_CHECK_ON_STORE", False)
    monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.TOPIC_ROUTING_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.DEDUP_LLM_ENABLED", False)


@contextmanager
def _allow_search_rbac():
    """Patch search-path RBAC so coach evals can pass namespace + agent_id."""
    with ExitStack() as stack:
        stack.enter_context(
            patch("archivist.app.handlers.tools_search.require_rbac", return_value=None)
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_search.filter_agents_for_read",
                side_effect=lambda caller, ids: (list(ids), []),
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_search.get_namespace_for_agent",
                side_effect=lambda aid: {
                    "coach-agent-rt": "coach-ns-rt",
                    "agent-a": "coach-ns-a",
                    "agent-b": "coach-ns-b",
                    "coach-agent": "coach-ns",
                }.get(aid, aid),
            )
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_search.is_permissive_mode", return_value=True)
        )
        yield


def _row_to_hit(row: Any, *, score: float = 0.91) -> dict[str, Any]:
    """Normalize a memory_chunks SQLite row into a retrieval hit dict."""
    data = dict(row) if hasattr(row, "keys") else {}
    mid = data.get("qdrant_id") or data.get("id") or ""
    return {
        "id": mid,
        "text": data.get("text") or "",
        "score": score,
        "namespace": data.get("namespace") or "",
        "source": data.get("source") or "",
        "subject": data.get("subject") or "",
        "purpose": data.get("purpose") or "",
        "sensitivity": data.get("sensitivity") or "standard",
        "statement_kind": data.get("statement_kind") or "",
        "confidence": data.get("confidence"),
        "agent_id": data.get("agent_id") or "",
        "is_suppressed": bool(data.get("is_suppressed")),
        "supersedes_id": data.get("supersedes_id") or "",
        "date": data.get("date") or "",
    }


async def _load_chunk(pool, memory_id: str) -> dict[str, Any] | None:
    async with pool.read() as conn:
        cur = await conn.execute(
            "SELECT qdrant_id, text, namespace, source, subject, purpose, "
            "sensitivity, statement_kind, confidence, agent_id, "
            "is_suppressed, supersedes_id FROM memory_chunks WHERE qdrant_id = ?",
            (memory_id,),
        )
        row = await cur.fetchone()
    return dict(row) if row is not None else None


async def _chunks_for_namespace(pool, namespace: str) -> list[dict[str, Any]]:
    async with pool.read() as conn:
        cur = await conn.execute(
            "SELECT qdrant_id, text, namespace, source, subject, purpose, "
            "sensitivity, statement_kind, confidence, agent_id, "
            "is_suppressed, supersedes_id FROM memory_chunks WHERE namespace = ?",
            (namespace,),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


def _assert_stable_memories(memories: list[dict], *, expected_id: str, text_substr: str) -> None:
    assert isinstance(memories, list) and memories, "expected non-empty memories[]"
    mem = next((m for m in memories if m.get("id") == expected_id), memories[0])
    assert mem["id"] == expected_id
    assert isinstance(mem.get("text"), str) and mem["text"].strip()
    assert text_substr in mem["text"]
    assert "score" in mem
    prov = mem.get("provenance")
    assert isinstance(prov, dict)
    assert "api_key" not in prov
    assert "password" not in prov
    assert "token" not in prov


async def _store_one(
    *,
    text: str,
    namespace: str,
    agent_id: str,
    entities: list[str],
    provenance: dict | None = None,
) -> dict:
    from archivist.app.handlers._registry import dispatch_tool

    args = {
        "text": text,
        "agent_id": agent_id,
        "namespace": namespace,
        "entities": entities,
        **(provenance or {}),
    }
    with (
        _store_patches(),
        patch(
            "archivist.app.handlers.tools_storage.get_namespace_for_agent",
            return_value=namespace,
        ),
    ):
        result = await dispatch_tool("archivist_store", args)
    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# ac-1: store → index → search/get_context round-trip
# ---------------------------------------------------------------------------


class TestCoachRoundTrip:
    async def test_store_index_search_get_context_round_trip(self, qa_pool, monkeypatch):
        _configure_coach_store_flags(monkeypatch)
        ns = "coach-ns-rt"
        agent = "coach-agent-rt"
        entity = "SleepDebtEntity008"
        body = f"{entity} rises after late meals — coach insight for round-trip"

        store = await _store_one(
            text=body,
            namespace=ns,
            agent_id=agent,
            entities=[entity],
            provenance={
                "source": "harness",
                "subject": "sleep",
                "purpose": "coaching",
                "sensitivity": "health",
                "statement_kind": "user",
                "confidence": 0.92,
            },
        )
        assert store.get("stored") is True, store
        mid = store["memory_id"]
        # INIT-004/SPEC-001: store-ack duration exposed for coach-path baselines.
        assert isinstance(store.get("duration_ms"), int)
        assert store["duration_ms"] >= 0
        assert store["provenance"]["subject"] == "sleep"
        assert store["provenance"]["sensitivity"] == "health"

        row = await _load_chunk(qa_pool, mid)
        assert row is not None
        assert row["namespace"] == ns
        assert row["subject"] == "sleep"

        with _allow_search_rbac():
            from archivist.app.handlers._registry import dispatch_tool

            index_result = await dispatch_tool(
                "archivist_index",
                {"agent_id": agent, "namespace": ns},
            )
        index_payload = json.loads(index_result[0].text)
        assert "markdown" in index_payload and "map" in index_payload
        assert entity in index_result[0].text
        assert ns in index_result[0].text

        hit = _row_to_hit(row)
        # Inject a secret-looking key to prove sanitize on the recall path.
        hit["api_key"] = "sk-should-never-appear"

        from archivist.retrieval.retrieval_filters import (
            apply_prerank_filters,
            attach_stable_memories,
            build_stable_memories,
        )

        async def _retrieve_from_sqlite(**kwargs):
            assert kwargs.get("namespace") == ns
            filtered = apply_prerank_filters(
                [hit],
                namespace=ns,
                subject=kwargs.get("subject") or "",
                purpose=kwargs.get("purpose") or "",
                sensitivity=kwargs.get("sensitivity") or "",
            )
            return attach_stable_memories({"answer": "", "sources": filtered})

        with (
            patch(
                "archivist.app.handlers.tools_search.recursive_retrieve",
                new=AsyncMock(side_effect=_retrieve_from_sqlite),
            ),
            _allow_search_rbac(),
        ):
            from archivist.app.handlers._registry import dispatch_tool

            search_out = await dispatch_tool(
                "archivist_search",
                {
                    "query": "sleep debt late meals",
                    "agent_id": agent,
                    "namespace": ns,
                    "subject": "sleep",
                    "purpose": "coaching",
                    "refine": False,
                },
            )
        search_payload = json.loads(search_out[0].text)
        _assert_stable_memories(
            search_payload["memories"], expected_id=mid, text_substr="late meals"
        )
        assert search_payload["memories"][0]["provenance"].get("subject") == "sleep"
        assert search_payload["memories"][0]["provenance"].get("purpose") == "coaching"
        # Stable memories[] must strip secrets even if raw hit rows carried them.
        assert "sk-should-never-appear" not in json.dumps(search_payload["memories"])
        assert "api_key" not in search_payload["memories"][0]["provenance"]

        from archivist.retrieval.context_api import ContextChunk, RelevantContext

        memories = build_stable_memories([hit])
        ctx = RelevantContext(
            answer="",
            sources=[
                ContextChunk(
                    memory_id=mid,
                    text=body,
                    score=0.91,
                    tier="l2",
                    file_path="",
                    date="",
                    agent_id=agent,
                    namespace=ns,
                    subject="sleep",
                    purpose="coaching",
                    sensitivity="health",
                    source="harness",
                    confidence=0.92,
                    statement_kind="user",
                )
            ],
            graph_facts=[],
            tips=[],
            total_tokens=12,
            budget_tokens=8000,
            over_budget=False,
            tier_distribution={"l2": 1},
            token_savings_pct=0.0,
            provenance=[mid],
            pack_policy="adaptive",
            memories=memories,
        )

        with (
            patch(
                "archivist.retrieval.context_api.get_relevant_context",
                new=AsyncMock(return_value=ctx),
            ),
            patch("archivist.app.handlers.tools_context.require_rbac", return_value=None),
        ):
            from archivist.app.handlers._registry import dispatch_tool

            ctx_out = await dispatch_tool(
                "archivist_get_context",
                {
                    "agent_id": agent,
                    "task_description": "coach sleep habits",
                    "namespace": ns,
                    "subject": "sleep",
                    "purpose": "coaching",
                },
            )
        ctx_payload = json.loads(ctx_out[0].text)
        _assert_stable_memories(ctx_payload["memories"], expected_id=mid, text_substr="late meals")
        assert ctx_payload["sources"][0]["text"]
        assert "sk-should-never-appear" not in json.dumps(ctx_payload["memories"])


# ---------------------------------------------------------------------------
# ac-2: dead/hanging Qdrant — fail-fast / ack (SPEC-004 compose)
# ---------------------------------------------------------------------------


class TestCoachDeadQdrantAck:
    async def test_store_acks_when_conflict_query_times_out(self, qa_pool, monkeypatch):
        """Coach path: hanging similarity check must not stall store ack."""
        monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
        monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", True)
        monkeypatch.setattr("archivist.core.config.CONFLICT_BLOCK_ON_STORE", False)
        monkeypatch.setattr("archivist.core.config.CONFLICT_QUERY_TIMEOUT_S", 0.05)
        monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_CHECK_ON_STORE", True)
        monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_BLOCK_ON_STORE", False)
        monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.TOPIC_ROUTING_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.DEDUP_LLM_ENABLED", False)

        async def _fake_wait_for(awaitable, timeout=None):
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            raise TimeoutError

        pending_before = await _count_outbox_pending(qa_pool)
        t0 = time.monotonic()
        with (
            _store_patches(
                upsert_side_effect=AssertionError("inline upsert must not run with outbox on")
            ) as mock_client,
            patch(
                "archivist.write.conflict_detection.asyncio.wait_for",
                side_effect=_fake_wait_for,
            ),
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_for_agent",
                return_value="coach-ns",
            ),
        ):
            from archivist.app.handlers._registry import dispatch_tool

            result = await dispatch_tool(
                "archivist_store",
                {
                    "text": "coach insight about sleep debt under dead qdrant",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": ["sleep"],
                },
            )
        elapsed = time.monotonic() - t0

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        assert "memory_id" in data
        assert elapsed < 2.0, f"store ack took {elapsed:.2f}s under timeout mock"
        assert mock_client.upsert.call_count == 0
        pending_after = await _count_outbox_pending(qa_pool)
        assert pending_after > pending_before

    async def test_store_acks_when_qdrant_unreachable(self, qa_pool, monkeypatch):
        monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
        monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", True)
        monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_CHECK_ON_STORE", True)
        monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.DEDUP_LLM_ENABLED", False)

        pending_before = await _count_outbox_pending(qa_pool)
        with (
            _store_patches() as mock_client,
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_for_agent",
                return_value="coach-ns",
            ),
        ):
            mock_client.query_points.side_effect = OSError("Qdrant unreachable")
            from archivist.app.handlers._registry import dispatch_tool

            result = await dispatch_tool(
                "archivist_store",
                {
                    "text": "coach memory while qdrant is down",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": ["focus"],
                },
            )

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        pending_after = await _count_outbox_pending(qa_pool)
        assert pending_after > pending_before


# ---------------------------------------------------------------------------
# ac-3: two-namespace isolation
# ---------------------------------------------------------------------------


class TestCoachNamespaceIsolation:
    async def test_two_namespace_no_cross_tenant_leakage(self, qa_pool, monkeypatch):
        _configure_coach_store_flags(monkeypatch)
        ns_a, ns_b = "coach-ns-a", "coach-ns-b"
        secret_a = "tenant-A-secret-plan-alpha-008"
        secret_b = "tenant-B-secret-plan-beta-008"

        store_a = await _store_one(
            text=f"Coach note A: {secret_a}",
            namespace=ns_a,
            agent_id="agent-a",
            entities=["TenantAEntity008"],
            provenance={"subject": "user-a", "purpose": "coaching", "source": "harness"},
        )
        store_b = await _store_one(
            text=f"Coach note B: {secret_b}",
            namespace=ns_b,
            agent_id="agent-b",
            entities=["TenantBEntity008"],
            provenance={"subject": "user-b", "purpose": "coaching", "source": "harness"},
        )
        assert store_a.get("stored") is True, store_a
        assert store_b.get("stored") is True, store_b
        mid_a, mid_b = store_a["memory_id"], store_b["memory_id"]

        rows_a = await _chunks_for_namespace(qa_pool, ns_a)
        rows_b = await _chunks_for_namespace(qa_pool, ns_b)
        assert any(r["qdrant_id"] == mid_a for r in rows_a)
        assert any(r["qdrant_id"] == mid_b for r in rows_b)
        assert all(r["namespace"] == ns_a for r in rows_a)
        assert all(r["namespace"] == ns_b for r in rows_b)

        mixed_hits = [_row_to_hit(r) for r in rows_a + rows_b]
        assert len(mixed_hits) >= 2

        from archivist.retrieval.retrieval_filters import (
            apply_prerank_filters,
            attach_stable_memories,
        )

        async def _retrieve_ns(**kwargs):
            ns = kwargs.get("namespace") or ""
            filtered = apply_prerank_filters(mixed_hits, namespace=ns)
            return attach_stable_memories({"answer": "", "sources": filtered})

        with (
            patch(
                "archivist.app.handlers.tools_search.recursive_retrieve",
                new=AsyncMock(side_effect=_retrieve_ns),
            ),
            _allow_search_rbac(),
        ):
            from archivist.app.handlers._registry import dispatch_tool

            out_a = await dispatch_tool(
                "archivist_search",
                {
                    "query": "secret plan",
                    "agent_id": "agent-a",
                    "namespace": ns_a,
                    "refine": False,
                },
            )
            out_b = await dispatch_tool(
                "archivist_search",
                {
                    "query": "secret plan",
                    "agent_id": "agent-b",
                    "namespace": ns_b,
                    "refine": False,
                },
            )

        payload_a = json.loads(out_a[0].text)
        payload_b = json.loads(out_b[0].text)
        text_a = json.dumps(payload_a)
        text_b = json.dumps(payload_b)

        assert secret_a in text_a
        assert secret_b not in text_a
        assert mid_b not in {m.get("id") for m in payload_a.get("memories", [])}

        assert secret_b in text_b
        assert secret_a not in text_b
        assert mid_a not in {m.get("id") for m in payload_b.get("memories", [])}

        # get_context RBAC/namespace gate: wrong-ns deny must not leak payload.
        from mcp.types import TextContent

        denied = [TextContent(type="text", text=json.dumps({"error": "access_denied"}))]
        with patch(
            "archivist.app.handlers.tools_context.require_rbac",
            return_value=denied,
        ):
            from archivist.app.handlers._registry import dispatch_tool

            denied_out = await dispatch_tool(
                "archivist_get_context",
                {
                    "agent_id": "agent-a",
                    "task_description": "try to read other tenant",
                    "namespace": ns_b,
                },
            )
        denied_payload = json.loads(denied_out[0].text)
        assert denied_payload.get("error") == "access_denied"
        assert secret_b not in json.dumps(denied_payload)


# ---------------------------------------------------------------------------
# INIT-004/SPEC-001: coach-path stage timing hooks
# ---------------------------------------------------------------------------


class TestCoachPathTimingHooks:
    """Evidence that coach-path timing fields stay wired for baselines."""

    def test_search_stage_timing_keys_in_retriever(self):
        import inspect

        from archivist.retrieval import rlm_retriever

        source = inspect.getsource(rlm_retriever.recursive_retrieve)
        assert '_stage_timings["embed_ms"]' in source
        assert '_stage_timings["vector_ms"]' in source
        assert "stage_timings" in source

    def test_index_rebuild_timing_hook_present(self):
        import inspect

        from archivist.storage import compressed_index as ci

        source = inspect.getsource(ci.build_namespace_index_payload)
        assert "compressed_index.rebuild_complete" in source
        assert "rebuild_ms" in source
        assert "INDEX_DURATION_MS" in source

    async def test_store_ack_returns_duration_ms(self, qa_pool, monkeypatch):
        _configure_coach_store_flags(monkeypatch)
        store = await _store_one(
            text="coach timing baseline memory for duration_ms",
            namespace="coach-ns-timing",
            agent_id="coach-agent",
            entities=["TimingEntity004"],
        )
        assert store.get("stored") is True, store
        assert isinstance(store.get("duration_ms"), int)
        assert store["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# INIT-004/SPEC-006: CE evals — TOC ceiling, GR-CE-001, bootstrap (SM-002)
# ---------------------------------------------------------------------------


class TestCoachCeEvals:
    """Lock ADR-004 CE contracts into the coach_core CI marker."""

    async def test_index_toc_token_ceiling_and_no_key_facts(self, qa_pool, monkeypatch):
        """SM-001 + GR-CE-001: map stays ≤~500 tok; no Key Facts prose."""
        from archivist.utils.tokenizer import count_tokens

        _configure_coach_store_flags(monkeypatch)
        ns = "coach-ns-ce-toc"
        agent = "coach-agent-rt"
        entity = "CeTocEntity006"
        fact_prose = f"{entity} drinks green tea every morning before training"

        store = await _store_one(
            text=fact_prose,
            namespace=ns,
            agent_id=agent,
            entities=[entity],
            provenance={
                "source": "harness",
                "subject": "habits",
                "purpose": "coaching",
                "sensitivity": "standard",
                "statement_kind": "user",
            },
        )
        assert store.get("stored") is True, store

        with _allow_search_rbac():
            from archivist.app.handlers._registry import dispatch_tool

            index_out = await dispatch_tool(
                "archivist_index",
                {"agent_id": agent, "namespace": ns},
            )
        payload = json.loads(index_out[0].text)
        assert "markdown" in payload and "map" in payload
        md = payload["markdown"]
        assert isinstance(md, str) and md.strip()

        # Token ceiling — prefer payload token_estimate; cross-check count_tokens.
        te = payload.get("token_estimate") or {}
        md_tokens = te.get("markdown_tokens")
        if md_tokens is None:
            md_tokens = count_tokens(md)
        assert isinstance(md_tokens, int)
        assert md_tokens <= INDEX_MARKDOWN_TOKEN_CEILING, (
            f"index markdown tokens {md_tokens} exceed ceiling "
            f"{INDEX_MARKDOWN_TOKEN_CEILING} (ADR-004 / REFERENCE ~500)"
        )
        assert count_tokens(md) <= INDEX_MARKDOWN_TOKEN_CEILING

        # GR-CE-001 — no key-fact prose / Key Facts section
        assert "Key Facts" not in md
        assert "key fact" not in md.lower()
        assert "green tea" not in md
        assert "every morning" not in md
        # Navigational pointer still present
        assert entity in md or entity in json.dumps(payload["map"])

    async def test_get_context_bootstrap_mode(self, qa_pool, monkeypatch):
        """SM-002: get_context(mode=bootstrap) compact session-start path."""
        from archivist.retrieval.context_api import BOOTSTRAP_DEFAULT_MAX_TOKENS

        _configure_coach_store_flags(monkeypatch)
        ns = "coach-ns-ce-boot"
        agent = "coach-agent-rt"
        entity = "CeBootEntity006"

        store = await _store_one(
            text=f"{entity} prefers morning workouts for session bootstrap",
            namespace=ns,
            agent_id=agent,
            entities=[entity],
            provenance={
                "source": "harness",
                "subject": "habits",
                "purpose": "coaching",
            },
        )
        assert store.get("stored") is True, store

        with patch("archivist.app.handlers.tools_context.require_rbac", return_value=None):
            from archivist.app.handlers._registry import dispatch_tool

            ctx_out = await dispatch_tool(
                "archivist_get_context",
                {
                    "agent_id": agent,
                    "task_description": "session start bootstrap",
                    "namespace": ns,
                    "mode": "bootstrap",
                },
            )
        payload = json.loads(ctx_out[0].text)
        assert payload.get("mode") == "bootstrap"
        status = payload.get("context_status") or {}
        assert status.get("mode") == "bootstrap"
        assert status.get("pack_policy") == "bootstrap"
        assert status.get("budget_tokens") == BOOTSTRAP_DEFAULT_MAX_TOKENS
        assert status.get("total_tokens") <= BOOTSTRAP_DEFAULT_MAX_TOKENS
        # Empty memories[] is success on bootstrap (GR-CE-003) — not invented from TOC
        assert isinstance(payload.get("memories"), list)
        assert payload["memories"] == []
        assert payload.get("sources") == []
        answer = payload.get("answer") or ""
        assert "Bootstrap Context" in answer or "Identity" in answer or answer.strip()
