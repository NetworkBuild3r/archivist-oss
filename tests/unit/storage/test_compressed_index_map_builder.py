"""INIT-004/SPEC-005 — map-only compressed index builder (service).

Covers empty namespace, populated map, suppress/supersede omission,
token estimate helper, and no LLM/embed on the builder path.
"""

from __future__ import annotations

import inspect
import json

import pytest

pytestmark = [pytest.mark.unit]


async def _seed_entity_fact(
    *,
    name: str,
    entity_type: str,
    fact_text: str,
    agent_id: str,
    namespace: str,
    retention_class: str = "standard",
) -> tuple[int, int]:
    from archivist.storage.graph import add_fact, upsert_entity

    eid = await upsert_entity(
        name,
        entity_type,
        retention_class=retention_class,
        namespace=namespace,
    )
    fid = await add_fact(
        eid,
        fact_text,
        "test.md",
        agent_id,
        retention_class=retention_class,
        namespace=namespace,
    )
    return eid, fid


class TestMapOnlyBuilderEmptyAndPopulated:
    async def test_empty_namespace_map(self, async_pool):
        from archivist.storage.compressed_index import build_namespace_index_payload

        payload = await build_namespace_index_payload("empty-map-ns-005-unique")
        assert "markdown" in payload and "map" in payload
        assert payload["map"]["empty"] is True
        assert payload["map"]["entities"] == []
        assert "No indexed knowledge" in payload["markdown"]
        assert "Key Facts" not in payload["markdown"]
        assert "token_estimate" in payload
        assert payload["token_estimate"]["markdown_tokens"] >= 1

    async def test_populated_map_no_key_fact_prose(self, async_pool):
        from archivist.storage.compressed_index import build_namespace_index_payload

        ns = "populated-map-ns-005"
        await _seed_entity_fact(
            name="MapAlice",
            entity_type="person",
            fact_text="MapAlice drinks green tea every morning",
            agent_id="map-agent-005",
            namespace=ns,
            retention_class="durable",
        )
        await _seed_entity_fact(
            name="MapPlan",
            entity_type="topic",
            fact_text="MapPlan is a 12-week strength cycle",
            agent_id="map-agent-005",
            namespace=ns,
        )

        payload = await build_namespace_index_payload(ns, agent_ids=["map-agent-005"])
        md = payload["markdown"]
        index_map = payload["map"]

        assert index_map["empty"] is False
        assert "Key Facts" not in md
        assert "green tea" not in md
        assert "12-week" not in md
        assert "MapAlice" in md or "MapAlice" in str(index_map)
        assert index_map.get("entity_types") or index_map.get("entities")
        assert any("archivist_search" in h for h in index_map["search_hints"])
        # Namespace hard-scope: foreign ns must not leak into map
        foreign = await build_namespace_index_payload("other-ns-005-unique")
        assert foreign["map"]["empty"] is True


class TestSuppressSupersedeOmitted:
    async def test_suppressed_fact_entity_omitted_when_only_loser(self, async_pool):
        from archivist.storage.compressed_index import build_namespace_index_payload
        from archivist.storage.sqlite_pool import pool

        ns = "suppress-map-ns-005"
        _eid, fid = await _seed_entity_fact(
            name="SuppressedOnlyPerson",
            entity_type="person",
            fact_text="SuppressedOnlyPerson secret habit detail",
            agent_id="sup-agent-005",
            namespace=ns,
        )
        async with pool.write() as conn:
            await conn.execute("UPDATE facts SET is_suppressed = 1 WHERE id = ?", (fid,))

        payload = await build_namespace_index_payload(ns, agent_ids=["sup-agent-005"])
        names = {e["name"] for e in payload["map"].get("entities", [])}
        assert "SuppressedOnlyPerson" not in names
        assert "secret habit" not in payload["markdown"]
        assert payload["map"]["empty"] is True or "SuppressedOnlyPerson" not in str(payload["map"])

    async def test_superseded_loser_omitted_winner_kept(self, async_pool):
        from archivist.storage.compressed_index import build_namespace_index_payload
        from archivist.storage.graph import add_fact, supersede_fact, upsert_entity

        ns = "supersede-map-ns-005"
        eid = await upsert_entity("SuperPerson", "person", namespace=ns)
        old_id = await add_fact(
            eid,
            "SuperPerson lives in Boston permanently",
            "test.md",
            "super-agent-005",
            namespace=ns,
        )
        new_id = await add_fact(
            eid,
            "SuperPerson relocated to Denver last year",
            "test.md",
            "super-agent-005",
            namespace=ns,
        )
        await supersede_fact(old_id, new_id)

        payload = await build_namespace_index_payload(ns, agent_ids=["super-agent-005"])
        md = payload["markdown"]
        # Entity still active via winner fact — navigational name OK
        assert "SuperPerson" in md or "SuperPerson" in str(payload["map"])
        # Loser prose must not appear (map-only; no fact dump)
        assert "lives in Boston" not in md
        assert "relocated to Denver" not in md


class TestTokenEstimateHelper:
    def test_estimate_index_tokens_empty_and_populated(self):
        from archivist.storage.compressed_index import estimate_index_tokens

        empty = estimate_index_tokens(markdown="", index_map=None)
        assert empty == {"markdown_tokens": 0, "map_tokens": 0, "total_tokens": 0}

        md = "# Memory Index — demo\n\n- **Person**: Alice"
        index_map = {
            "namespace": "demo",
            "empty": False,
            "entities": [{"name": "Alice", "type": "person"}],
            "search_hints": ['archivist_search query="Alice"'],
        }
        est = estimate_index_tokens(markdown=md, index_map=index_map)
        assert est["markdown_tokens"] >= 1
        assert est["map_tokens"] >= 1
        assert est["total_tokens"] == est["markdown_tokens"] + est["map_tokens"]
        # Deterministic for same inputs
        assert estimate_index_tokens(markdown=md, index_map=index_map) == est

    async def test_payload_includes_token_estimate(self, async_pool):
        from archivist.storage.compressed_index import (
            build_namespace_index_payload,
            estimate_index_tokens,
        )

        payload = await build_namespace_index_payload("token-est-ns-005")
        te = payload["token_estimate"]
        assert set(te) == {"markdown_tokens", "map_tokens", "total_tokens"}
        recomputed = estimate_index_tokens(markdown=payload["markdown"], index_map=payload["map"])
        assert te == recomputed


class TestNoLlmAndWakeUpMapOnly:
    def test_builder_source_has_no_llm_or_key_facts(self):
        from archivist.storage import compressed_index as ci

        source = inspect.getsource(ci.build_namespace_index_payload)
        full = inspect.getsource(ci)
        assert "Key Facts" not in source
        assert "_query_key_facts" not in full
        for forbidden in (
            "chat_completion",
            "embed_text",
            "embed_batch",
            "openai",
            "anthropic",
        ):
            assert forbidden not in source, forbidden
        assert "recall_visible_sql_facts" in inspect.getsource(ci._query_entities)
        assert "estimate_index_tokens" in source

    async def test_wake_up_toc_has_no_key_facts_section(self, async_pool):
        from archivist.storage.compressed_index import (
            build_wake_up_context,
            format_wake_up_text,
        )

        ns = "wake-map-ns-005"
        await _seed_entity_fact(
            name="WakePerson",
            entity_type="person",
            fact_text="WakePerson prefers morning runs",
            agent_id="wake-agent-005",
            namespace=ns,
            retention_class="durable",
        )

        ctx = await build_wake_up_context(ns, agent_id="wake-agent-005")
        toc = ctx.get("namespace_toc", "")
        assert "Key Facts" not in toc
        assert "prefers morning runs" not in toc

        rendered = format_wake_up_text(ctx, agent_id="wake-agent-005")
        assert "Key Facts" not in rendered
        # Ops L1 critical facts may exist separately; TOC must stay map-only
        if toc:
            assert "Navigational map only" in toc or "Memory Index" in toc or "WakePerson" in toc
        # Ensure payload is JSON-serializable without secrets keys
        dumped = json.dumps(ctx)
        assert "api_key" not in dumped.lower()
        assert "password" not in dumped.lower()
