"""INIT-004/SPEC-003 — archivist_index progressive-disclosure map contract.

Asserts dual {markdown, map} shape, no key-fact prose, navigational pointers,
RBAC enforcement, and no synchronous LLM on the index path (ADR-004 GR-CE).
"""

from __future__ import annotations

import inspect
import json
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import TextContent

pytestmark = [pytest.mark.unit]


def _index_tool():
    from archivist.app.handlers.tools_search import TOOLS

    return next(t for t in TOOLS if t.name == "archivist_index")


class TestIndexToolSchema:
    def test_description_documents_map_semantics(self):
        tool = _index_tool()
        desc = tool.description.lower()
        assert "progressive-disclosure" in desc or "map" in desc
        assert "not citable" in desc or "not evidence" in desc or "not citable evidence" in desc
        assert "key-fact" in desc or "key fact" in desc
        assert "llm" in desc
        assert "{markdown, map}" in tool.description or "markdown" in desc

    def test_reference_documents_map_contract(self):
        from pathlib import Path

        ref = Path(__file__).resolve().parents[4] / "docs" / "REFERENCE.md"
        text = ref.read_text(encoding="utf-8")
        assert "Progressive-disclosure map" in text or "progressive-disclosure map" in text
        assert "GR-CE-001" in text or "not evidence" in text.lower()
        assert "{markdown, map}" in text or "`{markdown, map}`" in text
        assert "No key-fact prose" in text or "no key-fact prose" in text.lower()
        assert "No synchronous LLM" in text or "no synchronous llm" in text.lower()


class TestIndexHandlerDualShape:
    async def test_success_returns_markdown_and_map(self):
        from archivist.app.handlers.tools_search import _handle_index

        payload = {
            "markdown": (
                "# Memory Index — coach-ns\n\n"
                "Navigational map only — not evidence.\n"
                "- **Person**: Alice\n\n"
                "**Search hints:**\n"
                '- archivist_search query="Alice"'
            ),
            "map": {
                "namespace": "coach-ns",
                "empty": False,
                "entity_types": {"person": ["Alice"]},
                "entities": [{"name": "Alice", "type": "person"}],
                "pinned": [],
                "recently_active": ["Alice"],
                "top_topics": ["Alice"],
                "search_hints": ['archivist_search query="Alice"'],
            },
        }
        with (
            patch(
                "archivist.app.handlers.tools_search.build_namespace_index_payload",
                new_callable=AsyncMock,
                return_value=payload,
            ),
            patch("archivist.app.handlers.tools_search.is_permissive_mode", return_value=True),
        ):
            result = await _handle_index({"agent_id": "coach", "namespace": "coach-ns"})

        assert isinstance(result, list) and result
        body = json.loads(result[0].text)
        assert "markdown" in body
        assert "map" in body
        assert body["map"]["entity_types"]["person"] == ["Alice"]
        assert "Key Facts" not in body["markdown"]
        assert "key fact" not in body["markdown"].lower()

    async def test_rbac_denied_unchanged(self):
        from archivist.app.handlers.tools_search import _handle_index

        denied = [TextContent(type="text", text=json.dumps({"error": "access_denied"}))]
        with (
            patch("archivist.app.handlers.tools_search.is_permissive_mode", return_value=False),
            patch(
                "archivist.app.handlers.tools_search.require_rbac",
                return_value=denied,
            ) as rbac,
            patch(
                "archivist.app.handlers.tools_search.build_namespace_index_payload",
                new_callable=AsyncMock,
            ) as builder,
        ):
            result = await _handle_index({"agent_id": "intruder", "namespace": "secret-ns"})

        assert result == denied
        rbac.assert_called_once_with("intruder", "read", "secret-ns")
        builder.assert_not_called()

    async def test_handler_source_has_no_llm_call(self):
        from archivist.app.handlers import tools_search

        source = inspect.getsource(tools_search._handle_index)
        # GR-CE-002: no model invocation on index path (docstring may mention LLM)
        for forbidden in (
            "chat_completion",
            "complete(",
            "embed_text",
            "openai",
            "anthropic",
        ):
            assert forbidden not in source, forbidden
        assert "build_namespace_index_payload" in source
        assert "success_response" in source


class TestIndexBuilderNoKeyFacts:
    async def test_populated_map_has_no_key_facts_section(self, async_pool):
        from archivist.storage.compressed_index import build_namespace_index_payload
        from archivist.storage.graph import add_fact, upsert_entity

        ns = "map-contract-ns"
        eid_person = await upsert_entity(
            "MapPerson", "person", retention_class="durable", namespace=ns
        )
        await add_fact(
            eid_person,
            "MapPerson likes espresso in the morning",
            "test.md",
            "map-agent",
            namespace=ns,
        )
        eid_topic = await upsert_entity("MapTopic", "topic", namespace=ns)
        await add_fact(
            eid_topic, "MapTopic is a training plan", "test.md", "map-agent", namespace=ns
        )

        payload = await build_namespace_index_payload(ns, agent_ids=["map-agent"])
        md = payload["markdown"]
        index_map = payload["map"]

        assert "Key Facts" not in md
        assert "likes espresso" not in md
        assert "training plan" not in md
        assert "MapPerson" in md or "MapPerson" in str(index_map)
        assert index_map.get("entity_types") or index_map.get("entities")
        assert index_map.get("search_hints")
        assert any("archivist_search" in h for h in index_map["search_hints"])

    async def test_empty_namespace_dual_shape(self, async_pool):
        from archivist.storage.compressed_index import build_namespace_index_payload

        payload = await build_namespace_index_payload("empty-map-ns-003-unique")
        assert "markdown" in payload and "map" in payload
        assert "No indexed knowledge" in payload["markdown"]
        assert payload["map"].get("empty") is True
        assert "Key Facts" not in payload["markdown"]

    def test_builder_source_has_no_key_facts_or_llm(self):
        from archivist.storage import compressed_index as ci

        source = inspect.getsource(ci.build_namespace_index_payload)
        full = inspect.getsource(ci)
        assert "Key Facts" not in source
        assert "_query_key_facts" not in full
        for forbidden in ("chat_completion", "embed_text", "embed_batch", "openai"):
            assert forbidden not in source, forbidden
        assert "_query_entities" in source
