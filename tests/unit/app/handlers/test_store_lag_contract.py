"""INIT-005/SPEC-006 — store success deferred / searchable-lag API contract.

Covers:
- embed_deferred + searchable_lag_hint when defer path used
- searchable_lag_metric + stage_timings always present on success
- defer off → embed_deferred false, no lag hint
- Namespace write RBAC still enforced (unchanged)
- Success JSON never includes embedding vectors
"""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from archivist.app.handlers.tools_storage import (
    _SEARCHABLE_LAG_HINT_DEFERRED,
    TOOLS,
)
from archivist.core import metrics as m

pytestmark = [pytest.mark.unit]


@contextmanager
def _store_patches(*, embed_return=None):
    """Minimal patches for store without real Qdrant/embed/LLM."""
    if embed_return is None:
        embed_return = [0.1] * 1024
    mock_client = MagicMock()
    mock_client.upsert = MagicMock()
    mock_client.query_points = MagicMock(return_value=MagicMock(points=[]))
    stack = ExitStack()
    try:
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=embed_return,
            )
        )
        stack.enter_context(
            patch(
                "archivist.write.conflict_detection.embed_text",
                new_callable=AsyncMock,
                return_value=embed_return,
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.llm_adjudicated_dedup",
                new_callable=AsyncMock,
                return_value=None,
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage._query_similar",
                new_callable=AsyncMock,
                return_value=(embed_return, []),
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.check_for_conflicts",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    has_conflict=False,
                    max_similarity=0.0,
                    conflicting_ids=[],
                    recommendation="",
                ),
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage._extract_and_store_entities",
                new_callable=AsyncMock,
                return_value={},
            )
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.qdrant_client", return_value=mock_client)
        )
        stack.enter_context(
            patch("archivist.write.conflict_detection.qdrant_client", return_value=mock_client)
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.ensure_collection",
                return_value="archivist_memories",
            )
        )
        stack.enter_context(patch("archivist.core.audit.log_memory_event", new_callable=AsyncMock))
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage._extract_needle_micro_chunks",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_for_agent",
                return_value="coach-ns",
            )
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.get_namespace_config", return_value=None)
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.require_rbac", return_value=None)
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.pre_extract", return_value={})
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.extract_needle_entities",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.conflict_vec_for_primary_embed",
                return_value=None,
            )
        )
        yield mock_client
    finally:
        stack.close()


def _configure_store_flags(monkeypatch, *, embed_defer: bool) -> None:
    monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
    monkeypatch.setattr("archivist.core.config.ARCHIVIST_EMBED_DEFER", embed_defer)
    monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", False)
    monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_CHECK_ON_STORE", False)
    monkeypatch.setattr("archivist.core.config.CONFLICT_BLOCK_ON_STORE", False)
    monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_BLOCK_ON_STORE", False)
    monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.TOPIC_ROUTING_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.DEDUP_LLM_ENABLED", False)


_STORE_ARGS = {
    "text": "Coach note: protein timing after evening training",
    "agent_id": "coach-agent",
    "namespace": "coach-ns",
    "entities": ["ProteinTiming"],
    "actor_id": "coach-agent",
    "actor_type": "agent",
}

_SECRET_KEYS = frozenset(
    {
        "vector",
        "vectors",
        "embedding",
        "embeddings",
        "api_key",
        "token",
        "password",
        "secret",
    }
)


def _assert_no_secret_or_vector_payload(obj: object, *, path: str = "$") -> None:
    """Recursively ensure success JSON has no embedding vectors or secret keys."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert key.lower() not in _SECRET_KEYS, f"forbidden key at {path}.{key}"
            if key == "searchable_lag_metric":
                assert isinstance(value, str)
                continue
            _assert_no_secret_or_vector_payload(value, path=f"{path}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            # Numeric vectors would be long float lists — reject large numeric lists.
            if (
                isinstance(item, int | float)
                and len(obj) >= 8
                and all(isinstance(x, int | float) for x in obj)
            ):
                raise AssertionError(f"numeric vector-like list at {path} (len={len(obj)})")
            _assert_no_secret_or_vector_payload(item, path=f"{path}[{i}]")


class TestStoreLagContract:
    async def test_defer_on_exposes_lag_fields(self, async_pool, monkeypatch):
        """ac-1: defer path → embed_deferred + lag hint + metric + stage_timings."""
        _configure_store_flags(monkeypatch, embed_defer=True)

        with _store_patches():
            import archivist.app.handlers.tools_storage as ts

            result = await ts._handle_store(dict(_STORE_ARGS))

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        assert data["embed_deferred"] is True
        assert data["searchable_lag_hint"] == _SEARCHABLE_LAG_HINT_DEFERRED
        assert data["searchable_lag_metric"] == m.SEARCHABLE_LAG_SECONDS
        assert data["searchable_lag_metric"] == "archivist_outbox_lag_seconds"
        assert isinstance(data.get("stage_timings"), dict)
        assert "embed_ms" in data["stage_timings"]
        assert isinstance(data["stage_timings"]["embed_ms"], int | float)
        # Existing ack fields preserved (additive contract).
        assert data.get("memory_id")
        assert data.get("namespace") == "coach-ns"
        assert "duration_ms" in data
        assert "provenance" in data
        _assert_no_secret_or_vector_payload(data)

    async def test_defer_off_omits_lag_hint(self, async_pool, monkeypatch):
        """ac-1: default path → embed_deferred false; lag hint absent; metric still present."""
        _configure_store_flags(monkeypatch, embed_defer=False)

        with _store_patches():
            import archivist.app.handlers.tools_storage as ts

            result = await ts._handle_store(dict(_STORE_ARGS))

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        assert data["embed_deferred"] is False
        assert "searchable_lag_hint" not in data
        assert data["searchable_lag_metric"] == m.SEARCHABLE_LAG_SECONDS
        assert isinstance(data.get("stage_timings"), dict)
        assert "embed_ms" in data["stage_timings"]
        _assert_no_secret_or_vector_payload(data)

    async def test_store_tool_description_mentions_lag_contract(self):
        """Tool schema advertises deferred / lag success fields (SPEC-006)."""
        store_tool = next(t for t in TOOLS if t.name == "archivist_store")
        desc = store_tool.description or ""
        assert "embed_deferred" in desc
        assert "searchable_lag" in desc.lower() or "ARCHIVIST_EMBED_DEFER" in desc
        assert "outbox" in desc.lower()

    async def test_namespace_write_rbac_unchanged(self, monkeypatch):
        """ac-4: cross-namespace store still requires write RBAC (fail closed)."""
        from mcp.types import TextContent

        _configure_store_flags(monkeypatch, embed_defer=True)
        denied = [TextContent(type="text", text='{"error":"forbidden"}')]

        with (
            patch(
                "archivist.app.handlers.tools_storage.require_rbac",
                return_value=denied,
            ) as mock_rbac,
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_for_agent",
                return_value="outsider-home",
            ),
            patch("archivist.app.handlers.tools_storage.get_namespace_config", return_value=None),
        ):
            import archivist.app.handlers.tools_storage as ts

            result = await ts._handle_store(
                {
                    "text": "should not store",
                    "agent_id": "outsider",
                    "namespace": "secret-ns",
                }
            )

        mock_rbac.assert_called_once_with("outsider", "write", "secret-ns")
        assert result is denied
        # Denied path must not look like a successful store with lag fields.
        body = json.loads(result[0].text)
        assert body.get("stored") is not True
        assert "embed_deferred" not in body
