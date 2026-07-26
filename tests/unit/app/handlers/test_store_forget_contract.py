"""INIT-003/SPEC-006 — store provenance, forget modes, index-after-store.

Covers:
- Additive store provenance persisted on memory_chunks
- Provenance size/enum validation
- archivist_delete mode=suppress|delete via SPEC-007 lifecycle
- Namespace write RBAC on mutations
- Index freshness after store (live rebuild)
- No net-new core tools (forget = delete + mode)
"""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit]


@contextmanager
def _store_patches():
    """Minimal patches for store without real Qdrant/embed/LLM."""
    mock_client = MagicMock()
    mock_client.upsert = MagicMock()
    mock_client.query_points = MagicMock(return_value=MagicMock(points=[]))
    stack = ExitStack()
    try:
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            )
        )
        stack.enter_context(
            patch(
                "archivist.write.conflict_detection.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
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
            patch("archivist.app.handlers.tools_storage.qdrant_client", return_value=mock_client)
        )
        stack.enter_context(
            patch("archivist.write.conflict_detection.qdrant_client", return_value=mock_client)
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.ensure_collection",
                return_value="test_col",
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
        yield mock_client
    finally:
        stack.close()


def _configure_store_flags(monkeypatch) -> None:
    monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
    monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", False)
    monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_CHECK_ON_STORE", False)
    monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.TOPIC_ROUTING_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.DEDUP_LLM_ENABLED", False)


# ---------------------------------------------------------------------------
# Provenance validation (pure)
# ---------------------------------------------------------------------------


class TestProvenanceValidation:
    def test_accepts_defaults(self):
        from archivist.app.handlers.tools_storage import _validate_store_provenance

        result = _validate_store_provenance({})
        assert isinstance(result, dict)
        assert result["sensitivity"] == "standard"
        assert result["statement_kind"] == "user"

    def test_rejects_oversized_subject(self):
        from archivist.app.handlers.tools_storage import _validate_store_provenance

        result = _validate_store_provenance({"subject": "x" * 200})
        assert isinstance(result, list)
        data = json.loads(result[0].text)
        assert data["error"] == "invalid_provenance"
        assert data["field"] == "subject"

    def test_rejects_bad_sensitivity_enum(self):
        from archivist.app.handlers.tools_storage import _validate_store_provenance

        result = _validate_store_provenance({"sensitivity": "topsecret"})
        assert isinstance(result, list)
        data = json.loads(result[0].text)
        assert data["field"] == "sensitivity"

    def test_rejects_bad_statement_kind(self):
        from archivist.app.handlers.tools_storage import _validate_store_provenance

        result = _validate_store_provenance({"statement_kind": "guess"})
        assert isinstance(result, list)
        data = json.loads(result[0].text)
        assert data["field"] == "statement_kind"

    def test_rejects_confidence_above_one(self):
        from archivist.app.handlers.tools_storage import _validate_store_provenance

        result = _validate_store_provenance({"confidence": 1.5})
        assert isinstance(result, list)
        data = json.loads(result[0].text)
        assert data["field"] == "confidence"


# ---------------------------------------------------------------------------
# Store provenance round-trip
# ---------------------------------------------------------------------------


class TestStoreProvenancePersist:
    async def test_store_persists_provenance_columns(self, async_pool, monkeypatch):
        _configure_store_flags(monkeypatch)

        with _store_patches():
            from archivist.app.handlers.tools_storage import _handle_store

            result = await _handle_store(
                {
                    "text": "Coach note: sleep debt rises after late meals",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": ["SleepDebtEntity"],
                    "source": "harness",
                    "subject": "sleep",
                    "purpose": "coaching",
                    "sensitivity": "health",
                    "statement_kind": "user",
                    "confidence": 0.9,
                    "actor_id": "coach-agent",
                    "actor_type": "agent",
                }
            )

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        mid = data["memory_id"]
        assert data["provenance"]["subject"] == "sleep"
        assert data["provenance"]["sensitivity"] == "health"

        async with async_pool.read() as conn:
            cur = await conn.execute(
                "SELECT source, subject, purpose, sensitivity, statement_kind, "
                "confidence, actor_id, namespace FROM memory_chunks WHERE qdrant_id = ?",
                (mid,),
            )
            row = await cur.fetchone()

        assert row is not None
        assert row["source"] == "harness"
        assert row["subject"] == "sleep"
        assert row["purpose"] == "coaching"
        assert row["sensitivity"] == "health"
        assert row["statement_kind"] == "user"
        assert float(row["confidence"]) == pytest.approx(0.9)
        assert row["namespace"] == "coach-ns"


# ---------------------------------------------------------------------------
# Forget / delete modes
# ---------------------------------------------------------------------------


class TestForgetModes:
    async def test_suppress_hides_from_default_recall(self, async_pool, monkeypatch):
        from archivist.app.handlers.tools_storage import _handle_delete
        from archivist.lifecycle.visibility import is_recall_visible
        from archivist.storage.graph import upsert_fts_chunk

        mid = "mem-suppress-006"
        ns = "coach-ns"
        await upsert_fts_chunk(
            mid,
            "suppress me",
            "explicit/a",
            0,
            agent_id="coach-agent",
            namespace=ns,
        )

        with (
            patch("archivist.app.handlers.tools_storage.require_rbac", return_value=None),
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_for_agent",
                return_value=ns,
            ),
            patch("archivist.lifecycle.correct.qdrant_client", return_value=MagicMock()),
            patch(
                "archivist.lifecycle.correct.collection_for",
                return_value="col",
            ),
            patch("archivist.core.audit.log_memory_event", new_callable=AsyncMock),
        ):
            result = await _handle_delete(
                {
                    "memory_id": mid,
                    "agent_id": "coach-agent",
                    "namespace": ns,
                    "mode": "suppress",
                }
            )

        data = json.loads(result[0].text)
        assert data.get("mode") == "suppress"
        assert data.get("suppressed") is True
        assert data.get("forgotten") is True

        async with async_pool.read() as conn:
            cur = await conn.execute(
                "SELECT is_suppressed, text FROM memory_chunks WHERE qdrant_id = ?",
                (mid,),
            )
            row = await cur.fetchone()

        assert row is not None
        assert int(row["is_suppressed"]) == 1
        assert row["text"] == "suppress me"  # record remains
        assert is_recall_visible(dict(row)) is False

    async def test_delete_mode_calls_lifecycle_delete(self, monkeypatch):
        from archivist.app.handlers.tools_storage import _handle_delete

        with (
            patch("archivist.app.handlers.tools_storage.require_rbac", return_value=None),
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_for_agent",
                return_value="coach-ns",
            ),
            patch(
                "archivist.lifecycle.correct.delete_memory",
                new_callable=AsyncMock,
                return_value={
                    "status": "soft_delete_initiated",
                    "memory_id": "m1",
                    "namespace": "coach-ns",
                    "op_id": "op-1",
                    "idempotent": False,
                },
            ) as mock_del,
            patch("archivist.app.handlers.tools_storage.hot_cache.invalidate_namespace"),
        ):
            result = await _handle_delete(
                {
                    "memory_id": "m1",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "mode": "delete",
                }
            )

        mock_del.assert_awaited_once_with("m1", "coach-ns", agent_id="coach-agent")
        data = json.loads(result[0].text)
        assert data["mode"] == "delete"
        assert data["deleted"] is True

    async def test_cross_namespace_rbac_denies_suppress(self):
        from mcp.types import TextContent

        from archivist.app.handlers.tools_storage import _handle_delete

        denied = [TextContent(type="text", text=json.dumps({"error": "access_denied"}))]
        with (
            patch(
                "archivist.app.handlers.tools_storage.require_rbac",
                return_value=denied,
            ) as mock_rbac,
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_for_agent",
                return_value="other-ns",
            ),
            patch(
                "archivist.lifecycle.correct.suppress_memory",
                new_callable=AsyncMock,
            ) as mock_sup,
        ):
            result = await _handle_delete(
                {
                    "memory_id": "m1",
                    "agent_id": "outsider",
                    "namespace": "secret-ns",
                    "mode": "suppress",
                }
            )

        mock_rbac.assert_called_once_with("outsider", "write", "secret-ns")
        mock_sup.assert_not_awaited()
        assert json.loads(result[0].text)["error"] == "access_denied"


# ---------------------------------------------------------------------------
# Index after store
# ---------------------------------------------------------------------------


class TestIndexAfterStore:
    async def test_index_reflects_entity_after_store(self, async_pool, monkeypatch):
        _configure_store_flags(monkeypatch)
        unique = "IndexFreshEntity006"

        with _store_patches():
            from archivist.app.handlers.tools_storage import _handle_store

            store_result = await _handle_store(
                {
                    "text": f"{unique} prefers morning workouts",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": [unique],
                }
            )
        assert json.loads(store_result[0].text).get("stored") is True

        with (
            patch("archivist.app.handlers.tools_search.require_rbac", return_value=None),
            patch(
                "archivist.app.handlers.tools_search.get_namespace_for_agent",
                return_value="coach-ns",
            ),
            patch("archivist.app.handlers.tools_search.is_permissive_mode", return_value=True),
        ):
            from archivist.app.handlers.tools_search import _handle_index

            index_result = await _handle_index({"agent_id": "coach-agent", "namespace": "coach-ns"})

        index_text = index_result[0].text
        assert unique in index_text
        assert "coach-ns" in index_text


# ---------------------------------------------------------------------------
# Core tool surface (GR-PROD-002)
# ---------------------------------------------------------------------------


class TestNoNetNewCoreTools:
    def test_core_has_delete_not_separate_forget(self):
        from archivist.app.handlers._registry import CORE_TOOL_NAMES
        from archivist.app.handlers.tools_storage import TOOLS

        assert "archivist_delete" in CORE_TOOL_NAMES
        assert "archivist_forget" not in CORE_TOOL_NAMES
        assert len(CORE_TOOL_NAMES) <= 12
        delete_tool = next(t for t in TOOLS if t.name == "archivist_delete")
        assert "mode" in delete_tool.inputSchema["properties"]
        assert set(delete_tool.inputSchema["properties"]["mode"]["enum"]) == {
            "delete",
            "suppress",
        }

    def test_store_schema_has_provenance_fields(self):
        from archivist.app.handlers.tools_storage import TOOLS

        store = next(t for t in TOOLS if t.name == "archivist_store")
        props = store.inputSchema["properties"]
        for field in (
            "source",
            "subject",
            "purpose",
            "sensitivity",
            "statement_kind",
            "correction_of",
        ):
            assert field in props, field
