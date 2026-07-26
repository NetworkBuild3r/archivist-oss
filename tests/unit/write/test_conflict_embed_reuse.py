"""INIT-005/SPEC-004 — conflict embed reuse into primary store.

Covers:
- Reuse when conflict text and primary embed_input are byte-identical
- Fresh embed when texts differ (e.g. contextual augmentation)
- No second tools_storage.embed_text call on hit
- Reused vector is call-local (never crosses namespace / store-call boundaries)
"""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------


class TestConflictVecForPrimaryEmbed:
    def test_hit_when_texts_byte_identical(self):
        from archivist.write.conflict_detection import conflict_vec_for_primary_embed

        shared = [0.42, 0.1, 0.0]
        out = conflict_vec_for_primary_embed(
            conflict_text="same text",
            embed_input="same text",
            shared_vec=shared,
        )
        assert out is shared

    def test_miss_when_texts_differ(self):
        from archivist.write.conflict_detection import conflict_vec_for_primary_embed

        shared = [0.42, 0.1, 0.0]
        out = conflict_vec_for_primary_embed(
            conflict_text="raw text",
            embed_input="[Agent: coach] raw text",
            shared_vec=shared,
        )
        assert out is None

    def test_miss_when_no_shared_vec(self):
        from archivist.write.conflict_detection import conflict_vec_for_primary_embed

        out = conflict_vec_for_primary_embed(
            conflict_text="same",
            embed_input="same",
            shared_vec=None,
        )
        assert out is None

    def test_no_cross_call_cache(self):
        """Helper is pure — sequential calls with different namespaces cannot leak."""
        from archivist.write.conflict_detection import conflict_vec_for_primary_embed

        ns_a_vec = [1.0, 0.0]
        ns_b_vec = [0.0, 1.0]
        hit_a = conflict_vec_for_primary_embed(
            conflict_text="memo",
            embed_input="memo",
            shared_vec=ns_a_vec,
        )
        hit_b = conflict_vec_for_primary_embed(
            conflict_text="memo",
            embed_input="memo",
            shared_vec=ns_b_vec,
        )
        assert hit_a is ns_a_vec
        assert hit_b is ns_b_vec
        assert hit_a is not hit_b


# ---------------------------------------------------------------------------
# Store path: primary embed call site
# ---------------------------------------------------------------------------


@contextmanager
def _store_patches(*, conflict_vec, primary_embed_mock: AsyncMock):
    """Minimal store patches; conflict and primary embeds are distinct mocks."""
    mock_client = MagicMock()
    mock_client.upsert = MagicMock()
    mock_client.query_points = MagicMock(return_value=MagicMock(points=[]))
    stack = ExitStack()
    try:
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.embed_text",
                new=primary_embed_mock,
            )
        )
        stack.enter_context(
            patch(
                "archivist.write.conflict_detection.embed_text",
                new_callable=AsyncMock,
                return_value=conflict_vec,
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


def _configure_reuse_flags(monkeypatch, *, conflict: bool, augment: bool) -> None:
    monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
    monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", conflict)
    monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_CHECK_ON_STORE", conflict)
    monkeypatch.setattr("archivist.core.config.CONFLICT_BLOCK_ON_STORE", False)
    monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_BLOCK_ON_STORE", False)
    monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", augment)
    monkeypatch.setattr("archivist.core.config.TOPIC_ROUTING_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.DEDUP_LLM_ENABLED", False)


class TestStoreEmbedReusePath:
    async def test_reuse_hit_skips_primary_embed_text(self, async_pool, monkeypatch):
        """When conflict ran and texts match, primary path must not call embed_text."""
        _configure_reuse_flags(monkeypatch, conflict=True, augment=False)
        conflict_vec = [0.42] * 32
        primary_embed = AsyncMock(return_value=[0.99] * 32)

        with _store_patches(conflict_vec=conflict_vec, primary_embed_mock=primary_embed):
            from archivist.app.handlers.tools_storage import _handle_store

            result = await _handle_store(
                {
                    "text": "coach insight about sleep debt",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": ["sleep"],
                }
            )

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        assert primary_embed.await_count == 0, (
            "primary embed_text must not run on reuse hit (INIT-005/SPEC-004)"
        )

    async def test_reuse_miss_when_augmentation_changes_text(self, async_pool, monkeypatch):
        """When embed_input differs from conflict text, primary embeds freshly."""
        _configure_reuse_flags(monkeypatch, conflict=True, augment=True)
        conflict_vec = [0.42] * 32
        primary_embed = AsyncMock(return_value=[0.99] * 32)

        with _store_patches(conflict_vec=conflict_vec, primary_embed_mock=primary_embed):
            from archivist.app.handlers.tools_storage import _handle_store

            result = await _handle_store(
                {
                    "text": "coach insight about sleep debt",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": ["sleep"],
                }
            )

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        assert primary_embed.await_count == 1
        # Fresh embed must use the augmented input, not the raw conflict text.
        called_input = primary_embed.await_args.args[0]
        assert called_input != "coach insight about sleep debt"
        assert "coach insight about sleep debt" in called_input

    async def test_no_reuse_when_conflict_check_disabled(self, async_pool, monkeypatch):
        """Without a conflict embed on this call, primary always embeds."""
        _configure_reuse_flags(monkeypatch, conflict=False, augment=False)
        conflict_vec = [0.42] * 32
        primary_embed = AsyncMock(return_value=[0.99] * 32)

        with _store_patches(conflict_vec=conflict_vec, primary_embed_mock=primary_embed):
            from archivist.app.handlers.tools_storage import _handle_store

            result = await _handle_store(
                {
                    "text": "solo store without conflict path",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": ["focus"],
                }
            )

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        assert primary_embed.await_count == 1
