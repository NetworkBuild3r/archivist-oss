"""INIT-005/SPEC-005 — outbox drain fills deferred embeds before upsert.

Covers:
- Drain embeds then upserts
- Namespace/collection mismatch fails closed
- No embeddings or embed-input fact text in fill logs
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit]


class TestFillDeferredUpsertVectors:
    async def test_fill_embeds_then_returns_vectors(self):
        from archivist.storage.outbox import _fill_deferred_upsert_vectors

        points = [
            {
                "id": "p1",
                "vector": [],
                "payload": {"namespace": "ns-a", "text": "secret fact"},
            }
        ]
        fake_vec = [0.25] * 8
        with (
            patch(
                "archivist.features.embeddings.embed_batch",
                new_callable=AsyncMock,
                return_value=[fake_vec],
            ),
            patch(
                "archivist.storage.collection_router.collection_for",
                return_value="col_a",
            ),
        ):
            filled = await _fill_deferred_upsert_vectors(
                collection="col_a",
                points=points,
                embed_inputs={"p1": "augmented embed input"},
            )
        assert filled[0]["vector"] == fake_vec

    async def test_namespace_mismatch_fails_closed(self):
        """When sharding is on, collection/namespace mismatch fails closed."""
        from archivist.storage.outbox import _fill_deferred_upsert_vectors

        points = [
            {
                "id": "p1",
                "vector": [],
                "payload": {"namespace": "other-ns"},
            }
        ]
        with (
            patch("archivist.core.config.NAMESPACE_SHARDING_ENABLED", True),
            patch("archivist.core.config.SINGLE_COLLECTION_MODE", False),
            patch(
                "archivist.storage.collection_router.collection_for",
                return_value="col_other",
            ),
            pytest.raises(ValueError, match="namespace/collection mismatch"),
        ):
            await _fill_deferred_upsert_vectors(
                collection="col_a",
                points=points,
                embed_inputs={"p1": "x"},
            )

    async def test_fill_log_has_no_embeddings_or_fact_text(self, caplog):
        from archivist.storage.outbox import _fill_deferred_upsert_vectors

        secret = "Coach note: sleep debt rises after late meals"
        points = [
            {
                "id": "p1",
                "vector": [],
                "payload": {"namespace": "coach-ns", "text": secret},
            }
        ]
        fake_vec = [0.11, 0.22, 0.33]
        with (
            patch(
                "archivist.features.embeddings.embed_batch",
                new_callable=AsyncMock,
                return_value=[fake_vec],
            ),
            patch(
                "archivist.storage.collection_router.collection_for",
                return_value="archivist_memories",
            ),
            caplog.at_level(logging.INFO, logger="archivist.outbox"),
        ):
            await _fill_deferred_upsert_vectors(
                collection="archivist_memories",
                points=points,
                embed_inputs={"p1": secret},
            )
        records = [r for r in caplog.records if "outbox.embed_deferred_filled" in r.message]
        assert len(records) == 1
        blob = f"{records[0].__dict__}"
        assert secret not in blob
        assert "0.11" not in blob
        assert "sleep debt" not in blob


class TestApplyEventDeferred:
    async def test_apply_upsert_fills_before_backend(self):
        from archivist.storage.outbox import EventType, OutboxProcessor

        backend = MagicMock()
        backend.upsert = AsyncMock()
        proc = OutboxProcessor(backend)
        fake_vec = [0.5] * 4
        payload = {
            "collection": "archivist_memories",
            "memory_id": "m1",
            "embed_deferred": True,
            "embed_inputs": {"m1": "text for embed"},
            "points": [
                {
                    "id": "m1",
                    "vector": [],
                    "payload": {"namespace": "coach-ns"},
                }
            ],
        }
        with (
            patch(
                "archivist.features.embeddings.embed_batch",
                new_callable=AsyncMock,
                return_value=[fake_vec],
            ) as emb,
            patch(
                "archivist.storage.collection_router.collection_for",
                return_value="archivist_memories",
            ),
        ):
            await proc._apply_event("evt-1", EventType.QDRANT_UPSERT, payload)

        emb.assert_awaited_once()
        backend.upsert.assert_awaited_once()
        args = backend.upsert.await_args
        assert args.args[0] == "archivist_memories"
        points = args.args[1]
        assert len(points) == 1
        assert list(points[0].vector) == fake_vec
