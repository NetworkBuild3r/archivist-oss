"""INIT-005/SPEC-003 — ack-budget hard-skips for optional store gates.

Covers:
- Conflict / dedup / extract hard-skip when budget expired or below floor
- No-skip path when budget has headroom
- Outbox still enqueued on success (GR-DUR-001)
- Skip observability via structured log (SM-002; no fact text)
- RBAC still enforced before optional gates
"""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit]


class _BudgetStub:
    """Minimal LatencyBudget stand-in for hard-skip predicate tests."""

    def __init__(self, *, expired: bool = False, remaining_ms: float = 500.0):
        self._expired = expired
        self._remaining_ms = remaining_ms

    def is_expired(self) -> bool:
        return self._expired

    def remaining_ms(self) -> float:
        return self._remaining_ms

    def summary(self) -> dict:
        return {
            "budget_ms": 4000,
            "elapsed_ms": round(4000 - self._remaining_ms, 1),
            "remaining_ms": round(self._remaining_ms, 1),
            "reserved_ms": 0.0,
            "reservations": {},
        }


@contextmanager
def _store_patches(*, conflict_check: bool = True):
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
            patch(
                "archivist.app.handlers.tools_storage._query_similar",
                new_callable=AsyncMock,
                return_value=([0.1] * 1024, []),
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
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.conflict_vec_for_primary_embed",
                return_value=None,
            )
        )
        yield mock_client
    finally:
        stack.close()


def _configure_store_flags(monkeypatch, *, conflict_check: bool = True) -> None:
    monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
    monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", conflict_check)
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.CONFLICT_CHECK_ON_STORE", conflict_check
    )
    monkeypatch.setattr("archivist.core.config.CONFLICT_BLOCK_ON_STORE", False)
    monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_BLOCK_ON_STORE", False)
    monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.TOPIC_ROUTING_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.DEDUP_LLM_ENABLED", True)


# ---------------------------------------------------------------------------
# Predicate / emit helpers
# ---------------------------------------------------------------------------


class TestHardSkipPredicate:
    def test_expired_budget_hard_skips(self):
        from archivist.app.handlers.tools_storage import _should_hard_skip_optional_gates

        assert _should_hard_skip_optional_gates(_BudgetStub(expired=True, remaining_ms=0)) is True

    def test_below_floor_hard_skips(self):
        from archivist.app.handlers.tools_storage import (
            _ACK_OPTIONAL_GATE_MIN_REMAINING_MS,
            _should_hard_skip_optional_gates,
        )

        below = _ACK_OPTIONAL_GATE_MIN_REMAINING_MS - 1
        assert (
            _should_hard_skip_optional_gates(_BudgetStub(expired=False, remaining_ms=below)) is True
        )

    def test_above_floor_does_not_skip(self):
        from archivist.app.handlers.tools_storage import (
            _ACK_OPTIONAL_GATE_MIN_REMAINING_MS,
            _should_hard_skip_optional_gates,
        )

        above = float(_ACK_OPTIONAL_GATE_MIN_REMAINING_MS + 50)
        assert (
            _should_hard_skip_optional_gates(_BudgetStub(expired=False, remaining_ms=above))
            is False
        )

    def test_emit_has_no_fact_text(self, caplog):
        from archivist.app.handlers.tools_storage import _emit_ack_hard_skip

        with caplog.at_level(logging.INFO, logger="archivist.mcp"):
            _emit_ack_hard_skip(
                "conflict",
                namespace="coach-ns",
                ack_budget=_BudgetStub(expired=True, remaining_ms=0),
            )
        records = [r for r in caplog.records if "store_pipeline.hard_skip" in r.message]
        assert len(records) == 1
        rec = records[0]
        assert rec.gate == "conflict"
        assert rec.namespace == "coach-ns"
        # SM-002: no fact/memory text in structured extras
        extra_blob = f"{rec.__dict__}"
        assert "Coach note" not in extra_blob
        assert "sleep debt" not in extra_blob


# ---------------------------------------------------------------------------
# Store path: skip + no-skip + outbox durability
# ---------------------------------------------------------------------------


class TestStoreHardSkipPaths:
    async def test_expired_budget_skips_conflict_and_dedup_keeps_outbox(
        self, async_pool, monkeypatch, caplog
    ):
        """REQ-003 + GR-DUR-001 + SM-002: hard-skip optional gates; outbox remains."""
        _configure_store_flags(monkeypatch, conflict_check=True)

        expired = _BudgetStub(expired=True, remaining_ms=0)
        monkeypatch.setattr(
            "archivist.core.latency_budget.LatencyBudget",
            lambda *a, **k: expired,
        )

        with _store_patches():
            import archivist.app.handlers.tools_storage as ts

            query_mock = ts._query_similar
            conflict_mock = ts.check_for_conflicts
            dedup_mock = ts.llm_adjudicated_dedup

            with caplog.at_level(logging.INFO, logger="archivist.mcp"):
                result = await ts._handle_store(
                    {
                        "text": "Coach note: sleep debt rises after late meals",
                        "agent_id": "coach-agent",
                        "namespace": "coach-ns",
                        "entities": ["SleepDebtEntity"],
                        "actor_id": "coach-agent",
                        "actor_type": "agent",
                    }
                )

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        mid = data["memory_id"]

        query_mock.assert_not_awaited()
        conflict_mock.assert_not_awaited()
        dedup_mock.assert_not_awaited()

        skip_records = [r for r in caplog.records if "store_pipeline.hard_skip" in r.message]
        gates = {r.gate for r in skip_records}
        assert "conflict" in gates
        assert "dedup" in gates
        for r in skip_records:
            assert "sleep debt" not in f"{r.__dict__}"

        async with async_pool.read() as conn:
            cur = await conn.execute("SELECT COUNT(*) AS c FROM outbox WHERE status='pending'")
            row = await cur.fetchone()
            assert int(row["c"] if isinstance(row, dict) else row[0]) >= 1
            cur = await conn.execute(
                "SELECT qdrant_id FROM memory_chunks WHERE qdrant_id = ?",
                (mid,),
            )
            assert await cur.fetchone() is not None

    async def test_expired_budget_skips_optional_extract(self, async_pool, monkeypatch, caplog):
        """Optional auto-extract hard-skips when no explicit entities and budget expired."""
        _configure_store_flags(monkeypatch, conflict_check=False)

        expired = _BudgetStub(expired=True, remaining_ms=0)
        monkeypatch.setattr(
            "archivist.core.latency_budget.LatencyBudget",
            lambda *a, **k: expired,
        )

        with _store_patches():
            import archivist.app.handlers.tools_storage as ts

            extract_mock = ts._extract_and_store_entities

            with caplog.at_level(logging.INFO, logger="archivist.mcp"):
                result = await ts._handle_store(
                    {
                        "text": "Alice met Bob at the cafe on Main St",
                        "agent_id": "coach-agent",
                        "namespace": "coach-ns",
                        # no entities → auto-extract path
                        "actor_id": "coach-agent",
                        "actor_type": "agent",
                    }
                )

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        extract_mock.assert_not_awaited()
        skip_records = [r for r in caplog.records if "store_pipeline.hard_skip" in r.message]
        assert any(r.gate == "extract" for r in skip_records)

        async with async_pool.read() as conn:
            cur = await conn.execute("SELECT COUNT(*) AS c FROM outbox WHERE status='pending'")
            row = await cur.fetchone()
            assert int(row["c"] if isinstance(row, dict) else row[0]) >= 1

    async def test_healthy_budget_runs_conflict_and_dedup(self, async_pool, monkeypatch):
        """No-skip path: optional gates still run when ack budget has headroom."""
        _configure_store_flags(monkeypatch, conflict_check=True)

        healthy = _BudgetStub(expired=False, remaining_ms=3500)
        monkeypatch.setattr(
            "archivist.core.latency_budget.LatencyBudget",
            lambda *a, **k: healthy,
        )

        with _store_patches():
            import archivist.app.handlers.tools_storage as ts

            query_mock = ts._query_similar
            conflict_mock = ts.check_for_conflicts
            dedup_mock = ts.llm_adjudicated_dedup

            result = await ts._handle_store(
                {
                    "text": "Healthy budget path should run optional gates",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": ["GateEntity"],
                    "actor_id": "coach-agent",
                    "actor_type": "agent",
                }
            )

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        query_mock.assert_awaited()
        conflict_mock.assert_awaited()
        dedup_mock.assert_awaited()

    async def test_rbac_denial_unchanged_under_expired_budget(self, monkeypatch):
        """Hard-skip is not an authz bypass — RBAC still short-circuits first."""
        _configure_store_flags(monkeypatch, conflict_check=True)

        expired = _BudgetStub(expired=True, remaining_ms=0)
        monkeypatch.setattr(
            "archivist.core.latency_budget.LatencyBudget",
            lambda *a, **k: expired,
        )

        denied = [
            type(
                "TC",
                (),
                {"text": json.dumps({"error": "forbidden", "stored": False})},
            )()
        ]

        with _store_patches():
            import archivist.app.handlers.tools_storage as ts

            with patch.object(ts, "require_rbac", return_value=denied):
                with patch.object(ts, "_handle_store_inner", new_callable=AsyncMock) as inner:
                    result = await ts._handle_store(
                        {
                            "text": "should never reach inner",
                            "agent_id": "coach-agent",
                            "namespace": "coach-ns",
                        }
                    )

        assert result is denied
        inner.assert_not_awaited()
