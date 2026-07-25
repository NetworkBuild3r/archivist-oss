"""Characterization tests for ``_handle_store``'s pre-refactor persisted state.

INIT-022/SPEC-008 (ac-6): captures the observable side effects of
``_handle_store`` for representative inputs *before* the ac-3 (`M8`, shared
``_persist_background_points`` helper) and ac-4 (`M7`, entity-extraction +
background-task-builder decomposition) refactor, so the same assertions can
be re-run unchanged afterward to prove the decomposition made no behavior
change beyond the named findings.

Covered trigger conditions:
    - Baseline: both ``REVERSE_HYDE_ENABLED`` and ``SYNTHETIC_QUESTIONS_ENABLED``
      off — only primary + micro-chunk points persisted.
    - Both background-task triggers on simultaneously — reverse-HyDE and
      synthetic-question points persisted via the (soon to be shared) helper.
    - Auto entity-extraction path (no explicit ``entities`` argument) —
      persisted ``entities`` rows match ``pre_extract`` + ``extract_needle_entities``
      output directly.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.models import PointStruct

from archivist.core.config import MAX_MICRO_CHUNKS_PER_MEMORY
from archivist.utils.chunking import _extract_needle_micro_chunks
from archivist.write.pre_extractor import extract_needle_entities, pre_extract

pytestmark = [pytest.mark.system, pytest.mark.mcp, pytest.mark.storage]


def _mock_vec(dim: int = 8) -> list[float]:
    return [0.1] * dim


async def _drain_background_tasks() -> None:
    """Yield control until every fire-and-forget task ``_handle_store`` spawned finishes."""
    current = asyncio.current_task()
    for _ in range(50):
        pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        if not pending:
            return
        await asyncio.sleep(0.01)


async def _mp_point_types(qa_pool, memory_id: str) -> list[str]:
    """Return ``point_type`` values from ``memory_points`` for ``memory_id``."""
    async with qa_pool.read() as conn:
        cur = await conn.execute(
            "SELECT point_type FROM memory_points WHERE memory_id=?",
            (memory_id,),
        )
        return [row["point_type"] for row in await cur.fetchall()]


class TestStoreBaselineCharacterization:
    """``_handle_store`` with both background-task triggers OFF."""

    async def test_store_persists_primary_and_micro_chunk_points(self, qa_pool, memory_factory):
        from archivist.app.handlers.tools_storage import _handle_store

        text = "Qdrant 192.168.1.10:6333 — CONFIG-8901 healthy 2026-04-17T11:00"
        mem = memory_factory(text=text)
        raw_micro_chunks = _extract_needle_micro_chunks(text)
        expected_micro_count = min(len(raw_micro_chunks), MAX_MICRO_CHUNKS_PER_MEMORY)
        assert expected_micro_count > 0, "fixture text must contain needle-matchable tokens"

        with (
            patch(
                "archivist.app.handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=_mock_vec(),
            ),
            patch(
                "archivist.app.handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                side_effect=lambda inputs: [_mock_vec() for _ in inputs],
            ),
            patch(
                "archivist.app.handlers.tools_storage.qdrant_client",
                return_value=MagicMock(upsert=MagicMock()),
            ),
            patch(
                "archivist.app.handlers.tools_storage.ensure_collection",
                return_value="test_col",
            ),
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_config",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_storage.require_rbac",
                return_value=None,
            ),
            patch("archivist.core.config.REVERSE_HYDE_ENABLED", False),
            patch("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False),
        ):
            result = await _handle_store(
                {
                    "text": text,
                    "agent_id": mem["agent_id"],
                    "namespace": mem["namespace"],
                    "actor_id": mem["actor_id"],
                    "actor_type": mem["actor_type"],
                    "force_skip_conflict_check": True,
                }
            )

        data = json.loads(result[0].text)
        assert data["stored"] is True
        pid = data["memory_id"]

        point_types = await _mp_point_types(qa_pool, pid)
        assert point_types.count("primary") == 1
        assert point_types.count("micro_chunk") == expected_micro_count
        assert "reverse_hyde" not in point_types
        assert "synthetic_question" not in point_types


class TestStoreBackgroundFeaturesEnabledCharacterization:
    """``_handle_store`` with both background-task triggers ON simultaneously."""

    async def test_store_persists_reverse_hyde_and_synthetic_question_points(
        self, qa_pool, memory_factory
    ):
        from archivist.app.handlers.tools_storage import _handle_store

        text = "deployment runs on 10.0.0.1/24 — ticket OPS-9012"
        mem = memory_factory(text=text)

        rh_questions = ["What runs on 10.0.0.1?", "Which ticket covers this deployment?"]
        sq_point = PointStruct(
            id="11111111-1111-1111-1111-111111111111",
            vector=_mock_vec(),
            payload={"text": "synthetic question?", "file_type": "synthetic_question"},
        )

        with (
            patch(
                "archivist.app.handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=_mock_vec(),
            ),
            patch(
                "archivist.app.handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                side_effect=lambda inputs: [_mock_vec() for _ in inputs],
            ),
            patch(
                "archivist.app.handlers.tools_storage.qdrant_client",
                return_value=MagicMock(upsert=MagicMock()),
            ),
            patch(
                "archivist.app.handlers.tools_storage.ensure_collection",
                return_value="test_col",
            ),
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_config",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_storage.require_rbac",
                return_value=None,
            ),
            patch("archivist.core.config.REVERSE_HYDE_ENABLED", True),
            patch("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", True),
            patch(
                "archivist.write.hyde.generate_reverse_hyde_questions",
                new_callable=AsyncMock,
                return_value=rh_questions,
            ),
            patch(
                "archivist.write.synthetic_questions.generate_and_embed_synthetic_points",
                new_callable=AsyncMock,
                return_value=[sq_point],
            ),
        ):
            result = await _handle_store(
                {
                    "text": text,
                    "agent_id": mem["agent_id"],
                    "namespace": mem["namespace"],
                    "actor_id": mem["actor_id"],
                    "actor_type": mem["actor_type"],
                    "force_skip_conflict_check": True,
                }
            )
            data = json.loads(result[0].text)
            assert data["stored"] is True
            pid = data["memory_id"]

            await _drain_background_tasks()

        point_types = await _mp_point_types(qa_pool, pid)
        assert point_types.count("reverse_hyde") == len(rh_questions)
        assert point_types.count("synthetic_question") == 1


class TestStoreEntityExtractionCharacterization:
    """``_handle_store``'s auto entity-extraction path (no explicit ``entities``)."""

    async def test_store_auto_extracts_entities_matching_pre_extractor_output(
        self, qa_pool, memory_factory
    ):
        from archivist.app.handlers.tools_storage import _handle_store

        text = "deployment runs on 10.0.0.1/24 — ticket OPS-9012"
        mem = memory_factory(text=text)
        agent_id = mem["agent_id"]

        hints = pre_extract(text)
        needle_entities = extract_needle_entities(text)
        expected_names = {agent_id.strip().lower()}
        for ent in hints.get("entities", []) + needle_entities:
            ename = ent["name"].strip()
            if ename and ename != agent_id:
                expected_names.add(ename.lower())
        assert len(expected_names) > 1, "fixture text must yield at least one extracted entity"

        with (
            patch(
                "archivist.app.handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=_mock_vec(),
            ),
            patch(
                "archivist.app.handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                side_effect=lambda inputs: [_mock_vec() for _ in inputs],
            ),
            patch(
                "archivist.app.handlers.tools_storage.qdrant_client",
                return_value=MagicMock(upsert=MagicMock()),
            ),
            patch(
                "archivist.app.handlers.tools_storage.ensure_collection",
                return_value="test_col",
            ),
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_config",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_storage.require_rbac",
                return_value=None,
            ),
            patch("archivist.core.config.REVERSE_HYDE_ENABLED", False),
            patch("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False),
        ):
            result = await _handle_store(
                {
                    "text": text,
                    "agent_id": agent_id,
                    "namespace": mem["namespace"],
                    "actor_id": mem["actor_id"],
                    "actor_type": mem["actor_type"],
                    "force_skip_conflict_check": True,
                }
            )

        data = json.loads(result[0].text)
        assert data["stored"] is True

        async with qa_pool.read() as conn:
            cur = await conn.execute("SELECT name FROM entities")
            stored_names = {row["name"].strip().lower() for row in await cur.fetchall()}

        assert stored_names == expected_names
