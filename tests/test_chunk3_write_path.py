"""Tests for Chunk 3: Write Path Fixes.

Covers:
- Reverse HyDE runs as fire-and-forget background task (N0-SYNC)
- Micro-chunks are capped at MAX_MICRO_CHUNKS_PER_MEMORY (N3-UNBOUNDED-MICRO)
- MAX_MICRO_CHUNKS_PER_MEMORY config exists and defaults to 5
- Reverse HyDE errors are logged at WARNING, not DEBUG (N0-LOG-LEVEL)
- Indexer reverse HyDE uses asyncio.gather with semaphore (N0-SERIAL-LLM)
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMaxMicroChunksConfig:
    """MAX_MICRO_CHUNKS_PER_MEMORY config exists with correct default."""

    def test_config_exists(self):
        from config import MAX_MICRO_CHUNKS_PER_MEMORY

        assert isinstance(MAX_MICRO_CHUNKS_PER_MEMORY, int)
        assert MAX_MICRO_CHUNKS_PER_MEMORY == 5

    def test_config_overridable(self, monkeypatch):
        monkeypatch.setenv("MAX_MICRO_CHUNKS_PER_MEMORY", "10")
        import importlib

        import config

        importlib.reload(config)
        assert config.MAX_MICRO_CHUNKS_PER_MEMORY == 10
        monkeypatch.setenv("MAX_MICRO_CHUNKS_PER_MEMORY", "5")
        importlib.reload(config)


class TestMicroChunkCap:
    """Micro-chunks are truncated to MAX_MICRO_CHUNKS_PER_MEMORY."""

    @pytest.mark.asyncio
    async def test_micro_chunks_capped(self, monkeypatch):
        many_chunks = [f"chunk-{i}: 192.168.1.{i}" for i in range(20)]

        with (
            patch("handlers.tools_storage._extract_needle_micro_chunks", return_value=many_chunks),
            patch(
                "handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
            patch(
                "conflict_detection.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024
            ),
            patch(
                "handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                return_value=[[0.1] * 1024] * 5,
            ),
            patch(
                "handlers.tools_storage.check_for_conflicts",
                new_callable=AsyncMock,
                return_value=MagicMock(has_conflict=False),
            ),
            patch(
                "handlers.tools_storage.llm_adjudicated_dedup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("handlers.tools_storage.qdrant_client") as mock_qc,
            patch("handlers.tools_storage.ensure_collection", return_value="test_col"),
            patch("audit.log_memory_event", new_callable=AsyncMock),
            patch("handlers.tools_storage.get_namespace_for_agent", return_value="test-ns"),
            patch("handlers.tools_storage.get_namespace_config", return_value=None),
            patch("handlers.tools_storage._rbac_gate", return_value=None),
        ):
            monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", False)
            monkeypatch.setattr("config.BM25_ENABLED", False)
            monkeypatch.setattr("config.MAX_MICRO_CHUNKS_PER_MEMORY", 3)

            mock_client = MagicMock()
            mock_qc.return_value = mock_client

            from handlers.tools_storage import _handle_store

            await _handle_store({"text": "test text with 10.0.0.1", "agent_id": "test"})

            upsert_calls = mock_client.upsert.call_args_list
            micro_points = []
            for c in upsert_calls:
                pts = c[1].get("points", []) if "points" in c[1] else c[0][0] if c[0] else []
                for p in pts:
                    if hasattr(p, "payload") and p.payload.get("parent_id"):
                        micro_points.append(p)
            assert len(micro_points) <= 3, f"Expected <= 3 micro-chunks, got {len(micro_points)}"


class TestReverseHydeFireAndForget:
    """Reverse HyDE runs as background task, not blocking store response."""

    @pytest.mark.asyncio
    async def test_store_returns_before_hyde_completes(self, monkeypatch):
        hyde_started = asyncio.Event()
        hyde_gate = asyncio.Event()

        async def slow_hyde(text):
            hyde_started.set()
            await hyde_gate.wait()
            return ["What is this?"]

        with (
            patch(
                "handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
            patch(
                "conflict_detection.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024
            ),
            patch(
                "handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                return_value=[[0.1] * 1024],
            ),
            patch(
                "handlers.tools_storage.check_for_conflicts",
                new_callable=AsyncMock,
                return_value=MagicMock(has_conflict=False),
            ),
            patch(
                "handlers.tools_storage.llm_adjudicated_dedup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("handlers.tools_storage.qdrant_client") as mock_qc,
            patch("handlers.tools_storage.ensure_collection", return_value="test_col"),
            patch("audit.log_memory_event", new_callable=AsyncMock),
            patch("handlers.tools_storage._extract_needle_micro_chunks", return_value=[]),
            patch("handlers.tools_storage.get_namespace_for_agent", return_value="test-ns"),
            patch("handlers.tools_storage.get_namespace_config", return_value=None),
            patch("handlers.tools_storage._rbac_gate", return_value=None),
            patch("hyde.generate_reverse_hyde_questions", side_effect=slow_hyde),
        ):
            monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", True)
            monkeypatch.setattr("config.BM25_ENABLED", False)

            mock_client = MagicMock()
            mock_qc.return_value = mock_client

            from handlers.tools_storage import _handle_store

            result = await _handle_store({"text": "test text", "agent_id": "test"})

            import json

            data = json.loads(result[0].text)
            assert data["stored"] is True

            hyde_gate.set()
            await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_hyde_failure_logs_warning(self, monkeypatch):
        logged_warnings = []
        original_warning = None

        import logging

        test_logger = logging.getLogger("archivist.mcp")
        original_warning = test_logger.warning

        def capture_warning(msg, *args, **kwargs):
            logged_warnings.append(msg % args if args else msg)

        async def failing_hyde(text):
            raise RuntimeError("LLM is down")

        with (
            patch(
                "handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
            patch(
                "conflict_detection.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024
            ),
            patch(
                "handlers.tools_storage.check_for_conflicts",
                new_callable=AsyncMock,
                return_value=MagicMock(has_conflict=False),
            ),
            patch(
                "handlers.tools_storage.llm_adjudicated_dedup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("handlers.tools_storage.qdrant_client") as mock_qc,
            patch("handlers.tools_storage.ensure_collection", return_value="test_col"),
            patch("audit.log_memory_event", new_callable=AsyncMock),
            patch("handlers.tools_storage._extract_needle_micro_chunks", return_value=[]),
            patch("handlers.tools_storage.get_namespace_for_agent", return_value="test-ns"),
            patch("handlers.tools_storage.get_namespace_config", return_value=None),
            patch("handlers.tools_storage._rbac_gate", return_value=None),
            patch("hyde.generate_reverse_hyde_questions", side_effect=failing_hyde),
            patch.object(test_logger, "warning", side_effect=capture_warning),
        ):
            monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", True)
            monkeypatch.setattr("config.BM25_ENABLED", False)

            mock_client = MagicMock()
            mock_qc.return_value = mock_client

            from handlers.tools_storage import _handle_store

            await _handle_store({"text": "test text", "agent_id": "test"})

            await asyncio.sleep(0.1)

        hyde_warnings = [w for w in logged_warnings if "Reverse HyDE" in w]
        assert len(hyde_warnings) >= 1, (
            f"Expected warning about HyDE failure, got: {logged_warnings}"
        )


class TestIndexerParallelReverseHyde:
    """Indexer uses asyncio.gather with semaphore for reverse HyDE."""

    def test_indexer_imports_asyncio(self):
        import indexer

        assert hasattr(indexer, "asyncio"), "indexer must import asyncio for gather/semaphore"

    def test_gather_uses_return_exceptions(self):
        """asyncio.gather call must have return_exceptions=True for resilience."""
        import indexer

        source = inspect.getsource(indexer.index_file)
        assert "return_exceptions=True" in source, (
            "asyncio.gather in indexer must use return_exceptions=True"
        )

    @pytest.mark.asyncio
    async def test_parallel_hyde_calls(self, monkeypatch, tmp_path):
        call_times = []

        async def tracked_hyde(text):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)
            return ["Q1?"]

        test_file = tmp_path / "memories" / "agent" / "test.md"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("Parent chunk 1. " * 50 + "\n\n" + "Parent chunk 2. " * 50)

        monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", True)
        monkeypatch.setattr("config.BM25_ENABLED", False)
        monkeypatch.setattr("config.TIERED_CONTEXT_ENABLED", False)
        monkeypatch.setattr("config.TOPIC_ROUTING_ENABLED", False)
        monkeypatch.setattr("config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
        monkeypatch.setattr("config.MEMORY_ROOT", str(tmp_path / "memories"))

        import indexer

        monkeypatch.setattr(indexer, "MEMORY_ROOT", str(tmp_path / "memories"))

        with (
            patch("indexer.embed_batch", new_callable=AsyncMock, return_value=[[0.1] * 1024] * 50),
            patch("indexer.qdrant_client") as mock_qc,
            patch("indexer.ensure_collection", return_value="test_col"),
            patch("indexer.delete_file_points", new_callable=AsyncMock),
            patch("indexer.get_namespace_for_agent", return_value="test-ns"),
            patch("indexer.get_namespace_config", return_value=None),
            patch("hyde.generate_reverse_hyde_questions", side_effect=tracked_hyde),
        ):
            mock_client = MagicMock()
            mock_qc.return_value = mock_client

            await indexer.index_file(str(test_file), hierarchical=True)

        assert len(call_times) >= 1, "Expected at least one HyDE call"
