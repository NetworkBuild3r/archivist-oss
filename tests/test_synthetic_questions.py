"""Tests for synthetic question generation (v2.1 index-time multi-representation)."""

import json
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestSyntheticQuestionParsing:
    def test_parse_json_array(self):
        from synthetic_questions import _parse_questions
        raw = '["What is the server IP?", "Where is the database hosted?", "Which port does redis use?"]'
        result = _parse_questions(raw, 5)
        assert len(result) == 3
        assert "server IP" in result[0]

    def test_parse_json_with_markdown_fences(self):
        from synthetic_questions import _parse_questions
        raw = '```json\n["Q1?", "Q2?", "Q3?"]\n```'
        result = _parse_questions(raw, 5)
        assert len(result) == 3

    def test_parse_fallback_line_per_question(self):
        from synthetic_questions import _parse_questions
        raw = "1. What is X?\n2. Where is Y?\n3. How does Z work?"
        result = _parse_questions(raw, 5)
        assert len(result) == 3

    def test_parse_respects_count_limit(self):
        from synthetic_questions import _parse_questions
        raw = '["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]'
        result = _parse_questions(raw, 2)
        assert len(result) == 2

    def test_parse_empty_input(self):
        from synthetic_questions import _parse_questions
        assert _parse_questions("", 5) == []

    def test_parse_malformed_json(self):
        from synthetic_questions import _parse_questions
        raw = '["Q1?", "Q2?", broken'
        result = _parse_questions(raw, 5)
        assert isinstance(result, list)


class TestSyntheticQuestionGeneration:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self, monkeypatch):
        import synthetic_questions
        monkeypatch.setattr(synthetic_questions, "SYNTHETIC_QUESTIONS_ENABLED", False)
        result = await synthetic_questions.generate_synthetic_questions("some text")
        assert result == []

    @pytest.mark.asyncio
    async def test_generates_questions_from_llm(self, monkeypatch):
        import synthetic_questions
        monkeypatch.setattr(synthetic_questions, "SYNTHETIC_QUESTIONS_ENABLED", True)
        monkeypatch.setattr(synthetic_questions, "SYNTHETIC_QUESTIONS_COUNT", 3)

        mock_response = '["What is the backup schedule?", "When do backups run?", "How is data backed up?"]'
        with patch("synthetic_questions.llm_query", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            synthetic_questions._cache.clear()
            result = await synthetic_questions.generate_synthetic_questions("Backups run every 6 hours via cron.")

        assert len(result) == 3
        assert "backup" in result[0].lower()

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_llm_call(self, monkeypatch):
        import synthetic_questions
        monkeypatch.setattr(synthetic_questions, "SYNTHETIC_QUESTIONS_ENABLED", True)

        text = "Cache test text"
        key = synthetic_questions._cache_key(text)
        synthetic_questions._cache_put(key, ["cached Q1?", "cached Q2?"])

        with patch("synthetic_questions.llm_query", new_callable=AsyncMock) as mock_llm:
            result = await synthetic_questions.generate_synthetic_questions(text)

        mock_llm.assert_not_called()
        assert result == ["cached Q1?", "cached Q2?"]
        synthetic_questions._cache.clear()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self, monkeypatch):
        import synthetic_questions
        monkeypatch.setattr(synthetic_questions, "SYNTHETIC_QUESTIONS_ENABLED", True)

        with patch("synthetic_questions.llm_query", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM down")
            synthetic_questions._cache.clear()
            result = await synthetic_questions.generate_synthetic_questions("Some text")

        assert result == []


class TestSyntheticQuestionPointGeneration:
    @pytest.mark.asyncio
    async def test_generates_qdrant_points(self, monkeypatch):
        import synthetic_questions
        monkeypatch.setattr(synthetic_questions, "SYNTHETIC_QUESTIONS_ENABLED", True)
        monkeypatch.setattr(synthetic_questions, "SYNTHETIC_QUESTIONS_COUNT", 2)

        mock_qs = '["What is X?", "Where is Y?"]'
        with patch("synthetic_questions.llm_query", new_callable=AsyncMock) as mock_llm, \
             patch("synthetic_questions.embed_batch", new_callable=AsyncMock) as mock_embed:
            mock_llm.return_value = mock_qs
            mock_embed.return_value = [[0.1] * 1024, [0.2] * 1024]
            synthetic_questions._cache.clear()

            points = await synthetic_questions.generate_and_embed_synthetic_points(
                chunk_point_id="test-chunk-id",
                chunk_text="The server IP is 10.0.0.1",
                base_payload={"agent_id": "test", "namespace": "default"},
            )

        assert len(points) == 2
        for p in points:
            assert p.payload["representation_type"] == "synthetic_question"
            assert p.payload["source_memory_id"] == "test-chunk-id"
            assert p.payload["parent_id"] == "test-chunk-id"
            assert "synthetic_question" in p.payload
            assert p.payload["text"] == "The server IP is 10.0.0.1"

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self, monkeypatch):
        import synthetic_questions
        monkeypatch.setattr(synthetic_questions, "SYNTHETIC_QUESTIONS_ENABLED", False)

        points = await synthetic_questions.generate_and_embed_synthetic_points(
            chunk_point_id="id",
            chunk_text="text",
            base_payload={},
        )
        assert points == []
