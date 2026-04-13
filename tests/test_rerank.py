"""Tests for the reranker module."""

import sys
import os

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from reranker import rerank_results


@pytest.mark.asyncio
async def test_rerank_graceful_fallback():
    """When sentence-transformers is not installed, reranker falls back to score-based sort."""
    results = [
        {"score": 0.5, "text": "medium"},
        {"score": 0.9, "text": "high"},
        {"score": 0.3, "text": "low"},
    ]
    reranked = await rerank_results("test query", results, model_name="nonexistent/model", limit=2)
    assert len(reranked) == 2
    assert reranked[0]["score"] >= reranked[1]["score"]


@pytest.mark.asyncio
async def test_rerank_empty_results():
    """Empty input returns empty output."""
    reranked = await rerank_results("test query", [], limit=5)
    assert reranked == []


@pytest.mark.asyncio
async def test_rerank_limit():
    """Limit parameter is respected."""
    results = [
        {"score": 0.8, "text": "a"},
        {"score": 0.7, "text": "b"},
        {"score": 0.6, "text": "c"},
        {"score": 0.5, "text": "d"},
    ]
    reranked = await rerank_results("query", results, model_name="nonexistent/model", limit=2)
    assert len(reranked) == 2


@pytest.mark.asyncio
async def test_rerank_preserves_fields():
    """Non-score fields are preserved through reranking."""
    results = [
        {"score": 0.8, "text": "hello", "file_path": "a.md", "date": "2024-01-01"},
    ]
    reranked = await rerank_results("hello", results, model_name="nonexistent/model", limit=5)
    assert reranked[0]["file_path"] == "a.md"
    assert reranked[0]["date"] == "2024-01-01"


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_rerank_graceful_fallback())
    asyncio.run(test_rerank_empty_results())
    asyncio.run(test_rerank_limit())
    asyncio.run(test_rerank_preserves_fields())
    print("All rerank tests passed ✓")
