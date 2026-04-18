import pytest

"""Tests for memory_fusion deduplication."""


from memory_fusion import dedupe_vector_hits

pytestmark = [pytest.mark.integration, pytest.mark.lifecycle]


def test_dedupe_drops_duplicate_file_chunk():
    hits = [
        {"file_path": "a.md", "chunk_index": 0, "text": "hello world", "score": 0.9},
        {"file_path": "a.md", "chunk_index": 0, "text": "hello world", "score": 0.8},
        {"file_path": "b.md", "chunk_index": 0, "text": "other", "score": 0.7},
    ]
    out = dedupe_vector_hits(hits)
    assert len(out) == 2

def test_dedupe_keeps_same_file_different_chunk():
    hits = [
        {"file_path": "a.md", "chunk_index": 0, "text": "one", "score": 0.9},
        {"file_path": "a.md", "chunk_index": 1, "text": "two", "score": 0.8},
    ]
    out = dedupe_vector_hits(hits)
    assert len(out) == 2
