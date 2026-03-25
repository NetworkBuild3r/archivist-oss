"""Benchmark: text chunking — flat vs hierarchical throughput."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_helpers import make_corpus_text


def _make_long_doc(n_paragraphs: int) -> str:
    return make_corpus_text(n_paragraphs)


@pytest.mark.parametrize("doc_size", [8, 40, 200])
def test_flat_chunking(benchmark, doc_size):
    """Measure flat chunk_text throughput at various document sizes."""
    from chunking import chunk_text

    text = _make_long_doc(doc_size)

    result = benchmark(chunk_text, text, size=800, overlap=100)
    assert len(result) >= 1


@pytest.mark.parametrize("doc_size", [8, 40, 200])
def test_hierarchical_chunking(benchmark, doc_size):
    """Measure hierarchical (parent/child) chunking throughput."""
    from chunking import chunk_text_hierarchical

    text = _make_long_doc(doc_size)

    result = benchmark(
        chunk_text_hierarchical, text, "bench/test.md",
        parent_size=2000, parent_overlap=200,
        child_size=500, child_overlap=100,
    )
    assert len(result) >= 1
    parents = [c for c in result if c["is_parent"]]
    children = [c for c in result if not c["is_parent"]]
    assert len(parents) >= 1
    assert len(children) >= 0
