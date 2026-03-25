"""Benchmark: FTS5/BM25 search latency at various corpus sizes."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_helpers import populate_fts_chunks


@pytest.mark.parametrize("corpus_size", [100, 1000, 5000])
def test_fts5_search_latency(benchmark, corpus_size, monkeypatch):
    """Measure search_fts latency at different corpus sizes."""
    import graph

    populate_fts_chunks(corpus_size, namespace="bench")

    benchmark(
        graph.search_fts,
        query='"deployment" OR "pipeline" OR "kubernetes"',
        namespace="bench",
        limit=30,
    )


def test_fts5_search_no_namespace_filter(benchmark, monkeypatch):
    """Measure search_fts without namespace filter (wider scan)."""
    import graph

    populate_fts_chunks(1000)

    benchmark(
        graph.search_fts,
        query='"ArgoCD" OR "Helm" OR "canary"',
        limit=30,
    )


def test_fts5_upsert_chunk_latency(benchmark, monkeypatch):
    """Measure single FTS5 upsert (the per-chunk write on indexing path)."""
    import graph
    import uuid

    i = 0

    def _upsert():
        nonlocal i
        graph.upsert_fts_chunk(
            qdrant_id=str(uuid.uuid4()),
            text=f"Chunk {i} about Kubernetes deployment and Helm charts for agent fleet.",
            file_path=f"agents/agent-0/2025-03-{(i % 28) + 1:02d}.md",
            chunk_index=i,
            agent_id="agent-0",
            namespace="bench",
        )
        i += 1

    benchmark(_upsert)
