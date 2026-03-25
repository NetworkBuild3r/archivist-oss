"""Shared data generators for Archivist benchmark suite.

Imported by individual bench_*.py files. Kept separate from conftest.py
to avoid import-name collisions with the root conftest.
"""

import uuid


def make_corpus_text(n_paragraphs: int = 8) -> str:
    """Generate a synthetic markdown document with N paragraphs."""
    paragraphs = []
    for i in range(n_paragraphs):
        paragraphs.append(
            f"## Section {i + 1}\n\n"
            f"This is paragraph {i + 1} of the benchmark corpus document. "
            f"It contains technical content about memory retrieval systems, "
            f"knowledge graphs, and multi-agent orchestration. "
            f"The system processes entity-relationship triples with temporal "
            f"decay scoring and hybrid BM25-vector fusion.\n"
        )
    return "\n".join(paragraphs)


def make_vector_results(n: int, base_score: float = 0.85) -> list[dict]:
    """Generate N synthetic vector search results for fusion benchmarks."""
    results = []
    for i in range(n):
        results.append({
            "id": str(uuid.uuid4()),
            "score": max(0.1, base_score - i * 0.02),
            "text": f"Vector result chunk {i}: memory about deployment pipeline "
                    f"and monitoring configuration for agent fleet operations.",
            "l0": f"Summary L0 for chunk {i}",
            "l1": f"Detailed summary L1 for chunk {i} covering deployment details",
            "agent_id": f"agent-{i % 5}",
            "file_path": f"agents/agent-{i % 5}/2025-03-{10 + i:02d}.md",
            "file_type": "daily",
            "date": f"2025-03-{10 + i:02d}",
            "team": "platform",
            "namespace": "pipeline",
            "chunk_index": i,
            "parent_id": None,
            "is_parent": i % 3 == 0,
        })
    return results


def make_bm25_results(n: int, base_score: float = 5.0) -> list[dict]:
    """Generate N synthetic BM25 results for fusion benchmarks."""
    results = []
    for i in range(n):
        results.append({
            "qdrant_id": str(uuid.uuid4()),
            "bm25_score": max(0.5, base_score - i * 0.3),
            "text": f"BM25 result chunk {i}: keyword match for deployment and "
                    f"pipeline configuration in the agent memory store.",
            "file_path": f"agents/agent-{i % 5}/2025-03-{15 + i:02d}.md",
            "chunk_index": i,
            "agent_id": f"agent-{i % 5}",
            "namespace": "pipeline",
            "date": f"2025-03-{15 + i:02d}",
            "memory_type": "general",
        })
    return results


def populate_fts_chunks(n: int, namespace: str = "bench") -> list[str]:
    """Insert N chunks into FTS5 for search benchmarks. Returns qdrant_ids."""
    from graph import upsert_fts_chunk

    ids = []
    for i in range(n):
        qid = str(uuid.uuid4())
        text = (
            f"Chunk {i}: This memory describes the deployment pipeline for the "
            f"Kubernetes cluster managed by agent-{i % 10}. "
            f"It includes Helm chart versioning, ArgoCD sync waves, "
            f"and Prometheus monitoring configuration. "
            f"The rollback policy uses canary analysis with Flagger."
        )
        upsert_fts_chunk(
            qdrant_id=qid,
            text=text,
            file_path=f"agents/agent-{i % 10}/2025-03-{(i % 28) + 1:02d}.md",
            chunk_index=i,
            agent_id=f"agent-{i % 10}",
            namespace=namespace,
            date=f"2025-03-{(i % 28) + 1:02d}",
        )
        ids.append(qid)
    return ids
