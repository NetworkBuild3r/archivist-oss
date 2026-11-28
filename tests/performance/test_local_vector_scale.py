"""Scale performance tests for SPEC-006 LocalVectorIndex (50K local hybrid & D3)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from archivist.storage.local_store import LocalMemoryStore

pytestmark = [pytest.mark.performance, pytest.mark.slow]


def test_50k_local_hybrid_retrieval_p50_and_sync_write_latency(tmp_path: Path, capsys):
    """50K chunks: local hybrid p50 <75ms, sync write <30ms, 0 stdout bytes."""
    db = tmp_path / "scale_50k.db"
    store = LocalMemoryStore(
        db_path=db,
        vector_backend="sqlite-vec",
        quantization="bit",
        offline_deterministic=True,
    )
    try:
        rng = np.random.default_rng(42)
        n = 50_000
        dim = store.vector_dim
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_bit = np.packbits((vecs > 0).astype(np.uint8), axis=-1, bitorder="little")

        # Fast bulk seed 50K chunks + vectors
        batch = 5000
        with store._write_conn:
            for start in range(0, n, batch):
                end = min(start + batch, n)
                store._write_conn.executemany(
                    "INSERT INTO memory_chunks "
                    "(id, scope_id, content, l0, l1, kind, tags_json, trust_tier, "
                    "actor_id, source_client, created_at, updated_at, content_hash, token_count) "
                    "VALUES (?, 'repo:scale', ?, ?, ?, 'fact', '[]', 'T1_AGENT', 'local', 'bench', "
                    "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', ?, 12)",
                    [
                        (
                            f"m_{i}",
                            f"Document chunk {i} covering subsystem_{i % 200} metric_{i % 50}",
                            f"Chunk {i}",
                            f"Chunk {i}",
                            f"hash_{i}",
                        )
                        for i in range(start, end)
                    ],
                )
                store._write_conn.executemany(
                    "INSERT INTO vec_id_map (vec_rowid, chunk_id, scope_id) VALUES (?, ?, 'repo:scale')",
                    [(i + 1, f"m_{i}") for i in range(start, end)],
                )
                store._write_conn.executemany(
                    "INSERT INTO chunk_vectors_f32 (vec_rowid, embedding) VALUES (?, ?)",
                    [(i + 1, vecs[i].tobytes()) for i in range(start, end)],
                )
                store._write_conn.executemany(
                    "INSERT INTO chunk_vectors_coarse (vec_rowid, scope_id, embedding) "
                    "VALUES (?, 'repo:scale', vec_bit(?))",
                    [(i + 1, vecs_bit[i].tobytes()) for i in range(start, end)],
                )

        # Measure sync write latency on top of 50K DB
        write_lats: list[float] = []
        for i in range(10):
            t0 = time.perf_counter()
            store.save_memory(
                f"Fresh decision {i}: use single-file SQLite WAL for deterministic zero-LLM writes",
                scope="repo:scale",
            )
            write_lats.append((time.perf_counter() - t0) * 1000.0)

        # Measure hybrid retrieval latency across 20 queries
        query_lats: list[float] = []
        for i in range(20):
            t0 = time.perf_counter()
            hits = store.search_memories(
                f"deterministic zero-LLM writes decision {i % 10}",
                scopes=["repo:scale"],
                limit=10,
            )
            query_lats.append((time.perf_counter() - t0) * 1000.0)
            assert hits

        write_p50 = float(np.percentile(write_lats, 50))
        query_p50 = float(np.percentile(query_lats, 50))
        assert write_p50 < 30.0, f"Sync write p50={write_p50:.2f}ms exceeded 30ms"
        assert query_p50 < 75.0, f"50K hybrid retrieval p50={query_p50:.2f}ms exceeded 75ms"
        assert capsys.readouterr().out == ""
    finally:
        store.close()
