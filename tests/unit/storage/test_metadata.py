"""Unit tests for INIT-023/SPEC-004 archivist_meta embedding mismatch guard."""

from __future__ import annotations

import sqlite3

import pytest

from archivist.storage.metadata import (
    EmbeddingModelMismatchError,
    read_embedding_metadata_sync,
    verify_or_record_embedding_metadata_sync,
)

pytestmark = [pytest.mark.unit]


class TestEmbeddingMetadataGuard:
    def test_records_on_first_boot_and_passes_on_match(self):
        conn = sqlite3.connect(":memory:")
        meta = verify_or_record_embedding_metadata_sync(
            conn, embed_model="BAAI/bge-small-en-v1.5", vector_dim=384, schema_version="3.0"
        )
        assert meta.embed_model == "BAAI/bge-small-en-v1.5"
        assert meta.vector_dim == 384

        # Matching second call succeeds cleanly
        again = verify_or_record_embedding_metadata_sync(
            conn, embed_model="BAAI/bge-small-en-v1.5", vector_dim=384
        )
        assert again.embed_model == "BAAI/bge-small-en-v1.5"
        stored = read_embedding_metadata_sync(conn)
        assert stored is not None
        assert stored.vector_dim == 384

    def test_raises_actionable_error_on_model_mismatch_without_dropping_data(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO memories VALUES ('m1', 'important memory')")
        conn.commit()

        verify_or_record_embedding_metadata_sync(
            conn, embed_model="text-embedding-3-small", vector_dim=1536
        )

        with pytest.raises(EmbeddingModelMismatchError, match="archivist reindex"):
            verify_or_record_embedding_metadata_sync(
                conn, embed_model="BAAI/bge-small-en-v1.5", vector_dim=384
            )

        # Verify zero destructive drops occurred
        row = conn.execute("SELECT body FROM memories WHERE id = 'm1'").fetchone()
        assert row == ("important memory",)
        stored = read_embedding_metadata_sync(conn)
        assert stored is not None
        assert stored.embed_model == "text-embedding-3-small"
        assert stored.vector_dim == 1536

    def test_raises_on_dim_mismatch_same_model(self):
        conn = sqlite3.connect(":memory:")
        verify_or_record_embedding_metadata_sync(
            conn, embed_model="nomic-embed-text-v1.5", vector_dim=768
        )
        with pytest.raises(EmbeddingModelMismatchError, match="vector_dim=512"):
            verify_or_record_embedding_metadata_sync(
                conn, embed_model="nomic-embed-text-v1.5", vector_dim=512
            )
