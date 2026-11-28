"""Database embedding metadata guard (``archivist_meta`` table — INIT-023/SPEC-004).

Persists ``(embed_model, vector_dim, schema_version)`` in SQLite or PostgreSQL
and refuses to silently drop or mix vector indexes when the configured embedding
model or vector dimension changes. Instead raises
:class:`EmbeddingModelMismatchError` pointing operators to ``archivist reindex``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

META_KEY_EMBED_MODEL = "embed_model"
META_KEY_VECTOR_DIM = "vector_dim"
META_KEY_SCHEMA_VERSION = "schema_version"
META_KEY_UPDATED_AT = "updated_at"

ARCHIVIST_META_DDL = """
CREATE TABLE IF NOT EXISTS archivist_meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class EmbeddingMetadata:
    """Persisted embedding model & vector dimension metadata."""

    embed_model: str
    vector_dim: int
    schema_version: str = "3.0"
    updated_at: str = ""


class EmbeddingModelMismatchError(RuntimeError):
    """Raised when the configured embedding model or dimension differs from the DB index.

    Never drops or mutates stored vectors; directs the caller to run
    ``archivist reindex`` explicitly.
    """

    def __init__(
        self,
        *,
        stored_model: str,
        stored_dim: int,
        requested_model: str,
        requested_dim: int,
    ) -> None:
        self.stored_model = stored_model
        self.stored_dim = stored_dim
        self.requested_model = requested_model
        self.requested_dim = requested_dim
        super().__init__(
            f"Embedding index mismatch: database was indexed with "
            f"embed_model={stored_model!r} (vector_dim={stored_dim}), "
            f"but runtime requested embed_model={requested_model!r} "
            f"(vector_dim={requested_dim}). Refusing to silently drop or mix "
            f"embeddings — run `archivist reindex` to rebuild the index safely."
        )


def _validate_inputs(embed_model: str, vector_dim: int) -> tuple[str, int]:
    model = (embed_model or "").strip()
    if not model:
        raise ValueError("embed_model must be a non-empty string")
    dim = int(vector_dim)
    if dim <= 0:
        raise ValueError(f"vector_dim must be positive, got {dim}")
    return model, dim


def ensure_meta_table_sync(conn: sqlite3.Connection) -> None:
    """Create ``archivist_meta`` table idempotently on a synchronous SQLite connection."""
    conn.execute(ARCHIVIST_META_DDL)
    conn.commit()


def read_embedding_metadata_sync(conn: sqlite3.Connection) -> EmbeddingMetadata | None:
    """Read stored ``(embed_model, vector_dim, schema_version)`` from SQLite, or ``None``."""
    ensure_meta_table_sync(conn)
    rows = conn.execute(
        "SELECT key, value, updated_at FROM archivist_meta WHERE key IN (?, ?, ?)",
        (META_KEY_EMBED_MODEL, META_KEY_VECTOR_DIM, META_KEY_SCHEMA_VERSION),
    ).fetchall()
    kv = {str(r[0]): str(r[1]) for r in rows}
    if META_KEY_EMBED_MODEL not in kv or META_KEY_VECTOR_DIM not in kv:
        return None
    updated = str(rows[0][2]) if rows else ""
    return EmbeddingMetadata(
        embed_model=kv[META_KEY_EMBED_MODEL],
        vector_dim=int(kv[META_KEY_VECTOR_DIM]),
        schema_version=kv.get(META_KEY_SCHEMA_VERSION, "3.0"),
        updated_at=updated,
    )


def verify_or_record_embedding_metadata_sync(
    conn: sqlite3.Connection,
    *,
    embed_model: str,
    vector_dim: int,
    schema_version: str = "3.0",
) -> EmbeddingMetadata:
    """Verify runtime ``(embed_model, vector_dim)`` against SQLite ``archivist_meta``.

    - On an uninitialized DB: records ``(embed_model, vector_dim, schema_version)``.
    - On a matching DB: returns the stored :class:`EmbeddingMetadata` untouched.
    - On a mismatch: raises :class:`EmbeddingModelMismatchError` without touching data.
    """
    req_model, req_dim = _validate_inputs(embed_model, vector_dim)
    existing = read_embedding_metadata_sync(conn)
    if existing is not None:
        if existing.embed_model != req_model or existing.vector_dim != req_dim:
            raise EmbeddingModelMismatchError(
                stored_model=existing.embed_model,
                stored_dim=existing.vector_dim,
                requested_model=req_model,
                requested_dim=req_dim,
            )
        return existing

    now_iso = datetime.now(UTC).isoformat()
    for k, v in (
        (META_KEY_EMBED_MODEL, req_model),
        (META_KEY_VECTOR_DIM, str(req_dim)),
        (META_KEY_SCHEMA_VERSION, schema_version),
    ):
        conn.execute(
            "INSERT INTO archivist_meta (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (k, v, now_iso),
        )
    conn.commit()
    return EmbeddingMetadata(
        embed_model=req_model,
        vector_dim=req_dim,
        schema_version=schema_version,
        updated_at=now_iso,
    )


async def ensure_meta_table(pool: Any) -> None:
    """Create ``archivist_meta`` idempotently via an async graph backend pool."""
    async with pool.write() as db:
        await db.execute(ARCHIVIST_META_DDL)
        await db.commit()


async def read_embedding_metadata(pool: Any) -> EmbeddingMetadata | None:
    """Read stored ``(embed_model, vector_dim, schema_version)`` via an async pool."""
    await ensure_meta_table(pool)
    async with pool.read() as db:
        cur = await db.execute(
            "SELECT key, value, updated_at FROM archivist_meta WHERE key IN (?, ?, ?)",
            (META_KEY_EMBED_MODEL, META_KEY_VECTOR_DIM, META_KEY_SCHEMA_VERSION),
        )
        rows = await cur.fetchall()
    kv = {str(r[0]): str(r[1]) for r in rows}
    if META_KEY_EMBED_MODEL not in kv or META_KEY_VECTOR_DIM not in kv:
        return None
    updated = str(rows[0][2]) if rows else ""
    return EmbeddingMetadata(
        embed_model=kv[META_KEY_EMBED_MODEL],
        vector_dim=int(kv[META_KEY_VECTOR_DIM]),
        schema_version=kv.get(META_KEY_SCHEMA_VERSION, "3.0"),
        updated_at=updated,
    )


async def verify_or_record_embedding_metadata(
    pool: Any,
    *,
    embed_model: str,
    vector_dim: int,
    schema_version: str = "3.0",
) -> EmbeddingMetadata:
    """Async counterpart to :func:`verify_or_record_embedding_metadata_sync`."""
    req_model, req_dim = _validate_inputs(embed_model, vector_dim)
    existing = await read_embedding_metadata(pool)
    if existing is not None:
        if existing.embed_model != req_model or existing.vector_dim != req_dim:
            raise EmbeddingModelMismatchError(
                stored_model=existing.embed_model,
                stored_dim=existing.vector_dim,
                requested_model=req_model,
                requested_dim=req_dim,
            )
        return existing

    now_iso = datetime.now(UTC).isoformat()
    async with pool.write() as db:
        for k, v in (
            (META_KEY_EMBED_MODEL, req_model),
            (META_KEY_VECTOR_DIM, str(req_dim)),
            (META_KEY_SCHEMA_VERSION, schema_version),
        ):
            await db.execute(
                "INSERT INTO archivist_meta (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (k, v, now_iso),
            )
        await db.commit()
    return EmbeddingMetadata(
        embed_model=req_model,
        vector_dim=req_dim,
        schema_version=schema_version,
        updated_at=now_iso,
    )
