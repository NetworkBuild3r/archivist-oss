"""Single-file local vector index protocol and backends (INIT-024/SPEC-006, D3).

Defines the :class:`LocalVectorIndex` protocol with two drop-in backends:

1. :class:`SqliteVecIndex` (default): single-file ACID ``sqlite-vec`` virtual
   table (``int8`` or ``bit`` coarse scan partitioned by ``scope_id``) paired
   with exact ``float32`` cosine rescoring inside the SAME SQLite transaction
   as ``memory_chunks`` and FTS5 — 100% atomic deletes, zero zombie vectors.
2. :class:`UsearchIndex` (optional HNSW escape hatch): ``usearch`` HNSW index
   behind the exact same protocol for single-scope corpora >50K–250K chunks
   where sub-millisecond ANN p50 is preferred over single-file linear scan.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import sqlite_vec  # type: ignore[import-untyped]

logger = logging.getLogger("archivist.local_vector")


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the ``sqlite-vec`` extension safely and immediately re-lock extension loading."""
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def _normalize_f32(vec: Sequence[float], expected_dim: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ValueError(f"Expected vector of dim {expected_dim}, got {arr.shape[0]}")
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return arr


def _quantize_int8(unit_vec: np.ndarray) -> bytes:
    return np.clip(np.round(unit_vec * 127.0), -127, 127).astype(np.int8).tobytes()


def _quantize_bit(unit_vec: np.ndarray) -> bytes:
    return np.packbits((unit_vec > 0).astype(np.uint8), bitorder="little").tobytes()


@runtime_checkable
class LocalVectorIndex(Protocol):
    """Protocol for local vector backends embedded in or paired with SQLite."""

    dim: int
    backend_name: str

    def init_schema(self, conn: sqlite3.Connection) -> None:
        """Create required SQLite tables/virtual tables idempotently."""
        ...

    def upsert(
        self,
        conn: sqlite3.Connection,
        chunk_id: str,
        vec: Sequence[float],
        *,
        scope_id: str = "user:global",
    ) -> None:
        """Insert or replace *chunk_id*'s vector within the caller's SQLite transaction."""
        ...

    def delete(self, conn: sqlite3.Connection, chunk_id: str) -> None:
        """Delete *chunk_id* atomically within the caller's SQLite transaction."""
        ...

    def search(
        self,
        conn: sqlite3.Connection,
        query_vec: Sequence[float],
        *,
        scope_id: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Return up to *limit* ``(chunk_id, cosine_similarity)`` pairs for *scope_id*."""
        ...

    def on_commit(self) -> None:
        """Apply any staged in-memory index mutations after SQLite commits."""
        ...

    def on_rollback(self) -> None:
        """Discard staged in-memory index mutations when SQLite rolls back."""
        ...


class SqliteVecIndex:
    """Single-file ACID ``sqlite-vec`` backend with ``float32`` exact rescoring.

    All writes execute on the caller-supplied ``sqlite3.Connection`` without
    calling ``conn.commit()``, guaranteeing that chunk rows, FTS5 entries, and
    vectors commit or roll back together in one atomic SQLite transaction.
    """

    def __init__(
        self,
        dim: int,
        *,
        quantization: str = "int8",
        coarse_multiplier: int = 4,
        min_coarse_k: int = 64,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        norm_q = quantization.strip().lower()
        if norm_q not in ("int8", "bit"):
            raise ValueError(f"quantization must be 'int8' or 'bit', got {quantization!r}")
        if norm_q == "bit" and dim % 8 != 0:
            raise ValueError(f"bit quantization requires dim divisible by 8, got {dim}")
        self.dim = int(dim)
        self.quantization = norm_q
        self.coarse_multiplier = max(1, int(coarse_multiplier))
        self.min_coarse_k = max(1, int(min_coarse_k))
        self.backend_name = f"sqlite-vec:{norm_q}"

    def init_schema(self, conn: sqlite3.Connection) -> None:
        """Initialize ``vec_id_map``, ``chunk_vectors_f32``, and ``chunk_vectors_coarse``."""
        load_sqlite_vec(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vec_id_map (
                vec_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id  TEXT UNIQUE NOT NULL,
                scope_id  TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vec_id_map_scope ON vec_id_map (scope_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_vectors_f32 (
                vec_rowid INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
            """
        )
        if self.quantization == "bit":
            vec_col = f"embedding bit[{self.dim}]"
        else:
            vec_col = f"embedding int8[{self.dim}] distance_metric=cosine"
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors_coarse USING vec0(
                vec_rowid INTEGER PRIMARY KEY,
                scope_id TEXT partition key,
                {vec_col}
            )
            """
        )

    def upsert(
        self,
        conn: sqlite3.Connection,
        chunk_id: str,
        vec: Sequence[float],
        *,
        scope_id: str = "user:global",
    ) -> None:
        unit = _normalize_f32(vec, self.dim)
        f32_bytes = unit.tobytes()

        existing = conn.execute(
            "SELECT vec_rowid FROM vec_id_map WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if existing is not None:
            vec_rowid = int(existing[0])
            conn.execute(
                "DELETE FROM chunk_vectors_coarse WHERE vec_rowid = ?",
                (vec_rowid,),
            )
            conn.execute(
                "UPDATE vec_id_map SET scope_id = ? WHERE vec_rowid = ?",
                (scope_id, vec_rowid),
            )
        else:
            cur = conn.execute(
                "INSERT INTO vec_id_map (chunk_id, scope_id) VALUES (?, ?)",
                (chunk_id, scope_id),
            )
            assert cur.lastrowid is not None
            vec_rowid = int(cur.lastrowid)

        conn.execute(
            "INSERT OR REPLACE INTO chunk_vectors_f32 (vec_rowid, embedding) VALUES (?, ?)",
            (vec_rowid, f32_bytes),
        )
        if self.quantization == "bit":
            conn.execute(
                "INSERT INTO chunk_vectors_coarse (vec_rowid, scope_id, embedding) "
                "VALUES (?, ?, vec_bit(?))",
                (vec_rowid, scope_id, _quantize_bit(unit)),
            )
        else:
            conn.execute(
                "INSERT INTO chunk_vectors_coarse (vec_rowid, scope_id, embedding) "
                "VALUES (?, ?, vec_int8(?))",
                (vec_rowid, scope_id, _quantize_int8(unit)),
            )

    def delete(self, conn: sqlite3.Connection, chunk_id: str) -> None:
        row = conn.execute(
            "SELECT vec_rowid FROM vec_id_map WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return
        vec_rowid = int(row[0])
        conn.execute("DELETE FROM chunk_vectors_coarse WHERE vec_rowid = ?", (vec_rowid,))
        conn.execute("DELETE FROM chunk_vectors_f32 WHERE vec_rowid = ?", (vec_rowid,))
        conn.execute("DELETE FROM vec_id_map WHERE vec_rowid = ?", (vec_rowid,))

    def search(
        self,
        conn: sqlite3.Connection,
        query_vec: Sequence[float],
        *,
        scope_id: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        if limit <= 0:
            return []
        unit_q = _normalize_f32(query_vec, self.dim)
        k_coarse = max(limit * self.coarse_multiplier, self.min_coarse_k)

        if self.quantization == "bit":
            q_param = _quantize_bit(unit_q)
            match_expr = "vec_bit(?)"
        else:
            q_param = _quantize_int8(unit_q)
            match_expr = "vec_int8(?)"

        if scope_id == "*":
            coarse_rows = conn.execute(
                f"SELECT vec_rowid, distance FROM chunk_vectors_coarse "
                f"WHERE embedding MATCH {match_expr} AND k = ? ORDER BY distance",
                (q_param, k_coarse),
            ).fetchall()
        else:
            coarse_rows = conn.execute(
                f"SELECT vec_rowid, distance FROM chunk_vectors_coarse "
                f"WHERE embedding MATCH {match_expr} AND scope_id = ? AND k = ? "
                f"ORDER BY distance",
                (q_param, scope_id, k_coarse),
            ).fetchall()

        if not coarse_rows:
            return []

        rowids = [int(r[0]) for r in coarse_rows]
        placeholders = ",".join("?" for _ in rowids)
        f32_rows = conn.execute(
            f"SELECT m.chunk_id, v.embedding "
            f"FROM chunk_vectors_f32 v "
            f"JOIN vec_id_map m ON m.vec_rowid = v.vec_rowid "
            f"WHERE v.vec_rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        if not f32_rows:
            return []

        chunk_ids = [str(r[0]) for r in f32_rows]
        mat = np.frombuffer(b"".join(bytes(r[1]) for r in f32_rows), dtype=np.float32).reshape(
            len(f32_rows), self.dim
        )
        scores = mat @ unit_q
        top_order = np.argsort(-scores)[:limit]
        return [(chunk_ids[int(i)], float(scores[int(i)])) for i in top_order]

    def on_commit(self) -> None:
        return None

    def on_rollback(self) -> None:
        return None


class UsearchIndex:
    """Optional ``USearch`` HNSW adapter behind :class:`LocalVectorIndex` (D3 escape hatch).

    Maintains an HNSW graph in memory (optionally persisted to ``index_path``)
    while storing authoritative ID/scope mappings and float32 vectors in SQLite:
    - Mutations stage until :meth:`on_commit` so SQLite rollbacks never corrupt HNSW.
    - Every startup reconciles HNSW keys against SQLite ``vec_id_map`` (zero crash drift).
    - :meth:`save` writes atomically via a ``0600`` temporary file + ``fsync`` + ``os.replace``.
    - Scoped searches avoid overfetch starvation by directly scanning small scopes and
      expanding overfetch iteratively for larger scopes.
    """

    def __init__(
        self,
        dim: int,
        *,
        index_path: Path | None = None,
        dtype: str = "i8",
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        from usearch.index import Index  # type: ignore[import-untyped]

        self.dim = int(dim)
        self.index_path = index_path
        self.backend_name = f"usearch:{dtype}"
        self._lock = threading.RLock()
        self._pending_upserts: dict[int, np.ndarray] = {}
        self._pending_deletes: set[int] = set()
        self._index: Any = Index(
            ndim=self.dim,
            metric="cos",
            dtype=dtype,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )
        if index_path is not None and index_path.exists():
            self._index.load(str(index_path))

    def init_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vec_id_map (
                vec_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id  TEXT UNIQUE NOT NULL,
                scope_id  TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vec_id_map_scope ON vec_id_map (scope_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_vectors_f32 (
                vec_rowid INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
            """
        )
        # Always reconcile HNSW keys against SQLite source-of-truth on startup
        # so crash/restart never drops committed vectors or leaves deleted ghosts.
        rows = conn.execute(
            "SELECT m.vec_rowid, v.embedding "
            "FROM vec_id_map m "
            "JOIN chunk_vectors_f32 v ON v.vec_rowid = m.vec_rowid"
        ).fetchall()
        sqlite_map: dict[int, bytes] = {int(r[0]): bytes(r[1]) for r in rows}

        with self._lock:
            existing_keys: set[int] = (
                {int(k) for k in self._index.keys} if len(self._index) > 0 else set()
            )
            stale_keys = existing_keys - sqlite_map.keys()
            for k in stale_keys:
                if self._index.contains(k):
                    self._index.remove(k)

            missing_keys = sorted(sqlite_map.keys() - existing_keys)
            if missing_keys:
                keys_arr = np.array(missing_keys, dtype=np.uint64)
                mat = np.frombuffer(
                    b"".join(sqlite_map[k] for k in missing_keys),
                    dtype=np.float32,
                ).reshape(len(missing_keys), self.dim)
                self._index.add(keys_arr, mat)

            self._pending_upserts.clear()
            self._pending_deletes.clear()

    def save(self) -> None:
        """Atomically persist the HNSW sidecar file with 0600 permissions (POSIX)."""
        if self.index_path is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".usearch-",
            suffix=".tmp",
            dir=str(self.index_path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            if os.name == "posix":
                os.fchmod(fd, 0o600)
            os.close(fd)
            with self._lock:
                self._index.save(str(tmp_path))
            if os.name == "posix":
                os.chmod(tmp_path, 0o600)
            with tmp_path.open("rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.index_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def upsert(
        self,
        conn: sqlite3.Connection,
        chunk_id: str,
        vec: Sequence[float],
        *,
        scope_id: str = "user:global",
    ) -> None:
        unit = _normalize_f32(vec, self.dim)
        existing = conn.execute(
            "SELECT vec_rowid FROM vec_id_map WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if existing is not None:
            old_rowid = int(existing[0])
            conn.execute("DELETE FROM chunk_vectors_f32 WHERE vec_rowid = ?", (old_rowid,))
            conn.execute("DELETE FROM vec_id_map WHERE vec_rowid = ?", (old_rowid,))
            self._pending_upserts.pop(old_rowid, None)
            self._pending_deletes.add(old_rowid)

        # Always allocate a fresh AUTOINCREMENT vec_rowid so crash reconciliation
        # detects updated embeddings via key set diff without false matches.
        cur = conn.execute(
            "INSERT INTO vec_id_map (chunk_id, scope_id) VALUES (?, ?)",
            (chunk_id, scope_id),
        )
        assert cur.lastrowid is not None
        vec_rowid = int(cur.lastrowid)

        conn.execute(
            "INSERT INTO chunk_vectors_f32 (vec_rowid, embedding) VALUES (?, ?)",
            (vec_rowid, unit.tobytes()),
        )
        self._pending_deletes.discard(vec_rowid)
        self._pending_upserts[vec_rowid] = unit

    def delete(self, conn: sqlite3.Connection, chunk_id: str) -> None:
        row = conn.execute(
            "SELECT vec_rowid FROM vec_id_map WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return
        vec_rowid = int(row[0])
        conn.execute("DELETE FROM chunk_vectors_f32 WHERE vec_rowid = ?", (vec_rowid,))
        conn.execute("DELETE FROM vec_id_map WHERE vec_rowid = ?", (vec_rowid,))
        self._pending_upserts.pop(vec_rowid, None)
        self._pending_deletes.add(vec_rowid)

    def on_commit(self) -> None:
        """Apply staged HNSW mutations strictly AFTER SQLite commits."""
        with self._lock:
            for rowid in self._pending_deletes:
                if self._index.contains(rowid):
                    self._index.remove(rowid)
            for rowid, unit in self._pending_upserts.items():
                if self._index.contains(rowid):
                    self._index.remove(rowid)
                self._index.add(rowid, unit)
            self._pending_deletes.clear()
            self._pending_upserts.clear()

    def on_rollback(self) -> None:
        """Discard staged HNSW mutations when SQLite rolls back."""
        with self._lock:
            self._pending_deletes.clear()
            self._pending_upserts.clear()

    def _exact_scope_scan(
        self,
        conn: sqlite3.Connection,
        unit_q: np.ndarray,
        *,
        scope_id: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        rows = conn.execute(
            "SELECT m.chunk_id, v.embedding "
            "FROM vec_id_map m "
            "JOIN chunk_vectors_f32 v ON v.vec_rowid = m.vec_rowid "
            "WHERE m.scope_id = ?",
            (scope_id,),
        ).fetchall()
        if not rows:
            return []
        chunk_ids = [str(r[0]) for r in rows]
        mat = np.frombuffer(b"".join(bytes(r[1]) for r in rows), dtype=np.float32).reshape(
            len(rows), self.dim
        )
        scores = mat @ unit_q
        top_order = np.argsort(-scores)[:limit]
        return [(chunk_ids[int(i)], float(scores[int(i)])) for i in top_order]

    def search(
        self,
        conn: sqlite3.Connection,
        query_vec: Sequence[float],
        *,
        scope_id: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        if limit <= 0:
            return []
        unit_q = _normalize_f32(query_vec, self.dim)

        with self._lock:
            total_indexed = len(self._index)
        if total_indexed == 0:
            return []

        if scope_id != "*":
            scope_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM vec_id_map WHERE scope_id = ?",
                    (scope_id,),
                ).fetchone()[0]
            )
            if scope_count == 0:
                return []
            # For small scopes inside a large multi-scope index, exact float32 scan is
            # sub-millisecond and eliminates HNSW post-filter starvation completely.
            if scope_count <= max(limit * 16, 512):
                return self._exact_scope_scan(conn, unit_q, scope_id=scope_id, limit=limit)

        rows: list[Any] = []
        overfetch = min(total_indexed, max(limit * 4, 64))
        while True:
            with self._lock:
                matches = self._index.search(unit_q, overfetch)
            keys = [int(k) for k in matches.keys]
            if not keys:
                return []

            placeholders = ",".join("?" for _ in keys)
            if scope_id == "*":
                rows = conn.execute(
                    f"SELECT m.vec_rowid, m.chunk_id, v.embedding "
                    f"FROM vec_id_map m "
                    f"JOIN chunk_vectors_f32 v ON v.vec_rowid = m.vec_rowid "
                    f"WHERE m.vec_rowid IN ({placeholders})",
                    keys,
                ).fetchall()
                break

            rows = conn.execute(
                f"SELECT m.vec_rowid, m.chunk_id, v.embedding "
                f"FROM vec_id_map m "
                f"JOIN chunk_vectors_f32 v ON v.vec_rowid = m.vec_rowid "
                f"WHERE m.scope_id = ? AND m.vec_rowid IN ({placeholders})",
                [scope_id, *keys],
            ).fetchall()
            if len(rows) >= limit or overfetch >= total_indexed:
                break
            overfetch = min(total_indexed, overfetch * 4)

        if not rows:
            return []

        chunk_ids = [str(r[1]) for r in rows]
        mat = np.frombuffer(b"".join(bytes(r[2]) for r in rows), dtype=np.float32).reshape(
            len(rows), self.dim
        )
        scores = mat @ unit_q
        top_order = np.argsort(-scores)[:limit]
        return [(chunk_ids[int(i)], float(scores[int(i)])) for i in top_order]


def create_local_vector_index(
    dim: int,
    *,
    backend: str = "sqlite-vec",
    index_path: Path | None = None,
    quantization: str = "int8",
) -> LocalVectorIndex:
    """Instantiate a :class:`LocalVectorIndex` backend (``sqlite-vec`` default or ``usearch``)."""
    norm = (backend or "sqlite-vec").strip().lower()
    if norm in ("sqlite-vec", "sqlite_vec", "sqlite"):
        return SqliteVecIndex(dim=dim, quantization=quantization)
    if norm == "usearch":
        return UsearchIndex(dim=dim, index_path=index_path)
    raise ValueError(
        f"Unknown local vector backend {backend!r}; expected 'sqlite-vec' or 'usearch'"
    )
