"""Memory-as-Product: scope snapshot, fork, export, and import.

Service-layer operations over a namespace (optionally agent-filtered) memory set:

* **Snapshot** — versioned archive of ``memory_chunks`` (+ optional vectors)
* **Fork** — derive a new target scope from a snapshot with lineage pointer
* **Export** — auditable bundle under ``BACKUP_DIR`` with a counts/versions manifest
* **Import** — restore an archive under ``BACKUP_DIR`` into a target scope (INIT-011/SPEC-002)

Path containment reuses ``SnapshotPathError`` / ``_snapshot_dir`` from
``backup_manager`` (PR #39). Vector mutations go through ``MemoryTransaction``
+ outbox when vectors are present. Callers must pass ``caller_agent_id`` so
namespace RBAC is enforced at this layer.

Provenance: INIT-001/SPEC-009; INIT-011/SPEC-002 (import + caps).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from archivist.storage.backup_manager import SnapshotPathError, _snapshot_dir, delete_snapshot

logger = logging.getLogger("archivist.memory_product")

MANIFEST_KIND = "memory_scope"
MANIFEST_VERSION = 1

# ADR-011 import blast-radius caps (INIT-011/SPEC-002).
MAP_IMPORT_MAX_CHUNKS = 50_000
MAP_IMPORT_MAX_BYTES = 256 * 1024 * 1024  # 256 MiB

# Keys that must never appear in exported manifests (defense in depth).
_SECRET_MANIFEST_KEYS = frozenset(
    {
        "api_key",
        "api_keys",
        "authorization",
        "password",
        "secret",
        "token",
        "archivist_api_key",
        "openai_api_key",
        "qdrant_api_key",
    }
)


class MemoryProductError(Exception):
    """Base error for Memory-as-Product operations."""


class MemoryProductAuthzError(PermissionError, MemoryProductError):
    """Caller lacks RBAC permission for the requested namespace action."""


class MemoryProductNotFoundError(LookupError, MemoryProductError):
    """Requested scope version or archive is missing."""


class MemoryProductConflictError(MemoryProductError):
    """Fork/export cannot proceed (e.g. incomplete prior fork cleanup)."""


@dataclass(frozen=True)
class ScopeVersionRecord:
    """One row from ``memory_scope_versions``."""

    id: str
    source_namespace: str
    source_agent_id: str
    version: int
    label: str
    parent_version_id: str | None
    chunk_count: int
    point_count: int
    archive_id: str
    operation: str
    created_by: str
    created_at: str
    lineage: dict[str, Any]


def _require_rbac(caller_agent_id: str, action: str, namespace: str) -> None:
    from archivist.core.rbac import check_access

    if not caller_agent_id:
        raise MemoryProductAuthzError("caller_agent_id is required for RBAC")
    if not namespace:
        raise ValueError("namespace is required")
    policy = check_access(caller_agent_id, action, namespace)
    if not policy.allowed:
        raise MemoryProductAuthzError(policy.reason or "access denied")


def _safe_label(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (label or ""))[:48]


def _new_archive_id(label: str = "") -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = _safe_label(label)
    if suffix:
        return f"memprod_{ts}_{suffix}"
    return f"memprod_{ts}_{uuid.uuid4().hex[:10]}"


def _strip_secrets(obj: Any) -> Any:
    """Recursively drop secret-looking keys from manifest-bound structures."""
    if isinstance(obj, dict):
        return {
            k: _strip_secrets(v) for k, v in obj.items() if k.lower() not in _SECRET_MANIFEST_KEYS
        }
    if isinstance(obj, list):
        return [_strip_secrets(v) for v in obj]
    return obj


def _row_to_record(row: Any) -> ScopeVersionRecord:
    data = dict(row)
    raw_lineage = data.get("lineage_json") or "{}"
    try:
        lineage = json.loads(raw_lineage)
        if not isinstance(lineage, dict):
            lineage = {}
    except json.JSONDecodeError:
        lineage = {}
    return ScopeVersionRecord(
        id=data["id"],
        source_namespace=data["source_namespace"],
        source_agent_id=data.get("source_agent_id") or "",
        version=int(data["version"]),
        label=data.get("label") or "",
        parent_version_id=data.get("parent_version_id"),
        chunk_count=int(data.get("chunk_count") or 0),
        point_count=int(data.get("point_count") or 0),
        archive_id=data.get("archive_id") or "",
        operation=data["operation"],
        created_by=data.get("created_by") or "",
        created_at=data["created_at"],
        lineage=lineage,
    )


async def _next_scope_version(namespace: str, agent_id: str = "") -> int:
    from archivist.storage.versioning import next_scope_version

    return await next_scope_version(namespace, agent_id)


async def _insert_scope_version(
    *,
    version_id: str,
    namespace: str,
    agent_id: str,
    version: int,
    label: str,
    parent_version_id: str | None,
    chunk_count: int,
    point_count: int,
    archive_id: str,
    operation: str,
    created_by: str,
    created_at: str,
    lineage: dict[str, Any],
) -> ScopeVersionRecord:
    from archivist.storage.versioning import insert_scope_version_row

    await insert_scope_version_row(
        version_id=version_id,
        namespace=namespace,
        agent_id=agent_id,
        version=version,
        label=label,
        parent_version_id=parent_version_id,
        chunk_count=chunk_count,
        point_count=point_count,
        archive_id=archive_id,
        operation=operation,
        created_by=created_by,
        created_at=created_at,
        lineage=lineage,
    )
    return ScopeVersionRecord(
        id=version_id,
        source_namespace=namespace,
        source_agent_id=agent_id,
        version=version,
        label=label,
        parent_version_id=parent_version_id,
        chunk_count=chunk_count,
        point_count=point_count,
        archive_id=archive_id,
        operation=operation,
        created_by=created_by,
        created_at=created_at,
        lineage=lineage,
    )


async def get_scope_version(version_id: str) -> ScopeVersionRecord | None:
    """Fetch a scope version by id (no RBAC — callers must authorize)."""
    from archivist.storage.sqlite_pool import pool

    if not version_id:
        raise ValueError("version_id is required")
    async with pool.read() as conn:
        cur = await conn.execute(
            """
            SELECT id, source_namespace, source_agent_id, version, label,
                   parent_version_id, chunk_count, point_count, archive_id,
                   operation, created_by, created_at, lineage_json
            FROM memory_scope_versions
            WHERE id = ?
            """,
            (version_id,),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    return _row_to_record(row)


async def list_scope_versions(
    *,
    namespace: str,
    caller_agent_id: str,
    agent_id: str = "",
    limit: int = 50,
) -> list[ScopeVersionRecord]:
    """List scope versions for a namespace (newest version first). Requires read."""
    from archivist.storage.sqlite_pool import pool

    _require_rbac(caller_agent_id, "read", namespace)
    if limit < 1:
        raise ValueError("limit must be >= 1")

    async with pool.read() as conn:
        if agent_id:
            cur = await conn.execute(
                """
                SELECT id, source_namespace, source_agent_id, version, label,
                       parent_version_id, chunk_count, point_count, archive_id,
                       operation, created_by, created_at, lineage_json
                FROM memory_scope_versions
                WHERE source_namespace = ? AND source_agent_id = ?
                ORDER BY version DESC
                LIMIT ?
                """,
                (namespace, agent_id, limit),
            )
        else:
            cur = await conn.execute(
                """
                SELECT id, source_namespace, source_agent_id, version, label,
                       parent_version_id, chunk_count, point_count, archive_id,
                       operation, created_by, created_at, lineage_json
                FROM memory_scope_versions
                WHERE source_namespace = ?
                ORDER BY version DESC
                LIMIT ?
                """,
                (namespace, limit),
            )
        rows = await cur.fetchall()
    return [_row_to_record(r) for r in rows]


async def _load_scope_chunks(
    *,
    namespace: str,
    agent_id: str = "",
) -> list[dict[str, Any]]:
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        if agent_id:
            cur = await conn.execute(
                """
                SELECT qdrant_id, text, file_path, chunk_index, agent_id, namespace,
                       date, memory_type, is_excluded, actor_id, actor_type,
                       importance, tier_label, ttl_at, decay_rate
                FROM memory_chunks
                WHERE namespace = ? AND agent_id = ? AND is_excluded = 0
                ORDER BY file_path, chunk_index
                """,
                (namespace, agent_id),
            )
        else:
            cur = await conn.execute(
                """
                SELECT qdrant_id, text, file_path, chunk_index, agent_id, namespace,
                       date, memory_type, is_excluded, actor_id, actor_type,
                       importance, tier_label, ttl_at, decay_rate
                FROM memory_chunks
                WHERE namespace = ? AND is_excluded = 0
                ORDER BY file_path, chunk_index
                """,
                (namespace,),
            )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


def _write_archive(
    archive_id: str,
    *,
    chunks: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> Path:
    """Write chunks/vectors/manifest under BACKUP_DIR via ``_snapshot_dir``.

    Raises:
        SnapshotPathError: if archive_id escapes BACKUP_DIR.
    """
    snap_dir = _snapshot_dir(archive_id)
    os.makedirs(snap_dir, exist_ok=True)
    try:
        with open(snap_dir / "chunks.ndjson", "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False, default=str) + "\n")
        with open(snap_dir / "vectors.ndjson", "w", encoding="utf-8") as f:
            for vec in vectors:
                f.write(json.dumps(vec, ensure_ascii=False, default=str) + "\n")
        safe_manifest = _strip_secrets(manifest)
        with open(snap_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(safe_manifest, f, indent=2, ensure_ascii=False)
    except Exception:
        # Avoid leaving a partial archive directory on write failure.
        if snap_dir.is_dir():
            delete_snapshot(archive_id)
        raise
    return snap_dir


def _read_archive(
    archive_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    snap_dir = _snapshot_dir(archive_id)
    manifest_path = snap_dir / "manifest.json"
    if not manifest_path.is_file():
        raise MemoryProductNotFoundError(f"Archive '{archive_id}' not found or missing manifest")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    chunks: list[dict[str, Any]] = []
    chunks_path = snap_dir / "chunks.ndjson"
    if chunks_path.is_file():
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    vectors: list[dict[str, Any]] = []
    vectors_path = snap_dir / "vectors.ndjson"
    if vectors_path.is_file():
        with open(vectors_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    vectors.append(json.loads(line))
    return manifest, chunks, vectors


def _archive_payload_bytes(archive_id: str) -> int:
    """Sum on-disk size of archive files under BACKUP_DIR (fail closed on escape)."""
    snap_dir = _snapshot_dir(archive_id)
    if not snap_dir.is_dir():
        raise MemoryProductNotFoundError(f"Archive '{archive_id}' not found")
    total = 0
    for path in snap_dir.iterdir():
        if path.is_file():
            total += path.stat().st_size
    return total


def _validate_import_manifest(manifest: dict[str, Any]) -> None:
    kind = str(manifest.get("kind") or "")
    if kind != MANIFEST_KIND:
        raise MemoryProductConflictError(
            f"unsupported archive kind '{kind}' (expected '{MANIFEST_KIND}')"
        )
    try:
        version = int(manifest.get("manifest_version") or 0)
    except (TypeError, ValueError) as exc:
        raise MemoryProductConflictError("invalid manifest_version") from exc
    if version < 1 or version > MANIFEST_VERSION:
        raise MemoryProductConflictError(
            f"unsupported manifest_version {version} (max {MANIFEST_VERSION})"
        )


async def _target_scope_chunk_count(namespace: str, agent_id: str = "") -> int:
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        if agent_id:
            cur = await conn.execute(
                """
                SELECT COUNT(*) AS c FROM memory_chunks
                WHERE namespace = ? AND agent_id = ? AND is_excluded = 0
                """,
                (namespace, agent_id),
            )
        else:
            cur = await conn.execute(
                """
                SELECT COUNT(*) AS c FROM memory_chunks
                WHERE namespace = ? AND is_excluded = 0
                """,
                (namespace,),
            )
        row = await cur.fetchone()
    return int(row["c"] if row is not None else 0)


def _build_manifest(
    *,
    archive_id: str,
    version_id: str,
    namespace: str,
    agent_id: str,
    version: int,
    label: str,
    operation: str,
    chunk_count: int,
    point_count: int,
    parent_version_id: str | None,
    created_by: str,
    created_at: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "kind": MANIFEST_KIND,
        "archive_id": archive_id,
        "version_id": version_id,
        "namespace": namespace,
        "agent_id": agent_id,
        "version": version,
        "label": label,
        "operation": operation,
        "chunk_count": chunk_count,
        "point_count": point_count,
        "parent_version_id": parent_version_id,
        "created_by": created_by,
        "created_at": created_at,
        "has_vectors": point_count > 0,
    }
    if extra:
        manifest.update(extra)
    return _strip_secrets(manifest)


async def create_scope_snapshot(
    *,
    namespace: str,
    caller_agent_id: str,
    agent_id: str = "",
    label: str = "",
    parent_version_id: str | None = None,
    vectors: list[dict[str, Any]] | None = None,
) -> ScopeVersionRecord:
    """Create a versioned snapshot of a scoped memory set (ac-1).

    ``vectors`` is optional — unit tests pass an empty/mock list; production
    callers may supply scrolled Qdrant points ``{id, vector, payload}``.
    Does not call live Qdrant itself.
    """
    _require_rbac(caller_agent_id, "read", namespace)

    if parent_version_id:
        parent = await get_scope_version(parent_version_id)
        if parent is None:
            raise MemoryProductNotFoundError(f"parent version '{parent_version_id}' not found")

    chunks = await _load_scope_chunks(namespace=namespace, agent_id=agent_id)
    vecs = list(vectors or [])
    version = await _next_scope_version(namespace, agent_id)
    version_id = str(uuid.uuid4())
    archive_id = _new_archive_id(label or "snapshot")
    created_at = datetime.now(UTC).isoformat()
    lineage = {
        "parent_version_id": parent_version_id,
        "source_namespace": namespace,
        "source_agent_id": agent_id,
    }
    manifest = _build_manifest(
        archive_id=archive_id,
        version_id=version_id,
        namespace=namespace,
        agent_id=agent_id,
        version=version,
        label=label,
        operation="snapshot",
        chunk_count=len(chunks),
        point_count=len(vecs),
        parent_version_id=parent_version_id,
        created_by=caller_agent_id,
        created_at=created_at,
    )

    try:
        snap_dir = _write_archive(archive_id, chunks=chunks, vectors=vecs, manifest=manifest)
    except SnapshotPathError:
        raise
    except Exception as exc:
        raise MemoryProductError(f"failed to write snapshot archive: {exc}") from exc

    try:
        record = await _insert_scope_version(
            version_id=version_id,
            namespace=namespace,
            agent_id=agent_id,
            version=version,
            label=label,
            parent_version_id=parent_version_id,
            chunk_count=len(chunks),
            point_count=len(vecs),
            archive_id=archive_id,
            operation="snapshot",
            created_by=caller_agent_id,
            created_at=created_at,
            lineage=lineage,
        )
    except Exception:
        # Transactional boundary: no orphan lineage row; clean archive.
        try:
            delete_snapshot(archive_id)
        except Exception:
            logger.warning(
                "snapshot rollback: could not delete archive_id=%s under %s",
                archive_id,
                snap_dir,
            )
        raise

    logger.info(
        "scope snapshot created version_id=%s namespace=%s version=%d chunks=%d archive_id=%s",
        version_id,
        namespace,
        version,
        len(chunks),
        archive_id,
    )
    return record


async def fork_from_snapshot(
    *,
    source_version_id: str,
    target_namespace: str,
    caller_agent_id: str,
    target_agent_id: str = "",
    label: str = "",
    vector_factory: Callable[[dict[str, Any]], list[float] | None] | None = None,
) -> ScopeVersionRecord:
    """Fork a snapshot into a new target scope with lineage pointer (ac-2).

    Copies chunk rows into ``target_namespace`` (rewriting qdrant_ids) inside a
    single ``MemoryTransaction``. When vectors are available in the archive (or
    via ``vector_factory``), enqueues Qdrant upserts on the outbox. On any
    failure before commit, SQLite + outbox roll back together — no partial fork.
    """
    source = await get_scope_version(source_version_id)
    if source is None:
        raise MemoryProductNotFoundError(f"source version '{source_version_id}' not found")

    _require_rbac(caller_agent_id, "read", source.source_namespace)
    _require_rbac(caller_agent_id, "write", target_namespace)

    if not source.archive_id:
        raise MemoryProductConflictError("source version has no archive_id")

    try:
        _manifest, chunks, vectors = _read_archive(source.archive_id)
    except SnapshotPathError:
        raise
    except MemoryProductNotFoundError:
        raise
    except Exception as exc:
        raise MemoryProductError(f"failed to read source archive: {exc}") from exc

    vectors_by_id = {str(v.get("id", "")): v for v in vectors if v.get("id")}
    created_at = datetime.now(UTC).isoformat()
    version = await _next_scope_version(target_namespace, target_agent_id)
    version_id = str(uuid.uuid4())
    archive_id = _new_archive_id(label or "fork")
    forked_chunks: list[dict[str, Any]] = []
    forked_vectors: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}

    from archivist.storage.collection_router import collection_for
    from archivist.storage.transaction import MemoryTransaction

    collection = collection_for(target_namespace)

    try:
        async with MemoryTransaction() as txn:
            for chunk in chunks:
                old_id = str(chunk.get("qdrant_id") or "")
                new_id = str(uuid.uuid4())
                if old_id:
                    id_map[old_id] = new_id
                dest_agent = target_agent_id or str(chunk.get("agent_id") or "")
                new_chunk = {
                    **chunk,
                    "qdrant_id": new_id,
                    "namespace": target_namespace,
                    "agent_id": dest_agent,
                }
                forked_chunks.append(new_chunk)
                await txn.upsert_fts_chunk(
                    qdrant_id=new_id,
                    text=str(new_chunk.get("text") or ""),
                    file_path=str(new_chunk.get("file_path") or ""),
                    chunk_index=int(new_chunk.get("chunk_index") or 0),
                    agent_id=dest_agent,
                    namespace=target_namespace,
                    date=str(new_chunk.get("date") or ""),
                    memory_type=str(new_chunk.get("memory_type") or "general"),
                    actor_id=str(new_chunk.get("actor_id") or ""),
                    actor_type=str(new_chunk.get("actor_type") or ""),
                    importance=float(new_chunk.get("importance") or 0.5),
                    tier_label=str(new_chunk.get("tier_label") or "l2"),
                )

                vector: list[float] | None = None
                src_vec = vectors_by_id.get(old_id)
                if src_vec and isinstance(src_vec.get("vector"), list):
                    vector = list(src_vec["vector"])
                elif vector_factory is not None:
                    vector = vector_factory(new_chunk)

                if vector is not None:
                    payload = {
                        "text": new_chunk.get("text") or "",
                        "file_path": new_chunk.get("file_path") or "",
                        "chunk_index": int(new_chunk.get("chunk_index") or 0),
                        "agent_id": dest_agent,
                        "namespace": target_namespace,
                        "memory_type": new_chunk.get("memory_type") or "general",
                        "forked_from_version": source_version_id,
                        "forked_from_qdrant_id": old_id,
                    }
                    point = {
                        "id": new_id,
                        "vector": vector,
                        "payload": payload,
                    }
                    forked_vectors.append(point)
                    # PointStruct-like dict; outbox serialises via model_dump/dict or as-is.
                    txn.enqueue_qdrant_upsert(
                        collection,
                        [point],
                        memory_id=new_id,
                    )

            # Record lineage inside the same transaction so fork metadata is atomic
            # with chunk writes.
            lineage = {
                "parent_version_id": source_version_id,
                "source_namespace": source.source_namespace,
                "source_agent_id": source.source_agent_id,
                "target_namespace": target_namespace,
                "target_agent_id": target_agent_id,
                "id_map_count": len(id_map),
            }
            await txn.execute(
                """
                INSERT INTO memory_scope_versions (
                    id, source_namespace, source_agent_id, version, label,
                    parent_version_id, chunk_count, point_count, archive_id,
                    operation, created_by, created_at, lineage_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    target_namespace,
                    target_agent_id,
                    version,
                    label,
                    source_version_id,
                    len(forked_chunks),
                    len(forked_vectors),
                    archive_id,
                    "fork",
                    caller_agent_id,
                    created_at,
                    json.dumps(lineage, separators=(",", ":")),
                ),
            )
    except MemoryProductAuthzError:
        raise
    except Exception as exc:
        # MemoryTransaction rolls back SQLite + outbox; no partial fork remains.
        raise MemoryProductError(f"fork failed (rolled back): {exc}") from exc

    # Persist a fork archive for audit/export after the transactional write succeeds.
    manifest = _build_manifest(
        archive_id=archive_id,
        version_id=version_id,
        namespace=target_namespace,
        agent_id=target_agent_id,
        version=version,
        label=label,
        operation="fork",
        chunk_count=len(forked_chunks),
        point_count=len(forked_vectors),
        parent_version_id=source_version_id,
        created_by=caller_agent_id,
        created_at=created_at,
        extra={"source_version_id": source_version_id, "source_archive_id": source.archive_id},
    )
    try:
        _write_archive(archive_id, chunks=forked_chunks, vectors=forked_vectors, manifest=manifest)
    except Exception as exc:
        # Fork data is already committed; archive is best-effort audit. Log and continue.
        logger.warning(
            "fork archive write failed version_id=%s archive_id=%s: %s",
            version_id,
            archive_id,
            exc,
        )

    logger.info(
        "scope fork complete version_id=%s source=%s target_ns=%s chunks=%d",
        version_id,
        source_version_id,
        target_namespace,
        len(forked_chunks),
    )
    record = await get_scope_version(version_id)
    if record is None:
        raise MemoryProductError("fork committed but version row missing")
    return record


async def export_scope(
    *,
    namespace: str,
    caller_agent_id: str,
    agent_id: str = "",
    version_id: str | None = None,
    label: str = "",
    vectors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Export a scoped memory set (or existing version) to BACKUP_DIR (ac-3).

    Returns ``{path, archive_id, manifest, version_id, bytes}`` where ``path`` is
    the resolved archive directory (contained under BACKUP_DIR) and ``bytes`` is
    the UTF-8 manifest payload for callers that want in-memory materialisation.
    """
    _require_rbac(caller_agent_id, "read", namespace)

    if version_id:
        existing = await get_scope_version(version_id)
        if existing is None:
            raise MemoryProductNotFoundError(f"version '{version_id}' not found")
        if existing.source_namespace != namespace:
            raise MemoryProductAuthzError("version belongs to a different namespace than requested")
        _require_rbac(caller_agent_id, "read", existing.source_namespace)
        if not existing.archive_id:
            raise MemoryProductConflictError("version has no archive_id to export")
        snap_dir = _snapshot_dir(existing.archive_id)
        manifest_path = snap_dir / "manifest.json"
        if not manifest_path.is_file():
            raise MemoryProductNotFoundError(f"archive '{existing.archive_id}' missing manifest")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = _strip_secrets(json.load(f))
        # Re-stamp an export lineage row pointing at the source version.
        export_version = await _next_scope_version(namespace, agent_id or existing.source_agent_id)
        export_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        lineage = {
            "parent_version_id": existing.id,
            "exported_archive_id": existing.archive_id,
            "operation": "export",
        }
        await _insert_scope_version(
            version_id=export_id,
            namespace=namespace,
            agent_id=agent_id or existing.source_agent_id,
            version=export_version,
            label=label or existing.label or "export",
            parent_version_id=existing.id,
            chunk_count=existing.chunk_count,
            point_count=existing.point_count,
            archive_id=existing.archive_id,
            operation="export",
            created_by=caller_agent_id,
            created_at=created_at,
            lineage=lineage,
        )
        manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        return {
            "path": str(snap_dir),
            "archive_id": existing.archive_id,
            "version_id": export_id,
            "manifest": manifest,
            "bytes": manifest_bytes,
            "chunk_count": existing.chunk_count,
            "point_count": existing.point_count,
        }

    # Fresh snapshot then return its archive path.
    record = await create_scope_snapshot(
        namespace=namespace,
        caller_agent_id=caller_agent_id,
        agent_id=agent_id,
        label=label or "export",
        vectors=vectors,
    )
    # Mark a dedicated export row with lineage to the snapshot just created.
    export_version = await _next_scope_version(namespace, agent_id)
    export_id = str(uuid.uuid4())
    created_at = datetime.now(UTC).isoformat()
    await _insert_scope_version(
        version_id=export_id,
        namespace=namespace,
        agent_id=agent_id,
        version=export_version,
        label=label or "export",
        parent_version_id=record.id,
        chunk_count=record.chunk_count,
        point_count=record.point_count,
        archive_id=record.archive_id,
        operation="export",
        created_by=caller_agent_id,
        created_at=created_at,
        lineage={
            "parent_version_id": record.id,
            "exported_archive_id": record.archive_id,
            "operation": "export",
        },
    )
    snap_dir = _snapshot_dir(record.archive_id)
    with open(snap_dir / "manifest.json", encoding="utf-8") as f:
        manifest = _strip_secrets(json.load(f))
    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    return {
        "path": str(snap_dir),
        "archive_id": record.archive_id,
        "version_id": export_id,
        "manifest": manifest,
        "bytes": manifest_bytes,
        "chunk_count": record.chunk_count,
        "point_count": record.point_count,
    }


async def import_scope(
    *,
    archive_id: str,
    target_namespace: str,
    caller_agent_id: str,
    target_agent_id: str = "",
    label: str = "",
    vector_factory: Callable[[dict[str, Any]], list[float] | None] | None = None,
) -> ScopeVersionRecord:
    """Restore an archive under BACKUP_DIR into a target scope (INIT-011/SPEC-002).

    Fail-closed when the target agent scope already has non-excluded chunks
    (ADR-011 conflict policy). Vectors are best-effort via MemoryTransaction /
    outbox when present in the archive or supplied by ``vector_factory``.
    """
    if not archive_id:
        raise ValueError("archive_id is required")

    _require_rbac(caller_agent_id, "write", target_namespace)

    # Path confinement first — SnapshotPathError must surface for escapes.
    try:
        payload_bytes = _archive_payload_bytes(archive_id)
    except SnapshotPathError:
        raise
    except MemoryProductNotFoundError:
        raise

    if payload_bytes > MAP_IMPORT_MAX_BYTES:
        raise MemoryProductConflictError(
            f"archive exceeds max bytes ({payload_bytes} > {MAP_IMPORT_MAX_BYTES})"
        )

    try:
        manifest, chunks, vectors = _read_archive(archive_id)
    except SnapshotPathError:
        raise
    except MemoryProductNotFoundError:
        raise
    except Exception as exc:
        raise MemoryProductError(f"failed to read import archive: {exc}") from exc

    _validate_import_manifest(manifest if isinstance(manifest, dict) else {})

    source_namespace = str(manifest.get("namespace") or "").strip()
    if not source_namespace:
        raise MemoryProductConflictError("import archive missing source namespace in manifest")
    # SEC-011-01: require source read (symmetric with fork) so archive_id is not
    # an unscoped capability to materialize another namespace's memory.
    _require_rbac(caller_agent_id, "read", source_namespace)

    if len(chunks) > MAP_IMPORT_MAX_CHUNKS:
        raise MemoryProductConflictError(
            f"archive exceeds max chunks ({len(chunks)} > {MAP_IMPORT_MAX_CHUNKS})"
        )

    existing = await _target_scope_chunk_count(target_namespace, target_agent_id)
    if existing > 0:
        raise MemoryProductConflictError(
            "target scope is not empty; import refuses silent merge "
            "(fork into a fresh scope or clear the target first)"
        )

    vectors_by_id = {str(v.get("id", "")): v for v in vectors if v.get("id")}
    created_at = datetime.now(UTC).isoformat()
    version = await _next_scope_version(target_namespace, target_agent_id)
    version_id = str(uuid.uuid4())
    import_archive_id = _new_archive_id(label or "import")
    parent_version_id = str(manifest.get("version_id") or "") or None
    imported_chunks: list[dict[str, Any]] = []
    imported_vectors: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}

    from archivist.storage.collection_router import collection_for
    from archivist.storage.transaction import MemoryTransaction

    collection = collection_for(target_namespace)
    source_agent_id = str(manifest.get("agent_id") or "")

    try:
        async with MemoryTransaction() as txn:
            for chunk in chunks:
                old_id = str(chunk.get("qdrant_id") or "")
                new_id = str(uuid.uuid4())
                if old_id:
                    id_map[old_id] = new_id
                dest_agent = target_agent_id or str(chunk.get("agent_id") or "")
                new_chunk = {
                    **chunk,
                    "qdrant_id": new_id,
                    "namespace": target_namespace,
                    "agent_id": dest_agent,
                }
                imported_chunks.append(new_chunk)
                await txn.upsert_fts_chunk(
                    qdrant_id=new_id,
                    text=str(new_chunk.get("text") or ""),
                    file_path=str(new_chunk.get("file_path") or ""),
                    chunk_index=int(new_chunk.get("chunk_index") or 0),
                    agent_id=dest_agent,
                    namespace=target_namespace,
                    date=str(new_chunk.get("date") or ""),
                    memory_type=str(new_chunk.get("memory_type") or "general"),
                    actor_id=str(new_chunk.get("actor_id") or ""),
                    actor_type=str(new_chunk.get("actor_type") or ""),
                    importance=float(new_chunk.get("importance") or 0.5),
                    tier_label=str(new_chunk.get("tier_label") or "l2"),
                )

                vector: list[float] | None = None
                src_vec = vectors_by_id.get(old_id)
                if src_vec and isinstance(src_vec.get("vector"), list):
                    vector = list(src_vec["vector"])
                elif vector_factory is not None:
                    vector = vector_factory(new_chunk)

                if vector is not None:
                    payload = {
                        "text": new_chunk.get("text") or "",
                        "file_path": new_chunk.get("file_path") or "",
                        "chunk_index": int(new_chunk.get("chunk_index") or 0),
                        "agent_id": dest_agent,
                        "namespace": target_namespace,
                        "memory_type": new_chunk.get("memory_type") or "general",
                        "imported_from_archive": archive_id,
                        "imported_from_qdrant_id": old_id,
                    }
                    point = {
                        "id": new_id,
                        "vector": vector,
                        "payload": payload,
                    }
                    imported_vectors.append(point)
                    txn.enqueue_qdrant_upsert(
                        collection,
                        [point],
                        memory_id=new_id,
                    )

            lineage = _strip_secrets(
                {
                    "parent_version_id": parent_version_id,
                    "source_archive_id": archive_id,
                    "source_namespace": source_namespace,
                    "source_agent_id": source_agent_id,
                    "target_namespace": target_namespace,
                    "target_agent_id": target_agent_id,
                    "id_map_count": len(id_map),
                    "operation": "import",
                }
            )
            await txn.execute(
                """
                INSERT INTO memory_scope_versions (
                    id, source_namespace, source_agent_id, version, label,
                    parent_version_id, chunk_count, point_count, archive_id,
                    operation, created_by, created_at, lineage_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    target_namespace,
                    target_agent_id,
                    version,
                    label,
                    parent_version_id,
                    len(imported_chunks),
                    len(imported_vectors),
                    import_archive_id,
                    "import",
                    caller_agent_id,
                    created_at,
                    json.dumps(lineage, separators=(",", ":")),
                ),
            )
    except MemoryProductAuthzError:
        raise
    except MemoryProductConflictError:
        raise
    except Exception as exc:
        raise MemoryProductError(f"import failed (rolled back): {exc}") from exc

    manifest_out = _build_manifest(
        archive_id=import_archive_id,
        version_id=version_id,
        namespace=target_namespace,
        agent_id=target_agent_id,
        version=version,
        label=label,
        operation="import",
        chunk_count=len(imported_chunks),
        point_count=len(imported_vectors),
        parent_version_id=parent_version_id,
        created_by=caller_agent_id,
        created_at=created_at,
        extra={"source_archive_id": archive_id},
    )
    try:
        _write_archive(
            import_archive_id,
            chunks=imported_chunks,
            vectors=imported_vectors,
            manifest=manifest_out,
        )
    except Exception as exc:
        logger.warning(
            "import archive write failed version_id=%s archive_id=%s: %s",
            version_id,
            import_archive_id,
            exc,
        )

    logger.info(
        "scope import complete version_id=%s source_archive=%s target_ns=%s chunks=%d",
        version_id,
        archive_id,
        target_namespace,
        len(imported_chunks),
    )
    record = await get_scope_version(version_id)
    if record is None:
        raise MemoryProductError("import committed but version row missing")
    return record
