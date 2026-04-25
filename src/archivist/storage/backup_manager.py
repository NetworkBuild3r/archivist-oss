"""Backup and restore engine for Archivist memory stores.

Backs up:
  - Qdrant collections via the native snapshot REST API
  - SQLite graph.db via Python's online backup API  (GRAPH_BACKEND=sqlite)
  - PostgreSQL database via ``pg_dump`` custom format (GRAPH_BACKEND=postgres)
  - Optionally, memory source files as a tarball

Each snapshot is a timestamped directory under BACKUP_DIR containing the above
artefacts plus a manifest.json with metadata for validation during restore.

Postgres backup requirements
----------------------------
``pg_dump`` and ``pg_restore`` / ``psql`` must be installed and on ``PATH``
(they are included in the ``postgresql-client`` package on most distributions
and in the official ``postgres`` Docker image).  The ``DATABASE_URL``
environment variable is used to derive the connection parameters.
"""

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from archivist.core.config import (
    BACKUP_DIR,
    BACKUP_INCLUDE_FILES,
    BACKUP_RETENTION_COUNT,
    DATABASE_URL,
    GRAPH_BACKEND,
    MEMORY_ROOT,
    QDRANT_COLLECTION,
    QDRANT_URL,
    SQLITE_PATH,
    VECTOR_DIM,
)
from archivist.storage.collection_router import collections_for_query
from archivist.storage.sqlite_pool import _get_graph_write_lock

logger = logging.getLogger("archivist.backup")

MANIFEST_VERSION = 1
ARCHIVIST_VERSION = "2.0.1"


def _snapshot_dir(snapshot_id: str) -> Path:
    return Path(BACKUP_DIR) / snapshot_id


def _ensure_backup_dir() -> None:
    os.makedirs(BACKUP_DIR, exist_ok=True)


def _qdrant_http(method: str, path: str, **kwargs) -> httpx.Response:
    """Synchronous Qdrant REST call with generous timeout for snapshot ops."""
    url = f"{QDRANT_URL}{path}"
    with httpx.Client(timeout=300) as client:
        resp = getattr(client, method)(url, **kwargs)
        resp.raise_for_status()
        return resp


def _collection_point_count(collection_name: str) -> int:
    try:
        resp = _qdrant_http("get", f"/collections/{collection_name}")
        return resp.json().get("result", {}).get("points_count", 0)
    except Exception:
        return -1


# ── Snapshot creation ────────────────────────────────────────────────────────


def create_snapshot(label: str = "") -> dict:
    """Create a full backup snapshot. Returns summary dict with snapshot_id and paths."""
    _ensure_backup_dir()
    ts = datetime.now(UTC)
    snapshot_id = ts.strftime("%Y%m%dT%H%M%SZ")
    if label:
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        snapshot_id = f"{snapshot_id}_{safe_label}"

    snap_dir = _snapshot_dir(snapshot_id)
    os.makedirs(snap_dir, exist_ok=True)
    t0 = time.monotonic()

    errors: list[str] = []
    collection_info: dict[str, dict] = {}

    colls = collections_for_query("")
    for coll in colls:
        try:
            info = _backup_qdrant_collection(coll, snap_dir)
            collection_info[coll] = info
        except Exception as e:
            msg = f"Qdrant snapshot failed for '{coll}': {e}"
            logger.error(msg)
            errors.append(msg)

    _is_pg = (GRAPH_BACKEND or "sqlite").lower() == "postgres"

    if _is_pg:
        try:
            _backup_postgres(snap_dir)
        except Exception as e:
            msg = f"Postgres backup failed: {e}"
            logger.error(msg)
            errors.append(msg)
    else:
        try:
            _backup_sqlite(snap_dir)
        except Exception as e:
            msg = f"SQLite backup failed: {e}"
            logger.error(msg)
            errors.append(msg)

    if BACKUP_INCLUDE_FILES and os.path.isdir(MEMORY_ROOT):
        try:
            _backup_memory_files(snap_dir)
        except Exception as e:
            msg = f"Memory files backup failed: {e}"
            logger.warning(msg)
            errors.append(msg)

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "archivist_version": ARCHIVIST_VERSION,
        "snapshot_id": snapshot_id,
        "label": label,
        "created_at": ts.isoformat(),
        "vector_dim": VECTOR_DIM,
        "graph_backend": (GRAPH_BACKEND or "sqlite").lower(),
        "collections": collection_info,
        "sqlite_backed_up": os.path.isfile(snap_dir / "graph.db"),
        "postgres_backed_up": os.path.isfile(snap_dir / "graph.pgdump"),
        "files_backed_up": os.path.isfile(snap_dir / "memories.tar.gz"),
        "errors": errors,
        "elapsed_ms": elapsed_ms,
    }
    with open(snap_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        "Snapshot created: %s (%d collections, %.1fms, %d errors)",
        snapshot_id,
        len(collection_info),
        elapsed_ms,
        len(errors),
    )
    return manifest


def _backup_qdrant_collection(collection_name: str, snap_dir: Path) -> dict:
    """Create and download a Qdrant snapshot for a single collection."""
    point_count = _collection_point_count(collection_name)

    resp = _qdrant_http("post", f"/collections/{collection_name}/snapshots")
    snap_info = resp.json().get("result", {})
    snap_name = snap_info.get("name", "")

    if not snap_name:
        raise RuntimeError(f"Qdrant returned no snapshot name for '{collection_name}'")

    dl_url = f"/collections/{collection_name}/snapshots/{snap_name}"
    dl_resp = _qdrant_http("get", dl_url)

    out_path = snap_dir / f"qdrant_{collection_name}.snapshot"
    with open(out_path, "wb") as f:
        f.write(dl_resp.content)

    try:
        _qdrant_http("delete", dl_url)
    except Exception:
        pass

    size_bytes = out_path.stat().st_size
    logger.info(
        "Qdrant snapshot: %s — %d points, %.1f MB",
        collection_name,
        point_count,
        size_bytes / (1024 * 1024),
    )
    return {
        "points": point_count,
        "snapshot_file": out_path.name,
        "size_bytes": size_bytes,
    }


def _backup_sqlite(snap_dir: Path) -> None:
    """Online backup of SQLite graph.db using the backup API."""
    if not os.path.isfile(SQLITE_PATH):
        logger.warning("SQLite database not found at %s — skipping", SQLITE_PATH)
        return

    dest_path = str(snap_dir / "graph.db")
    source = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    dest = sqlite3.connect(dest_path)
    try:
        source.backup(dest)
    finally:
        dest.close()
        source.close()

    size_bytes = os.path.getsize(dest_path)
    logger.info("SQLite backup: %.1f MB", size_bytes / (1024 * 1024))


def _pg_env() -> dict[str, str]:
    """Build a minimal environment for pg_dump / psql from DATABASE_URL.

    Parses ``postgresql://user:password@host:port/dbname`` and sets the
    standard ``PG*`` env variables.  Any component that is absent in the URL
    is omitted so that pg_dump falls back to its own defaults (unix socket,
    current user, etc.).
    """
    env = os.environ.copy()
    if not DATABASE_URL:
        return env

    try:
        parsed = urlparse(DATABASE_URL)
        if parsed.hostname:
            env["PGHOST"] = parsed.hostname
        if parsed.port:
            env["PGPORT"] = str(parsed.port)
        if parsed.username:
            env["PGUSER"] = parsed.username
        if parsed.password:
            env["PGPASSWORD"] = parsed.password
        # Path starts with '/' so strip leading slash to get dbname
        if parsed.path and parsed.path.lstrip("/"):
            env["PGDATABASE"] = parsed.path.lstrip("/")
    except Exception as exc:
        logger.warning("Could not parse DATABASE_URL for pg env: %s", exc)

    return env


def _backup_postgres(snap_dir: Path) -> None:
    """Dump PostgreSQL database to a custom-format ``pg_dump`` archive.

    The output file is ``graph.pgdump`` inside the snapshot directory.
    Uses ``pg_dump --format=custom`` which produces a compressed binary
    archive that ``pg_restore`` can reload selectively.

    Raises:
        RuntimeError: if ``pg_dump`` exits with a non-zero status.
        FileNotFoundError: if ``pg_dump`` is not on PATH.
    """
    dest_path = snap_dir / "graph.pgdump"
    pg_env = _pg_env()

    cmd = [
        "pg_dump",
        "--format=custom",
        "--no-password",
        f"--file={dest_path}",
    ]
    # If DATABASE_URL contains the dbname we already set PGDATABASE above;
    # pass it as the positional argument too for older pg_dump versions.
    if DATABASE_URL:
        cmd.append(DATABASE_URL)

    logger.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(
        cmd,
        env=pg_env,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minutes
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pg_dump failed (exit {result.returncode}):\n{result.stderr}"
        )

    size_bytes = dest_path.stat().st_size
    logger.info("Postgres backup: %.1f MB → %s", size_bytes / (1024 * 1024), dest_path.name)


def _restore_postgres(backup_path: Path) -> None:
    """Restore a PostgreSQL database from a ``pg_dump`` custom-format archive.

    Uses ``pg_restore --clean --if-exists`` to drop and recreate all objects,
    making the restore idempotent against a pre-existing database.

    Args:
        backup_path: Path to the ``graph.pgdump`` file.

    Raises:
        RuntimeError: if ``pg_restore`` exits with a non-zero status.
        FileNotFoundError: if ``pg_restore`` is not on PATH or backup missing.
    """
    if not backup_path.is_file():
        raise FileNotFoundError(f"Postgres backup file not found: {backup_path}")

    pg_env = _pg_env()

    cmd = [
        "pg_restore",
        "--clean",
        "--if-exists",
        "--no-password",
    ]
    if DATABASE_URL:
        cmd += ["--dbname", DATABASE_URL]
    cmd.append(str(backup_path))

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        env=pg_env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    # pg_restore exits 1 for warnings (e.g. object doesn't exist for --clean);
    # treat exit > 1 as a hard failure.
    if result.returncode > 1:
        raise RuntimeError(
            f"pg_restore failed (exit {result.returncode}):\n{result.stderr}"
        )
    if result.returncode == 1:
        logger.warning("pg_restore completed with warnings:\n%s", result.stderr)

    logger.info("Postgres database restored from %s", backup_path.name)


def _backup_memory_files(snap_dir: Path) -> None:
    """Create a tarball of MEMORY_ROOT markdown files."""
    tar_path = str(snap_dir / "memories.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for root, _dirs, files in os.walk(MEMORY_ROOT):
            for fname in files:
                if fname.endswith(".md"):
                    full = os.path.join(root, fname)
                    arcname = os.path.relpath(full, MEMORY_ROOT)
                    tar.add(full, arcname=arcname)
    size_bytes = os.path.getsize(tar_path)
    logger.info("Memory files backup: %.1f MB", size_bytes / (1024 * 1024))


# ── Snapshot listing ─────────────────────────────────────────────────────────


def list_snapshots() -> list[dict]:
    """List all available snapshots sorted newest-first."""
    _ensure_backup_dir()
    snapshots = []
    backup_path = Path(BACKUP_DIR)
    if not backup_path.is_dir():
        return []

    for entry in sorted(backup_path.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if manifest_path.is_file():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                total_points = sum(
                    c.get("points", 0) for c in manifest.get("collections", {}).values()
                )
                total_size = sum(
                    c.get("size_bytes", 0) for c in manifest.get("collections", {}).values()
                )
                if manifest.get("sqlite_backed_up"):
                    db_path = entry / "graph.db"
                    if db_path.is_file():
                        total_size += db_path.stat().st_size
                if manifest.get("postgres_backed_up"):
                    pg_path = entry / "graph.pgdump"
                    if pg_path.is_file():
                        total_size += pg_path.stat().st_size
                snapshots.append(
                    {
                        "snapshot_id": manifest.get("snapshot_id", entry.name),
                        "label": manifest.get("label", ""),
                        "created_at": manifest.get("created_at", ""),
                        "graph_backend": manifest.get("graph_backend", "sqlite"),
                        "collections": len(manifest.get("collections", {})),
                        "total_points": total_points,
                        "total_size_mb": round(total_size / (1024 * 1024), 2),
                        "sqlite": manifest.get("sqlite_backed_up", False),
                        "postgres": manifest.get("postgres_backed_up", False),
                        "files": manifest.get("files_backed_up", False),
                        "errors": len(manifest.get("errors", [])),
                    }
                )
            except (json.JSONDecodeError, OSError):
                snapshots.append(
                    {
                        "snapshot_id": entry.name,
                        "label": "",
                        "created_at": "",
                        "error": "corrupt manifest",
                    }
                )
    return snapshots


# ── Snapshot restore ─────────────────────────────────────────────────────────


def restore_snapshot(snapshot_id: str, target: str = "all") -> dict:
    """Restore from a snapshot directory.

    Args:
        snapshot_id: The snapshot directory name under BACKUP_DIR.
        target: What to restore — "all", "qdrant", or "sqlite".

    Returns summary of restored components.
    """
    snap_dir = _snapshot_dir(snapshot_id)
    manifest_path = snap_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snapshot '{snapshot_id}' not found or missing manifest")

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("vector_dim") and manifest["vector_dim"] != VECTOR_DIM:
        raise ValueError(
            f"Vector dimension mismatch: snapshot has {manifest['vector_dim']}-dim "
            f"but current config is {VECTOR_DIM}-dim"
        )

    t0 = time.monotonic()
    restored: dict[str, object] = {"snapshot_id": snapshot_id}
    errors: list[str] = []

    if target in ("all", "qdrant"):
        for coll, info in manifest.get("collections", {}).items():
            snap_file = snap_dir / info.get("snapshot_file", "")
            if snap_file.is_file():
                try:
                    _restore_qdrant_collection(coll, snap_file)
                    restored[f"qdrant_{coll}"] = "restored"
                except Exception as e:
                    msg = f"Qdrant restore failed for '{coll}': {e}"
                    logger.error(msg)
                    errors.append(msg)
            else:
                errors.append(f"Snapshot file missing for collection '{coll}'")

    if target in ("all", "sqlite"):
        _snap_backend = manifest.get("graph_backend", "sqlite").lower()
        if _snap_backend == "postgres":
            pg_backup = snap_dir / "graph.pgdump"
            if pg_backup.is_file():
                try:
                    _restore_postgres(pg_backup)
                    restored["postgres"] = "restored"
                except Exception as e:
                    msg = f"Postgres restore failed: {e}"
                    logger.error(msg)
                    errors.append(msg)
            else:
                errors.append("Postgres backup file (graph.pgdump) not found in snapshot")
        else:
            db_backup = snap_dir / "graph.db"
            if db_backup.is_file():
                try:
                    _restore_sqlite(db_backup)
                    restored["sqlite"] = "restored"
                except Exception as e:
                    msg = f"SQLite restore failed: {e}"
                    logger.error(msg)
                    errors.append(msg)
            else:
                errors.append("SQLite backup file not found in snapshot")

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    restored["elapsed_ms"] = elapsed_ms
    restored["errors"] = errors
    logger.info("Restore complete: %s (%.1fms, %d errors)", snapshot_id, elapsed_ms, len(errors))
    return restored


def _restore_qdrant_collection(collection_name: str, snapshot_file: Path) -> None:
    """Upload a snapshot file to restore a Qdrant collection."""
    with open(snapshot_file, "rb") as f:
        _qdrant_http(
            "post",
            f"/collections/{collection_name}/snapshots/upload",
            files={"snapshot": (snapshot_file.name, f, "application/octet-stream")},
            params={"priority": "snapshot"},
        )
    logger.info("Qdrant collection '%s' restored from snapshot", collection_name)


def _restore_sqlite(backup_path: Path) -> None:
    """Restore SQLite from a backup file using the backup API.

    Acquires GRAPH_WRITE_LOCK_ASYNC via the running event loop to prevent
    concurrent writes during the restore.  restore_snapshot() is always
    invoked from asyncio.to_thread(), so the running loop is accessible.
    """
    import asyncio as _asyncio

    loop = _asyncio.get_event_loop()

    async def _acquire_and_restore():
        async with _get_graph_write_lock():
            source = sqlite3.connect(str(backup_path))
            dest = sqlite3.connect(SQLITE_PATH)
            try:
                source.backup(dest)
            finally:
                dest.close()
                source.close()

    future = _asyncio.run_coroutine_threadsafe(_acquire_and_restore(), loop)
    future.result(timeout=120)
    logger.info("SQLite database restored from %s", backup_path.name)


# ── Snapshot deletion / pruning ──────────────────────────────────────────────


def delete_snapshot(snapshot_id: str) -> bool:
    """Delete a snapshot directory. Returns True if deleted."""
    snap_dir = _snapshot_dir(snapshot_id)
    if not snap_dir.is_dir():
        return False
    shutil.rmtree(snap_dir)
    logger.info("Deleted snapshot: %s", snapshot_id)
    return True


def prune_snapshots(keep: int = 0) -> list[str]:
    """Remove old snapshots, keeping the most recent ``keep`` snapshots.

    Returns list of pruned snapshot IDs.
    """
    keep = keep or BACKUP_RETENTION_COUNT
    all_snaps = list_snapshots()
    if len(all_snaps) <= keep:
        return []

    to_prune = all_snaps[keep:]
    pruned = []
    for snap in to_prune:
        sid = snap["snapshot_id"]
        if delete_snapshot(sid):
            pruned.append(sid)

    if pruned:
        logger.info("Pruned %d old snapshots (kept %d)", len(pruned), keep)
    return pruned


# ── Per-agent export / import (NDJSON) ───────────────────────────────────────


def export_agent(agent_id: str, output_dir: str = "") -> dict:
    """Export all memories for a single agent to NDJSON.

    Scrolls Qdrant for all points matching agent_id, writes one JSON line per
    point (id, vector, payload). Also exports matching SQLite graph rows.

    Returns summary with path and counts.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    out_dir = Path(output_dir) if output_dir else Path(BACKUP_DIR) / "exports"
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_agent = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent_id)
    filename = f"agent_{safe_agent}_{ts}.ndjson"
    out_path = out_dir / filename

    from archivist.storage.qdrant import qdrant_client as get_client

    client = get_client()

    agent_filter = Filter(must=[FieldCondition(key="agent_id", match=MatchValue(value=agent_id))])

    total_points = 0
    colls = collections_for_query("")

    with open(out_path, "w", encoding="utf-8") as f:
        for coll in colls:
            offset = None
            while True:
                try:
                    results, next_offset = client.scroll(
                        collection_name=coll,
                        scroll_filter=agent_filter,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=True,
                    )
                except Exception as e:
                    logger.warning("Scroll failed for %s (offset=%s): %s", coll, offset, e)
                    break

                for point in results:
                    record = {
                        "id": str(point.id),
                        "collection": coll,
                        "vector": point.vector
                        if isinstance(point.vector, list)
                        else list(point.vector),
                        "payload": dict(point.payload) if point.payload else {},
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_points += 1

                if next_offset is None:
                    break
                offset = next_offset

    graph_rows = _export_agent_graph(agent_id, out_dir, safe_agent, ts)

    size_bytes = out_path.stat().st_size
    summary = {
        "agent_id": agent_id,
        "file": str(out_path),
        "points": total_points,
        "graph_rows": graph_rows,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
    }
    logger.info("Agent export: %s — %d points, %d graph rows", agent_id, total_points, graph_rows)
    return summary


def _export_agent_graph(agent_id: str, out_dir: Path, safe_agent: str, ts: str) -> int:
    """Export agent-related graph data (entities, facts) to NDJSON.

    Uses the async pool so the export works correctly for both SQLite and
    Postgres backends without needing a direct sqlite3 connection.
    """
    graph_path = out_dir / f"agent_{safe_agent}_{ts}_graph.ndjson"
    count = 0

    async def _run() -> int:
        from archivist.storage.sqlite_pool import pool

        rows_written = 0
        async with pool.read() as conn:
            with open(graph_path, "w", encoding="utf-8") as f:
                for row in await conn.fetchall(
                    "SELECT * FROM facts WHERE agent_id = ? ORDER BY created_at",
                    (agent_id,),
                ):
                    f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
                    rows_written += 1

                for row in await conn.fetchall(
                    "SELECT mc.* FROM memory_chunks mc WHERE mc.agent_id = ?",
                    (agent_id,),
                ):
                    record = dict(row)
                    record["_table"] = "memory_chunks"
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    rows_written += 1
        return rows_written

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Called from a sync context inside asyncio.to_thread()

            future = asyncio.run_coroutine_threadsafe(_run(), loop)
            count = future.result(timeout=120)
        else:
            count = loop.run_until_complete(_run())
    except Exception as exc:
        logger.warning("_export_agent_graph pool query failed, falling back to sqlite3: %s", exc)
        # Fallback for SQLite-only deployments where the pool may not be initialized
        if (GRAPH_BACKEND or "sqlite").lower() != "postgres" and os.path.isfile(SQLITE_PATH):
            conn_s = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
            conn_s.row_factory = sqlite3.Row
            try:
                with open(graph_path, "w", encoding="utf-8") as f:
                    for row in conn_s.execute(
                        "SELECT * FROM facts WHERE agent_id = ? ORDER BY created_at", (agent_id,)
                    ):
                        f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
                        count += 1
                    for row in conn_s.execute(
                        "SELECT mc.* FROM memory_chunks mc WHERE mc.agent_id = ?", (agent_id,)
                    ):
                        record = dict(row)
                        record["_table"] = "memory_chunks"
                        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                        count += 1
            finally:
                conn_s.close()

    return count


def import_agent(ndjson_path: str, dry_run: bool = False) -> dict:
    """Import agent memories from an NDJSON export file.

    Upserts points back into Qdrant and rebuilds FTS entries.
    """
    from qdrant_client.models import PointStruct

    from archivist.core.config import BM25_ENABLED
    from archivist.storage.collection_router import ensure_collection
    from archivist.storage.graph import upsert_fts_chunk

    path = Path(ndjson_path)
    if not path.is_file():
        raise FileNotFoundError(f"Import file not found: {ndjson_path}")

    from archivist.storage.qdrant import qdrant_client as get_client

    client = get_client()

    imported = 0
    skipped = 0
    fts_rebuilt = 0

    with open(path, encoding="utf-8") as f:
        batch: list[tuple[str, PointStruct]] = []

        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d", line_num)
                skipped += 1
                continue

            if "_table" in record:
                skipped += 1
                continue

            point_id = record.get("id", "")
            vector = record.get("vector", [])
            payload = record.get("payload", {})
            coll = record.get("collection", QDRANT_COLLECTION)

            if not point_id or not vector:
                skipped += 1
                continue

            if not dry_run:
                target_coll = ensure_collection(payload.get("namespace", ""))
                point = PointStruct(id=point_id, vector=vector, payload=payload)
                batch.append((target_coll, point))

                if len(batch) >= 100:
                    _flush_import_batch(client, batch)
                    imported += len(batch)
                    batch.clear()

                if BM25_ENABLED and payload.get("text"):
                    asyncio.run(
                        upsert_fts_chunk(
                            qdrant_id=point_id,
                            text=payload["text"],
                            file_path=payload.get("file_path", ""),
                            chunk_index=payload.get("chunk_index", 0),
                            agent_id=payload.get("agent_id", ""),
                            namespace=payload.get("namespace", ""),
                            date=payload.get("date", ""),
                            memory_type=payload.get("memory_type", "general"),
                        )
                    )
                    fts_rebuilt += 1
            else:
                imported += 1

        if batch and not dry_run:
            _flush_import_batch(client, batch)
            imported += len(batch)

    summary = {
        "file": str(path),
        "imported": imported,
        "skipped": skipped,
        "fts_rebuilt": fts_rebuilt,
        "dry_run": dry_run,
    }
    logger.info(
        "Agent import: %d points imported, %d skipped, %d FTS rebuilt",
        imported,
        skipped,
        fts_rebuilt,
    )
    return summary


def _flush_import_batch(client, batch: list[tuple[str, object]]) -> None:
    """Upsert a batch of points grouped by collection."""
    by_coll: dict[str, list] = {}
    for coll, point in batch:
        by_coll.setdefault(coll, []).append(point)

    for coll, points in by_coll.items():
        try:
            client.upsert(collection_name=coll, points=points)
        except Exception as e:
            logger.error("Batch upsert to '%s' failed (%d points): %s", coll, len(points), e)
