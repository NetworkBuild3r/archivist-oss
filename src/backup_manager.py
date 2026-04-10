"""Backup and restore engine for Archivist memory stores.

Backs up:
  - Qdrant collections via the native snapshot REST API
  - SQLite graph.db via Python's online backup API
  - Optionally, memory source files as a tarball

Each snapshot is a timestamped directory under BACKUP_DIR containing the above
artefacts plus a manifest.json with metadata for validation during restore.
"""

import json
import logging
import os
import shutil
import sqlite3
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config import (
    BACKUP_DIR,
    BACKUP_RETENTION_COUNT,
    BACKUP_INCLUDE_FILES,
    QDRANT_URL,
    QDRANT_COLLECTION,
    SQLITE_PATH,
    MEMORY_ROOT,
    VECTOR_DIM,
)
from collection_router import collections_for_query
from graph import get_db, GRAPH_WRITE_LOCK

logger = logging.getLogger("archivist.backup")

MANIFEST_VERSION = 1
ARCHIVIST_VERSION = "1.10.0"


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
    ts = datetime.now(timezone.utc)
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
        "collections": collection_info,
        "sqlite_backed_up": os.path.isfile(snap_dir / "graph.db"),
        "files_backed_up": os.path.isfile(snap_dir / "memories.tar.gz"),
        "errors": errors,
        "elapsed_ms": elapsed_ms,
    }
    with open(snap_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        "Snapshot created: %s (%d collections, %.1fms, %d errors)",
        snapshot_id, len(collection_info), elapsed_ms, len(errors),
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
        collection_name, point_count, size_bytes / (1024 * 1024),
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
                snapshots.append({
                    "snapshot_id": manifest.get("snapshot_id", entry.name),
                    "label": manifest.get("label", ""),
                    "created_at": manifest.get("created_at", ""),
                    "collections": len(manifest.get("collections", {})),
                    "total_points": total_points,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "sqlite": manifest.get("sqlite_backed_up", False),
                    "files": manifest.get("files_backed_up", False),
                    "errors": len(manifest.get("errors", [])),
                })
            except (json.JSONDecodeError, OSError):
                snapshots.append({
                    "snapshot_id": entry.name,
                    "label": "",
                    "created_at": "",
                    "error": "corrupt manifest",
                })
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
    """Restore SQLite from a backup file using the backup API."""
    with GRAPH_WRITE_LOCK:
        source = sqlite3.connect(str(backup_path))
        dest = sqlite3.connect(SQLITE_PATH)
        try:
            source.backup(dest)
        finally:
            dest.close()
            source.close()
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
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    out_dir = Path(output_dir) if output_dir else Path(BACKUP_DIR) / "exports"
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_agent = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent_id)
    filename = f"agent_{safe_agent}_{ts}.ndjson"
    out_path = out_dir / filename

    from qdrant import qdrant_client as get_client
    client = get_client()

    agent_filter = Filter(
        must=[FieldCondition(key="agent_id", match=MatchValue(value=agent_id))]
    )

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
                        "vector": point.vector if isinstance(point.vector, list) else list(point.vector),
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
    """Export agent-related graph data (entities, facts) to NDJSON."""
    graph_path = out_dir / f"agent_{safe_agent}_{ts}_graph.ndjson"
    count = 0

    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        with open(graph_path, "w", encoding="utf-8") as f:
            for row in conn.execute(
                "SELECT * FROM facts WHERE agent_id = ? ORDER BY created_at", (agent_id,)
            ):
                f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
                count += 1

            for row in conn.execute(
                "SELECT mc.* FROM memory_chunks mc WHERE mc.agent_id = ?", (agent_id,)
            ):
                record = dict(row)
                record["_table"] = "memory_chunks"
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                count += 1
    finally:
        conn.close()

    return count


def import_agent(ndjson_path: str, dry_run: bool = False) -> dict:
    """Import agent memories from an NDJSON export file.

    Upserts points back into Qdrant and rebuilds FTS entries.
    """
    from qdrant_client.models import PointStruct
    from collection_router import ensure_collection
    from graph import upsert_fts_chunk
    from config import BM25_ENABLED

    path = Path(ndjson_path)
    if not path.is_file():
        raise FileNotFoundError(f"Import file not found: {ndjson_path}")

    from qdrant import qdrant_client as get_client
    client = get_client()

    imported = 0
    skipped = 0
    fts_rebuilt = 0

    with open(path, "r", encoding="utf-8") as f:
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
    logger.info("Agent import: %d points imported, %d skipped, %d FTS rebuilt", imported, skipped, fts_rebuilt)
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
