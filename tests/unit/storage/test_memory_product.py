"""Unit tests for Memory-as-Product service (INIT-001/SPEC-009; INIT-011/SPEC-002).

Covers scope snapshot, fork lineage, export manifest, import restore, RBAC
boundaries, and BACKUP_DIR path containment — without a live Qdrant
(vectors mocked / omitted).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POSTGRES_SQL = _REPO_ROOT / "src" / "archivist" / "storage" / "schema_postgres.sql"


def test_sqlite_schema_creates_memory_scope_versions_idempotently(tmp_path, monkeypatch):
    """Schema creates memory_scope_versions idempotently on SQLite."""
    db_path = str(tmp_path / "msv.db")
    monkeypatch.setenv("SQLITE_PATH", db_path)
    monkeypatch.setattr("archivist.core.config.SQLITE_PATH", db_path)
    monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "sqlite")

    from archivist.storage import graph_schema

    monkeypatch.setattr(graph_schema, "SQLITE_PATH", db_path, raising=False)
    graph_schema.init_schema()
    graph_schema.init_schema()

    conn = sqlite3.connect(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memory_scope_versions)")}
        assert {
            "id",
            "source_namespace",
            "source_agent_id",
            "version",
            "label",
            "parent_version_id",
            "chunk_count",
            "point_count",
            "archive_id",
            "operation",
            "created_by",
            "created_at",
            "lineage_json",
        } <= cols
        idx_sql = " ".join(
            (r[0] or "")
            for r in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' "
                "AND tbl_name='memory_scope_versions'"
            ).fetchall()
        )
        assert "idx_msv_namespace_version" in idx_sql
        assert "idx_msv_parent" in idx_sql
    finally:
        conn.close()


def test_postgres_sql_artifact_defines_memory_scope_versions():
    """Postgres DDL artifact includes memory_scope_versions + rollback note."""
    assert _POSTGRES_SQL.is_file()
    ddl = _POSTGRES_SQL.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS memory_scope_versions" in ddl
    assert "idx_msv_namespace_version" in ddl
    assert "DROP TABLE IF EXISTS memory_scope_versions" in ddl


async def _seed_chunks(pool, *, namespace: str, agent_id: str, n: int = 2) -> list[str]:
    ids: list[str] = []
    async with pool.write() as conn:
        for i in range(n):
            qid = f"q-{namespace}-{i}"
            ids.append(qid)
            await conn.execute(
                """
                INSERT INTO memory_chunks (
                    qdrant_id, text, file_path, chunk_index, agent_id, namespace,
                    date, memory_type, is_excluded
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (qid, f"text {i}", f"mem/{i}.md", i, agent_id, namespace, "2026-07-25", "general"),
            )
    return ids


@pytest.mark.asyncio
async def test_create_scope_snapshot_and_lineage(async_pool, tmp_path, monkeypatch, rbac_config):
    """ac-1 / ac-5: snapshot scoped chunks with version lineage; no live Qdrant."""
    from archivist.storage import memory_product as mp
    from archivist.storage.backup_manager import _snapshot_dir, delete_snapshot

    backup = tmp_path / "backups"
    backup.mkdir()
    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(backup))
    monkeypatch.setattr("archivist.storage.memory_product._snapshot_dir", _snapshot_dir)
    monkeypatch.setattr("archivist.storage.memory_product.delete_snapshot", delete_snapshot)

    await _seed_chunks(async_pool, namespace="chief", agent_id="chief", n=3)

    # Mock vectors (no Qdrant)
    mock_vectors = [
        {"id": "q-chief-0", "vector": [0.1, 0.2], "payload": {"namespace": "chief"}},
        {"id": "q-chief-1", "vector": [0.3, 0.4], "payload": {"namespace": "chief"}},
    ]

    record = await mp.create_scope_snapshot(
        namespace="chief",
        caller_agent_id="chief",
        agent_id="chief",
        label="v1",
        vectors=mock_vectors,
    )
    assert record.operation == "snapshot"
    assert record.version == 1
    assert record.chunk_count == 3
    assert record.point_count == 2
    assert record.parent_version_id is None
    assert record.archive_id.startswith("memprod_")

    snap_dir = Path(backup) / record.archive_id
    assert (snap_dir / "manifest.json").is_file()
    assert (snap_dir / "chunks.ndjson").is_file()
    manifest = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_count"] == 3
    assert manifest["version"] == 1
    assert "api_key" not in manifest
    assert "archivist_api_key" not in json.dumps(manifest).lower()

    listed = await mp.list_scope_versions(
        namespace="chief", caller_agent_id="chief", agent_id="chief"
    )
    assert len(listed) == 1
    assert listed[0].id == record.id


@pytest.mark.asyncio
async def test_fork_from_snapshot_lineage_and_isolation(
    async_pool, tmp_path, monkeypatch, rbac_config
):
    """ac-2: fork copies into target scope with parent lineage pointer."""
    from archivist.storage import memory_product as mp
    from archivist.storage.backup_manager import _snapshot_dir, delete_snapshot

    backup = tmp_path / "backups"
    backup.mkdir()
    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(backup))
    monkeypatch.setattr("archivist.storage.memory_product._snapshot_dir", _snapshot_dir)
    monkeypatch.setattr("archivist.storage.memory_product.delete_snapshot", delete_snapshot)
    # shared is writable by all in rbac_config fixture
    await _seed_chunks(async_pool, namespace="shared", agent_id="chief", n=2)

    source = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="chief",
        agent_id="chief",
        label="src",
        vectors=[
            {"id": "q-shared-0", "vector": [1.0, 0.0], "payload": {}},
            {"id": "q-shared-1", "vector": [0.0, 1.0], "payload": {}},
        ],
    )

    forked = await mp.fork_from_snapshot(
        source_version_id=source.id,
        target_namespace="pipeline",
        caller_agent_id="gitbob",  # write ACL on pipeline
        target_agent_id="gitbob",
        label="fork-1",
    )
    assert forked.operation == "fork"
    assert forked.parent_version_id == source.id
    assert forked.source_namespace == "pipeline"
    assert forked.chunk_count == 2
    assert forked.lineage.get("source_namespace") == "shared"
    assert forked.lineage.get("target_namespace") == "pipeline"

    # Target scope has new chunks; source unchanged count
    async with async_pool.read() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) AS c FROM memory_chunks WHERE namespace = ?",
            ("pipeline",),
        )
        row = await cur.fetchone()
        assert int(row["c"]) == 2
        cur = await conn.execute(
            "SELECT COUNT(*) AS c FROM memory_chunks WHERE namespace = ?",
            ("shared",),
        )
        row = await cur.fetchone()
        assert int(row["c"]) == 2
        cur = await conn.execute(
            "SELECT agent_id, namespace FROM memory_chunks WHERE namespace = ?",
            ("pipeline",),
        )
        rows = await cur.fetchall()
        assert all(r["agent_id"] == "gitbob" for r in rows)


@pytest.mark.asyncio
async def test_export_produces_manifest_path_and_bytes(
    async_pool, tmp_path, monkeypatch, rbac_config
):
    """ac-3: export returns path + manifest bytes with counts/versions."""
    from archivist.storage import memory_product as mp
    from archivist.storage.backup_manager import _snapshot_dir, delete_snapshot

    backup = tmp_path / "backups"
    backup.mkdir()
    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(backup))
    monkeypatch.setattr("archivist.storage.memory_product._snapshot_dir", _snapshot_dir)
    monkeypatch.setattr("archivist.storage.memory_product.delete_snapshot", delete_snapshot)

    await _seed_chunks(async_pool, namespace="chief", agent_id="chief", n=1)
    result = await mp.export_scope(
        namespace="chief",
        caller_agent_id="chief",
        agent_id="chief",
        label="export-me",
        vectors=[{"id": "q-chief-0", "vector": [0.5], "payload": {}}],
    )
    assert Path(result["path"]).is_dir()
    assert Path(result["path"]).resolve().is_relative_to(backup.resolve())
    assert result["manifest"]["chunk_count"] == 1
    assert result["manifest"]["version"] >= 1
    assert isinstance(result["bytes"], bytes | bytearray)
    assert b"chunk_count" in result["bytes"]
    assert b"api_key" not in result["bytes"].lower()


@pytest.mark.asyncio
async def test_rbac_denies_unauthorized_snapshot(async_pool, tmp_path, monkeypatch, rbac_config):
    """ac-4: cannot snapshot/export across unauthorized namespaces."""
    from archivist.storage import memory_product as mp
    from archivist.storage.backup_manager import _snapshot_dir, delete_snapshot

    backup = tmp_path / "backups"
    backup.mkdir()
    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(backup))
    monkeypatch.setattr("archivist.storage.memory_product._snapshot_dir", _snapshot_dir)
    monkeypatch.setattr("archivist.storage.memory_product.delete_snapshot", delete_snapshot)

    await _seed_chunks(async_pool, namespace="deployer", agent_id="argo", n=1)

    with pytest.raises(mp.MemoryProductAuthzError):
        await mp.create_scope_snapshot(
            namespace="deployer",
            caller_agent_id="gitbob",  # no read on deployer
        )

    with pytest.raises(mp.MemoryProductAuthzError):
        await mp.export_scope(
            namespace="deployer",
            caller_agent_id="gitbob",
        )


@pytest.mark.asyncio
async def test_fork_rbac_requires_write_on_target(async_pool, tmp_path, monkeypatch, rbac_config):
    """ac-4: fork denied when caller lacks write on target namespace."""
    from archivist.storage import memory_product as mp
    from archivist.storage.backup_manager import _snapshot_dir, delete_snapshot

    backup = tmp_path / "backups"
    backup.mkdir()
    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(backup))
    monkeypatch.setattr("archivist.storage.memory_product._snapshot_dir", _snapshot_dir)
    monkeypatch.setattr("archivist.storage.memory_product.delete_snapshot", delete_snapshot)

    await _seed_chunks(async_pool, namespace="shared", agent_id="chief", n=1)
    source = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="gitbob",
        label="ok",
    )
    # gitbob can read shared but cannot write deployer
    with pytest.raises(mp.MemoryProductAuthzError):
        await mp.fork_from_snapshot(
            source_version_id=source.id,
            target_namespace="deployer",
            caller_agent_id="gitbob",
        )


def test_export_archive_id_path_traversal_rejected(tmp_path, monkeypatch):
    """Security: export paths confined via SnapshotPathError / _snapshot_dir."""
    from archivist.storage.backup_manager import SnapshotPathError
    from archivist.storage.memory_product import _write_archive

    backup = tmp_path / "backups"
    backup.mkdir()
    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(backup))

    with pytest.raises(SnapshotPathError):
        _write_archive(
            "../../etc/passwd",
            chunks=[],
            vectors=[],
            manifest={"kind": "memory_scope"},
        )


def test_manifest_strips_secret_keys():
    from archivist.storage.memory_product import _strip_secrets

    cleaned = _strip_secrets(
        {
            "chunk_count": 1,
            "api_key": "SHOULD_NOT_LEAK",
            "nested": {"token": "x", "ok": True},
        }
    )
    assert cleaned == {"chunk_count": 1, "nested": {"ok": True}}


@pytest.mark.asyncio
async def test_fork_rolls_back_on_mid_transaction_failure(
    async_pool, tmp_path, monkeypatch, rbac_config
):
    """Technical req: no partial fork — transaction rolls back on error."""
    from archivist.storage import memory_product as mp
    from archivist.storage.backup_manager import _snapshot_dir, delete_snapshot
    from archivist.storage.transaction import MemoryTransaction

    backup = tmp_path / "backups"
    backup.mkdir()
    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(backup))
    monkeypatch.setattr("archivist.storage.memory_product._snapshot_dir", _snapshot_dir)
    monkeypatch.setattr("archivist.storage.memory_product.delete_snapshot", delete_snapshot)

    await _seed_chunks(async_pool, namespace="shared", agent_id="chief", n=2)
    source = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="chief",
        vectors=[{"id": "q-shared-0", "vector": [1.0], "payload": {}}],
    )

    original_upsert = MemoryTransaction.upsert_fts_chunk
    call_count = {"n": 0}

    async def boom(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("simulated mid-fork failure")
        return await original_upsert(self, *args, **kwargs)

    monkeypatch.setattr(MemoryTransaction, "upsert_fts_chunk", boom)

    with pytest.raises(mp.MemoryProductError, match="rolled back"):
        await mp.fork_from_snapshot(
            source_version_id=source.id,
            target_namespace="pipeline",
            caller_agent_id="gitbob",
            target_agent_id="gitbob",
        )

    async with async_pool.read() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) AS c FROM memory_chunks WHERE namespace = ?",
            ("pipeline",),
        )
        assert int((await cur.fetchone())["c"]) == 0
        cur = await conn.execute(
            "SELECT COUNT(*) AS c FROM memory_scope_versions WHERE operation = 'fork'"
        )
        assert int((await cur.fetchone())["c"]) == 0


# --- INIT-011/SPEC-002: import_scope ---


def _map_backup_patches(monkeypatch, backup: Path) -> None:
    from archivist.storage.backup_manager import _snapshot_dir, delete_snapshot

    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(backup))
    monkeypatch.setattr("archivist.storage.memory_product._snapshot_dir", _snapshot_dir)
    monkeypatch.setattr("archivist.storage.memory_product.delete_snapshot", delete_snapshot)


@pytest.mark.asyncio
async def test_import_scope_restores_chunks_and_lineage(
    async_pool, tmp_path, monkeypatch, rbac_config
):
    """ac-1 / ac-4 / ac-5: import restores chunks; lineage operation=import; vectors optional."""
    from archivist.storage import memory_product as mp

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)

    await _seed_chunks(async_pool, namespace="shared", agent_id="chief", n=2)
    source = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="chief",
        agent_id="chief",
        label="export-src",
        vectors=[
            {"id": "q-shared-0", "vector": [1.0, 0.0], "payload": {}},
            {"id": "q-shared-1", "vector": [0.0, 1.0], "payload": {}},
        ],
    )

    imported = await mp.import_scope(
        archive_id=source.archive_id,
        target_namespace="pipeline",
        caller_agent_id="gitbob",
        target_agent_id="gitbob",
        label="import-1",
    )
    assert imported.operation == "import"
    assert imported.source_namespace == "pipeline"
    assert imported.chunk_count == 2
    assert imported.point_count == 2
    assert imported.lineage.get("source_archive_id") == source.archive_id
    assert imported.lineage.get("operation") == "import"
    assert imported.parent_version_id == source.id
    assert "api_key" not in json.dumps(imported.lineage).lower()

    async with async_pool.read() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) AS c FROM memory_chunks WHERE namespace = ? AND agent_id = ?",
            ("pipeline", "gitbob"),
        )
        assert int((await cur.fetchone())["c"]) == 2
        cur = await conn.execute(
            "SELECT COUNT(*) AS c FROM memory_scope_versions WHERE operation = 'import'"
        )
        assert int((await cur.fetchone())["c"]) == 1


@pytest.mark.asyncio
async def test_import_scope_chunks_without_vectors(async_pool, tmp_path, monkeypatch, rbac_config):
    """ac-5: chunks durable when archive has no vectors."""
    from archivist.storage import memory_product as mp

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)

    await _seed_chunks(async_pool, namespace="shared", agent_id="chief", n=1)
    source = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="chief",
        agent_id="chief",
        vectors=[],
    )
    imported = await mp.import_scope(
        archive_id=source.archive_id,
        target_namespace="pipeline",
        caller_agent_id="gitbob",
        target_agent_id="gitbob",
    )
    assert imported.chunk_count == 1
    assert imported.point_count == 0


@pytest.mark.asyncio
async def test_import_rbac_requires_write_on_target(async_pool, tmp_path, monkeypatch, rbac_config):
    """ac-3: import denied without namespace write."""
    from archivist.storage import memory_product as mp

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)

    await _seed_chunks(async_pool, namespace="shared", agent_id="chief", n=1)
    source = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="chief",
        label="src",
    )
    with pytest.raises(mp.MemoryProductAuthzError):
        await mp.import_scope(
            archive_id=source.archive_id,
            target_namespace="deployer",
            caller_agent_id="gitbob",
        )


@pytest.mark.asyncio
async def test_import_rbac_requires_read_on_source(async_pool, tmp_path, monkeypatch, rbac_config):
    """SEC-011-01: import denied without read on archive source namespace."""
    from archivist.storage import memory_product as mp

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)

    await _seed_chunks(async_pool, namespace="deployer", agent_id="argo", n=1)
    source = await mp.create_scope_snapshot(
        namespace="deployer",
        caller_agent_id="argo",
        agent_id="argo",
        label="secret-src",
    )
    # gitbob can write pipeline but cannot read deployer
    with pytest.raises(mp.MemoryProductAuthzError):
        await mp.import_scope(
            archive_id=source.archive_id,
            target_namespace="pipeline",
            caller_agent_id="gitbob",
            target_agent_id="gitbob",
        )


@pytest.mark.asyncio
async def test_import_path_traversal_rejected(async_pool, tmp_path, monkeypatch, rbac_config):
    """ac-2: path escape via archive_id fails closed (SnapshotPathError)."""
    from archivist.storage import memory_product as mp
    from archivist.storage.backup_manager import SnapshotPathError

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)

    with pytest.raises(SnapshotPathError):
        await mp.import_scope(
            archive_id="../../etc/passwd",
            target_namespace="pipeline",
            caller_agent_id="gitbob",
            target_agent_id="gitbob",
        )


@pytest.mark.asyncio
async def test_import_missing_archive_not_found(async_pool, tmp_path, monkeypatch, rbac_config):
    """ac-2: missing archive raises MemoryProductNotFoundError."""
    from archivist.storage import memory_product as mp

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)

    with pytest.raises(mp.MemoryProductNotFoundError):
        await mp.import_scope(
            archive_id="memprod_missing_does_not_exist",
            target_namespace="pipeline",
            caller_agent_id="gitbob",
            target_agent_id="gitbob",
        )


@pytest.mark.asyncio
async def test_import_rejects_nonempty_target(async_pool, tmp_path, monkeypatch, rbac_config):
    """ADR-011: fail-closed when target scope already has chunks."""
    from archivist.storage import memory_product as mp

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)

    await _seed_chunks(async_pool, namespace="shared", agent_id="chief", n=1)
    await _seed_chunks(async_pool, namespace="pipeline", agent_id="gitbob", n=1)
    source = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="chief",
        agent_id="chief",
    )
    with pytest.raises(mp.MemoryProductConflictError, match="not empty"):
        await mp.import_scope(
            archive_id=source.archive_id,
            target_namespace="pipeline",
            caller_agent_id="gitbob",
            target_agent_id="gitbob",
        )


@pytest.mark.asyncio
async def test_import_rejects_oversized_chunk_count(async_pool, tmp_path, monkeypatch, rbac_config):
    """Security: oversized archive rejected per ADR caps."""
    from archivist.storage import memory_product as mp

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)
    monkeypatch.setattr(mp, "MAP_IMPORT_MAX_CHUNKS", 1)

    await _seed_chunks(async_pool, namespace="shared", agent_id="chief", n=2)
    source = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="chief",
        agent_id="chief",
    )
    with pytest.raises(mp.MemoryProductConflictError, match="max chunks"):
        await mp.import_scope(
            archive_id=source.archive_id,
            target_namespace="pipeline",
            caller_agent_id="gitbob",
            target_agent_id="gitbob",
        )


@pytest.mark.asyncio
async def test_import_rejects_oversized_bytes(async_pool, tmp_path, monkeypatch, rbac_config):
    """Security: archive byte cap rejected per ADR-011."""
    from archivist.storage import memory_product as mp

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)
    monkeypatch.setattr(mp, "MAP_IMPORT_MAX_BYTES", 10)

    await _seed_chunks(async_pool, namespace="shared", agent_id="chief", n=1)
    source = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="chief",
        agent_id="chief",
    )
    with pytest.raises(mp.MemoryProductConflictError, match="max bytes"):
        await mp.import_scope(
            archive_id=source.archive_id,
            target_namespace="pipeline",
            caller_agent_id="gitbob",
            target_agent_id="gitbob",
        )


@pytest.mark.asyncio
async def test_import_rejects_wrong_manifest_kind(async_pool, tmp_path, monkeypatch, rbac_config):
    """Edge: corrupt / wrong MANIFEST_KIND fails closed."""
    from archivist.storage import memory_product as mp
    from archivist.storage.backup_manager import _snapshot_dir

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)

    archive_id = "memprod_bad_kind"
    snap = _snapshot_dir(archive_id)
    snap.mkdir(parents=True)
    (snap / "manifest.json").write_text(
        json.dumps({"kind": "not_memory_scope", "manifest_version": 1}),
        encoding="utf-8",
    )
    (snap / "chunks.ndjson").write_text("", encoding="utf-8")

    with pytest.raises(mp.MemoryProductConflictError, match="unsupported archive kind"):
        await mp.import_scope(
            archive_id=archive_id,
            target_namespace="pipeline",
            caller_agent_id="gitbob",
            target_agent_id="gitbob",
        )
