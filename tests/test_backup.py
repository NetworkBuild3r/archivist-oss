"""Tests for the backup / restore / export / import system."""

import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestManifest(unittest.TestCase):
    """Test manifest creation and validation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = [
            patch("backup_manager.BACKUP_DIR", self.tmpdir),
            patch("backup_manager.QDRANT_URL", "http://localhost:6333"),
            patch("backup_manager.QDRANT_COLLECTION", "test_memories"),
            patch("backup_manager.SQLITE_PATH", os.path.join(self.tmpdir, "graph.db")),
            patch("backup_manager.MEMORY_ROOT", os.path.join(self.tmpdir, "memories")),
            patch("backup_manager.VECTOR_DIM", 768),
            patch("backup_manager.BACKUP_INCLUDE_FILES", False),
            patch("backup_manager.BACKUP_RETENTION_COUNT", 5),
        ]
        for p in self.patches:
            p.start()

        db_path = os.path.join(self.tmpdir, "graph.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("backup_manager.collections_for_query")
    @patch("backup_manager._qdrant_http")
    def test_create_snapshot_writes_manifest(self, mock_http, mock_colls):
        mock_colls.return_value = ["test_memories"]
        snap_content = b"\x00" * 100

        def http_side_effect(method, path, **kwargs):
            resp = MagicMock()
            if method == "get" and "collections" in path and "snapshots" not in path:
                resp.json.return_value = {"result": {"points_count": 42}}
            elif method == "post" and "snapshots" in path:
                resp.json.return_value = {"result": {"name": "snap_001.snapshot"}}
            elif method == "get" and "snapshots" in path:
                resp.content = snap_content
            elif method == "delete":
                pass
            return resp

        mock_http.side_effect = http_side_effect

        from backup_manager import create_snapshot

        result = create_snapshot(label="test-backup")

        assert result["manifest_version"] == 1
        assert result["label"] == "test-backup"
        assert result["vector_dim"] == 768
        assert "test_memories" in result["collections"]
        assert result["collections"]["test_memories"]["points"] == 42
        assert result["sqlite_backed_up"] is True
        assert result["snapshot_id"].endswith("_test-backup")

        snap_id = result["snapshot_id"]
        manifest_path = Path(self.tmpdir) / snap_id / "manifest.json"
        assert manifest_path.is_file()

        with open(manifest_path) as f:
            on_disk = json.load(f)
        assert on_disk["snapshot_id"] == snap_id
        assert on_disk["archivist_version"] == "2.0.0"

    @patch("backup_manager.collections_for_query")
    @patch("backup_manager._qdrant_http")
    def test_list_snapshots_returns_sorted(self, mock_http, mock_colls):
        mock_colls.return_value = ["test_memories"]
        snap_content = b"\x00" * 50

        def http_side_effect(method, path, **kwargs):
            resp = MagicMock()
            if method == "get" and "collections" in path and "snapshots" not in path:
                resp.json.return_value = {"result": {"points_count": 10}}
            elif method == "post" and "snapshots" in path:
                resp.json.return_value = {"result": {"name": "s.snapshot"}}
            elif method == "get" and "snapshots" in path:
                resp.content = snap_content
            elif method == "delete":
                pass
            return resp

        mock_http.side_effect = http_side_effect

        from backup_manager import create_snapshot, list_snapshots

        create_snapshot(label="first")
        time.sleep(0.05)
        create_snapshot(label="second")

        snapshots = list_snapshots()
        assert len(snapshots) == 2
        assert snapshots[0]["label"] == "second"
        assert snapshots[1]["label"] == "first"


class TestSQLiteBackup(unittest.TestCase):
    """Test SQLite online backup round-trip."""

    def test_backup_and_restore_roundtrip(self):
        tmpdir = tempfile.mkdtemp()
        try:
            src_path = os.path.join(tmpdir, "source.db")
            conn = sqlite3.connect(src_path)
            conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, text TEXT)")
            conn.execute("INSERT INTO memories VALUES ('m1', 'hello world')")
            conn.execute("INSERT INTO memories VALUES ('m2', 'test data')")
            conn.commit()
            conn.close()

            backup_path = os.path.join(tmpdir, "backup.db")
            source = sqlite3.connect(src_path)
            dest = sqlite3.connect(backup_path)
            source.backup(dest)
            dest.close()
            source.close()

            restored = sqlite3.connect(backup_path)
            rows = restored.execute("SELECT * FROM memories ORDER BY id").fetchall()
            restored.close()

            assert len(rows) == 2
            assert rows[0][0] == "m1"
            assert rows[0][1] == "hello world"
            assert rows[1][0] == "m2"
            assert rows[1][1] == "test data"
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


class TestSnapshotDeleteAndPrune(unittest.TestCase):
    """Test snapshot deletion and retention pruning."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = [
            patch("backup_manager.BACKUP_DIR", self.tmpdir),
            patch("backup_manager.BACKUP_RETENTION_COUNT", 2),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fake_snapshot(self, snapshot_id: str, label: str = "") -> None:
        snap_dir = Path(self.tmpdir) / snapshot_id
        snap_dir.mkdir(parents=True)
        manifest = {
            "manifest_version": 1,
            "snapshot_id": snapshot_id,
            "label": label,
            "created_at": "2025-01-01T00:00:00+00:00",
            "vector_dim": 768,
            "collections": {},
            "sqlite_backed_up": False,
            "files_backed_up": False,
            "errors": [],
        }
        with open(snap_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

    def test_delete_snapshot(self):
        from backup_manager import delete_snapshot

        self._make_fake_snapshot("snap_001")
        assert (Path(self.tmpdir) / "snap_001").is_dir()
        assert delete_snapshot("snap_001") is True
        assert not (Path(self.tmpdir) / "snap_001").is_dir()

    def test_delete_nonexistent(self):
        from backup_manager import delete_snapshot

        assert delete_snapshot("nonexistent") is False

    def test_prune_keeps_n_most_recent(self):
        from backup_manager import list_snapshots, prune_snapshots

        self._make_fake_snapshot("20250101T000000Z_a", "a")
        self._make_fake_snapshot("20250102T000000Z_b", "b")
        self._make_fake_snapshot("20250103T000000Z_c", "c")

        before = list_snapshots()
        assert len(before) == 3

        pruned = prune_snapshots(keep=2)
        assert len(pruned) == 1

        after = list_snapshots()
        assert len(after) == 2


class TestRestoreValidation(unittest.TestCase):
    """Test restore validation (dimension mismatch, missing snapshot)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patches = [
            patch("backup_manager.BACKUP_DIR", self.tmpdir),
            patch("backup_manager.VECTOR_DIM", 768),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_restore_missing_snapshot(self):
        from backup_manager import restore_snapshot

        with self.assertRaises(FileNotFoundError):
            restore_snapshot("nonexistent_snapshot")

    def test_restore_dimension_mismatch(self):
        from backup_manager import restore_snapshot

        snap_dir = Path(self.tmpdir) / "dim_mismatch"
        snap_dir.mkdir()
        manifest = {
            "manifest_version": 1,
            "snapshot_id": "dim_mismatch",
            "vector_dim": 1024,
            "collections": {},
        }
        with open(snap_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        with self.assertRaises(ValueError) as ctx:
            restore_snapshot("dim_mismatch")
        assert "dimension mismatch" in str(ctx.exception).lower()


class TestNDJSONExportImport(unittest.TestCase):
    """Test per-agent NDJSON export and import round-trip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "graph.db")
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY, entity_id INTEGER, fact_text TEXT,
                source_file TEXT, agent_id TEXT, created_at TEXT,
                retention_class TEXT DEFAULT 'standard',
                is_active INTEGER DEFAULT 1,
                valid_from TEXT, valid_until TEXT, superseded_by INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_chunks (
                rowid INTEGER PRIMARY KEY, qdrant_id TEXT, text TEXT,
                file_path TEXT, chunk_index INTEGER DEFAULT 0,
                agent_id TEXT DEFAULT '', namespace TEXT DEFAULT '',
                date TEXT DEFAULT '', memory_type TEXT DEFAULT 'general'
            )
        """)
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, source_file, agent_id, created_at) "
            "VALUES (1, 'test fact', 'test.md', 'agent-nova', '2025-01-01')"
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ndjson_format(self):
        ndjson_path = os.path.join(self.tmpdir, "test.ndjson")
        records = [
            {
                "id": "p1",
                "collection": "test_coll",
                "vector": [0.1, 0.2, 0.3],
                "payload": {"agent_id": "nova", "text": "hello"},
            },
            {
                "id": "p2",
                "collection": "test_coll",
                "vector": [0.4, 0.5, 0.6],
                "payload": {"agent_id": "nova", "text": "world"},
            },
        ]
        with open(ndjson_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        with open(ndjson_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["id"] == "p1"
        assert lines[0]["vector"] == [0.1, 0.2, 0.3]
        assert lines[1]["payload"]["text"] == "world"

    @patch("backup_manager.collections_for_query")
    def test_export_agent_calls_scroll(self, mock_colls):
        mock_colls.return_value = ["test_memories"]

        mock_point = MagicMock()
        mock_point.id = "point-001"
        mock_point.vector = [0.1, 0.2, 0.3]
        mock_point.payload = {"agent_id": "test-agent", "text": "memory text"}

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([mock_point], None)

        with (
            patch("backup_manager.BACKUP_DIR", self.tmpdir),
            patch("backup_manager.SQLITE_PATH", self.db_path),
            patch("qdrant.qdrant_client", return_value=mock_client),
        ):
            from backup_manager import export_agent

            result = export_agent("test-agent")

        assert result["agent_id"] == "test-agent"
        assert result["points"] == 1
        assert os.path.isfile(result["file"])

        with open(result["file"]) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["id"] == "point-001"
        assert lines[0]["payload"]["text"] == "memory text"

    def test_import_dry_run(self):
        ndjson_path = os.path.join(self.tmpdir, "import.ndjson")
        records = [
            {"id": "p1", "vector": [0.1, 0.2], "payload": {"agent_id": "nova", "text": "hello"}},
            {"id": "p2", "vector": [0.3, 0.4], "payload": {"agent_id": "nova", "text": "world"}},
            {"_table": "memory_chunks", "text": "skip me"},
        ]
        with open(ndjson_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        from backup_manager import import_agent

        result = import_agent(ndjson_path, dry_run=True)

        assert result["imported"] == 2
        assert result["skipped"] == 1
        assert result["dry_run"] is True

    def test_import_missing_file(self):
        from backup_manager import import_agent

        with self.assertRaises(FileNotFoundError):
            import_agent("/nonexistent/path.ndjson")


class TestPrePruneHook(unittest.TestCase):
    """Test that the pre-prune hook debounces correctly."""

    def test_debounce_prevents_rapid_snapshots(self):
        import curator_queue as cq

        cq._last_pre_prune_snapshot = 0.0

        call_count = 0
        original_time = time.time()

        def mock_create_snapshot(label=""):
            nonlocal call_count
            call_count += 1
            return {"snapshot_id": "test"}

        with (
            patch("curator_queue.BACKUP_PRE_PRUNE", True, create=True),
            patch.dict("sys.modules", {"backup_manager": MagicMock()}),
        ):
            import importlib

            importlib.reload(cq)

            with patch("config.BACKUP_PRE_PRUNE", True):
                mock_bm = MagicMock()
                mock_bm.create_snapshot = mock_create_snapshot
                mock_bm.prune_snapshots = MagicMock()

                with patch.dict("sys.modules", {"backup_manager": mock_bm}):
                    cq._last_pre_prune_snapshot = 0.0
                    cq._maybe_pre_prune_snapshot()
                    first_ts = cq._last_pre_prune_snapshot
                    assert first_ts > 0

                    cq._maybe_pre_prune_snapshot()
                    assert cq._last_pre_prune_snapshot == first_ts


class TestBackupMemoryFiles(unittest.TestCase):
    """Test optional memory file tarball backup."""

    def test_tarball_contains_md_files(self):
        tmpdir = tempfile.mkdtemp()
        try:
            mem_dir = os.path.join(tmpdir, "memories")
            os.makedirs(os.path.join(mem_dir, "agents", "nova"))
            with open(os.path.join(mem_dir, "agents", "nova", "note.md"), "w") as f:
                f.write("# Test Note\nSome memory content.")
            with open(os.path.join(mem_dir, "agents", "nova", "skip.txt"), "w") as f:
                f.write("not a markdown file")

            snap_dir = Path(tmpdir) / "snap"
            snap_dir.mkdir()

            with patch("backup_manager.MEMORY_ROOT", mem_dir):
                from backup_manager import _backup_memory_files

                _backup_memory_files(snap_dir)

            tar_path = snap_dir / "memories.tar.gz"
            assert tar_path.is_file()

            import tarfile

            with tarfile.open(str(tar_path), "r:gz") as tar:
                names = tar.getnames()
            assert any("note.md" in n for n in names)
            assert not any("skip.txt" in n for n in names)
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
