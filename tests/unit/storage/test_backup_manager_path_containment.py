"""Path-containment tests for backup_manager's snapshot_id handling.

INIT-014/SPEC-001 — RSCH-002 identified that DELETE /admin/backup/{snapshot_id}
and POST /admin/restore built a filesystem path directly from an unvalidated
caller-supplied ``snapshot_id`` with no directory-traversal containment check.
``_snapshot_dir()`` (shared by ``create_snapshot``, ``delete_snapshot``, and
``restore_snapshot``) now resolves the candidate path and rejects it unless it
is a contained descendant of the resolved ``BACKUP_DIR``.

These are pure-logic tests against ``_snapshot_dir()`` / ``delete_snapshot()`` /
``restore_snapshot()`` directly (no HTTP layer) — the full-stack route tests live
in ``tests/integration/storage/test_backup_admin_routes.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]


def _make_manifest(snap_dir: Path, snapshot_id: str) -> None:
    snap_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": 1,
        "snapshot_id": snapshot_id,
        "label": "",
        "created_at": "2026-07-23T00:00:00+00:00",
        "vector_dim": 768,
        "graph_backend": "sqlite",
        "collections": {},
        "sqlite_backed_up": False,
        "postgres_backed_up": False,
        "files_backed_up": False,
        "errors": [],
    }
    with open(snap_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)


class TestSnapshotDirContainment:
    """Unit tests for ``_snapshot_dir()`` in isolation."""

    def setup_method(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        self.patcher = patch("archivist.storage.backup_manager.BACKUP_DIR", self.tmpdir)
        self.patcher.start()

    def teardown_method(self):
        import shutil

        self.patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_legitimate_relative_id_resolves_inside_backup_dir(self):
        from archivist.storage.backup_manager import _snapshot_dir

        resolved = _snapshot_dir("20260723T191500Z_nightly")
        assert resolved == Path(self.tmpdir).resolve() / "20260723T191500Z_nightly"

    def test_dotdot_traversal_rejected(self):
        from archivist.storage.backup_manager import SnapshotPathError, _snapshot_dir

        with pytest.raises(SnapshotPathError):
            _snapshot_dir("../../etc/passwd")

    def test_dotdot_traversal_rejected_even_with_valid_prefix(self):
        from archivist.storage.backup_manager import SnapshotPathError, _snapshot_dir

        with pytest.raises(SnapshotPathError):
            _snapshot_dir("valid_looking_id/../../../etc")

    def test_absolute_path_rejected(self):
        from archivist.storage.backup_manager import SnapshotPathError, _snapshot_dir

        with pytest.raises(SnapshotPathError):
            _snapshot_dir("/etc/passwd")

    def test_backup_dir_itself_rejected(self):
        """An empty-effective traversal that resolves to BACKUP_DIR itself must
        also be rejected — it is never a valid single snapshot."""
        from archivist.storage.backup_manager import SnapshotPathError, _snapshot_dir

        with pytest.raises(SnapshotPathError):
            _snapshot_dir(".")

    def test_empty_snapshot_id_rejected(self):
        from archivist.storage.backup_manager import SnapshotPathError, _snapshot_dir

        with pytest.raises(SnapshotPathError):
            _snapshot_dir("")

    def test_symlink_escape_rejected(self):
        """A pre-existing symlink inside BACKUP_DIR pointing outside it must not
        be usable to escape containment — resolution follows the symlink before
        the containment check runs."""
        import os
        import tempfile

        from archivist.storage.backup_manager import SnapshotPathError, _snapshot_dir

        outside = tempfile.mkdtemp()
        try:
            link_path = Path(self.tmpdir) / "escape_link"
            os.symlink(outside, link_path)
            with pytest.raises(SnapshotPathError):
                _snapshot_dir("escape_link")
        finally:
            import shutil

            shutil.rmtree(outside, ignore_errors=True)


class TestDeleteSnapshotContainment:
    """Unit tests for ``delete_snapshot()``'s use of the containment check."""

    def setup_method(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        self.patcher = patch("archivist.storage.backup_manager.BACKUP_DIR", self.tmpdir)
        self.patcher.start()

    def teardown_method(self):
        import shutil

        self.patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_delete_rejects_traversal_without_touching_filesystem(self):
        """A malicious snapshot_id must be rejected before shutil.rmtree() runs —
        assert via a spy that rmtree is never called."""
        from archivist.storage.backup_manager import SnapshotPathError, delete_snapshot

        with patch("archivist.storage.backup_manager.shutil.rmtree") as mock_rmtree:
            with pytest.raises(SnapshotPathError):
                delete_snapshot("../../etc")
            mock_rmtree.assert_not_called()

    def test_delete_rejects_absolute_path_without_touching_filesystem(self):
        from archivist.storage.backup_manager import SnapshotPathError, delete_snapshot

        with patch("archivist.storage.backup_manager.shutil.rmtree") as mock_rmtree:
            with pytest.raises(SnapshotPathError):
                delete_snapshot("/etc/passwd")
            mock_rmtree.assert_not_called()

    def test_delete_legitimate_snapshot_still_succeeds(self):
        """Positive-path regression test: a real, non-traversal snapshot_id must
        still delete successfully (ac-5)."""
        from archivist.storage.backup_manager import delete_snapshot

        snap_dir = Path(self.tmpdir) / "20260723T191500Z_nightly"
        _make_manifest(snap_dir, "20260723T191500Z_nightly")
        assert snap_dir.is_dir()

        assert delete_snapshot("20260723T191500Z_nightly") is True
        assert not snap_dir.is_dir()


class TestRestoreSnapshotContainment:
    """Unit tests for ``restore_snapshot()``'s use of the containment check."""

    def setup_method(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        self.patcher = patch("archivist.storage.backup_manager.BACKUP_DIR", self.tmpdir)
        self.patcher.start()

    def teardown_method(self):
        import shutil

        self.patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_restore_rejects_traversal_before_reading_manifest(self):
        """A malicious snapshot_id must never reach the manifest.json read."""
        from archivist.storage.backup_manager import SnapshotPathError, restore_snapshot

        with pytest.raises(SnapshotPathError):
            restore_snapshot("../../etc/passwd")

    def test_restore_rejects_absolute_path(self):
        from archivist.storage.backup_manager import SnapshotPathError, restore_snapshot

        with pytest.raises(SnapshotPathError):
            restore_snapshot("/etc/passwd")

    def test_restore_missing_snapshot_still_raises_file_not_found(self):
        """Non-traversal but nonexistent snapshot_id must still 404 (not 4xx from
        the new containment check) — existing behavior is preserved."""
        from archivist.storage.backup_manager import restore_snapshot

        with pytest.raises(FileNotFoundError):
            restore_snapshot("nonexistent_but_well_formed_id")
