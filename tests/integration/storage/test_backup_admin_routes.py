"""Full-stack tests for the admin backup/restore routes' path-containment fix.

INIT-014/SPEC-001 (ac-3, ac-4, ac-5): ``DELETE /admin/backup/{snapshot_id}`` and
``POST /admin/restore`` must reject any ``snapshot_id`` that would resolve outside
``BACKUP_DIR`` with a 4xx error, and must not touch the filesystem for such an ID —
while a legitimate, non-traversal ``snapshot_id`` must still succeed end-to-end.

Unit-level tests for the containment logic itself live in
``tests/unit/storage/test_backup_manager_path_containment.py``.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

pytestmark = [pytest.mark.integration, pytest.mark.storage]


async def _noop_startup():
    """Avoid Qdrant/SQLite/background-task init — only the HTTP routing + backup
    manager under test are exercised."""
    return None


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


@pytest.fixture
def backup_dir(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setattr("backup_manager.BACKUP_DIR", tmpdir)
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def client(monkeypatch):
    import main

    monkeypatch.setattr(main, "_startup", _noop_startup)
    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "", raising=False)
    with TestClient(main.app) as c:
        yield c


class TestDeleteBackupRouteContainment:
    def test_rejects_dotdot_traversal_with_4xx_and_no_rmtree(self, client, backup_dir):
        """``%2e%2e`` (percent-encoded ``..``) survives client-side URL normalization
        and is decoded by Starlette to snapshot_id='..' before reaching the handler
        — a real single-path-segment traversal vector for this route. A literal
        unencoded '..' segment is collapsed by URL normalization before the request
        is even sent, so it never reaches the server at all; the encoded form is
        what an actual attacker would send."""
        with patch("backup_manager.shutil.rmtree") as mock_rmtree:
            resp = client.delete("/admin/backup/%2e%2e")
            assert 400 <= resp.status_code < 500, resp.text
            mock_rmtree.assert_not_called()

    def test_multi_segment_traversal_rejected_at_routing_layer(self, client, backup_dir):
        """A path containing an encoded '/' (e.g. an absolute-path attempt) does not
        even match the single-segment {snapshot_id} route — Starlette's routing
        itself returns 404 before our handler runs. This is defense-in-depth on top
        of the explicit containment check exercised by the other tests here; still
        asserted to be a 4xx (never a 5xx or a successful delete)."""
        with patch("backup_manager.shutil.rmtree") as mock_rmtree:
            resp = client.delete("/admin/backup/%2e%2e%2f%2e%2e%2fetc")
            assert 400 <= resp.status_code < 500, resp.text
            mock_rmtree.assert_not_called()

    def test_legitimate_snapshot_id_still_deletes(self, client, backup_dir):
        """Positive-path regression test (ac-5)."""
        snap_dir = backup_dir / "20260723T191500Z_nightly"
        _make_manifest(snap_dir, "20260723T191500Z_nightly")
        assert snap_dir.is_dir()

        resp = client.delete("/admin/backup/20260723T191500Z_nightly")
        assert resp.status_code == 200, resp.text
        assert resp.json()["deleted"] == "20260723T191500Z_nightly"
        assert not snap_dir.is_dir()


class TestRestoreRouteContainment:
    def test_rejects_dotdot_traversal_with_4xx(self, client, backup_dir):
        resp = client.post("/admin/restore", json={"snapshot_id": "../../etc/passwd"})
        assert 400 <= resp.status_code < 500, resp.text

    def test_rejects_absolute_path_with_4xx(self, client, backup_dir):
        resp = client.post("/admin/restore", json={"snapshot_id": "/etc/passwd"})
        assert 400 <= resp.status_code < 500, resp.text

    def test_traversal_id_never_reaches_manifest_read(self, client, backup_dir):
        """The rejection must happen before any manifest.json read is attempted."""
        with patch("backup_manager.json.load") as mock_json_load:
            resp = client.post("/admin/restore", json={"snapshot_id": "../../etc/passwd"})
            assert 400 <= resp.status_code < 500, resp.text
            mock_json_load.assert_not_called()

    def test_legitimate_snapshot_id_reaches_manifest_read(self, client, backup_dir, monkeypatch):
        """Positive-path regression test (ac-5): a real, non-traversal snapshot_id
        must still be resolved and its manifest read — restore proceeds past the
        containment check into the real restore logic (here there is no graph.db
        file in the snapshot, so it reports that as a per-component error in a
        200 response rather than being blocked by the security check)."""
        monkeypatch.setattr("backup_manager.VECTOR_DIM", 768)
        snap_dir = backup_dir / "20260723T191500Z_nightly"
        _make_manifest(snap_dir, "20260723T191500Z_nightly")

        resp = client.post("/admin/restore", json={"snapshot_id": "20260723T191500Z_nightly"})
        assert resp.status_code == 200, resp.text
        assert any("SQLite backup file not found" in e for e in resp.json()["errors"])

    def test_missing_but_well_formed_snapshot_still_404s(self, client, backup_dir):
        """Non-traversal but nonexistent snapshot_id keeps its existing 404 —
        confirms the new containment check does not change unrelated behavior."""
        resp = client.post("/admin/restore", json={"snapshot_id": "nonexistent_snapshot"})
        assert resp.status_code == 404
