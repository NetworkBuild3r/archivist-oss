"""Tests for config.py startup validation (v1.12)."""

import os

import pytest


def _reload_config(monkeypatch, overrides: dict):
    """Re-import config with overridden environment variables."""
    import sys

    for k, v in overrides.items():
        monkeypatch.setenv(k, str(v))

    if "archivist.core.config" in sys.modules:
        del sys.modules["archivist.core.config"]
    if "config" in sys.modules:
        del sys.modules["config"]

    import archivist.core.config as cfg

    return cfg


class TestConfigDefaults:
    def test_sqlite_wal_autocheckpoint_default(self):
        import archivist.core.config as cfg

        assert int(os.getenv("SQLITE_WAL_AUTOCHECKPOINT", "1000")) == cfg.SQLITE_WAL_AUTOCHECKPOINT

    def test_sqlite_busy_timeout_default(self):
        import archivist.core.config as cfg

        assert int(os.getenv("SQLITE_BUSY_TIMEOUT_MS", "5000")) == cfg.SQLITE_BUSY_TIMEOUT_MS


class TestConfigValidation:
    def test_invalid_chunk_overlap_raises(self, monkeypatch, tmp_path):
        """CHUNK_OVERLAP >= CHUNK_SIZE should fail validation."""
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "graph.db"))
        monkeypatch.setenv("CHUNK_SIZE", "100")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")

        import sys

        for mod in list(sys.modules.keys()):
            if "archivist.core.config" in mod or mod == "config":
                del sys.modules[mod]

        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            pass

    def test_negative_wal_autocheckpoint_raises(self, monkeypatch, tmp_path):
        """Negative SQLITE_WAL_AUTOCHECKPOINT should fail validation."""
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "graph.db"))
        monkeypatch.setenv("SQLITE_WAL_AUTOCHECKPOINT", "-1")

        import sys

        for mod in list(sys.modules.keys()):
            if "archivist.core.config" in mod or mod == "config":
                del sys.modules[mod]

        with pytest.raises(ValueError, match="SQLITE_WAL_AUTOCHECKPOINT"):
            pass

    def test_negative_busy_timeout_raises(self, monkeypatch, tmp_path):
        """Negative SQLITE_BUSY_TIMEOUT_MS should fail validation."""
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "graph.db"))
        monkeypatch.setenv("SQLITE_BUSY_TIMEOUT_MS", "-500")

        import sys

        for mod in list(sys.modules.keys()):
            if "archivist.core.config" in mod or mod == "config":
                del sys.modules[mod]

        with pytest.raises(ValueError, match="SQLITE_BUSY_TIMEOUT_MS"):
            pass
