"""Shared fixtures for Archivist benchmark suite.

Provides isolated SQLite, mock LLM/Qdrant, and env isolation for all benchmarks.
Data generators live in bench_helpers.py (avoids root conftest name collision).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(autouse=True)
def _bench_env(monkeypatch, tmp_path):
    """Isolate every benchmark from real config and filesystem state."""
    db_path = str(tmp_path / "bench_graph.db")
    mem_root = str(tmp_path / "memories")
    os.makedirs(mem_root, exist_ok=True)

    monkeypatch.setenv("MEMORY_ROOT", mem_root)
    monkeypatch.setenv("SQLITE_PATH", db_path)
    monkeypatch.setenv("NAMESPACES_CONFIG_PATH", str(tmp_path / "ns.yaml"))
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("LLM_URL", "http://localhost:4000")
    monkeypatch.setenv("LLM_API_KEY", "bench-key")
    monkeypatch.setenv("EMBED_URL", "http://localhost:4000")
    monkeypatch.setenv("BM25_ENABLED", "true")
    monkeypatch.setenv("HOT_CACHE_ENABLED", "true")
    monkeypatch.setenv("METRICS_ENABLED", "true")
    monkeypatch.setenv("TRAJECTORY_EXPORT_ENABLED", "true")
    monkeypatch.setenv("HOTNESS_WEIGHT", "0.15")

    import config
    import graph

    monkeypatch.setattr(config, "SQLITE_PATH", db_path)
    monkeypatch.setattr(config, "MEMORY_ROOT", mem_root)
    monkeypatch.setattr(config, "BM25_ENABLED", True)
    monkeypatch.setattr(config, "HOT_CACHE_ENABLED", True)
    monkeypatch.setattr(graph, "SQLITE_PATH", db_path)
    graph.init_schema()

    for mod_name in ("trajectory", "skills", "curator_queue", "retrieval_log", "hotness"):
        try:
            mod = __import__(mod_name)
            if hasattr(mod, "_SCHEMA_APPLIED"):
                monkeypatch.setattr(mod, "_SCHEMA_APPLIED", False)
        except ImportError:
            pass


@pytest.fixture
def graph_db(tmp_path, monkeypatch):
    """Provide an initialized SQLite graph database and return the path."""
    db_path = str(tmp_path / "bench_test_graph.db")
    monkeypatch.setenv("SQLITE_PATH", db_path)

    import importlib
    import config

    importlib.reload(config)
    monkeypatch.setattr("config.SQLITE_PATH", db_path)

    import graph

    monkeypatch.setattr(graph, "SQLITE_PATH", db_path)
    graph.init_schema()
    return db_path
