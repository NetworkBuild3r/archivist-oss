"""Micro-benchmark: namespace inventory snapshot (SQL aggregation)."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def seeded_inventory_db(tmp_path, monkeypatch):
    import graph
    import namespace_inventory as ni

    db_path = str(tmp_path / "inv.db")
    monkeypatch.setenv("SQLITE_PATH", db_path)
    import importlib
    import config

    importlib.reload(config)
    monkeypatch.setattr(config, "SQLITE_PATH", db_path)
    monkeypatch.setattr(graph, "SQLITE_PATH", db_path)
    graph.init_schema()
    ni.invalidate_all()

    from graph import upsert_fts_chunk

    for i in range(200):
        upsert_fts_chunk(
            f"q-{i}",
            f"chunk {i}",
            f"path/{i}.md",
            i,
            "agent",
            "bench-ns",
            "2026-01-01",
            "skill" if i % 2 == 0 else "experience",
        )
    return "bench-ns"


def test_get_inventory_latency(benchmark, seeded_inventory_db, monkeypatch):
    import namespace_inventory as ni

    monkeypatch.setattr(ni, "INVENTORY_TTL_SECONDS", 0)
    ns = seeded_inventory_db

    def run():
        ni.invalidate(ns)
        ni.get_inventory(ns)

    benchmark(run)
