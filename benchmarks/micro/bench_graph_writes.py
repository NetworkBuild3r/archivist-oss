"""Benchmark: graph CRUD — entity/relationship/fact write throughput under lock."""

import sys
import os
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def test_upsert_entity_latency(benchmark):
    """Measure single entity upsert (insert or increment mention_count)."""
    import graph

    i = 0

    def _upsert():
        nonlocal i
        graph.upsert_entity(f"entity-{i}", entity_type="service", agent_id=f"agent-{i % 5}")
        i += 1

    benchmark(_upsert)


def test_upsert_entity_existing(benchmark):
    """Measure upsert on an already-existing entity (UPDATE path)."""
    import graph

    graph.upsert_entity("stable-entity", entity_type="service")

    benchmark(graph.upsert_entity, "stable-entity", entity_type="service")


def test_add_relationship_latency(benchmark):
    """Measure relationship insertion with UPSERT conflict handling."""
    import graph

    src = graph.upsert_entity("source-node", entity_type="service")
    tgt = graph.upsert_entity("target-node", entity_type="service")

    i = 0

    def _add_rel():
        nonlocal i
        graph.add_relationship(src, tgt, f"rel-type-{i % 10}", f"evidence for relation {i}", agent_id="agent-0")
        i += 1

    benchmark(_add_rel)


def test_add_fact_latency(benchmark):
    """Measure fact insertion throughput."""
    import graph

    eid = graph.upsert_entity("fact-entity", entity_type="concept")
    i = 0

    def _add_fact():
        nonlocal i
        graph.add_fact(eid, f"Fact number {i} about deployment pipelines", source_file="bench.md", agent_id="agent-0")
        i += 1

    benchmark(_add_fact)


def test_search_entities_latency(benchmark):
    """Measure entity search after populating 500 entities."""
    import graph

    for i in range(500):
        graph.upsert_entity(f"benchmark-entity-{i:04d}", entity_type="service")

    benchmark(graph.search_entities, "benchmark-entity", limit=10)


def test_concurrent_entity_writes():
    """Measure throughput with N threads competing for GRAPH_WRITE_LOCK."""
    import graph

    n_threads = 8
    ops_per_thread = 50
    results = {"total_ops": 0, "total_time": 0.0}
    lock = threading.Lock()

    def _worker(tid):
        t0 = time.perf_counter()
        for i in range(ops_per_thread):
            graph.upsert_entity(f"thread-{tid}-entity-{i}", entity_type="service")
        elapsed = time.perf_counter() - t0
        with lock:
            results["total_ops"] += ops_per_thread
            results["total_time"] = max(results["total_time"], elapsed)

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(n_threads)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_time = time.perf_counter() - t_start

    total = n_threads * ops_per_thread
    ops_sec = total / wall_time
    assert ops_sec > 0, f"Concurrent writes: {total} ops in {wall_time:.3f}s = {ops_sec:.0f} ops/sec"
