"""Adversarial ReDoS performance tests for SPEC-010 re2 Memory Firewall."""

from __future__ import annotations

import time

import numpy as np
import pytest

from archivist.security.firewall import MemoryFirewall

pytestmark = [pytest.mark.performance]


def test_firewall_100kb_redos_payloads_under_5ms_p95():
    """100KB pathological inputs must complete in <5ms p95 under linear-time RE2."""
    fw = MemoryFirewall()
    payloads = [
        "a" * 100_000,
        ("api_key=" + "a" * 10_000 + "\n") * 10,
        "Bearer " + ("A" * 50_000) + " " + ("!" * 50_000),
        "-----BEGIN PRIVATE KEY-----" + ("A" * 100_000),
        "ignore " + ("all " * 25_000) + "previous instructions",
        "eyJ" + ("a" * 33_000) + ".eyJ" + ("b" * 33_000) + "." + ("c" * 33_000),
        ("hello \u202e world " * 8_000),
        ("postgresql://user:" + ("x" * 10_000) + "@host:5432/db\n") * 5,
    ]

    # Warm up DFA states once
    for p in payloads:
        fw.scan(p)

    latencies_ms: list[float] = []
    for _ in range(10):
        for p in payloads:
            t0 = time.perf_counter()
            res = fw.scan(p)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            assert res.allowed is True

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    assert p95 < 5.0, f"Memory Firewall 100KB ReDoS p95={p95:.3f}ms (p50={p50:.3f}ms) exceeded 5ms"
