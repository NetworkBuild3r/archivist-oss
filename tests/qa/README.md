# tests/qa — Archivist QA Test Suite

Local, self-contained QA tests for Archivist Phase 3 + 3.5 guarantees.
All tests run against an in-process SQLite DB with a mocked Qdrant backend.
No external services, no internet access required.

## Suite overview

| File | What it covers |
|------|----------------|
| `conftest.py` | Shared fixtures: `qa_pool`, `mock_vector_backend`, `memory_factory`, `_enable_outbox` autouse |
| `test_storage_transactional.py` | Atomicity, rollback, shim guards, concurrent transactions |
| `test_chaos_fault_injection.py` | Crash simulation, transient/sustained failures, lock contention |
| `test_outbox_full_lifecycle.py` | Full enqueue→drain→retry→dead-letter→prune cycle |
| `test_merge_delete_cascade.py` | Merge error paths, delete cascade, conn-shim graph writes |
| `test_mcp_tools_integrity.py` | Smoke test every MCP storage tool handler |
| `test_performance_regression.py` | Local latency/throughput bounds for the transactional path |
| `test_rbac_and_namespaces.py` | Permission isolation, namespace ACLs, concurrent agents |

## Run commands

```bash
# Full QA suite (fast — all tests, quiet output)
pytest tests/qa/ -q --tb=no

# Full QA suite with short tracebacks
pytest tests/qa/ -q --tb=short

# Chaos + fault-injection tests only
pytest tests/qa/test_chaos_fault_injection.py -v --tb=short -m chaos

# Performance regression benchmarks (prints timing)
pytest tests/qa/test_performance_regression.py -v --tb=short -s

# Single test module
pytest tests/qa/test_storage_transactional.py -v

# Run with coverage (appended to main coverage)
pytest tests/qa/ --cov=src/archivist --cov-append --cov-report=term-missing

# Run only fast tests (skip chaos + perf)
pytest tests/qa/ -m "not chaos" -q --tb=short
```

## Key design decisions

### `OUTBOX_ENABLED=True` by default in this suite

The `_enable_outbox` autouse fixture forces `OUTBOX_ENABLED=True` for every
test, so the transactional outbox path is exercised everywhere.  Individual
tests that want the legacy (disabled) path can override:

```python
def test_legacy_path(monkeypatch):
    import archivist.core.config as _cfg
    monkeypatch.setattr(_cfg, "OUTBOX_ENABLED", False)
    ...
```

### `qa_pool` — isolated per-test SQLite

Each test gets a fresh temp-file SQLite database with the full Archivist
schema applied.  The `archivist.storage.sqlite_pool.pool` module singleton
is patched for the duration of the test so `MemoryTransaction` and
`OutboxProcessor` use the isolated pool.

### No real Qdrant

The `mock_vector_backend` fixture provides a `MagicMock` satisfying the
`VectorBackend` protocol with all methods as `AsyncMock`.  Tests that need
specific backend behaviour override the mock inline.

### Chaos marker

Tests marked `@pytest.mark.chaos` are fault-injection / long-running tests.
Run them selectively:

```bash
pytest tests/qa/ -m chaos -v --tb=short
```

## Prerequisites

Standard project dependencies — no additional packages required beyond what
`pyproject.toml` already lists:

```
aiosqlite
pytest
pytest-asyncio
```

## Thresholds (performance regression)

| Operation | Limit |
|-----------|-------|
| Single `MemoryTransaction` open/close | 50 ms |
| Enqueue 50 events in one transaction | 100 ms |
| Drain 50 pending events (mock backend) | 500 ms |
| 100 sequential writers | 5 000 ms |
| `payload_json()` × 1 000 | 10 ms |
| Full schema DDL | 200 ms |
| 1 000 needle_registry rows insert + read | 200 ms |
| 2 concurrent drains × 100 events | 2 000 ms |

Tighten these thresholds each quarter as the system matures.
