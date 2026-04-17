# Quality assurance

Archivist ships three layers of verification: **automated unit and integration tests**, a dedicated **`tests/qa/`** package for transactional storage guarantees, and a **manual MCP/HTTP checklist** for release validation.

## Automated tests

### Default suite

```bash
pip install -r requirements.txt -r requirements-test.txt
python -m pytest tests/ -q --tb=no
```

CI runs this matrix on Python 3.12 and 3.13 with coverage gates; see [`.github/workflows/ci.yml`](https://github.com/NetworkBuild3r/archivist-oss/blob/main/.github/workflows/ci.yml).

### QA package (Phase 3 + 3.5)

The `tests/qa/` directory exercises `MemoryTransaction`, the SQLite `outbox` table, `OutboxProcessor`, and fault-injection paths **without** a live Qdrant instance.

```bash
python -m pytest tests/qa/ -q --tb=no
```

Details, markers, and optional chaos-only runs: [`tests/qa/README.md`](../tests/qa/README.md).

### Lint and types (local)

```bash
ruff check . --fix && ruff format .
python -m mypy src/archivist/ --config-file pyproject.toml
```

Mypy uses a ratchet in CI; do not increase the error budget without fixing real issues.

## Manual and fleet QA

- **Operator checklist** — [`QA_CHECKLIST.md`](../QA_CHECKLIST.md): environment, HTTP endpoints, every MCP tool, pipeline stages, RBAC, degradation matrix, sign-off table.
- **Tool schemas** — The checklist appendix includes parameter schemas; regenerate with `PYTHONPATH=src python -m archivist.app.handlers._schema_dump` when tools change.

## Benchmarks and regression

- **In-repo pipeline** — [`benchmarks/README.md`](../benchmarks/README.md) and [`docs/BENCHMARKS.md`](BENCHMARKS.md): reproduction commands, variant definitions, and recorded snapshots.
- **Performance sanity** — See [`QA_CHECKLIST.md`](../QA_CHECKLIST.md) §19; for sustained regression tracking, store harness JSON under `.benchmarks/` (gitignored) and attach paths to release notes.

## Chaos and resilience

Chaos-oriented tests live in `tests/qa/test_chaos_fault_injection.py` (network blips, stuck `processing` rows, concurrent drains). They complement the outbox unit tests in [`tests/test_outbox.py`](../tests/test_outbox.py).

## Storage architecture reference

For the transactional boundary, outbox event types, and `conn=` shim pattern, see [`docs/rearchitect_storage_phase3.md`](rearchitect_storage_phase3.md) and [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) §Storage transaction model.
