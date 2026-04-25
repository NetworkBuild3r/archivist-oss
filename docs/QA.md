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

### PostgreSQL integration tests

Two integration test files exercise Postgres-specific behaviour and dual-backend parity. They require a live PostgreSQL database; set `POSTGRES_TEST_DSN` to enable them (otherwise they are skipped automatically).

```bash
# Start Postgres (Docker quickstart)
docker run -d --name pg-test -e POSTGRES_USER=archivist -e POSTGRES_PASSWORD=archivist \
  -e POSTGRES_DB=archivist_test -p 5432:5432 postgres:16-alpine

# Run Postgres-specific integration tests
POSTGRES_TEST_DSN="postgresql://archivist:archivist@localhost:5432/archivist_test" \
  pytest tests/integration/storage/test_postgres_backend.py -v

# Run dual-backend tests (SQLite always, Postgres when DSN set)
POSTGRES_TEST_DSN="postgresql://archivist:archivist@localhost:5432/archivist_test" \
  pytest tests/integration/storage/test_dual_backend.py -v
```

The dual-backend suite validates that `upsert_entity`, `add_fact`, `search_entities`, needle registry, and `fetchval` behave identically on both backends. Unit tests for SQL translation (`_translate_sql`) are always-run in `tests/unit/storage/test_backends.py`.

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

For the transactional boundary, outbox event types, and `conn=` shim pattern, see [`docs/rearchitect_storage_phase3.md`](rearchitect_storage_phase3.md) and [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) §Storage transaction model. For the PostgreSQL backend, schema, and backup mechanics, see [`docs/DOCKER.md`](DOCKER.md) §PostgreSQL backend.
