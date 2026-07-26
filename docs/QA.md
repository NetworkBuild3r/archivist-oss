# Quality assurance

Archivist ships three layers of verification: **automated unit and integration tests**, a dedicated **`tests/qa/`** package for transactional storage guarantees, and a **manual MCP/HTTP checklist** for release validation.

## Automated tests

### Default suite

```bash
pip install -r requirements.txt -r requirements-test.txt
python -m pytest tests/ -q --tb=no
```

CI runs this matrix on Python 3.12 and 3.13 with coverage gates; see [`.github/workflows/ci.yml`](https://github.com/NetworkBuild3r/archivist-oss/blob/main/.github/workflows/ci.yml).

### Coach-path evals (INIT-003/SPEC-008 + INIT-004/SPEC-006 CE)

Personal-production coach-path scenarios (store → index → search/`get_context`,
dead-Qdrant store ack, two-namespace isolation) live under `tests/system/mcp/`
with the focused pytest marker `coach_core`. They run on the SQLite CI path
(no live Qdrant; no myaifitness harness). The Integration & System CI job already
collects `tests/system/`, so no dedicated workflow job is required.

**INIT-004/SPEC-006 CE asserts** (same marker / SQLite path; ADR-004):

| Assert | What it locks |
|--------|----------------|
| TOC token ceiling | `archivist_index` markdown ≤ `INDEX_MARKDOWN_TOKEN_CEILING` (500; REFERENCE ~500-token intent) via `token_estimate.markdown_tokens` / `count_tokens` |
| No key-fact prose (GR-CE-001) | Index markdown has no `Key Facts` section and no citable fact sentences from stored prose |
| Bootstrap mode (SM-002) | `archivist_get_context(mode=bootstrap)` returns compact session-start payload under bootstrap budget; empty `memories[]` is success |

```bash
# Focused coach-core suite (includes CE asserts)
python -m pytest -m coach_core -q --tb=short

# Explicit module path
python -m pytest tests/system/mcp/test_coach_core_evals.py -q --tb=short
```

### Coach-path stage timing baselines (INIT-004/SPEC-001)

Measure-before-optimize hooks for the personal-production coach path. Prefer
these existing fields over inventing new instrumentation. No secrets belong in
timing payloads (namespace / counts / durations only).

| Stage | Where to read | Assert / observe |
|-------|---------------|------------------|
| Store ack wall clock | `archivist_store` success JSON `duration_ms`; structured log `store_pipeline.complete` → `duration_ms` (budget: `STORE_ACK_BUDGET_MS` / `ack_budget`) | `isinstance(duration_ms, int)` and `duration_ms >= 0`; warning log `store_pipeline.ack_budget_exceeded` when over budget |
| Search embed | Search/`recursive_retrieve` → `retrieval_trace.stage_timings.embed_ms`; also on `retrieval_pipeline.complete` → `stage_timings` | Key present; numeric `>= 0` |
| Search vector | Same `stage_timings.vector_ms` | Key present; numeric `>= 0` |
| Index rebuild | Structured log `compressed_index.rebuild_complete` → `rebuild_ms`; Prometheus histogram `archivist_index_duration_ms` | Log + metric observed on every `build_namespace_index` / `archivist_index` call (empty namespaces included) |

**How to assert in CI (SQLite path):**

```bash
# Hook / contract tests (unit)
python -m pytest \
  tests/unit/core/test_write_observability.py \
  tests/unit/storage/test_compressed_index_timing.py \
  -q --tb=short

# coach_core includes a store ack duration_ms check
python -m pytest -m coach_core -q --tb=short
```

**Manual / local log inspection:** run a store → index → search turn with
`ARCHIVIST_LOG_LEVEL=INFO` (or the host logging config) and grep:

```text
store_pipeline.complete
compressed_index.rebuild_complete
retrieval_pipeline.complete
```

Confirm `duration_ms` / `rebuild_ms` / `stage_timings.embed_ms` /
`stage_timings.vector_ms` appear. Scrape `GET /metrics` for
`archivist_index_duration_ms` after an `archivist_index` call when
`METRICS_ENABLED=true`.

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

The dual-backend suite validates that `upsert_entity`, `add_fact`, `search_entities`, needle registry, `fetchval`, and `retrieval_log` roundtrips (including token savings columns) behave identically on both backends. Unit tests for SQL translation (`_translate_sql`) are always-run in `tests/unit/storage/test_backends.py`.

### Lint and types (local)

```bash
ruff check . --fix && ruff format .
python -m mypy src/archivist/ --config-file pyproject.toml
```

Mypy uses a hard-fail ratchet in CI (`MYPY_MAX_ERRORS`, currently 175 real
errors, excluding `[import-not-found]`/`[import-untyped]`) — the job fails the
build if the count exceeds the ceiling; it no longer runs with
`continue-on-error`. `[tool.mypy] python_version` is pinned to `3.12` to match
the CI/Docker interpreter floor (INIT-001/SPEC-005) — a lower target previously
made mypy abort after ~3 errors on numpy's PEP 695 stub syntax, silently
disabling the ratchet against the full codebase. Do not raise the ceiling
without fixing real issues; lower it as errors are cleaned up.

Coverage floor (`[tool.coverage.report] fail_under` in `pyproject.toml`) is
49% as of INIT-001/SPEC-005 (up from 46%), based on a measured 53.3%
combined unit+regression+integration+system run. The `test` CI job overrides
this to `--cov-fail-under=0` because it only runs the fast unit+regression
subset (~24% by design); the floor is the full-suite target, raised roughly
2–5 points per quarter as coverage grows.

### Reproduce CI locally (INIT-002)

Use these commands before pushing to a PR. They mirror
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Do **not** treat a
narrow derived-scope pytest run as sufficient for Mode D completion.

| CI job | Local reproduction | INIT-002 owner |
|--------|-------------------|----------------|
| Pre-commit Hooks | `pre-commit run --all-files` | SPEC-002 |
| Lint & Format (Ruff) | `ruff format --check src/ tests/` then `ruff check src/ tests/` | SPEC-002 |
| Type Check (mypy) | Install the **same pinned mypy** as CI (see below), then run the ratchet snippet | SPEC-005 |
| Unit & Regression | `python -m pytest tests/unit tests/regression -q` (CI also runs coverage flags on this job) | SPEC-003, SPEC-004 |
| Integration & System | `python -m pytest tests/integration tests/system -q` | SPEC-004 |
| Chaos | `python -m pytest tests/qa/test_chaos_fault_injection.py -q` | keep green |

**Mypy ratchet (match CI):**

```bash
# Pin must match CI after INIT-002/SPEC-005 (do not `pip install mypy` unpinned).
pip install "mypy==1.19.1" types-PyYAML types-tqdm
OUTPUT=$(python -m mypy src/archivist/ --config-file pyproject.toml 2>&1) || true
echo "$OUTPUT"
COUNT=$(echo "$OUTPUT" | grep "^src/archivist/.*: error:" | grep -v "\[import-not-found\]\|\[import-untyped\]" | wc -l | tr -d ' ' || true)
echo "mypy real errors: $COUNT / ceiling 175"
test "${COUNT:-0}" -le 175
```

**Hard rules:** no new ruff ignores, no raising `MYPY_MAX_ERRORS`, no reintroducing
Phase-5 shims (`import graph`), no flipping `OUTBOX_ENABLED` default back to
false to green tests — fix fixtures/asserts for the durable outbox path.

## Manual and fleet QA

- **Operator checklist** — [`QA_CHECKLIST.md`](../QA_CHECKLIST.md): environment, HTTP endpoints, every MCP tool, pipeline stages, RBAC, degradation matrix, sign-off table.
- **Tool schemas** — The checklist appendix includes parameter schemas; regenerate with `PYTHONPATH=src python -m archivist.app.handlers._schema_dump` when tools change.

## Benchmarks and regression

- **In-repo pipeline** — [`benchmarks/README.md`](../benchmarks/README.md) and [`docs/BENCHMARKS.md`](BENCHMARKS.md): reproduction commands, variant definitions, and recorded snapshots.
- **Token efficiency benchmark** — `benchmarks/token_efficiency.py`: 49 representative queries across 3 packing policies (`adaptive`, `l0_first`, `l2_first`). Run with:
  ```bash
  PYTHONPATH=src python -m benchmarks.token_efficiency --queries 0 --output .benchmarks/token_efficiency_$(date +%Y%m%d).json
  ```
  Output includes per-query savings % and a cross-policy comparison table. Results land in `.benchmarks/` (gitignored).
- **Performance sanity** — See [`QA_CHECKLIST.md`](../QA_CHECKLIST.md) §19; for sustained regression tracking, store harness JSON under `.benchmarks/` (gitignored) and attach paths to release notes.

## Chaos and resilience

Chaos-oriented tests live in `tests/qa/test_chaos_fault_injection.py` (network blips, stuck `processing` rows, concurrent drains). They complement the outbox unit tests in [`tests/test_outbox.py`](../tests/test_outbox.py).

## Answer Finder tests (v2.3)

The v2.3 Answer Finder ships 75 dedicated unit tests across five files:

```bash
python -m pytest \
  tests/unit/retrieval/test_context_packer.py \
  tests/unit/retrieval/test_context_api.py \
  tests/unit/retrieval/test_phase5_observability.py \
  tests/unit/retrieval/test_session_store.py \
  tests/unit/retrieval/test_auto_compress.py \
  -v --tb=short
```

These cover: tier-aware packing policies, `get_relevant_context` + `HandoffPacket` round-trips, `SessionStore` TTL and flush, auto-compress overflow, and the `retrieval_logs` token savings stats pipeline.

## Storage architecture reference

For the transactional boundary, outbox event types, and `conn=` shim pattern, see [`docs/rearchitect_storage_phase3.md`](rearchitect_storage_phase3.md) and [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) §Storage transaction model. For the PostgreSQL backend, schema, and backup mechanics, see [`docs/DOCKER.md`](DOCKER.md) §PostgreSQL backend.
