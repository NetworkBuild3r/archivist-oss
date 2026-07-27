# Quality assurance

Archivist ships three layers of verification: **automated unit and integration tests**, a dedicated **`tests/qa/`** package for transactional storage guarantees, and a **manual MCP/HTTP checklist** for release validation.

## Automated tests

### Default suite

```bash
pip install -r requirements.txt -r requirements-test.txt
python -m pytest tests/ -q --tb=no
```

CI runs this matrix on Python 3.12 and 3.13 with coverage gates; see [`.github/workflows/ci.yml`](https://github.com/NetworkBuild3r/archivist-oss/blob/main/.github/workflows/ci.yml).

### Coach-path evals (INIT-003/SPEC-008 + INIT-004/SPEC-006 CE + INIT-005/SPEC-007)

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

**INIT-005/SPEC-007 perf / lag / durability asserts** (same marker / SQLite path; ADR-005):

| Assert | What it locks |
|--------|----------------|
| Dead-Qdrant / ack SLO | Existing `TestCoachDeadQdrantAck` — store acks with outbox pending; no sync Qdrant wait (INIT-003 spirit / SM-003) |
| Hard-skip observability | Expired ack budget → `store_pipeline.hard_skip` for conflict/dedup (no fact text in log extras); outbox row still enqueued |
| Embed-defer + lag fields | In-test `ARCHIVIST_EMBED_DEFER=true` (default remains **false**) → success JSON `embed_deferred`, `searchable_lag_hint`, `searchable_lag_metric`, `stage_timings.embed_ms` |
| Drain-to-searchable SLO | Deterministic fake `embed_batch` + in-test `drain_outbox` → applied upsert with filled vectors in **≤ 5s** wall clock (ADR-005 CI bar; not live-provider wall-clock) |
| Prior CE contracts | TOC ≤500 / no Key Facts / bootstrap still collected under `-m coach_core` |

```bash
# Focused coach-core suite (CE + INIT-005 perf asserts)
python -m pytest -m coach_core -q --tb=short

# Explicit module path
python -m pytest tests/system/mcp/test_coach_core_evals.py -q --tb=short
```

### Agentic memory evals (INIT-006 / ADR-006)

<!-- INIT-006/SPEC-005 -->

MemoryArena-*inspired* multi-session **Memory→Action** scenarios (not a full
MemoryArena / web-nav port). Sibling pytest marker **`agentic_memory`** — keep
**`-m coach_core` required and green** (ADR-006 GR-COACH-001 / REQ-006). Action
selection is a **test-only oracle** under `tests/system/mcp/agentic_memory_harness.py`
(GR-LAYER-001); no production agent runtime.

**Success metrics locked by these evals:**

| ID | Intent |
|----|--------|
| SM-001 | Omit Session-A store → Session-B action fails or **refuse** (memory is necessary) |
| SM-002 | Stale / contradictory / ambiguous evidence does **not** invent `order_express` |
| SM-003 | `pytest -m coach_core` remains green |

**Scenario inventory (shipped):**

| File | Classes / focus |
|------|-----------------|
| `tests/system/mcp/agentic_memory_harness.py` | Shared harness: store flags, fake-embed patches, `AgenticSession`, `choose_action` |
| `tests/system/mcp/test_agentic_memory_harness_smoke.py` | `TestAgenticMemoryHarnessSmoke` — oracle + SQLite store smoke |
| `tests/system/mcp/test_agentic_memory_positive.py` | `TestAgenticMemoryPositive` — Session A→B `order_express`; omit-store refuse; TOC not evidence; ns isolation |
| `tests/system/mcp/test_agentic_memory_negative.py` | `TestAgenticMemoryNegative` — suppressed stale; eligible+ineligible → `needs_clarification`; ambiguous/empty; index TOC never sufficient |
| `tests/system/mcp/test_agentic_memory_procedure.py` | `TestAgenticMemoryProcedureAction` — procedure tips → action (INIT-007); omit/archived tip refuse |
| `tests/system/mcp/test_agentic_memory_coordination.py` | Tip-via-handoff + share `tip_ids` multi-agent scenarios (INIT-009 Diff #5) |
| `tests/system/mcp/test_agentic_memory_self_curation.py` | Relevance forget → suppress blocks `order_express` (INIT-010 Diff #6) |

**Baselines / contracts:** see [ADR-006](adr/ADR-006-agentic-memory-eval-gym.md). SQLite CI path +
fake embed / stub Qdrant only (GR-EVAL-002). Index markdown alone is never action evidence
(GR-CE-001). Procedure→action tip contracts: [ADR-007](adr/ADR-007-procedural-memory-wedge.md).
Diff #6 self-curation product loop + flags: [ADR-010](adr/ADR-010-intelligent-self-curation.md) /
[REFERENCE.md](REFERENCE.md#intelligent-self-curation-diff-6). Integration coverage:
`tests/integration/lifecycle/test_self_curation_product.py`.

```bash
# Focused agentic-memory suite (Memory→Action + procedure tips)
python -m pytest -m agentic_memory -q --tb=short

# Explicit modules
python -m pytest \
  tests/system/mcp/test_agentic_memory_harness_smoke.py \
  tests/system/mcp/test_agentic_memory_positive.py \
  tests/system/mcp/test_agentic_memory_negative.py \
  tests/system/mcp/test_agentic_memory_procedure.py \
  tests/system/mcp/test_agentic_memory_coordination.py \
  tests/system/mcp/test_agentic_memory_self_curation.py \
  -q --tb=short

# coach_core remains required (run both for full coach+agentic coverage)
python -m pytest -m coach_core -q --tb=short
```

### Procedural tips / procedure evals (INIT-007 / ADR-007)

<!-- INIT-007/SPEC-005 -->

Tips-first procedural memory on the existing trajectory→`tips`→`get_context`
path ([ADR-007](adr/ADR-007-procedural-memory-wedge.md)). **Tip-only lessons** —
the MCP skill registry is **retired** ([ADR-008](adr/ADR-008-retire-skills-tip-lessons.md));
do not use or reintroduce `archivist_*skill*` tools. Cross-agent lesson sharing:
**handoff tips** (primary) and **`archivist_share_*` `tip_ids`** on **ops**/**full**
([ADR-009](adr/ADR-009-native-multi-agent-coordination.md) / INIT-009 Diff #5).

**How tips reach `archivist_get_context` (core read path):**

| Step | Behavior |
|------|----------|
| Storage | SQLite `tips.tip_text` (rows from trajectory extraction / curator / test seed) |
| Surfacing | `include_tips=true` (default) maps **`tip_text`** into response `tips[]` (list of strings) — INIT-007/SPEC-002 |
| Ranking | Non-empty `task_description` → keyword/fingerprint-conditioned recall (not pure recency); empty query → recency fallback — INIT-007/SPEC-003 |
| Usage | Successful conditioned retrieves may bump `usage_count` / `last_used_at` |
| Bootstrap | Fleet tips via wake-up builder (`tip_text`); `include_tips` ignored in bootstrap mode (existing ADR-004 behavior) |

**Core vs ops/full write tools:**

| Profile | Tip-related tools |
|---------|-------------------|
| **core** (default) | **Read** tips via `archivist_get_context` only. No `log_trajectory` / `archivist_tips` in core. |
| **ops** / **full** | `archivist_log_trajectory` (extract tips), `archivist_tips` (direct tip list); **`archivist_share_*`** on ops+full for selective `tip_ids` / memory grants (ADR-009). No skill-registry tools (ADR-008). Checkpoints remain **full**-only. |

CI procedure scenarios **seed** the `tips` table directly in the harness (no ops profile required).

**Procedure→action (SM-003 spirit):** under `-m agentic_memory`,
`TestAgenticMemoryProcedureAction` requires `PROCEDURE_EXPRESS_CUE` in **tips**
from get_context. Memories / index TOC alone must **not** unlock `order_express`
when `require_tip_evidence=True`. Omit tip / archived tip → `refuse`.

**Still required (do not drop):**

```bash
# Procedure scenarios are part of agentic_memory (not a new marker)
python -m pytest -m agentic_memory -q --tb=short

# coach_core remains mandatory (ADR-006 GR-COACH-001 / ADR-007 GR-COACH-001)
python -m pytest -m coach_core -q --tb=short

# Tip mapping + conditioned recall unit/integration (optional focused)
python -m pytest \
  tests/unit/core/test_tip_row_text.py \
  tests/unit/core/test_tip_conditioned_recall.py \
  tests/integration/features/test_trajectory.py \
  -q --tb=short
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

### Coach-path store stage timings + searchable lag (INIT-005/SPEC-002)

ADR-005 / GR-LAG-001 baselines for the write path. Timing payloads are
**numeric / ids / namespace only** — never memory fact text or secrets.

| Stage / signal | Where to read | Assert / observe |
|----------------|---------------|------------------|
| Store embed | `archivist_store` success JSON `stage_timings.embed_ms`; log `store_pipeline.complete` → `stage_timings`; histogram `archivist_store_embed_duration_ms` | Key present on success; numeric `>= 0` |
| Store conflict (optional) | Same `stage_timings.conflict_ms` when conflict check ran; histogram `archivist_store_conflict_duration_ms` | Present only when check executed; numeric `>= 0` |
| Store ack wall | Existing `duration_ms` (INIT-004) | Unchanged |
| Searchable-vector lag | Prometheus gauge `archivist_outbox_lag_seconds` (alias constant `SEARCHABLE_LAG_SECONDS`); refreshed by storage gauges loop from oldest pending outbox age; log field `searchable_lag_metric` names the hook | Gauge `>= 0`; empty queue → `0` |

**SLO spirit (ADR-005):** with fake embed / coach_core, p95 drain-to-searchable ≤ 5s
is asserted under `-m coach_core` (INIT-005/SPEC-007). This section locks the
**instrumentation hooks** so lag is never silent.


```bash
# Hook / contract tests (unit)
python -m pytest tests/unit/core/test_write_observability.py -q --tb=short
```

**Manual / local:** after `archivist_store` with `OUTBOX_ENABLED=true` and
`METRICS_ENABLED=true`, confirm success JSON includes `stage_timings.embed_ms`,
grep `store_pipeline.complete` for `stage_timings` (no fact text), and scrape
`GET /metrics` for `archivist_outbox_lag_seconds` / `archivist_store_embed_duration_ms`.

### Coach-path embed-defer + searchable-lag SLO (INIT-005/SPEC-005)

ADR-005 / GR-LAG-001 / GR-DUR-001. Opt-in via `ARCHIVIST_EMBED_DEFER` (default
**false**). When enabled with `OUTBOX_ENABLED=true`:

| Signal | Where to read | Assert / observe |
|--------|---------------|------------------|
| Defer on store | Success JSON `embed_deferred: true`; log `store_pipeline.embed_deferred` | Primary `embed_text` not awaited on ack path |
| Durable ack | Graph + outbox row (unchanged INIT-003) | FTS/needle usable at ack; vector may lag |
| Drain fill | Outbox drain embeds **before** Qdrant upsert; log `outbox.embed_deferred_filled` (counts/duration only — **no** vectors or fact text) | Point eventually vector-searchable |
| Searchable lag metric | `archivist_outbox_lag_seconds` / alias `SEARCHABLE_LAG_SECONDS` | Hook for lag SLO; never silent |
| Dead Qdrant | Store ack still within INIT-003 spirit (`STORE_ACK_BUDGET_MS` / no sync Qdrant wait) | SM-003 — no regression |

**SLO (ADR-005):** p95 drain-to-searchable **≤ 5s** on fake-embed / coach_core
lanes — hard-asserted in `tests/system/mcp/test_coach_core_evals.py`
(`TestCoachEmbedDeferLagSlo`; INIT-005/SPEC-007). Production real-embed: measure
via lag gauge + “eventually searchable” — do not hard-gate wall-clock against
live providers.

```bash
# Embed-defer unit + drain-fill tests
python -m pytest tests/unit/app/handlers/test_embed_deferred_store.py \
  tests/unit/storage/test_outbox_embed_deferred.py -q --tb=short

# coach_core lag SLO + hard-skip observability (SPEC-007)
python -m pytest -m coach_core -q --tb=short
```

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
