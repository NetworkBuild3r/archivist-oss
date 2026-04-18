# Archivist-OSS — Testing Architecture Plan

> **Status**: PROPOSAL — awaiting approval before any code changes.
> **Scope**: Analysis + re-architecture plan only. Zero test files are moved or modified here.

---

## 1. Current State: Audit & Problems

### 1.1 Numbers at a Glance

| Metric | Value |
|--------|-------|
| Total test files | 52 (40 flat + 12 in `tests/qa/`) |
| Total collected tests | 835 |
| Lines of test code | ~11,000 in `tests/test_*.py` alone |
| Markers actually used | `chaos` (11), `asyncio` (75 explicit, rest auto), `integration` (1), `parametrize` (7) |
| conftest.py locations | 2 (`conftest.py` root + `tests/qa/conftest.py`) |
| Explicit `sys.path.insert` hacks | 8 files |

### 1.2 Critical Structural Problems

**Problem 1: Flat chaos — 40 files dumped in one directory with no domain grouping**

```
tests/
  test_background_tasks.py      # infrastructure
  test_backup.py                # storage
  test_chunk1_dry.py            # write pipeline
  test_chunk2_cascade.py        # lifecycle
  test_delete_cascade.py        # lifecycle (overlaps chunk2!)
  test_graph.py                 # storage
  test_new_modules.py           # 14 unrelated classes (1,386 lines!)
  test_phase2_tiered_context.py # retrieval
  test_phase7_curator_intelligence.py  # curator
  test_sprint1_needle.py        # retrieval
  ...
```

There is no way to run "all storage tests" or "all retrieval tests" without grep-hacking file names.

**Problem 2: Phase/Sprint/Chunk naming encodes temporal history, not domain**

Files named `test_phase1_foundations.py`, `test_chunk3_write_path.py`, `test_sprint2_needle.py` describe *when* the code was written, not *what* is being tested. A new engineer cannot discover what is tested where.

**Problem 3: `test_new_modules.py` is a 1,386-line omnibus with 14 unrelated test classes**

```python
class TestTokenizer:          # utils/tokenizer
class TestContextManager:     # utils/context_manager
class TestFTSSearch:          # storage/fts_search
class TestCuratorChecksum:    # core/curator
class TestReranker:           # retrieval/reranker
class TestBenchmarkMetrics:   # benchmarks
class TestPhase3NominateThenRerank:  # retrieval pipeline
class TestGetReferenceDocs:   # MCP/tools_docs
```

A failure in any one class forces reading 1,386 lines to locate it.

**Problem 4: Cascading overlap — two files cover the same delete/cascade module**

`test_chunk2_cascade.py` (63 tests) and `test_delete_cascade.py` (58 tests) both test `cascade.py` and `memory_lifecycle.py`. There is no canonical home for cascade tests, and coverage gaps are invisible.

**Problem 5: 69 explicit `@pytest.mark.asyncio` decorators are dead weight**

`asyncio_mode = "auto"` in `pyproject.toml` means these are no-ops. They add noise and will break if mode is ever switched.

**Problem 6: 8 files use `sys.path.insert` hacks instead of the configured `pythonpath`**

```python
# tests/test_backup.py:13
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
```

This works accidentally because `pythonpath = ["src"]` in `pyproject.toml` already adds `src/` to path. The hacks are fragile and will break on path changes.

**Problem 7: Two import namespaces used inconsistently**

Some tests import `from graph import upsert_entity` (flat `src/` imports), others import `from archivist.storage.sqlite_pool import SQLitePool` (package imports). This inconsistency signals unclear src layout and will cause import confusion as the package matures.

**Problem 8: `tests/qa/` is a partial solution that introduces a second fixture system**

`tests/qa/conftest.py` duplicates the full DDL schema (850+ lines) because it cannot reuse the root `conftest.py`. The two fixture systems diverge silently.

**Problem 9: No marker-based fast/slow separation**

Running `pytest -m "not slow"` currently selects 835 tests (same as full run) because almost nothing is marked `slow`. The full suite takes ~120 seconds. Engineers have no way to get a fast feedback loop during development.

**Problem 10: Performance tests mixed with unit tests**

`tests/qa/test_performance_regression.py` (timing assertions like "under 500ms") and `tests/test_sprint3_enterprise.py::TestLatencyBudget` live alongside pure unit tests. A flapping performance test breaks unrelated CI.

**Problem 11: `test_code_quality.py` runs `ruff` as a subprocess inside pytest**

```python
def test_ruff_format_check_src():
    result = _run([sys.executable, "-m", "ruff", "format", "--check", "src/"])
```

This is a pre-commit hook concern, not a pytest test. It runs slowly, produces confusing output, and is already enforced by CI's `ruff` pre-commit step.


---

## 2. Proposed Directory Structure

```
tests/
├── conftest.py                     # ROOT: shared fixtures, isolation, factories
│
├── unit/                           # Pure logic, zero I/O — fastest tier
│   ├── conftest.py                 # Unit-only fixtures (none currently needed)
│   ├── storage/
│   │   ├── test_chunking.py        # chunking.py patterns, NEEDLE_PATTERNS DRY
│   │   ├── test_graph_sync.py      # sync graph helpers (non-DB)
│   │   ├── test_result_types.py    # ResultCandidate dataclass + factory methods
│   │   └── test_fts_quality.py     # strip_augmentation_header, FTS5 text handling
│   ├── retrieval/
│   │   ├── test_rank_fusion.py     # RRF algorithm, BM25 modes
│   │   ├── test_tiering.py         # select_tier, tiered context
│   │   ├── test_threshold.py       # threshold filtering
│   │   ├── test_reranker.py        # reranker module
│   │   └── test_graph_retrieval.py # contradictions, entity_brief (mocked DB)
│   ├── core/
│   │   ├── test_config.py          # config validation, feature flag logging
│   │   ├── test_tokenizer.py       # count_tokens, count_message_tokens
│   │   ├── test_context_manager.py # check_context, check_memories_budget
│   │   ├── test_retry.py           # retry decorator, backoff logic
│   │   ├── test_rbac.py            # check_access, namespace resolution
│   │   └── test_provenance.py      # SourceTrace, provenance propagation
│   └── write/
│       ├── test_write_pipeline.py  # write path logic (mocked DB/Qdrant)
│       ├── test_hyde.py            # HyDE cache key, config reads
│       └── test_contextual_augment.py  # augmentation header format/strip
│
├── integration/                    # Real SQLite, mocked Qdrant/LLM
│   ├── conftest.py                 # Integration fixtures: graph_db, async_pool, rbac_config
│   ├── storage/
│   │   ├── test_graph.py           # entity/fact CRUD with real SQLite
│   │   ├── test_sqlite_pool.py     # pool lifecycle, concurrent writes
│   │   ├── test_fts_search.py      # FTS5 queries against real DB
│   │   ├── test_backup.py          # snapshot create/restore/export/import
│   │   └── test_outbox.py          # outbox enqueue, drain, idempotency
│   ├── lifecycle/
│   │   ├── test_delete_cascade.py  # delete_memory_complete, all artifact types
│   │   ├── test_archive_cascade.py # archive_memory_complete, FTS exclusion
│   │   ├── test_merge.py           # merge_memories, outbox events
│   │   └── test_compaction.py      # compaction thresholds, dedup
│   ├── retrieval/
│   │   ├── test_retrieval_pipeline.py  # recursive_retrieve end-to-end (mocked Qdrant)
│   │   ├── test_parent_child.py    # hierarchical chunking, parent text injection
│   │   ├── test_needle_registry.py # needle token store/lookup/isolation
│   │   └── test_synthetic_questions.py # synthetic Q generation + quality
│   ├── mcp/
│   │   ├── test_dispatch.py        # MCP registry, ALL_TOOLS, dispatch errors
│   │   ├── test_http_transport.py  # HTTP transport wiring, SSE compat
│   │   └── test_storage_handlers.py # tools_storage handlers (store/merge/pin/delete)
│   └── features/
│       ├── test_trajectory.py      # trajectory log, annotations, ratings, tips
│       ├── test_skills.py          # skill register, events, health, lessons
│       ├── test_curator.py         # curator_queue scheduling, hotness scoring
│       └── test_background_tasks.py # task tracking, exception logging
│
├── system/                         # Full handler coverage across all 37 MCP tools
│   ├── conftest.py                 # System-level fixtures (pool + mocked external)
│   └── mcp/
│       ├── test_smoke_all_handlers.py  # ← current tests/qa/test_mcp_handler_smoke.py
│       ├── test_integrity_storage.py   # ← current tests/qa/test_mcp_tools_integrity.py
│       └── test_namespace_rbac.py      # ← current tests/qa/test_rbac_and_namespaces.py
│
├── chaos/                          # Fault injection, concurrency, adversarial
│   ├── conftest.py
│   ├── test_fault_injection.py     # ← current tests/qa/test_chaos_fault_injection.py
│   ├── test_write_lock_contract.py # ← current tests/qa/test_write_lock_contract.py
│   └── test_outbox_chaos.py        # chaos scenarios from current tests/test_outbox.py
│
├── performance/                    # Timing assertions — isolated from unit/integration
│   ├── conftest.py
│   └── test_regression.py          # ← current tests/qa/test_performance_regression.py
│
├── regression/                     # Specific bug regression guards (named by issue)
│   ├── test_missing_awaits.py      # ← current tests/test_missing_awaits.py
│   ├── test_unawaited_graph_calls.py   # ← current tests/qa regression guards
│   └── test_merge_consistency.py   # ← current tests/test_merge_consistency.py
│
└── fixtures/                       # Shared factories and helpers (NOT test files)
    ├── __init__.py
    ├── factories.py                # memory_factory, entity_factory, trajectory_factory
    ├── schema.py                   # _SCHEMA_SQL, _build_schema() — single source
    └── mocks.py                    # _mock_embed, mock_qdrant_client, mock_llm_response
```

### 2.1 Key Principles

1. **Domain over chronology** — directories are named after the system domain (`storage/`, `lifecycle/`, `retrieval/`), never after development phases (`phase3/`, `sprint2/`).
2. **One conftest per tier** — each tier (`unit/`, `integration/`, `system/`, `chaos/`, `performance/`) gets its own `conftest.py` for tier-specific fixtures. All tiers inherit from the root `conftest.py`.
3. **`fixtures/` is not a test directory** — `factories.py`, `schema.py`, `mocks.py` are pure Python helpers imported by tests. pytest does not collect them.
4. **Flat within domain** — within each domain directory, files are flat (not nested further). `integration/storage/test_graph.py` not `integration/storage/graph/test_entity_operations.py`.


---

## 3. Marker Strategy

### 3.1 Full Marker Set

```toml
# pyproject.toml — [tool.pytest.ini_options]
markers = [
    # Speed / tier
    "unit: pure logic tests, no I/O (fastest — run on every save)",
    "integration: real SQLite + mocked external services",
    "system: full handler-level coverage across all MCP tools",
    "chaos: fault injection, adversarial, concurrent stress tests",
    "performance: timing assertions (run isolated, not in main CI pass)",
    "regression: specific bug regression guards",
    # Domain
    "storage: tests touching SQLite, Qdrant, outbox, or backup",
    "lifecycle: tests for delete/archive/merge/cascade paths",
    "retrieval: tests for search, ranking, graph retrieval, synthesis",
    "mcp: tests for MCP tool handlers and transport layer",
    "rbac: tests for namespace access control",
    # Infra
    "slow: tests taking >2 seconds (deselect during development)",
    "integration: requires external services (Qdrant, real LLM)",
]
```

### 3.2 Marker Application Rules

| File pattern | Required markers | Optional |
|---|---|---|
| `unit/**/*.py` | `@pytest.mark.unit` | domain marker |
| `integration/**/*.py` | `@pytest.mark.integration` | domain marker |
| `system/**/*.py` | `@pytest.mark.system` | `mcp`, `rbac` |
| `chaos/**/*.py` | `@pytest.mark.chaos` | `slow` |
| `performance/**/*.py` | `@pytest.mark.performance`, `@pytest.mark.slow` | domain |
| `regression/**/*.py` | `@pytest.mark.regression` | domain |

**Rule**: apply markers at the **class or module level** where all tests share the same tier. Apply at function level only for exceptions.

```python
# Good — module-level marker, applies to all tests in file
pytestmark = [pytest.mark.unit, pytest.mark.storage]

class TestFtsQuality:
    def test_strip_header_basic(self): ...
    def test_strip_header_missing_delimiter(self): ...
```

```python
# Good — class-level marker for mixed files
class TestBackupRestore:
    pytestmark = pytest.mark.integration  # all tests here are integration

class TestBackupPathValidation:
    pytestmark = pytest.mark.unit         # pure logic, no DB
```

### 3.3 Drop `@pytest.mark.asyncio` everywhere

`asyncio_mode = "auto"` already handles all `async def test_*` functions. The 75 explicit decorators in `tests/test_*.py` are removed during migration as each file is touched. No new tests should ever add `@pytest.mark.asyncio`.

---

## 4. pyproject.toml Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"

markers = [
    "unit: pure logic tests, no I/O",
    "integration: real SQLite, mocked Qdrant/LLM",
    "system: full MCP handler coverage",
    "chaos: fault injection and concurrency stress",
    "performance: timing regression assertions",
    "regression: specific bug regression guards",
    "storage: storage layer tests",
    "lifecycle: delete/archive/merge/cascade tests",
    "retrieval: search and ranking tests",
    "mcp: MCP tool handler tests",
    "rbac: access control tests",
    "slow: tests taking more than 2 seconds",
]

addopts = [
    "--strict-markers",
    "--tb=short",
    "-q",
]

# Coverage (when run with --cov):
# [tool.coverage.run]
# source = ["src"]
# omit = ["src/main.py", "*/migrations/*"]
# [tool.coverage.report]
# fail_under = 70
# show_missing = true
```

### 4.1 Removed from pyproject.toml

- `bench_*.py` removed from `python_files` — benchmarks live in `benchmarks/`, not `tests/`
- `"chaos: marks chaos / fault-injection tests"` updated to full new marker list

---

## 5. Migration Strategy

### 5.1 Principles

1. **No test behavior changes during migration** — we move files and add markers; we do not refactor test logic.
2. **One PR per domain group** — migrate storage tests, get CI green, then lifecycle, then retrieval, etc.
3. **Keep old files until new file CI-verified** — rename/move atomically, never leave a gap.
4. **Update `conftest.py` imports** when fixtures move to tier-specific conftests.

### 5.2 Migration Map — All 40 Flat Test Files

| Current file | Destination | Notes |
|---|---|---|
| `test_chunk1_dry.py` | `unit/storage/test_chunking.py` + `unit/write/test_write_pipeline.py` | Split by class: NeedlePatterns → chunking; HyDE/RlmRetriever → write pipeline |
| `test_chunk2_cascade.py` | `integration/lifecycle/test_delete_cascade.py` | Consolidate with `test_delete_cascade.py` |
| `test_chunk3_write_path.py` | `integration/mcp/test_storage_handlers.py` | MCP handler write path tests |
| `test_chunk4_result_types.py` | `unit/storage/test_result_types.py` | Pure dataclass tests |
| `test_chunk5_fts_quality.py` | `unit/storage/test_fts_quality.py` | `strip_augmentation_header` is pure logic |
| `test_chunk6_observability.py` | `unit/core/test_config.py` (config logging) + `unit/core/test_write_observability.py` | Split by class |
| `test_config.py` | `unit/core/test_config.py` | Merge with chunk6 config tests |
| `test_delete_cascade.py` | `integration/lifecycle/test_delete_cascade.py` | Merge with chunk2 cascade tests |
| `test_dispatch.py` | `integration/mcp/test_dispatch.py` | MCP registry tests |
| `test_graph.py` | `integration/storage/test_graph.py` | Real SQLite, keep async_pool fixture |
| `test_mcp_http_transport.py` | `integration/mcp/test_http_transport.py` | Transport wiring |
| `test_memory_awareness.py` | `integration/retrieval/test_namespace_inventory.py` | Query classifier, namespace inventory |
| `test_memory_fusion.py` | `integration/lifecycle/test_merge.py` | Merge with merge_consistency |
| `test_merge_consistency.py` | `regression/test_merge_consistency.py` | Regression-specific |
| `test_metrics_v111.py` | `integration/features/test_metrics.py` | Observability/metrics |
| `test_missing_awaits.py` | `regression/test_missing_awaits.py` | Explicit regression file |
| `test_new_modules.py` | **Split into 8 files** (see §5.3) | Largest migration |
| `test_observability_v110.py` | `integration/features/test_observability.py` | |
| `test_outbox.py` | `integration/storage/test_outbox.py` | Core outbox; chaos scenarios → chaos/ |
| `test_parent_child.py` | `integration/retrieval/test_parent_child.py` | |
| `test_phase1_foundations.py` | `unit/core/test_rbac.py` + `unit/retrieval/test_retrieval_trace.py` | Split |
| `test_phase2_tiered_context.py` | `unit/retrieval/test_tiering.py` + `unit/retrieval/test_graph_retrieval.py` | |
| `test_phase3_trajectory.py` | `integration/features/test_trajectory.py` | |
| `test_phase4_skills.py` | `integration/features/test_skills.py` | |
| `test_phase5_memory_arch.py` | `integration/features/test_hot_cache.py` + `unit/core/test_archivist_uri.py` | Split |
| `test_phase6_observability.py` | `integration/features/test_observability.py` | Merge with metrics |
| `test_phase7_curator_intelligence.py` | `integration/features/test_curator.py` | |
| `test_provenance.py` | `unit/core/test_provenance.py` | Pure SourceTrace tests |
| `test_provenance_integration.py` | `integration/mcp/test_storage_handlers.py` | Provenance through full handler pipeline |
| `test_rbac.py` | `unit/core/test_rbac.py` | Pure RBAC logic |
| `test_rerank.py` | `unit/retrieval/test_reranker.py` | |
| `test_retry_utils.py` | `unit/core/test_retry.py` | |
| `test_sprint1_needle.py` | `unit/retrieval/test_rank_fusion.py` (RRF, BM25) + `integration/retrieval/test_needle_registry.py` | Split |
| `test_sprint2_needle.py` | `unit/write/test_hyde.py` + `unit/write/test_contextual_augment.py` + `integration/retrieval/test_retrieval_pipeline.py` | Split |
| `test_sprint3_enterprise.py` | `unit/storage/test_collection_router.py` + `unit/retrieval/test_latency_budget.py` + `integration/storage/test_cache_backend.py` + `performance/test_regression.py` | Split |
| `test_sqlite_pool.py` | `integration/storage/test_sqlite_pool.py` | |
| `test_synthetic_questions.py` | `integration/retrieval/test_synthetic_questions.py` | |
| `test_threshold.py` | `unit/retrieval/test_threshold.py` | |
| `test_background_tasks.py` | `integration/features/test_background_tasks.py` | |
| `test_backup.py` | `integration/storage/test_backup.py` | |

### 5.3 `test_new_modules.py` — Split Plan (14 classes → 8 files)

| Class(es) | Destination |
|---|---|
| `TestTokenizer`, `TestContextManager` | `unit/core/test_tokenizer.py`, `unit/core/test_context_manager.py` |
| `TestFTSSearch` | `unit/storage/test_fts_quality.py` (merge with chunk5) |
| `TestEntityExtraction` | `unit/write/test_pre_extractor.py` |
| `TestCuratorChecksum` | `unit/core/test_curator_checksum.py` |
| `TestReranker`, `TestBenchmarkMetrics` | `unit/retrieval/test_reranker.py`, `unit/retrieval/test_benchmark_metrics.py` |
| `TestPhase1QueryExpansionKill` | `unit/retrieval/test_query_expansion.py` |
| `TestPhase2SyntheticQuestionPipeline` through `TestPhase5SemanticChunking` | `integration/retrieval/test_retrieval_pipeline.py` |
| `TestGetReferenceDocs` | `integration/mcp/test_dispatch.py` (merge with docs handler) |

### 5.4 `tests/qa/` — Migration Map

| Current file | Destination |
|---|---|
| `conftest.py` | Merged into `tests/conftest.py` (schema DDL) + `tests/fixtures/schema.py` |
| `test_chaos_fault_injection.py` | `chaos/test_fault_injection.py` |
| `test_code_quality.py` | **Deleted** — enforced by pre-commit, not pytest |
| `test_mcp_handler_smoke.py` | `system/mcp/test_smoke_all_handlers.py` |
| `test_mcp_tools_integrity.py` | `system/mcp/test_integrity_storage.py` |
| `test_merge_delete_cascade.py` | `integration/lifecycle/test_delete_cascade.py` (merge) |
| `test_outbox_full_lifecycle.py` | `integration/storage/test_outbox.py` (merge) |
| `test_performance_regression.py` | `performance/test_regression.py` |
| `test_rbac_and_namespaces.py` | `system/mcp/test_namespace_rbac.py` |
| `test_storage_transactional.py` | `integration/storage/test_sqlite_pool.py` (merge) |
| `test_write_lock_contract.py` | `chaos/test_write_lock_contract.py` |


---

## 6. Fixture Strategy

### 6.1 Root `conftest.py` — Shared by All Tiers

The root `conftest.py` (at project root, sibling to `src/`) keeps:

- `_isolate_env` (autouse) — temp dir, env vars, schema reset
- `async_pool` — async SQLitePool fixture for graph tests
- `graph_db` — sync SQLite path for legacy tests
- `rbac_config` — writes test `namespaces.yaml`
- `mock_llm` — patches `llm.llm_query` with `AsyncMock`

**No changes to these fixtures during migration.** They are the stable base.

### 6.2 New: `tests/fixtures/schema.py` — Single Schema Source

The DDL that is currently duplicated between `conftest.py` (root) and `tests/qa/conftest.py` is extracted to:

```python
# tests/fixtures/schema.py
_SCHEMA_SQL: str = """..."""   # full DDL, single source of truth
_FTS5_SQL: list[str] = [...]   # FTS5 virtual tables

async def build_schema(conn) -> None:
    """Execute full schema DDL on an aiosqlite connection."""
    await conn.executescript(_SCHEMA_SQL)
    for stmt in _FTS5_SQL:
        await conn.executescript(stmt)
```

Both `conftest.py` and `tests/qa/conftest.py` replacement fixtures import from here.

### 6.3 New: `tests/fixtures/factories.py` — Test Data Factories

```python
# tests/fixtures/factories.py
import uuid
from typing import Any

def memory_factory(
    text: str | None = None,
    agent_id: str = "agent-test",
    namespace: str = "test-ns",
    **overrides: Any,
) -> dict:
    """Return a minimal valid memory payload dict."""
    return {
        "text": text or f"Test memory {uuid.uuid4().hex[:8]}",
        "agent_id": agent_id,
        "namespace": namespace,
        "actor_id": overrides.get("actor_id", agent_id),
        "actor_type": overrides.get("actor_type", "human"),
        **overrides,
    }

def entity_factory(
    name: str = "TestEntity",
    entity_type: str = "concept",
    namespace: str = "test-ns",
) -> dict: ...

def trajectory_factory(agent_id: str = "agent-test") -> dict: ...
```

### 6.4 `tests/integration/conftest.py` — Integration-Tier Fixtures

```python
# tests/integration/conftest.py
import pytest
from tests.fixtures.schema import build_schema
from archivist.storage.sqlite_pool import SQLitePool

@pytest.fixture
async def integration_pool(tmp_path):
    """Full-schema async pool for integration tests.

    Replaces the duplicate pool+schema setup in tests/qa/conftest.py.
    """
    db = str(tmp_path / "integration.db")
    p = SQLitePool()
    await p.initialize(db)
    async with p.write() as conn:
        await build_schema(conn)
    yield p
    await p.close()
```

### 6.5 `tests/system/conftest.py` — System-Tier Fixtures

```python
# tests/system/conftest.py
import pytest
from tests.fixtures.schema import build_schema
from tests.fixtures.mocks import mock_qdrant_client
from archivist.storage.sqlite_pool import SQLitePool

@pytest.fixture
async def qa_pool(tmp_path):
    """Isolated pool + schema for system/MCP handler tests.

    Direct replacement for tests/qa/conftest.py::qa_pool.
    """
    ...

@pytest.fixture
def memory_factory():
    from tests.fixtures.factories import memory_factory as _f
    return _f
```

### 6.6 Fixture Scope Rules

| Fixture | Scope | Rationale |
|---|---|---|
| `_isolate_env` | function (autouse) | Every test needs fresh env/paths |
| `async_pool` | function | Pool state must not bleed between tests |
| `integration_pool` | function | Same — schema state is mutable |
| `qa_pool` | function | Same |
| `rbac_config` | function | RBAC config is loaded at module level |
| `mock_llm` | function | LLM mock state must be independent |
| `memory_factory` | session | Factory is stateless, safe to share |

---

## 7. How to Run Different Test Types

```bash
# ── Development inner loop (pure unit, ~5 seconds) ───────────────────────────
pytest tests/unit/ -q --tb=short

# ── Unit + integration (pre-push, ~30 seconds) ───────────────────────────────
pytest tests/unit/ tests/integration/ -q

# ── Full suite except performance/chaos (~60 seconds) ────────────────────────
pytest -m "not (performance or chaos)" -q

# ── By domain ────────────────────────────────────────────────────────────────
pytest -m storage -q
pytest -m lifecycle -q
pytest -m retrieval -q
pytest -m mcp -q
pytest -m rbac -q

# ── Fault injection / chaos ───────────────────────────────────────────────────
pytest tests/chaos/ -q -m chaos

# ── Performance regressions (isolated run, flap-prone) ───────────────────────
pytest tests/performance/ -m performance -v

# ── Regression guards only ───────────────────────────────────────────────────
pytest tests/regression/ -m regression -q

# ── Full suite with coverage ─────────────────────────────────────────────────
pytest tests/ --cov=src/archivist --cov-report=term-missing --cov-fail-under=70

# ── Parallel (requires pytest-xdist) ─────────────────────────────────────────
pytest tests/unit/ tests/integration/ -n auto -q

# ── Collect only (verify discovery) ──────────────────────────────────────────
pytest tests/ --collect-only -q | tail -5
```

---

## 8. CI Configuration

### 8.1 Fast/Slow Job Split

```yaml
# .github/workflows/ci.yml

jobs:
  unit:
    name: Unit tests
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/unit/ -m unit -q --tb=short
    # Expected: <15s — runs on every push

  integration:
    name: Integration tests
    runs-on: ubuntu-latest
    needs: unit
    steps:
      - run: pytest tests/integration/ tests/regression/ -q --tb=short
    # Expected: ~60s — runs on every push after unit passes

  system:
    name: System/MCP smoke tests
    runs-on: ubuntu-latest
    needs: integration
    steps:
      - run: pytest tests/system/ -q --tb=short
    # Expected: ~35s

  chaos:
    name: Chaos tests
    runs-on: ubuntu-latest
    needs: integration
    steps:
      - run: pytest tests/chaos/ -m chaos -q
    # Expected: ~20s — can run in parallel with system

  performance:
    name: Performance regression
    runs-on: ubuntu-latest
    needs: integration
    if: github.ref == 'refs/heads/main'   # only on main merges
    steps:
      - run: pytest tests/performance/ -m "performance and slow" -v
    # Timing assertions are flap-prone on shared runners; only gate on main
```

### 8.2 Matrix Testing

```yaml
strategy:
  matrix:
    python-version: ["3.12", "3.13"]
# Apply to unit + integration jobs only — system/chaos/performance on 3.12 only
```

### 8.3 Coverage Gate

```yaml
# Added to integration job
- run: pytest tests/unit/ tests/integration/ --cov=src/archivist
         --cov-report=xml --cov-fail-under=70
```

---

## 9. New Tools and Plugins

### 9.1 Currently Installed (keep)

- `pytest` — test runner
- `pytest-asyncio` — `asyncio_mode = "auto"`, handles all async tests
- `pytest-mock` — `mocker` fixture (use where appropriate vs `unittest.mock`)

### 9.2 Add

| Package | Why |
|---|---|
| `pytest-cov` | Coverage reporting, `--cov-fail-under` gate |
| `pytest-xdist` | Parallel test execution (`-n auto`) for unit/integration |
| `pytest-timeout` | Per-test timeout (`@pytest.mark.timeout(5)`) to catch hung tests |

### 9.3 Explicitly Not Adding

| Package | Reason |
|---|---|
| `pytest-benchmark` | Already used in `benchmarks/micro/` separately; should not bleed into `tests/` |
| `hypothesis` | Property-based testing is valuable long-term but out of scope for this refactor |
| `pytest-randomly` | Adds non-determinism; not worth it until isolation is proven |

---

## 10. Test Data Management

### 10.1 Synthetic Data Factory

All test data goes through `tests/fixtures/factories.py`. No test should construct raw dicts inline unless they are testing a specific field structure.

### 10.2 Schema DDL — Single Source

`tests/fixtures/schema.py` extracts the full DDL that is currently duplicated across:
- `conftest.py` (root) — calls `graph.init_schema()` which reads from `src/graph.py`
- `tests/qa/conftest.py` — inlines ~850 lines of DDL

After migration, both use `from tests.fixtures.schema import build_schema`.

### 10.3 No Production Data in Tests

All test data uses:
- `agent_id` values: `"agent-test"`, `"agent-smoke"`, `"agent-regression"`, or `"agent-{role}"`
- `namespace` values: `"test-ns"`, `"default"`, `"qa-{domain}"`
- Memory text: deterministic synthetic text from factories, never real user content

---

## 11. Invariants During Migration

1. **All 835 existing tests must still collect and pass after each PR.**
2. **`pytest --collect-only` must show zero warnings** about unknown markers after pyproject.toml update.
3. **`sys.path.insert` hacks are removed** as each file is migrated — the `pythonpath = ["src"]` config handles it.
4. **`@pytest.mark.asyncio` decorators are removed** as each file is migrated.
5. **No test imports from `tests/qa/conftest.py`** after migration — all imports go through `tests/fixtures/`.
6. **`test_code_quality.py` is deleted** (pre-commit handles ruff).
7. **`tests/qa/` directory is deleted** once all its files are migrated.

---

## 12. Worked Example — Migrating `test_retry_utils.py`

**Before** (`tests/test_retry_utils.py`, 285 lines):
```python
import os
import sys
# no marker, no tier declaration
class TestRetryDecorator:
    async def test_success_no_retry(self): ...
```

**After** (`tests/unit/core/test_retry.py`):
```python
"""Unit tests for src/retry.py — retry decorator, backoff logic, error classification."""
import pytest
pytestmark = [pytest.mark.unit, pytest.mark.regression]  # regression: was a bug source

class TestRetryDecorator:
    async def test_success_no_retry(self): ...
```

Changes:
- Moved to `tests/unit/core/`
- Added `pytestmark`
- Removed `sys.path.insert` (not present in this file — kept clean)
- Removed `@pytest.mark.asyncio` decorators (if any)
- **Zero logic changes**

---

## 13. Approval Gate

This document describes the target state. **No files are moved or changed until the user replies:**

```
TESTING PLAN APPROVED — PROCEED
```

At that point, migration proceeds in this order:

1. `pyproject.toml` — add new markers (non-breaking, safe first)
2. `tests/fixtures/` — create `schema.py`, `factories.py`, `mocks.py`
3. `tests/unit/` — migrate all pure-logic tests (fastest, lowest risk)
4. `tests/integration/` — migrate integration tests domain by domain
5. `tests/system/`, `tests/chaos/`, `tests/performance/`, `tests/regression/`
6. Delete `tests/qa/` and all 40 flat `tests/test_*.py` files
7. Run `ruff check . --fix && pytest --collect-only | head -100`
8. Run full suite, confirm 835+ tests collected and passing
