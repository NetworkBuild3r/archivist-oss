# Enterprise Benchmark Suite — Phase 1

Performance validation suite for store + search latency, throughput, and SQLite vs Postgres comparison.

## Quick Start

```bash
# Install benchmark deps (adds psutil)
pip install -r requirements.txt -r requirements-test.txt

# Smoke run — mocked Qdrant/embed, ~30s, no external services needed
python -m benchmarks.enterprise.scenarios.smoke

# Dry-run (validates harness, no real I/O, always passes)
python -m benchmarks.enterprise.scenarios.smoke --dry-run

# SQLite vs Postgres comparison (requires DATABASE_URL)
DATABASE_URL=postgresql://archivist:archivist@localhost:5432/archivist \
    python -m benchmarks.enterprise.scenarios.sqlite_vs_pg
```

## Scenarios

| Scenario | File | Requires | Duration |
|----------|------|----------|---------|
| Smoke | `scenarios/smoke.py` | Nothing (mocked) | ~30–60 s |
| SQLite vs Postgres | `scenarios/sqlite_vs_pg.py` | `DATABASE_URL` + asyncpg | ~5–10 min |

## Output

Results land in `.benchmarks/enterprise/` (gitignored per `.benchmarks/` contract).
A committed `baselines.json` stores the CI regression thresholds.

```
.benchmarks/enterprise/
├── smoke_sqlite.json          # CI smoke result
└── comparison.json            # SQLite vs Postgres comparison
```

## CI Smoke Job

The `benchmark-smoke` GitHub Actions job runs `scenarios/smoke.py` on every PR to `main`.
It fails if any operation's p99 exceeds `2 × baselines.json[op].p99_ms`.
Results are uploaded as a CI artifact (`smoke-results.json`, 14-day retention).

## Baseline Gate

```bash
# Check if current run would pass the baseline gate
python -m benchmarks.enterprise.scenarios.smoke --check-baselines

# Update baselines after intentional performance improvements
# 1. Run without baseline check to get current numbers:
python -m benchmarks.enterprise.scenarios.smoke --no-check-baselines \
    --output .benchmarks/enterprise/calibration.json

# 2. Review .benchmarks/enterprise/calibration.json and manually
#    update benchmarks/enterprise/baselines.json with new p99_ms values.
# 3. Commit baselines.json.
```

## Architecture

```
benchmarks/enterprise/
├── harness.py              # LatencyHistogram, BenchmarkResult, BackendFixture, run_workload, write_json
├── workloads/
│   ├── store.py            # Write-path workload (injectable Qdrant/embed mocks)
│   └── search.py           # Retrieval workload (injectable mocks + recall@5)
├── scenarios/
│   ├── smoke.py            # CI smoke: store×20 + search×20, SQLite only
│   └── sqlite_vs_pg.py     # Comparison: identical workload on both backends
├── baselines.json          # Committed CI regression thresholds
└── README.md               # This file
```

## Metrics Collected

| Metric | Description |
|--------|-------------|
| `p50_ms` / `p95_ms` / `p99_ms` | Latency percentiles per operation |
| `ops_per_sec` | Throughput at the configured concurrency |
| `error_rate` | Fraction of calls that raised an exception |
| `recall_at_5` | Search quality: fraction of queries with a keyword hit in top 5 |
| `rss_mb_delta` | RSS memory growth during the run (via psutil) |
| `cpu_percent_mean` | Mean CPU usage during the run (via psutil) |

## Phase 2 Roadmap

Phase 2 (separate PR, after smoke job is stable in CI) will add:

- `workloads/fts.py` — FTS5/tsvector indexing + query latency
- `workloads/curator.py` — curator drain timing at queue depths 10/100/500
- `workloads/lifecycle.py` — cascade delete, archive, merge latency
- `workloads/mixed.py` — multi-agent writer + reader + curator interleaved
- `scenarios/sustained.py` — 5-minute sustained load
- `scenarios/chaos.py` — Qdrant downtime, pool exhaustion, FTS health fallback
- `reporters/html_reporter.py` — standalone HTML dashboard
- `PrometheusPoller` — scrape `/metrics` during runs
- `benchmark-full.yml` — weekly scheduled workflow with Qdrant service
