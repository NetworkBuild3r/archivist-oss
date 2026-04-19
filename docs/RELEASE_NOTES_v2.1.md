# Archivist 2.1.0 — Release Notes

**Release date:** 2026-04-18
**Git tag:** `v2.1.0`
**Branch:** `feature/observability-clean` (merge target: `main`)

## Summary

Archivist **2.1** is the first production-ready release of the full v2 platform. It bundles four major engineering phases delivered since v2.0.1:

1. **Transactional outbox + `MemoryTransaction`** (Phase 3 + 3.5) — atomic cross-store writes
2. **Pydantic Settings v2** — validated, frozen configuration model
3. **Pluggable GraphBackend + PostgreSQL backend with FTS parity** (Phase 4) — first-class Postgres support
4. **Production-grade observability** — enriched `/health`, `/debug/config`, 12 new Prometheus metrics, structured logging

For users already on v2.0.x with the default SQLite backend, **this is a zero-migration drop-in upgrade**. See [`docs/MIGRATION.md`](MIGRATION.md) for all upgrade paths including the new PostgreSQL backend.

---

## What's included

| Phase | Description | Reference |
|-------|-------------|-----------|
| Phase 3 + 3.5 | Transactional outbox, `MemoryTransaction`, `OutboxProcessor`, `conn=` shims | [`rearchitect_storage_phase3.md`](rearchitect_storage_phase3.md) |
| Config (v2) | Pydantic Settings v2 `ArchivistSettings`; UPPER_CASE compat re-exports preserved | [`core/config.py`](../src/archivist/core/config.py) |
| Phase 4 | `GraphBackend` protocol; `AsyncpgGraphBackend`; `schema_postgres.sql`; tsvector FTS parity | [`ARCHITECTURE.md`](ARCHITECTURE.md#postgresql-backend-v21) |
| Observability | 12 new metrics; `/health` subsystems map + 503; `/debug/config`; structured logging | [`ARCHITECTURE.md`](ARCHITECTURE.md#observability-v21) |

---

## Highlights

### Transactional outbox (Phase 3 + 3.5)

`async with MemoryTransaction()` wraps a single `pool.write()` lock, exposing `txn.conn` for all SQLite mutations. FTS, needle registry, `memory_points`, entity/facts, and outbox rows all commit in a single atomic transaction when `OUTBOX_ENABLED=true`. A background `OutboxProcessor` drains pending rows to Qdrant with retries and idempotency. Crash windows that previously left cross-store orphans on write, index, merge, and delete paths are closed.

See [`tests/qa/`](../tests/qa/) for atomicity and chaos-style fault injection test coverage, and [`docs/QA.md`](QA.md) for the test guide.

### Pluggable GraphBackend + PostgreSQL

`GRAPH_BACKEND=postgres` + `DATABASE_URL` switches all graph, FTS, and structured-state storage to PostgreSQL 14+. Full-text search uses `tsvector`/`tsquery` (with `ts_rank_cd` scoring) in place of FTS5 — retrieval results are identical. Schema initialises automatically on first boot. A `?`-to-`$N` placeholder translation layer in `AsyncpgConnection` means no SQL strings needed to change.

```bash
# One-command Postgres stack
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
```

`asyncpg` is an optional dependency not installed by default. Install with:
```bash
pip install asyncpg                     # bare metal
pip install -e ".[postgres]"            # editable install with extras
docker build --build-arg EXTRAS=postgres .  # Docker
```

### Pydantic Settings v2

All 130+ configuration variables are validated by a frozen `ArchivistSettings` model at startup. Field validators enforce constraint combinations (e.g. `chunk_overlap < chunk_size`). The UPPER_CASE module-level re-exports that all consumer code uses are preserved as a Phase A compatibility layer — zero changes needed.

### Production-grade observability

**`GET /health`** now returns a structured JSON document including per-subsystem health, initialisation latency, and an overall `"healthy"` / `"degraded"` status. HTTP 503 is returned when any subsystem is unhealthy — the container is still running and serving, 503 indicates a degraded state. Update external monitors to accept 503 as a valid "alive" response.

**`GET /debug/config`** (auth required) returns a non-secret config snapshot for runtime inspection.

**Prometheus metrics** — 12 new metric families:

| Metric | Description |
|--------|-------------|
| `archivist_pg_pool_acquire_ms` | Pool connection acquire latency |
| `archivist_pg_pool_query_ms` | Per-query execution latency |
| `archivist_pg_pool_errors_total` | Pool + query errors |
| `archivist_pg_pool_size` | Current pool size (gauge) |
| `archivist_fts_search_duration_ms` | FTS search duration by backend |
| `archivist_fts_search_total` | FTS search call count by backend |
| `archivist_fts_upsert_total` | FTS upsert count by backend |
| `archivist_fts_upsert_errors_total` | FTS upsert failures by backend |
| `archivist_index_duration_ms` | Write-pipeline indexing duration |
| `archivist_curator_extract_duration_ms` | Curator extract phase |
| `archivist_curator_decay_duration_ms` | Curator decay phase |
| `archivist_subsystem_healthy` | 1.0/0.0 per subsystem (gauge) |

---

## Breaking changes

| Change | Impact | Mitigation |
|--------|--------|-----------|
| `GET /health` returns HTTP 503 when degraded (was always 200) | External monitors that treat 503 as "down" may alert on startup/transient degradation | Whitelist 503 as a valid "alive" status; see [`docs/MIGRATION.md`](MIGRATION.md#healthcheck-behaviour-change) |
| `GET /health` response body changed from `{"status":"ok"}` to structured object | Code that parses the health response will need updating | Update parser to read `response["status"]` field |

**No breaking changes for:** SQLite deployments, MCP tool signatures, config env vars, Python import paths.

---

## Upgrade instructions

See **[`docs/MIGRATION.md`](MIGRATION.md)** for the full guide including:
- SQLite users: drop-in upgrade steps
- New Postgres deployment: step-by-step setup
- SQLite → Postgres data migration options
- Kubernetes liveness probe adjustments

**Quick upgrade (SQLite users):**
```bash
docker compose pull && docker compose up -d
# or rebuild: docker compose up -d --build
curl http://localhost:3100/health  # verify: {"status":"healthy",...}
```

---

## Docker

```bash
# Standard (SQLite backend):
docker pull ghcr.io/networkbuild3r/archivist-oss:v2.1.0

# PostgreSQL backend (asyncpg included):
docker build --build-arg EXTRAS=postgres -t archivist:v2.1.0-postgres .
```

---

## References

- [`CHANGELOG.md`](../CHANGELOG.md) — Full change list
- [`docs/MIGRATION.md`](MIGRATION.md) — Upgrade guide
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — PostgreSQL backend and observability architecture
- [`docs/rearchitect_storage_phase3.md`](rearchitect_storage_phase3.md) — Transactional outbox design reference
- [`README.md`](../README.md) — Quick start and configuration reference

---

*Archivist — Memory-as-a-Service for AI agent fleets.*
