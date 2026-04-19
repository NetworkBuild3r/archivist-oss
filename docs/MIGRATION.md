# Archivist v2.1 — Migration Guide

This document covers the upgrade path from **v2.0.x → v2.1** for all deployment configurations.

---

## At a glance

| Scenario | Action required |
|----------|----------------|
| **Existing SQLite deployment** | Update image / package only. Zero config changes. |
| **New deployment, Postgres backend** | Install `asyncpg`, set `GRAPH_BACKEND=postgres` + `DATABASE_URL`. |
| **Existing SQLite → migrate data to Postgres** | Manual export/import per namespace (no automated tool in v2.1). |
| **External health monitors** | Whitelist HTTP 503 as a valid "alive" status from `/health`. |
| **Pydantic config consumers** | No action needed — UPPER_CASE re-exports are preserved. |
| **Kubernetes / Docker healthcheck** | `failureThreshold ≥ 3`; see [HEALTHCHECK change](#healthcheck-behaviour-change). |

---

## SQLite users (default) — zero-migration upgrade

If you run Archivist with the default `GRAPH_BACKEND=sqlite` (or without setting `GRAPH_BACKEND` at all), upgrading to v2.1 is a drop-in image swap:

```bash
# Docker Compose
docker compose pull  # or rebuild: docker compose up -d --build
docker compose up -d

# Verify
curl http://localhost:3100/health
```

No schema changes, no config changes, no data migration. All MCP tool signatures are identical.

The only externally-visible change is the `/health` response shape — see [below](#healthcheck-behaviour-change).

---

## New deployment: PostgreSQL backend

Follow these steps to deploy Archivist with `GRAPH_BACKEND=postgres`.

### 1 — Prerequisites

- PostgreSQL 14 or later
- Python: `asyncpg>=0.29.0`

### 2 — Install asyncpg

**Host / bare-metal:**
```bash
pip install asyncpg
# or, with the package extras:
pip install -e ".[postgres]"
```

**Docker (rebuilding from source):**
```bash
docker build --build-arg EXTRAS=postgres -t archivist:local .
```

**Docker Compose (one-command stack):**
```bash
# docker-compose.postgres.yml adds a postgres:16-alpine service and sets
# GRAPH_BACKEND=postgres + DATABASE_URL automatically.
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
```

### 3 — Configure

Add to your `.env` (or set as environment variables):

```bash
GRAPH_BACKEND=postgres
DATABASE_URL=postgresql://archivist:archivist@localhost:5432/archivist

# Optional — tune pool size (defaults shown):
# PG_POOL_MIN=5
# PG_POOL_MAX=20
```

`SQLITE_PATH` must still be set — it is used for Qdrant outbox, backup manifests, and other operational tables even when `GRAPH_BACKEND=postgres`.

### 4 — Start Archivist

```bash
python -m archivist.app.main
# or: docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d
```

On first boot, `AsyncpgGraphBackend.initialize()` automatically runs `schema_postgres.sql`, creating all required tables, FTS indexes, and GIN indexes. This operation is idempotent — it is safe to run against an already-initialised database.

### 5 — Verify

```bash
curl http://localhost:3100/health
```

Expected response (HTTP 200):
```json
{
  "status": "healthy",
  "service": "archivist",
  "version": "2.1.0",
  "subsystems": {
    "postgres": { "healthy": true, "latency_ms": 42.1, ... }
  },
  "timestamp": "..."
}
```

If `"postgres": { "healthy": false }` appears (HTTP 503), check your `DATABASE_URL` and that Postgres is accepting connections.

---

## SQLite → Postgres data migration (brownfield)

There is **no automated data migration tool** in v2.1. The two practical options are:

### Option A — Fresh start (recommended for most deployments)

Start Archivist pointing at Postgres with no data. Re-index your memory files from `MEMORY_ROOT` using the file watcher or `archivist_index` MCP tool.

### Option B — Export/import per namespace

Use the `archivist_export_agent` and `archivist_import_agent` MCP tools to transfer memories namespace-by-namespace:

1. Keep the SQLite instance running and reachable.
2. Stand up a new Postgres-backed Archivist instance.
3. For each agent namespace: call `archivist_export_agent` on the SQLite instance, then `archivist_import_agent` on the Postgres instance.
4. Validate retrieval quality on the Postgres instance.
5. Cut over traffic and decommission the SQLite instance.

Qdrant vector data is stored independently of the graph backend and does not need migration — both instances can share the same Qdrant collection or use separate ones.

---

## Configuration changes (Pydantic Settings v2)

The configuration system was migrated from `os.getenv` to a frozen `ArchivistSettings` Pydantic v2 model. All UPPER_CASE module-level re-exports (`LLM_URL`, `GRAPH_BACKEND`, `VECTOR_DIM`, etc.) are preserved as a **Phase A compatibility layer** and will remain available throughout v2.x. No changes are required in existing `.env` files, compose configurations, or integrations that read environment variables.

If you import config values directly in Python (e.g. `from archivist.core.config import LLM_URL`), those imports continue to work unchanged.

---

## HEALTHCHECK behaviour change

**Previous behaviour:** `GET /health` always returned HTTP 200 with `{"status": "ok"}`.

**v2.1 behaviour:** `GET /health` returns:
- **HTTP 200** — all registered subsystems healthy (or no subsystems registered — e.g. SQLite-only deployment).
- **HTTP 503** — one or more subsystems are unhealthy (e.g. Postgres connection lost, Qdrant unreachable). The container is alive and serving; 503 indicates a *degraded* state.

The full response body includes `status`, `service`, `version`, `timestamp`, and a `subsystems` map.

### Impact on monitoring

**Docker HEALTHCHECK** — The updated `Dockerfile` already handles this correctly (accepts both 200 and 503 as healthy container states).

**Kubernetes liveness probes** — Ensure `failureThreshold` is at least 3 and `periodSeconds` is ≥ 10 so transient degraded states (e.g. Postgres restart) do not cycle the pod:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 3100
  initialDelaySeconds: 15
  periodSeconds: 10
  failureThreshold: 3
```

If your liveness probe health check rejects 503, add a custom check that accepts both status codes, or switch to a TCP probe.

**External monitors (Uptime Robot, Datadog Synthetics, etc.)** — Add HTTP 503 to your list of accepted status codes for the `/health` endpoint, or configure the monitor to treat `"status": "degraded"` as a warning rather than a failure.

---

## New endpoints in v2.1

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | Exempt | Structured subsystem status; 200 healthy / 503 degraded |
| `GET /debug/config` | Required | Non-secret config + feature flag snapshot |
| `GET /metrics` | Configurable (`METRICS_AUTH_EXEMPT`) | Prometheus metrics including 12 new v2.1 families |

---

## Further reading

- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — PostgreSQL backend architecture, FTS details, observability reference
- [`docs/RELEASE_NOTES_v2.1.md`](RELEASE_NOTES_v2.1.md) — Full release notes
- [`CHANGELOG.md`](../CHANGELOG.md) — Detailed change log
- [`docker-compose.postgres.yml`](../docker-compose.postgres.yml) — Postgres compose overlay
