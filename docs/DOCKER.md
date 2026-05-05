# Docker deployment (dev / single node)

Archivist runs as a **non-root** container image with `WORKDIR /app`; the process entrypoint is **`python3 -m archivist.app.main`** so package imports resolve correctly. This document covers Compose wiring, host vLLM, and volume overrides.

The repo ships **`docker-compose.yml`** with:

| Service    | Port (host) | Role |
|------------|----------------|------|
| `archivist` | `3100` (default) | MCP HTTP (Streamable HTTP + SSE), file watcher, curator |
| `qdrant`    | `6333`, `6334`   | Vector store |

## Quick start

```bash
cp .env.example .env
# Edit .env: LLM_URL, LLM_API_KEY, EMBED_URL, EMBED_MODEL, VECTOR_DIM, etc.

docker compose up -d --build
curl -s http://localhost:3100/health
```

### Git on a server without SSH keys

If `git pull` fails with `Permission denied (publickey)`, use HTTPS:

```bash
git remote set-url origin https://github.com/NetworkBuild3r/archivist-oss.git
git pull origin feature/v1.6-memory-awareness
```

Or use a [GitHub personal access token](https://github.com/settings/tokens) as the password when prompted.

MCP endpoint: `http://localhost:3100/mcp` (preferred Streamable HTTP; legacy SSE remains at `http://localhost:3100/mcp/sse`).

## Layout

- **Archivist image** is built from the root [`Dockerfile`](../Dockerfile) (`CMD` → `python3 -m archivist.app.main`).
- **Persistent data**: named volume `archivist-data` → `/data` (SQLite graph, config dir).
- **Memories**: read-only mount `./sample-memories` → `/data/memories` by default; override with env **`MEMORY_DIR`** (path on the host).

### Host paths (persistent memories on disk)

```bash
sudo mkdir -p /data/archivist-memories
sudo chown "$USER:$USER" /data/archivist-memories
cp docker-compose.override.example.yml docker-compose.override.yml
# Edit override: set paths that match your environment
docker compose up -d --build
```

`docker-compose.override.yml` is gitignored; only the **example** file is committed.

## Embeddings on the host (vLLM) + cloud LLM (xAI)

Typical on a GPU host: **vLLM** serves OpenAI-compatible embeddings on `:8000`; **xAI** serves chat.

`.env`:

```bash
EMBED_URL=http://host.docker.internal:8000
EMBED_MODEL=BAAI/bge-base-en-v1.5
EMBED_API_KEY=
VECTOR_DIM=768

LLM_URL=https://api.x.ai
LLM_API_KEY=xai-your-key
LLM_MODEL=grok-2-latest
```

`extra_hosts: host.docker.internal:host-gateway` is set in compose so Linux resolves the host from inside containers.

**Do not** set `LLM_URL` to `https://api.x.ai/v1` — the app appends `/v1/chat/completions` itself.

## Qdrant URL

Inside the **archivist** container, `QDRANT_URL` is always **`http://qdrant:6333`** (compose overrides `.env` for that variable). Your `.env` may still list `localhost:6333` for documentation when you run benchmarks or the app **without** Docker.

## Optional: all services on one machine without compose

Run Qdrant and Archivist on the host; point `QDRANT_URL`, `EMBED_URL`, and `LLM_URL` at `127.0.0.1` as in [`.env.example`](../.env.example).

## Troubleshooting

- **Compose no longer depends on Qdrant “health”** — the official `qdrant` image has no `curl`/`wget`, and shell TCP probes can fail on some hosts. **Archivist waits up to 120s** for Qdrant to accept connections on startup (see `ensure_qdrant_collection` in `main.py`).
- **Port conflict**: if host port `6333` is already in use, change `QDRANT_PORT` in `.env` or stop the other service.
- **Archivist cannot reach embeddings on host**: confirm vLLM listens on `0.0.0.0:8000`, not only `127.0.0.1`, if needed.
- **Vector dimension mismatch**: `VECTOR_DIM` must match the embedding model (e.g. `768` for `bge-base-en-v1.5`). Recreate the Qdrant collection if you change dimension after data was indexed.

## PostgreSQL backend (production-grade)

The default backend is SQLite — zero config, works instantly, and supports dozens of concurrent agents. Switch to PostgreSQL for **large fleets or horizontal scaling**; Postgres MVCC replaces the single-writer `asyncio.Lock` with connection-pool concurrency.

### Compose quickstart (managed Postgres in Docker)

```bash
cp .env.example .env
# Edit .env for LLM / embed as usual — no DATABASE_URL needed for the managed service

docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
curl http://localhost:3100/health
```

`docker-compose.postgres.yml` starts a `postgres:16-alpine` service, sets `GRAPH_BACKEND=postgres`, and wires `DATABASE_URL` automatically. Data persists in the `pg-data` named Docker volume.

To customise credentials or port, add these to `.env` before composing up:

```bash
PG_USER=myuser
PG_PASSWORD=secret
PG_DB=archivist
PG_PORT=5432
PG_POOL_MIN=2
PG_POOL_MAX=10
```

### External Postgres (RDS, Supabase, Cloud SQL, etc.)

Add to `.env` and use the standard compose file:

```bash
GRAPH_BACKEND=postgres
DATABASE_URL=postgresql://user:password@host:5432/archivist
PG_POOL_MIN=2
PG_POOL_MAX=10
```

The schema (`schema_postgres.sql`) is applied automatically on first startup. All statements are idempotent (`IF NOT EXISTS`), so re-running is safe.

### Requirements

- `asyncpg` — already in `requirements.txt`
- `pg_dump` / `pg_restore` on PATH for backup MCP tools (`postgresql-client` package)

### Schema highlights

Full DDL: [`src/archivist/storage/schema_postgres.sql`](../src/archivist/storage/schema_postgres.sql)

| SQLite | Postgres equivalent |
|--------|---------------------|
| `INTEGER PRIMARY KEY AUTOINCREMENT` | `SERIAL PRIMARY KEY` |
| `TEXT NOT NULL UNIQUE COLLATE NOCASE` | `CITEXT NOT NULL UNIQUE` |
| FTS5 virtual tables | `tsvector` columns + GIN indexes |
| `INSERT OR IGNORE` | `INSERT INTO … ON CONFLICT DO NOTHING` (auto-translated) |
| `INSERT OR REPLACE` | `INSERT INTO … ON CONFLICT DO UPDATE SET …` (auto-translated) |

### Backups

`archivist_backup` / `archivist_restore` detect the active backend automatically:

- **SQLite** — Python Online Backup API (`graph.db`)
- **Postgres** — `pg_dump --format=custom` (`graph.pgdump`) / `pg_restore --clean`

Both are stored under `BACKUP_DIR` alongside the Qdrant snapshots in a timestamped directory with `manifest.json` recording `graph_backend`.

### Postgres integration tests

```bash
POSTGRES_TEST_DSN="postgresql://archivist:archivist@localhost:5432/archivist_test" \
  pytest tests/integration/storage/test_postgres_backend.py \
         tests/integration/storage/test_dual_backend.py -v
```

### Switching back to SQLite

Remove `GRAPH_BACKEND` and `DATABASE_URL` from `.env`. SQLite resumes on next restart — the Postgres data is unaffected.

---

## Answer Finder configuration (v2.3)

Archivist v2.3 ships an Answer Finder layer that assembles token-budgeted context for every agent query. The defaults work out-of-the-box; tune the variables below when you need different behavior.

All variables are optional. Uncomment them in `.env` or pass them as Docker Compose environment entries.

### Context packing policy

Controls how the tier-aware packer chooses between L0 headlines, L1 summaries, and L2 full content.

```bash
# adaptive (default): 3-pass — L0 first, upgrade to L1/L2 by score
# l0_first: maximum compression (fewest tokens)
# l2_first: greedy full content (legacy behavior)
CONTEXT_PACK_POLICY=adaptive

# Fraction of max_tokens reserved for L0 summaries in the adaptive first pass
CONTEXT_L0_BUDGET_SHARE=0.30

# Minimum results always upgraded to their best tier, regardless of budget
CONTEXT_MIN_FULL_RESULTS=3
```

### Auto-compression (opt-in)

When `AUTO_COMPRESS_ENABLED=true`, results that overflow the token budget are LLM-summarized and re-injected as a synthetic L1 chunk. This prevents silent truncation.

```bash
AUTO_COMPRESS_ENABLED=false         # set true to enable
AUTO_COMPRESS_THRESHOLD=0.85        # budget utilization fraction that triggers compression
```

This feature requires an LLM endpoint (`LLM_URL`). If the LLM is unavailable, overflow items are silently dropped (same behavior as when disabled).

### Ephemeral session store

In-process per-session scratch memory. Entries expire after `SESSION_STORE_TTL_SECONDS` and are never written to SQLite or Qdrant unless explicitly promoted.

```bash
SESSION_STORE_MAX_ENTRIES=512       # total entries across all sessions
SESSION_STORE_TTL_SECONDS=3600      # 1 hour per entry
```

Entries marked `promoted=true` during a session are flushed to durable memory when `archivist_session_end` is called with `persist_ephemeral=true`.

### Token savings observability

No configuration needed. Every retrieval logs `tokens_returned`, `tokens_naive`, `savings_pct`, and `pack_policy` to `retrieval_logs` automatically. View the dashboard:

```bash
# via MCP tool
archivist_savings_dashboard(window_days=7, heatmap_top_n=50)

# or via REST
curl http://localhost:3100/admin/dashboard
```

### Running the token efficiency benchmark

Requires a running Archivist stack with memories loaded:

```bash
cd /opt/appdata/archivist-oss   # or your checkout
cp .env.example .env            # configure LLM_URL, EMBED_URL, QDRANT_URL

# Quick run (10 queries)
python -m benchmarks.token_efficiency --queries 10

# Full benchmark (50 queries × 3 policies)
python -m benchmarks.token_efficiency \
  --output .benchmarks/token_efficiency_$(date +%Y%m%d).json

# Results are printed as a formatted table and saved to .benchmarks/
```

