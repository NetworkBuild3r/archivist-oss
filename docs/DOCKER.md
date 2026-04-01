# Docker deployment (dev / single node)

The repo ships **`docker-compose.yml`** with:

| Service    | Port (host) | Role |
|------------|----------------|------|
| `archivist` | `3100` (default) | MCP HTTP + SSE, file watcher, curator |
| `qdrant`    | `6333`, `6334`   | Vector store |

## Quick start

```bash
cp .env.example .env
# Edit .env: LLM_URL, LLM_API_KEY, EMBED_URL, EMBED_MODEL, VECTOR_DIM, etc.

docker compose up -d --build
curl -s http://localhost:3100/health
```

MCP endpoint: `http://localhost:3100/mcp/sse` (or your published host/port).

## Layout

- **Archivist image** is built from the root [`Dockerfile`](../Dockerfile) (`python main.py`).
- **Persistent data**: named volume `archivist-data` → `/data` (SQLite graph, config dir).
- **Memories**: read-only mount `./sample-memories` → `/data/memories` by default; override with env **`MEMORY_DIR`** (path on the host).

### Host paths (e.g. `/opt/appdata`)

```bash
sudo mkdir -p /opt/appdata/archivist-memories
sudo chown "$USER:$USER" /opt/appdata/archivist-memories
cp docker-compose.override.example.yml docker-compose.override.yml
# Edit override: point volumes to /opt/appdata/...
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

- **Qdrant unhealthy**: ensure ports `6333`/`6334` are free; check `docker compose logs qdrant`.
- **Archivist cannot reach embeddings on host**: confirm vLLM listens on `0.0.0.0:8000`, not only `127.0.0.1`, if needed.
- **Vector dimension mismatch**: `VECTOR_DIM` must match the embedding model (e.g. `768` for `bge-base-en-v1.5`). Recreate the Qdrant collection if you change dimension after data was indexed.
