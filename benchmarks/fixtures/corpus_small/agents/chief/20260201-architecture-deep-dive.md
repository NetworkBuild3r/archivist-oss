# Platform Architecture Deep Dive (chief)

**Agent:** chief
**Date:** 2026-02-01

This document records illustrative architecture decisions and operational
patterns used in the **benchmark** corpus.  It is intentionally long to
exercise semantic chunking at index time.  All hostnames, paths, and routing
identifiers are **synthetic** (not a real deployment).

## Service Topology

The example platform runs across three availability zones in a fictional
region.  Each zone hosts a replica of every stateful service.

Core services and their **example** hostnames:

- **api-gateway** (demo-api.example.internal:8080) — TLS termination, rate
  limiting, JWT validation.  Runs 6 replicas minimum.  Horizontal pod
  autoscaler triggers at 70 % CPU.
- **memory-service** (demo-memory.example.internal:3100) — Archivist MCP
  server.  Each instance is stateless; Qdrant and SQLite are the only
  persistent stores.  Target replica count: 4.
- **curator-worker** (demo-curator.example.internal:9000) — background dedup,
  compaction, and entity extraction.  Single-replica per namespace to prevent
  concurrent SQLite write contention.
- **embedding-proxy** (demo-embed.example.internal:4000) — round-robin proxy
  in front of three GPU nodes running a generic embedding model.  p99 embed
  latency SLO: 80 ms for 512-token inputs.
- **qdrant-cluster** (demo-vectors.example.internal:6333) — 3-node Qdrant
  cluster with replication_factor=2.  Collection: demo_memories.  HNSW m=32,
  ef_construct=256.

## Database Design

### Qdrant Collections

Single collection mode is enabled (SINGLE_COLLECTION_MODE=true).  All
namespaces share the demo_memories collection.  Namespace isolation is
enforced via a required payload filter on every query.

Key payload fields:

| Field | Type | Purpose |
|-------|------|---------|
| namespace | string | Tenant isolation |
| agent_id | string | Per-agent filtering |
| is_parent | bool | Parent vs child chunk flag |
| parent_id | string | Child → parent link |
| parent_text | string | Full parent context (index-time) |
| chunk_type | string | child \| parent \| needle \| synthetic |
| file_date | string | ISO-8601 date from filename |
| importance | float | Curator-assigned importance score |

Vector dimension: 1024.  Distance metric: Cosine.

### SQLite Schema

The SQLite database at /data/demo/graph.db holds three subsystems:

1. **FTS5 virtual table** (fts_chunks) — full-text search over raw chunk
   content.  Tokeniser: unicode61 with diacritics=false.
2. **Entity graph** (entities, facts, entity_mentions) — named-entity
   extraction results and fact statements.
3. **Needle registry** (needle_tokens) — high-specificity token index for
   IP addresses, UUIDs, cron expressions, and ticket-style IDs.

SQLite is opened in WAL mode.  All writes acquire GRAPH_WRITE_LOCK.  Reads
do not acquire the lock; WAL mode handles concurrent read safety.

## Monitoring Setup

### Prometheus Metrics

All services expose /metrics on port 9090 without authentication
(METRICS_AUTH_EXEMPT=true for lab scrape).  Scrape interval: 15 s.

Critical alerts (example routing key: PD-DEMO-001):

- **archivist_embed_duration_p99 > 200ms** for 5 minutes → page on-call SRE.
- **archivist_qdrant_error_total rate > 0.1/s** for 2 minutes → page on-call.
- **archivist_store_total rate < 0.01/s** for 10 minutes (store stall) → page
  on-call SRE and tech lead.
- **SQLite WAL file > 500 MB** → warning to a team channel (e.g. #demo-alerts).

### Grafana Dashboards

Dashboard UIDs:

- **archivist-overview**: d/archivist-overview — golden signals, per-namespace
  store/search rates, curator queue depth.
- **archivist-retrieval**: d/archivist-retrieval — R@1, R@5, NDCG@5 from
  shadow evaluation runs, reranker latency percentiles.
- **archivist-qdrant**: d/archivist-qdrant — collection size, vector search
  latency, HNSW ef_search tuning.

Retention: 90 days on a hosted metrics tier; local Prometheus TSDB retains
15 days.

## Deployment Pipeline

### CI (example: GitLab)

Pipeline stages: lint → unit-test → build → integration-test → deploy.

Key environment variables injected from CI variables:

```
QDRANT_URL=http://vectors.demo.local:6333
LLM_URL=http://llm.demo.local:4000
EMBED_URL=http://embed.demo.local:4000
ARCHIVIST_API_KEY=${ARCHIVIST_API_KEY}
RERANKER_ENABLED=true
SYNTHETIC_QUESTIONS_ENABLED=true
CHUNKING_STRATEGY=semantic
```

Deployment uses Helm chart `demo-archivist-chart` version 1.12.0.  Rolling
update strategy with maxUnavailable=0 and maxSurge=1.

### Runbook: Emergency Rollback

1. `helm rollback demo-archivist-chart -n demo-bench`
2. Verify pod restarts: `kubectl rollout status deployment/memory-service -n demo-bench`
3. Check /health endpoint on all replicas: expected `{"status":"ok"}`.
4. If Qdrant data is corrupt, restore from latest snapshot:
   `qdrant-backup restore --collection demo_memories --snapshot latest`

## Security Posture

All inter-service traffic is mTLS via a service mesh.  External traffic
terminates TLS at the api-gateway.

API key authentication is enforced on all MCP tool endpoints
(ARCHIVIST_API_KEY must be set).  The /health and /metrics endpoints are
exempt from API key auth for liveness probes and Prometheus.

Secrets are stored in a central secrets store; injected at pod startup via a
sidecar.  Secret lease renewal: 24 h.  Auto-renew: enabled.

RBAC configuration lives in /data/demo/config/namespaces.yaml.  Each
namespace entry declares allowed_agents, retention_class, and consistency
(strong | eventual).
