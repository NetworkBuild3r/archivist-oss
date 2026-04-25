# Archivist MCP tool reference

Quick reference for **37** MCP tools exposed by the Archivist server. For full parameter schemas, defaults, and examples, see [`CURSOR_SKILL.md`](CURSOR_SKILL.md).

## Search & Retrieval (9)

| Tool | Purpose |
|------|---------|
| `archivist_search` | Semantic search + 10-stage RLM pipeline with optional LLM refinement, tier selection, date filters, multi-agent fleet support |
| `archivist_recall` | Entity-centric multi-hop graph lookup (entities, relationships, facts) |
| `archivist_timeline` | Chronological slice for a topic with configurable lookback |
| `archivist_insights` | Cross-agent topic discovery across accessible namespaces |
| `archivist_deref` | Dereference a memory by ID for full L2 detail (drill-down after L0/L1 search) |
| `archivist_index` | Compressed navigational index of namespace knowledge (~500 tokens) |
| `archivist_contradictions` | Surface contradicting facts about an entity across agents |
| `archivist_entity_brief` | Structured knowledge card for an entity: facts, relationships, retention class, mention count, timeline. Supports `as_of` for point-in-time views. |
| `archivist_wake_up` | Bootstrap session context — agent identity, critical pinned facts, namespace overview in ~200 tokens |

## Storage & Memory Management (6)

| Tool | Purpose |
|------|---------|
| `archivist_store` | Write a memory with entity extraction, conflict checks, LLM dedup |
| `archivist_delete` | Soft-delete a memory by ID — hides from all search paths in ~5 ms, background hard-cascade |
| `archivist_merge` | Merge conflicting entries (latest / concat / semantic / manual) |
| `archivist_compress` | Archive memories and return compact summaries (flat or structured Goal/Progress/Decisions/Next Steps) |
| `archivist_pin` | Pin a memory or entity to retention class `permanent` — sets importance to 1.0 |
| `archivist_unpin` | Remove the permanent pin from a memory or entity |

## Trajectory & Feedback (5)

| Tool | Purpose |
|------|---------|
| `archivist_log_trajectory` | Log execution trajectory (task + actions + outcome), auto-extract tips |
| `archivist_annotate` | Add quality annotations (note, correction, stale, verified, quality) to a memory |
| `archivist_rate` | Rate a memory as helpful (+1) or unhelpful (-1) |
| `archivist_tips` | Retrieve strategy/recovery/optimization tips from past trajectories |
| `archivist_session_end` | Summarize a session into durable memory |

## Skill Registry (6)

| Tool | Purpose |
|------|---------|
| `archivist_register_skill` | Register or update a skill (MCP tool) with provider, version, endpoint |
| `archivist_skill_event` | Log invocation outcome (success/partial/failure) for health scoring |
| `archivist_skill_lesson` | Record failure modes, workarounds, best practices |
| `archivist_skill_health` | Get health grade, success rate, recent failures, substitutes |
| `archivist_skill_relate` | Create relations between skills (similar_to, depend_on, compose_with, replaced_by) |
| `archivist_skill_dependencies` | Get skill dependency/relation graph |

## Admin & Context Management (8)

| Tool | Purpose |
|------|---------|
| `archivist_context_check` | Pre-reasoning token counting against a budget with compaction hints |
| `archivist_namespaces` | List namespaces visible to an agent |
| `archivist_audit_trail` | View immutable audit log entries |
| `archivist_resolve_uri` | Resolve `archivist://` URIs to their underlying resource |
| `archivist_retrieval_logs` | Export/analyze retrieval pipeline execution traces |
| `archivist_health_dashboard` | Single-pane health: memory counts, stale %, conflict rate, skills, cache |
| `archivist_batch_heuristic` | Recommended batch size (1-10) from health signals |
| `archivist_backup` | Create, list, restore, or delete memory snapshots (Qdrant + SQLite/Postgres). Supports `export_agent` / `import_agent` for portable agent migration. |

## Cache Management (2)

| Tool | Purpose |
|------|---------|
| `archivist_cache_stats` | Hot cache stats (entries per agent, TTL, hit rate) |
| `archivist_cache_invalidate` | Manual eviction by namespace, agent, or all |

## Reference Docs (1)

| Tool | Purpose |
|------|---------|
| `archivist_get_reference_docs` | Return the full Archivist tool skill reference from inside the server. Optionally pass `section` to filter to a heading (e.g. `search`, `storage`, `admin`). |

## Usage Hints

- `min_score` / `RETRIEVAL_THRESHOLD`: set to `0` to disable score filtering for a single call when debugging recall.
- Prefer `archivist_search` first; refine with `archivist_recall` when entities are known.
- Use `archivist_entity_brief` when you need a structured knowledge card — faster than multiple search/recall calls.
- Call `archivist_wake_up` once at session start to pre-load critical context in ~200 tokens.
- Use `archivist_context_check` before reasoning to decide if context compaction is needed.
- Use `archivist_compress` with `format: structured` for Goal/Progress/Decisions/Next Steps summaries.
- Log trajectories so future searches benefit from outcome-aware retrieval scoring.
- Pin critical facts (host IPs, credentials, ownership) with `archivist_pin` so the curator never forgets them.

## REST Endpoints (non-MCP)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness probe (no auth required) |
| `/metrics` | GET | Prometheus text exposition (see **Prometheus metrics** below) |
| `/admin/invalidate` | POST/GET | Delete expired memories (TTL-based) |
| `/admin/retrieval-logs` | GET | Export retrieval pipeline execution traces |
| `/admin/dashboard` | GET | Health dashboard JSON (add `?batch=true` for batch heuristic) |
| `/mcp` | GET/POST/DELETE | MCP Streamable HTTP transport entrypoint (preferred) |
| `/mcp/sse` | GET | Legacy MCP SSE transport entrypoint |
| `/mcp/messages/` | POST | Legacy SSE message handler |

## Prometheus metrics

Archivist exposes a **text exposition** endpoint at **`GET /metrics`** on the same port as MCP (see `MCP_PORT`, default **3100**). Implementation is in-repo (`src/metrics.py`); no extra Python dependency.

| Env var | Default | Purpose |
|---------|---------|---------|
| `METRICS_ENABLED` | `true` | When `false`, recording is disabled and `/metrics` returns **404**. |
| `METRICS_AUTH_EXEMPT` | `false` | When `true`, `/metrics` does not require `ARCHIVIST_API_KEY` (use for in-cluster Prometheus scrape). |
| `METRICS_COLLECT_INTERVAL_SECONDS` | `60` | How often storage/availability gauges refresh (minimum enforced in the loop: 5s). |

**Cardinality:** labels use tool names, namespaces, coarse status strings, and collection names — not raw queries or UUIDs.

**Names (representative):**

| Metric | Type | Labels (if any) | Meaning |
|--------|------|-----------------|--------|
| `archivist_mcp_tool_duration_ms` | histogram | `tool` | MCP tool call latency (ms). |
| `archivist_mcp_tool_errors_total` | counter | `tool` | Unhandled handler exceptions per tool. |
| `archivist_search_total` | counter | — | Completed retrieval pipeline runs. |
| `archivist_search_duration_ms` | histogram | — | End-to-end search latency (ms). |
| `archivist_search_results` | histogram | `namespace` | Count of items in `sources` returned. |
| `archivist_store_total` | counter | `namespace` | Successful stores. |
| `archivist_cache_hit_total` / `archivist_cache_miss_total` | counter | — | Hot-cache hits/misses. |
| `archivist_embed_duration_ms` | histogram | — | Embedding API latency (ms). |
| `archivist_embed_cache_hit_total` / `archivist_embed_cache_miss_total` | counter | — | In-process embed LRU cache. |
| `archivist_qdrant_query_duration_ms` | histogram | — | Qdrant query latency (ms). |
| `archivist_llm_duration_ms` | histogram | — | LLM call latency (ms). |
| `archivist_total_memories` | gauge | `namespace` | Distinct memory IDs in audit log whose latest action is not `delete`. |
| `archivist_sqlite_size_bytes` | gauge | — | Size of `SQLITE_PATH` on disk. |
| `archivist_qdrant_vectors_total` | gauge | `collection` | Qdrant collection `points_count`. |
| `archivist_qdrant_available` / `archivist_sqlite_available` | gauge | — | `1` if dependency responds, else `0` (Qdrant also reflects `health` registry when set). |

**Kubernetes (Prometheus Operator):** point a `ServiceMonitor` at the Service port that serves HTTP (same as MCP), path `/metrics`, and set `METRICS_AUTH_EXEMPT=true` **or** configure scrape auth to send your `ARCHIVIST_API_KEY`.

## Timeout troubleshooting

Slow or hanging requests can come from **downstream dependencies** (embedding API, Qdrant, LLM) or from **infrastructure in front of Archivist** (reverse proxies, gateways, MCP bridges). Archivist logs dependency timings in Prometheus histograms (`archivist_embed_duration_ms`, `archivist_qdrant_query_duration_ms`, `archivist_llm_duration_ms`) and MCP tool duration (`archivist_mcp_tool_duration_ms`). When you set `SLOW_EMBED_MS`, `SLOW_QDRANT_MS`, or `SLOW_LLM_MS` (milliseconds; `0` disables), a **`slow_path`** warning is emitted if a step exceeds the threshold (includes `request_id` when present).

If the client reports **`ETIMEDOUT`** or similar before Archivist logs complete, treat the **gateway or client** as first suspect: increase timeouts, inspect gateway logs, and correlate with **`X-Request-ID`** (Archivist accepts this header on MCP HTTP transports and propagates it into logs and tool lines). Full root-cause analysis usually requires gateway-side logs; Archivist does not duplicate them here.

## Pruning and TTL (vector store)

The **`/admin/invalidate`** endpoint scans Qdrant for points whose `ttl_expires_at` payload is in the past, deletes those vectors, and appends matching rows to the **immutable audit log** (`delete` actions with `reason: ttl_expired`). The HTTP response returns `{"invalidated": N}`. Logs include a structured line `invalidation.complete` with `count`, `duration_ms`, and samples of point IDs and namespaces.

**Optional export:** set **`ARCHIVIST_INVALIDATION_EXPORT_PATH`** to a file path to append **one JSON object per invalidation run** (not per point), e.g. `count`, `sample_ids`, `sample_namespaces`, `duration_ms`, `reason`. Operators should rotate or truncate this file (logrotate, sidecar shipper) on long-lived clusters; full memory text is not written (IDs/namespaces only).

## Curator vs vector TTL

The background **`curator.cycle`** log line (one per successful loop) summarizes file processing, graph fact decay, hotness scoring, tip consolidation, and wake-up cache refreshes. **Graph decay** (`facts_decayed`) soft-deactivates old or superseded facts in SQLite; **vector TTL** is enforced separately via `/admin/invalidate` and payload `ttl_expires_at`. They address different layers: the graph ages knowledge; Qdrant TTL removes embedded chunks after expiry.

## See Also

- [CURSOR_SKILL.md](CURSOR_SKILL.md) — full parameter schemas and examples
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design and module map
- [INSPIRATION.md](INSPIRATION.md) — ReMe comparison and design rationale
