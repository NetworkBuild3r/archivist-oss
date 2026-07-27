# Archivist MCP tool reference

Quick reference for **47** MCP tools exposed by the Archivist server. For full parameter schemas, defaults, and examples, see [`CURSOR_SKILL.md`](CURSOR_SKILL.md).

## Tool profiles (`ARCHIVIST_TOOL_PROFILE`)

<!-- INIT-003/SPEC-003 -->

`list_tools` / `call_tool` honor **`ARCHIVIST_TOOL_PROFILE`** (default **`core`**):

| Profile | Surface |
|---------|---------|
| `core` | Coach-path tools only: `archivist_store`, `archivist_search`, `archivist_get_context`, `archivist_index`, `archivist_delete` (forget path), plus small helpers (`archivist_health_dashboard`, `archivist_namespaces`, `archivist_get_reference_docs`). ≤12 tools. |
| `ops` | Operator-oriented middle set — all tools except unfinished `archivist_checkpoint_*` wedge. Includes **`archivist_share_*`** (Diff #5 / [ADR-009](adr/ADR-009-native-multi-agent-coordination.md)). |
| `full` | Entire registry (including checkpoints). |

Hidden tools remain in the codebase but are omitted from `list_tools` and **fail closed** on `call_tool` with a clear error. Set `ARCHIVIST_TOOL_PROFILE=full` (or `ops`) when you need the broader surface. See [ADR-003](adr/ADR-003-coach-core-reliability.md).

The tables below describe the **full** registry; only the active profile is advertised to MCP clients.

## Search & Retrieval (9)

| Tool | Purpose |
|------|---------|
| `archivist_search` | Semantic search + 10-stage RLM pipeline with optional LLM refinement, tier selection, date filters, multi-agent fleet support |
| `archivist_recall` | Entity-centric multi-hop graph lookup (entities, relationships, facts) |
| `archivist_timeline` | Chronological slice for a topic with configurable lookback |
| `archivist_insights` | Cross-agent topic discovery across accessible namespaces |
| `archivist_deref` | Dereference a memory by ID for full L2 detail (drill-down after L0/L1 search) |
| `archivist_index` | Progressive-disclosure map (~500 tokens): entity/type pointers + search hints; dual `{markdown, map}`; not citable evidence; live rebuild each call |
| `archivist_contradictions` | Surface contradicting facts about an entity across agents |
| `archivist_entity_brief` | Structured knowledge card for an entity: facts, relationships, retention class, mention count, timeline. Supports `as_of` for point-in-time views. |
| `archivist_wake_up` | Bootstrap session context — agent identity, critical pinned facts, namespace overview in ~200 tokens |

## Storage & Memory Management (6)

<!-- INIT-003/SPEC-004: store ack = durable outbox commit, not Qdrant sync -->
<!-- INIT-003/SPEC-006: store provenance + forget modes -->

With **`OUTBOX_ENABLED=true` (default)**, `archivist_store` acknowledges after the
durable graph + outbox commit. Qdrant upsert runs asynchronously via the outbox
drain. Pre-store similarity queries are fail-open and timeout-bounded
(`CONFLICT_QUERY_TIMEOUT_S`) so a dead Qdrant cannot stall the coach write path.
See [ADR-003 § Store ack semantics](adr/ADR-003-coach-core-reliability.md#store-ack-semantics-init-003spec-004).

| Tool | Purpose |
|------|---------|
| `archivist_store` | Write a memory with entity extraction, conflict checks, LLM dedup, optional provenance envelope; ack after durable outbox commit; additive `embed_deferred` / lag fields when defer enabled |
| `archivist_delete` | Forget path: `mode=delete` (default soft-delete) or `mode=suppress` (hide, keep record). Namespace write RBAC required. |
| `archivist_merge` | Merge conflicting entries (latest / concat / semantic / manual) |
| `archivist_compress` | Archive memories and return compact summaries (flat or structured Goal/Progress/Decisions/Next Steps) |
| `archivist_pin` | Pin a memory or entity to retention class `permanent` — sets importance to 1.0 |
| `archivist_unpin` | Remove the permanent pin from a memory or entity |

## Store / Index / Forget contract

<!-- INIT-003/SPEC-006 -->
<!-- INIT-004/SPEC-003: archivist_index map-only dual shape -->
<!-- INIT-005/SPEC-006: store deferred / searchable-lag success fields -->

Coach-path enrichment of the ADR-003 five-tool contract (no net-new core tools):

| Tool | Contract |
|------|----------|
| **`archivist_store`** | Additive provenance args: `source`, `subject`, `purpose`, `sensitivity` (`standard`\|`sensitive`\|`secret`\|`health`\|`public`), `statement_kind` (`user`\|`inferred`), plus existing `actor_*` / `confidence`. Persisted on `memory_chunks` (+ Qdrant payload). Optional `correction_of` links the new id as superseding the prior via SPEC-007 `correct_memory`. Size/enum validated. Namespace **write** RBAC. Success JSON is additive (see **Store ack lag fields** below). |
| **`archivist_index`** | **Progressive-disclosure map, not evidence** ([ADR-004](adr/ADR-004-llm-native-coach-memory-surfaces.md) GR-CE-001). Returns JSON `{markdown, map}` under one tool: entity/type pointers, pinned/recent names, and search hints (~500-token intent). **No key-fact prose** — do not cite the TOC; use `archivist_search` / `archivist_get_context` `memories[]` for provenance-bearing facts. **No synchronous LLM** on this path (GR-CE-002). Rebuilds from the **live graph every call** (no index TTL). After store, the next index includes new entities; store also invalidates the search hot cache (`HOT_CACHE_TTL_SECONDS`, default 600s). Recommended coach sequence: `get_context(mode=bootstrap)` → turn evidence via `search`/`get_context` → navigational refresh via `index` as a **map only**. |
| **`archivist_delete`** (ADR: forget) | `mode=delete` (default): governed soft-delete via SPEC-007 `delete_memory`. `mode=suppress`: SPEC-007 `suppress_memory` — hidden from default recall, row remains. Cross-namespace mutations require write RBAC on the target namespace. |

### Store ack lag fields (`ARCHIVIST_EMBED_DEFER`)

<!-- INIT-005/SPEC-006 · ADR-005 GR-LAG-001 / GR-DUR-001 -->

Ack still means **durable graph + outbox** ([ADR-003](adr/ADR-003-coach-core-reliability.md)); Qdrant upsert remains async. Opt-in **`ARCHIVIST_EMBED_DEFER`** (default **`false`**) additionally skips the blocking primary embed on the ack path so vector rank may lag until outbox drain embeds then upserts. See [ADR-005](adr/ADR-005-coach-path-performance.md).

| Success field | When | Semantics |
|---------------|------|-----------|
| `stored`, `memory_id`, `uri`, `namespace`, `provenance`, … | Always on success | Unchanged INIT-003 / INIT-004 ack fields — **not removed**. |
| `duration_ms` | Always on success | Store-ack wall clock (ms). |
| `stage_timings` | Always on success | Numeric stage map (e.g. `embed_ms`, optional `conflict_ms`). Observability only — **no** fact text or embedding vectors. |
| `embed_deferred` | Always on success (`bool`) | `true` when this store used the embed-defer path (primary and/or micro-chunk vectors left empty for drain fill). |
| `searchable_lag_hint` | Only when `embed_deferred` is `true` | Stable string `vector_rank_may_lag_until_outbox_drain`. Clients must not treat hybrid/vector rank as ready at ack; FTS/needle/graph durability at ack is unchanged. Empty or incomplete vector hit during lag is **cite-or-refuse OK** (ADR-004 empty-OK). |
| `searchable_lag_metric` | Always on success | Prometheus gauge name for searchable-vector lag: `archivist_outbox_lag_seconds` (code alias `SEARCHABLE_LAG_SECONDS`). Scraped via `GET /metrics`. |

**SLO pointer (ADR-005):** p95 drain-to-searchable **≤ 5s** on fake-embed / coach_core lanes; production real-embed uses the lag gauge + “eventually searchable” rather than a brittle live-provider wall-clock. Operator details: [`docs/QA.md`](QA.md) (INIT-005 embed-defer section).

**Security:** store success JSON must **never** include embedding vectors, API keys, or raw secret material — only flags, timings, ids, and validated provenance labels.

## Trajectory & Feedback (5)

| Tool | Purpose |
|------|---------|
| `archivist_log_trajectory` | Log execution trajectory (task + actions + outcome), auto-extract tips |
| `archivist_annotate` | Add quality annotations (note, correction, stale, verified, quality) to a memory |
| `archivist_rate` | Rate a memory as helpful (+1) or unhelpful (-1) |
| `archivist_tips` | Retrieve strategy/recovery/optimization tips from past trajectories |
| `archivist_session_end` | Summarize a session into durable memory |

## Context Assembly & Handoff (3)

| Tool | Purpose |
|------|---------|
| `archivist_get_context` | High-level token-budgeted context assembly — tiers, graph facts, and **procedural tips** (the tip-only lesson API) in one call. Supports `mode=normal\|bootstrap` (INIT-004/SPEC-004). Tips: `include_tips` (default true) returns `tips[]` as strings from SQLite `tip_text`, ranked by task/query when provided ([ADR-007](adr/ADR-007-procedural-memory-wedge.md); skill registry retired — [ADR-008](adr/ADR-008-retire-skills-tip-lessons.md)). |
| `archivist_handoff` | Package a session's summary, goals, tips, hottest memories, and knowledge snapshot into a structured `HandoffPacket`. |
| `archivist_receive_handoff` | Inject a `HandoffPacket` into the receiving agent's ephemeral `SessionStore`. |

## Agent Checkpoints (5)

| Tool | Purpose |
|------|---------|
| `archivist_checkpoint_save` | Persist agent-state checkpoint (payload + optional parent) scoped to namespace. |
| `archivist_checkpoint_list` | List session checkpoints in a namespace (oldest first). |
| `archivist_checkpoint_get` | Fetch one checkpoint by id **and** namespace (no cross-tenant get by id alone). |
| `archivist_checkpoint_resume` | Inject resume packet into the caller's `SessionStore` for that session only. |
| `archivist_checkpoint_replay` | Read-only parent-chain reconstruction (metadata + payloads). |

## Coordination Beyond Handoff (5)

<!-- INIT-009/SPEC-004 -->

Selective share + consensus v1 (explicit accept/reject + audit) — Unique Differentiator
**#5** productized ([ADR-009](adr/ADR-009-native-multi-agent-coordination.md)). Available on
**ops** and **full** (not **core**). Extends — does not replace —
`archivist_handoff` / `archivist_receive_handoff` (GR-HANDOFF-001).

**Lessons / tips:** Procedural lessons remain tip-only ([ADR-007](adr/ADR-007-procedural-memory-wedge.md) /
[ADR-008](adr/ADR-008-retire-skills-tip-lessons.md)). Cross-agent tip transfer:
(1) primary — tips in `HandoffPacket`; (2) selective — `tip_ids` on
`archivist_share_propose` (grant metadata + SessionStore `share_tip_ids` on accept).

Conflict outcomes use `contradiction_resolve` actions: `supersede` \| `merge` \| `keep_both`.
Optional `apply=true` on attach invokes `apply_resolution` (dry_run defaults true).

| Tool | Purpose |
|------|---------|
| `archivist_share_propose` | Propose selective share of `memory_ids`, `tip_ids`, and/or `scope` to another agent (pending grant). Proposer needs namespace read. |
| `archivist_share_accept` | Recipient accepts a grant (audited; idempotent). Injects `share_memory_ids` / `share_tip_ids` into SessionStore when present. Optional `materialize_namespace` requires write RBAC. |
| `archivist_share_reject` | Recipient rejects a grant (audited; idempotent). |
| `archivist_share_attach_conflict` | Attach conflict outcome (`supersede` / `merge` / `keep_both`); optional `apply` → resolver. |
| `archivist_share_get` | Fetch one grant by id + namespace (proposer or recipient). |

## Admin & Context Management (10)

| Tool | Purpose |
|------|---------|
| `archivist_context_check` | Pre-reasoning token counting against a budget with compaction hints |
| `archivist_namespaces` | List namespaces visible to an agent |
| `archivist_audit_trail` | View immutable audit log entries |
| `archivist_resolve_uri` | Resolve `archivist://` URIs to their underlying resource |
| `archivist_retrieval_logs` | Export/analyze retrieval pipeline execution traces |
| `archivist_health_dashboard` | Single-pane health: memory counts, stale %, conflict rate, cache |
| `archivist_batch_heuristic` | Recommended batch size (1-10) from health signals |
| `archivist_savings_dashboard` | Token savings stats: avg/min/max savings %, total tokens saved, per-policy breakdown, estimated USD (`null` if `TOKEN_USD_PER_1K` unset), hotness heatmap (top-N memories). |
| `archivist_memory_lineage` | Lineage edges for a memory/entity (provenance, versions, audit, retrieval mentions). RBAC on namespace reads; no secrets/full text. |
| `archivist_backup` | Create, list, restore, or delete memory snapshots (Qdrant + SQLite/Postgres). Supports `export_agent` / `import_agent` for portable agent migration. |

### Memory-as-Product (service layer)

Service APIs in `archivist.storage.memory_product` (INIT-001/SPEC-009) — not MCP tools yet:

| API | Purpose |
|-----|---------|
| `create_scope_snapshot` | Versioned snapshot of a namespace (optional `agent_id` filter); archive under `BACKUP_DIR` |
| `fork_from_snapshot` | Copy snapshot into a target namespace with `parent_version_id` lineage; vectors via outbox |
| `export_scope` | Export archive path + manifest (counts/versions); paths confined by `SnapshotPathError` / `_snapshot_dir` |

All three require `caller_agent_id` and enforce namespace RBAC. Distinct from Phase-7 `agent_checkpoints`.

## Cache Management (2)

| Tool | Purpose |
|------|---------|
| `archivist_cache_stats` | Hot cache stats (entries per agent, TTL, hit rate) |
| `archivist_cache_invalidate` | Manual eviction by namespace, agent, or all |

## Reference Docs (1)

| Tool | Purpose |
|------|---------|
| `archivist_get_reference_docs` | Return the full Archivist tool skill reference from inside the server. Optionally pass `section` to filter to a heading (e.g. `search`, `storage`, `admin`). |

## get_context modes & budgets

<!-- INIT-004/SPEC-004 -->

`archivist_get_context` accepts `mode`:

| Mode | Purpose | Default `max_tokens` |
|------|---------|----------------------|
| `normal` (default) | Turn recall — tier-packed sources + stable `memories[]` | **2000** (coach-oriented; was 8000) |
| `bootstrap` | Session start — compact identity / critical pointers / map slice (~200–400 tok spirit) | **400** |

- **Explicit `max_tokens` always wins** over the mode default.
- Bootstrap keeps the same success shape (`memories`, `sources`, `answer`,
  `context_status`, …). It does **not** invent `memories[]` rows from the map;
  empty `memories[]` is success (GR-CE-003). Prefer bootstrap over promoting
  `archivist_wake_up` into the **core** profile (`wake_up` stays ops/full).
- Recommended coach sequence: `get_context(mode=bootstrap)` → turn evidence via
  `search` / `get_context` (budgeted) → navigational refresh via `archivist_index`
  (map only — see ADR-004).
- Namespace RBAC remains on the handler for both modes.

## Stable recall (search / get_context)

<!-- INIT-003/SPEC-005 -->

Both `archivist_search` and `archivist_get_context` return a **canonical**
`memories` array alongside existing keys (`sources`, `answer`, …) for
backward compatibility:

```json
{
  "answer": "",
  "memories": [
    {
      "id": "<memory_id>",
      "text": "<non-empty usable text when hits exist>",
      "score": 0.87,
      "provenance": {
        "namespace": "agents-alice",
        "subject": "optional",
        "purpose": "optional",
        "sensitivity": "standard",
        "source": "user|system|…",
        "confidence": 0.9,
        "statement_kind": "user|inferred",
        "agent_id": "alice",
        "date": "2026-07-25"
      }
    }
  ],
  "sources": [ "…legacy shape preserved…" ]
}
```

**Contract notes**

- When hits exist, each `memories[].text` is non-empty and usable. Prefer
  `memories` over dual-calling search + get_context for text. `answer` may
  still be empty when `refine=false` (search) / default get_context path.
- **Empty recall is OK (GR-CE-003 / INIT-004/SPEC-004):** when filters or
  thresholds yield nothing, `memories: []` is a valid success — cite-or-refuse;
  do not invent facts. Bootstrap mode likewise returns empty `memories[]`
  rather than fabricating memory rows from the map.
- **Pre-rank filters** (applied before ranking; cannot be disabled to widen
  tenants): suppressed rows omitted; superseded losers omitted by default
  (SPEC-007 visibility helpers); `namespace` / optional `subject` hard-scope
  isolation; optional `purpose` / `sensitivity` further narrow (default = no
  purpose restriction within namespace).
- RBAC / namespace read gates remain on both handlers.
- Provenance in the response is a **safe subset** (ids, axes, timestamps) —
  never API keys, tokens, passwords, or other secrets.

## Usage Hints

- `min_score` / `RETRIEVAL_THRESHOLD`: set to `0` to disable score filtering for a single call when debugging recall.
- Prefer `archivist_get_context` for agent pre-prompt injection — it assembles tiers, graph facts, and tips in one token-budgeted call. Read `memories[].text` for usable recall when `answer` is empty.
- Use `archivist_get_context(mode=bootstrap)` at session start for a compact identity/pointers/map payload (~200–400 tokens). `archivist_wake_up` remains on ops/full profiles and is not required on core.
- Use `archivist_search` for explicit queries; `archivist_recall` when entity names are known.
- Use `archivist_entity_brief` when you need a structured knowledge card — faster than multiple search/recall calls.
- Use `archivist_context_check` before reasoning to decide if context compaction is needed.
- Use `archivist_compress` with `format: structured` for Goal/Progress/Decisions/Next Steps summaries.
- Log trajectories so future searches benefit from outcome-aware retrieval scoring.
- Pin critical facts (host IPs, credentials, ownership) with `archivist_pin` so the curator never forgets them.
- Use `archivist_handoff` + `archivist_receive_handoff` to transfer session context **and tips** between agents (primary tip-transfer channel).
- Use `archivist_share_propose` / `accept` / `reject` on **ops**/**full** for selective memory/`tip_ids` grants (Diff #5; does not replace handoff; not on **core**).
- Use `archivist_share_attach_conflict` to record conflict outcomes (`supersede` / `merge` / `keep_both`); set `apply=true` to invoke the contradiction resolver (dry_run default).
- Use `archivist_checkpoint_save` / `resume` / `replay` for Phase-7 agent-state time-travel (**full** profile; distinct from handoff and from L0–L2 tiers).
- Check `archivist_savings_dashboard` to measure how much token waste the Answer Finder is eliminating (set `TOKEN_USD_PER_1K` for estimated USD fields).
- Use `archivist_memory_lineage` to inspect provenance/version/audit/retrieval edges for a memory or entity (requires namespace read access).

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

## Lifecycle (suppress / supersede / delete)

<!-- INIT-003/SPEC-007 -->

Service APIs in `archivist.lifecycle.correct` (and visibility helpers in
`archivist.lifecycle.visibility`) govern coach-path memory state. Handlers
(SPEC-006) and retrieval (SPEC-005) should call these — not flip flags ad hoc.

| Operation | Effect | Default recall |
|-----------|--------|----------------|
| **suppress** (`suppress_memory`) | Sets `memory_chunks.is_suppressed=1` (namespace-scoped). Record remains for audit/ops. Not a hard erase. | Hidden |
| **unsuppress** (`unsuppress_memory`) | Explicit restore only — no silent resurrection of suppressed rows as instruction-grade. | Visible again |
| **supersede / correct** (`supersede_memory` / `correct_memory`) | Winner links `supersedes_id` → prior id. Loser stays on disk. Prefer `correct_memory` when store has `correction_of`. | Loser hidden; winner present |
| **delete** (`delete_memory`) | Soft-delete / tombstone via existing `soft_delete_memory` + background cascade. Second delete is idempotent (`already_deleted`). | Hidden |

Default recall predicates (`is_recall_visible`, `recall_visible_sql_chunks` /
`recall_visible_sql_facts`) exclude suppressed rows and superseded losers
unless the caller opts in. Lifecycle audit metadata carries ids/status only —
never memory text or secrets.

## See Also

- [CURSOR_SKILL.md](CURSOR_SKILL.md) — full parameter schemas and examples
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design and module map
- [INSPIRATION.md](INSPIRATION.md) — ReMe comparison and design rationale
