# Archivist Architecture

## Overview

Archivist is a memory service for AI agent fleets. It combines three storage backends:

1. **Qdrant** — Vector store for semantic search over chunked documents
2. **SQLite** — Temporal knowledge graph for entities, relationships, and facts
3. **File System** — Source of truth (markdown files watched for changes)

## Data Flow

### Ingestion

```
Markdown Files (MEMORY_ROOT)
    │
    ├──→ File Watcher (watchfiles)
    │        │
    │        ▼
    │    Chunker (indexer.py)
    │        │
    │        ├──→ Flat chunks (legacy mode)
    │        │
    │        └──→ Hierarchical chunks (Phase 1)
    │                 │
    │                 ├── Parent chunks (2000 chars)
    │                 │       │
    │                 │       └── Child chunks (500 chars)
    │                 │             with parent_id reference
    │                 │
    │                 ▼
    │            Embedding API
    │                 │
    │                 ▼
    │           Qdrant Upsert
    │
    └──→ Curator (background loop)
             │
             ▼
         LLM Extraction
             │
             ├──→ Entities → SQLite entities table
             ├──→ Facts → SQLite facts table
             └──→ Relationships → SQLite relationships table
```

### Retrieval (RLM Pipeline)

```
Query
  │
  ▼
Stage 1: Coarse Vector Search (Qdrant, optional date_from/date_to range filter)
  │
  ▼
Stage 1a: Graph Augmentation (v0.5)
  │        (entity extraction → multi-hop KG context → merge into results)
  │
  ▼
Stage 1b: Dedupe (memory_fusion — same file/chunk/text)
  │
  ▼
Stage 1c: Temporal Decay (v0.5)
  │        (exponential decay based on age, configurable halflife)
  │
  ▼
Stage 1d: Hotness Scoring (v1.0)
  │        (frequency × recency boost from precomputed scores)
  │
  ▼
Stage 2: Threshold Filter (score >= RETRIEVAL_THRESHOLD or per-call min_score)
  │
  ▼
Stage 3: Cross-Encoder Rerank (optional)
  │
  ▼
Stage 4: Parent-Child Enrichment
  │        (fetch parent context for child matches)
  │
  ▼
Stage 4b: Context Budget (v0.5)
  │         (cap chunks by max_tokens; tier selection: l0/l1/l2)
  │
  ▼
Stage 5: LLM Refinement
  │        (per-chunk relevance extraction, graph context included)
  │
  ▼
Stage 6: LLM Synthesis
  │        (combine extractions into answer)
  │
  ▼
Response with sources + retrieval_trace + context_status (v1.0)
```

## Module Map

| Module | Responsibility |
|--------|---------------|
| `main.py` | Entry point, Starlette app, startup tasks, background task tracking |
| `mcp_server.py` | Thin MCP orchestrator (~30 lines), delegates to `mcp/` package |
| `mcp/_registry.py` | Central tool registry: aggregates tools and handlers from domain modules |
| `mcp/_common.py` | Shared helpers: `_rbac_gate`, error/success formatters |
| `mcp/tools_search.py` | 7 search/retrieval handlers (search, recall, timeline, insights, deref, index, contradictions) |
| `mcp/tools_storage.py` | 3 storage handlers (store, merge, compress) |
| `mcp/tools_trajectory.py` | 5 trajectory handlers (log_trajectory, annotate, rate, tips, session_end) |
| `mcp/tools_skills.py` | 6 skill registry handlers (register, event, lesson, health, relate, dependencies) |
| `mcp/tools_admin.py` | 7 admin handlers (context_check, namespaces, audit_trail, resolve_uri, retrieval_logs, health_dashboard, batch_heuristic) |
| `mcp/tools_cache.py` | 2 cache handlers (cache_stats, cache_invalidate) |
| `rlm_retriever.py` | RLM recursive retrieval pipeline |
| `reranker.py` | Cross-encoder reranking (optional) |
| `indexer.py` | File chunking, embedding, Qdrant indexing |
| `graph.py` | SQLite schema, entity/relationship CRUD |
| `graph_retrieval.py` | Hybrid vector+graph, temporal decay, multi-hop, contradictions |
| `tiering.py` | L0/L1/L2 summary generation and selection |
| `compressed_index.py` | Per-namespace semantic TOC generation |
| `trajectory.py` | Trajectory logging, attribution, tips, annotations, ratings |
| `skills.py` | Skill registry, version tracking, lessons learned, usage events, health scoring |
| `hot_cache.py` | Per-agent LRU/TTL hot cache (middle layer of three-tier hierarchy) |
| `archivist_uri.py` | `archivist://` URI scheme — parse, construct, resolve addressable references |
| `retrieval_log.py` | Retrieval trajectory logging, export, and aggregate analytics |
| `metrics.py` | Prometheus-compatible metrics: counters, histograms, gauges, text exposition |
| `webhooks.py` | Async webhook dispatcher for memory/conflict/skill events |
| `dashboard.py` | Health dashboard aggregation and batch-size heuristic |
| `curator_queue.py` | Write-ahead curator queue: enqueue/drain/stats for deferred curation ops |
| `hotness.py` | Hotness scoring (frequency × recency) with batch aggregation |
| `embeddings.py` | Embedding API client |
| `llm.py` | LLM API client |
| `config.py` | Environment variable configuration |
| `rbac.py` | Namespace access control |
| `curator.py` | Background knowledge extraction |
| `audit.py` | Immutable audit logging |
| `merge.py` | Memory merge strategies |
| `versioning.py` | Memory version tracking |
| `conflict_detection.py` | Pre-write conflict detection |
| `context_manager.py` | Token budget checks, message splitting, compaction hints |
| `compaction.py` | Structured conversation compaction (Goal/Progress/Decisions/Next Steps) |
| `tokenizer.py` | Token counting (tiktoken cl100k_base with chars//4 fallback) |

## Storage Schema

### Qdrant Payload Fields

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | keyword | Source agent |
| `file_path` | keyword | Relative path from MEMORY_ROOT |
| `file_type` | keyword | daily, durable, system, explicit, merged |
| `team` | keyword | Agent's team |
| `date` | keyword | ISO date |
| `namespace` | keyword | RBAC namespace |
| `text` | text | Chunk content (L2 — full detail) |
| `l0` | text | One-sentence abstract (~100 tokens) |
| `l1` | text | Overview paragraph (~500 tokens) |
| `chunk_index` | integer | Position in source file |
| `parent_id` | keyword | Parent chunk ID (hierarchical) |
| `is_parent` | bool | Whether this is a parent chunk |
| `version` | integer | Monotonic version |
| `importance_score` | float | 0.0–1.0 retention score |
| `ttl_expires_at` | integer | Unix timestamp for expiry |
| `checksum` | keyword | Content hash for dedup |

### SQLite Tables

- **entities** — Named entities with type, mention count, first/last seen
- **relationships** — Typed edges between entities with evidence and confidence
- **facts** — Text facts linked to entities, with active/superseded status
- **curator_state** — Key-value store for curator bookkeeping
- **audit_log** — Immutable log of all memory operations
- **memory_versions** — Version history per memory ID
- **trajectories** — Execution traces with task, actions, outcome, linked memory IDs
- **tips** — Strategy/recovery/optimization learnings extracted from trajectories
- **annotations** — Quality notes and corrections on memory points
- **ratings** — Up/down votes on memories
- **memory_outcomes** — Links memories to trajectory outcomes for retrieval scoring
- **skills** — Registered skill/tool catalog (name, provider, endpoint, version, status)
- **skill_versions** — Per-skill version history with changelog and breaking changes
- **skill_lessons** — Operational lessons learned per skill (failure modes, workarounds, best practices)
- **skill_events** — Usage event log for invocation outcomes, duration, errors, trajectory links
- **retrieval_logs** — Persisted retrieval pipeline executions (query, trace, duration, cache hit)
- **curator_queue** — Write-ahead queue for deferred curation operations (merge, archive, consolidate, hotness)
- **memory_hotness** — Precomputed hotness scores per memory ID (frequency × recency)
- **skill_relations** — Typed edges between skills (similar_to, depend_on, compose_with, replaced_by)

## v0.4.0 operational notes

- **HTTP auth** — Set `ARCHIVIST_API_KEY`; clients send `Authorization: Bearer <key>` or `X-API-Key`. `GET /health` is never authenticated (Kubernetes probes).
- **SQLite writes** — Graph mutations, audit inserts, version inserts, and curator fact decay take a process-wide `GRAPH_WRITE_LOCK` to avoid concurrent write races.
- **Retrieval trace** — `archivist_search` JSON includes `retrieval_trace`: coarse hit counts, dedupe/threshold/rerank stages, and rerank settings.
- **Store conflicts** — Before `archivist_store`, optional Qdrant similarity vs *other* agents in the same namespace; block or allow via env + `force_skip_conflict_check`.
- **Explicit store IDs** — Qdrant point IDs for explicit stores are UUIDs (not content hashes).

## v0.5.0 operational notes

- **Tiered context** — On ingest, parent chunks get auto-generated L0 (~20 words) and L1 (~2–4 sentences) summaries via LLM. Child chunks inherit their parent's summaries. Controlled by `TIERED_CONTEXT_ENABLED`; falls back to truncation if the LLM is unreachable.
- **Graph-augmented retrieval** — Entity mentions in the query are matched against the knowledge graph; facts and relationships up to `MULTI_HOP_DEPTH` hops are merged into vector results with weight `GRAPH_RETRIEVAL_WEIGHT`.
- **Temporal decay** — Results are weighted by recency: `score × exp(-ln2 × age_days / halflife)`. Set `TEMPORAL_DECAY_HALFLIFE_DAYS=0` to disable.
- **Context budget** — `archivist_search` accepts `max_tokens` to cap approximate context size and `tier` (l0/l1/l2) to control detail level. Use `l0` + `max_tokens=400` for pre-message injection; `l2` for deep research.
- **Progressive dereference** — `archivist_deref` fetches full L2 text for a point ID. Intended for L0/L1→L2 drill-down after a compact search.
- **Compressed index** — `archivist_index` returns a ~500-token navigational summary of what entities/topics exist in a namespace, enabling cross-domain bridging.
- **Contradiction surfacing** — `archivist_contradictions` uses keyword-heuristic detection on facts from different agents about the same entity.
- **Date range filters** — `archivist_search` accepts `date_from` and `date_to` for Qdrant payload range filtering.

## v0.6.0 operational notes

- **Trajectory logging** — `archivist_log_trajectory` stores task/actions/outcome in a `trajectories` SQLite table and creates a graph fact. If `memory_ids_used` is provided, the handler runs LLM-based attribution (linking outcomes to influential memories) and tip extraction (strategy/recovery/optimization learnings).
- **Outcome-aware retrieval** — After threshold filtering, the retriever checks `memory_outcomes` for each hit. Memories linked to successful trajectories get `+OUTCOME_RETRIEVAL_BOOST` × influence weight; failures get `−OUTCOME_RETRIEVAL_PENALTY`. Logged in `retrieval_trace.outcome_adjustments`.
- **Annotations** — `archivist_annotate` stores typed notes (note/correction/stale/verified/quality) with optional quality scores. All writes are audit-logged.
- **Ratings** — `archivist_rate` records +1/−1 votes per agent per memory, with aggregate summaries available via the response. Clamped to [-1, 1].
- **Tips** — `archivist_tips` retrieves past learnings, filterable by category.
- **Session-end** — `archivist_session_end` aggregates all trajectories for a session_id, produces an LLM summary, and optionally stores it as a durable memory with importance 0.8.

## v0.7.0 operational notes

- **Skill registry** — Four new SQLite tables (`skills`, `skill_versions`, `skill_lessons`, `skill_events`) track externally provided MCP tools. All writes are serialized via `GRAPH_WRITE_LOCK`.
- **Skill tools** — `archivist_register_skill` (upsert by name+provider), `archivist_skill_event` (log invocation outcome), `archivist_skill_lesson` (add failure mode / workaround / best practice), `archivist_skill_health` (compute success rate, latency, health grade from recent events).
- **Health scoring** — Aggregates over a configurable `window_days` period. Grades: `healthy` (≥80% success), `warning` (50–80%), `degraded` (<50%), `broken`/`deprecated` (status override).
- **Version tracking** — Each skill maintains a version log with changelog and breaking change annotations. Version bumps are automatically recorded on registration updates.
- **Memory type routing** — `archivist_store` accepts `memory_type` (experience / skill / general). `archivist_search` can filter by type, enabling experience-vs-skill retrieval routing. Qdrant payload index on `memory_type`.

## v0.8.0 operational notes

- **Three-layer hierarchy** — Session/ephemeral → per-agent hot cache → long-term (Qdrant + SQLite). The hot cache (`hot_cache.py`) is an in-process LRU/TTL store keyed by `(agent_id, query_hash)`. Configurable via `HOT_CACHE_ENABLED`, `HOT_CACHE_MAX_PER_AGENT` (default 128), `HOT_CACHE_TTL_SECONDS` (default 600).
- **Cache invalidation** — `archivist_store` automatically evicts all cache entries for the written namespace. Manual control via `archivist_cache_invalidate` (by namespace, agent, or all). `archivist_cache_stats` returns per-agent entry counts.
- **`archivist://` URIs** — Format: `archivist://{namespace}/{memory|entity|namespace|skill}/{id}`. `archivist_store` and `archivist_deref` responses include `uri` fields. `archivist_resolve_uri` resolves any URI to its underlying resource (dispatches to deref, recall, index, or skill_health internally).
- **Retrieval trajectory logging** — Every pipeline execution is persisted to `retrieval_logs` SQLite table with query, agent, tier, cache hit flag, duration_ms, and full `retrieval_trace` JSON. Available via `archivist_retrieval_logs` (MCP) and `GET /admin/retrieval-logs` (REST).
- **Retrieval analytics** — `archivist_retrieval_logs` with `stats_only=true` returns aggregate stats: total queries, cache hit rate, avg/min/max duration, top agents.
- **Consistency config** — `DEFAULT_CONSISTENCY` env var (default `eventual`) sets the baseline consistency level. Namespace configs can override to `session` or `strong`. The consistency level is stored in Qdrant payloads and used by RBAC/merge logic.

## v0.9.0 operational notes

- **Prometheus metrics** — `GET /metrics` returns counters, histograms, and gauges in text exposition format. No external dependency — pure-Python implementation in `metrics.py`. Instrumented: search total/duration, store total, conflicts, cache hit/miss, webhook fire/fail, skill events, LLM calls/errors.
- **Webhooks** — `webhooks.py` fires HTTP POST to `WEBHOOK_URL` on `memory_store`, `memory_conflict`, and `skill_event` events. Filter with `WEBHOOK_EVENTS` (comma-separated). Fire-and-forget via `asyncio.create_task` — never blocks the caller. Failures are metrics-counted and logged.
- **Health dashboard** — `dashboard.py` aggregates Qdrant stats (point count, status), stale memory estimate (TTL-based), audit conflict rate, retrieval stats (cache hit rate, avg latency), skill health overview (status breakdown, success rate), and cache status. Available via `archivist_health_dashboard` MCP tool and `GET /admin/dashboard`.
- **Batch heuristic** — Recommends batch size (1–10) from health signals. Downgrades on high conflict rate (>20% → −2, >5% → −1), high stale % (>30% → −2, >10% → −1), low cache hit rate (<10% → −0.5), and degraded skills (>2 → −1). Available via `archivist_batch_heuristic` MCP tool and `GET /admin/dashboard?batch=true`.

## v1.0.0 operational notes

- **Write-ahead curator queue** — `curator_queue.py` provides `enqueue(op_type, payload)` and `drain(limit)`. Op types: `merge_memory`, `delete_memory`, `consolidate_tips`, `update_hotness`, `archive_memory`, `skip_store`. Drained every `CURATOR_QUEUE_DRAIN_INTERVAL` seconds (default 30) by a background asyncio task in `main.py`. All ops applied under `GRAPH_WRITE_LOCK`.
- **LLM-adjudicated dedup** — On `archivist_store`, if vector similarity exceeds `DEDUP_LLM_THRESHOLD` (default 0.80), the LLM decides: `skip` (reject store, return existing URI), `create` (store normally), `merge` (enqueue merge op), `delete` (enqueue archival of old). Below threshold, fast cosine-only check. Controlled by `DEDUP_LLM_ENABLED`.
- **Tip consolidation** — `trajectory.consolidate_tips()` clusters tips by embedding similarity (≥0.85), LLM-merges clusters of 3+ into canonical tips with `negative_example` anti-patterns, archives originals. Budget-capped by `CURATOR_TIP_BUDGET` (default 20) LLM calls per cycle.
- **`archivist_compress`** — Agents call this tool with a list of `memory_ids`. The handler fetches L2 text, generates (or accepts) a summary, stores it as `memory_type: "compressed"`, and enqueues archival of originals (set `archived: true` in Qdrant payload — excluded from default search).
- **Context-status signaling** — Every `archivist_search` response includes `context_status` in `retrieval_trace`: `result_tokens_approx`, `budget_tokens`, `budget_used_pct`, `tier`, and `hint` ("compress" if budget >80% used).
- **Hotness scoring** — `hotness.py` computes `sigmoid(log1p(retrieval_count)) × exp(−ln2 × days/halflife)`. Batch aggregation from `retrieval_logs` into `memory_hotness` table. Blended into RLM results via `(1 − HOTNESS_WEIGHT) + HOTNESS_WEIGHT × hotness`. Config: `HOTNESS_WEIGHT` (default 0.15), `HOTNESS_HALFLIFE_DAYS` (default 7).
- **Skill relation graph** — `skill_relations` table tracks typed edges between skills: `similar_to`, `depend_on`, `compose_with`, `replaced_by`. `archivist_skill_relate` creates/updates relations. `archivist_skill_dependencies` returns the relation subgraph (configurable depth). `archivist_skill_health` now includes `related_skills` (substitutes/dependencies).
- **Curator agent persona** — `prompts/curator.md` defines the NemoClaw Curator: a memory librarian that checks health, compresses stale memories, resolves contradictions, and reports findings. Designed for cron-scheduled execution in an OpenClaw sandbox.
- **Curator metrics** — Five new Prometheus metrics instrument the curation pipeline: `archivist_curator_queue_depth` (gauge), `archivist_curator_dedup_decisions_total` (counter by decision label), `archivist_curator_tip_consolidations_total` (counter), `archivist_curator_llm_calls_total` (counter), `archivist_curator_drain_duration_ms` (histogram).

## v1.0.1 operational notes

- **MCP server refactor** — Split monolithic 1,694-line `mcp_server.py` into `src/mcp/` package with 8 modules by domain (search, storage, trajectory, skills, admin, cache) plus shared helpers (`_common.py`) and central registry (`_registry.py`). `mcp_server.py` is now ~30 lines. Handler dispatch uses `dict.get()` instead of 58-branch if/elif chain.
- **Fix `decay_old_entries()` row count** — Replaced `conn.total_changes` (cumulative for connection) with `cursor.rowcount` (per-statement) in `curator.py`.
- **Track background tasks** — `startup()` stores task references in `_background_tasks` with named tasks and `done_callback` for crash logging. Same pattern applied to `webhooks.py` `fire_background()`.
- **Non-root Docker** — Container runs as `archivist` user (UID/GID 1000 by default, configurable via `--build-arg`). Data directories pre-created and chowned.

## v1.1.0 operational notes

- **Token counting** — New `tokenizer.py` with tiktoken cl100k_base (lazy-loaded) + `chars//4` fallback. `count_tokens()` for strings, `count_message_tokens()` for chat message lists with per-message overhead.
- **Context manager** — New `context_manager.py`: `check_context()` analyzes messages against token budget, returns usage %, hint (ok/compress/critical), split recommendations. `check_memories_budget()` for raw text lists.
- **Structured compaction** — New `compaction.py`: `compact_structured()` produces Goal/Progress/Decisions/Next Steps/Critical Context JSON via LLM. Supports incremental compaction with `previous_summary`. Falls back to flat summary on parse failure.
- **`archivist_context_check` MCP tool** — Pre-reasoning context check: agents send messages or memory texts, get token count, budget usage, and compaction hint. Added to `tools_admin.py`.
- **Upgraded `archivist_compress`** — New `format` parameter (`flat`/`structured`) and `previous_summary` for incremental compaction. Delegates to `compaction` module.
- **Retriever token counting** — `rlm_retriever.py` budget cap and context-status signaling now use `tokenizer.count_tokens()` instead of `len(text) // 4`.
- **Config** — `DEFAULT_CONTEXT_BUDGET` env var (default 128000 tokens).
