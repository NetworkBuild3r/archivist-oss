---
name: archivist
description: Persistent, cross-agent memory — store, search, and recall structured knowledge across sessions.
metadata:
  openclaw:
    emoji: "🧠"
    homepage: "https://github.com/NetworkBuild3r/archivist-oss"
---

# Archivist Memory Skill

Archivist is a long-term memory service for agent fleets. It combines semantic vector search (Qdrant), full-text BM25 keyword search (SQLite FTS5 / PostgreSQL), and a knowledge graph (entities + facts + relationships) into a single MCP interface. Use it whenever you need to remember something across sessions, share knowledge between agents, or recall structured facts about entities.

---

## When to Use Archivist

| Trigger | Action |
|---------|--------|
| User mentions something worth remembering (decisions, preferences, facts) | `archivist_store` |
| User asks "do you remember…" or queries past context | `archivist_search` |
| User asks about a specific entity ("what do we know about X?") | `archivist_entity_brief` or `archivist_recall` |
| Session starts — bootstrap working memory | `archivist_wake_up` |
| Session ends — persist a summary | `archivist_session_end` |
| User asks about available namespaces or agent scopes | `archivist_namespaces` |
| A critical fact must never be forgotten | `archivist_pin` |
| Memory is stale, wrong, or redundant | `archivist_delete` or `archivist_merge` |
| Context window is filling up | `archivist_context_check` → `archivist_compress` |
| User wants a chronological view of a topic | `archivist_timeline` |
| User wants cross-agent insights on a topic | `archivist_insights` |

---

## Connection

Archivist exposes **31 MCP tools** over Streamable HTTP (default) or SSE.

```
Streamable HTTP: http://<host>:3100/mcp
SSE (opt-in):    http://<host>:3100/mcp/sse
```

Authentication: `Authorization: Bearer <ARCHIVIST_API_KEY>` or `X-API-Key: <key>`.

---

## Session Startup Protocol

Call `archivist_wake_up` **once at the beginning of every session** to load identity, critical facts, and namespace overview in ~200 tokens.

```json
{
  "name": "archivist_wake_up",
  "arguments": {
    "agent_id": "your-agent-id"
  }
}
```

---

## Core Tools Reference

### Search & Retrieval

#### `archivist_search` — Semantic search (primary retrieval tool)

Runs the full 10-stage RLM pipeline: vector search → BM25 fusion → graph augmentation → decay → rerank → LLM refinement.

```json
{
  "name": "archivist_search",
  "arguments": {
    "query": "What decisions were made about the database migration?",
    "agent_id": "my-agent",
    "namespace": "engineering",
    "refine": true,
    "tier": "l1",
    "limit": 20
  }
}
```

**Key parameters:**
- `query` *(required)* — Natural language search query
- `agent_id` — Filter to one agent's memories; omit for fleet-wide search
- `agent_ids` — Search a specific set of agents (OR logic)
- `namespace` — Scope to a namespace (see Namespaces section)
- `refine` *(default: true)* — Enable LLM synthesis; set `false` for faster raw results
- `tier` *(default: l2)* — `l0` = abstract/title, `l1` = overview, `l2` = full text
- `limit` *(default: 20)* — Max chunks returned
- `min_score` — Override similarity threshold (0.0–1.0); `0` disables cutoff
- `date_from` / `date_to` — ISO date range filter (e.g. `"2026-01-01"`)
- `max_tokens` — Token budget for returned context
- `memory_type` — `experience`, `skill`, or `general`

**Tip:** Use `tier: l0` with `max_tokens` for lightweight pre-message context injection without consuming the full context window.

---

#### `archivist_recall` — Knowledge graph entity lookup

Multi-hop lookup for entities and the relationships/facts between them.

```json
{
  "name": "archivist_recall",
  "arguments": {
    "entity": "Kubernetes",
    "related_to": "ArgoCD",
    "agent_id": "my-agent"
  }
}
```

---

#### `archivist_entity_brief` — Structured knowledge card for an entity

Returns all known facts, relationships, mention count, and timeline for a named entity. Better than searching through unstructured memories for entity-centric questions.

```json
{
  "name": "archivist_entity_brief",
  "arguments": {
    "entity": "prod-db-01",
    "agent_id": "my-agent",
    "as_of": "2026-01-15"
  }
}
```

Use `as_of` (ISO date) to query facts valid at a point in time.

---

#### `archivist_timeline` — Chronological memory view

```json
{
  "name": "archivist_timeline",
  "arguments": {
    "query": "PostgreSQL migration",
    "agent_id": "my-agent",
    "days": 30
  }
}
```

---

#### `archivist_insights` — Cross-agent knowledge discovery

Surfaces what agents across the fleet know about a topic.

```json
{
  "name": "archivist_insights",
  "arguments": {
    "topic": "infrastructure cost reduction",
    "agent_id": "my-agent",
    "limit": 10
  }
}
```

---

#### `archivist_deref` — Fetch a specific memory by ID

Used after a search returns IDs. Retrieves full L2 text and metadata.

```json
{
  "name": "archivist_deref",
  "arguments": {
    "memory_id": "a1b2c3d4-...",
    "agent_id": "my-agent"
  }
}
```

---

#### `archivist_index` — Compressed navigational summary (~500 tokens)

Returns entity categories and top topics for a namespace. Use for cross-domain bridging or as a lightweight overview before searching.

```json
{
  "name": "archivist_index",
  "arguments": {
    "agent_id": "my-agent",
    "namespace": "engineering"
  }
}
```

---

#### `archivist_contradictions` — Surface conflicting facts

Finds contradicting facts about an entity from different agents via the knowledge graph. Call before storing sensitive factual updates.

```json
{
  "name": "archivist_contradictions",
  "arguments": {
    "entity": "deployment-pipeline",
    "agent_id": "my-agent"
  }
}
```

---

### Storage & Memory Management

#### `archivist_store` — Persist a memory

```json
{
  "name": "archivist_store",
  "arguments": {
    "text": "The migration to PostgreSQL was approved. Target date: Q2 2026. Owner: @alice.",
    "agent_id": "my-agent",
    "namespace": "engineering",
    "entities": ["PostgreSQL", "migration"],
    "importance_score": 0.9,
    "memory_type": "experience"
  }
}
```

**Key parameters:**
- `text` *(required)* — The memory content
- `agent_id` *(required)* — Your agent's ID
- `namespace` — Target namespace (auto-detected from `agent_id` if omitted)
- `entities` — Entity names to tag; auto-extracted if omitted
- `importance_score` *(default: 0.5)* — Retention priority 0.0–1.0. Use `>0.9` to prevent TTL expiry
- `memory_type` — `experience` (events/outcomes), `skill` (how-to), `general` (facts)
- `force_skip_conflict_check` — Set `true` only when you know the store is safe and speed matters

**What happens internally:** Entity extraction, conflict check, LLM-adjudicated dedup, FTS5 indexing, Qdrant upsert, graph update.

---

#### `archivist_pin` — Make a memory permanent

Sets `retention_class = permanent` and `importance_score = 1.0`. Use for things the agent must never lose: host IPs, credentials, org structure, service ownership.

```json
{
  "name": "archivist_pin",
  "arguments": {
    "agent_id": "my-agent",
    "memory_id": "a1b2c3d4-...",
    "reason": "Production DB host — must always be available"
  }
}
```

You can also pin by entity name instead of `memory_id`.

---

#### `archivist_delete` — Remove a memory

```json
{
  "name": "archivist_delete",
  "arguments": {
    "memory_id": "a1b2c3d4-...",
    "agent_id": "my-agent"
  }
}
```

---

#### `archivist_merge` — Resolve conflicting memories

```json
{
  "name": "archivist_merge",
  "arguments": {
    "agent_id": "my-agent",
    "memory_ids": ["id-1", "id-2"],
    "strategy": "semantic",
    "namespace": "engineering"
  }
}
```

Strategies: `latest` | `concat` | `semantic` | `manual`

---

#### `archivist_compress` — Compact a block of memories

Archives originals and returns a compact summary. Use when the context window is filling up.

```json
{
  "name": "archivist_compress",
  "arguments": {
    "agent_id": "my-agent",
    "namespace": "engineering",
    "memory_ids": ["id-1", "id-2", "id-3"],
    "format": "structured"
  }
}
```

`format: structured` produces a Goal/Progress/Decisions/Next Steps summary — ideal for ongoing projects.

---

### Namespace Management

#### `archivist_namespaces` — List accessible namespaces

Always call this when you are uncertain which namespaces you can read or write.

```json
{
  "name": "archivist_namespaces",
  "arguments": {
    "agent_id": "my-agent"
  }
}
```

---

### Context Management

#### `archivist_context_check` — Token budget check

Call before heavy reasoning to decide if compression is needed.

```json
{
  "name": "archivist_context_check",
  "arguments": {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize the deployment history."}
    ],
    "budget_tokens": 128000
  }
}
```

Returns: `{ "hint": "ok" | "compress" | "critical", "used_tokens": N, "budget_pct": N }`

When `hint` is `compress` or `critical`, call `archivist_compress` before continuing.

---

### Session Lifecycle

#### `archivist_session_end` — Summarize and persist session

```json
{
  "name": "archivist_session_end",
  "arguments": {
    "agent_id": "my-agent",
    "session_id": "session-2026-04-19",
    "store_as_memory": true
  }
}
```

---

### Trajectory & Feedback

#### `archivist_log_trajectory` — Record what you did and how it went

Logging trajectories enables outcome-aware retrieval — future searches will weight memories that led to successful outcomes.

```json
{
  "name": "archivist_log_trajectory",
  "arguments": {
    "agent_id": "my-agent",
    "task_description": "Migrate the auth service DB schema",
    "actions": [
      {"action": "search memories", "result": "found prior migration notes"},
      {"action": "run migration script", "result": "schema updated successfully"}
    ],
    "outcome": "success",
    "outcome_score": 0.95,
    "memory_ids_used": ["id-1", "id-2"],
    "session_id": "session-2026-04-19"
  }
}
```

---

#### `archivist_tips` — Retrieve tips from past trajectories

```json
{
  "name": "archivist_tips",
  "arguments": {
    "agent_id": "my-agent",
    "category": "recovery",
    "limit": 5
  }
}
```

---

### Admin & Observability

| Tool | Purpose |
|------|---------|
| `archivist_audit_trail` | Immutable audit log for memory operations |
| `archivist_retrieval_logs` | Pipeline execution traces for debugging |
| `archivist_health_dashboard` | Memory counts, stale %, conflict rate, retrieval stats |
| `archivist_cache_stats` | Hot cache hit rate and TTL |
| `archivist_cache_invalidate` | Manually flush the hot cache |
| `archivist_resolve_uri` | Resolve `archivist://` URIs |
| `archivist_get_reference_docs` | Full tool reference — call when unsure |

---

## Namespaces

Namespaces are the primary isolation boundary. Each agent has a default namespace derived from its `agent_id`. Always set `namespace` explicitly when writing shared knowledge.

**Conventions:**
- `engineering` — Technical decisions, architecture, infrastructure facts
- `product` — Requirements, roadmaps, stakeholder notes
- `people` — Team structure, preferences, contact info
- `{agent_id}` — Agent-private memories (auto-default)
- `shared` — Cross-agent shared knowledge accessible to all

**Rules:**
1. Call `archivist_namespaces(agent_id="your-id")` before writing to a namespace you have not used before
2. Do not write to another agent's private namespace without explicit permission
3. Use `caller_agent_id` when searching on behalf of a user — it propagates the correct RBAC identity

---

## Storing Memories: What to Include

When calling `archivist_store`, write text that is self-contained and searchable:

| Include | Example |
|---------|---------|
| Who, what, when, why | "On 2026-04-18, Alice approved the Postgres migration for Q2 2026 due to SQLite write-lock bottlenecks." |
| Named entities | Service names, person names, system names |
| Outcome or decision, not just process | "Decision: use blue-green deployment" not "we discussed deployment" |
| Importance signal | `importance_score: 0.9` for decisions; `0.5` for observations; `1.0` + pin for permanent facts |

Avoid storing raw conversation transcripts — summarize to the key fact or decision.

---

## Error Handling & Fallback

| Error | Response |
|-------|----------|
| RBAC denied | Call `archivist_namespaces` to confirm accessible scopes; adjust namespace |
| No results from search | Lower `min_score` toward `0`; try `tier: l2`; remove date filters |
| Conflict detected on store | Review the conflict; use `archivist_merge` if duplicates exist, or `force_skip_conflict_check: true` if store is deliberate |
| Context hint = critical | Immediately call `archivist_compress` with `format: structured`; do not continue reasoning |
| Unknown tool behavior | Call `archivist_get_reference_docs` with the relevant `section` name |

---

## Best Practices for the Agent

1. **Always call `archivist_wake_up` at session start.** It loads critical facts and namespace context in ~200 tokens — cheap and essential.
2. **Search before storing.** Check for existing memories with `archivist_search` before writing to avoid duplicates. Archivist also runs dedup internally, but explicit checks prevent unnecessary conflict resolution overhead.
3. **Use `importance_score` deliberately.** Default `0.5` is correct for observations. Use `0.9+` for decisions, `1.0` + `archivist_pin` for permanent facts that must survive TTL expiry.
4. **Prefer `tier: l0` or `l1` for pre-message injection.** Full `l2` text is rich but token-expensive. Use `l0` summaries to inform reasoning without consuming context budget.
5. **Log trajectories.** `archivist_log_trajectory` makes future retrievals smarter — memories associated with successful outcomes rank higher. Takes 30 seconds and pays dividends.
6. **Use `archivist_entity_brief` for entity-centric questions.** It's faster and more structured than a semantic search for "what do we know about X".
7. **Check context budget proactively.** Call `archivist_context_check` before long reasoning chains. A `compress` hint means compress now, not later.
8. **Set `refine: false` for speed.** LLM refinement improves synthesis quality but adds latency. Disable it for simple lookups or when working under time constraints.
9. **End sessions cleanly.** `archivist_session_end` persists a durable summary and enables outcome-aware retrieval for the next session.
10. **Do not store secrets as plain text.** Archivist logs are auditable; treat stored memories as semi-public within the namespace.

---

## Example Usage Flows

### Flow 1: Start of session

```
1. archivist_wake_up(agent_id="my-agent")
   → loads identity + critical facts + namespace overview

2. archivist_search(query="current sprint goals", agent_id="my-agent", tier="l0")
   → lightweight context for planning
```

---

### Flow 2: Store a key decision

```
User: "We've decided to use Redis for session caching, not in-memory maps."

1. archivist_search(query="session caching decision", agent_id="my-agent")
   → check for conflicting prior decision

2. archivist_store(
     text="2026-04-19: Decided to use Redis for session caching (not in-memory maps). 
           Rationale: horizontal scalability across pods.",
     agent_id="my-agent",
     namespace="engineering",
     entities=["Redis", "session caching"],
     importance_score=0.9,
     memory_type="experience"
   )
```

---

### Flow 3: Answer "what do we know about this system?"

```
User: "What do we know about prod-db-01?"

1. archivist_entity_brief(entity="prod-db-01", agent_id="my-agent")
   → structured card: facts, relationships, timeline

2. If more context needed:
   archivist_search(query="prod-db-01 incidents performance", agent_id="my-agent", tier="l1")
```

---

### Flow 4: Context window management

```
1. archivist_context_check(messages=[...current messages...], budget_tokens=128000)
   → returns { hint: "compress", budget_pct: 82 }

2. archivist_search(query="current task summary", agent_id="my-agent", tier="l0")
   → get IDs of recent memory blocks

3. archivist_compress(
     agent_id="my-agent",
     namespace="engineering",
     memory_ids=["id-1", "id-2", "id-3"],
     format="structured"
   )
   → Goal/Progress/Decisions/Next Steps summary replaces verbose history
```

---

### Flow 5: End of session

```
1. archivist_log_trajectory(
     agent_id="my-agent",
     task_description="...",
     actions=[...],
     outcome="success",
     memory_ids_used=["id-1"]
   )

2. archivist_session_end(
     agent_id="my-agent",
     session_id="session-2026-04-19",
     store_as_memory=true
   )
```
