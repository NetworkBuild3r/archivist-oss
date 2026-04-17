# Archivist MCP Skill Reference

## Overview

Archivist exposes 31 memory tools via the Model Context Protocol (MCP) over Streamable HTTP (preferred) or legacy HTTP SSE. Any MCP-compatible client can connect and use these tools.

## Connection

```text
Streamable HTTP endpoint (default): http://<host>:3100/mcp

Legacy SSE endpoints (opt-in, set MCP_SSE_ENABLED=true):
  http://<host>:3100/mcp/sse
  http://<host>:3100/mcp/messages/
```

---

## Search & Retrieval (7 tools)

### archivist_search

Semantic search with 10-stage RLM recursive retrieval pipeline. Supports fleet-wide, single-agent, or multi-agent queries with RBAC.

**Parameters:**
- `query` (string, **required**) -- Search query
- `agent_id` (string) -- Filter to one agent's memories
- `agent_ids` (array of string) -- Search these agents' memories (OR). Omit for fleet-wide.
- `caller_agent_id` (string) -- Invoking agent for RBAC when reading others' namespaces
- `namespace` (string) -- Memory namespace to search
- `team` (string) -- Filter by team
- `refine` (boolean, default: true) -- Enable LLM refinement
- `limit` (integer, default: 20) -- Max chunks to refine/synthesize
- `min_score` (number) -- Minimum vector similarity 0-1. Overrides `RETRIEVAL_THRESHOLD`. Use `0` to disable.
- `tier` (enum: `l0`, `l1`, `l2`, default: `l2`) -- Context tier: l0 abstract, l1 overview, l2 full
- `date_from` (string) -- ISO date lower bound, e.g. `2026-01-01`
- `date_to` (string) -- ISO date upper bound
- `max_tokens` (integer) -- Approximate token budget for returned context
- `memory_type` (enum: `experience`, `skill`, `general`) -- Filter by memory type

**Example:**
```json
{
  "name": "archivist_search",
  "arguments": {
    "query": "What decisions were made about the database migration?",
    "agent_id": "alice",
    "refine": true,
    "tier": "l1"
  }
}
```

### archivist_recall

Multi-hop knowledge graph lookup for entities and relationships/facts.

**Parameters:**
- `entity` (string, **required**) -- Entity name to look up
- `related_to` (string) -- Second entity to find connections
- `agent_id` (string) -- Calling agent for RBAC
- `caller_agent_id` (string) -- Identity for read access checks
- `namespace` (string) -- Scope

**Example:**
```json
{
  "name": "archivist_recall",
  "arguments": {
    "entity": "Kubernetes",
    "related_to": "ArgoCD"
  }
}
```

### archivist_timeline

Chronological timeline of memories about a topic.

**Parameters:**
- `query` (string, **required**) -- Topic to build timeline for
- `agent_id` (string) -- Filter to specific agent
- `caller_agent_id` (string) -- Invoker identity for RBAC
- `namespace` (string) -- Memory namespace
- `days` (integer, default: 14) -- Lookback window

### archivist_insights

Cross-agent knowledge discovery for a topic across accessible namespaces.

**Parameters:**
- `topic` (string, **required**) -- Topic to get insights on
- `agent_id` (string) -- Calling agent for RBAC
- `caller_agent_id` (string) -- Invoker identity for RBAC
- `namespace` (string) -- Namespace scope
- `limit` (integer, default: 10) -- Max insights

### archivist_deref

Dereference a memory by ID. Returns full L2 text and metadata. Use after L0/L1 search for drill-down.

**Parameters:**
- `memory_id` (string, **required**) -- Qdrant point ID
- `agent_id` (string) -- Calling agent for RBAC

### archivist_index

Compressed navigational index of knowledge in a namespace (~500 tokens). Lists entity categories and top topics for cross-domain bridging.

**Parameters:**
- `agent_id` (string, **required**) -- Calling agent for RBAC and default namespace resolution
- `namespace` (string) -- Namespace to index

### archivist_contradictions

Surface contradicting facts about an entity from different agents via the knowledge graph.

**Parameters:**
- `entity` (string, **required**) -- Entity name to check
- `agent_id` (string) -- Calling agent for RBAC

---

## Storage & Memory Management (3 tools)

### archivist_store

Store a memory with entity extraction, conflict checks, and LLM-adjudicated dedup.

**Parameters:**
- `text` (string, **required**) -- Memory content to store
- `agent_id` (string, **required**) -- Storing agent
- `namespace` (string) -- Target namespace
- `entities` (array of string) -- Entity names (auto-extracted if empty)
- `importance_score` (number, default: 0.5) -- 0.0-1.0 retention priority
- `memory_type` (enum: `experience`, `skill`, `general`, default: `general`) -- Memory type tag
- `force_skip_conflict_check` (boolean, default: false) -- Skip conflict check (use sparingly)

**Example:**
```json
{
  "name": "archivist_store",
  "arguments": {
    "text": "The migration to PostgreSQL was approved. Target date: Q2 2026.",
    "agent_id": "chief",
    "entities": ["PostgreSQL", "migration"],
    "importance_score": 0.9,
    "memory_type": "experience"
  }
}
```

### archivist_merge

Merge conflicting memory entries.

**Parameters:**
- `agent_id` (string, **required**) -- Calling agent
- `memory_ids` (array of string, **required**) -- Point IDs to merge
- `strategy` (enum: `latest`, `concat`, `semantic`, `manual`, **required**) -- Merge strategy
- `namespace` (string) -- Namespace for merged result

### archivist_compress

Archive memory blocks and return compact summaries. Supports flat (paragraph) and structured (Goal/Progress/Decisions/Next Steps) output.

**Parameters:**
- `agent_id` (string, **required**) -- Agent requesting compression
- `namespace` (string, **required**) -- Target namespace
- `memory_ids` (array of string, **required**) -- Point IDs to compress
- `summary` (string) -- Agent-provided summary (LLM generates if omitted)
- `format` (enum: `flat`, `structured`, default: `flat`) -- Output format
- `previous_summary` (string) -- Prior structured summary JSON for incremental compaction

---

## Trajectory & Feedback (5 tools)

### archivist_log_trajectory

Log an execution trajectory with auto-extracted tips via LLM post-mortem.

**Parameters:**
- `agent_id` (string, **required**) -- Agent that executed the trajectory
- `task_description` (string, **required**) -- What the agent was trying to accomplish
- `actions` (array of object, **required**) -- Ordered list of actions, e.g. `[{"action": "search", "result": "found X"}]`
- `outcome` (enum: `success`, `partial`, `failure`, `unknown`, **required**) -- Overall outcome
- `outcome_score` (number) -- Optional 0.0-1.0 quality score
- `memory_ids_used` (array of string) -- Memory IDs that informed decisions (enables outcome-aware retrieval)
- `session_id` (string) -- Session grouping key

### archivist_annotate

Add quality annotations to a memory point.

**Parameters:**
- `memory_id` (string, **required**) -- Point ID to annotate
- `agent_id` (string, **required**) -- Annotating agent
- `content` (string, **required**) -- Annotation text
- `annotation_type` (enum: `note`, `correction`, `stale`, `verified`, `quality`, default: `note`)
- `quality_score` (number) -- Optional 0.0-1.0 quality assessment

### archivist_rate

Rate a memory as helpful (+1) or unhelpful (-1).

**Parameters:**
- `memory_id` (string, **required**) -- Point ID to rate
- `agent_id` (string, **required**) -- Rating agent
- `rating` (integer, **required**) -- `+1` (helpful) or `-1` (unhelpful)
- `context` (string) -- Optional context for the rating

### archivist_tips

Retrieve tips from past trajectories.

**Parameters:**
- `agent_id` (string, **required**) -- Agent whose tips to retrieve
- `category` (enum: `strategy`, `recovery`, `optimization`) -- Filter by category
- `limit` (integer, default: 10) -- Max tips to return

### archivist_session_end

Summarize a session and optionally store it as durable memory.

**Parameters:**
- `agent_id` (string, **required**) -- Agent whose session to summarize
- `session_id` (string, **required**) -- Session identifier
- `store_as_memory` (boolean, default: true) -- Also store summary as durable memory

---

## Skill Registry (6 tools)

### archivist_register_skill

Register or update a skill (MCP tool) with provider, endpoint, and version.

**Parameters:**
- `name` (string, **required**) -- Skill/tool name
- `agent_id` (string, **required**) -- Registering agent
- `provider` (string) -- Provider name (e.g. `openai`, `internal`)
- `mcp_endpoint` (string) -- MCP server endpoint URL
- `version` (string, default: `0.0.0`) -- Version string
- `description` (string) -- What the skill does
- `changelog` (string) -- What changed in this version
- `breaking_changes` (string) -- Known breaking changes

### archivist_skill_event

Log a skill invocation outcome for health scoring.

**Parameters:**
- `skill_name` (string, **required**) -- Skill name
- `agent_id` (string, **required**) -- Agent that used the skill
- `outcome` (enum: `success`, `partial`, `failure`, **required**)
- `provider` (string) -- Provider to disambiguate
- `skill_version` (string) -- Version used
- `duration_ms` (integer) -- Execution time in milliseconds
- `error_message` (string) -- Error details if failed
- `trajectory_id` (string) -- Link to a trajectory

### archivist_skill_lesson

Record failure modes, workarounds, or best practices for a skill.

**Parameters:**
- `skill_name` (string, **required**) -- Skill name
- `title` (string, **required**) -- Short title
- `content` (string, **required**) -- Full lesson content
- `agent_id` (string, **required**) -- Contributing agent
- `provider` (string) -- Provider to disambiguate
- `lesson_type` (enum: `failure_mode`, `workaround`, `best_practice`, `breaking_change`, `general`, default: `general`)
- `skill_version` (string) -- Version this lesson applies to

### archivist_skill_health

Get health grade, success rate, recent failures, and substitutes for a skill.

**Parameters:**
- `skill_name` (string, **required**) -- Skill to check
- `provider` (string) -- Provider to disambiguate
- `window_days` (integer, default: 30) -- History window
- `include_lessons` (boolean, default: true) -- Include recent lessons

### archivist_skill_relate

Create or update a relation between two skills.

**Parameters:**
- `skill_a` (string, **required**) -- First skill name
- `skill_b` (string, **required**) -- Second skill name
- `relation_type` (enum: `similar_to`, `depend_on`, `compose_with`, `replaced_by`, **required**)
- `agent_id` (string, **required**) -- Agent creating the relation
- `confidence` (number, default: 1.0) -- Confidence 0-1
- `evidence` (string) -- Why this relation exists
- `provider_a` (string) -- Provider for skill_a
- `provider_b` (string) -- Provider for skill_b

### archivist_skill_dependencies

Return the dependency/relation graph for a skill.

**Parameters:**
- `skill_name` (string, **required**) -- Skill to get relations for
- `provider` (string) -- Provider to disambiguate
- `depth` (integer, default: 1) -- Graph traversal depth

---

## Admin & Context Management (7 tools)

### archivist_context_check

Pre-reasoning context check. Returns token count, budget usage %, and hint (ok / compress / critical).

**Parameters:**
- `messages` (array of object) -- Chat messages `[{role, content}]` to count tokens for
- `memory_texts` (array of string) -- Raw texts to count tokens for (alternative to messages)
- `budget_tokens` (integer) -- Token budget (defaults to `DEFAULT_CONTEXT_BUDGET` env)
- `reserve_from_tail` (integer, default: 2000) -- Tokens to reserve for recent messages

**Example:**
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

### archivist_namespaces

List memory namespaces accessible to the calling agent.

**Parameters:**
- `agent_id` (string, **required**) -- The calling agent's ID

### archivist_audit_trail

View immutable audit log for memory operations.

**Parameters:**
- `agent_id` (string, **required**) -- Calling agent
- `memory_id` (string) -- Specific memory ID to audit
- `target_agent` (string) -- Agent whose activity to view
- `limit` (integer, default: 50) -- Max entries

### archivist_resolve_uri

Resolve an `archivist://` URI to its underlying resource. Supports memory, entity, namespace, and skill URIs.

**Parameters:**
- `uri` (string, **required**) -- An `archivist://` URI
- `agent_id` (string) -- Calling agent for RBAC

### archivist_retrieval_logs

Export retrieval pipeline execution traces for debugging and analytics.

**Parameters:**
- `agent_id` (string) -- Filter by agent
- `limit` (integer, default: 20) -- Max entries
- `since` (string) -- ISO datetime lower bound
- `stats_only` (boolean, default: false) -- Return aggregate stats instead
- `window_days` (integer, default: 7) -- Stats aggregation window

### archivist_health_dashboard

Comprehensive health dashboard: memory counts, stale %, conflict rate, retrieval stats, skill health, cache status.

**Parameters:**
- `window_days` (integer, default: 7) -- Analysis window

### archivist_batch_heuristic

Recommend batch size (1-10) from health signals. Considers conflict rate, stale %, cache hit rate, degraded skills.

**Parameters:**
- `window_days` (integer, default: 7) -- Analysis window

---

## Cache Management (2 tools)

### archivist_cache_stats

Return hot cache statistics: entries per agent, TTL, hit rate.

**Parameters:** *(none)*

### archivist_cache_invalidate

Manually invalidate the hot cache.

**Parameters:**
- `namespace` (string) -- Invalidate entries for this namespace
- `agent_id` (string) -- Invalidate entries for this agent
- `all` (boolean, default: false) -- Invalidate entire cache

---

## Reference Docs (1 tool)

### archivist_get_reference_docs

Return the full Archivist agent skill reference (this document) or a single named section.  Call this on first connection or whenever you are unsure how to use a tool.

**Parameters:**
- `section` (string, optional) -- Heading keyword to filter (e.g. `"search"`, `"storage"`, `"trajectory"`, `"skills"`, `"admin"`, `"tips"`). Omit to return the full reference.

---

## Tips

1. **Start with `archivist_search`** for most queries -- it handles the full retrieval pipeline
2. **Use `archivist_recall`** when you know the entity name and want structured data
3. **Set `refine: false`** on `archivist_search` for faster results (skips LLM refinement)
4. **Use `tier: l0`** with `max_tokens` for lightweight pre-message injection
5. **Use `archivist_store`** with high `importance_score` (>0.9) to prevent TTL expiry
6. **Check `archivist_namespaces`** to see what you can access
7. **Use `archivist_context_check`** before reasoning to decide if compaction is needed
8. **Use `archivist_compress` with `format: structured`** for Goal/Progress/Decisions/Next Steps summaries
9. **Log trajectories** with `archivist_log_trajectory` so future searches benefit from outcome-aware retrieval
10. **Check skill health** with `archivist_skill_health` before invoking unreliable external tools
