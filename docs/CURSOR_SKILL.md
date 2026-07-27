# Archivist MCP skill reference

Authoritative parameter lists and examples for every MCP tool. A condensed table lives in [`REFERENCE.md`](REFERENCE.md); operational testing is covered in [`QA.md`](QA.md) and [`QA_CHECKLIST.md`](../QA_CHECKLIST.md).

## Overview

Archivist exposes 41 memory tools via the Model Context Protocol (MCP) over Streamable HTTP (preferred) or legacy HTTP SSE. Any MCP-compatible client can connect and use these tools.

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

## Admin & Context Management (8 tools)

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

Resolve an `archivist://` URI to its underlying resource. Supports memory, entity, and namespace URIs.

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

Comprehensive health dashboard: memory counts, stale %, conflict rate, retrieval stats, cache status.

**Parameters:**
- `window_days` (integer, default: 7) -- Analysis window

### archivist_batch_heuristic

Recommend batch size (1-10) from health signals. Considers conflict rate, stale %, cache hit rate.

**Parameters:**
- `window_days` (integer, default: 7) -- Analysis window

### archivist_savings_dashboard

Token savings statistics from the Answer Finder engine: average/min/max savings %, total tokens saved vs naive full-history retrieval, per-policy breakdown, estimated USD (`estimated_usd_*`, `null` when `TOKEN_USD_PER_1K` unset), and hotness heatmap.

**Parameters:**
- `window_days` (integer, default: 7) -- Analysis window for retrieval log aggregation
- `heatmap_top_n` (integer, default: 50) -- Number of top memories to include in hotness heatmap

### archivist_memory_lineage

Lineage edges for a memory or entity from provenance, version history, audit actors, and retrieval-log mentions. Requires namespace read access. Returns observability edges only — no secrets or full memory text.

**Parameters:**
- `agent_id` (string, **required**) -- Calling agent for RBAC
- `memory_id` (string) -- Memory id (mutually exclusive with `entity_id`)
- `entity_id` (string) -- Entity numeric id or name
- `namespace` (string) -- Namespace for RBAC (required for entity lineage under strict RBAC)
- `limit` (integer, default: 50) -- Max edges per source

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

## Context Assembly & Handoff (3 tools)

Handoff tools are unchanged. For durable agent-state resume/time-travel, see **Agent Checkpoints** below.

### archivist_get_context

High-level token-budgeted context assembly for agents. Returns tiered memories, graph facts, and procedural tips in a single call — replaces multi-step search + recall patterns.

**Parameters:**
- `agent_id` (string, **required**) -- Calling agent
- `task_description` (string, **required**) -- What the agent is about to do
- `namespace` (string) -- Memory namespace to search
- `max_tokens` (integer, default: 8000) -- Token budget for returned context
- `include_graph` (boolean, default: true) -- Include knowledge graph facts
- `include_tips` (boolean, default: true) -- Include procedural tips from past trajectories
- `extra_memory_ids` (array of string) -- Pin-include specific memory IDs regardless of score
- `pack_policy` (enum: `adaptive`, `l0_first`, `l2_first`) -- Override packing policy for this call

**Example:**
```json
{
  "name": "archivist_get_context",
  "arguments": {
    "agent_id": "planner",
    "task_description": "Plan the next sprint for the payments team",
    "namespace": "payments",
    "max_tokens": 6000
  }
}
```

### archivist_handoff

Package the current session (summary, active goals, recovery tips, top memories, knowledge snapshot, ephemeral notes) into a typed `HandoffPacket` for transfer to another agent.

**Parameters:**
- `agent_id` (string, **required**) -- Sending agent
- `session_id` (string, **required**) -- Current session identifier
- `namespace` (string) -- Namespace scope for the knowledge snapshot
- `target_agent_id` (string) -- Intended recipient agent (informational; used to filter context)

### archivist_receive_handoff

Inject a `HandoffPacket` into the receiving agent's ephemeral `SessionStore`. The receiving agent can then call `archivist_get_context` to include the handoff context in retrieval.

**Parameters:**
- `packet` (object, **required**) -- A `HandoffPacket` returned by `archivist_handoff`
- `receiving_agent_id` (string, **required**) -- Agent receiving the handoff
- `session_id` (string) -- Session to inject into (defaults to a new session)

---

## Agent Checkpoints (8 tools)

<!-- INIT-012/SPEC-005 -->

Diff #7 agent-state checkpoints (resume / time-travel / branch / thin HITL) —
[ADR-012](adr/ADR-012-checkpoint-time-travel.md). On **ops** and **full** (not **core**).
Namespace-scoped; distinct from L0–L2 memory tiers and from handoff packets.
Do **not** store API keys or other secrets in checkpoint payloads.

### archivist_checkpoint_save

Persist a checkpoint payload for an agent session.

**Parameters:**
- `agent_id` (string, **required**)
- `session_id` (string, **required**)
- `namespace` (string, **required**) -- RBAC write required
- `payload` (object) -- State blob (max 256 KiB JSON)
- `metadata` (object) -- Optional labels (max 64 KiB JSON)
- `parent_checkpoint_id` (string) -- Optional parent in the same namespace

### archivist_checkpoint_list

List checkpoints for `agent_id` + `session_id` in `namespace` (oldest first).

### archivist_checkpoint_get

Fetch one checkpoint by `checkpoint_id` **and** `namespace`. Id alone is never sufficient.

### archivist_checkpoint_resume

Load a checkpoint and inject a resume packet into the caller's `SessionStore` for `session_id` only (does not mutate other agents). Returns `resume_packet` with `injected_keys`, `extra_memory_ids`, and `summary` for `archivist_get_context`. Owner-agent bind required. Fails with `hitl_interrupted` until `archivist_checkpoint_approve` if interrupted.

### archivist_checkpoint_replay

Read-only parent-chain walk from a leaf checkpoint (root → leaf). No SessionStore mutation.

### archivist_checkpoint_branch

Create a child checkpoint from a **required** `parent_checkpoint_id` in the same namespace. Owner must match the parent. Optional `payload` overrides the parent copy.

### archivist_checkpoint_interrupt

Mark a checkpoint HITL-interrupted (`hitl_status=interrupted` in metadata). Optional `reason`.

### archivist_checkpoint_approve

Clear HITL interrupt (`hitl_status=approved`) so resume may proceed. Idempotent if already approved.

---

## Coordination Beyond Handoff (5 tools)

<!-- INIT-009/SPEC-004 -->

Selective share + consensus v1 (explicit accept/reject + audit) — Diff #5 /
[ADR-009](adr/ADR-009-native-multi-agent-coordination.md). On **ops** and **full**
(not **core**). Extends — does not replace — handoff (GR-HANDOFF-001).

**Tip / lesson share:** Tips are the only procedural lesson API ([ADR-007](adr/ADR-007-procedural-memory-wedge.md) /
[ADR-008](adr/ADR-008-retire-skills-tip-lessons.md)). Prefer `archivist_handoff` for tip
strings in `HandoffPacket`. Optionally pass `tip_ids` on propose for selective tip-id
grants (metadata + SessionStore `share_tip_ids` on accept).

### archivist_share_propose

Propose a selective share of `memory_ids`, `tip_ids`, and/or `scope` to `recipient_agent_id`
in a `namespace`. Proposer needs namespace **read**. Creates a pending grant.

**Parameters:**
- `agent_id` (string, **required**) -- Proposer
- `recipient_agent_id` (string, **required**)
- `namespace` (string, **required**)
- `memory_ids` (array of string) -- Selective memory IDs
- `tip_ids` (array of string) -- Optional tip/lesson IDs (stored on grant metadata)
- `scope` (string) -- Optional Memory-as-Product scope label
- `reason` (string) -- Audited rationale

At least one of `memory_ids`, `tip_ids`, or `scope` is required.

### archivist_share_accept / archivist_share_reject

Recipient-only decisions. Audited; idempotent when already in the target status. Accept may
inject `share_memory_ids` / `share_tip_ids` / `share_scope` into the recipient `SessionStore`
for `session_id`. Optional `materialize_namespace` on accept requires **write** RBAC.

### archivist_share_attach_conflict

Attach a conflict/consensus outcome to a grant. `action` must be `supersede`, `merge`, or
`keep_both` (same vocabulary as `contradiction_resolve`). Optional `apply=true` builds a
`ResolutionProposal` and calls `apply_resolution` (`dry_run` defaults **true** — set
`dry_run=false` to mutate facts; requires `entity_id` + fact pair ids).

### archivist_share_get

Fetch one grant by `grant_id` + `namespace`. Visible to proposer or recipient with namespace read.

---

## Memory as a Product (6 tools)

<!-- INIT-011/SPEC-005 · ADR-011 -->

Versioned scope snapshot / fork / export / **import** — Diff #4 /
[ADR-011](adr/ADR-011-memory-as-product-mcp.md). On **ops** and **full** (not **core**).
Archives use opaque `archive_id` under `BACKUP_DIR` (no client absolute paths).
Do not put secrets in manifests. Short recipe: [`demos/map-roundtrip.md`](demos/map-roundtrip.md).

### archivist_map_list / archivist_map_get

List or fetch scope versions. Requires `caller_agent_id` + namespace **read**.

### archivist_map_snapshot

Create a versioned archive of a memory scope (`namespace`, optional `agent_id` / `label`).

### archivist_map_fork

Fork a `source_version_id` into `target_namespace` (source **read** + target **write**).

### archivist_map_export

Export a scope or existing `version_id` to `BACKUP_DIR` (returns `archive_id`, path, manifest metadata).

### archivist_map_import

Restore `archive_id` into `target_namespace` (source namespace **read** from
manifest + target **write**). Fail-closed if the target agent scope is already
nonempty — fork into a fresh scope or clear first.

---

## Intelligent Self-Curation (Diff #6)

<!-- INIT-010/SPEC-005 · ADR-010 -->

Runs in the **background curator cycle** — not new MCP tools on **core**. Defaults
stay safe: masters **off**, dry-run **on** when you enable a path.

| Path | Env (defaults) | Effect when applied |
|------|----------------|---------------------|
| Reconsolidation | `RECONSOLIDATION_ENABLED=false`, `RECONSOLIDATION_DRY_RUN=true` | Summarize L2 groups → L1 chunk (same agent/namespace) |
| Relevance forget | `RELEVANCE_FORGET_ENABLED=false`, `RELEVANCE_FORGET_DRY_RUN=true` | Suppress cold/low-importance chunks from default recall |
| Contradiction resolve | `CONTRADICTION_RESOLVE_ENABLED=false`, `CONTRADICTION_RESOLVE_DRY_RUN=true` | Propose/apply supersede/merge/keep_both |
| Reflection (optional) | `REFLECTION_ENABLED=false`, `REFLECTION_DRY_RUN=true` | Tip artifacts from trajectories |

**Operator ladder:** enable with dry-run → review audit → flip dry-run false.
Mutating share `attach_conflict` still needs namespace **write** + resolve enabled.
Full flag matrix: [REFERENCE.md](REFERENCE.md#intelligent-self-curation-diff-6) /
[ADR-010](adr/ADR-010-intelligent-self-curation.md). Prefer `archivist_pin` for
facts that must never decay/forget.

---

## Reference Docs (1 tool)

### archivist_get_reference_docs

Return the full Archivist agent skill reference (this document) or a single named section.  Call this on first connection or whenever you are unsure how to use a tool.

**Parameters:**
- `section` (string, optional) -- Heading keyword to filter (e.g. `"search"`, `"storage"`, `"trajectory"`, `"admin"`, `"tips"`). Omit to return the full reference.

---

## Tips

1. **Start with `archivist_get_context`** for pre-prompt injection — tier-aware, token-budgeted, single call
2. **Use `archivist_search`** for explicit ad-hoc queries within a task
3. **Use `archivist_recall`** when you know the entity name and want structured facts
4. **Set `refine: false`** on `archivist_search` for faster results (skips LLM refinement)
5. **Use `tier: l0`** with `max_tokens` for lightweight pre-message injection
6. **Use `archivist_store`** with high `importance_score` (>0.9) to prevent TTL expiry
7. **Check `archivist_namespaces`** to see what you can access
8. **Use `archivist_context_check`** before reasoning to decide if compaction is needed
9. **Use `archivist_compress` with `format: structured`** for Goal/Progress/Decisions/Next Steps summaries
10. **Log trajectories** with `archivist_log_trajectory` so future searches benefit from outcome-aware retrieval
11. **Hand off sessions** with `archivist_handoff` + `archivist_receive_handoff` to transfer context **and tips** between agents (primary tip-transfer channel)
12. **Selectively share memories / tip_ids** with `archivist_share_propose` / `accept` / `reject` on **ops**/**full** (Diff #5 / ADR-009; not on **core**; does not replace handoff)
13. **Attach conflict outcomes** with `archivist_share_attach_conflict` (`supersede` / `merge` / `keep_both`; optional `apply` → resolver; mutating apply needs write + `CONTRADICTION_RESOLVE_ENABLED`)
14. **Version / fork / export / import memory scopes** with `archivist_map_*` on **ops**/**full** (Diff #4 / ADR-011; not on **core**; opaque `archive_id` under `BACKUP_DIR`)
15. **Self-curation** is flag-driven in the curator (Diff #6 / ADR-010) — do not invent core MCP tools; stage `RECONSOLIDATION_*` / `RELEVANCE_FORGET_*` / `CONTRADICTION_RESOLVE_*`
16. **Resume / branch / HITL agent state** with `archivist_checkpoint_*` on **ops**/**full** (Diff #7 / ADR-012; not on **core**; interrupt→approve before resume when gated; no secrets in payloads)
17. **Monitor token savings** with `archivist_savings_dashboard` to confirm the Answer Finder is reducing noise
