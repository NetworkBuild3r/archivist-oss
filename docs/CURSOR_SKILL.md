# Archivist MCP Skill Reference

## Overview

Archivist exposes memory tools via the Model Context Protocol (MCP) over HTTP SSE. Any MCP-compatible client can connect and use these tools.

## Connection

```
SSE endpoint: http://<host>:3100/mcp/sse
Message endpoint: http://<host>:3100/mcp/messages/
```

## Available Tools

### archivist_search

Semantic search with recursive LLM retrieval pipeline.

**Parameters:**
- `query` (string, required) — Search query
- `agent_id` (string) — Filter to a single agent’s memories
- `agent_ids` (array of strings) — Search only these agents’ memories (OR). Omit for fleet-wide search.
- `caller_agent_id` (string) — Who is calling the tool; used for RBAC when reading other agents’ namespaces. Defaults to `agent_id` if set.
- `namespace` (string) — Filter to specific namespace
- `team` (string) — Filter by team
- `refine` (boolean, default: true) — Enable LLM refinement
- `limit` (integer, default: 20) — Max chunks to refine after retrieval
- `min_score` (number, optional) — Minimum vector similarity (0–1). Overrides the `RETRIEVAL_THRESHOLD` environment default for this call. Use `0` to disable score filtering.

**Returns:** Synthesized answer with source citations. Responses may include `multi_agent`, `agents_scoped`, and `below_threshold` as documented in the main README.

**Example:**
```json
{
  "name": "archivist_search",
  "arguments": {
    "query": "What decisions were made about the database migration?",
    "agent_id": "alice",
    "refine": true
  }
}
```

### archivist_recall

Knowledge graph lookup for entities and their relationships.

**Parameters:**
- `entity` (string, required) — Entity name to look up
- `related_to` (string) — Second entity for connection finding
- `agent_id` (string) — Calling agent for RBAC
- `namespace` (string) — Scope

**Returns:** Entity details, facts, and relationships.

### archivist_store

Explicitly store a memory with optional entity tagging.

**Parameters:**
- `text` (string, required) — Memory content
- `agent_id` (string, required) — Storing agent
- `namespace` (string) — Target namespace
- `entities` (array of strings) — Entity names
- `importance_score` (number, 0.0-1.0) — Retention priority

**Returns:** Memory ID and storage confirmation.

### archivist_timeline

Chronological view of memories about a topic.

**Parameters:**
- `query` (string, required) — Topic
- `agent_id` (string) — Filter to agent
- `namespace` (string) — Filter namespace
- `days` (integer, default: 14) — Lookback window

### archivist_insights

Cross-agent knowledge discovery.

**Parameters:**
- `topic` (string, required) — Topic to explore
- `agent_id` (string) — Calling agent
- `namespace` (string) — Scope
- `limit` (integer, default: 10) — Max insights

### archivist_namespaces

List namespaces accessible to an agent.

**Parameters:**
- `agent_id` (string, required) — Agent ID

### archivist_audit_trail

View audit log entries.

**Parameters:**
- `agent_id` (string, required) — Calling agent
- `memory_id` (string) — Specific memory to audit
- `target_agent` (string) — Agent whose activity to view
- `limit` (integer, default: 50) — Max entries

### archivist_merge

Merge conflicting memory entries.

**Parameters:**
- `agent_id` (string, required) — Calling agent
- `memory_ids` (array of strings, required) — IDs to merge
- `strategy` (string, required) — One of: `latest`, `concat`, `semantic`, `manual`
- `namespace` (string) — Namespace for result

## Tips

1. **Start with `archivist_search`** for most queries — it handles the full retrieval pipeline
2. **Use `archivist_recall`** when you know the entity name and want structured data
3. **Set `refine: false`** on `archivist_search` for faster results (skips LLM refinement)
4. **Use `archivist_store`** with high `importance_score` (>0.9) to prevent TTL expiry
5. **Check `archivist_namespaces`** to see what you can access
