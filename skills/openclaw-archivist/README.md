# Archivist OpenClaw Skill

Give any OpenClaw agent persistent, cross-session memory backed by Archivist's semantic search, BM25 full-text search, and knowledge graph.

## Prerequisites

A running Archivist instance (v2.0+). See the [Archivist docs](https://github.com/NetworkBuild3r/archivist-oss) for deployment instructions.

## Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ARCHIVIST_MCP_URL` | Archivist MCP endpoint | `http://archivist:3100/mcp` |
| `ARCHIVIST_API_KEY` | API key set on the Archivist server | `sk-archivist-...` |

Set these in your shell, `.env` file, or OpenClaw secrets config before starting.

## Installation

### 1. Copy the skill

```bash
mkdir -p ~/.openclaw/skills/archivist
cp SKILL.md ~/.openclaw/skills/archivist/SKILL.md
```

### 2. Register the MCP server in `openclaw.json`

Add an entry under `mcpServers` so OpenClaw knows how to reach Archivist:

```json
{
  "mcpServers": {
    "archivist": {
      "url": "${ARCHIVIST_MCP_URL}",
      "headers": {
        "Authorization": "Bearer ${ARCHIVIST_API_KEY}"
      }
    }
  }
}
```

### 3. Enable the skill for your agent

Under the relevant agent in `openclaw.json`, add `"archivist"` to its `skills` list:

```json
{
  "agents": {
    "my-agent": {
      "skills": ["archivist"]
    }
  }
}
```

### 4. Verify the connection

Start OpenClaw and check that the Archivist tools are visible:

```bash
openclaw tools list --agent my-agent | grep archivist
```

You should see tools like `archivist_search`, `archivist_store`, `archivist_wake_up`, etc.

## Quick smoke test

```bash
openclaw run --agent my-agent \
  "Call archivist_wake_up with agent_id='my-agent' and tell me what namespaces you can access."
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `401 Unauthorized` | Verify `ARCHIVIST_API_KEY` matches the key configured on the server (`ARCHIVIST_API_KEY` env on the Archivist container) |
| `Connection refused` | Check `ARCHIVIST_MCP_URL` — confirm the host and port are reachable from the OpenClaw process |
| Tools not listed | Confirm the `mcpServers` entry uses the exact key `"archivist"` and that the skill name in `skills` matches |
| RBAC errors on search | Call `archivist_namespaces(agent_id="your-id")` to see what the agent can access |

## Further reading

- [Archivist architecture](https://github.com/NetworkBuild3r/archivist-oss/blob/main/docs/ARCHITECTURE.md)
- [Full MCP tool reference](https://github.com/NetworkBuild3r/archivist-oss/blob/main/docs/CURSOR_SKILL.md)
- [Docker deployment guide](https://github.com/NetworkBuild3r/archivist-oss/blob/main/docs/DOCKER.md)
