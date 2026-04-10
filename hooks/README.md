# Archivist Auto-Save Hooks

Hooks that automatically persist session context to Archivist, preventing
information loss across context window boundaries. Supports periodic
auto-save and pre-compaction emergency save.

## Hooks

| Hook | Purpose | When to fire |
|------|---------|-------------|
| `archivist_save_hook.sh` | Save session trajectories as durable memory | End of turn, periodic interval |
| `archivist_precompact_hook.sh` | Emergency save + structured compress before context shrinks | Before context compaction |

## Environment Variables

All hooks share these variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARCHIVIST_URL` | `http://localhost:3100` | Archivist MCP endpoint |
| `AGENT_ID` | `claude` | Agent identifier for namespacing |
| `SESSION_ID` | `auto-<timestamp>` | Session identifier (auto-generated if unset) |
| `ARCHIVIST_API_KEY` | *(empty)* | Optional Bearer token for auth |
| `NAMESPACE` | *(empty)* | Memory namespace (precompact hook only) |

## Setup: Claude Code

Add hooks to `.claude/settings.json` in your project root:

```json
{
  "hooks": {
    "Stop": [
      {
        "command": "bash hooks/archivist_save_hook.sh",
        "description": "Save session context to Archivist"
      }
    ],
    "Compact": [
      {
        "command": "bash hooks/archivist_precompact_hook.sh",
        "description": "Emergency save before context compaction"
      }
    ]
  }
}
```

The `Stop` hook fires at the end of each assistant turn, while `Compact`
fires before Claude Code shrinks the context window.

## Setup: Cursor

Cursor doesn't have built-in hook events, but you can achieve similar
behavior through these approaches:

### Option A: Cursor Rules (recommended)

Add to `.cursor/rules/archivist-save.mdc`:

```markdown
---
description: Auto-save session context to Archivist
globs: ["**/*"]
alwaysApply: true
---

At the end of each major task or when context is getting large, call
the archivist_session_end MCP tool with your agent_id to persist
the session. If context compaction is needed, call archivist_compress
on recent memories first.
```

### Option B: Task runner

Add to your `Makefile` or `package.json`:

```makefile
archivist-save:
	@bash hooks/archivist_save_hook.sh

archivist-compact:
	@bash hooks/archivist_precompact_hook.sh
```

Then invoke manually or via a periodic task.

### Option C: Cron (periodic auto-save)

```bash
# Save every 15 minutes while working
*/15 * * * * cd /path/to/project && AGENT_ID=cursor bash hooks/archivist_save_hook.sh
```

## Setup: Other MCP Clients

Any MCP client can call the underlying tools directly:

- **`archivist_session_end`** — Summarize and store session trajectories
  - Required: `agent_id`, `session_id`
  - Optional: `store_as_memory` (default: true)

- **`archivist_compress`** — Archive and compress memory blocks
  - Required: `agent_id`, `namespace`, `memory_ids`
  - Optional: `format` ("flat" or "structured"), `summary`, `previous_summary`

## Troubleshooting

**Hook exits silently (no error, no save):**
Archivist isn't running. Start it with `docker compose up -d` or
`python src/main.py`.

**"401 Unauthorized":**
Set `ARCHIVIST_API_KEY` to match your server's `ARCHIVIST_API_KEY` env var.

**No trajectories to save:**
`archivist_session_end` requires prior `archivist_log_trajectory` calls in
the same session. If using the save hook standalone, ensure your agent is
logging trajectories during the session.

**Pre-compact finds no memories:**
The hook queries recent memories via `archivist_recall`. If the namespace
is empty or memories were already compressed, it exits cleanly.
