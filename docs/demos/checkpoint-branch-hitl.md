# Checkpoint branch + HITL recipe

<!-- INIT-012/SPEC-005 -->

Requires **`ARCHIVIST_TOOL_PROFILE=ops`** (or `full`). Tools are **not** on **core**.
See [ADR-012](../adr/ADR-012-checkpoint-time-travel.md) and
[REFERENCE](../REFERENCE.md#agent-checkpoints-8).

Do **not** put API keys or other secrets in checkpoint payloads.

## Save → branch → interrupt → approve → resume

1. Caller needs namespace **write** (mutations) / **read** (resume/replay) and must be the
   checkpoint **owner** (`agent_id` bind).
2. Save a root checkpoint:

```json
{
  "tool": "archivist_checkpoint_save",
  "arguments": {
    "agent_id": "worker",
    "session_id": "sess-1",
    "namespace": "shared",
    "payload": { "summary": "mid-task", "step": 0 }
  }
}
```

3. Branch from the parent:

```json
{
  "tool": "archivist_checkpoint_branch",
  "arguments": {
    "agent_id": "worker",
    "parent_checkpoint_id": "<root id>",
    "namespace": "shared",
    "payload": { "summary": "forked plan", "step": 1 }
  }
}
```

4. Interrupt (HITL wait) — resume will fail until approve:

```json
{
  "tool": "archivist_checkpoint_interrupt",
  "arguments": {
    "agent_id": "worker",
    "checkpoint_id": "<branch id>",
    "namespace": "shared",
    "reason": "need human review"
  }
}
```

5. Approve, then resume into SessionStore:

```json
{
  "tool": "archivist_checkpoint_approve",
  "arguments": {
    "agent_id": "worker",
    "checkpoint_id": "<branch id>",
    "namespace": "shared"
  }
}
```

```json
{
  "tool": "archivist_checkpoint_resume",
  "arguments": {
    "agent_id": "worker",
    "session_id": "sess-1",
    "namespace": "shared",
    "checkpoint_id": "<branch id>"
  }
}
```

6. Optional: `archivist_checkpoint_replay` on the branch id for a read-only parent chain.
