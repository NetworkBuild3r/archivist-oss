# Memory as a Product — round-trip recipe

<!-- INIT-011/SPEC-005 -->

Requires **`ARCHIVIST_TOOL_PROFILE=ops`** (or `full`). Tools are **not** on **core**.
See [ADR-011](../adr/ADR-011-memory-as-product-mcp.md) and [REFERENCE](../REFERENCE.md#memory-as-a-product-6).

Do **not** put API keys or other secrets in archives/manifests.

## Snapshot → export → import

1. Ensure the caller has **read** on the source namespace and **write** on the target.
2. Snapshot:

```json
{
  "tool": "archivist_map_snapshot",
  "arguments": {
    "namespace": "shared",
    "agent_id": "chief",
    "caller_agent_id": "chief",
    "label": "demo-v1"
  }
}
```

3. Export (optional if you already have `archive_id` from snapshot):

```json
{
  "tool": "archivist_map_export",
  "arguments": {
    "namespace": "shared",
    "caller_agent_id": "chief",
    "version_id": "<version_id from snapshot>"
  }
}
```

4. Import into an **empty** target agent scope (fail-closed if nonempty):

```json
{
  "tool": "archivist_map_import",
  "arguments": {
    "archive_id": "<archive_id>",
    "target_namespace": "pipeline",
    "target_agent_id": "gitbob-demo",
    "caller_agent_id": "gitbob",
    "label": "demo-import"
  }
}
```

5. Verify with `archivist_map_list` on the target namespace / agent.

## Fork instead of import

Use `archivist_map_fork` with `source_version_id` when cloning from a known version
into a fresh target scope (source **read** + target **write**).
