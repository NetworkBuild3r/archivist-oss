# Demo: Observability billboard (Diff #8)

<!-- INIT-013/SPEC-005 -->

Thin control-plane UI over existing lineage / audit / savings / health JSON
([ADR-013](../adr/ADR-013-observability-billboard.md)).

## Prerequisites

- Archivist running (default MCP port **3100**)
- Prefer `ARCHIVIST_API_KEY` set on any shared/exposed network
- Optional: stored memories so lineage/audit have edges

## Open the billboard

1. Browse to `http://localhost:3100/admin/ui/`
2. If the server requires a key, paste it into **API key** and Apply (optional
   sessionStorage remember — clears when the tab session ends)
3. **Health & savings** loads `GET /admin/dashboard` automatically
4. **Lineage** — enter `memory_id` (or `entity_id` + namespace) and agent id under RBAC
5. **Audit** — filter by `memory_id` or `agent_id`
6. **Retrieval** — recent retrieval logs / token fields

## Equivalent HTTP

```http
GET /admin/dashboard?window_days=7
GET /admin/lineage?memory_id=<id>&agent_id=<caller>&limit=50
GET /admin/audit?memory_id=<id>&limit=50
GET /admin/retrieval-logs?limit=50
```

Headers when keyed: `X-API-Key: <key>` or `Authorization: Bearer <key>`.

## MCP (unchanged)

Operators can still use `archivist_memory_lineage`, `archivist_audit_trail`,
`archivist_savings_dashboard`, `archivist_health_dashboard`, and
`archivist_retrieval_logs` on ops/full — the billboard is the browser surface, not a
replacement for MCP.
