# ADR-013: Observability control-plane billboard (Diff #8 productize)

<!-- INIT-013/SPEC-001 -->

**Status:** Accepted
**Date:** 2026-07-27
**Deciders:** Archivist maintainers
**Source:** [BRAIN-005 — Complete Unique Differentiators](../../sdd/brainstorms/BRAIN-005-complete-unique-differentiators/decision-document.md)
(Phase 4); plumbing: `app/lineage.py`, `app/dashboard.py`, `core/audit.py`,
`handlers/tools_admin.py`, `GET /admin/dashboard` + `/admin/retrieval-logs`; →
[INIT-013](../../sdd/initiatives/INIT-013-observability-billboard/INIT-013-observability-billboard-initiative.md);
→ prior: [ADR-012](ADR-012-checkpoint-time-travel.md) (Diff #7 Done scoped; deferred Diff #8),
[ADR-001](ADR-001-platform-coherence-sequencing.md) (lineage/cost side-track).

## Decision

**Productize** Unique Differentiator **#8** (Observability Dashboard) by shipping a **thin,
served control-plane billboard UI** over **existing** lineage / audit / savings / health JSON —
and adding the **missing browser-facing HTTP** for lineage and audit — **without** inventing a
second memory runtime, React/Vite product shell, cookie auth rewrite, or net-new **core** MCP
tools.

BRAIN-005 **rejected** Option E (mark Done via JSON/MCP alone). Diff #8 Done requires a real
explorer operators can open in a browser.

### Plumbing vs Diff #8 product

| Layer | Meaning | Status entering INIT-013 |
|---|---|---|
| **Observability plumbing** | MCP: `archivist_memory_lineage`, `archivist_audit_trail`, `archivist_savings_dashboard`, `archivist_health_dashboard`, `archivist_retrieval_logs`; HTTP: `GET /admin/dashboard`, `GET /admin/retrieval-logs`; builders in `lineage.py` / `dashboard.py` / `audit.py` | **Shipped** |
| **Differentiator #8 product** | Served UI + HTTP lineage/audit + auth honesty + tests + docs — ROADMAP Done + Phase 9 checkbox under this ADR | **This INIT** |

### Product contract (INIT-013)

1. **UI tech (GR-THIN-001)** — **Static HTML + CSS + JS** (no React/Vite/npm app). Optional
   Chart.js via **CDN or vendored** file only if KPI charts need it; tables alone are acceptable
   for v1. Assets live at:

   ```text
   src/archivist/app/static/admin_ui/
     index.html
     app.js
     styles.css
   ```

2. **Mount path** — Serve under **`/admin/ui/`** (trailing-slash index) via Starlette
   `StaticFiles` (or equivalent) from `main.py`. Relative fetches to `/admin/*` JSON APIs.
   Mutating `/admin/*` (backup/restore/import) are **not** first-class billboard panels
   (document links in REFERENCE only).

3. **HTTP read APIs (REQ-002)** — Add browser-consumable JSON (reuse builders; do not
   duplicate SQL):

   | Method + path | Behavior | Authz |
   |---|---|---|
   | `GET /admin/lineage` | Memory or entity lineage (query: `namespace`, `memory_id` and/or `entity_id`, `limit`) | Same API-key middleware as other `/admin/*`; **namespace read RBAC** when namespace is provided (parity with MCP `archivist_memory_lineage`) |
   | `GET /admin/audit` | Audit trail (query: `agent_id`, `memory_id`, `limit`) | Same API-key middleware; fail closed on invalid params; cap `limit` |
   | `GET /admin/dashboard` | **Reuse** (health + savings aggregates) | Existing |
   | `GET /admin/retrieval-logs` | **Reuse** | Existing |

   No dedicated `/admin/savings` alias required (dashboard already carries savings stats).

4. **Auth (GR-AUTH-001)**

   - Billboard and new GETs inherit **`ArchivistAuthMiddleware`** (Bearer / `X-API-Key` when
     `ARCHIVIST_API_KEY` is set).
   - **No cookie sessions**, no weaker static-only bypass, no anonymous write admin.
   - When `ARCHIVIST_API_KEY` is **unset**, behavior matches today (open port + critical log
     warning) — operators **must** set a key on any exposed deployment; REFERENCE must say so
     explicitly for the UI.
   - UI: operator pastes API key into a prompt/field; optional `sessionStorage` only (not
     committed; clear warning). Prefer `X-API-Key` header on `fetch`.

5. **Panels (v1 read-only)**

   | Panel | Source |
   |---|---|
   | Health / savings KPIs | `/admin/dashboard` |
   | Lineage explorer | `/admin/lineage` |
   | Audit trail | `/admin/audit` |
   | Retrieval / cost | `/admin/retrieval-logs` + dashboard token fields |

6. **MCP** — Existing admin MCP tools **remain**. INIT-013 does **not** add core tools and
   does **not** require new ops MCP tools for Diff #8 Done (HTTP is the browser contract).

7. **XSS / disclosure** — Render untrusted memory/audit strings via `textContent` / escaped
   HTML only. Cap list sizes server-side. Do not log API keys.

### Frozen guardrails

| ID | Rule |
|---|---|
| **GR-DIFF8-001** | Diff #8 = **served thin UI** + HTTP lineage/audit over existing builders — not JSON-only Done. |
| **GR-UI-001** | Explorer/viz only — **not** a second memory runtime or agent orchestrator. |
| **GR-THIN-001** | Static HTML/JS; no React/Vite toolchain this INIT. |
| **GR-AUTH-001** | Same middleware as `/admin/*`; no cookie invent; no static auth bypass. |
| **GR-PROD-002** | **No net-new core MCP tools.** |
| **GR-WEDGE-001** | Diff #8 only — no MaP/checkpoint/curator revisits; no skill OS. |
| **GR-CE-001** / **GR-COACH-001** (carry) | Cite-or-refuse; `-m coach_core` / `agentic_memory` green. |

### Diff #8 Done criteria (ROADMAP claim)

Diff #8 may be marked **Done** and Phase **9** tracking checked only when all hold:

1. ADR-013 Accepted (this doc).
2. `GET /admin/lineage` + `GET /admin/audit` + `/admin/ui/` mount shipped (INIT-013 SPEC-002).
3. Billboard UI panels (health/savings, lineage, audit, retrieval/cost) shipped (SPEC-003).
4. Smoke/e2e + auth denial tests; markers `coach_core` / `agentic_memory` green (SPEC-004).
5. REFERENCE / README / CHANGELOG / ROADMAP #8 → **Done**; Phase 9 checked; Immediate Next →
   BRAIN-005 program close / maintenance (SPEC-005).
6. Security Review: 0 unresolved Critical/High (SPEC-006).
7. Architecture Mermaid for browser → HTTP → builders (SPEC-007).

**Out of Diff #8 Done (deferred):** React control plane; write-heavy admin console; Grafana
replacement; multi-tenant SSO; institutional tier DDL (still ADR-012 deferred).

## Context

ROADMAP Diff #8 has long been **Partial**: operators can query lineage/audit/savings via MCP
or scrape `/admin/dashboard` JSON, but there is no served memory-explorer UI. Lineage and audit
lack HTTP, so a browser billboard cannot work without SPEC-002. INIT-013 closes that gap under
one ADR so SPEC-002/003 do not invent conflicting stacks.

## Alternatives considered

| Option | Verdict | Why |
|---|---|---|
| A. Mark Done via MCP/JSON only (BRAIN-005 Option E) | **Reject** | Dishonest “dashboard” claim |
| B. Full React/Vite SPA + design system | **Reject** | GR-THIN-001 / GR-UI-001; scope explosion |
| C. Embed Grafana / external APM only | **Reject** | Leaves Archivist Diff #8 Partial; optional later |
| D. Cookie session auth rewrite | **Reject** | Out of wedge; keep API-key middleware |
| E. **Static `/admin/ui/` + HTTP lineage/audit + reuse dashboard** | **Chosen** | Matches BRAIN-005 Phase 4 |

## Consequences

### Positive

- Honest Diff #8 / Phase 9 Done with a real operator-facing explorer.
- Reuses lineage/dashboard/audit builders — no parallel query stack.
- Core coach MCP surface unchanged.

### Negative / follow-ups

- Static UI will look utilitarian vs a design-system product — acceptable under GR-THIN-001.
- Open-port-when-key-unset remains a deployment footgun — docs + existing critical log; do not
  silently break local-dev open mode without a separate INIT.
- XSS and IDOR become first-class Security Review focus (SPEC-006).

### Spec mapping

| Spec | Implements |
|---|---|
| SPEC-001 | This ADR (Accepted) |
| SPEC-002 | HTTP lineage/audit + `/admin/ui/` mount |
| SPEC-003 | Billboard static UI |
| SPEC-004 | Smoke / e2e + markers |
| SPEC-005 | ROADMAP / REFERENCE / CHANGELOG |
| SPEC-006 | Security Review |
| SPEC-007 | Architecture diagrams |

## References

- BRAIN-005 decision / action-plan Phase 4 / GR-UI-001
- `src/archivist/app/lineage.py`, `dashboard.py`, `main.py`
- `src/archivist/core/audit.py`
- `src/archivist/app/handlers/tools_admin.py`
- Offline-only `docs/benchmark-dashboard.html` is **not** Diff #8 (benchmarks, not memory explorer)
