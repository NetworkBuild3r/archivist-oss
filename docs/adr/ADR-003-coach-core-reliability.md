# ADR-003: Coach-core reliability contract

<!-- INIT-003/SPEC-001 -->

**Status:** Accepted
**Date:** 2026-07-25
**Deciders:** Archivist maintainers
**Source:** [BRAIN-001 amendment — personal production](../../sdd/brainstorms/BRAIN-001-best-multi-agent-memory/amendment-personal-production.md)
→ [INIT-003](../../sdd/initiatives/INIT-003-coach-core-reliability/INIT-003-coach-core-reliability-initiative.md)

## Decision

Freeze Archivist’s **immediate engineering north star** as **coach-core
reliability**: make the **five-tool coach path** boringly reliable under real
turn latency, then enrich that same contract — do **not** broaden into Phase
7–10 product wedges or an OSS “billboard” success model.

The frozen **core tool contract** is exactly these five MCP tools (names and
roles):

| Tool | Role |
|---|---|
| `archivist_store` | Durable write of coaching insights; fast ack under outbox |
| `archivist_search` | Recall / search with stable, usable payload shape |
| `archivist_get_context` | Token-budgeted packed context for a turn |
| `archivist_index` | Navigational MEMORY INDEX catalog with clear refresh semantics |
| `archivist_forget` | Idempotent prune / remove (delete path the coach already uses) |

That contract is enriched along **five pillars** (REQ-003…REQ-007). Later
INIT-003 specs implement them; this ADR freezes the meaning so specs do not
re-litigate scope:

1. **Provenance envelope** — store and retrieve carry source, subject,
   confidence, sensitivity, purpose, inferred vs user-stated, and timestamps.
2. **Pre-rank filters** — `search` / `get_context` filter by subject, purpose,
   sensitivity, supersession, and freshness **before** ranking.
3. **Supersede / suppress** — corrections supersede prior memories; suppress
   hides without erase; governed delete remains available via forget.
4. **Fast durable write** — `archivist_store` acknowledges within a
   client-aligned budget (**≤4s** ack target under healthy deps / outbox
   default-on); Qdrant lag must not stall the turn.
5. **Stable recall shape** — normalized memory-list payload with usable text
   always present in a documented field; end dual-call ambiguity between
   search and get_context.

**Default surface:** deploy / `list_tools` defaults to a **core** profile
(~5–12 tools) that includes this five-tool set. Broader `fleet` / `ops` /
`full` profiles remain opt-in. Unfinished wedges stay off the default surface
(GR-PROD-003).

This ADR is the authoritative answer to “what is next?” after INIT-001’s
platform-coherence sequencing ([ADR-001](ADR-001-platform-coherence-sequencing.md)):
INIT-003 coach-core reliability, not Phase 7–10 breadth.

## Context

[BRAIN-001](../../sdd/brainstorms/BRAIN-001-best-multi-agent-memory/) originally
oriented toward Memory-as-Product packaging, fleet demos, and broader
coordination. The **personal-production amendment** (2026-07-25) pivots: a
reference consumer (My AI Fitness / myaicoach-harness pattern — **out of this
repo’s edit scope**) already talks to Archivist over streamable HTTP MCP and
uses **only** the five tools above. The harness compensates today for unfinished
server behavior (write timeouts ~4s, circuit breaker, dedup, parallel
get_context+search when refine leaves answers empty, local pruner).

Unfinished ~50-tool surface area and parked INIT-001 product wedges hurt more
than missing differentiators. Success for INIT-003 is **coach-path reliability
for that personal production pattern** — OSS community reaction and Phase 7–10
breadth are **explicitly out of success criteria** (GR-PROD-001).

Two prior guardrails still apply:

- **GR-001** — do not re-vendor ReMe / OpenViking / zer0dex trees.
- **GR-002** — keep L0–L2 Answer Finder tiers separate from Phase-7
  checkpoint/time-travel models; this ADR parks checkpoint UX entirely for
  INIT-003.

## Alternatives considered

| Option | Description | Why not chosen |
|---|---|---|
| **Continue ADR-001 product track next** — Phase 8 lifecycle → Phase 7 checkpoints → MaP → coordination (Phase 10) → observability (Phase 9) | Keeps the INIT-001 differentiator calendar intact | The live consumer does not call share_*, checkpoint_*, or MaP MCP APIs. Shipping those next leaves the five-tool path unfinished while the harness keeps compensating. ADR-001 remains valid history for INIT-001; it is **not** the immediate engineering focus. |
| **Fleet / OSS billboard first** — MaP MCP golden path, three marketing eval scenarios, tool packaging for Cursor/Claude | Maximizes external narrative | Amendment and GR-PROD-001 reject OSS adoption as a success metric; personal-prod reliability is the north star. |
| **Net-new coach tools** — add specialized MCP tools for provenance, filters, or “smart recall” | Cleaner API for new fields | GR-PROD-002: enrich the existing five tools unless that contract is proven insufficient. New names force harness churn for no consumer gain. |
| **Keep default `list_tools` = full registry** — hide nothing; document “use these five” in prose only | Avoids profile plumbing | Operators and agents still see unfinished wedges; default core profile (GR-PROD-003) is the decision so unfinished surface stays off by default. |

## Consequences

**Positive:**

- INIT-003 specs inherit a frozen tool list, pillar list, write SLO (≤4s),
  default core profile, and explicit park list — no re-deciding MaP vs core
  inside later data/service/api work.
- Harness-shaped success criteria (store → index refresh → recall; fail-fast
  under dead Qdrant; namespace isolation) can drive evals without marketing
  scenario sprawl.
- ADR-001’s Phase 7–10 sequencing remains on disk as INIT-001 history; this
  ADR redirects **immediate** engineering attention without rewriting that
  initiative’s internal order.

**Negative / accepted trade-offs:**

- Differentiator phases (checkpoints, MaP MCP, share/coordination UX,
  observability depth) stay parked for this initiative even where partial code
  already exists — intentional under GR-PROD-001 / GR-PROD-002.
- Enriching five tools (additive fields, filters, supersession) may require
  careful backward-compatible shaping so the reference harness does not break;
  contract tests in later specs own that risk.
- Domain MCP / `myaifitness-android` changes are out of scope here; if the
  consumer needs client edits, that is a separate effort (GR-PROD-004).

## Non-goals (INIT-003)

Explicitly **out of scope** for work governed by this ADR:

- **MaP MCP** and Memory-as-Product fleet / golden-path demos
- **`share_*`** coordination depth and related UX
- **Checkpoint** / time-travel UX (Phase 7 product track)
- **Edits inside `myaifitness-android`** (or other consumer repos) — reference
  only
- **OSS billboard** metrics (stars, marketing evals, community adoption) as
  success criteria
- Net-new MCP tool names unless the five-tool contract is later proven
  insufficient (would require a follow-up ADR)

## Implementing specs

| Spec | What it delivers |
|---|---|
| [SPEC-001](../../sdd/initiatives/INIT-003-coach-core-reliability/specs/SPEC-001-docs-adr-coach-core-contract.md) | This ADR; ROADMAP “Immediate Next Steps” pointer (current spec) |
| SPEC-002 | Provenance + supersede/suppress schema |
| SPEC-003 | Default **core** MCP tool profile |
| SPEC-004 | Fast durable store (outbox ack SLO ≤4s) |
| SPEC-005 | Stable recall + pre-rank filters on search/get_context |
| SPEC-006 | Store/index/forget(+suppress) contract enrichment |
| SPEC-007 | Supersede/suppress/correct lifecycle |
| SPEC-008 | Coach-path CI eval scenarios |
| SPEC-009 / SPEC-010 | Security review and architecture/design diagrams (initiative close) |

See [`docs/ROADMAP.md`](../ROADMAP.md) for the product-facing pointer to this
ADR, and the BRAIN-001 amendment for the personal-production rationale.

## Store ack semantics (INIT-003/SPEC-004)

With **`OUTBOX_ENABLED=true` (default)**, `archivist_store` **acks after the
durable graph commit** that includes the outbox row (SQLite/Postgres). That
ack means the write is durable and will be applied to Qdrant asynchronously by
`OutboxProcessor` — **not** that Qdrant has already upserted the point.

Pre-transaction similarity checks (`conflict_detection._query_similar`) are
**fail-open and budgeted** (`CONFLICT_QUERY_TIMEOUT_S`, default 1s) so a dead
or hanging Qdrant cannot approach the shared client’s ~30s timeout and stall
coach clients toward ~15s. Soft ack budget logging uses `STORE_ACK_BUDGET_MS`
(default 4000). If the graph pool is not initialized, store fails fast with
`error=graph_pool_unavailable` (no Qdrant wait).
