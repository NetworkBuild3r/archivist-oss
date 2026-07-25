# ADR-001: Platform coherence sequencing

<!-- INIT-001/SPEC-001 -->

**Status:** Accepted
**Date:** 2026-07-24
**Deciders:** Archivist maintainers
**Source:** [ASMT-001](../../sdd/assessments/ASMT-001-archivist-oss/) (assessment) →
[INIT-001](../../sdd/initiatives/INIT-001-platform-coherence-and-memory-product/INIT-001-platform-coherence-and-memory-product-initiative.md)

## Decision

Sequence INIT-001 as **platform-debt-first, then product-differentiators, in
this fixed order**:

1. **P0 — Coherence** (this spec): docs/version sync so planning stops
   thrashing against a stale narrative.
2. **P1 — Compounding platform debt**: shim removal (`SPEC-002`) →
   `graph.py` decomposition (`SPEC-003`) → outbox default-on (`SPEC-004`),
   with CI quality ratchet (`SPEC-005`) parallel after `SPEC-002`.
3. **Phase 8 — Intelligent lifecycle** (`SPEC-006`): contradiction resolution
   + reflection hooks, built on the now-default outbox.
4. **Phase 7 — Checkpointing** (`SPEC-007` data, `SPEC-008` API): a store
   distinct from the L0–L2 Answer Finder delivery tiers (GR-002).
5. **Memory as a Product** (`SPEC-009`): fork/export/version, building on
   `versioning.py` + backup fragments and reusing the `SnapshotPathError`
   containment helper from PR #39.
6. **Coordination beyond handoff** (`SPEC-010`): depends on both lifecycle
   (step 3) and Memory-as-Product (step 5).
7. **Observability depth** (`SPEC-011`): independent of the product track,
   runs parallel to steps 3–6 once `SPEC-005` lands.
8. **Security review** (`SPEC-012`) and **architecture/design diagrams**
   (`SPEC-013`) close the initiative.

This ADR is the authoritative answer to "what order" for INIT-001; specs
cite it instead of re-litigating sequencing.

## Context

ASMT-001 found the codebase materially ahead of its own documentation:
`INSPIRATION.md` was frozen at v1.3 while four releases (v2.0–v2.3) shipped;
`ROADMAP.md` listed Pydantic Settings as greenfield future work when
`ArchivistSettings` already existed (only the Phase-B alias removal
remained); and `__version__` (`2.0.1`) had drifted from the documented
release (`2.3.0`). Independently, PR #39 (Security Hardening) landed after
the assessment and **rejected** a previously-documented OpenClaw
compatibility bypass (accepting the literal, unresolved
`Bearer ${ARCHIVIST_API_KEY}` string as valid auth) — `ROADMAP.md` Phase 6.5
still described the old, since-reverted behavior as current.

Two guardrails constrain the sequencing choice:

- **GR-001** — do not re-vendor ReMe / OpenViking / zer0dex trees; inspiration
  is credited, not re-implemented.
- **GR-002** — keep the L0–L2 Answer Finder tiers separate from the Phase-7
  checkpoint/time-travel data model; do not conflate delivery tiering with
  agent-state checkpointing.

## Alternatives considered

| Option | Description | Why not chosen |
|---|---|---|
| **Product-first** — ship Phase 8/7/Memory-as-Product immediately, defer shim removal / `graph.py` split / outbox default-on | Gets differentiator features (checkpointing, Memory-as-Product) in front of users sooner | New service/API work (`SPEC-006`–`SPEC-010`) would build on top of the dual-write outbox path, the `graph.py` monolith, and Phase-5 import shims — each new feature inherits and compounds that debt instead of retiring it. Checkpoint durability specifically needs the outbox as the default write path, not an opt-in. |
| **Docs-last** — treat INSPIRATION/ROADMAP/version sync as low-priority cleanup after the code work | Avoids "wasting" a spec slot on non-functional docs | ASMT-001's own top P0 finding is that stale docs are actively driving planning thrash (e.g., ROADMAP arguing for Pydantic work that mostly already exists). Sequencing product work atop an incoherent narrative risks re-deciding already-settled scope inside later specs — this ADR exists specifically to prevent that. |
| **Big-bang platform rewrite** — split `graph.py`, flip the outbox default, and remove shims in one combined spec | Fewer PRs / gate cycles | Violates single-domain spec sizing (SDD guidance: 200–800 LOC / 2–6h per spec) and mixes `data` (schema/module split) with `service` (outbox semantics) with `infrastructure` (import canonicalization) domains, each with different blast radius and rollback needs. Kept as three specs (`SPEC-002`, `SPEC-003`, `SPEC-004`) with a linear dependency chain instead. |
| **Reintroduce the OpenClaw Bearer-placeholder acceptance** while updating docs to describe it as intentional | Keeps the original "compatibility fix" framing intact | PR #39 already rejected this on security grounds — accepting a literal, unresolved template string as auth is a fixed, guessable-in-advance bypass, not a compatibility shim. This ADR does not reopen that decision; `ROADMAP.md` Phase 6.5 is rewritten to document the **current** (rejecting) behavior and the supported `X-API-Key` / resolved-Bearer fix instead. |

## Consequences

**Positive:**

- Every subsequent INIT-001 spec inherits a coherent, version-accurate
  narrative — no spec needs to re-derive "is Pydantic done?" or "what does
  `__version__` say?" from first principles.
- Product-track specs (`SPEC-006`–`SPEC-011`) build on a retired-shim,
  split-graph, outbox-default platform instead of adding new surface area to
  debt that would otherwise need a second migration later.
- The OpenClaw guidance in `ROADMAP.md` now matches the shipped, security-
  reviewed behavior — no future spec can accidentally "fix" the 401 by
  reintroducing the placeholder-acceptance bypass without contradicting a
  documented decision.

**Negative / accepted trade-offs:**

- Differentiator features (checkpointing, Memory-as-Product, coordination —
  ASMT-001's stated high-value items) are pushed later in the calendar
  (`SPEC-006` onward) rather than shipped first; this is an explicit trade
  of short-term visible product progress for platform durability.
- `SPEC-005` (CI ratchet) and `SPEC-011` (observability) run on a side track
  parallel to the product chain; if either slips, it does not block Phase
  7/8/Memory-as-Product but does mean quality-gate hardening and lineage/cost
  observability may lag behind the product surface they should be
  instrumenting.
- This ADR fixes an order; if a later assessment finds new debt or a new
  security finding (as PR #39 did mid-initiative), the sequencing may need
  a follow-up ADR rather than silent renumbering of specs.

## Implementing specs

| Spec | What it delivers |
|---|---|
| [SPEC-001](../../sdd/initiatives/INIT-001-platform-coherence-and-memory-product/specs/SPEC-001-docs-coherence-adr-version-sync.md) | This ADR; INSPIRATION/ROADMAP/version coherence (current spec) |
| SPEC-002 | Shim removal, canonical `archivist.*` imports |
| SPEC-003 | `graph.py` → schema / entities-facts / FTS module split |
| SPEC-004 | Outbox production default; inline Qdrant path deprecation window |
| SPEC-005 | CI quality ratchet (mypy, coverage, ruff) |
| SPEC-006 | Phase 8 contradiction resolution + reflection |
| SPEC-007 / SPEC-008 | Phase 7 checkpoint store + resume/replay MCP tools |
| SPEC-009 | Memory-as-Product fork/export/version |
| SPEC-010 | Coordination primitives beyond handoff |
| SPEC-011 | Observability lineage + cost signals |
| SPEC-012 / SPEC-013 | Security review and architecture/design diagrams (initiative close) |

See [`docs/ROADMAP.md`](../ROADMAP.md) for the product-facing phase
descriptions and [`docs/INSPIRATION.md`](../INSPIRATION.md#post-v13-v20v23)
for what shipped between v1.3 and v2.3.
