# ADR-014: Benchmark showcase validation (Option B contract)

<!-- INIT-014/SPEC-001 -->

**Status:** Accepted
**Date:** 2026-07-27
**Deciders:** Archivist maintainers
**Source:** [RSCH-001 — Benchmark showcase with real numbers](../../sdd/research/RSCH-001-benchmark-showcase-real-numbers/RSCH-001-benchmark-showcase-real-numbers-report.md)
(Option B recommendation); →
[INIT-014](../../sdd/initiatives/INIT-014-benchmark-showcase-validation/INIT-014-benchmark-showcase-validation-initiative.md);
→ prior: [ADR-013](ADR-013-observability-billboard.md) (Diff #8 UI productized — **not** a
retrieval-quality claim), [ADR-001](ADR-001-platform-coherence-sequencing.md) (bench honesty /
publish path).

## Decision

**Freeze** the operator contract for INIT-014: validate RSCH-001’s recommended **Option B —
Product showcase pack** against a live OpenAI-compatible stack, archive fresh
`.benchmarks/*.json` artifacts, and publish honest **v2.5.x** tables into
`docs/BENCHMARKS.md` (with README measured-savings links).

This ADR locks **endpoints**, **pack commands**, **honesty rules**, and **publish targets**
before any live preflight or harness run. It does **not** invent benchmark numbers.

### REQ-001 coverage

| Area | Frozen decision |
|---|---|
| **Endpoints** | Primary / curator / embed / Qdrant binding in § Endpoint map |
| **Pack** | Option B three-suite command freeze in § Showcase pack |
| **Honesty** | Option B only; BEIR footnote; Diff #8 UI ≠ retrieval scores; no invented numbers |
| **Publish path** | `docs/BENCHMARKS.md` header + tables from archived JSON; README measured link |

## Endpoint map (operator binding)

> Verified 2026-07-27 over Tailscale (INIT-014 planning). Prefer Tailscale IP/hostname from the
> laptop. Do **not** publish private LAN-only addresses in this ADR.

| Role | Preferred base (**no** `/v1` in `.env`) | Served model | Notes |
|---|---|---|---|
| **Primary LLM** | `http://100.91.115.22:11435` | `gemma4-uncensored` | Spark `vllm-fast`; **primary for this INIT** |
| **Primary LLM (optional alt)** | `https://vllm-mm-dev-leo-public.tail160017.ts.net` | *smoke `/v1/models` first* | MagicDNS; **optional** — use only when healthy; was **502** at plan time → fallback to Spark `:11435` |
| **Curator LLM** | `http://100.91.115.22:11436` | `qwen2.5-1.5b-instruct` | Spark `vllm-curator` |
| **Embeddings** | `http://100.91.115.22:8000` | `BAAI/bge-m3` | **VECTOR_DIM=1024** (Qdrant collection must match) |
| **Qdrant** | `http://127.0.0.1:6333` | — | Local compose for host-side benchmarks |
| **API keys** | empty / `EMPTY` | — | Lab vLLM has **no auth** (Tailscale-trust boundary) |

**Primary selection rule:** Prefer Spark `http://100.91.115.22:11435` until leo-public
`/v1/models` is healthy; then leo-public may be used as an optional primary. Unhealthy
leo-public → **always** fall back to Spark `:11435`.

### `.env` base URL shape

Bases must **not** include a trailing `/v1`. The app appends `/v1/chat/completions` (and
embeddings paths) to the configured base. The LLM client also strips a trailing `/v1` before
appending, but operators should still configure bases **without** `/v1` so embeddings and other
clients never see a **`/v1/v1/...`** double path.

Illustrative shape (write only into **gitignored** `.env` — never commit):

```env
LLM_URL=http://100.91.115.22:11435
LLM_MODEL=gemma4-uncensored
LLM_API_KEY=
CURATOR_LLM_URL=http://100.91.115.22:11436
CURATOR_LLM_MODEL=qwen2.5-1.5b-instruct
CURATOR_LLM_API_KEY=
EMBED_URL=http://100.91.115.22:8000
EMBED_MODEL=BAAI/bge-m3
VECTOR_DIM=1024
QDRANT_URL=http://127.0.0.1:6333
# Optional LongMemEval judge (default = primary):
# BENCHMARK_JUDGE_LLM_URL=http://100.91.115.22:11435
# BENCHMARK_JUDGE_LLM_MODEL=gemma4-uncensored
```

**GR-DIM-001:** Embeddings = `BAAI/bge-m3` → **VECTOR_DIM=1024**.

## Showcase pack (Option B — command freeze)

Execute in order after SPEC-002 preflight (`.env`, Qdrant, `/v1/models` smoke, CI markers).
Prefer host-side `PYTHONPATH=src` (GR-CE-001). Date stamp in filenames is illustrative — use the
actual run date.

```bash
# A) Token efficiency (Answer Finder / measured-savings claim)
PYTHONPATH=src python -m benchmarks.token_efficiency \
  --output .benchmarks/token_efficiency_YYYYMMDD.json

# B) Pipeline product slice (retrieval quality / latency)
env REVERSE_HYDE_ENABLED=false TIERED_CONTEXT_ENABLED=false QUERY_EXPANSION_ENABLED=false \
  PYTHONPATH=src python -m benchmarks.pipeline.evaluate \
  --memory-scale small --variants clean_reranker --no-refine \
  --output .benchmarks/pipeline_small_clean_reranker_YYYYMMDD.json --print-slices

# C) LongMemEval thin (industry memory-assistant frame)
SKIP_BEIR=1 LIMIT_LM=50 BENCHMARK_FAST=1 bash benchmarks/scripts/run_thin_reference.sh
# Optional later credibility bump (not required for INIT-014 Done):
# SKIP_BEIR=1 LIMIT_LM=200 BENCHMARK_FAST=0 bash benchmarks/scripts/run_thin_reference.sh
```

**Suite scope (normative):**

1. **token_efficiency** — product token-savings claim.
2. **pipeline** — `small` + `clean_reranker` only (not full ablation / academic maximal).
3. **LongMemEval thin** — `LIMIT_LM≥50`, label `BENCHMARK_FAST` / thin limits in publish docs.

Archive JSON under `.benchmarks/` (gitignored). Publish summaries only via SPEC-004.

## Honesty rules (frozen)

| ID | Rule |
|---|---|
| **GR-RSCH-001** | **Option B only** — no academic maximal BEIR headline as the showcase. |
| **GR-BEIR-001** | BEIR thin (if run at all) is a **footnote** / ROADMAP regression log only — not the product headline. |
| **GR-DIFF8-001** | Do **not** claim Diff #8 `/admin/ui/` improves LongMemEval or retrieval scores. UI is control-plane observability, not a retrieval stage. |
| **GR-HONEST-001** | Numbers must come from archived JSON for this initiative’s run date; label thin / `BENCHMARK_FAST` limits. **No invented numbers.** |
| **GR-SECRET-001** | Never commit `.env`, API keys, or HF tokens. Tailscale hostnames/IPs are operator-owned and OK here; prefer Tailscale over private LAN in public docs. |

## Publish targets

| Target | Contract |
|---|---|
| `docs/BENCHMARKS.md` | Refresh header to current package/`__version__` at run time (expect **v2.5.x**) + dated run; tables sourced from the three Option B JSON artifacts |
| README measured savings | Link to the refreshed BENCHMARKS section (SPEC-004) |
| Version stamp | Use live package version at publish time — do not invent a version string in this ADR |

Publishing tables is **SPEC-004**. This ADR only freezes *what* may be published and from which artifacts.

## Non-goals

- Academic **BEIR** as the headline showcase metric (footnote / regression log only).
- Claiming Diff #8 `/admin/ui/` (or any observability billboard) as a **retrieval-quality** win.
- Running benchmarks or editing BENCHMARKS tables in this spec (SPEC-001 = contract only).
- Option A (CI markers alone as the public showcase) or Option C (academic maximal pack).
- Committing `.env`, secrets, or private LAN-only ops notes into the public tree.
- Inventing or backfilling numeric results before SPEC-003 produces JSON.

## Context

Canonical `docs/BENCHMARKS.md` still headers **v2.3.0 — 2026-04-25**. RSCH-001 scored
**Option B** highest for honesty vs claims, external credibility, and in-repo maintainability,
but live numbers were not run during research (missing `.env` / Qdrant). INIT-014 executes that
runbook. Diff #8 is Done under ADR-013; it must not be conflated with retrieval/LongMemEval
scores.

## Alternatives considered

| Option | Verdict | Why |
|---|---|---|
| A. CI markers only (`coach_core` / `agentic_memory`) | **Reject** for showcase | Proves contracts, not publishable LongMemEval/pipeline/token tables (RSCH-001) |
| B. **Product showcase pack** (token_efficiency + pipeline small/`clean_reranker` + LongMemEval thin) | **Chosen** | RSCH-001 weighted winner; maps to marketed claims |
| C. Academic maximal (full LongMemEval 500 + full BEIR + scale sweep) | **Reject** | TCO / time; README already rejects BEIR as headline |

## Consequences

### Positive

- SPEC-002+ share one endpoint and pack contract — no conflicting primaries mid-run.
- Honesty rules are ADR-normative before any table rewrite.
- Publish path is explicit: JSON → BENCHMARKS.md → README link.

### Negative / follow-ups

- leo-public MagicDNS may remain flaky — Spark `:11435` is the durable primary.
- LongMemEval wall-clock dominates; thin `LIMIT_LM=50` + `BENCHMARK_FAST=1` must be labeled.
- Unauthenticated lab vLLM over Tailscale is accepted for this operator lab — document and do
  not put secrets in public docs (SPEC-005 deepens the security review).

### Spec mapping

| Spec | Implements |
|---|---|
| SPEC-001 | This ADR (Accepted) — contract freeze |
| SPEC-002 | Preflight: `.env`, Qdrant, LLM/embed smoke, CI markers |
| SPEC-003 | Execute Option B pack → `.benchmarks/` JSON |
| SPEC-004 | Publish `docs/BENCHMARKS.md` + README measured link |
| SPEC-005 | Security Review |
| SPEC-006 | Architecture / design diagrams |

## Security notes (light)

- **No API keys or tokens** belong in this ADR or in committed docs.
- Lab vLLM endpoints are **unauthenticated** OpenAI-compatible APIs reachable over **Tailscale**;
  treat Tailscale membership as the trust boundary for this validation run.
- `.env` remains gitignored; SPEC-002 may write local env only under the `secret_change` gate
  discipline (never commit).

## References

- RSCH-001 report — Option B recommendation + command order
- INIT-014 initiative §5 endpoint inventory; §3 REQ-001 / guardrails
- `docs/BENCHMARKS.md` — current (stale) publish surface
- `benchmarks/README.md`, `benchmarks/token_efficiency.py`, `benchmarks/pipeline/evaluate.py`,
  `benchmarks/scripts/run_thin_reference.sh`
- `src/archivist/features/llm.py` — OpenAI base normalization (avoid `/v1/v1`)
- ADR-013 — Diff #8 UI (orthogonal to retrieval scores)
