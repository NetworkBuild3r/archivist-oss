# Archivist benchmark results

> **Latest showcase snapshot: v2.5.0 — 2026-07-27** (INIT-014 / Option B product pack)
>
> **Models:** primary `gemma4-uncensored` · curator `qwen2.5-1.5b-instruct` · embeddings `BAAI/bge-m3` (1024-dim) · local Qdrant.
> Lab inference was OpenAI-compatible over Tailscale (unauthenticated; Tailscale membership = trust boundary). Model IDs only below — no private LAN topology.
>
> **Labels:** `BENCHMARK_FAST=1` · LongMemEval thin `LIMIT_LM=50` · **BEIR skipped** (not a product headline; see footnote).
> **Diff #8** `/admin/ui/` observability billboard is **not** a retrieval-quality claim (ADR-013 / ADR-014).
>
> Evidence JSON (gitignored locally; do not commit):
> `.benchmarks/token_efficiency_20260727.json` ·
> `.benchmarks/pipeline_small_clean_reranker_20260727.json` ·
> `.benchmarks/longmemeval_thin.json`

For **CI-style test commands** (unit suite, `tests/qa/`, lint, mypy), see [`QA.md`](QA.md).

---

## Latest showcase pack (v2.5.0 / 2026-07-27)

Product pack per [ADR-014](adr/ADR-014-benchmark-showcase-validation.md): token efficiency + pipeline `small`/`clean_reranker` + LongMemEval thin. Numbers are **actuals** from the archived JSON — not targets, not rounded-up marketing claims.

### Token efficiency (Answer Finder)

**Artifact:** `.benchmarks/token_efficiency_20260727.json` (`run_at` 2026-07-27T20:11:55Z) · agent `benchmark_agent` · 49 queries · `max_tokens=8000`

| Policy | Queries | Avg savings % | Min % | Max % | Tokens saved | Avg duration (ms) |
|--------|---------|---------------|-------|-------|--------------|-------------------|
| adaptive | 49 | **0.0** | 0.0 | 0.0 | 0 | 719.7 |
| l0_first | 49 | 0.0 | 0.0 | 0.0 | 0 | 61.3 |
| l2_first | 49 | 0.0 | 0.0 | 0.0 | 0 | 35.4 |

**Why 0.0%:** the live store had **no memories** for `benchmark_agent`, so every query returned empty sources (`result_count=0`, `tokens_returned`/`tokens_naive` null). With nothing to pack, savings vs naive L2 is definitionally zero. This run does **not** meet a ≥60% adaptive target — do not cite it as such. Re-run after indexing a real corpus for the measured-savings claim.

### Pipeline — `memory_scale=small` / `clean_reranker`

**Artifact:** `.benchmarks/pipeline_small_clean_reranker_20260727.json` · 108 queries · refine off · CrossEncoder via `sentence-transformers`

| Variant | R@1 | R@5 | R@10 | NDCG@5 | NDCG@10 | p50 (ms) | p95 (ms) | Tok/Q | Pool | Wall (s) |
|---------|-----|-----|------|--------|---------|----------|----------|-------|------|----------|
| clean_reranker | 0.4815 | **0.5975** | 0.6813 | **0.8693** | 0.8853 | 1422.8 | 3135.6 | 1378.0 | 57.2 | 208.6 |

#### Retention slices

| Variant | temporal (R@5 / NDCG@5) | needle (R@5 / NDCG@5) | multi_hop (R@5 / NDCG@5) | single_hop (R@5 / NDCG@5) |
|---------|-------------------------|----------------------|--------------------------|---------------------------|
| small/clean_reranker | 0.7333 / 0.9722 (n=4) | — (none in small filter) | 0.4970 / 0.8985 (n=11) | 0.5737 / 0.8167 (n=52) |

### LongMemEval thin (`LIMIT_LM=50`, `BENCHMARK_FAST`)

**Artifact:** `.benchmarks/longmemeval_thin.json` · judge model `gemma4-uncensored` · refine off · curator off · **BEIR not run**

| Metric | Value |
|--------|-------|
| evaluated_questions | 50 |
| overall_accuracy | **0.04** |
| task_averaged_accuracy | 0.04 |
| retrieval recall@5 / NDCG@5 | **0.0** / 0.0 |
| retrieval recall@10 / NDCG@10 | 0.0 / 0.0 |

Thin / fast limits apply — treat as a labeled credibility probe, not a full LongMemEval (n=500) result.

### BEIR (footnote only)

**BEIR was skipped** on this showcase run (`SKIP_BEIR=1`). Per ADR-014 / GR-BEIR-001, BEIR is a regression-log / academic footnote — **not** the product headline metric.

### How to reproduce (Option B)

```bash
# A) Token efficiency
PYTHONPATH=src python -m benchmarks.token_efficiency \
  --output .benchmarks/token_efficiency_YYYYMMDD.json

# B) Pipeline product slice
env REVERSE_HYDE_ENABLED=false TIERED_CONTEXT_ENABLED=false QUERY_EXPANSION_ENABLED=false \
  OUTBOX_ENABLED=false \
  PYTHONPATH=src python -m benchmarks.pipeline.evaluate \
  --memory-scale small --variants clean_reranker --no-refine \
  --output .benchmarks/pipeline_small_clean_reranker_YYYYMMDD.json --print-slices

# C) LongMemEval thin (BEIR skipped)
SKIP_BEIR=1 LIMIT_LM=50 BENCHMARK_FAST=1 bash benchmarks/scripts/run_thin_reference.sh
```

Configure `.env` with your OpenAI-compatible primary/curator/embed bases (**without** a trailing `/v1`) and local `QDRANT_URL`. See [ADR-014](adr/ADR-014-benchmark-showcase-validation.md).

---

## Historical — pipeline evaluation (v2.3.0 / 2026-04-25) — superseded

> **Superseded** as the “latest” header by the v2.5.0 showcase above. Kept for comparison only.

**Run:** `scale=small`, **2 variants** (`clean_reranker`, `vector_plus_synth`), **108 queries** each.
**Artifacts:** Full JSON is written locally to `.benchmarks/phase5_semantic_chunking.json` when you reproduce the run (directory is gitignored).

**Session log (excerpt):**

```text
vector_plus_synth: 100%|…| 108/108 [11:11<00:00,  6.22s/q, mrr=0.741, recall=0.578]
R@1=0.3451  R@5=0.5778  R@10=0.6819  NDCG@5=0.7407  p50=7577ms  tok/q=1176  wall=671.8s
Session complete: scale=small  2 variants  2212.9s total
```

### Overall by variant (historical)

| Variant | R@1 | R@5 | R@10 | NDCG@5 | NDCG@10 | p50 (ms) | p95 (ms) | Tok/Q | Synth Hits | Pool | Wall (s) |
|---------|-----|-----|------|--------|---------|----------|----------|-------|------------|------|----------|
| clean_reranker | 0.4389 | 0.6208 | 0.7185 | 0.8580 | 0.8591 | 112 | 11091 | 1207 | 1689 (108q) | 80 | 190.9 |
| vector_plus_synth | 0.3451 | 0.5778 | 0.6819 | 0.7407 | 0.7321 | 7577 | 11052 | 1176 | 3078 (108q) | 0 | 671.8 |

### Per-query-type slices (historical)

| Variant | temporal (R@5 / NDCG@5) | needle (R@5 / NDCG@5) | multi_hop (R@5 / NDCG@5) | single_hop (R@5 / NDCG@5) |
|---------|-------------------------|----------------------|--------------------------|---------------------------|
| small/clean_reranker | 0.6333 / 0.9342 (n=4) | — | 0.5273 / 0.8845 (n=11) | 0.6353 / 0.8346 (n=52) |
| small/vector_plus_synth | 0.7333 / 0.8868 (n=4) | — | 0.4727 / 0.6653 (n=11) | 0.6096 / 0.7774 (n=52) |

**Notes (historical):**

- **clean_reranker** — Cross-encoder reranking after coarse retrieval; strong NDCG and low p50 latency on this slice.
- **vector_plus_synth** — Index-time synthetic question embeddings (`vector_plus_synth`); higher synthetic-hit counts and different latency profile (see p50). Compare variants using the same corpus and `questions.json` only.

---

## Historical snapshot (v1.5 era)

> Generated: 2026-03-25
> Version: v1.5.0
> Stack: Qdrant + vLLM `BAAI/bge-base-en-v1.5` (768-dim) + xAI Grok (LLM)

## Overview

This report contains benchmark results across two tiers:

1. **Pipeline Ablation** — Retrieval quality measured across 7 pipeline configurations
2. **Micro-benchmarks** — Component-level performance (ops/sec, latency)

---

## Tier 1: Pipeline Ablation

Tested against a live Archivist stack with 100 queries across 6 query types against a 50-document, 155-chunk agent memory corpus. Each variant adds one more pipeline stage to show cumulative effect.

### Overall Results

| Pipeline Configuration | Recall@5 | Recall@10 | MRR | p50 Latency | p95 Latency | Tokens/Query |
|------------------------|----------|-----------|-----|-------------|-------------|--------------|
| Vector search only | 89.2% | 89.2% | 0.7352 | 794 ms | 943 ms | 4,462 |
| + BM25 keyword fusion | 89.2% | 89.2% | 0.7352 | 727 ms | 886 ms | 4,462 |
| + Knowledge graph augmentation | 89.2% | 89.2% | 0.7352 | 747 ms | 815 ms | 4,462 |
| + Temporal decay (365d halflife) | 87.7% | 87.7% | 0.7033 | 779 ms | 967 ms | 4,631 |
| + Hotness scoring | 87.7% | 87.7% | 0.7033 | 730 ms | 907 ms | 4,631 |
| + Reranking | 87.7% | 87.7% | 0.7033 | 766 ms | 908 ms | 4,631 |
| **Full pipeline** | **87.7%** | **87.7%** | **0.7033** | **823 ms** | **1,049 ms** | **4,631** |

### By Query Type (Full Pipeline)

| Query Type | Count | Recall | MRR | p50 Latency | What it tests |
|------------|-------|--------|-----|-------------|---------------|
| Single-hop | 50 | 99.0% | 0.785 | 805 ms | Direct factual lookup |
| Multi-hop | 10 | 57.5% | 0.716 | 811 ms | Cross-document reasoning |
| Temporal | 5 | 100.0% | 0.867 | 871 ms | Time-aware retrieval |
| Adversarial | 5 | 40.0% | 0.200 | 963 ms | Ambiguous/confusing queries |
| Agent-scoped | 5 | 83.3% | 0.667 | 987 ms | RBAC-filtered retrieval |
| Broad | 25 | 85.0% | 0.611 | 770 ms | Open-ended exploration |

### Key Observations

- **99% single-hop recall**: Archivist reliably finds direct factual answers across the corpus.
- **100% temporal recall**: Time-aware queries are handled perfectly, a key differentiator over plain vector search.
- **BM25 reduces latency**: Adding keyword fusion improves p50 from 794ms to 727ms (8.4% improvement) with no recall loss, because exact keyword matches resolve faster.
- **Temporal decay correctly penalizes stale data**: The 1.5% recall drop from temporal decay is expected -- the benchmark corpus is ~12 months old. With current-date memories, temporal decay *improves* effective recall by prioritizing fresh context.
- **Adversarial queries remain challenging**: 40% recall on deliberately ambiguous queries is an area for improvement, likely addressable with better LLM refinement.

### How to run

```bash
# Configure .env with Qdrant, embedding, and LLM endpoints (see “Remote stack” below).
# Module form (recommended):
python -m benchmarks.pipeline.evaluate --no-refine --output .benchmarks/pipeline.json

# With LLM refinement (slower, higher quality):
python -m benchmarks.pipeline.evaluate --output .benchmarks/pipeline.json

# Skip indexing if corpus already loaded:
python -m benchmarks.pipeline.evaluate --no-refine --skip-index --output .benchmarks/pipeline.json
```

**Progress and partial results:** With `--output`, the harness writes an atomic checkpoint beside the JSON (e.g. `.benchmarks/full_medium.run_state.json`) after each query. By default a **tqdm** progress bar on stderr updates each query (rolling R/MRR in the postfix). Use `--no-progress-bar` to fall back to milestone `PROGRESS` log lines every `--progress-pct` (default 10). Use `--checkpoint /path/state.json` explicitly, or `--no-checkpoint` to disable the file. HTTP client request spam is suppressed (`httpx` / `httpcore` at WARNING).

### Dual-track corpus (small / medium / large)

Regenerate questions (adds `needle`, `contradiction`, `tags`, `scales` fields):

```bash
python benchmarks/fixtures/generate_corpus.py --questions-only
```

Generate scaled corpora under `benchmarks/fixtures/corpus_<preset>/` (gitignored by default):

```bash
python benchmarks/fixtures/generate_corpus.py --preset small --corpus-only
python benchmarks/fixtures/generate_corpus.py --preset medium --corpus-only
python benchmarks/fixtures/generate_corpus.py --preset large --corpus-only
```

Run the harness against one preset (re-indexes Qdrant from that tree):

```bash
python -m benchmarks.pipeline.evaluate --memory-scale small --no-refine --output .benchmarks/small.json
```

Optional: after indexing, run one-shot LLM graph extraction over all agent markdown (`curator.extract_all_agent_memories`) so SQLite/KG is populated — **significant LLM cost**:

```bash
python -m benchmarks.pipeline.evaluate --memory-scale medium --run-curator --no-refine --output .benchmarks/medium.json
```

Scale sweep (same variants across all three presets; skips missing `corpus_*` dirs):

```bash
python -m benchmarks.pipeline.evaluate --scale-sweep --variants vector_only,full_pipeline --no-refine --output .benchmarks/sweep.json
```

JSON output includes `benchmark_meta.hypotheses` (H1–H3) for pre-registered claims; sweep runs add `by_scale` with per-preset summaries and traces.

### Remote stack (e.g. Docker on another host)

Point `.env` at the machine running Qdrant and embedding/LLM services (prefer Tailscale hostnames or `127.0.0.1` for local Qdrant; avoid publishing private LAN-only addresses):

```bash
QDRANT_URL=http://127.0.0.1:6333
EMBED_URL=http://<embed-host>:8000
LLM_URL=http://<primary-llm-host>:<port>
# Optional curator:
# CURATOR_LLM_URL=http://<curator-host>:<port>
```

Run the evaluate module from a checkout of this repo with that `.env`; `MEMORY_ROOT` is overridden per run to the chosen corpus directory.

### Limitations (what we prove vs. not)

- **Legacy row** in the table above reflects one corpus size and query mix; dual-track runs are required to compare small vs large memory honestly.
- **Needle** and **contradiction** slices depend on the generated markdown matching `questions.json` (regenerate both after changing `NEEDLE_SECRET` / contradiction facts in `generate_corpus.py`).
- **Curator** metrics are not automatic: without `--run-curator`, graph-augmented stages may see an empty KG (hypothesis H3).

---

## Tier 2: Micro-Benchmarks

Isolated component performance measured with `pytest-benchmark` on Python 3.14, Windows, no GPU. No external services required.

### Hot Cache (LRU/TTL)

| Operation | Mean | Min | Rounds |
|-----------|------|-----|--------|
| Cache hit | 0.001 ms | 0.001 ms | 129,871 |
| Cache miss | 0.001 ms | 0.001 ms | 38,023 |
| Cache put | 0.002 ms | 0.001 ms | 19,961 |
| LRU eviction | 0.002 ms | 0.001 ms | 169,492 |
| Invalidate namespace (100 agents x 50 entries) | 0.142 ms | 0.107 ms | 1,999 |

### Hybrid Search (BM25 + Vector Fusion)

| Operation | Mean | Min | Rounds |
|-----------|------|-----|--------|
| Vector-only merge | 0.000 ms | 0.000 ms | 196,080 |
| BM25-only merge | 0.004 ms | 0.003 ms | 116,280 |
| Fusion (10 results) | 0.011 ms | 0.008 ms | 34,723 |
| Fusion (50 results) | 0.055 ms | 0.037 ms | 13,423 |
| Fusion (200 results) | 0.233 ms | 0.152 ms | 3,746 |
| Fusion with overlap | 0.044 ms | 0.030 ms | 19,456 |

### FTS5 (BM25 Keyword Search)

| Operation | Mean | Min | Rounds |
|-----------|------|-----|--------|
| Search (100 docs) | 3.350 ms | 2.169 ms | 375 |
| Search (1,000 docs) | 3.841 ms | 2.752 ms | 313 |
| Search (5,000 docs) | 6.511 ms | 5.410 ms | 152 |
| Search (no namespace filter) | 3.920 ms | 2.793 ms | 230 |
| Upsert chunk | 7.251 ms | 5.568 ms | 138 |

### Knowledge Graph (SQLite)

| Operation | Mean | Min | Rounds |
|-----------|------|-----|--------|
| Upsert entity (new) | 6.809 ms | 5.202 ms | 136 |
| Upsert entity (existing) | 5.950 ms | 4.777 ms | 138 |
| Add relationship | 6.523 ms | 5.119 ms | 163 |
| Add fact | 7.229 ms | 5.226 ms | 137 |
| Search entities | 2.445 ms | 1.939 ms | 381 |

### Chunking & Tokenization

| Operation | Mean | Min | Rounds |
|-----------|------|-----|--------|
| Flat chunking (8 KB) | 0.010 ms | 0.008 ms | 14,578 |
| Flat chunking (40 KB) | 0.071 ms | 0.050 ms | 5,932 |
| Flat chunking (200 KB) | 0.338 ms | 0.266 ms | 1,964 |
| Hierarchical chunking (8 KB) | 0.043 ms | 0.035 ms | 7,605 |
| Hierarchical chunking (40 KB) | 0.236 ms | 0.189 ms | 3,322 |
| Hierarchical chunking (200 KB) | 1.342 ms | 1.022 ms | 855 |
| Token count (short) | 0.000 ms | 0.000 ms | 1,289 |
| Token count (medium) | 0.000 ms | 0.000 ms | 107,527 |
| Token count (large) | 0.000 ms | 0.000 ms | 74,627 |
| Message token count | 0.001 ms | 0.000 ms | 147,059 |

### Hotness Scoring, Temporal Decay & Metrics

| Operation | Mean | Min | Rounds |
|-----------|------|-----|--------|
| Hotness (single) | 0.000 ms | 0.000 ms | 120,483 |
| Hotness batch (100) | 0.029 ms | 0.021 ms | 27,701 |
| Hotness batch (1,000) | 0.277 ms | 0.210 ms | 4,004 |
| Hotness batch (10,000) | 2.609 ms | 2.148 ms | 444 |
| Apply hotness to results | 2.583 ms | 1.936 ms | 363 |
| Temporal decay (20 results) | 0.071 ms | 0.061 ms | 477 |
| Temporal decay (100 results) | 0.408 ms | 0.305 ms | 2,000 |
| Temporal decay (500 results) | 2.502 ms | 1.603 ms | 503 |
| Metrics render (100 series) | 0.050 ms | 0.040 ms | 13,459 |
| Metrics render (1,000 series) | 0.335 ms | 0.283 ms | 3,060 |
| Metrics render (10,000 series) | 3.204 ms | 2.701 ms | 332 |
| Metrics inc throughput | 0.000 ms | 0.000 ms | 200,000 |
| Metrics observe throughput | 0.001 ms | 0.000 ms | 181,818 |

### How to run

```bash
pip install pytest-benchmark
python -m pytest benchmarks/micro/ --benchmark-json=.benchmarks/micro.json -v
```

---

## Token Efficiency (Answer Finder)

Measures how much token waste the Answer Finder eliminates versus naive full-L2 retrieval.

### Latest measured run (v2.5.0 / 2026-07-27)

See [Latest showcase pack](#latest-showcase-pack-v250--2026-07-27) — adaptive **avg_savings_pct = 0.0** because the `benchmark_agent` store was empty. Evidence: `.benchmarks/token_efficiency_20260727.json`.

### Running the benchmark

```bash
# Default: all 49 queries, 8 000-token budget, 3 policies
PYTHONPATH=src python -m benchmarks.token_efficiency \
  --output .benchmarks/token_efficiency_$(date +%Y%m%d).json

# Quick smoke run (10 queries)
PYTHONPATH=src python -m benchmarks.token_efficiency --queries 10 --verbose
```

Output is a JSON file under `.benchmarks/` (gitignored) plus a printed comparison table. Example shape (illustrative — **not** this initiative’s actuals):

```text
Policy        Queries  Avg Savings%  Min%   Max%   Avg Tokens  Avg Naive   Avg ms
adaptive         49       …          …      …         …          …          …
```

**Regression target (when the store is populated):** `adaptive` ≥60% average savings. The 2026-07-27 empty-store run is **not** evidence that target was met.

### How savings are calculated

`savings_pct = 1 - (tokens_returned / tokens_naive)` where `tokens_naive` is the token count if the full L2 result set (no budget cap) were returned verbatim. Empty retrieval → 0% savings.

---

## Feature Overview

| Feature | Archivist |
|---------|-----------|
| Hybrid search (vector + BM25) | Yes (0.7/0.3 fusion) |
| Temporal knowledge graph | Yes (SQLite + FTS5 / Postgres GIN) |
| Active curation (background) | Yes (LLM dedup, tip consolidation) |
| Multi-agent RBAC | Yes (namespace ACLs) |
| Cross-encoder reranking | Yes (BAAI/bge-reranker-v2-m3) |
| Hotness scoring | Yes (freq × recency × importance) |
| Conflict detection | Yes (vector + LLM adjudication) |
| Hierarchical tiered memory | Yes (L0/L1/L2/ephemeral, v2.3) |
| Token-budgeted context packing | Yes (adaptive/l0_first/l2_first, v2.3) |
| Multi-agent handoff protocol | Yes (HandoffPacket, v2.3) |
| Token savings observability | Yes (retrieval_logs + dashboard, v2.3) |
| Self-hosted / Apache 2.0 | Yes |

---

## Visual Dashboard

Open [`docs/benchmark-dashboard.html`](benchmark-dashboard.html) in a browser for an interactive visual dashboard with:

- Pipeline ablation bar chart (recall by stage)
- Recall and MRR by query type
- Latency vs quality tradeoff scatter plot
- Full results table

---

## Reproducing Results

```bash
# Prerequisites
docker compose up -d qdrant
pip install pytest-benchmark

# Configure .env with your Qdrant, embedding, and LLM endpoints
cp .env.example .env
# Edit .env with your endpoints

# Micro-benchmarks (no external services needed)
python -m pytest benchmarks/micro/ --benchmark-json=.benchmarks/micro.json -v

# Pipeline ablation (requires Qdrant + embedding API)
python benchmarks/pipeline/evaluate.py --no-refine --output .benchmarks/pipeline.json

# Generate visual dashboard
# Open docs/benchmark-dashboard.html in a browser
```
