# Archivist Benchmark Results

> Generated: 2026-03-25 19:10 UTC  
> Version: v1.5.0

## Overview

This report contains benchmark results across three tiers:

1. **Micro-benchmarks** — Component-level performance (ops/sec, latency)
2. **Pipeline Ablation** — Retrieval quality with stages toggled on/off
3. **Academic Benchmarks** — LoCoMo (long conversation memory) and HaluMem (hallucination detection)

---

## Tier 1: Micro-Benchmarks

Isolated component performance measured with `pytest-benchmark`. No external services required.

_No micro-benchmark data available. Run:_
```
python -m pytest benchmarks/micro/ --benchmark-json=.benchmarks/micro.json
```


### How to run

```bash
python -m pytest benchmarks/micro/ --benchmark-json=.benchmarks/micro.json
```

---

## Tier 2: Pipeline Ablation

Each row adds one pipeline stage to measure its marginal contribution to retrieval quality.

_No pipeline ablation data available. Run:_
```
python -m benchmarks.pipeline.evaluate --output pipeline_results.json
```


### How to run

```bash
# Index corpus + run all variants
python -m benchmarks.pipeline.evaluate --output pipeline_results.json

# Faster: skip LLM refinement
python -m benchmarks.pipeline.evaluate --no-refine --output pipeline_results.json
```

---

## Tier 3: Academic Benchmarks

### LoCoMo (Long Conversation Memory)

Tests memory retention and reasoning over 300-600 turn dialogues across 5 QA types.

_No LoCoMo data available. Run:_
```
python -m benchmarks.academic.locomo.adapter --data-dir data/locomo
```


### HaluMem (Hallucination in Memory)

Tests whether the memory system introduces hallucinated information during extraction, updating, or question answering.

_No HaluMem data available. Run:_
```
python -m benchmarks.academic.halumem.adapter --data-dir data/halumem
```


---

## Competitive Positioning

| System | LoCoMo QA | HaluMem Composite | Architecture |
|--------|-----------|-------------------|-------------|
| **Archivist** | **TBD** | **TBD** | 10-stage RLM pipeline, hybrid search, temporal KG, active curation |
| Zep (Graphiti) | ~85% | — | Temporal knowledge graph |
| Letta/MemGPT | ~83.2% | — | Self-managed 3-tier agent memory |
| Mem0 | ~58-66% | — | Vector similarity + knowledge graph (Pro) |
| Memobase | — | See HaluMem paper | — |
| MemOS | — | See HaluMem paper | — |
| Supermemory | — | See HaluMem paper | — |

### Archivist Differentiators

| Feature | Archivist | Mem0 | Zep | Letta |
|---------|-----------|------|-----|-------|
| Hybrid search (vector + BM25) | Yes (0.7/0.3 fusion) | Vector only (free) | Graph-based | Vector |
| Temporal knowledge graph | Yes (SQLite + FTS5) | Pro only ($249/mo) | Yes (Graphiti) | No |
| Active curation (background) | Yes (LLM dedup, tip consolidation) | No | No | Self-managed |
| Multi-agent RBAC | Yes (namespace ACLs) | No | No | Per-agent isolation |
| Cross-encoder reranking | Yes (BAAI/bge-reranker-v2-m3) | No | No | No |
| Hotness scoring | Yes (freq x recency) | No | Temporal decay | No |
| Conflict detection | Yes (vector + LLM adjudication) | No | Temporal versioning | No |
| Self-hosted / Apache 2.0 | Yes | Open core | Yes | Yes |

---

## Reproducing Results

```bash
# Prerequisites
docker compose up -d qdrant
pip install pytest-benchmark rouge-score nltk

# Tier 1: Micro-benchmarks (no external services)
python -m pytest benchmarks/micro/ --benchmark-json=.benchmarks/micro.json

# Tier 2: Pipeline ablation (requires Qdrant + embedding API)
python -m benchmarks.pipeline.evaluate --output pipeline_results.json

# Tier 3: Academic benchmarks (requires Qdrant + LLM + embedding API)
git clone https://github.com/snap-research/locomo.git data/locomo
python -m benchmarks.academic.locomo.adapter --data-dir data/locomo --output locomo_results.json

git clone https://github.com/MemTensor/HaluMem.git data/halumem
python -m benchmarks.academic.halumem.adapter --data-dir data/halumem --output halumem_results.json

# Generate report
python -m benchmarks.report \
    --micro-json .benchmarks/micro.json \
    --pipeline-json pipeline_results.json \
    --locomo-json locomo_results.json \
    --halumem-json halumem_results.json \
    --output docs/BENCHMARKS.md
```
