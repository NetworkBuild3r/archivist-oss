# Archivist Benchmark Results

> Generated: 2026-03-25 20:03 UTC  
> Version: v1.5.0

## Overview

This report contains benchmark results across three tiers:

1. **Micro-benchmarks** — Component-level performance (ops/sec, latency)
2. **Pipeline Ablation** — Retrieval quality with stages toggled on/off
3. **Academic Benchmarks** — LoCoMo (long conversation memory) and HaluMem (hallucination detection)

---

## Tier 1: Micro-Benchmarks

Isolated component performance measured with `pytest-benchmark`. No external services required.

| Benchmark | Min (ms) | Mean (ms) | Max (ms) | StdDev | Rounds |
|-----------|----------|-----------|---------|--------|--------|
| test_add_fact_latency | 5.226 | 7.229 | 16.088 | 1.576 | 137 |
| test_add_relationship_latency | 5.119 | 6.523 | 12.312 | 0.996 | 163 |
| test_apply_hotness_to_results | 1.936 | 2.583 | 6.225 | 0.586 | 363 |
| test_cache_hit_latency | 0.001 | 0.001 | 1.478 | 0.005 | 129871 |
| test_cache_lru_eviction | 0.001 | 0.002 | 0.331 | 0.002 | 169492 |
| test_cache_miss_latency | 0.001 | 0.001 | 0.063 | 0.001 | 38023 |
| test_cache_put_latency | 0.001 | 0.002 | 2.814 | 0.021 | 19961 |
| test_compute_hotness_batch[10000] | 2.148 | 2.609 | 8.405 | 0.425 | 444 |
| test_compute_hotness_batch[1000] | 0.210 | 0.277 | 1.702 | 0.085 | 4004 |
| test_compute_hotness_batch[100] | 0.021 | 0.029 | 0.446 | 0.014 | 27701 |
| test_compute_hotness_single | 0.000 | 0.000 | 0.345 | 0.002 | 120483 |
| test_count_message_tokens | 0.000 | 0.001 | 0.294 | 0.001 | 147059 |
| test_count_tokens_large | 0.000 | 0.000 | 0.014 | 0.000 | 74627 |
| test_count_tokens_medium | 0.000 | 0.000 | 0.039 | 0.000 | 107527 |
| test_count_tokens_short | 0.000 | 0.000 | 0.002 | 0.000 | 1289 |
| test_flat_chunking[200] | 0.266 | 0.338 | 1.206 | 0.093 | 1964 |
| test_flat_chunking[40] | 0.050 | 0.071 | 1.054 | 0.031 | 5932 |
| test_flat_chunking[8] | 0.008 | 0.010 | 0.260 | 0.005 | 14578 |
| test_fts5_search_latency[1000] | 2.752 | 3.841 | 7.177 | 0.788 | 313 |
| test_fts5_search_latency[100] | 2.169 | 3.350 | 6.569 | 0.792 | 375 |
| test_fts5_search_latency[5000] | 5.410 | 6.511 | 13.182 | 1.093 | 152 |
| test_fts5_search_no_namespace_filter | 2.793 | 3.920 | 33.355 | 2.084 | 230 |
| test_fts5_upsert_chunk_latency | 5.568 | 7.251 | 10.766 | 1.009 | 138 |
| test_hierarchical_chunking[200] | 1.022 | 1.342 | 6.099 | 0.383 | 855 |
| test_hierarchical_chunking[40] | 0.189 | 0.236 | 0.851 | 0.071 | 3322 |
| test_hierarchical_chunking[8] | 0.035 | 0.043 | 0.473 | 0.022 | 7605 |
| test_invalidate_namespace_100_agents | 0.107 | 0.142 | 0.829 | 0.052 | 1999 |
| test_merge_bm25_only | 0.003 | 0.004 | 2.991 | 0.010 | 116280 |
| test_merge_fusion_latency[10] | 0.008 | 0.011 | 0.770 | 0.009 | 34723 |
| test_merge_fusion_latency[200] | 0.152 | 0.233 | 4.544 | 0.139 | 3746 |
| test_merge_fusion_latency[50] | 0.037 | 0.055 | 2.415 | 0.038 | 13423 |
| test_merge_fusion_with_overlap | 0.030 | 0.044 | 1.165 | 0.019 | 19456 |
| test_merge_vector_only | 0.000 | 0.000 | 0.003 | 0.000 | 196080 |
| test_metrics_inc_throughput | 0.000 | 0.000 | 0.105 | 0.000 | 200000 |
| test_metrics_observe_throughput | 0.000 | 0.001 | 0.516 | 0.002 | 181818 |
| test_metrics_render_cold | 0.001 | 0.001 | 23.600 | 0.061 | 153847 |
| test_metrics_render_latency[10000] | 2.701 | 3.204 | 5.487 | 0.415 | 332 |
| test_metrics_render_latency[1000] | 0.283 | 0.335 | 0.919 | 0.062 | 3060 |
| test_metrics_render_latency[100] | 0.040 | 0.050 | 4.052 | 0.038 | 13459 |
| test_search_entities_latency | 1.939 | 2.445 | 4.815 | 0.412 | 381 |
| test_temporal_decay_halflife_sweep[30] | 0.309 | 0.453 | 1.445 | 0.126 | 1746 |
| test_temporal_decay_halflife_sweep[365] | 0.307 | 0.457 | 5.384 | 0.212 | 2497 |
| test_temporal_decay_halflife_sweep[7] | 0.306 | 0.432 | 2.086 | 0.135 | 1848 |
| test_temporal_decay_halflife_sweep[90] | 0.314 | 0.454 | 1.866 | 0.117 | 1735 |
| test_temporal_decay_latency[100] | 0.305 | 0.408 | 2.293 | 0.128 | 2000 |
| test_temporal_decay_latency[20] | 0.061 | 0.071 | 0.260 | 0.017 | 477 |
| test_temporal_decay_latency[500] | 1.603 | 2.502 | 30.662 | 1.493 | 503 |
| test_temporal_decay_preserves_order_for_same_date | 0.030 | 0.041 | 0.596 | 0.019 | 12920 |
| test_upsert_entity_existing | 4.777 | 5.950 | 9.307 | 0.637 | 138 |
| test_upsert_entity_latency | 5.202 | 6.809 | 10.180 | 0.746 | 136 |

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
