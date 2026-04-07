# LongMemEval Benchmark Adapter

Evaluates Archivist against **LongMemEval** (ICLR 2025) — a 500-question benchmark testing five core long-term memory abilities:

| Category | What it tests | Question types |
|----------|--------------|----------------|
| Information Extraction | Recall specific facts from history | single-session-user, single-session-assistant, single-session-preference |
| Multi-Session Reasoning | Synthesize information across sessions | multi-session |
| Knowledge Updates | Recognize changed information over time | knowledge-update |
| Temporal Reasoning | Understand temporal aspects | temporal-reasoning |
| Abstention | Refrain from answering unknown questions | *_abs suffixed questions |

## Setup

### 1. Download dataset

```bash
mkdir -p data/longmemeval && cd data/longmemeval
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
# Optional (large, ~500 sessions per question):
# wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
cd ../..
```

### 2. Ensure Qdrant is running

```bash
docker compose up -d qdrant
```

### 3. Run

```bash
# Full run (500 questions, with LLM refinement)
python -m benchmarks.academic.longmemeval.adapter \
    --data-file data/longmemeval/longmemeval_s_cleaned.json \
    --output .benchmarks/longmemeval_results.json

# Quick test (10 questions, no refinement)
python -m benchmarks.academic.longmemeval.adapter \
    --data-file data/longmemeval/longmemeval_s_cleaned.json \
    --limit 10 --no-refine \
    --output .benchmarks/longmemeval_quick.json

# With curator (KG entity extraction for entity-anchored retrieval)
python -m benchmarks.academic.longmemeval.adapter \
    --data-file data/longmemeval/longmemeval_s_cleaned.json \
    --run-curator \
    --output .benchmarks/longmemeval_with_kg.json
```

## Metrics

| Metric | Description |
|--------|-------------|
| **keyword_recall** | Fraction of ground-truth words found in answer + sources (R@k proxy) |
| **session_recall** | Fraction of evidence sessions retrieved in top-k results |
| **f1** | Token-level F1 between answer and ground truth |

## Reference

```bibtex
@article{wu2024longmemeval,
    title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
    author={Wu, Di and Wang, Hongwei and Yu, Wenhao and Zhang, Yuwei and Chang, Kai-Wei and Yu, Dong},
    year={2024},
    journal={arXiv preprint arXiv:2410.10813},
}
```
