# LongMemEval Benchmark Adapter

Evaluates Archivist against **LongMemEval** (ICLR 2025) — a 500-question benchmark testing five core long-term memory abilities:

| Category | What it tests | Question types |
|----------|--------------|----------------|
| Information Extraction | Recall specific facts from history | single-session-user, single-session-assistant, single-session-preference |
| Multi-Session Reasoning | Synthesize information across sessions | multi-session |
| Knowledge Updates | Recognize changed information over time | knowledge-update |
| Temporal Reasoning | Understand temporal aspects | temporal-reasoning |
| Abstention | Refrain from answering unknown questions | *_abs suffixed questions |

## Evaluation Protocol

This adapter uses the **official LongMemEval evaluation protocol** from the paper:

- **QA Accuracy** via **LLM-as-judge** with task-specific prompts (from `src/evaluation/evaluate_qa.py` in the official repo). Each question type has its own judge prompt to handle nuances like off-by-one tolerance for temporal questions, update precedence for knowledge-update, and preference rubrics.
- **Retrieval metrics**: **Recall@k** and **NDCG@k** (k=5, 10) computed against ground-truth `answer_session_ids`, matching `src/retrieval/eval_utils.py`.
- Results are broken down by the 6 official question types for direct comparison with published numbers.

The official protocol uses GPT-4o as judge; we use whatever `LLM_MODEL` is configured (e.g., Qwen). The judge prompts are identical to the paper's.

## Setup

### 1. Download dataset

```bash
mkdir -p data/longmemeval && cd data/longmemeval
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
cd ../..
```

### 2. Run

```bash
# Full run (500 questions)
python -m benchmarks.academic.longmemeval.adapter \
    --data-file data/longmemeval/longmemeval_s_cleaned.json \
    --output .benchmarks/longmemeval_results.json

# Quick test (10 questions, no refinement)
python -m benchmarks.academic.longmemeval.adapter \
    --data-file data/longmemeval/longmemeval_s_cleaned.json \
    --limit 10 --no-refine \
    --output .benchmarks/longmemeval_quick.json

# Ablation across pipeline variants
python -m benchmarks.academic.longmemeval.adapter \
    --data-file data/longmemeval/longmemeval_s_cleaned.json \
    --ablation --output .benchmarks/longmemeval_ablation.json

# Docker (uses compose benchmark profile + Qdrant):
bash benchmarks/scripts/run_academic_benchmarks_docker.sh --limit 50
```

## Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| **QA Accuracy** | Official (LLM-as-judge) | Binary correct/incorrect per question, task-specific prompts |
| **Task-Averaged Accuracy** | Official | Mean accuracy across the 6 question types (not micro-averaged) |
| **Abstention Accuracy** | Official | Accuracy on unanswerable questions (*_abs) |
| **Recall@k** | Official (eval_utils.py) | Fraction of evidence sessions retrieved in top-k |
| **NDCG@k** | Official (eval_utils.py) | Normalized DCG over evidence session rankings |

## Reference

```bibtex
@inproceedings{wu2025longmemeval,
    title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
    author={Wu, Di and Wang, Hongwei and Yu, Wenhao and Zhang, Yuwei and Chang, Kai-Wei and Yu, Dong},
    booktitle={ICLR},
    year={2025},
}
```
