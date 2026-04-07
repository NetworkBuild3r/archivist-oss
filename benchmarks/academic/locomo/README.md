# LoCoMo Benchmark Adapter

Evaluates Archivist against the [LoCoMo](https://github.com/snap-research/locomo) (Long Conversation Memory) benchmark from ACL 2024.

## What LoCoMo Tests

LoCoMo contains multi-session dialogues (~300-600 turns, ~16K tokens) with 5 QA types:

| QA Type | What it tests |
|---------|---------------|
| Single-hop | Direct fact recall from memory |
| Multi-hop | Reasoning across multiple stored facts |
| Temporal | "When did X happen?" — requires date awareness |
| Commonsense | Inference combining memory + world knowledge |
| Adversarial | Questions about things NOT in memory |

## Setup

```bash
# 1. Clone the dataset
git clone https://github.com/snap-research/locomo.git data/locomo

# 2. Install evaluation dependencies
pip install rouge-score nltk

# 3. Ensure Qdrant is running
docker compose up -d qdrant

# 4. Run the benchmark
python -m benchmarks.academic.locomo.adapter --data-dir data/locomo --output locomo_results.json

# Quick test with limited dialogues
python -m benchmarks.academic.locomo.adapter --data-dir data/locomo --limit 5 --no-refine
```

## How It Works

1. Loads LoCoMo conversation sessions
2. Converts each session to markdown files in a temporary `MEMORY_ROOT`
3. Indexes all files through Archivist's full pipeline (chunking + embedding + FTS5)
4. Triggers the curator cycle for entity extraction
5. For each benchmark question, calls `recursive_retrieve` with `refine=true`
6. Compares the synthesized answer against ground truth using F1, BLEU-1, and ROUGE-L

## Archivist Scores

| Metric | Score |
|--------|-------|
| **Archivist** | Run the adapter to measure |

## What Archivist Features Are Exercised

- Temporal knowledge graph (entity/relationship extraction)
- Graph-augmented retrieval with multi-hop traversal
- Temporal decay scoring
- Tiered context (L0/L1/L2 summaries)
- Hybrid search (vector + BM25 keyword fusion)
- LLM-based refinement and synthesis
