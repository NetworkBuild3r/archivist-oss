# HaluMem Benchmark Adapter

Evaluates Archivist against the [HaluMem](https://github.com/MemTensor/HaluMem) (Hallucination in Memory) benchmark.

## What HaluMem Tests

HaluMem is the first operation-level hallucination evaluation benchmark for memory systems. It decomposes memory into three tasks:

| Task | What it tests | Hallucination types detected |
|------|---------------|------------------------------|
| **Extraction** | Can the system accurately store facts from conversations? | Fabrication, omission |
| **Updating** | When new facts arrive, does the system correctly overwrite old ones? | Conflicts, stale memories |
| **QA** | Can the system answer questions without hallucinating? | Fabrication, errors, conflicts, omissions |

### Datasets

- **HaluMem-Medium**: ~160K tokens context, ~1.5K dialogue turns per user
- **HaluMem-Long**: ~1M tokens context, ~2.6K dialogue turns per user
- ~15,000 memory points and ~3,500 questions across 20 users

## Setup

```bash
# 1. Get the dataset (either method)
git clone https://github.com/MemTensor/HaluMem.git data/halumem
# OR
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('IAAR-Shanghai/HaluMem')"

# 2. Ensure Qdrant is running
docker compose up -d qdrant

# 3. Run the benchmark
python -m benchmarks.academic.halumem.adapter --data-dir data/halumem --output halumem_results.json

# Quick test with limited users
python -m benchmarks.academic.halumem.adapter --data-dir data/halumem --limit 3
```

## How It Works

1. Loads user conversation histories from HaluMem dataset
2. For each user:
   a. Ingests conversation history as markdown files via Archivist's indexer
   b. **Extraction eval**: Checks if ground-truth facts are present in stored chunks
   c. **Update eval**: Writes new facts, re-indexes, verifies retrieval reflects updates
   d. **QA eval**: Calls `recursive_retrieve` and checks for hallucination

## Scoring

The composite score weights the three tasks:
- Extraction F1: 30%
- Update correctness: 30%
- QA non-hallucination rate: 40%

## What Archivist Features Are Exercised

- Conflict detection on `archivist_store` (catches contradictions)
- Curator deduplication (LLM-adjudicated merge/skip decisions)
- Knowledge graph contradiction detection
- RBAC namespace isolation (no cross-user leakage)
- Memory versioning and supersession
- BM25 + vector hybrid search for update verification

## Notes

Archivist's active curation, conflict detection, and knowledge graph should provide advantages on the update and QA tasks where stale/contradictory memories cause hallucination.
