# Benchmarks

## In-repo pipeline (regression / product-shaped)

```bash
cp .env.example .env   # LLM_URL, EMBED_URL, VECTOR_DIM, QDRANT_URL
docker compose up -d qdrant
env REVERSE_HYDE_ENABLED=false TIERED_CONTEXT_ENABLED=false QUERY_EXPANSION_ENABLED=false \
  python -m benchmarks.pipeline.evaluate \
  --memory-scale small --variants vector_only --no-refine \
  --output .benchmarks/pipeline_small.json --print-slices
```

## Thin industry-reference runs

These are intentionally small defaults; increase limits for serious numbers.

### A) One-shot script (host Python + Qdrant)

Requires: `pip install -r requirements.txt` and `pip install -r requirements-benchmark.txt` (BEIR).

```bash
docker compose up -d qdrant
bash benchmarks/scripts/run_thin_reference.sh
# Optional: LIMIT_LM=50 LIMIT_BEIR=200 bash benchmarks/scripts/run_thin_reference.sh
```

Outputs:

- `.benchmarks/longmemeval_thin.json` — **LongMemEval** via Archivist (index + `recursive_retrieve` + LLM judge).
- `.benchmarks/beir_nfcorpus_thin.json` — **BEIR NFCorpus** dense bi-encoder baseline (standard nDCG@k; **not** the full RLM pipeline).

### B) LongMemEval only (host)

```bash
mkdir -p data/longmemeval .benchmarks
wget -q -O data/longmemeval/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

python -m benchmarks.academic.longmemeval.adapter \
  --data-file data/longmemeval/longmemeval_s_cleaned.json \
  --limit 20 --no-refine \
  --output .benchmarks/longmemeval_thin.json
```

### C) LongMemEval only (Docker benchmark service)

Uses `Dockerfile.benchmark` + `qdrant` on `archivist-net` (LLM/embed typically `host.docker.internal`).

```bash
bash benchmarks/scripts/run_longmemeval_docker.sh --limit 10 --no-refine
```

### D) BEIR only (host, no Qdrant)

```bash
pip install -r requirements-benchmark.txt
python -m benchmarks.academic.beir_thin \
  --dataset nfcorpus \
  --limit-queries 50 \
  --output .benchmarks/beir_nfcorpus_thin.json
```

Use `--model` to match your production embedding model id when comparing families.
