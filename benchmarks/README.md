# Benchmarks

This directory holds the **in-repo pipeline** harness and thin reference scripts (LongMemEval, BEIR). Canonical result tables and competitive discussion live in **[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md)**. For pytest and `tests/qa/`, see **[`docs/QA.md`](../docs/QA.md)**.

## Using your local setup

Work from **your repo clone** (example: `cd /opt/appdata/archivist-oss`). Do **not** copy the placeholder path from older docs.

- **Config:** If you already have a working `.env`, keep it. Only create one when missing:
  `cp -n .env.example .env` (`-n` = do not overwrite an existing file).
- **Python:** Use whatever you normally use (conda base, `venv`, etc.). Core runtime: `python -m pip install -r requirements.txt`. Benchmark-only deps (BEIR, large wheels): **add** `pip install -r requirements-benchmark.txt` when you need them — they are **not** part of the default image/install.
- **Qdrant:** Start the stack you already use, e.g. `docker compose up -d qdrant`, and ensure `QDRANT_URL` in `.env` matches (e.g. `http://127.0.0.1:6333` when Qdrant is on the host).
- **LongMemEval only:** `SKIP_BEIR=1 bash benchmarks/scripts/run_thin_reference.sh` skips BEIR and extra packages.

## In-repo pipeline (regression / product-shaped)

```bash
cp -n .env.example .env   # only if .env missing; set LLM_URL, EMBED_URL, VECTOR_DIM, QDRANT_URL
docker compose up -d qdrant
env REVERSE_HYDE_ENABLED=false TIERED_CONTEXT_ENABLED=false QUERY_EXPANSION_ENABLED=false \
  python -m benchmarks.pipeline.evaluate \
  --memory-scale small --variants vector_only --no-refine \
  --output .benchmarks/pipeline_small.json --print-slices
```

## Thin industry-reference runs

These are intentionally small defaults; increase limits for serious numbers.

### A) One-shot script (host Python + Qdrant)

From the repo root, with `.env` loaded (the script sources `.env` automatically):

```bash
docker compose up -d qdrant   # or use your existing Qdrant; match QDRANT_URL in .env
python -m pip install -r requirements.txt
python -m pip install -r requirements-benchmark.txt   # omit if SKIP_BEIR=1

bash benchmarks/scripts/run_thin_reference.sh
# Optional: LIMIT_LM=50 LIMIT_BEIR=200 bash benchmarks/scripts/run_thin_reference.sh
# LongMemEval only (no BEIR): SKIP_BEIR=1 bash benchmarks/scripts/run_thin_reference.sh
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

## Speed (LongMemEval)

Indexing was slow because **reverse HyDE** and related options trigger **many LLM calls per chunk**. `run_thin_reference.sh` sets **`BENCHMARK_FAST=1` by default** (after sourcing `.env`), which exports:

- `REVERSE_HYDE_ENABLED=false`
- `SYNTHETIC_QUESTIONS_ENABLED=false`
- `QUERY_EXPANSION_ENABLED=false`
- `TIERED_CONTEXT_ENABLED=false`
- `CONTEXTUAL_AUGMENTATION_ENABLED=false`

For a **full-fidelity** run (slow), use:

```bash
BENCHMARK_FAST=0 bash benchmarks/scripts/run_thin_reference.sh
```

**More knobs:**

- **`LIMIT_LM=5`** — fewer questions while you tune hardware.
- **Faster judge (separate host)** — keep `LLM_URL` for retrieval/synthesis on your workstation; set **`BENCHMARK_JUDGE_LLM_URL`**, **`BENCHMARK_JUDGE_LLM_MODEL`**, and optionally **`BENCHMARK_JUDGE_LLM_API_KEY`** to a fast Ollama/vLLM on the LAN (e.g. DGX). URLs are API roots **without** `/v1`. Qwen3-class judge IDs get `/no_think` automatically in `llm.py`.
- **GPU / batching** — keep **embeddings** on a fast endpoint; local Ollama on CPU for everything will bound wall time.
- **`SKIP_BEIR=1`** — skip BEIR if you only care about LongMemEval right now.
