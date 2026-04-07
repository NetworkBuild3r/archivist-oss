"""Benchmark report aggregator — collects results from all three tiers and
generates a unified markdown report for docs/BENCHMARKS.md.

Usage:
    # After running benchmarks, aggregate:
    python -m benchmarks.report \
        --micro-json .benchmarks/micro.json \
        --pipeline-json pipeline_results.json \
        --locomo-json locomo_results.json \
        --halumem-json halumem_results.json \
        --output docs/BENCHMARKS.md

    # From pytest-benchmark output:
    python -m pytest benchmarks/micro/ --benchmark-json=.benchmarks/micro.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def _load_json(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _format_micro_benchmarks(data: dict) -> str:
    """Format pytest-benchmark JSON output into markdown table."""
    if not data or "benchmarks" not in data:
        return "_No micro-benchmark data available. Run:_\n```\npython -m pytest benchmarks/micro/ --benchmark-json=.benchmarks/micro.json\n```\n"

    lines = [
        "| Benchmark | Min (ms) | Mean (ms) | Max (ms) | StdDev | Rounds |",
        "|-----------|----------|-----------|---------|--------|--------|",
    ]

    for b in sorted(data["benchmarks"], key=lambda x: x["group"] + x["name"] if x.get("group") else x["name"]):
        name = b["name"]
        stats = b["stats"]
        lines.append(
            f"| {name} | {stats['min']*1000:.3f} | {stats['mean']*1000:.3f} | "
            f"{stats['max']*1000:.3f} | {stats['stddev']*1000:.3f} | {stats['rounds']} |"
        )

    return "\n".join(lines)


def _format_pipeline_ablation(data: dict) -> str:
    """Format pipeline ablation results into markdown table."""
    if not data or "summaries" not in data:
        return "_No pipeline ablation data available. Run:_\n```\npython -m benchmarks.pipeline.evaluate --output pipeline_results.json\n```\n"

    lines = [
        "| Variant | Recall@5 | Recall@10 | MRR | p50 Latency | p95 Latency | Tokens/Query |",
        "|---------|----------|-----------|-----|-------------|-------------|--------------|",
    ]

    for s in data["summaries"]:
        lines.append(
            f"| {s['variant']} | {s['recall_at_5']:.4f} | {s['recall_at_10']:.4f} | "
            f"{s['mrr']:.4f} | {s['latency_p50_ms']:.0f}ms | {s['latency_p95_ms']:.0f}ms | "
            f"{s['avg_tokens_per_query']:.0f} |"
        )

    if data["summaries"]:
        best = max(data["summaries"], key=lambda s: s["recall_at_5"])
        baseline = min(data["summaries"], key=lambda s: s["recall_at_5"])
        if best["variant"] != baseline["variant"]:
            delta = best["recall_at_5"] - baseline["recall_at_5"]
            lines.append("")
            lines.append(
                f"> **{best['variant']}** achieves +{delta:.1%} Recall@5 improvement "
                f"over **{baseline['variant']}** baseline."
            )

    return "\n".join(lines)


def _format_locomo(data: dict) -> str:
    """Format LoCoMo benchmark results."""
    if not data or "summary" not in data:
        return "_No LoCoMo data available. Run:_\n```\npython -m benchmarks.academic.locomo.adapter --data-dir data/locomo\n```\n"

    s = data["summary"]
    lines = [
        f"**Dialogues evaluated:** {s['dialogues']}  ",
        f"**Questions evaluated:** {s['evaluated_questions']}  ",
        f"**Total chunks indexed:** {s['total_chunks']}",
        "",
        "| Metric | Archivist |",
        "|--------|-----------|",
        f"| F1 | **{s['overall_f1']:.4f}** |",
        f"| BLEU-1 | {s['overall_bleu']:.4f} |",
        f"| ROUGE-L | {s['overall_rouge_l']:.4f} |",
    ]

    if s.get("by_category"):
        lines.extend([
            "",
            "**By QA category:**",
            "",
            "| Category | F1 | BLEU | ROUGE-L | Count |",
            "|----------|----|------|---------|-------|",
        ])
        for cat, vals in s["by_category"].items():
            lines.append(
                f"| {cat} | {vals['f1']:.4f} | {vals['bleu']:.4f} | "
                f"{vals['rouge_l']:.4f} | {vals['count']} |"
            )

    return "\n".join(lines)


def _format_halumem(data: dict) -> str:
    """Format HaluMem benchmark results."""
    if not data or "summary" not in data:
        return "_No HaluMem data available. Run:_\n```\npython -m benchmarks.academic.halumem.adapter --data-dir data/halumem\n```\n"

    s = data["summary"]
    lines = [
        f"**Users evaluated:** {s['users_evaluated']}  ",
        f"**Composite score:** {s['composite_score']:.4f}",
        "",
        "| Task | Metric | Score |",
        "|------|--------|-------|",
        f"| Extraction | Recall | {s['extraction']['avg_recall']:.4f} |",
        f"| Extraction | F1 | {s['extraction']['avg_f1']:.4f} |",
        f"| Updating | Correctness | {s['updating']['correctness_rate']:.4f} |",
        f"| Updating | New fact recall | {s['updating']['new_fact_recall']:.4f} |",
        f"| QA | Hallucination rate | {s['qa']['hallucination_rate']:.4f} |",
        f"| QA | F1 | {s['qa']['avg_f1']:.4f} |",
    ]

    if s["qa"].get("hallucination_types"):
        lines.extend([
            "",
            "**Hallucination breakdown:**",
            "",
        ])
        for h_type, count in s["qa"]["hallucination_types"].items():
            lines.append(f"- {h_type}: {count}")

    return "\n".join(lines)


def generate_report(
    micro_data: dict | None = None,
    pipeline_data: dict | None = None,
    locomo_data: dict | None = None,
    halumem_data: dict | None = None,
) -> str:
    """Generate the full benchmark report as markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report = f"""# Archivist Benchmark Results

> Generated: {now}  
> Version: v1.5.0

## Overview

This report contains benchmark results across three tiers:

1. **Micro-benchmarks** — Component-level performance (ops/sec, latency)
2. **Pipeline Ablation** — Retrieval quality with stages toggled on/off
3. **Academic Benchmarks** — LoCoMo (long conversation memory) and HaluMem (hallucination detection)

---

## Tier 1: Micro-Benchmarks

Isolated component performance measured with `pytest-benchmark`. No external services required.

{_format_micro_benchmarks(micro_data)}

### How to run

```bash
python -m pytest benchmarks/micro/ --benchmark-json=.benchmarks/micro.json
```

---

## Tier 2: Pipeline Ablation

Each row adds one pipeline stage to measure its marginal contribution to retrieval quality.

{_format_pipeline_ablation(pipeline_data)}

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

{_format_locomo(locomo_data)}

### HaluMem (Hallucination in Memory)

Tests whether the memory system introduces hallucinated information during extraction, updating, or question answering.

{_format_halumem(halumem_data)}

---

## Archivist Features

| Feature | Archivist |
|---------|-----------|
| Hybrid search (vector + BM25) | Yes (0.7/0.3 fusion) |
| Temporal knowledge graph | Yes (SQLite + FTS5) |
| Active curation (background) | Yes (LLM dedup, tip consolidation) |
| Multi-agent RBAC | Yes (namespace ACLs) |
| Cross-encoder reranking | Yes (BAAI/bge-reranker-v2-m3) |
| Hotness scoring | Yes (freq x recency) |
| Conflict detection | Yes (vector + LLM adjudication) |
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
python -m benchmarks.report \\
    --micro-json .benchmarks/micro.json \\
    --pipeline-json pipeline_results.json \\
    --locomo-json locomo_results.json \\
    --halumem-json halumem_results.json \\
    --output docs/BENCHMARKS.md
```
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Aggregate Archivist benchmark results into a markdown report")
    parser.add_argument("--micro-json", type=str, help="pytest-benchmark JSON output")
    parser.add_argument("--pipeline-json", type=str, help="Pipeline ablation results JSON")
    parser.add_argument("--locomo-json", type=str, help="LoCoMo benchmark results JSON")
    parser.add_argument("--halumem-json", type=str, help="HaluMem benchmark results JSON")
    parser.add_argument("--output", type=str, default="docs/BENCHMARKS.md", help="Output markdown path")
    args = parser.parse_args()

    micro = _load_json(args.micro_json)
    pipeline = _load_json(args.pipeline_json)
    locomo = _load_json(args.locomo_json)
    halumem = _load_json(args.halumem_json)

    report = generate_report(micro, pipeline, locomo, halumem)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {args.output}")
    print(f"  Micro-benchmarks: {'included' if micro else 'missing'}")
    print(f"  Pipeline ablation: {'included' if pipeline else 'missing'}")
    print(f"  LoCoMo: {'included' if locomo else 'missing'}")
    print(f"  HaluMem: {'included' if halumem else 'missing'}")


if __name__ == "__main__":
    main()
