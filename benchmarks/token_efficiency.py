"""Token efficiency benchmark — measures Archivist's token savings vs naive full-L2 retrieval.

Usage
-----
    cd /opt/appdata/archivist-oss
    python -m benchmarks.token_efficiency [--queries N] [--window-days D] [--output PATH]

The script runs a fixed query set (or a user-supplied size) against the live
Archivist retrieval pipeline with three policies (adaptive / l0_first / l2_first)
and produces a JSON result file in .benchmarks/.

It also prints a summary table comparing:
  - Archivist adaptive       (this implementation)
  - Archivist l2_first       (greedy full-content — our old behavior)
  - Archivist l0_first       (maximum compression)

Output: .benchmarks/token_efficiency_<date>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ── Add src to path for standalone invocation ────────────────────────────────
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "src"))

# ── Load .env ────────────────────────────────────────────────────────────────
try:
    from benchmarks.env_loader import load_env  # type: ignore[import]

    load_env()
except ImportError:
    from dotenv import load_dotenv

    load_dotenv(_repo_root / ".env", override=False)

# ── Imports after env loaded ──────────────────────────────────────────────────
from archivist.retrieval.context_packer import pack_context
from archivist.retrieval.rlm_retriever import recursive_retrieve

logger = logging.getLogger("benchmarks.token_efficiency")

# ---------------------------------------------------------------------------
# Fixed query set (50 representative queries across 3 domains)
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES: list[dict] = [
    # --- domain: engineering ---
    {"query": "How do I fix a SQLite locked database error?", "domain": "engineering"},
    {"query": "What is the difference between asyncio.gather and asyncio.wait?", "domain": "engineering"},
    {"query": "How does hotness scoring work in Archivist?", "domain": "engineering"},
    {"query": "What Qdrant collection configuration is needed for semantic search?", "domain": "engineering"},
    {"query": "How do I run integration tests for the Postgres backend?", "domain": "engineering"},
    {"query": "What is the tier-aware context packing algorithm?", "domain": "engineering"},
    {"query": "How does Archivist handle multi-agent memory isolation?", "domain": "engineering"},
    {"query": "What config variables control token budgeting?", "domain": "engineering"},
    {"query": "How does FTS5 BM25 ranking work in SQLite?", "domain": "engineering"},
    {"query": "What is the difference between l0, l1, and l2 memory tiers?", "domain": "engineering"},
    {"query": "How does the knowledge graph store entity relationships?", "domain": "engineering"},
    {"query": "How do I set up Archivist with a Postgres database?", "domain": "engineering"},
    {"query": "What is reciprocal rank fusion and how is it used here?", "domain": "engineering"},
    {"query": "How does the needle registry work?", "domain": "engineering"},
    {"query": "What happens during archivist_session_end?", "domain": "engineering"},
    {"query": "How does temporal decay affect memory retrieval scores?", "domain": "engineering"},
    {"query": "What is the compact_flat function and when is it called?", "domain": "engineering"},
    # --- domain: operations ---
    {"query": "How do I back up Archivist memory data?", "domain": "operations"},
    {"query": "What Docker services are needed to run Archivist?", "domain": "operations"},
    {"query": "How do I migrate from SQLite to Postgres?", "domain": "operations"},
    {"query": "What environment variables must be set for production?", "domain": "operations"},
    {"query": "How do I check the health of the Archivist stack?", "domain": "operations"},
    {"query": "What is the batch heuristic and when should I use it?", "domain": "operations"},
    {"query": "How do I delete a specific memory?", "domain": "operations"},
    {"query": "How do I restore a memory snapshot?", "domain": "operations"},
    {"query": "What metrics does Archivist expose for monitoring?", "domain": "operations"},
    {"query": "How do I configure hot cache TTL?", "domain": "operations"},
    {"query": "What does the audit log track?", "domain": "operations"},
    {"query": "How do I add a custom namespace for an agent team?", "domain": "operations"},
    {"query": "How does RBAC work in Archivist?", "domain": "operations"},
    {"query": "What is the Qdrant collection replication strategy?", "domain": "operations"},
    {"query": "How do I export retrieval logs for debugging?", "domain": "operations"},
    {"query": "What is the session store max entries limit?", "domain": "operations"},
    # --- domain: agent-usage ---
    {"query": "How should an agent call archivist_store?", "domain": "agent-usage"},
    {"query": "What is the best way to get context for a new task?", "domain": "agent-usage"},
    {"query": "How does an agent hand off work to another agent?", "domain": "agent-usage"},
    {"query": "What should an agent do at the start of a session?", "domain": "agent-usage"},
    {"query": "How do I search memories for a specific time range?", "domain": "agent-usage"},
    {"query": "How does the tips system work?", "domain": "agent-usage"},
    {"query": "What is the right way to store ephemeral task notes?", "domain": "agent-usage"},
    {"query": "How do I promote a session memory to durable storage?", "domain": "agent-usage"},
    {"query": "What is archivist_context_check used for?", "domain": "agent-usage"},
    {"query": "How should a coding agent record a decision?", "domain": "agent-usage"},
    {"query": "What is a trajectory and how should an agent use it?", "domain": "agent-usage"},
    {"query": "How does the multi-hop graph search work?", "domain": "agent-usage"},
    {"query": "What should an agent include when calling archivist_store?", "domain": "agent-usage"},
    {"query": "How do I retrieve only my own agent's memories?", "domain": "agent-usage"},
    {"query": "What is the difference between archivist_recall and archivist_search?", "domain": "agent-usage"},
    {"query": "How do I use archivist_get_context instead of archivist_search?", "domain": "agent-usage"},
]

# Sampling budget: max 100 results returned per query for naive comparison
_NAIVE_LIMIT = 100


# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------


async def measure_query(
    query: str,
    agent_id: str = "benchmark_agent",
    max_tokens: int = 8000,
    policies: list[str] | None = None,
) -> dict:
    """Run *query* through the retrieval pipeline for each policy and measure token savings.

    Returns a dict with per-policy measurements:
      tokens_returned, tokens_naive, savings_pct, duration_ms
    """
    if policies is None:
        policies = ["adaptive", "l0_first", "l2_first"]

    measurements: dict[str, dict] = {}

    for policy in policies:
        t0 = time.monotonic()
        try:
            result = await recursive_retrieve(
                query=query,
                agent_id=agent_id,
                max_tokens=max_tokens,
                refine=False,
            )
            sources = result.get("sources", [])
            ctx_status = result.get("retrieval_trace", {}).get("context_status", {})

            tokens_returned = ctx_status.get("result_tokens_approx")
            tokens_naive: int | None = None

            if sources:
                packed = pack_context(sources, max_tokens=max_tokens, tier_policy=policy)
                tokens_returned = packed.total_tokens
                tokens_naive = packed.naive_tokens
                savings_pct = packed.token_savings_pct
                tier_dist = packed.tier_distribution
            else:
                savings_pct = 0.0
                tier_dist = {}

            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            measurements[policy] = {
                "tokens_returned": tokens_returned,
                "tokens_naive": tokens_naive,
                "savings_pct": savings_pct,
                "tier_distribution": tier_dist,
                "result_count": len(sources),
                "duration_ms": elapsed_ms,
                "over_budget": packed.over_budget if sources else False,
            }
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            measurements[policy] = {
                "error": str(exc),
                "tokens_returned": None,
                "tokens_naive": None,
                "savings_pct": None,
                "duration_ms": elapsed_ms,
            }

    return measurements


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _compute_summary(results: list[dict]) -> dict:
    """Compute aggregate statistics across all measured queries."""
    policies = list(results[0]["measurements"].keys()) if results else []
    summary: dict = {}

    for policy in policies:
        policy_results = [r["measurements"].get(policy, {}) for r in results]
        savings_vals = [p["savings_pct"] for p in policy_results if p.get("savings_pct") is not None]
        token_saved_vals = [
            (p.get("tokens_naive") or 0) - (p.get("tokens_returned") or 0)
            for p in policy_results
            if p.get("tokens_naive") is not None
        ]
        dur_vals = [p["duration_ms"] for p in policy_results if p.get("duration_ms") is not None]

        summary[policy] = {
            "queries_measured": len(savings_vals),
            "avg_savings_pct": round(sum(savings_vals) / len(savings_vals), 1) if savings_vals else None,
            "min_savings_pct": round(min(savings_vals), 1) if savings_vals else None,
            "max_savings_pct": round(max(savings_vals), 1) if savings_vals else None,
            "total_tokens_saved": sum(token_saved_vals),
            "avg_duration_ms": round(sum(dur_vals) / len(dur_vals), 1) if dur_vals else None,
        }

    return summary


def _print_summary_table(summary: dict, total_queries: int) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"\n{'=' * 68}"
    print(header)
    print(f"  Archivist Token Efficiency Benchmark — {total_queries} queries")
    print(header)
    print(f"  {'Policy':<20} {'Avg Savings %':>14} {'Min':>8} {'Max':>8} {'Tokens Saved':>14} {'Avg ms':>8}")
    print(f"  {'-' * 64}")
    for policy, stats in summary.items():
        avg_s = f"{stats['avg_savings_pct']:.1f}%" if stats["avg_savings_pct"] is not None else "N/A"
        min_s = f"{stats['min_savings_pct']:.1f}%" if stats["min_savings_pct"] is not None else "N/A"
        max_s = f"{stats['max_savings_pct']:.1f}%" if stats["max_savings_pct"] is not None else "N/A"
        saved = f"{stats['total_tokens_saved']:,}" if stats["total_tokens_saved"] is not None else "N/A"
        dur = f"{stats['avg_duration_ms']:.0f}" if stats["avg_duration_ms"] is not None else "N/A"
        print(f"  {policy:<20} {avg_s:>14} {min_s:>8} {max_s:>8} {saved:>14} {dur:>8}")
    print(header)
    print()

    # Competitor reference (documented baselines — not live-measured)
    print("  Competitor Reference (documented / estimated — full-history baselines)")
    print(f"  {'-' * 64}")
    competitors = [
        ("Mem0 (flat list)", "~0%", "Returns all memories as flat list; no token budgeting"),
        ("Letta/MemGPT (paging)", "~20%", "In-context paging; drops old blocks, not tier-aware"),
        ("Zep (entity graph)", "~15%", "Entity extraction only; no L0/L1/L2 tiers"),
        ("Graphiti (graph only)", "~30%", "Graph traversal only; no FTS rescue; no budget API"),
    ]
    for name, savings, note in competitors:
        print(f"  {name:<25} {savings:>8}    {note}")
    print(header + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> None:
    queries = BENCHMARK_QUERIES[: args.queries] if args.queries else BENCHMARK_QUERIES
    policies = ["adaptive", "l0_first", "l2_first"]
    agent_id = args.agent_id or "benchmark_agent"
    max_tokens = int(args.max_tokens)

    logger.info("Running token efficiency benchmark on %d queries", len(queries))
    print(f"\nRunning token efficiency benchmark — {len(queries)} queries × {len(policies)} policies")
    print(f"max_tokens={max_tokens}  agent_id={agent_id}\n")

    results: list[dict] = []
    errors = 0

    for i, q in enumerate(queries, 1):
        print(f"  [{i:3d}/{len(queries)}] {q['query'][:70]}", end=" ... ", flush=True)
        measurements = await measure_query(
            query=q["query"],
            agent_id=agent_id,
            max_tokens=max_tokens,
            policies=policies,
        )
        results.append(
            {
                "query": q["query"],
                "domain": q.get("domain", ""),
                "measurements": measurements,
            }
        )
        # Quick line summary
        adaptive = measurements.get("adaptive", {})
        if adaptive.get("savings_pct") is not None:
            print(f"savings={adaptive['savings_pct']:.0f}%  tokens={adaptive.get('tokens_returned', '?')}")
        elif adaptive.get("error"):
            print(f"ERROR: {adaptive['error'][:60]}")
            errors += 1
        else:
            print("no data")

    summary = _compute_summary(results)
    _print_summary_table(summary, len(queries))

    # ── Write output file ────────────────────────────────────────────────────
    output_dir = Path(args.output).parent if args.output else _repo_root / ".benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output)
        if args.output
        else output_dir / f"token_efficiency_{date_str}.json"
    )

    payload = {
        "benchmark": "token_efficiency",
        "version": "1.0",
        "run_at": datetime.now(UTC).isoformat(),
        "config": {
            "queries_run": len(queries),
            "policies": policies,
            "max_tokens": max_tokens,
            "agent_id": agent_id,
        },
        "summary": summary,
        "results": results,
        "errors": errors,
    }

    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Results written to: {output_path}")
    if errors:
        print(f"WARNING: {errors} queries had errors — check 'error' keys in results.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archivist token efficiency benchmark — measure token savings vs naive full-L2 retrieval"
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=0,
        help=f"Number of queries to run (0 = all {len(BENCHMARK_QUERIES)})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Token budget per query (default 8000)",
    )
    parser.add_argument(
        "--agent-id",
        type=str,
        default="benchmark_agent",
        help="Agent ID to use for retrieval (default: benchmark_agent)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON path (default: .benchmarks/token_efficiency_<date>.json)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Retrieval log window for savings stats comparison (default 7)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
