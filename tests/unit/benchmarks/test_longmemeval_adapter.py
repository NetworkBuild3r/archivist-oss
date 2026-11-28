"""Unit tests for INIT-023/SPEC-003 (LongMemEval B1-B7, cache, budget, hygiene) and SPEC-004."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from benchmarks.academic.longmemeval.ablation_onnx import (
    LOCAL_ONNX_EMBEDDERS,
    LOCAL_ONNX_RERANKERS,
    evaluate_selection_gate,
    run_onnx_ablation,
    select_stratified_dev100,
)
from benchmarks.academic.longmemeval.adapter import (
    DEFAULT_JUDGE_MODEL,
    JudgeEvaluationError,
    _llm_judge,
    _ndcg_at_k,
    _recall_at_k,
    run_longmemeval_benchmark,
)
from benchmarks.cache import BudgetExceededError, DiskCache, SpendTracker

pytestmark = [pytest.mark.unit]


def _synthetic_longmemeval_dataset(n: int = 35) -> list[dict]:
    qtypes = [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    ]
    rows: list[dict] = []
    for i in range(n):
        qtype = qtypes[i % len(qtypes)]
        is_abs = i % 10 == 0
        qid = f"q_{i}_abs" if is_abs else f"q_{i}"
        s1_id = f"sess_{i}_a"
        s2_id = f"sess_{i}_b"
        rows.append(
            {
                "question_id": qid,
                "question_type": qtype,
                "question": f"What color did Alice pick for project {i}?",
                "question_date": "2023/05/30 (Tue) 14:00",
                "answer": "emerald green" if not is_abs else "I don't have that information.",
                "answer_session_ids": [s2_id],
                "haystack_session_ids": [s1_id, s2_id],
                "haystack_dates": ["2023/05/01 (Mon) 10:00", "2023/05/15 (Mon) 11:00"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": f"We started planning project {i}."},
                        {"role": "assistant", "content": "Sounds great!"},
                    ],
                    [
                        {
                            "role": "user",
                            "content": f"Alice chose emerald green for project {i}.",
                        },
                        {"role": "assistant", "content": "Noted emerald green."},
                    ],
                ],
            }
        )
    return rows


class TestLongMemEvalMetricsAndFixes:
    """Covers B1-B7 fixes in benchmarks/academic/longmemeval/adapter.py."""

    def test_b3_ndcg_uses_string_session_ids_and_true_ideal_dcg(self):
        # 2 relevant string session IDs; only 1 retrieved at rank 2 -> ideal DCG has 2 docs, not 1
        retrieved = ["sess_noise", "sess_gold_1", "sess_other"]
        relevant = {"sess_gold_1", "sess_gold_2"}
        score = _ndcg_at_k(retrieved, relevant, k=5)
        # DCG = 1/log2(3) ~= 0.6309; IDCG = 1/log2(2) + 1/log2(3) ~= 1.6309 -> ~0.3869
        assert 0.35 < score < 0.42
        assert _recall_at_k(retrieved, relevant, k=5) == pytest.approx(0.5)

    def test_b3_ndcg_perfect_when_all_relevant_at_top(self):
        retrieved = ["sess_gold_2", "sess_gold_1", "sess_noise"]
        relevant = {"sess_gold_1", "sess_gold_2"}
        assert _ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_b1_b2_b4_b5_b7_retrieval_only_adapter_contract(self, tmp_path):
        data = _synthetic_longmemeval_dataset(3)
        data_file = tmp_path / "lme.json"
        data_file.write_text(json.dumps(data))

        seen_sqlite_paths: list[str] = []
        captured_calls: list[dict] = []

        async def fake_index(q_mem_dir: str, sqlite_path: str, md_files: list[str]) -> int:
            seen_sqlite_paths.append(sqlite_path)
            return len(md_files)

        async def fake_retrieve(**kwargs) -> dict:
            captured_calls.append(dict(kwargs))
            idx = kwargs["query"].rstrip("?").split()[-1]
            return {
                "sources": [
                    {
                        "file_path": f"longmemeval-q_{idx}/session-0002-2023-05-15.md",
                        "tier_text": f"Alice chose emerald green for project {idx}.",
                        "text": f"Alice chose emerald green for project {idx}.",
                    }
                ],
                "total_tokens": 25,
            }

        out = await run_longmemeval_benchmark(
            data_file=str(data_file),
            limit=3,
            mode="retrieval-only",
            profile="default",
            max_tokens=4000,
            cache_dir=str(tmp_path / "cache"),
            checkpoint_path=str(tmp_path / "results.jsonl"),
            index_fn=fake_index,
            retrieve_fn=fake_retrieve,
        )
        summary = out["summary"]
        # B1: retrieval-only MUST NOT report keyword recall as QA accuracy
        assert summary["overall_accuracy"] is None
        assert summary["task_averaged_accuracy"] is None
        assert summary["retrieval"]["recall@5"] == pytest.approx(1.0)
        assert summary["retrieval"]["ndcg@5"] == pytest.approx(1.0)
        # B2: every question gets its own isolated SQLite path
        assert len(seen_sqlite_paths) == 3
        assert len(set(seen_sqlite_paths)) == 3
        # B4: question_date threaded into retrieve_fn as reference_time & as_of
        assert all(c["reference_time"] == "2023/05/30 (Tue) 14:00" for c in captured_calls)
        assert all(c["as_of"] == "2023/05/30 (Tue) 14:00" for c in captured_calls)
        # B5: profile recorded honestly as default
        assert summary["profile"] == "default"
        assert out["manifest"]["profile"] == "default"
        # B7: max_tokens=4000 passed explicitly
        assert all(c["max_tokens"] == 4000 for c in captured_calls)
        assert summary["max_tokens"] == 4000

    @pytest.mark.asyncio
    async def test_b6_judge_raises_error_instead_of_keyword_fallback(self):
        with patch(
            "archivist.features.llm.llm_query",
            new=AsyncMock(side_effect=RuntimeError("429 rate limit")),
        ):
            with pytest.raises(JudgeEvaluationError):
                await _llm_judge(
                    question="What color?",
                    ground_truth="emerald green",
                    hypothesis="emerald green",
                    question_type="single-session-user",
                    is_abstention=False,
                    judge_model=DEFAULT_JUDGE_MODEL,
                    max_retries=2,
                    retry_base_delay_s=0.001,
                )

    @pytest.mark.asyncio
    async def test_b6_full_mode_uses_cache_on_rerun_and_resumes_atomic_jsonl(self, tmp_path):
        data = _synthetic_longmemeval_dataset(2)
        data_file = tmp_path / "lme.json"
        data_file.write_text(json.dumps(data))
        cache_dir = tmp_path / "cache"
        ckpt_path = tmp_path / "results.jsonl"

        gen_calls = 0
        judge_calls = 0

        async def fake_llm(prompt: str, system: str = "", **kwargs) -> str:
            nonlocal gen_calls, judge_calls
            if "yes or no" in prompt.lower() or "model response:" in prompt.lower():
                judge_calls += 1
                return "yes"
            gen_calls += 1
            return "emerald green"

        async def fake_index(q_mem_dir: str, sqlite_path: str, md_files: list[str]) -> int:
            return len(md_files)

        async def fake_retrieve(**kwargs) -> dict:
            idx = kwargs["query"].rstrip("?").split()[-1]
            return {
                "sources": [
                    {
                        "file_path": f"longmemeval-q_{idx}/session-0002-2023-05-15.md",
                        "tier_text": f"Alice chose emerald green for project {idx}.",
                        "text": f"Alice chose emerald green for project {idx}.",
                    }
                ],
                "total_tokens": 20,
            }

        with patch("archivist.features.llm.llm_query", new=fake_llm):
            # Run 1 writes results.jsonl and populates SHA-256 disk cache
            out1 = await run_longmemeval_benchmark(
                data_file=str(data_file),
                limit=2,
                mode="full",
                cache_dir=str(cache_dir),
                checkpoint_path=str(ckpt_path),
                index_fn=fake_index,
                retrieve_fn=fake_retrieve,
            )
            assert out1["summary"]["overall_accuracy"] == pytest.approx(1.0)
            assert gen_calls == 2
            assert judge_calls == 2
            assert ckpt_path.is_file()
            assert len(ckpt_path.read_text(encoding="utf-8").splitlines()) == 2

            # Run 2 with a fresh checkpoint path hits SHA-256 disk cache -> 0 new LLM calls
            out2 = await run_longmemeval_benchmark(
                data_file=str(data_file),
                limit=2,
                mode="full",
                cache_dir=str(cache_dir),
                checkpoint_path=str(tmp_path / "results_run2.jsonl"),
                index_fn=fake_index,
                retrieve_fn=fake_retrieve,
            )
            assert out2["summary"]["overall_accuracy"] == pytest.approx(1.0)
            assert gen_calls == 2
            assert judge_calls == 2
            assert out2["summary"]["cache"]["hits"] >= 4
            assert out2["summary"]["cache"]["misses"] == 0


class TestCacheAndBudgetGuard:
    """Content-addressed SHA-256 cache and hard USD spend cap."""

    def test_disk_cache_roundtrip_and_miss(self, tmp_path):
        cache = DiskCache(tmp_path / "c")
        found, val = cache.get("emb", {"model": "m", "text": "hello"})
        assert found is False and val is None
        cache.set("emb", {"model": "m", "text": "hello"}, {"vec": [0.1, 0.2]})
        found2, val2 = cache.get("emb", {"model": "m", "text": "hello"})
        assert found2 is True
        assert val2 == {"vec": [0.1, 0.2]}
        assert cache.hits == 1
        assert cache.misses == 1

    def test_spend_tracker_enforces_hard_cap(self):
        tracker = SpendTracker(max_budget_usd=1.00)
        tracker.check_and_reserve(
            stage="judge", model="gpt-4o-2024-08-06", input_tokens=100_000, output_tokens=0
        )
        assert tracker.spent_usd < 1.00
        with pytest.raises(BudgetExceededError):
            tracker.check_and_reserve(
                stage="judge", model="gpt-4o-2024-08-06", input_tokens=500_000, output_tokens=0
            )


class TestOnnxAblationHarness:
    """Covers INIT-023/SPEC-004 stratification, dry-run harness, and selection gate."""

    def test_stratified_dev100_preserves_counts_and_is_deterministic(self):
        dataset = _synthetic_longmemeval_dataset(200)
        s1 = select_stratified_dev100(dataset, target_n=100, seed=42)
        s2 = select_stratified_dev100(dataset, target_n=100, seed=42)
        assert s1["question_ids"] == s2["question_ids"]
        assert s1["total_selected"] == 100
        assert s1["abstention_count"] > 0
        assert sum(s1["strata_counts"].values()) == 100

    @pytest.mark.asyncio
    async def test_dry_run_ablation_and_cache_reuse(self, tmp_path):
        dataset = _synthetic_longmemeval_dataset(12)
        data_file = tmp_path / "lme_small.json"
        data_file.write_text(json.dumps(dataset))
        cache_dir = tmp_path / "ablation_cache"

        r1 = await run_onnx_ablation(
            data_file=str(data_file),
            split_path=tmp_path / "nonexistent_split.json",
            mode="dry-run",
            cache_dir=str(cache_dir),
            max_budget_usd=5.00,
        )
        expected_pairs = len(LOCAL_ONNX_EMBEDDERS) * len(LOCAL_ONNX_RERANKERS) + 1
        assert len(r1["pairs"]) == expected_pairs
        assert r1["spend"]["estimated_usd"] == 0.0
        assert r1["spend"]["max_budget_usd"] == 5.00

        r2 = await run_onnx_ablation(
            data_file=str(data_file),
            split_path=tmp_path / "nonexistent_split.json",
            mode="dry-run",
            cache_dir=str(cache_dir),
            max_budget_usd=5.00,
        )
        assert r2["cache"]["misses"] == 0
        assert r2["cache"]["hits"] > 0

    def test_selection_gate_applies_plan_decision_rule(self):
        rows = [
            {
                "pair_key": "BAAI/bge-small-en-v1.5 + ms-marco-MiniLM-L-6-v2",
                "embedder": "BAAI/bge-small-en-v1.5",
                "embedder_dim": 384,
                "embedder_backend": "onnx",
                "reranker": "ms-marco-MiniLM-L-6-v2",
                "recall@5": 0.72,
                "ndcg@5": 0.65,
                "qa_accuracy": 0.78,
                "combined_cpu_p50_ms": 22.0,
            },
            {
                "pair_key": "snowflake-arctic-embed-m-v2.0 + bge-reranker-base-int8",
                "embedder": "snowflake-arctic-embed-m-v2.0",
                "embedder_dim": 768,
                "embedder_backend": "onnx",
                "reranker": "bge-reranker-base-int8",
                "recall@5": 0.76,
                "ndcg@5": 0.69,
                "qa_accuracy": 0.80,
                "combined_cpu_p50_ms": 95.0,
            },
            {
                "pair_key": "text-embedding-3-small + none",
                "embedder": "text-embedding-3-small",
                "embedder_dim": 1536,
                "embedder_backend": "cloud",
                "reranker": "none",
                "recall@5": 0.75,
                "ndcg@5": 0.68,
                "qa_accuracy": 0.80,
                "combined_cpu_p50_ms": 150.0,
            },
        ]
        gate = evaluate_selection_gate(rows)
        assert gate["status"] == "locked"
        # snowflake (95ms) fails the <80ms latency gate; bge-small (22ms, within 2.0pp) wins!
        assert gate["winner"]["embedder"] == "BAAI/bge-small-en-v1.5"
        assert len(gate["passing"]) == 1


class TestBenchmarkHygiene:
    """SPEC-003 hygiene: no fabricated competitor rows and no BENCHMARK_FAST=1 overrides."""

    def test_token_efficiency_has_no_fabricated_competitor_rows(self):
        import benchmarks.token_efficiency as te

        src = inspect.getsource(te)
        assert "estimated from paper" not in src
        assert "Mem0 (default)" not in src
        assert "Zep / Graphiti" not in src

    def test_benchmark_scripts_do_not_export_benchmark_fast(self):
        root = Path(__file__).resolve().parents[3]
        for rel in (
            "benchmarks/scripts/run_thin_reference.sh",
            "benchmarks/scripts/run_longmemeval_ablation.sh",
        ):
            content = (root / rel).read_text(encoding="utf-8")
            assert "BENCHMARK_FAST=1" not in content
