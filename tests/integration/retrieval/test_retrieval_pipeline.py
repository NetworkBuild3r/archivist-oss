"""Integration tests for retrieval pipeline — synthetic questions, nominate-then-rerank, parent text, semantic chunking.

Migrated from tests/test_new_modules.py (TestPhase2SyntheticQuestionPipeline,
TestPhase3NominateThenRerank, TestPhase4ParentTextAtIndexTime, TestPhase5SemanticChunking).
"""

import asyncio
import importlib
import inspect
import os
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.retrieval]


_BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks", "pipeline")

def _add_bench_to_path():
    if _BENCH_DIR not in sys.path:
        sys.path.insert(0, _BENCH_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Synthetic question pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2SyntheticQuestionPipeline:
    """Phase 2: verify synthetic question embeddings flow through the search path."""

    def test_search_vectors_returns_representation_type(self):
        """search_vectors result rows must include representation_type."""
        from rlm_retriever import search_vectors

        source = inspect.getsource(search_vectors)
        assert "representation_type" in source
        assert "synthetic_question" in source
        assert "source_memory_id" in source

    def test_search_vectors_filters_synthetic_when_disabled(self):
        """search_vectors must exclude synthetic_question points when the feature is off."""
        from rlm_retriever import search_vectors

        source = inspect.getsource(search_vectors)
        assert "SYNTHETIC_QUESTIONS_ENABLED" in source
        assert "must_not_filters" in source

    def test_dedupe_keeps_best_score(self):
        """When a chunk and its synthetic question both match, keep the higher score."""
        from memory_fusion import dedupe_vector_hits

        chunk_hit = {
            "id": "chunk-1",
            "file_path": "agents/chief/2026-04-01.md",
            "chunk_index": 0,
            "text": "The backup runs at 04:15 UTC every Sunday",
            "score": 0.75,
            "representation_type": "chunk",
        }
        synth_hit = {
            "id": "synth-1",
            "file_path": "agents/chief/2026-04-01.md",
            "chunk_index": 0,
            "text": "The backup runs at 04:15 UTC every Sunday",
            "score": 0.92,
            "representation_type": "synthetic_question",
            "synthetic_question": "What time does the backup run?",
        }
        result = dedupe_vector_hits([chunk_hit, synth_hit])
        assert len(result) == 1
        assert result[0]["score"] == 0.92
        assert result[0]["synthetic_match"] is True

    def test_dedupe_marks_synthetic_match_when_chunk_wins(self):
        """Even when the chunk score wins, synthetic_match should be True."""
        from memory_fusion import dedupe_vector_hits

        chunk_hit = {
            "id": "chunk-1",
            "file_path": "agents/chief/daily.md",
            "chunk_index": 2,
            "text": "PostgreSQL backup via pg_dump",
            "score": 0.95,
            "representation_type": "chunk",
        }
        synth_hit = {
            "id": "synth-2",
            "file_path": "agents/chief/daily.md",
            "chunk_index": 2,
            "text": "PostgreSQL backup via pg_dump",
            "score": 0.80,
            "representation_type": "synthetic_question",
        }
        result = dedupe_vector_hits([chunk_hit, synth_hit])
        assert len(result) == 1
        assert result[0]["score"] == 0.95
        assert result[0]["synthetic_match"] is True

    def test_dedupe_no_synthetic_match_for_pure_chunks(self):
        """Chunks without a synthetic twin should NOT have synthetic_match."""
        from memory_fusion import dedupe_vector_hits

        hits = [
            {
                "id": "c1",
                "file_path": "f1.md",
                "chunk_index": 0,
                "text": "some text",
                "score": 0.8,
                "representation_type": "chunk",
            },
            {
                "id": "c2",
                "file_path": "f2.md",
                "chunk_index": 0,
                "text": "other text",
                "score": 0.7,
                "representation_type": "chunk",
            },
        ]
        result = dedupe_vector_hits(hits)
        assert len(result) == 2
        for r in result:
            assert "synthetic_match" not in r

    def test_trace_includes_synthetic_hits_key(self):
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        assert "synthetic_hits" in source

    def test_force_invalidate_all_clears_cache(self):
        import hot_cache

        old_enabled = hot_cache.HOT_CACHE_ENABLED
        try:
            hot_cache.HOT_CACHE_ENABLED = True
            hot_cache.put("test_agent", "test query", {"answer": "cached"})
            assert hot_cache.get("test_agent", "test query") is not None
            evicted = hot_cache.force_invalidate_all()
            assert evicted >= 1
            assert hot_cache.get("test_agent", "test query") is None
        finally:
            hot_cache.HOT_CACHE_ENABLED = old_enabled
            hot_cache.force_invalidate_all()

    def test_synthetic_point_payload_structure(self):
        """Synthetic question points must carry required fields for the search path."""
        from unittest.mock import AsyncMock, patch

        import synthetic_questions

        mock_qs = '["How do backups work?", "When is the backup window?"]'
        with (
            patch.object(synthetic_questions, "SYNTHETIC_QUESTIONS_ENABLED", True),
            patch.object(synthetic_questions, "SYNTHETIC_QUESTIONS_COUNT", 2),
            patch("synthetic_questions.llm_query", new_callable=AsyncMock) as mock_llm,
            patch("synthetic_questions.embed_batch", new_callable=AsyncMock) as mock_embed,
        ):
            mock_llm.return_value = mock_qs
            mock_embed.return_value = [[0.1] * 1024, [0.2] * 1024]
            synthetic_questions._cache.clear()

            points = asyncio.get_event_loop().run_until_complete(
                synthetic_questions.generate_and_embed_synthetic_points(
                    chunk_point_id="parent-chunk-id",
                    chunk_text="Backups run at 04:15 UTC on Sunday",
                    base_payload={
                        "agent_id": "chief",
                        "namespace": "default",
                        "file_path": "agents/chief/daily.md",
                        "chunk_index": 3,
                    },
                )
            )

        assert len(points) == 2
        for p in points:
            assert p.payload["representation_type"] == "synthetic_question"
            assert p.payload["source_memory_id"] == "parent-chunk-id"
            assert p.payload["text"] == "Backups run at 04:15 UTC on Sunday"
            assert p.payload["file_path"] == "agents/chief/daily.md"
            assert p.payload["chunk_index"] == 3
            assert p.payload["agent_id"] == "chief"

    def test_benchmark_summary_includes_synthetic_counts(self):
        _add_bench_to_path()
        import evaluate

        source = inspect.getsource(evaluate.run_variant)
        assert "total_synthetic_hits" in source
        assert "queries_with_synthetic_hits" in source

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Nominate-then-rerank pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase3NominateThenRerank:
    """Phase 3: verify the clean nominate-then-rerank pipeline."""

    def test_v2_path_bypasses_rrf_merge(self):
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        lines = source.split("\n")
        in_v2 = False
        for line in lines:
            if "v2 CLEAN PATH" in line:
                in_v2 = True
            if in_v2 and "LEGACY PATH" in line:
                break
            if in_v2:
                assert "rrf_merge" not in line

    def test_v2_path_bypasses_bm25_merge(self):
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        assert "not RERANKER_ENABLED" in source

    def test_v2_path_does_id_based_dedup(self):
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        assert "candidate_pool" in source

    def test_v2_pool_collects_all_nomination_sources(self):
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        for source_name in ("vec_results", "bm25_hits", "literal_hits", "_registry_hits"):
            assert source_name in source

    def test_v2_trace_includes_nomination_pool_size(self):
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        assert "nomination_pool_size" in source

    def test_v2_pool_keeps_best_score_per_id(self):
        """When duplicate IDs appear from different nomination sources, keep the highest score."""
        candidate_pool: dict[str, dict] = {}
        test_items = [
            {"id": "abc", "score": 0.5, "file_path": "a.md", "representation_type": "chunk"},
            {"id": "abc", "score": 0.9, "file_path": "a.md", "representation_type": "chunk"},
            {"id": "def", "score": 0.7, "file_path": "b.md", "representation_type": "chunk"},
        ]
        for r in test_items:
            rid = str(r.get("id", ""))
            existing = candidate_pool.get(rid)
            if existing is None or r.get("score", 0) > existing.get("score", 0):
                candidate_pool[rid] = dict(r)

        assert len(candidate_pool) == 2
        assert candidate_pool["abc"]["score"] == 0.9
        assert candidate_pool["def"]["score"] == 0.7

    def test_v2_pool_marks_synthetic_match(self):
        """Synthetic question hits must be tagged with synthetic_match=True in the pool."""
        candidate_pool: dict[str, dict] = {}
        test_items = [
            {
                "id": "c1",
                "score": 0.8,
                "file_path": "a.md",
                "chunk_index": 0,
                "representation_type": "chunk",
            },
            {
                "id": "s1",
                "score": 0.85,
                "file_path": "a.md",
                "chunk_index": 0,
                "representation_type": "synthetic_question",
            },
        ]
        for r in test_items:
            rid = str(r.get("id", ""))
            existing = candidate_pool.get(rid)
            if existing is None or r.get("score", 0) > existing.get("score", 0):
                candidate_pool[rid] = dict(r)
            if r.get("representation_type") == "synthetic_question":
                candidate_pool[rid]["synthetic_match"] = True

        assert candidate_pool["s1"].get("synthetic_match") is True
        assert candidate_pool["c1"].get("synthetic_match") is not True

    def test_v2_pool_preserves_needle_registry_hit(self):
        """Needle registry hits must keep their needle_registry_hit flag through dedup."""
        candidate_pool: dict[str, dict] = {}
        test_items = [
            {
                "id": "n1",
                "score": 1.0,
                "file_path": "needle.md",
                "needle_registry_hit": True,
                "representation_type": "chunk",
            },
            {
                "id": "n1",
                "score": 0.9,
                "file_path": "needle.md",
                "representation_type": "chunk",
            },
        ]
        for r in test_items:
            rid = str(r.get("id", ""))
            existing = candidate_pool.get(rid)
            if existing is None or r.get("score", 0) > existing.get("score", 0):
                candidate_pool[rid] = dict(r)
            if r.get("needle_registry_hit"):
                candidate_pool[rid]["needle_registry_hit"] = True

        assert candidate_pool["n1"]["needle_registry_hit"] is True
        assert candidate_pool["n1"]["score"] == 1.0

    def test_clean_reranker_variant_exists(self):
        _add_bench_to_path()
        import evaluate

        importlib.reload(evaluate)
        v = evaluate.VARIANTS["clean_reranker"]
        assert v["RERANKER_ENABLED"] == "true"
        assert v["BM25_ENABLED"] == "true"
        assert v["GRAPH_RETRIEVAL_ENABLED"] == "true"
        assert v["SYNTHETIC_QUESTIONS_ENABLED"] == "true"
        assert v["QUERY_EXPANSION_ENABLED"] == "false"

    def test_clean_reranker_variant_no_legacy_scoring(self):
        _add_bench_to_path()
        import evaluate

        importlib.reload(evaluate)
        v = evaluate.VARIANTS["clean_reranker"]
        assert v.get("HOTNESS_WEIGHT") == "0"
        assert v.get("TEMPORAL_DECAY_HALFLIFE_DAYS") == "0"
        assert v.get("RERANK_ENABLED") == "false"

    def test_legacy_path_preserved(self):
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        assert "LEGACY PATH" in source
        assert "apply_temporal_decay" in source
        assert "apply_hotness_to_results" in source
        assert "apply_retrieval_threshold" in source

    def test_v2_no_threshold_filter(self):
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        lines = source.split("\n")
        in_v2 = False
        for line in lines:
            if "v2 CLEAN PATH" in line:
                in_v2 = True
            if in_v2 and "LEGACY PATH" in line:
                break
            if in_v2:
                assert "apply_dynamic_threshold" not in line
                assert "apply_retrieval_threshold" not in line

    def test_benchmark_summary_includes_nomination_pool(self):
        _add_bench_to_path()
        import evaluate

        importlib.reload(evaluate)
        source = inspect.getsource(evaluate.run_variant)
        assert "nomination_pool_size" in source
        assert "avg_nomination_pool_size" in source

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Parent text at index time
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase4ParentTextAtIndexTime:
    """Phase 4: verify parent context is stored at index time, not fetched at runtime."""

    def test_enrich_with_parent_deleted(self):
        import rlm_retriever

        assert not hasattr(rlm_retriever, "enrich_with_parent")

    def test_no_enrich_with_parent_calls(self):
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        assert "enrich_with_parent" not in source

    def test_search_vectors_returns_parent_text(self):
        from rlm_retriever import search_vectors

        source = inspect.getsource(search_vectors)
        assert '"parent_text"' in source or "'parent_text'" in source

    def test_indexer_stores_parent_text_hierarchical(self):
        from indexer import index_file

        source = inspect.getsource(index_file)
        assert "parent_text" in source
        assert "_parent_text_map" in source

    def test_indexer_parent_text_map_correctness(self):
        hier_chunks = [
            {"id": "p1", "parent_id": None, "content": "Parent text here", "is_parent": True},
            {"id": "c1", "parent_id": "p1", "content": "Child text", "is_parent": False},
            {"id": "c2", "parent_id": "p1", "content": "Another child", "is_parent": False},
            {"id": "p2", "parent_id": None, "content": "Second parent", "is_parent": True},
        ]
        _parent_text_map = {c["id"]: c["content"] for c in hier_chunks if c["is_parent"]}
        assert _parent_text_map["p1"] == "Parent text here"
        assert _parent_text_map["p2"] == "Second parent"
        assert "c1" not in _parent_text_map

        for c in hier_chunks:
            if not c["is_parent"] and c["parent_id"]:
                parent_text = _parent_text_map.get(c["parent_id"], "")
                assert parent_text == "Parent text here"

    def test_indexer_flat_chunks_have_empty_parent_text(self):
        from indexer import index_file

        source = inspect.getsource(index_file)
        assert source.count('"parent_text"') >= 2

    def test_reranker_reads_parent_text(self):
        from reranker import _build_pair

        candidate_with_text = {
            "text": "Child chunk content",
            "parent_text": "Full parent context from indexing",
        }
        pair = _build_pair("test query", candidate_with_text)
        assert "Full parent context from indexing" in pair
        assert "Parent context:" in pair

    def test_reranker_backward_compat_parent_context(self):
        from reranker import _build_pair

        candidate_legacy = {
            "text": "Child chunk content",
            "parent_context": "Legacy parent context",
        }
        pair = _build_pair("test query", candidate_legacy)
        assert "Legacy parent context" in pair

    def test_reranker_no_parent_text_omits_section(self):
        from reranker import _build_pair

        candidate_no_parent = {
            "text": "Standalone chunk content",
            "parent_text": "",
        }
        pair = _build_pair("test query", candidate_no_parent)
        assert "Parent context:" not in pair

    def test_refine_uses_parent_text(self):
        from rlm_retriever import _refine_one_chunk

        source = inspect.getsource(_refine_one_chunk)
        assert "parent_text" in source
        assert "parent_context" not in source

    def test_v2_trace_parent_enriched_uses_parent_text(self):
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        assert "parent_text" in source
        assert "parent_context" not in source

# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Semantic chunking
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase5SemanticChunking:
    """Phase 5: Semantic / meaning-boundary chunking."""

    def test_chunking_strategy_config_exists(self):
        import os

        import config

        assert hasattr(config, "CHUNKING_STRATEGY")
        assert config.CHUNKING_STRATEGY in ("semantic", "fixed")
        if "CHUNKING_STRATEGY" not in os.environ:
            assert config.CHUNKING_STRATEGY == "semantic"

    def test_chunk_text_semantic_exported(self):
        from chunking import chunk_text_semantic

        assert callable(chunk_text_semantic)

    def test_chunk_text_semantic_short_doc_fast_path(self):
        from chunking import chunk_text_semantic

        text = "## Title\n\nShort content that fits in one chunk easily."
        result = chunk_text_semantic(text, size=2000)
        assert len(result) == 1
        assert result[0] == text.strip()

    def test_chunk_text_semantic_long_doc_splits(self):
        from chunking import chunk_text_semantic

        body = "Content word. " * 50
        text = f"## Section A\n\n{body}\n\n## Section B\n\n{body}"
        result = chunk_text_semantic(text, size=800)
        assert len(result) >= 2

    def test_chunk_text_semantic_no_cross_section_merge(self):
        from chunking import chunk_text_semantic

        body = "unique_marker_{n}. " * 40
        text = f"## One\n\n{body.format(n='A')}\n\n## Two\n\n{body.format(n='B')}"
        result = chunk_text_semantic(text, size=500)
        for chunk in result:
            has_a = "unique_marker_A" in chunk
            has_b = "unique_marker_B" in chunk
            assert not (has_a and has_b)

    def test_hierarchical_accepts_strategy_param(self):
        from chunking import chunk_text_hierarchical

        sig = inspect.signature(chunk_text_hierarchical)
        assert "strategy" in sig.parameters
        default = sig.parameters["strategy"].default
        assert default == "semantic"

    def test_hierarchical_semantic_uses_chunk_text_semantic(self, monkeypatch):
        import chunking

        calls = []
        original = chunking.chunk_text_semantic

        def spy(text, **kwargs):
            calls.append(text)
            return original(text, **kwargs)

        monkeypatch.setattr(chunking, "chunk_text_semantic", spy)
        body = "word " * 100
        text = f"## Section\n\n{body}"
        chunking.chunk_text_hierarchical(text, "test.md", parent_size=300, strategy="semantic")
        assert len(calls) > 0

    def test_hierarchical_fixed_does_not_use_chunk_text_semantic(self, monkeypatch):
        import chunking

        calls = []
        original = chunking.chunk_text_semantic

        def spy(text, **kwargs):
            calls.append(text)
            return original(text, **kwargs)

        monkeypatch.setattr(chunking, "chunk_text_semantic", spy)
        body = "word " * 100
        text = f"## Section\n\n{body}"
        chunking.chunk_text_hierarchical(text, "test.md", parent_size=300, strategy="fixed")
        assert len(calls) == 0

    def test_indexer_imports_chunking_strategy(self):
        import indexer

        source = inspect.getsource(indexer)
        assert "CHUNKING_STRATEGY" in source

    def test_indexer_passes_strategy_to_hierarchical(self):
        import indexer

        source = inspect.getsource(indexer)
        assert "strategy=CHUNKING_STRATEGY" in source

    def test_short_document_strategy_invariant(self):
        from chunking import chunk_text_hierarchical

        text = "## Note\n\nA short note that fits comfortably inside one parent chunk."
        sem = chunk_text_hierarchical(text, "note.md", parent_size=2000, strategy="semantic")
        fix = chunk_text_hierarchical(text, "note.md", parent_size=2000, strategy="fixed")
        sem_parents = [c["content"] for c in sem if c["is_parent"]]
        fix_parents = [c["content"] for c in fix if c["is_parent"]]
        assert sem_parents == fix_parents
