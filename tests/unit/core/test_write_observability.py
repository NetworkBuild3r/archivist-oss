"""Unit tests for Chunk 6: observability structured logs and code quality invariants."""

import inspect
import logging

import pytest

pytestmark = [pytest.mark.unit]


class TestConfigFeatureFlagLogging:
    """config.py logs all feature flags at module load."""

    def test_log_feature_flags_function_exists(self):
        import config

        assert hasattr(config, "_log_feature_flags")
        assert callable(config._log_feature_flags)

    def test_log_feature_flags_emits_info(self, caplog):
        import config

        with caplog.at_level(logging.INFO, logger="archivist.config"):
            config._log_feature_flags()
        flag_records = [r for r in caplog.records if "feature_flags" in r.message]
        assert len(flag_records) >= 1
        rec = flag_records[0]
        assert hasattr(rec, "enabled")
        assert hasattr(rec, "disabled")
        assert hasattr(rec, "enabled_count")
        assert hasattr(rec, "disabled_count")

    def test_log_feature_flags_covers_key_flags(self, caplog):
        import config

        with caplog.at_level(logging.INFO, logger="archivist.config"):
            config._log_feature_flags()
        flag_records = [r for r in caplog.records if "feature_flags" in r.message]
        rec = flag_records[0]
        all_flags = rec.enabled + rec.disabled
        for expected in [
            "BM25_ENABLED",
            "REVERSE_HYDE_ENABLED",
            "GRAPH_RETRIEVAL_ENABLED",
            "HOT_CACHE_ENABLED",
            "RERANK_ENABLED",
            "METRICS_ENABLED",
        ]:
            assert expected in all_flags


class TestStorePipelineLog:
    """tools_storage._handle_store emits store_pipeline.complete."""

    def test_store_log_contains_required_fields(self):
        import handlers.tools_storage as ts

        source = inspect.getsource(ts._handle_store)
        assert "store_pipeline.complete" in source
        for field in [
            "memory_id",
            "namespace",
            "agent_id",
            "chunk_count",
            "micro_chunk_count",
            "entity_count",
            "reverse_hyde_queued",
            "duration_ms",
        ]:
            assert field in source

    def test_store_log_uses_time_import(self):
        import handlers.tools_storage as ts

        source = inspect.getsource(ts)
        assert "import time" in source


class TestRetrievalPipelineLog:
    """rlm_retriever.recursive_retrieve emits retrieval_pipeline.complete."""

    def test_retrieval_log_contains_required_fields(self):
        import rlm_retriever

        source = inspect.getsource(rlm_retriever.recursive_retrieve)
        assert "retrieval_pipeline.complete" in source
        for field in [
            "query_length",
            "namespace",
            "agent_id",
            "registry_hits",
            "vector_results",
            "bm25_results",
            "graph_entities",
            "post_threshold",
            "final_count",
            "expansion_variants",
            "hyde_used",
            "ltr_used",
            "duration_ms",
        ]:
            assert field in source


class TestDocstringsAndTypeHints:
    """All new public functions from Chunks 1-5 have docstrings and return annotations."""

    @pytest.mark.parametrize(
        "module_name,func_name",
        [
            ("memory_lifecycle", "delete_memory_complete"),
            ("memory_lifecycle", "archive_memory_complete"),
            ("contextual_augment", "strip_augmentation_header"),
            ("contextual_augment", "augment_chunk"),
            ("graph", "delete_fts_chunks_by_qdrant_id"),
            ("graph", "delete_needle_tokens_by_memory"),
            ("chunking", "_extract_needle_micro_chunks"),
        ],
    )
    def test_has_docstring(self, module_name, func_name):
        import importlib

        mod = importlib.import_module(module_name)
        func = getattr(mod, func_name)
        assert func.__doc__ is not None
        assert len(func.__doc__.strip()) > 10

    @pytest.mark.parametrize(
        "module_name,func_name",
        [
            ("memory_lifecycle", "delete_memory_complete"),
            ("memory_lifecycle", "archive_memory_complete"),
            ("contextual_augment", "strip_augmentation_header"),
            ("contextual_augment", "augment_chunk"),
            ("graph", "delete_fts_chunks_by_qdrant_id"),
            ("graph", "delete_needle_tokens_by_memory"),
        ],
    )
    def test_has_return_annotation(self, module_name, func_name):
        import importlib

        mod = importlib.import_module(module_name)
        func = getattr(mod, func_name)
        sig = inspect.signature(func)
        assert sig.return_annotation is not inspect.Parameter.empty

    def test_result_candidate_factory_docstrings(self):
        from result_types import ResultCandidate

        for method in [
            "from_qdrant_payload",
            "from_registry_hit",
            "from_bm25_hit",
            "update_from_payload",
            "to_dict",
        ]:
            func = getattr(ResultCandidate, method)
            assert func.__doc__ is not None

    def test_delete_result_has_total_property(self):
        from memory_lifecycle import DeleteResult

        dr = DeleteResult(qdrant_primary=1, fts_entries=2, registry_tokens=3)
        assert dr.total == 6
