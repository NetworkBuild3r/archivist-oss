"""Unit tests for Chunk 6: observability structured logs and code quality invariants."""

import inspect
import logging

import pytest

pytestmark = [pytest.mark.unit]


class TestConfigFeatureFlagLogging:
    """config.py logs all feature flags at module load."""

    def test_log_feature_flags_function_exists(self):
        import archivist.core.config as config

        assert hasattr(config, "_log_feature_flags")
        assert callable(config._log_feature_flags)

    def test_log_feature_flags_emits_info(self, caplog):
        import archivist.core.config as config

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
        import archivist.core.config as config

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
    """tools_storage store path emits store_pipeline.complete."""

    def test_store_log_contains_required_fields(self):
        import archivist.app.handlers.tools_storage as ts

        # INIT-003/SPEC-004 split wrapper (_handle_store) vs body (_handle_store_inner).
        source = inspect.getsource(ts._handle_store_inner)
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
            "stage_timings",
        ]:
            assert field in source

    def test_store_ack_duration_ms_in_response(self):
        """Store success payload exposes duration_ms for coach_core baselines."""
        import archivist.app.handlers.tools_storage as ts

        source = inspect.getsource(ts._handle_store_inner)
        assert '"duration_ms": _duration_ms' in source

    def test_store_stage_timings_embed_ms_hook(self):
        """INIT-005/SPEC-002: store success exposes stage_timings.embed_ms."""
        import archivist.app.handlers.tools_storage as ts

        source = inspect.getsource(ts._handle_store_inner)
        assert 'stage_timings["embed_ms"]' in source
        assert '"stage_timings": stage_timings' in source
        assert "STORE_EMBED_MS" in source

    def test_store_stage_timings_optional_conflict_ms(self):
        """INIT-005/SPEC-002: optional conflict_ms when conflict check runs."""
        import archivist.app.handlers.tools_storage as ts

        source = inspect.getsource(ts._handle_store_inner)
        assert 'stage_timings["conflict_ms"]' in source
        assert "STORE_CONFLICT_MS" in source

    def test_store_complete_log_stage_timings_no_text_fields(self):
        """Timing log extras must not include memory text / secrets keys."""
        import archivist.app.handlers.tools_storage as ts

        source = inspect.getsource(ts._handle_store_inner)
        # Prefer the success complete log that carries stage_timings.
        marker = '"searchable_lag_metric": m.SEARCHABLE_LAG_SECONDS'
        assert marker in source
        lag_idx = source.index(marker)
        # Window covering the complete-log extras around the lag/stage fields.
        snippet = source[max(0, lag_idx - 600) : lag_idx + 120]
        assert '"stage_timings": stage_timings' in snippet
        # Must not attach fact body to the complete log extras.
        assert '"text":' not in snippet
        assert "embed_input" not in snippet

    def test_store_log_uses_time_import(self):
        import archivist.app.handlers.tools_storage as ts

        source = inspect.getsource(ts)
        assert "import time" in source
        assert "store_pipeline.complete" in inspect.getsource(ts._handle_store_inner)


class TestSearchableLagInstrumentation:
    """INIT-005/SPEC-002: searchable-lag SLO hook reuses OUTBOX_LAG_SECONDS."""

    def test_searchable_lag_aliases_outbox_lag(self):
        from archivist.core import metrics as m

        assert m.SEARCHABLE_LAG_SECONDS == m.OUTBOX_LAG_SECONDS
        assert m.SEARCHABLE_LAG_SECONDS == "archivist_outbox_lag_seconds"

    def test_store_stage_metric_constants_exist(self):
        from archivist.core import metrics as m

        assert m.STORE_EMBED_MS == "archivist_store_embed_duration_ms"
        assert m.STORE_CONFLICT_MS == "archivist_store_conflict_duration_ms"
        assert m.STORE_EMBED_MS != m.STORE_CONFLICT_MS

    def test_outbox_lag_gauge_set_in_async_collector(self):
        """Hook is testable: async gauge collector writes OUTBOX_LAG_SECONDS."""
        import archivist.core.metrics as m

        source = inspect.getsource(m._collect_db_gauges_async)
        assert "OUTBOX_LAG_SECONDS" in source
        assert "gauge_set" in source


class TestRetrievalPipelineLog:
    """archivist.retrieval.rlm_retriever.recursive_retrieve emits retrieval_pipeline.complete."""

    def test_retrieval_log_contains_required_fields(self):
        import archivist.retrieval.rlm_retriever as rlm_retriever

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
            "stage_timings",
        ]:
            assert field in source

    def test_retrieval_stage_timing_keys_recorded(self):
        """Coach-path baselines (INIT-004/SPEC-001): embed_ms + vector_ms hooks."""
        import archivist.retrieval.rlm_retriever as rlm_retriever

        source = inspect.getsource(rlm_retriever.recursive_retrieve)
        assert '_stage_timings["embed_ms"]' in source
        assert '_stage_timings["vector_ms"]' in source


class TestDocstringsAndTypeHints:
    """All new public functions from Chunks 1-5 have docstrings and return annotations."""

    @pytest.mark.parametrize(
        "module_name,func_name",
        [
            ("archivist.lifecycle.memory_lifecycle", "delete_memory_complete"),
            ("archivist.lifecycle.memory_lifecycle", "archive_memory_complete"),
            ("archivist.write.contextual_augment", "strip_augmentation_header"),
            ("archivist.write.contextual_augment", "augment_chunk"),
            ("archivist.storage.graph", "delete_fts_chunks_by_qdrant_id"),
            ("archivist.storage.graph", "delete_needle_tokens_by_memory"),
            ("archivist.utils.chunking", "_extract_needle_micro_chunks"),
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
            ("archivist.lifecycle.memory_lifecycle", "delete_memory_complete"),
            ("archivist.lifecycle.memory_lifecycle", "archive_memory_complete"),
            ("archivist.write.contextual_augment", "strip_augmentation_header"),
            ("archivist.write.contextual_augment", "augment_chunk"),
            ("archivist.storage.graph", "delete_fts_chunks_by_qdrant_id"),
            ("archivist.storage.graph", "delete_needle_tokens_by_memory"),
        ],
    )
    def test_has_return_annotation(self, module_name, func_name):
        import importlib

        mod = importlib.import_module(module_name)
        func = getattr(mod, func_name)
        sig = inspect.signature(func)
        assert sig.return_annotation is not inspect.Parameter.empty

    def test_result_candidate_factory_docstrings(self):
        from archivist.core.result_types import ResultCandidate

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
        from archivist.lifecycle.memory_lifecycle import DeleteResult

        dr = DeleteResult(qdrant_primary=1, fts_entries=2, registry_tokens=3)
        assert dr.total == 6
