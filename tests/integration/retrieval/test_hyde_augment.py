import pytest

"""Tests for Sprint 2 needle-finding improvements (v1.10).

Covers:
  - HyDE needle detection heuristic + cache
  - LTR feature extraction + graceful fallback
  - Contextual chunk augmentation
  - Iterative retrieval (_is_retry parameter)
"""


pytestmark = [pytest.mark.integration, pytest.mark.retrieval]

# ── HyDE Tests ───────────────────────────────────────────────────────────────


class TestHyDE:
    def test_needle_detection_exact_keyword(self):
        from hyde import is_needle_query

        assert is_needle_query("What is the exact backup window?")
        assert is_needle_query("Find the specific IP address of the gateway")
        assert is_needle_query("Who is the on-call DBA?")
        assert is_needle_query("What is the cron schedule?")

    def test_needle_detection_negative(self):
        from hyde import is_needle_query

        assert not is_needle_query("Tell me about kubernetes")
        assert not is_needle_query("Describe the architecture")
        assert not is_needle_query("How does monitoring work?")

    def test_needle_detection_ip_literal(self):
        from hyde import is_needle_query

        assert is_needle_query("Find 192.168.1.1 in the config")

    def test_hyde_cache_roundtrip(self):
        from hyde import _cache_get, _cache_put

        _cache_put("test query", "hypothetical answer")
        assert _cache_get("test query") == "hypothetical answer"

    def test_hyde_cache_miss(self):
        from hyde import _cache_get

        assert _cache_get("never_stored_hyde_xyz_999") is None

    def test_hyde_cache_ttl_expiry(self):
        from hyde import _cache_get, _cache_put, _hyde_cache, _hyde_lock

        _cache_put("hyde_ttl_test", "old answer")
        with _hyde_lock:
            ts, doc = _hyde_cache["hyde_ttl_test"]
            _hyde_cache["hyde_ttl_test"] = (ts - 1200, doc)
        assert _cache_get("hyde_ttl_test") is None


# ── LTR Feature Extraction Tests ────────────────────────────────────────────


class TestLTRFeatures:
    def test_extract_features_basic(self):
        from ranker import FEATURE_NAMES, extract_features

        result = {
            "score": 0.85,
            "vector_score": 0.85,
            "bm25_score": 3.2,
            "rrf_score": 0.03,
            "hotness": 0.7,
            "importance_score": 0.9,
            "temporal_decay": 0.95,
            "outcome_adjustment": 0.1,
            "graph_hop": 1,
            "mention_count": 5,
            "rerank_score": 0.88,
            "text": "Some chunk text here",
            "date": "2026-01-15",
        }
        features = extract_features(result)
        assert len(features) == len(FEATURE_NAMES)
        assert features[0] == 0.85  # vector_score
        assert features[1] == 3.2  # bm25_score
        assert features[10] > 0  # chunk_length

    def test_extract_features_missing_fields(self):
        from ranker import FEATURE_NAMES, extract_features

        result = {"score": 0.5, "text": "hello"}
        features = extract_features(result)
        assert len(features) == len(FEATURE_NAMES)
        assert features[0] == 0.5
        assert all(isinstance(f, float) for f in features)

    def test_ltr_not_available_without_model(self):
        from ranker import ltr_available

        assert not ltr_available()

    def test_rank_results_passthrough_without_model(self):
        from ranker import rank_results

        results = [{"score": 0.9, "text": "a"}, {"score": 0.5, "text": "b"}]
        ranked = rank_results(results)
        assert ranked == results  # unchanged — no model loaded

    def test_feature_names_count(self):
        from ranker import FEATURE_NAMES

        assert len(FEATURE_NAMES) == 12


# ── Contextual Augmentation Tests ────────────────────────────────────────────


class TestContextualAugment:
    def test_augment_adds_metadata_header(self):
        from contextual_augment import augment_chunk

        result = augment_chunk(
            "The backup window is 04:15 UTC.",
            agent_id="needleagent",
            file_path="policies/backup.md",
            date="2026-01-20",
            topic="operations",
        )
        assert "[Agent: needleagent" in result
        assert "File: policies/backup.md" in result
        assert "Date: 2026-01-20" in result
        assert "Topic: operations" in result
        assert "---" in result
        assert "backup window" in result

    def test_augment_no_metadata(self):
        from contextual_augment import augment_chunk

        result = augment_chunk("plain text, no metadata")
        assert "plain text, no metadata" in result

    def test_augment_preserves_original_text(self):
        from contextual_augment import augment_chunk

        original = "ArchivistNeedleV1: the approved production backup window"
        result = augment_chunk(original, agent_id="test")
        assert original in result

    def test_augment_with_entities(self):
        from contextual_augment import augment_chunk

        text = "**Kubernetes** cluster running on **AWS** with **PostgreSQL** backend."
        result = augment_chunk(text, agent_id="infra-bot")
        assert "Key entities:" in result

    def test_augment_thought_type_general_omitted(self):
        from contextual_augment import augment_chunk

        result = augment_chunk("test", thought_type="general")
        assert "Type:" not in result

    def test_augment_thought_type_specific_included(self):
        from contextual_augment import augment_chunk

        result = augment_chunk("test", thought_type="decision")
        assert "Type: decision" in result


# ── Iterative Retrieval Tests ────────────────────────────────────────────────


class TestIterativeRetrieval:
    def test_recursive_retrieve_accepts_is_retry(self):
        """Verify the _is_retry parameter is accepted by the function signature."""
        import inspect

        from rlm_retriever import recursive_retrieve

        sig = inspect.signature(recursive_retrieve)
        assert "_is_retry" in sig.parameters

    def test_is_retry_default_false(self):
        import inspect

        from rlm_retriever import recursive_retrieve

        sig = inspect.signature(recursive_retrieve)
        assert sig.parameters["_is_retry"].default is False


# ── Ranker Training Data Builder Tests ───────────────────────────────────────


class TestRankerTrainData:
    def test_feature_names_match(self):
        from ranker import FEATURE_NAMES as inference_names
        from ranker_train import FEATURE_NAMES as train_names

        assert inference_names == train_names

    def test_load_training_data_empty_db(self, tmp_path):
        """Training on an empty DB should return empty lists."""
        import sqlite3

        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.close()
        from ranker_train import _load_training_data

        features, labels, groups = _load_training_data(db_path)
        assert features == []
        assert labels == []
        assert groups == []
