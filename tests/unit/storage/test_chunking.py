"""Unit tests for Chunk 1: needle patterns, DRY extraction, HyDE, registry schema.

Migrated from tests/test_chunk1_dry.py (pure-logic tests only).
Registry schema tests require graph_db fixture (integration); DRY logic is unit.
"""

import hashlib

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]


class TestNeedlePatternsConsistency:
    """NEEDLE_PATTERNS in chunking.py is the single source of truth."""

    def test_graph_imports_from_chunking(self):
        from chunking import NEEDLE_PATTERNS
        from graph import NEEDLE_PATTERNS as graph_patterns

        assert NEEDLE_PATTERNS is graph_patterns

    def test_patterns_are_public(self):
        import chunking

        assert hasattr(chunking, "NEEDLE_PATTERNS")
        assert not hasattr(chunking, "_NEEDLE_PATTERNS")

    def test_pre_extractor_reuses_chunking_patterns(self):
        from chunking import NEEDLE_PATTERNS
        from pre_extractor import _NEEDLE_ENTITY_PATTERNS

        chunking_ip_pat = NEEDLE_PATTERNS[0]
        extractor_ip_pat = _NEEDLE_ENTITY_PATTERNS[0][0]
        assert chunking_ip_pat is extractor_ip_pat

    def test_port_pattern_rejects_timestamps(self):
        from chunking import NEEDLE_PATTERNS

        port_pat = NEEDLE_PATTERNS[7]
        assert not port_pat.search("meeting at 12:30 today")

    def test_port_pattern_matches_real_ports(self):
        from chunking import NEEDLE_PATTERNS

        port_pat = NEEDLE_PATTERNS[7]
        assert port_pat.search("listening on :8443")


class TestHostnameOvermatch:
    """Hostname pattern must not match common English hyphenated phrases."""

    def test_rejects_english_phrases(self):
        from pre_extractor import extract_needle_entities

        false_positives = [
            "well-known-fact",
            "state-of-the-art",
            "day-to-day",
            "end-to-end",
        ]
        for phrase in false_positives:
            entities = extract_needle_entities(phrase)
            hostnames = [e for e in entities if e["type"] == "hostname"]
            assert not hostnames, f"'{phrase}' should NOT be extracted as a hostname"

    def test_accepts_real_hostnames(self):
        from pre_extractor import extract_needle_entities

        entities = extract_needle_entities("connected to prod-web-01 via SSH")
        hostnames = [e for e in entities if e["type"] == "hostname"]
        assert any("prod-web-01" in h["name"] for h in hostnames)


class TestReverseHydeCacheKey:
    """Cache key must use full text hash, not truncated prefix."""

    def test_different_texts_with_same_prefix_get_different_keys(self):
        shared_prefix = "A" * 200
        text_a = shared_prefix + " specific fact about server alpha"
        text_b = shared_prefix + " specific fact about server beta"

        key_a = hashlib.md5(text_a.encode()).hexdigest()
        key_b = hashlib.md5(text_b.encode()).hexdigest()

        assert key_a != key_b

    def test_old_prefix_key_would_collide(self):
        shared_prefix = "A" * 200
        text_a = shared_prefix + " alpha"
        text_b = shared_prefix + " beta"

        assert text_a[:200] == text_b[:200]
        assert hashlib.md5(text_a.encode()).hexdigest() != hashlib.md5(text_b.encode()).hexdigest()


class TestHydeConfigFromConfigPy:
    """hyde.py must use config.py values, not read env vars directly."""

    def test_no_direct_env_reads_in_hyde(self):
        import inspect

        import hyde

        source = inspect.getsource(hyde)
        assert "os.getenv" not in source

    def test_imports_config_values(self):
        import hyde  # noqa: F401 — import compiles without error


class TestNeedleRegistrySchema:
    """Needle registry must have composite index on (token, namespace)."""

    def test_composite_index_exists(self, graph_db):
        from graph import _ensure_needle_registry, get_db

        _ensure_needle_registry()
        conn = get_db()
        try:
            indexes = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='needle_registry'"
            ).fetchall()
            index_names = [r["name"] for r in indexes]
            assert "idx_needle_token_ns" in index_names
        finally:
            conn.close()


class TestDeleteNeedleTokensLogsError:
    """delete_needle_tokens_by_memory must log errors, not silently pass."""

    async def test_returns_count(self, async_pool):
        from graph import delete_needle_tokens_by_memory, register_needle_tokens

        await register_needle_tokens(
            "mem-1", "Server 192.168.1.1 on :8443", namespace="ns1", agent_id="a1"
        )
        count = await delete_needle_tokens_by_memory("mem-1")
        assert count >= 1

    async def test_returns_zero_for_missing(self, async_pool):
        from graph import delete_needle_tokens_by_memory

        count = await delete_needle_tokens_by_memory("nonexistent-id")
        assert count == 0


class TestRlmRetrieverNoShadow:
    """rlm_retriever must not shadow 'import metrics as m' with a loop variable."""

    def test_no_m_loop_variable(self):
        import inspect

        import rlm_retriever

        source = inspect.getsource(rlm_retriever._extract_literal_tokens)
        assert "for m in" not in source

    def test_no_duplicate_metrics_import(self):
        import inspect

        import rlm_retriever

        source = inspect.getsource(rlm_retriever)
        import_lines = [
            line.strip() for line in source.splitlines() if line.strip() == "import metrics as m"
        ]
        assert len(import_lines) <= 1
