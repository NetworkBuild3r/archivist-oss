"""Tests for Chunk 5: BM25 IDF Dilution + FTS5 Quality.

Covers:
- strip_augmentation_header() correctly strips metadata and handles edge cases
- tools_storage passes raw text (not augmented) to FTS5
- indexer passes raw text (not augmented) to FTS5
- Embedding cache stores tuples and returns without copying
"""

import ast
import inspect

import pytest

# ── strip_augmentation_header ────────────────────────────────────────────────


class TestStripAugmentationHeader:
    """strip_augmentation_header() extracts raw content from augmented text."""

    def test_strips_header(self):
        from contextual_augment import strip_augmentation_header

        augmented = (
            "[Agent: alice | File: logs/2025-03-10.md | Date: 2025-03-10]\n"
            "Key entities: Redis, PostgreSQL\n"
            "---\n"
            "Redis was restarted at 3am."
        )
        assert strip_augmentation_header(augmented) == "Redis was restarted at 3am."

    def test_no_header_returns_unchanged(self):
        from contextual_augment import strip_augmentation_header

        raw = "Redis was restarted at 3am."
        assert strip_augmentation_header(raw) == raw

    def test_empty_string(self):
        from contextual_augment import strip_augmentation_header

        assert strip_augmentation_header("") == ""

    def test_separator_only(self):
        from contextual_augment import strip_augmentation_header

        assert strip_augmentation_header("\n---\n") == ""

    def test_multiple_separators_only_strips_first(self):
        from contextual_augment import strip_augmentation_header

        text = "header\n---\ncontent part 1\n---\ncontent part 2"
        assert strip_augmentation_header(text) == "content part 1\n---\ncontent part 2"

    def test_roundtrip_with_augment_chunk(self):
        from contextual_augment import augment_chunk, strip_augmentation_header

        original = "Port 5432 was unreachable from host-01."
        augmented = augment_chunk(
            original,
            agent_id="bob",
            file_path="logs/db.md",
            date="2025-06-01",
        )
        assert augmented != original
        assert strip_augmentation_header(augmented) == original


# ── FTS5 receives raw text (not augmented) ───────────────────────────────────


class TestToolsStorageFtsRawText:
    """tools_storage.py passes raw text to upsert_fts_chunk, not augmented."""

    def test_fts_receives_raw_text(self):
        """The upsert_fts_chunk call in tools_storage uses the raw `text` variable,
        not embed_input / augmented text."""
        import handlers.tools_storage as ts

        source = inspect.getsource(ts)
        # After the BM25_ENABLED check, upsert_fts_chunk should use text=text
        # and NOT text=_fts_text or text=embed_input
        assert "text=_fts_text" not in source, (
            "tools_storage still passes _fts_text (augmented) to upsert_fts_chunk"
        )

    def test_micro_chunk_fts_receives_raw_text(self):
        """Micro-chunk FTS should use raw mc text, not _micro_embed_inputs."""
        import handlers.tools_storage as ts

        source = inspect.getsource(ts)
        assert "_mc_fts" not in source, (
            "tools_storage still uses _mc_fts variable for micro-chunk FTS"
        )


class TestIndexerFtsRawText:
    """indexer.py passes raw text to upsert_fts_chunk."""

    def test_fts_uses_raw_text_payload(self):
        """indexer should use p.payload['text'], not text_augmented."""
        import indexer

        source = inspect.getsource(indexer)
        # Should NOT prefer text_augmented for FTS
        assert (
            'get("text_augmented"' not in source
            or 'text=p.payload.get("text_augmented"' not in source
        ), "indexer still uses text_augmented for FTS5 insertion"


# ── Embedding cache immutability ─────────────────────────────────────────────


class TestEmbeddingCacheImmutability:
    """Embedding cache stores tuples and returns them without defensive copy."""

    def test_cache_stores_tuple(self):
        from embeddings import _cache_put, _embed_cache, _embed_cache_lock

        _cache_put("hello", "model-a", [1.0, 2.0, 3.0])
        with _embed_cache_lock:
            for _key, (ts, vec) in _embed_cache.items():
                assert isinstance(vec, tuple), f"Cache stores {type(vec).__name__}, expected tuple"
                break

    def test_cache_get_returns_tuple(self):
        from embeddings import _cache_get, _cache_put

        _cache_put("world", "model-b", [4.0, 5.0, 6.0])
        result = _cache_get("world", "model-b")
        assert result is not None
        assert isinstance(result, tuple), f"Cache returns {type(result).__name__}, expected tuple"
        assert result == (4.0, 5.0, 6.0)

    def test_cache_get_no_copy(self):
        """Returned value should be the exact same object, not a copy."""
        from embeddings import _cache_get, _cache_put

        _cache_put("identity", "model-c", [7.0, 8.0])
        result1 = _cache_get("identity", "model-c")
        result2 = _cache_get("identity", "model-c")
        assert result1 is result2, "Cache returns copies instead of the same immutable tuple"

    def test_cache_no_list_copy_in_source(self):
        """_cache_get should not call list() on the cached vector."""
        from embeddings import _cache_get

        source = inspect.getsource(_cache_get)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "list":
                    pytest.fail("_cache_get still calls list() — defensive copy remains")
