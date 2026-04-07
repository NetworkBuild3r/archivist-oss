"""Tests for v1.1-1.3 modules: tokenizer, context_manager, compaction, fts_search, curator checksum."""

import pytest


class TestTokenizer:
    def test_count_tokens_nonempty(self):
        from tokenizer import count_tokens

        n = count_tokens("Hello, world!")
        assert n > 0

    def test_count_tokens_empty(self):
        from tokenizer import count_tokens

        n = count_tokens("")
        assert n >= 0

    def test_count_message_tokens(self):
        from tokenizer import count_message_tokens

        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        n = count_message_tokens(msgs)
        assert n > 10  # overhead + content

    def test_count_message_tokens_empty(self):
        from tokenizer import count_message_tokens

        n = count_message_tokens([])
        assert n == 0

    def test_fallback_approximation(self):
        from tokenizer import count_tokens

        text = "a" * 400
        n = count_tokens(text)
        assert n >= 50  # at least chars//4 ballpark


class TestContextManager:
    def test_check_context_under_budget(self):
        from context_manager import check_context

        msgs = [{"role": "user", "content": "Short message."}]
        result = check_context(msgs, budget_tokens=10000)
        assert result["over_budget"] is False
        assert result["hint"] == "ok"

    def test_check_context_empty(self):
        from context_manager import check_context

        result = check_context([], budget_tokens=1000)
        assert result["total_tokens"] == 0
        assert result["hint"] == "ok"

    def test_check_context_over_budget(self):
        from context_manager import check_context

        msgs = [{"role": "user", "content": "x " * 5000}]
        result = check_context(msgs, budget_tokens=100)
        assert result["over_budget"] is True
        assert result["hint"] in ("compress", "critical")

    def test_check_memories_budget(self):
        from context_manager import check_memories_budget

        result = check_memories_budget(["short text"], budget_tokens=10000)
        assert result["over_budget"] is False
        assert result["memory_count"] == 1

    def test_check_memories_budget_over(self):
        from context_manager import check_memories_budget

        texts = ["word " * 1000 for _ in range(10)]
        result = check_memories_budget(texts, budget_tokens=100)
        assert result["over_budget"] is True


class TestCompaction:
    @pytest.mark.asyncio
    async def test_compact_flat(self, mock_llm):
        from compaction import compact_flat

        mock_llm.return_value = "This is a flat summary."
        result = await compact_flat([("id1", "Some memory text")])
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_compact_structured(self, mock_llm):
        import json
        from compaction import compact_structured

        mock_llm.return_value = json.dumps({
            "goal": "Deploy app",
            "progress": ["Step 1 done"],
            "decisions": ["Use ArgoCD"],
            "next_steps": ["Run tests"],
            "critical_context": "Cluster is prod-us-east-1",
        })
        result = await compact_structured([("id1", "Memory about deployment")])
        assert isinstance(result, dict)
        assert "goal" in result
        assert "progress" in result

    @pytest.mark.asyncio
    async def test_compact_structured_fallback(self, mock_llm):
        from compaction import compact_structured

        mock_llm.return_value = "Not valid JSON at all"
        result = await compact_structured([("id1", "Some memory")])
        assert isinstance(result, dict)
        assert "progress" in result

    def test_format_structured_summary(self):
        from compaction import format_structured_summary

        data = {
            "goal": "Migrate database",
            "progress": ["Schema created", "Data exported"],
            "decisions": ["Use PostgreSQL"],
            "next_steps": ["Run migration script"],
            "critical_context": "Prod cluster: us-east-1",
        }
        md = format_structured_summary(data)
        assert "## Goal" in md
        assert "## Progress" in md
        assert "PostgreSQL" in md


class TestFTSSearch:
    def test_fts5_safe_query_basic(self):
        from fts_search import _fts5_safe_query

        q = _fts5_safe_query("kubernetes deployment")
        assert '"kubernetes"' in q
        assert '"deployment"' in q

    def test_fts5_safe_query_special_chars(self):
        from fts_search import _fts5_safe_query

        q = _fts5_safe_query("NOT (foo AND bar)")
        assert "NOT" not in q or '"NOT"' in q

    def test_fts5_safe_query_empty(self):
        from fts_search import _fts5_safe_query

        assert _fts5_safe_query("") == ""

    def test_search_bm25_disabled(self, monkeypatch):
        monkeypatch.setattr("fts_search.BM25_ENABLED", False)
        from fts_search import search_bm25

        results = search_bm25("test query")
        assert results == []

    def test_merge_vector_and_bm25_empty_bm25(self):
        from fts_search import merge_vector_and_bm25

        vec = [{"id": "a", "score": 0.9, "text": "t"}]
        result = merge_vector_and_bm25(vec, [])
        assert result == vec

    def test_merge_vector_and_bm25_fusion(self):
        from fts_search import merge_vector_and_bm25

        vec = [{"id": "a", "score": 0.9, "text": "t1", "qdrant_id": "a"}]
        bm25 = [{"qdrant_id": "a", "bm25_score": 5.0, "text": "t1"}]
        result = merge_vector_and_bm25(vec, bm25)
        assert len(result) >= 1
        assert result[0]["score"] > 0.9 * 0.7  # vector contrib + bm25 contrib


class TestEntityExtraction:
    def test_ngram_expansion(self, graph_db):
        from graph import upsert_entity
        from graph_retrieval import extract_entity_mentions

        upsert_entity("Argo CD", "tool")
        upsert_entity("hot cache", "concept")

        results = extract_entity_mentions("How does Argo CD use the hot cache?")
        names = [e["name"] for e in results]
        assert any("Argo CD" in n for n in names)
        assert any("hot cache" in n for n in names)

    def test_short_entities(self, graph_db):
        from graph import upsert_entity
        from graph_retrieval import extract_entity_mentions

        upsert_entity("AI", "concept")
        results = extract_entity_mentions("How does AI work?")
        names = [e["name"] for e in results]
        assert any("AI" in n for n in names)


class TestCuratorChecksum:
    def test_file_checksum_deterministic(self):
        from curator import _file_checksum

        h1 = _file_checksum("hello world")
        h2 = _file_checksum("hello world")
        assert h1 == h2

    def test_file_checksum_differs(self):
        from curator import _file_checksum

        h1 = _file_checksum("hello world")
        h2 = _file_checksum("hello world!")
        assert h1 != h2
