"""Unit tests for src/context_manager.py."""

import pytest

pytestmark = [pytest.mark.unit]


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
    async def test_compact_flat(self, mock_llm):
        from compaction import compact_flat

        mock_llm.return_value = "This is a flat summary."
        result = await compact_flat([("id1", "Some memory text")])
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_compact_structured(self, mock_llm):
        import json

        from compaction import compact_structured

        mock_llm.return_value = json.dumps(
            {
                "goal": "Deploy app",
                "progress": ["Step 1 done"],
                "decisions": ["Use ArgoCD"],
                "next_steps": ["Run tests"],
                "critical_context": "Cluster is prod-us-east-1",
            }
        )
        result = await compact_structured([("id1", "Memory about deployment")])
        assert isinstance(result, dict)
        assert "goal" in result
        assert "progress" in result

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
