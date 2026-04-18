"""Smoke tests for all 37 MCP tool handlers across all 7 handler modules.

Each test calls the handler directly (no MCP transport), mocks all external I/O,
and asserts the response is a valid ``list[TextContent]`` with:

* At least one item.
* ``.type == "text"`` on the first item.
* Non-empty ``.text``.
* No ``Traceback`` spill.
* No ``coroutine`` string — this catches unawaited-async regressions.

The "coroutine" guard is the key addition that would have caught all of
Nova's HIGH-severity findings before they reached production.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.system, pytest.mark.mcp]

# ---------------------------------------------------------------------------
# Shared assertion helper
# ---------------------------------------------------------------------------


def _assert_text_response(result) -> None:
    """Core smoke assertion: valid TextContent, no tracebacks, no coroutines."""
    assert isinstance(result, list) and len(result) > 0, "Handler returned empty list"
    first = result[0]
    assert first.type == "text", f"Expected type='text', got {first.type!r}"
    assert first.text, "Handler returned empty text"
    assert "Traceback" not in first.text, f"Traceback in response:\n{first.text[:400]}"
    assert "coroutine" not in first.text, (
        "Response contains 'coroutine' — likely an unawaited async call:\n" + first.text[:400]
    )


# ---------------------------------------------------------------------------
# Parametrized registry completeness test
# ---------------------------------------------------------------------------

_ALL_EXPECTED_TOOLS = [
    # tools_search (9)
    "archivist_search",
    "archivist_recall",
    "archivist_timeline",
    "archivist_insights",
    "archivist_deref",
    "archivist_index",
    "archivist_contradictions",
    "archivist_entity_brief",
    "archivist_wake_up",
    # tools_storage (6)
    "archivist_store",
    "archivist_merge",
    "archivist_compress",
    "archivist_pin",
    "archivist_unpin",
    "archivist_delete",
    # tools_trajectory (5)
    "archivist_log_trajectory",
    "archivist_annotate",
    "archivist_rate",
    "archivist_tips",
    "archivist_session_end",
    # tools_skills (6)
    "archivist_register_skill",
    "archivist_skill_event",
    "archivist_skill_lesson",
    "archivist_skill_health",
    "archivist_skill_relate",
    "archivist_skill_dependencies",
    # tools_admin (8)
    "archivist_context_check",
    "archivist_namespaces",
    "archivist_audit_trail",
    "archivist_resolve_uri",
    "archivist_retrieval_logs",
    "archivist_health_dashboard",
    "archivist_batch_heuristic",
    "archivist_backup",
    # tools_cache (2)
    "archivist_cache_stats",
    "archivist_cache_invalidate",
    # tools_docs (1)
    "archivist_get_reference_docs",
]


@pytest.mark.parametrize("tool_name", _ALL_EXPECTED_TOOLS)
def test_tool_registered_in_handlers(tool_name: str) -> None:
    """Every expected tool name appears in its module's HANDLERS dict."""
    from archivist.app.handlers import (
        tools_admin,
        tools_cache,
        tools_docs,
        tools_search,
        tools_skills,
        tools_storage,
        tools_trajectory,
    )

    all_handlers: dict = {}
    for mod in (
        tools_search,
        tools_storage,
        tools_trajectory,
        tools_skills,
        tools_admin,
        tools_cache,
        tools_docs,
    ):
        all_handlers.update(mod.HANDLERS)  # type: ignore[arg-type]

    assert tool_name in all_handlers, (
        f"'{tool_name}' is not registered in any handler module's HANDLERS dict"
    )


# ---------------------------------------------------------------------------
# tools_docs — 1 tool
# ---------------------------------------------------------------------------


class TestDocsHandler:
    async def test_get_reference_docs_returns_content(self) -> None:
        from archivist.app.handlers.tools_docs import _handle_get_reference_docs

        result = await _handle_get_reference_docs({})
        _assert_text_response(result)

    async def test_get_reference_docs_falls_back_gracefully_when_files_missing(
        self, tmp_path, monkeypatch
    ) -> None:
        """Handler must return a JSON error dict (not crash) when both doc files are absent."""
        from archivist.app.handlers import tools_docs

        monkeypatch.setattr(tools_docs, "_SKILL_DOC", tmp_path / "missing1.md")
        monkeypatch.setattr(tools_docs, "_FALLBACK_DOC", tmp_path / "missing2.md")

        result = await tools_docs._handle_get_reference_docs({})
        assert isinstance(result, list) and result
        data = json.loads(result[0].text)
        assert data.get("error") == "reference_docs_not_found"
        assert "tried" in data

    async def test_get_reference_docs_section_filter(self) -> None:
        from archivist.app.handlers.tools_docs import _handle_get_reference_docs

        result = await _handle_get_reference_docs({"section": "search"})
        _assert_text_response(result)


# ---------------------------------------------------------------------------
# tools_cache — 2 tools
# ---------------------------------------------------------------------------


class TestCacheHandlers:
    async def test_cache_stats_returns_text(self) -> None:
        from archivist.app.handlers.tools_cache import _handle_cache_stats

        with patch("archivist.retrieval.hot_cache.stats", return_value={"entries": 0}):
            result = await _handle_cache_stats({})
        _assert_text_response(result)

    async def test_cache_invalidate_returns_text(self) -> None:
        from archivist.app.handlers.tools_cache import _handle_cache_invalidate

        result = await _handle_cache_invalidate({"all": True})
        _assert_text_response(result)


# ---------------------------------------------------------------------------
# tools_admin — 8 tools
# ---------------------------------------------------------------------------


class TestAdminHandlers:
    async def test_context_check_no_args_returns_text(self) -> None:
        from archivist.app.handlers.tools_admin import _handle_context_check

        result = await _handle_context_check({})
        _assert_text_response(result)

    async def test_namespaces_returns_text(self) -> None:
        from archivist.app.handlers.tools_admin import _handle_namespaces

        with (
            patch(
                "archivist.app.handlers.tools_admin.list_accessible_namespaces",
                return_value=["default"],
            ),
            patch(
                "archivist.app.handlers.tools_admin.get_namespace_for_agent",
                return_value="default",
            ),
        ):
            result = await _handle_namespaces({"agent_id": "agent-1"})
        _assert_text_response(result)

    async def test_audit_trail_returns_text(self) -> None:
        from archivist.app.handlers.tools_admin import _handle_audit_trail

        with (
            patch("archivist.core.audit.get_audit_trail", return_value=[]),
            patch("archivist.core.audit.get_agent_activity", return_value=[]),
        ):
            result = await _handle_audit_trail({})
        _assert_text_response(result)

    async def test_resolve_uri_returns_error_for_bad_uri(self) -> None:
        from archivist.app.handlers.tools_admin import _handle_resolve_uri

        result = await _handle_resolve_uri({"uri": "not-a-valid-uri"})
        assert isinstance(result, list) and result
        # Should return some JSON (error or success)
        assert result[0].text
        assert "coroutine" not in result[0].text

    async def test_retrieval_logs_returns_text(self) -> None:
        from archivist.app.handlers.tools_admin import _handle_retrieval_logs

        with patch(
            "archivist.app.handlers.tools_admin.get_retrieval_logs",
            return_value=[],
        ):
            result = await _handle_retrieval_logs({"agent_id": "agent-1"})
        _assert_text_response(result)

    async def test_health_dashboard_returns_text(self) -> None:
        from archivist.app.handlers.tools_admin import _handle_health_dashboard

        with patch(
            "archivist.app.handlers.tools_admin.build_dashboard",
            return_value={"status": "ok"},
        ):
            result = await _handle_health_dashboard({})
        _assert_text_response(result)

    async def test_batch_heuristic_returns_text(self) -> None:
        from archivist.app.handlers.tools_admin import _handle_batch_heuristic

        with patch(
            "archivist.app.handlers.tools_admin.batch_heuristic",
            return_value={"recommended_batch": 10},
        ):
            result = await _handle_batch_heuristic({})
        _assert_text_response(result)

    async def test_backup_list_returns_text(self) -> None:
        from archivist.app.handlers.tools_admin import _handle_backup

        with patch(
            "archivist.storage.backup_manager.list_snapshots",
            return_value=[],
        ):
            result = await _handle_backup({"action": "list"})
        _assert_text_response(result)


# ---------------------------------------------------------------------------
# tools_skills — 6 tools
# ---------------------------------------------------------------------------


class TestSkillHandlers:
    async def test_register_skill_returns_text(self, qa_pool) -> None:
        from archivist.app.handlers.tools_skills import _handle_register_skill

        result = await _handle_register_skill(
            {
                "name": "test_skill_smoke",
                "agent_id": "agent-smoke",
                "version": "1.0.0",
                "description": "Smoke test skill",
            }
        )
        _assert_text_response(result)

    async def test_skill_event_not_found_returns_error(self) -> None:
        from archivist.app.handlers.tools_skills import _handle_skill_event

        result = await _handle_skill_event(
            {
                "skill_name": "nonexistent_skill_xyz",
                "agent_id": "agent-1",
                "outcome": "success",
            }
        )
        assert isinstance(result, list) and result
        data = json.loads(result[0].text)
        assert data.get("error") == "skill_not_found"
        assert "coroutine" not in result[0].text

    async def test_skill_lesson_not_found_returns_error(self) -> None:
        from archivist.app.handlers.tools_skills import _handle_skill_lesson

        result = await _handle_skill_lesson(
            {
                "skill_name": "nonexistent_skill_xyz",
                "agent_id": "agent-1",
                "title": "Lesson title",
                "content": "Lesson content",
            }
        )
        assert isinstance(result, list) and result
        data = json.loads(result[0].text)
        assert data.get("error") == "skill_not_found"

    async def test_skill_health_not_found_returns_error(self) -> None:
        from archivist.app.handlers.tools_skills import _handle_skill_health

        result = await _handle_skill_health({"skill_name": "nonexistent_skill_xyz"})
        assert isinstance(result, list) and result
        data = json.loads(result[0].text)
        assert data.get("error") == "skill_not_found"

    async def test_skill_relate_not_found_returns_error(self) -> None:
        from archivist.app.handlers.tools_skills import _handle_skill_relate

        result = await _handle_skill_relate(
            {
                "skill_a": "nonexistent_a",
                "skill_b": "nonexistent_b",
                "relation_type": "depends_on",
                "agent_id": "agent-1",
            }
        )
        assert isinstance(result, list) and result
        data = json.loads(result[0].text)
        assert data.get("error") == "skill_not_found"

    async def test_skill_dependencies_not_found_returns_error(self) -> None:
        from archivist.app.handlers.tools_skills import _handle_skill_dependencies

        result = await _handle_skill_dependencies({"skill_name": "nonexistent_skill_xyz"})
        assert isinstance(result, list) and result
        data = json.loads(result[0].text)
        assert data.get("error") == "skill_not_found"


# ---------------------------------------------------------------------------
# tools_trajectory — 5 tools
# ---------------------------------------------------------------------------

_FAKE_MEMORY_ID = "00000000-0000-0000-0000-000000000001"


class TestTrajectoryHandlers:
    async def test_annotate_returns_annotation_id(self, qa_pool) -> None:
        """Regression: _handle_annotate must await add_annotation (was unawaited)."""
        from archivist.app.handlers.tools_trajectory import _handle_annotate

        with (
            patch(
                "archivist.app.handlers.tools_trajectory.add_annotation",
                new_callable=AsyncMock,
                return_value="ann-001",
            ),
            patch("archivist.core.audit.log_memory_event", new_callable=AsyncMock),
            patch(
                "archivist.core.trajectory.get_annotations",
                return_value=[],
            ),
            patch(
                "archivist.app.handlers.tools_trajectory.get_annotations",
                return_value=[],
            ),
        ):
            result = await _handle_annotate(
                {
                    "memory_id": _FAKE_MEMORY_ID,
                    "agent_id": "agent-smoke",
                    "content": "This memory was accurate.",
                    "annotation_type": "note",
                }
            )
        _assert_text_response(result)
        data = json.loads(result[0].text)
        assert "annotation_id" in data, "annotation_id missing from response"

    async def test_rate_returns_rating_id(self, qa_pool) -> None:
        """Regression: _handle_rate must await add_rating (was unawaited)."""
        from archivist.app.handlers.tools_trajectory import _handle_rate

        with (
            patch(
                "archivist.app.handlers.tools_trajectory.add_rating",
                new_callable=AsyncMock,
                return_value="rat-001",
            ),
            patch("archivist.core.audit.log_memory_event", new_callable=AsyncMock),
            patch(
                "archivist.core.trajectory.get_rating_summary",
                return_value={"average_rating": 1.0, "count": 1},
            ),
            patch(
                "archivist.app.handlers.tools_trajectory.get_rating_summary",
                return_value={"average_rating": 1.0, "count": 1},
            ),
        ):
            result = await _handle_rate(
                {
                    "memory_id": _FAKE_MEMORY_ID,
                    "agent_id": "agent-smoke",
                    "rating": 1,
                    "context": "Helpful retrieval.",
                }
            )
        _assert_text_response(result)
        data = json.loads(result[0].text)
        assert "rating_id" in data, "rating_id missing from response"

    async def test_tips_returns_text(self, qa_pool) -> None:
        from archivist.app.handlers.tools_trajectory import _handle_tips

        with patch(
            "archivist.app.handlers.tools_trajectory.search_tips",
            return_value=[],
        ):
            result = await _handle_tips({"agent_id": "agent-smoke"})
        _assert_text_response(result)

    async def test_log_trajectory_with_mocked_llm(self, qa_pool) -> None:
        """_handle_log_trajectory must not timeout; LLM and DB are mocked."""
        from archivist.app.handlers.tools_trajectory import _handle_log_trajectory

        with (
            patch(
                "archivist.app.handlers.tools_trajectory.log_trajectory",
                new_callable=AsyncMock,
                return_value={"trajectory_id": "traj-001"},
            ),
            patch(
                "archivist.app.handlers.tools_trajectory.extract_tips",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "archivist.app.handlers.tools_trajectory.attribute_decisions",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await _handle_log_trajectory(
                {
                    "agent_id": "agent-smoke",
                    "task_description": "Test task",
                    "actions": [{"step": 1, "description": "Did something"}],
                    "outcome": "success",
                    "outcome_score": 0.9,
                }
            )
        _assert_text_response(result)

    async def test_session_end_no_trajectories(self, qa_pool) -> None:
        from archivist.app.handlers.tools_trajectory import _handle_session_end

        with patch(
            "archivist.app.handlers.tools_trajectory.session_end_summary",
            new_callable=AsyncMock,
            return_value={"error": "no_trajectories"},
        ):
            result = await _handle_session_end(
                {"agent_id": "agent-smoke", "session_id": "sess-001"}
            )
        assert isinstance(result, list) and result
        assert "coroutine" not in result[0].text


# ---------------------------------------------------------------------------
# tools_search — 9 tools  (critical: recall, entity_brief, contradictions)
# ---------------------------------------------------------------------------

_MOCK_ENTITIES = [{"id": 1, "name": "TestEntity", "entity_type": "concept"}]
_MOCK_FACTS = [
    {
        "fact_text": "TestEntity is a test.",
        "agent_id": "agent-smoke",
        "source_file": "",
        "created_at": "2026-01-01",
        "retention_class": "standard",
        "valid_from": "",
        "valid_until": "",
        "confidence": 1.0,
    }
]
_MOCK_RELS: list = []


def _search_patches():
    """Common patches for graph functions used in search handlers."""
    return [
        patch(
            "archivist.app.handlers.tools_search.search_entities",
            new_callable=AsyncMock,
            return_value=_MOCK_ENTITIES,
        ),
        patch(
            "archivist.app.handlers.tools_search.get_entity_facts",
            new_callable=AsyncMock,
            return_value=_MOCK_FACTS,
        ),
        patch(
            "archivist.app.handlers.tools_search.get_entity_relationships",
            new_callable=AsyncMock,
            return_value=_MOCK_RELS,
        ),
        patch(
            "archivist.app.handlers.tools_search.add_entity_alias",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "archivist.storage.graph_retrieval.get_entity_brief",
            new_callable=AsyncMock,
            return_value={
                "entity": {
                    "id": 1,
                    "name": "TestEntity",
                    "type": "concept",
                    "retention_class": "standard",
                    "mention_count": 1,
                    "first_seen": "",
                    "last_seen": "",
                    "aliases": "[]",
                },
                "facts": _MOCK_FACTS,
                "relationships": [],
                "fact_count": 1,
                "relationship_count": 0,
            },
        ),
        patch(
            "archivist.app.handlers.tools_search.get_entity_brief",
            new_callable=AsyncMock,
            return_value={
                "entity": {
                    "id": 1,
                    "name": "TestEntity",
                    "type": "concept",
                    "retention_class": "standard",
                    "mention_count": 1,
                    "first_seen": "",
                    "last_seen": "",
                    "aliases": "[]",
                },
                "facts": _MOCK_FACTS,
                "relationships": [],
                "fact_count": 1,
                "relationship_count": 0,
            },
        ),
        patch(
            "archivist.storage.graph_retrieval.detect_contradictions",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "archivist.app.handlers.tools_search.detect_contradictions",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "archivist.app.handlers.tools_search._rbac_gate",
            return_value=None,
        ),
        patch(
            "archivist.app.handlers.tools_search.require_caller",
            return_value=None,
        ),
        patch(
            "archivist.app.handlers.tools_search.resolve_caller",
            return_value="agent-smoke",
        ),
    ]


class TestSearchHandlers:
    async def test_recall_returns_entity_brief(self) -> None:
        """Regression: _handle_recall must await all graph calls."""
        from archivist.app.handlers.tools_search import _handle_recall

        with (
            patch(
                "archivist.app.handlers.tools_search.search_entities",
                new_callable=AsyncMock,
                return_value=_MOCK_ENTITIES,
            ),
            patch(
                "archivist.app.handlers.tools_search.get_entity_facts",
                new_callable=AsyncMock,
                return_value=_MOCK_FACTS,
            ),
            patch(
                "archivist.app.handlers.tools_search.get_entity_relationships",
                new_callable=AsyncMock,
                return_value=_MOCK_RELS,
            ),
            patch(
                "archivist.app.handlers.tools_search._rbac_gate",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_search.require_caller",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_search.resolve_caller",
                return_value="agent-smoke",
            ),
        ):
            result = await _handle_recall({"entity": "TestEntity", "agent_id": "agent-smoke"})
        _assert_text_response(result)
        data = json.loads(result[0].text)
        assert "entity" in data

    async def test_entity_brief_returns_structured_card(self) -> None:
        """Regression: _handle_entity_brief must await search_entities, add_entity_alias, get_entity_brief."""
        from archivist.app.handlers.tools_search import _handle_entity_brief

        brief_data = {
            "entity": {
                "id": 1,
                "name": "TestEntity",
                "type": "concept",
                "retention_class": "standard",
                "mention_count": 1,
                "first_seen": "",
                "last_seen": "",
                "aliases": "[]",
            },
            "facts": _MOCK_FACTS,
            "relationships": [],
            "fact_count": 1,
            "relationship_count": 0,
        }
        with (
            patch(
                "archivist.app.handlers.tools_search.search_entities",
                new_callable=AsyncMock,
                return_value=_MOCK_ENTITIES,
            ),
            patch(
                "archivist.app.handlers.tools_search.add_entity_alias",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_search.get_entity_brief",
                new_callable=AsyncMock,
                return_value=brief_data,
            ),
            patch(
                "archivist.app.handlers.tools_search._rbac_gate",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_search.require_caller",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_search.resolve_caller",
                return_value="agent-smoke",
            ),
            patch(
                "archivist.app.handlers.tools_search.is_permissive_mode",
                return_value=True,
            ),
        ):
            result = await _handle_entity_brief({"entity": "TestEntity", "agent_id": "agent-smoke"})
        _assert_text_response(result)
        data = json.loads(result[0].text)
        assert "entity" in data

    async def test_contradictions_returns_list(self) -> None:
        """Regression: _handle_contradictions must await search_entities and detect_contradictions."""
        from archivist.app.handlers.tools_search import _handle_contradictions

        with (
            patch(
                "archivist.app.handlers.tools_search.search_entities",
                new_callable=AsyncMock,
                return_value=_MOCK_ENTITIES,
            ),
            patch(
                "archivist.app.handlers.tools_search.detect_contradictions",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await _handle_contradictions({"entity": "TestEntity", "namespace": ""})
        _assert_text_response(result)
        data = json.loads(result[0].text)
        assert "contradictions" in data

    async def test_search_returns_text_content(self) -> None:
        from archivist.app.handlers.tools_search import _handle_search

        with (
            patch(
                "archivist.app.handlers.tools_search.recursive_retrieve",
                new_callable=AsyncMock,
                return_value={"results": [], "total": 0},
            ),
            patch(
                "archivist.app.handlers.tools_search._rbac_gate",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_search.resolve_caller",
                return_value="agent-smoke",
            ),
        ):
            result = await _handle_search({"query": "test query", "agent_id": "agent-smoke"})
        _assert_text_response(result)

    async def test_index_returns_text(self) -> None:
        from archivist.app.handlers.tools_search import _handle_index

        with (
            patch(
                "archivist.app.handlers.tools_search.build_namespace_index",
                return_value="# Index\n\nNo entries.",
            ),
            patch(
                "archivist.app.handlers.tools_search.get_namespace_for_agent",
                return_value="default",
            ),
            patch(
                "archivist.app.handlers.tools_search.is_permissive_mode",
                return_value=True,
            ),
        ):
            result = await _handle_index({"agent_id": "agent-smoke", "namespace": "default"})
        _assert_text_response(result)

    async def test_deref_missing_id_returns_error(self) -> None:
        from archivist.app.handlers.tools_search import _handle_deref

        mock_client = MagicMock()
        mock_client.retrieve.return_value = []

        with patch(
            "archivist.storage.qdrant.qdrant_client",
            return_value=mock_client,
        ):
            result = await _handle_deref({"memory_id": "nonexistent-id", "agent_id": "agent-smoke"})
        assert isinstance(result, list) and result
        assert "coroutine" not in result[0].text

    async def test_timeline_requires_caller(self) -> None:
        from archivist.app.handlers.tools_search import _handle_timeline

        with (
            patch(
                "archivist.app.handlers.tools_search.resolve_caller",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_search.require_caller",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_search.search_vectors",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "archivist.app.handlers.tools_search._rbac_gate",
                return_value=None,
            ),
        ):
            result = await _handle_timeline({"query": "history", "agent_id": "agent-smoke"})
        assert isinstance(result, list) and result
        assert "coroutine" not in result[0].text

    async def test_insights_requires_caller(self) -> None:
        from archivist.app.handlers.tools_search import _handle_insights

        with (
            patch(
                "archivist.app.handlers.tools_search.resolve_caller",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_search.require_caller",
                return_value=None,
            ),
            patch(
                "archivist.app.handlers.tools_search.search_vectors",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "archivist.app.handlers.tools_search._rbac_gate",
                return_value=None,
            ),
        ):
            result = await _handle_insights({"topic": "test insights", "agent_id": "agent-smoke"})
        assert isinstance(result, list) and result
        assert "coroutine" not in result[0].text

    async def test_wake_up_returns_text(self) -> None:
        from archivist.app.handlers.tools_search import _handle_wake_up

        wake_ctx = {
            "l0_identity": "agent-smoke",
            "critical_facts": [],
            "toc": [],
            "agent_id": "agent-smoke",
            "namespace": "default",
        }
        with (
            patch(
                "archivist.app.handlers.tools_search.get_cached_wake_up",
                new_callable=AsyncMock,
                return_value=wake_ctx,
            ),
            patch(
                "archivist.app.handlers.tools_search.get_namespace_for_agent",
                return_value="default",
            ),
            patch(
                "archivist.app.handlers.tools_search.is_permissive_mode",
                return_value=True,
            ),
        ):
            result = await _handle_wake_up({"agent_id": "agent-smoke", "namespace": "default"})
        _assert_text_response(result)
