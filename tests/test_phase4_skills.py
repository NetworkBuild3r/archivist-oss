"""Phase 4 tests — skill registry, lessons, events, health scoring, memory_type routing."""

import sqlite3
import sys
import os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_test_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


_test_conn = None


def _patch_dbs():
    global _test_conn
    _test_conn = _make_test_db()
    import graph
    import skills as skills_mod
    graph._original_get_db = graph.get_db
    skills_mod._original_get_db = skills_mod.get_db
    graph.get_db = lambda: _test_conn
    skills_mod.get_db = lambda: _test_conn
    skills_mod._SCHEMA_APPLIED = False


def _unpatch_dbs():
    import graph
    import skills as skills_mod
    graph.get_db = graph._original_get_db
    skills_mod.get_db = skills_mod._original_get_db
    if _test_conn:
        _test_conn.close()


def test_register_and_find_skill():
    _patch_dbs()
    try:
        from skills import register_skill, find_skill, list_skills

        result = register_skill(
            name="web_search",
            provider="openai",
            version="1.0.0",
            description="Search the web",
            registered_by="agent-alpha",
        )
        assert result["action"] == "registered"
        assert result["version"] == "1.0.0"
        skill_id = result["skill_id"]

        found = find_skill("web_search", "openai")
        assert found is not None
        assert found["id"] == skill_id
        assert found["current_version"] == "1.0.0"

        result2 = register_skill(
            name="web_search",
            provider="openai",
            version="1.1.0",
            registered_by="agent-alpha",
        )
        assert result2["action"] == "updated"
        assert result2["version"] == "1.1.0"

        all_skills = list_skills()
        assert len(all_skills) >= 1
    finally:
        _unpatch_dbs()


def test_lessons_crud():
    _patch_dbs()
    try:
        from skills import register_skill, add_lesson, get_lessons

        reg = register_skill(name="code_exec", provider="internal", registered_by="a1")
        sid = reg["skill_id"]

        lid = add_lesson(
            skill_id=sid,
            title="Timeout on large inputs",
            content="code_exec times out on inputs >50KB. Chunk first.",
            lesson_type="failure_mode",
            skill_version="1.0.0",
            agent_id="a1",
        )
        assert lid

        lessons = get_lessons(sid)
        assert len(lessons) == 1
        assert lessons[0]["title"] == "Timeout on large inputs"
        assert lessons[0]["lesson_type"] == "failure_mode"

        add_lesson(sid, "Use JSON input", "Always pass JSON, not raw text.", "best_practice", agent_id="a2")
        lessons2 = get_lessons(sid, lesson_type="failure_mode")
        assert len(lessons2) == 1
    finally:
        _unpatch_dbs()


def test_skill_events_and_health():
    _patch_dbs()
    try:
        from skills import register_skill, log_skill_event, get_skill_health

        reg = register_skill(name="calculator", provider="builtin", registered_by="a1")
        sid = reg["skill_id"]

        for _ in range(7):
            log_skill_event(sid, "a1", "success", duration_ms=100)
        for _ in range(3):
            log_skill_event(sid, "a1", "failure", error_message="division by zero")

        health = get_skill_health(sid)
        assert health["total_events"] == 10
        assert health["successes"] == 7
        assert health["failures"] == 3
        assert health["success_rate"] == 0.7
        assert health["health"] == "warning"
        assert health["last_failure"] is not None
    finally:
        _unpatch_dbs()


def test_health_grades():
    _patch_dbs()
    try:
        from skills import register_skill, log_skill_event, get_skill_health, update_skill_status

        reg = register_skill(name="good_tool", provider="x", registered_by="a1")
        for _ in range(10):
            log_skill_event(reg["skill_id"], "a1", "success")
        h = get_skill_health(reg["skill_id"])
        assert h["health"] == "healthy"

        reg2 = register_skill(name="bad_tool", provider="x", registered_by="a1")
        for _ in range(3):
            log_skill_event(reg2["skill_id"], "a1", "success")
        for _ in range(7):
            log_skill_event(reg2["skill_id"], "a1", "failure")
        h2 = get_skill_health(reg2["skill_id"])
        assert h2["health"] == "degraded"

        update_skill_status(reg2["skill_id"], "deprecated")
        h3 = get_skill_health(reg2["skill_id"])
        assert h3["health"] == "deprecated"
    finally:
        _unpatch_dbs()


def test_version_tracking():
    _patch_dbs()
    try:
        from skills import register_skill, record_version, get_skill_health

        reg = register_skill(name="api_tool", provider="acme", version="1.0.0", registered_by="a1")
        sid = reg["skill_id"]

        record_version(sid, "1.1.0", changelog="Added caching", reported_by="a1")
        record_version(sid, "2.0.0", changelog="New API", breaking_changes="Removed v1 endpoints", reported_by="a1")

        health = get_skill_health(sid)
        assert health["current_version"] == "2.0.0"
        assert len(health["recent_versions"]) >= 2
        v2 = next(v for v in health["recent_versions"] if v["version"] == "2.0.0")
        assert "Removed v1" in v2["breaking_changes"]
    finally:
        _unpatch_dbs()
