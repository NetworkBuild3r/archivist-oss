import pytest

pytestmark = [pytest.mark.integration]
"""Phase 4 tests — skill registry, lessons, events, health scoring."""


async def test_register_and_find_skill(async_pool):
    from skills import find_skill, list_skills, register_skill

    result = await register_skill(
        name="web_search",
        provider="openai",
        version="1.0.0",
        description="Search the web",
        registered_by="agent-alpha",
    )
    assert result["action"] == "registered"
    assert result["version"] == "1.0.0"
    skill_id = result["skill_id"]

    found = await find_skill("web_search", "openai")
    assert found is not None
    assert found["id"] == skill_id
    assert found["current_version"] == "1.0.0"

    result2 = await register_skill(
        name="web_search",
        provider="openai",
        version="1.1.0",
        registered_by="agent-alpha",
    )
    assert result2["action"] == "updated"
    assert result2["version"] == "1.1.0"

    all_skills = await list_skills()
    assert len(all_skills) >= 1


async def test_lessons_crud(async_pool):
    from skills import add_lesson, get_lessons, register_skill

    reg = await register_skill(name="code_exec", provider="internal", registered_by="a1")
    sid = reg["skill_id"]

    lid = await add_lesson(
        skill_id=sid,
        title="Timeout on large inputs",
        content="code_exec times out on inputs >50KB. Chunk first.",
        lesson_type="failure_mode",
        skill_version="1.0.0",
        agent_id="a1",
    )
    assert lid

    lessons = await get_lessons(sid)
    assert len(lessons) == 1
    assert lessons[0]["title"] == "Timeout on large inputs"
    assert lessons[0]["lesson_type"] == "failure_mode"

    await add_lesson(
        sid, "Use JSON input", "Always pass JSON, not raw text.", "best_practice", agent_id="a2"
    )
    lessons2 = await get_lessons(sid, lesson_type="failure_mode")
    assert len(lessons2) == 1


async def test_skill_events_and_health(async_pool):
    from skills import get_skill_health, log_skill_event, register_skill

    reg = await register_skill(name="calculator", provider="builtin", registered_by="a1")
    sid = reg["skill_id"]

    for _ in range(7):
        await log_skill_event(sid, "a1", "success", duration_ms=100)
    for _ in range(3):
        await log_skill_event(sid, "a1", "failure", error_message="division by zero")

    health = await get_skill_health(sid)
    assert health["total_events"] == 10
    assert health["successes"] == 7
    assert health["failures"] == 3
    assert health["success_rate"] == 0.7
    assert health["health"] == "warning"
    assert health["last_failure"] is not None


async def test_health_grades(async_pool):
    from skills import get_skill_health, log_skill_event, register_skill, update_skill_status

    reg = await register_skill(name="good_tool", provider="x", registered_by="a1")
    for _ in range(10):
        await log_skill_event(reg["skill_id"], "a1", "success")
    h = await get_skill_health(reg["skill_id"])
    assert h["health"] == "healthy"

    reg2 = await register_skill(name="bad_tool", provider="x", registered_by="a1")
    for _ in range(3):
        await log_skill_event(reg2["skill_id"], "a1", "success")
    for _ in range(7):
        await log_skill_event(reg2["skill_id"], "a1", "failure")
    h2 = await get_skill_health(reg2["skill_id"])
    assert h2["health"] == "degraded"

    await update_skill_status(reg2["skill_id"], "deprecated")
    h3 = await get_skill_health(reg2["skill_id"])
    assert h3["health"] == "deprecated"


async def test_version_tracking(async_pool):
    from skills import get_skill_health, record_version, register_skill

    reg = await register_skill(
        name="api_tool", provider="acme", version="1.0.0", registered_by="a1"
    )
    sid = reg["skill_id"]

    await record_version(sid, "1.1.0", changelog="Added caching", reported_by="a1")
    await record_version(
        sid, "2.0.0", changelog="New API", breaking_changes="Removed v1 endpoints", reported_by="a1"
    )

    health = await get_skill_health(sid)
    assert health["current_version"] == "2.0.0"
    assert len(health["recent_versions"]) >= 2
    v2 = next(v for v in health["recent_versions"] if v["version"] == "2.0.0")
    assert "Removed v1" in v2["breaking_changes"]
