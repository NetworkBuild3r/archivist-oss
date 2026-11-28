"""Integration tests for SPEC-008 5-tool coding flow, 5 Resources, 3 Prompts & SPEC-005 docs."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from archivist.app.handlers._registry import dispatch_tool
from archivist.app.handlers.scope_resolver import set_session_roots
from archivist.app.handlers.tools_docs import _doc_path, read_bundled_doc
from archivist.app.handlers.v3_coding import set_local_store
from archivist.app.handlers.v3_resources import (
    PROMPTS,
    get_prompt_by_name,
    get_resource_templates,
    get_static_resources,
    read_resource_uri,
)
from archivist.storage.local_store import LocalMemoryStore

pytestmark = [pytest.mark.integration, pytest.mark.mcp]


@pytest.fixture
def local_store_fixture(tmp_path: Path, monkeypatch):
    import archivist.core.config as config

    monkeypatch.setattr(config, "TOOL_PROFILE", "coding")
    store = LocalMemoryStore(db_path=tmp_path / "memory.db", offline_deterministic=True)
    set_local_store(store)
    try:
        yield store
    finally:
        set_local_store(None)
        store.close()


class TestV3CodingEndToEnd:
    @pytest.mark.asyncio
    async def test_five_tool_round_trip_with_git_roots_auto_scope(
        self, local_store_fixture: LocalMemoryStore, tmp_path: Path
    ):
        # Create a temp git repo with SSH origin remote and bind via MCP roots/list
        repo_dir = tmp_path / "demo-repo"
        repo_dir.mkdir()
        git_bin = shutil.which("git")
        assert git_bin is not None
        subprocess.run([git_bin, "init", str(repo_dir)], check=True, capture_output=True)
        subprocess.run(
            [
                git_bin,
                "-C",
                str(repo_dir),
                "remote",
                "add",
                "origin",
                "git@github.com:Acme/DemoApp.git",
            ],
            check=True,
            capture_output=True,
        )

        token = set_session_roots([repo_dir.as_uri()])
        try:
            # 1. memory_save with ZERO scope/agent_id args (auto-scoped + secret redacted)
            save_res = await dispatch_tool(
                "memory_save",
                {
                    "content": "Always run ruff check before pushing; key AKIAIOSFODNN7EXAMPLE",
                    "kind": "decision",
                    "tags": ["ci", "lint"],
                },
            )
            saved = json.loads(save_res[0].text)
            assert saved["scope"] == "repo:github.com/acme/demoapp"
            assert saved["redacted_count"] == 1
            assert saved["quarantined"] is False
            mid = saved["memory_id"]

            # 2. memory_search
            search_res = await dispatch_tool(
                "memory_search", {"query": "ruff check before pushing"}
            )
            search_data = json.loads(search_res[0].text)
            assert search_data["count"] >= 1
            assert search_data["results"][0]["id"] == mid
            assert "AKIAIOSFODNN7EXAMPLE" not in search_data["results"][0]["summary"]

            # 3. memory_context (framed <archivist_memory> output)
            ctx_res = await dispatch_tool(
                "memory_context", {"query": "ruff check before pushing", "max_tokens": 2000}
            )
            ctx_data = json.loads(ctx_res[0].text)
            assert "<archivist_memory " in ctx_data["context"]
            assert 'trust_tier="T1_AGENT"' in ctx_data["context"]
            assert "AKIAIOSFODNN7EXAMPLE" not in ctx_data["context"]

            # 4. memory_inspect
            insp_res = await dispatch_tool("memory_inspect", {"memory_id": mid})
            insp_data = json.loads(insp_res[0].text)
            assert insp_data["found"] is True
            assert insp_data["memory"]["scope"] == "repo:github.com/acme/demoapp"

            # 5. All 5 MCP Resources
            static_uris = {str(r.uri) for r in get_static_resources()}
            assert "archivist://health" in static_uris
            assert "archivist://stats/savings" in static_uris
            assert "archivist://docs/skill" in static_uris
            template_uris = {t.uriTemplate for t in get_resource_templates()}
            assert "archivist://index/{scope}" in template_uris
            assert "archivist://{scope}/memory/{id}" in template_uris
            assert "archivist://docs/{topic}" in template_uris

            health_doc = json.loads(read_resource_uri("archivist://health"))
            assert health_doc["status"] == "ok"
            assert health_doc["active_memories"] == 1

            savings_doc = json.loads(read_resource_uri("archivist://stats/savings"))
            assert savings_doc["schema_tools"] == 5

            index_doc = json.loads(
                read_resource_uri("archivist://index/repo:github.com/acme/demoapp")
            )
            assert index_doc["total_shown"] == 1

            mem_doc = json.loads(
                read_resource_uri(f"archivist://repo:github.com/acme/demoapp/memory/{mid}")
            )
            assert mem_doc["id"] == mid

            skill_doc = read_resource_uri("archivist://docs/skill")
            assert len(skill_doc) > 100

            # 6. All 3 MCP Prompts
            assert {p.name for p in PROMPTS} == {
                "bootstrap-session",
                "save-session-lessons",
                "audit-contradictions",
            }
            for prompt_name in (
                "bootstrap-session",
                "save-session-lessons",
                "audit-contradictions",
            ):
                res_prompt = get_prompt_by_name(prompt_name, {})
                assert res_prompt.messages
                assert "repo:github.com/acme/demoapp" in res_prompt.messages[0].content.text

            # 7. memory_forget
            forget_res = await dispatch_tool("memory_forget", {"memory_id": mid, "mode": "delete"})
            forget_data = json.loads(forget_res[0].text)
            assert forget_data["affected"] == 1

            after_search = json.loads(
                (await dispatch_tool("memory_search", {"query": "ruff check"}))[0].text
            )
            assert after_search["count"] == 0
        finally:
            set_session_roots(())
            _ = token

    def test_bundled_docs_resolve_via_importlib_resources(self):
        """SPEC-005: tools_docs.py resolves bundled package docs, not broken src/archivist/docs/."""
        doc = _doc_path()
        assert doc.exists()
        assert "docs_data" in str(doc)
        assert read_bundled_doc("skill") is not None
        assert read_bundled_doc("reference") is not None
