"""Hard CI schema-token budget gate & scope canonicalization tests (INIT-024/SPEC-008)."""

from __future__ import annotations

import json

import pytest
import tiktoken

from archivist.app.handlers._registry import CODING_TOOL_NAMES, dispatch_tool, get_all_tools
from archivist.app.handlers.scope_resolver import (
    GLOBAL_SCOPE,
    canonicalize_git_remote,
    resolve_read_scopes,
    resolve_write_scope,
)

pytestmark = [pytest.mark.unit, pytest.mark.mcp]


class TestV3CodingSchemaBudget:
    def test_default_coding_profile_has_exactly_5_tools_under_900_tokens(self, monkeypatch):
        import archivist.core.config as config

        monkeypatch.setattr(config, "TOOL_PROFILE", "coding")
        tools = get_all_tools()
        assert len(tools) == 5
        assert {t.name for t in tools} == CODING_TOOL_NAMES
        assert {t.name for t in tools} == {
            "memory_context",
            "memory_save",
            "memory_search",
            "memory_forget",
            "memory_inspect",
        }

        serialized = json.dumps([t.model_dump(exclude_none=True) for t in tools])
        for enc_name in ("cl100k_base", "o200k_base"):
            enc = tiktoken.get_encoding(enc_name)
            token_count = len(enc.encode(serialized))
            assert token_count < 900, (
                f"5-tool coding schema exceeded hard 900-token CI budget in {enc_name}: "
                f"{token_count} tokens"
            )

    def test_coding_tools_have_zero_required_agent_or_namespace_args(self, monkeypatch):
        import archivist.core.config as config

        monkeypatch.setattr(config, "TOOL_PROFILE", "coding")
        for tool in get_all_tools():
            required = set(tool.inputSchema.get("required") or [])
            assert "agent_id" not in required
            assert "namespace" not in required
            assert "caller_agent_id" not in required

    @pytest.mark.asyncio
    async def test_non_coding_tools_fail_closed_under_default_profile(self, monkeypatch):
        import archivist.core.config as config

        monkeypatch.setattr(config, "TOOL_PROFILE", "coding")
        for blocked_name in (
            "archivist_store",
            "archivist_share_propose",
            "archivist_checkpoint_save",
        ):
            res = await dispatch_tool(blocked_name, {})
            payload = json.loads(res[0].text)
            assert "error" in payload
            assert "not available" in payload["error"]
            assert "coding" in payload["error"]


class TestCanonicalGitScopeResolver:
    @pytest.mark.parametrize(
        "remote_url",
        [
            "git@github.com:Org/Repo.git",
            "https://github.com/org/repo",
            "https://github.com/Org/Repo.git/",
            "https://ghp_secretTokenValue@github.com/Org/Repo.git",
            "ssh://git@github.com:22/Org/Repo.git",
        ],
    )
    def test_ssh_and_https_remotes_canonicalize_identically(self, remote_url: str):
        assert canonicalize_git_remote(remote_url) == "repo:github.com/org/repo"

    def test_read_scopes_blend_repo_with_user_global(self):
        scopes = resolve_read_scopes("repo:github.com/org/repo")
        assert scopes == ["repo:github.com/org/repo", GLOBAL_SCOPE]
        assert resolve_read_scopes(GLOBAL_SCOPE) == [GLOBAL_SCOPE]
        assert resolve_write_scope("repo:github.com/org/repo") == "repo:github.com/org/repo"
