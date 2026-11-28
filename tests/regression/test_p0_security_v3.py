"""Regression tests for INIT-023/SPEC-003 P0 security fixes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from archivist.app.main import ArchivistAuthMiddleware
from archivist.core import rbac as rbac_mod
from archivist.core.config import ArchivistSettings
from archivist.core.rbac import (
    NamespaceConfig,
    RBACConfig,
    can_read_agent_memory,
    check_access,
)

pytestmark = [pytest.mark.regression]


class TestLoopbackDefaultBind:
    """P0-3: default host bind must be 127.0.0.1; Dockerfile opts into 0.0.0.0."""

    def test_default_archivist_host_is_loopback(self):
        assert ArchivistSettings.model_fields["archivist_host"].default == "127.0.0.1"

    def test_dockerfile_sets_archivist_host_explicitly(self):
        dockerfile = Path(__file__).resolve().parents[2] / "Dockerfile"
        content = dockerfile.read_text(encoding="utf-8")
        assert "ENV ARCHIVIST_HOST=0.0.0.0" in content


class TestAdminRouteScopeSplit:
    """P0-4: /admin/* and /debug/config must require ARCHIVIST_ADMIN_API_KEY when set."""

    def _build_client(self) -> TestClient:
        async def _ok(request):
            return JSONResponse({"ok": True})

        app = Starlette(
            routes=[
                Route("/health", _ok),
                Route("/mcp", _ok),
                Route("/admin/stats", _ok),
                Route("/debug/config", _ok),
            ]
        )
        app.add_middleware(ArchivistAuthMiddleware)
        return TestClient(app)

    def test_standard_api_key_rejected_on_admin_routes_when_admin_key_set(self):
        client = self._build_client()
        with (
            patch("archivist.app.main.ARCHIVIST_API_KEY", "std-secret-key"),
            patch("archivist.app.main.ARCHIVIST_ADMIN_API_KEY", "admin-secret-key"),
        ):
            # Standard key works on /mcp
            assert (
                client.get("/mcp", headers={"Authorization": "Bearer std-secret-key"}).status_code
                == 200
            )
            # Standard key is FORBIDDEN (403) on /admin/* and /debug/config
            r_admin = client.get("/admin/stats", headers={"Authorization": "Bearer std-secret-key"})
            assert r_admin.status_code == 403
            r_debug = client.get(
                "/debug/config", headers={"Authorization": "Bearer std-secret-key"}
            )
            assert r_debug.status_code == 403
            # Admin key succeeds on /admin/* and /debug/config
            assert (
                client.get(
                    "/admin/stats", headers={"Authorization": "Bearer admin-secret-key"}
                ).status_code
                == 200
            )
            assert (
                client.get(
                    "/debug/config", headers={"Authorization": "Bearer admin-secret-key"}
                ).status_code
                == 200
            )
            # Wrong/missing key returns 401
            assert client.get("/admin/stats").status_code == 401
            assert (
                client.get(
                    "/admin/stats", headers={"Authorization": "Bearer wrong-key"}
                ).status_code
                == 401
            )

    def test_admin_key_only_protects_admin_routes_even_without_std_key(self):
        client = self._build_client()
        with (
            patch("archivist.app.main.ARCHIVIST_API_KEY", ""),
            patch("archivist.app.main.ARCHIVIST_ADMIN_API_KEY", "admin-only-key"),
        ):
            assert client.get("/admin/stats").status_code == 401
            assert (
                client.get(
                    "/admin/stats", headers={"Authorization": "Bearer admin-only-key"}
                ).status_code
                == 200
            )

    def test_trailing_slash_debug_config_enforces_admin_scope(self):
        """AUDIT-MIN-1: /debug/config/ must be treated as an admin path (no 307 bypass)."""
        client = self._build_client()
        with (
            patch("archivist.app.main.ARCHIVIST_API_KEY", "std-secret-key"),
            patch("archivist.app.main.ARCHIVIST_ADMIN_API_KEY", "admin-secret-key"),
        ):
            r_unauth = client.get("/debug/config/", follow_redirects=False)
            assert r_unauth.status_code == 401
            r_std = client.get(
                "/debug/config/",
                headers={"Authorization": "Bearer std-secret-key"},
                follow_redirects=False,
            )
            assert r_std.status_code == 403

    @pytest.mark.asyncio
    async def test_non_ascii_header_bytes_return_401_not_500(self):
        """AUDIT-MAJ-4: raw non-ASCII latin-1 header bytes must return 401 cleanly, never TypeError 500."""

        async def _ok(request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/mcp", _ok), Route("/admin/stats", _ok)])
        app.add_middleware(ArchivistAuthMiddleware)

        async def _send_raw(path: str, headers: list[tuple[bytes, bytes]]) -> int:
            status_code = 0

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = int(message["status"])

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": b"",
                "headers": headers,
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8100),
            }
            await app(scope, receive, send)
            return status_code

        with (
            patch("archivist.app.main.ARCHIVIST_API_KEY", "std-secret-key"),
            patch("archivist.app.main.ARCHIVIST_ADMIN_API_KEY", "admin-secret-key"),
        ):
            assert await _send_raw("/mcp", [(b"x-api-key", b"caf\xe9")]) == 401
            assert await _send_raw("/mcp", [(b"authorization", b"Bearer caf\xe9")]) == 401
            assert await _send_raw("/admin/stats", [(b"x-api-key", b"caf\xe9")]) == 401

    def test_startup_warns_when_admin_key_set_but_mcp_unprotected(self, caplog):
        """AUDIT-MIN-1: setting only ARCHIVIST_ADMIN_API_KEY must still warn that /mcp is open."""
        import logging

        from archivist.app.main import _warn_if_no_access_control

        with (
            patch("archivist.app.main.ARCHIVIST_API_KEY", ""),
            patch("archivist.app.main.ARCHIVIST_ADMIN_API_KEY", "admin-only-key"),
            patch("archivist.app.main.NAMESPACES_CONFIG_PATH", ""),
            caplog.at_level(logging.CRITICAL, logger="archivist.server"),
        ):
            _warn_if_no_access_control()
        assert any("NO MCP ACCESS CONTROL" in rec.message for rec in caplog.records)


class TestRBACSecurityFixes:
    """P0-5 & P0-6: similar_namespaces enumeration oracle and empty target_agent_id bypass."""

    def test_similar_namespaces_never_leaks_foreign_tenant_names(self, monkeypatch):
        cfg = RBACConfig(
            namespaces={
                "tenant_alpha_prod": NamespaceConfig(
                    id="tenant_alpha_prod", read=["agent_a"], write=["agent_a"]
                ),
                "tenant_secret_merger": NamespaceConfig(
                    id="tenant_secret_merger", read=["agent_b"], write=["agent_b"]
                ),
                "tenant_secret_merge": NamespaceConfig(
                    id="tenant_secret_merge", read=["agent_b"], write=["agent_b"]
                ),
            },
        )
        monkeypatch.setattr(rbac_mod, "_config", cfg)
        monkeypatch.setattr(rbac_mod, "_permissive_fallback", False)

        result = check_access("agent_a", "read", "tenant_secret_mergr")
        assert result.allowed is False
        assert "tenant_secret_merger" not in result.similar_namespaces
        assert "tenant_secret_merge" not in result.similar_namespaces
        assert "tenant_secret_merger" not in (result.hint or "")
        assert "tenant_secret_merge" not in (result.hint or "")

        # Typo against caller's OWN accessible namespace IS suggested
        own_typo = check_access("agent_a", "read", "tenant_alpha_pro")
        assert own_typo.allowed is False
        assert "tenant_alpha_prod" in own_typo.similar_namespaces

    def test_can_read_agent_memory_fails_closed_on_empty_target_under_active_rbac(
        self, monkeypatch
    ):
        cfg = RBACConfig(
            namespaces={
                "team_a": NamespaceConfig(id="team_a", read=["agent_a"], write=["agent_a"]),
                "*": NamespaceConfig(id="*", read=["super_admin"], write=["super_admin"]),
            },
        )
        monkeypatch.setattr(rbac_mod, "_config", cfg)
        monkeypatch.setattr(rbac_mod, "_permissive_fallback", False)

        assert can_read_agent_memory("agent_a", "") is False
        assert can_read_agent_memory("agent_a", "   ") is False
        assert can_read_agent_memory("super_admin", "") is True
