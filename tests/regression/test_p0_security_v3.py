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
