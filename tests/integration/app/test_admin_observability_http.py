"""HTTP smoke for Diff #8 admin lineage/audit + UI mount.

INIT-013/SPEC-002 — route wiring + RBAC/auth.
INIT-013/SPEC-004 — billboard smoke / e2e (HTTP-level; no Playwright).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

pytestmark = [pytest.mark.integration]


async def _noop_startup():
    return None


@pytest.fixture
def client(monkeypatch):
    import archivist.app.main as main

    monkeypatch.setattr(main, "_startup", _noop_startup)
    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "", raising=False)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def authed_client(monkeypatch):
    import archivist.app.main as main

    monkeypatch.setattr(main, "_startup", _noop_startup)
    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "test-secret-key", raising=False)
    with TestClient(main.app) as c:
        yield c


class TestAdminLineageAuditHttp:
    def test_lineage_requires_id(self, client):
        resp = client.get("/admin/lineage")
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_arguments"

    def test_lineage_happy_path_permissive(self, client, monkeypatch):
        monkeypatch.setattr(
            "archivist.app.admin_observability.is_permissive_mode",
            lambda: True,
        )
        fake = {
            "resource_type": "memory",
            "resource_id": "mem-1",
            "namespace": "ns-a",
            "edge_count": 0,
            "edges": [],
            "sources": [],
        }
        with patch(
            "archivist.app.lineage.build_memory_lineage",
            new_callable=AsyncMock,
            return_value=dict(fake),
        ):
            with patch(
                "archivist.app.lineage.validate_memory_id",
                return_value="mem-1",
            ):
                resp = client.get("/admin/lineage", params={"memory_id": "mem-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["resource_id"] == "mem-1"
        assert body["namespace"] == "ns-a"

    def test_lineage_rbac_denied(self, client, monkeypatch):
        monkeypatch.setattr(
            "archivist.app.admin_observability.is_permissive_mode",
            lambda: False,
        )

        class _Denied:
            allowed = False
            reason = "nope"
            hint = "ask"
            similar_namespaces = []
            next_steps = []

        with patch(
            "archivist.app.lineage.validate_memory_id",
            return_value="mem-1",
        ):
            with patch(
                "archivist.app.lineage.resolve_memory_namespace",
                new_callable=AsyncMock,
                return_value="secret-ns",
            ):
                with patch(
                    "archivist.app.admin_observability.check_access",
                    return_value=_Denied(),
                ):
                    resp = client.get(
                        "/admin/lineage",
                        params={"memory_id": "mem-1", "agent_id": "agent-x"},
                    )
        assert resp.status_code == 403
        assert resp.json()["error"] == "access_denied"

    def test_audit_happy_path(self, client):
        entries = [{"event": "write", "memory_id": "m1"}]
        with patch(
            "archivist.core.audit.get_audit_trail",
            new_callable=AsyncMock,
            return_value=entries,
        ):
            resp = client.get("/admin/audit", params={"memory_id": "m1", "limit": "10"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["entries"] == entries

    def test_audit_by_agent(self, client):
        with patch(
            "archivist.core.audit.get_agent_activity",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_act:
            resp = client.get("/admin/audit", params={"agent_id": "agent-1"})
        assert resp.status_code == 200
        mock_act.assert_awaited()
        assert resp.json()["count"] == 0


class TestAdminUiMount:
    def test_ui_index_served(self, client):
        resp = client.get("/admin/ui/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Archivist" in resp.text
        assert "Observability" in resp.text
        assert 'id="panel-lineage"' in resp.text
        assert 'id="panel-audit"' in resp.text
        assert "app.js" in resp.text

    def test_ui_requires_api_key_when_configured(self, authed_client):
        denied = authed_client.get("/admin/ui/")
        assert denied.status_code == 401
        ok = authed_client.get(
            "/admin/ui/",
            headers={"X-API-Key": "test-secret-key"},
        )
        assert ok.status_code == 200

    def test_lineage_requires_api_key_when_configured(self, authed_client):
        denied = authed_client.get("/admin/lineage", params={"memory_id": "m"})
        assert denied.status_code == 401
