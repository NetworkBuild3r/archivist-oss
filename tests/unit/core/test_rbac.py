"""Unit tests for RBAC — check_access, namespace resolution, filtering."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.rbac]


class TestCheckAccess:
    def test_read_own_namespace(self, rbac_config):
        pol = rbac_config.check_access("chief", "read", "chief")
        assert pol.allowed is True

    def test_write_own_namespace(self, rbac_config):
        pol = rbac_config.check_access("chief", "write", "chief")
        assert pol.allowed is True

    def test_read_other_namespace_allowed(self, rbac_config):
        pol = rbac_config.check_access("chief", "read", "pipeline")
        assert pol.allowed is True

    def test_write_other_namespace_denied(self, rbac_config):
        pol = rbac_config.check_access("chief", "write", "pipeline")
        assert pol.allowed is False

    def test_unknown_namespace_denied(self, rbac_config):
        pol = rbac_config.check_access("chief", "read", "nonexistent")
        assert pol.allowed is False
        assert "Unknown namespace" in pol.reason

    def test_all_wildcard_read(self, rbac_config):
        pol = rbac_config.check_access("random_agent", "read", "shared")
        assert pol.allowed is True

    def test_all_wildcard_write(self, rbac_config):
        pol = rbac_config.check_access("random_agent", "write", "shared")
        assert pol.allowed is True

    def test_deployer_write_by_argo(self, rbac_config):
        pol = rbac_config.check_access("argo", "write", "deployer")
        assert pol.allowed is True

    def test_deployer_write_denied_for_gitbob(self, rbac_config):
        pol = rbac_config.check_access("gitbob", "write", "deployer")
        assert pol.allowed is False


class TestNamespaceResolution:
    def test_agent_namespace_mapping(self, rbac_config):
        assert rbac_config.get_namespace_for_agent("chief") == "chief"
        assert rbac_config.get_namespace_for_agent("gitbob") == "pipeline"
        assert rbac_config.get_namespace_for_agent("argo") == "deployer"

    def test_unknown_agent_falls_back(self, rbac_config):
        ns = rbac_config.get_namespace_for_agent("unknown_agent")
        assert isinstance(ns, str)


class TestFilterAgents:
    def test_filter_allowed_denied(self, rbac_config):
        allowed, denied = rbac_config.filter_agents_for_read(
            "chief", ["gitbob", "argo", "kubekate"]
        )
        assert "gitbob" in allowed
        assert "argo" in allowed

    def test_filter_empty_ids(self, rbac_config):
        allowed, denied = rbac_config.filter_agents_for_read("chief", ["", ""])
        assert allowed == []
        assert denied == []


class TestListAccessible:
    def test_chief_sees_multiple_namespaces(self, rbac_config):
        nss = rbac_config.list_accessible_namespaces("chief")
        ns_ids = [n["namespace"] for n in nss]
        assert "chief" in ns_ids
        assert "pipeline" in ns_ids
        assert "deployer" in ns_ids
        assert "shared" in ns_ids

    def test_gitbob_cannot_see_deployer(self, rbac_config):
        nss = rbac_config.list_accessible_namespaces("gitbob")
        ns_ids = [n["namespace"] for n in nss]
        assert "pipeline" in ns_ids
        assert "deployer" not in ns_ids

    def test_namespace_flags(self, rbac_config):
        nss = rbac_config.list_accessible_namespaces("chief")
        chief_ns = next(n for n in nss if n["namespace"] == "chief")
        assert chief_ns["can_read"] is True
        assert chief_ns["can_write"] is True

        pipeline_ns = next(n for n in nss if n["namespace"] == "pipeline")
        assert pipeline_ns["can_read"] is True
        assert pipeline_ns["can_write"] is False


class TestPermissiveMode:
    def test_permissive_when_path_empty(self, monkeypatch):
        import rbac

        monkeypatch.setenv("NAMESPACES_CONFIG_PATH", "")
        rbac._config = None
        rbac._permissive_fallback = False
        rbac.load_config("")

        assert rbac.is_permissive_mode() is True
        pol = rbac.check_access("anyone", "write", "anything")
        assert pol.allowed is True

    def test_permissive_when_config_missing(self, tmp_path, monkeypatch):
        import rbac

        rbac._config = None
        rbac._permissive_fallback = False
        rbac.load_config(str(tmp_path / "missing.yaml"))

        assert rbac.is_permissive_mode() is True
        pol = rbac.check_access("anyone", "write", "anything")
        assert pol.allowed is True
