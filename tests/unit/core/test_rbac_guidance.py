"""Unit tests for RBAC error guidance — similar namespace suggestions, next_steps,
and the universal get_help pointer in every denial response.

These tests verify the agent-facing error payloads so that when an agent makes a
bad namespace call it receives enough information to self-correct without human
intervention.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.rbac]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NAMESPACES_YAML = """
namespaces:
  - id: agents-nova
    read: [nova, all]
    write: [nova]
  - id: agents-athena
    read: [athena]
    write: [athena]
  - id: athena-identity
    read: [athena, nova]
    write: [athena]
  - id: shared
    read: [all]
    write: [nova, athena]
"""


@pytest.fixture(autouse=True)
def _reset_rbac(tmp_path):
    """Load a known RBAC config before each test and reset after."""
    import archivist.core.rbac as rbac

    yaml_path = tmp_path / "namespaces.yaml"
    yaml_path.write_text(_NAMESPACES_YAML)
    rbac.load_config(str(yaml_path))
    yield
    # Reset to permissive so other tests are not affected
    rbac.load_config("")


# ---------------------------------------------------------------------------
# _similar_namespaces
# ---------------------------------------------------------------------------


class TestSimilarNamespaces:
    def test_exact_typo_surfaces_best_match(self):
        from archivist.core.rbac import _similar_namespaces

        result = _similar_namespaces("agents-novia", ["agents-nova", "agents-athena", "shared"])
        assert "agents-nova" in result

    def test_prefix_match_surfaces_candidates(self):
        from archivist.core.rbac import _similar_namespaces

        result = _similar_namespaces("athena-ident", ["agents-nova", "athena-identity", "shared"])
        assert "athena-identity" in result

    def test_completely_unrelated_returns_empty(self):
        from archivist.core.rbac import _similar_namespaces

        result = _similar_namespaces("zzz-xyz-999", ["agents-nova", "agents-athena", "shared"])
        assert result == []

    def test_top_n_respected(self):
        from archivist.core.rbac import _similar_namespaces

        candidates = ["agents-nova", "agents-athena", "athena-identity", "shared"]
        result = _similar_namespaces("agents", candidates, top_n=2)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# check_access — unknown namespace
# ---------------------------------------------------------------------------


class TestCheckAccessUnknownNamespace:
    def test_unknown_namespace_returns_denied(self):
        from archivist.core.rbac import check_access

        policy = check_access("nova", "write", "agent-nova")  # typo: agent- vs agents-
        assert policy.allowed is False

    def test_unknown_namespace_reason_contains_namespace_name(self):
        from archivist.core.rbac import check_access

        policy = check_access("nova", "write", "agent-nova")
        assert "agent-nova" in policy.reason

    def test_unknown_namespace_hint_contains_did_you_mean(self):
        from archivist.core.rbac import check_access

        policy = check_access("nova", "write", "agents-novia")  # close to agents-nova
        assert policy.hint is not None
        assert "agents-nova" in policy.hint  # fuzzy match found

    def test_unknown_namespace_similar_namespaces_populated(self):
        from archivist.core.rbac import check_access

        policy = check_access("nova", "write", "agents-novia")
        assert "agents-nova" in policy.similar_namespaces

    def test_unknown_namespace_next_steps_populated(self):
        from archivist.core.rbac import check_access

        policy = check_access("nova", "write", "agent-nova")
        assert len(policy.next_steps) > 0
        full_text = " ".join(policy.next_steps)
        assert "archivist_index" in full_text
        assert "archivist_namespaces" in full_text
        assert "archivist_get_reference_docs" in full_text

    def test_unknown_namespace_hint_lists_accessible_namespaces(self):
        from archivist.core.rbac import check_access

        policy = check_access("nova", "read", "does-not-exist")
        # nova has access to agents-nova and shared (read=all)
        assert policy.hint is not None
        assert "agents-nova" in policy.hint or "shared" in policy.hint


# ---------------------------------------------------------------------------
# check_access — permission denied (namespace exists but agent lacks access)
# ---------------------------------------------------------------------------


class TestCheckAccessPermissionDenied:
    def test_denied_agent_gets_denied_policy(self):
        from archivist.core.rbac import check_access

        # athena-identity: write = [athena] only, nova not in list
        policy = check_access("nova", "write", "athena-identity")
        assert policy.allowed is False

    def test_denied_hint_contains_archivist_namespaces_call(self):
        from archivist.core.rbac import check_access

        policy = check_access("nova", "write", "athena-identity")
        assert policy.hint is not None
        # The hint now directs the agent to the inline permitted_namespaces field;
        # the actual archivist_namespaces call appears in next_steps instead.
        assert "permitted_namespaces" in policy.hint

    def test_denied_permitted_namespaces_populated(self):
        from archivist.core.rbac import check_access

        policy = check_access("nova", "write", "athena-identity")
        # permitted_namespaces must be a list (may be empty if nova has no access)
        assert isinstance(policy.permitted_namespaces, list)

    def test_denied_next_steps_populated(self):
        from archivist.core.rbac import check_access

        policy = check_access("nova", "write", "athena-identity")
        assert len(policy.next_steps) > 0
        full_text = " ".join(policy.next_steps)
        assert "archivist_namespaces" in full_text

    def test_denied_similar_namespaces_empty_for_permission_error(self):
        from archivist.core.rbac import check_access

        # The namespace exists — no fuzzy suggestions needed
        policy = check_access("nova", "write", "athena-identity")
        assert policy.similar_namespaces == []

    def test_allowed_agent_gets_allowed_policy(self):
        from archivist.core.rbac import check_access

        policy = check_access("nova", "write", "agents-nova")
        assert policy.allowed is True
        assert policy.next_steps == []


# ---------------------------------------------------------------------------
# _rbac_gate — full JSON error payload shape
# ---------------------------------------------------------------------------


class TestRbacGatePayload:
    def test_unknown_namespace_payload_has_all_keys(self):
        from archivist.app.handlers._common import _rbac_gate

        result = _rbac_gate("nova", "write", "agent-nova")
        assert result is not None
        payload = json.loads(result)
        assert payload["error"] == "access_denied"
        assert "reason" in payload
        assert "hint" in payload
        assert "next_steps" in payload
        assert "get_help" in payload
        assert "archivist_get_reference_docs" in payload["get_help"]

    def test_unknown_namespace_payload_includes_similar_namespaces(self):
        from archivist.app.handlers._common import _rbac_gate

        result = _rbac_gate("nova", "write", "agents-novia")  # typo
        payload = json.loads(result)
        assert "similar_namespaces" in payload
        assert "agents-nova" in payload["similar_namespaces"]

    def test_permission_denied_payload_has_all_keys(self):
        from archivist.app.handlers._common import _rbac_gate

        result = _rbac_gate("nova", "write", "athena-identity")
        assert result is not None
        payload = json.loads(result)
        assert payload["error"] == "access_denied"
        assert "next_steps" in payload
        assert "get_help" in payload

    def test_allowed_access_returns_none(self):
        from archivist.app.handlers._common import _rbac_gate

        result = _rbac_gate("nova", "write", "agents-nova")
        assert result is None

    def test_next_steps_is_a_list(self):
        from archivist.app.handlers._common import _rbac_gate

        result = _rbac_gate("nova", "read", "does-not-exist")
        payload = json.loads(result)
        assert isinstance(payload["next_steps"], list)
        assert len(payload["next_steps"]) >= 2

    def test_next_steps_reference_archivist_index(self):
        from archivist.app.handlers._common import _rbac_gate

        result = _rbac_gate("nova", "write", "bad-ns")
        payload = json.loads(result)
        combined = " ".join(payload["next_steps"])
        assert "archivist_index" in combined

    def test_get_help_always_points_to_reference_docs(self):
        from archivist.app.handlers._common import _rbac_gate

        # Both denial types must include get_help
        for ns in ("agent-nova", "athena-identity"):
            result = _rbac_gate("nova", "write", ns)
            payload = json.loads(result)
            assert "archivist_get_reference_docs" in payload["get_help"]


# ---------------------------------------------------------------------------
# require_caller — guidance in missing-agent_id response
# ---------------------------------------------------------------------------


class TestRequireCallerGuidance:
    def test_empty_caller_in_strict_mode_returns_error(self):
        import archivist.core.rbac as rbac
        from archivist.app.handlers._common import require_caller

        # Ensure strict RBAC mode (not permissive)
        assert not rbac.is_permissive_mode()
        result = require_caller("")
        assert result is not None

    def test_error_payload_has_next_steps(self):
        import archivist.core.rbac as rbac
        from archivist.app.handlers._common import require_caller

        assert not rbac.is_permissive_mode()
        result = require_caller("")
        assert result is not None
        payload = json.loads(result[0].text)
        assert "next_steps" in payload
        assert "get_help" in payload

    def test_error_payload_tells_agent_to_add_agent_id(self):
        import archivist.core.rbac as rbac
        from archivist.app.handlers._common import require_caller

        assert not rbac.is_permissive_mode()
        result = require_caller("")
        payload = json.loads(result[0].text)
        assert "agent_id" in payload["hint"]

    def test_non_empty_caller_returns_none(self):
        from archivist.app.handlers._common import require_caller

        assert require_caller("nova") is None
