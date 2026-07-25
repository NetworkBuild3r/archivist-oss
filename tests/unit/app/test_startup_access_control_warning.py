"""Unit tests for the startup no-access-control warning.

INIT-014/SPEC-001 (ac-6, ac-7): both ``ARCHIVIST_API_KEY`` and
``NAMESPACES_CONFIG_PATH`` defaulting to unset is a supported permissive-mode
configuration (single-operator/local-dev) — but it previously shipped with zero
operator-visible signal. ``_warn_if_no_access_control()`` makes that choice loud
without blocking startup or forcing auth on.
"""

from __future__ import annotations

import logging

import pytest

pytestmark = [pytest.mark.unit]


def test_warns_when_both_unset(monkeypatch, caplog):
    import archivist.app.main as main

    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "", raising=False)
    monkeypatch.setattr(main, "NAMESPACES_CONFIG_PATH", "", raising=False)

    with caplog.at_level(logging.CRITICAL, logger="archivist"):
        main._warn_if_no_access_control()

    assert any("NO ACCESS CONTROL" in r.message for r in caplog.records), (
        "expected a CRITICAL 'no access control' log line when both are unset"
    )


def test_does_not_warn_when_api_key_set(monkeypatch, caplog):
    import archivist.app.main as main

    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "some-real-key", raising=False)
    monkeypatch.setattr(main, "NAMESPACES_CONFIG_PATH", "", raising=False)

    with caplog.at_level(logging.CRITICAL, logger="archivist"):
        main._warn_if_no_access_control()

    assert not any("NO ACCESS CONTROL" in r.message for r in caplog.records)


def test_does_not_warn_when_namespaces_config_set(monkeypatch, caplog):
    import archivist.app.main as main

    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "", raising=False)
    monkeypatch.setattr(
        main, "NAMESPACES_CONFIG_PATH", "/etc/archivist/namespaces.yaml", raising=False
    )

    with caplog.at_level(logging.CRITICAL, logger="archivist"):
        main._warn_if_no_access_control()

    assert not any("NO ACCESS CONTROL" in r.message for r in caplog.records)


def test_does_not_warn_when_both_set(monkeypatch, caplog):
    import archivist.app.main as main

    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "some-real-key", raising=False)
    monkeypatch.setattr(
        main, "NAMESPACES_CONFIG_PATH", "/etc/archivist/namespaces.yaml", raising=False
    )

    with caplog.at_level(logging.CRITICAL, logger="archivist"):
        main._warn_if_no_access_control()

    assert not any("NO ACCESS CONTROL" in r.message for r in caplog.records)


def test_warning_does_not_leak_credential_values(monkeypatch, caplog):
    """Security AC: the warning states *that* auth/RBAC are permissive, never a
    credential value — this test would fail if a future edit accidentally
    interpolated ARCHIVIST_API_KEY into the message."""
    import archivist.app.main as main

    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "", raising=False)
    monkeypatch.setattr(main, "NAMESPACES_CONFIG_PATH", "", raising=False)

    with caplog.at_level(logging.CRITICAL, logger="archivist"):
        main._warn_if_no_access_control()

    for record in caplog.records:
        assert "some-real-key" not in record.message
        assert "secret" not in record.message.lower()


def test_permissive_mode_server_still_starts_and_serves_requests(monkeypatch):
    """ac-7: the warning is informational only — the server must still start
    successfully and serve requests when both are unset."""
    from starlette.testclient import TestClient

    import archivist.app.main as main
    import archivist.core.health as health

    async def _noop_startup():
        # Exercise the real warning function as part of startup, but skip the
        # heavy Qdrant/SQLite/background-task init this unit test doesn't need.
        main._warn_if_no_access_control()

    monkeypatch.setattr(main, "_startup", _noop_startup)
    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "", raising=False)
    monkeypatch.setattr(main, "NAMESPACES_CONFIG_PATH", "", raising=False)

    # `archivist.core.health` is a process-wide, module-level registry (`_status`)
    # with no per-test reset — an unrelated test elsewhere in the suite that marks
    # a subsystem unhealthy (e.g. to test degraded-mode `/health` responses) leaks
    # that state into every later test in the same process. This test's only
    # concern is ac-7 (the server starts and serves requests in permissive mode,
    # independent of subsystem health), so isolate it from that pre-existing
    # global-state leak rather than asserting on real subsystem health here.
    monkeypatch.setattr(health, "_status", {})

    with TestClient(main.app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200, "server did not start/serve requests in permissive mode"
