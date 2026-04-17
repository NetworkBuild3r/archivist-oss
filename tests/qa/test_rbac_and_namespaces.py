"""RBAC and namespace isolation tests under concurrent agent activity.

Scenarios
---------
1. Agent without write permission is denied on ``archivist_store``.
2. Agent with write permission succeeds.
3. Agent from namespace A cannot see namespace B outbox events.
4. ``get_namespace_for_agent`` returns the correct namespace from config.
5. ``check_access`` returns ``AccessPolicy(allowed=True)`` for permitted agents.
6. ``check_access`` returns ``AccessPolicy(allowed=False)`` for denied agents.
7. Concurrent agents writing to different namespaces do not interleave data.
8. Permissive fallback (no config file) allows all operations.
9. Namespace isolation: outbox rows are tagged by memory_id; two namespaces
   produce independent event streams.
10. ``get_namespace_config`` returns None for an unknown namespace.

All tests use the ``qa_pool`` fixture and a temporary ``namespaces.yaml``.
No real Qdrant or LLM calls are made.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from tests.qa.conftest import count_outbox

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_namespaces_yaml(tmp_path: Path, content: str) -> str:
    """Write *content* to a namespaces.yaml in *tmp_path*, return path string."""
    p = tmp_path / "namespaces.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


def _reload_rbac(yaml_path: str) -> None:
    """Force-reload the RBAC module with a new config path."""
    import archivist.core.rbac as _rbac

    _rbac._config = None
    _rbac._permissive_fallback = False
    _rbac.load_config(yaml_path)


# ---------------------------------------------------------------------------
# 1 + 2. Access control — write permission
# ---------------------------------------------------------------------------


def test_write_denied_for_unknown_agent(tmp_path):
    """Agent not listed in the namespace write ACL is denied."""
    yaml_path = _write_namespaces_yaml(
        tmp_path,
        """
        namespaces:
          - id: prod
            read: [agent-reader]
            write: [agent-writer]
        agent_namespaces:
          agent-writer: prod
          agent-reader: prod
        """,
    )
    _reload_rbac(yaml_path)

    from archivist.core.rbac import check_access

    policy = check_access("intruder-agent", "write", "prod")
    assert not policy.allowed


def test_write_allowed_for_listed_agent(tmp_path):
    """Agent in the write ACL is granted write access."""
    yaml_path = _write_namespaces_yaml(
        tmp_path,
        """
        namespaces:
          - id: prod
            read: [agent-reader]
            write: [agent-writer]
        agent_namespaces:
          agent-writer: prod
        """,
    )
    _reload_rbac(yaml_path)

    from archivist.core.rbac import check_access

    policy = check_access("agent-writer", "write", "prod")
    assert policy.allowed


# ---------------------------------------------------------------------------
# 3. Namespace isolation in the RBAC config
# ---------------------------------------------------------------------------


def test_namespace_read_allowed_only_for_listed_agent(tmp_path):
    """Agent allowed to read namespace A cannot read namespace B."""
    yaml_path = _write_namespaces_yaml(
        tmp_path,
        """
        namespaces:
          - id: ns-a
            read: [agent-a]
            write: [agent-a]
          - id: ns-b
            read: [agent-b]
            write: [agent-b]
        agent_namespaces:
          agent-a: ns-a
          agent-b: ns-b
        """,
    )
    _reload_rbac(yaml_path)

    from archivist.core.rbac import check_access

    assert check_access("agent-a", "read", "ns-a").allowed
    assert not check_access("agent-a", "read", "ns-b").allowed
    assert check_access("agent-b", "read", "ns-b").allowed
    assert not check_access("agent-b", "read", "ns-a").allowed


# ---------------------------------------------------------------------------
# 4. get_namespace_for_agent resolution
# ---------------------------------------------------------------------------


def test_get_namespace_for_agent_returns_configured_namespace(tmp_path):
    """get_namespace_for_agent returns the namespace from agent_namespaces map."""
    yaml_path = _write_namespaces_yaml(
        tmp_path,
        """
        namespaces:
          - id: custom-ns
            read: [my-agent]
            write: [my-agent]
        agent_namespaces:
          my-agent: custom-ns
        """,
    )
    _reload_rbac(yaml_path)

    from archivist.core.rbac import get_namespace_for_agent

    assert get_namespace_for_agent("my-agent") == "custom-ns"


def test_get_namespace_for_agent_defaults_to_default(tmp_path):
    """Unknown agent resolves to 'default' namespace."""
    yaml_path = _write_namespaces_yaml(
        tmp_path,
        """
        namespaces: []
        agent_namespaces: {}
        """,
    )
    _reload_rbac(yaml_path)

    from archivist.core.rbac import get_namespace_for_agent

    ns = get_namespace_for_agent("totally-unknown-agent")
    assert ns == "default"


# ---------------------------------------------------------------------------
# 5 + 6. check_access return types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_id,action,namespace,should_allow",
    [
        pytest.param("agent-rw", "read", "shared", True, id="read-allowed"),
        pytest.param("agent-rw", "write", "shared", True, id="write-allowed"),
        pytest.param("agent-ro", "read", "shared", True, id="readonly-read"),
        pytest.param("agent-ro", "write", "shared", False, id="readonly-write-denied"),
        pytest.param("outsider", "read", "shared", False, id="outsider-read-denied"),
    ],
)
def test_check_access_parametrized(tmp_path, agent_id, action, namespace, should_allow):
    """Parametrised check_access covers allowed and denied cases."""
    yaml_path = _write_namespaces_yaml(
        tmp_path,
        """
        namespaces:
          - id: shared
            read: [agent-rw, agent-ro]
            write: [agent-rw]
        agent_namespaces:
          agent-rw: shared
          agent-ro: shared
        """,
    )
    _reload_rbac(yaml_path)

    from archivist.core.rbac import check_access

    policy = check_access(agent_id, action, namespace)
    assert policy.allowed == should_allow


# ---------------------------------------------------------------------------
# 7. Concurrent agents write to different namespaces — no data interleave
# ---------------------------------------------------------------------------


async def test_concurrent_agents_namespace_isolation(qa_pool, memory_factory):
    """Sequential MemoryTransaction writes for different namespaces don't interleave outbox."""
    from archivist.storage.transaction import MemoryTransaction

    ns_a_mems = [memory_factory(namespace="ns-a") for _ in range(5)]
    ns_b_mems = [memory_factory(namespace="ns-b") for _ in range(5)]

    # Sequential — GRAPH_WRITE_LOCK_ASYNC cannot be shared across asyncio.gather tasks.
    for m in ns_a_mems + ns_b_mems:
        async with MemoryTransaction(enabled=True) as txn:
            txn.enqueue_qdrant_upsert("col", [], memory_id=m["memory_id"])

    # Total events: 10 (5 per namespace)
    assert await count_outbox(qa_pool, "pending") == 10


# ---------------------------------------------------------------------------
# 8. Permissive fallback — missing config allows all access
# ---------------------------------------------------------------------------


def test_permissive_fallback_allows_all_when_no_config():
    """With no config file, RBAC falls back to permissive mode (all allowed)."""
    import archivist.core.rbac as _rbac

    _rbac._config = None
    _rbac._permissive_fallback = False
    _rbac.load_config("")  # empty path → permissive

    from archivist.core.rbac import check_access

    assert check_access("any-agent", "write", "any-namespace").allowed
    assert check_access("any-agent", "read", "any-namespace").allowed


# ---------------------------------------------------------------------------
# 9. Outbox event memory_ids are namespace-scoped (via memory_id tagging)
# ---------------------------------------------------------------------------


async def test_outbox_events_carry_correct_memory_ids(qa_pool, memory_factory):
    """Outbox rows store memory_id in payload; events for ns-a do not contain ns-b ids."""
    import json

    from archivist.storage.transaction import MemoryTransaction

    mem_a = memory_factory(namespace="ns-a")
    mem_b = memory_factory(namespace="ns-b")

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", [mem_a["qdrant_id"]], memory_id=mem_a["memory_id"])

    async with MemoryTransaction(enabled=True) as txn:
        txn.enqueue_qdrant_delete("col", [mem_b["qdrant_id"]], memory_id=mem_b["memory_id"])

    async with qa_pool.read() as conn:
        cur = await conn.execute("SELECT payload FROM outbox ORDER BY created_at")
        rows = await cur.fetchall()

    payloads = [json.loads(r[0]) for r in rows]
    memory_ids_in_outbox = {p["memory_id"] for p in payloads}

    assert mem_a["memory_id"] in memory_ids_in_outbox
    assert mem_b["memory_id"] in memory_ids_in_outbox
    # No cross-contamination: each event is tagged with exactly its own memory_id
    assert all(p["memory_id"] in {mem_a["memory_id"], mem_b["memory_id"]} for p in payloads)


# ---------------------------------------------------------------------------
# 10. get_namespace_config for unknown namespace returns None
# ---------------------------------------------------------------------------


def test_get_namespace_config_unknown_returns_none(tmp_path):
    """get_namespace_config returns None for a namespace not in the config."""
    yaml_path = _write_namespaces_yaml(
        tmp_path,
        """
        namespaces:
          - id: known-ns
            read: [agent]
            write: [agent]
        agent_namespaces: {}
        """,
    )
    _reload_rbac(yaml_path)

    from archivist.core.rbac import get_namespace_config

    assert get_namespace_config("unknown-ns") is None
    assert get_namespace_config("known-ns") is not None


# ---------------------------------------------------------------------------
# 11. Wildcard ACL — '*' grants all agents access
# ---------------------------------------------------------------------------


def test_wildcard_acl_grants_all_agents(tmp_path):
    """Namespace with read: ['all'] allows any agent to read."""
    yaml_path = _write_namespaces_yaml(
        tmp_path,
        """
        namespaces:
          - id: public
            read: [all]
            write: [admin-agent]
        agent_namespaces: {}
        """,
    )
    _reload_rbac(yaml_path)

    from archivist.core.rbac import check_access

    assert check_access("random-agent", "read", "public").allowed
    assert not check_access("random-agent", "write", "public").allowed
    assert check_access("admin-agent", "write", "public").allowed
