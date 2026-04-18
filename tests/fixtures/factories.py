"""Deterministic test data factories for the Archivist test suite.

All test data should originate here so that field contracts are enforced in one
place and tests remain independent of production data.

Convention:
- ``agent_id`` values: ``"agent-test"``, ``"agent-smoke"``, ``"agent-{role}"``
- ``namespace`` values: ``"test-ns"``, ``"default"``, ``"qa-{domain}"``
- Memory text: synthetic; contains NEEDLE_PATTERN-matchable tokens (timestamps,
  ticket IDs, IPs) so ``register_needle_tokens()`` actually inserts rows.
"""

from __future__ import annotations

import uuid
from typing import Any

# Synthetic memory texts with NEEDLE_PATTERN-matchable tokens.
_FAKE_TOPICS: list[str] = [
    "user prefers dark mode — config USER-1234 set 2026-01-17T09:00",
    "project deadline 2026-06-30T00:00 ticket PROJ-5678",
    "deployment runs on 10.0.0.1/24 — ticket OPS-9012",
    "embedding model EMB-3456 at 2026-02-01T12:00 dimension 1536",
    "API rate limit 1000 req/min — TICKET-7890 logged 2026-03-15T08:30",
    "memory retention 90 days — POLICY-2345 since 2025-09-01T00:00",
    "outbox drain KEY=2 — OPS-4567 updated 2026-04-17T10:00",
    "Qdrant 192.168.1.10:6333 — CONFIG-8901 healthy 2026-04-17T11:00",
    "SQLite WAL mode — CONF-1122 at 2026-01-01T00:00",
    "phase 3 rollout FEAT-3344 closed 2026-04-01T17:00",
]


def make_memory_id() -> str:
    """Return a fresh UUID string suitable for use as a memory ID."""
    return str(uuid.uuid4())


def make_qdrant_id() -> str:
    """Return a fresh UUID string suitable for use as a Qdrant point ID."""
    return str(uuid.uuid4())


def memory_payload(
    text: str | None = None,
    agent_id: str = "agent-test",
    namespace: str = "test-ns",
    actor_id: str | None = None,
    actor_type: str = "human",
    memory_type: str = "general",
    **overrides: Any,
) -> dict[str, Any]:
    """Return a minimal valid memory payload dict.

    Suitable for use directly in handler argument dicts or for seeding
    ``memory_chunks`` / ``memory_points`` rows.
    """
    return {
        "memory_id": make_memory_id(),
        "qdrant_id": make_qdrant_id(),
        "text": text or f"Test memory {uuid.uuid4().hex[:8]}",
        "agent_id": agent_id,
        "namespace": namespace,
        "actor_id": actor_id or agent_id,
        "actor_type": actor_type,
        "memory_type": memory_type,
        "file_path": f"test/{agent_id}/mem.md",
        "chunk_index": 0,
        "date": "2026-01-17",
        **overrides,
    }


class MemoryFactory:
    """Stateful factory that generates unique memories in a sequence.

    Round-robins through ``_FAKE_TOPICS`` so each call produces a distinct,
    needle-token-rich text without requiring explicit ``text`` arguments.
    """

    def __init__(self) -> None:
        self._counter = 0

    def __call__(
        self,
        namespace: str = "default",
        agent_id: str = "qa-agent",
        actor_id: str = "user-qa",
        actor_type: str = "human",
        memory_type: str = "general",
        text: str | None = None,
    ) -> dict[str, Any]:
        idx = self._counter % len(_FAKE_TOPICS)
        self._counter += 1
        return {
            "memory_id": make_memory_id(),
            "qdrant_id": make_qdrant_id(),
            "text": text or _FAKE_TOPICS[idx],
            "namespace": namespace,
            "agent_id": agent_id,
            "actor_id": actor_id,
            "actor_type": actor_type,
            "memory_type": memory_type,
            "file_path": f"qa/{agent_id}/{idx}.md",
            "chunk_index": idx,
            "date": "2026-01-17",
        }


def entity_payload(
    name: str = "TestEntity",
    entity_type: str = "concept",
    namespace: str = "test-ns",
    agent_id: str = "agent-test",
) -> dict[str, Any]:
    """Return a minimal valid entity payload dict."""
    return {
        "name": name,
        "entity_type": entity_type,
        "namespace": namespace,
        "agent_id": agent_id,
        "actor_id": agent_id,
        "actor_type": "system",
    }


def trajectory_entry(
    agent_id: str = "agent-test",
    action: str = "store",
    result: str = "success",
) -> dict[str, Any]:
    """Return a minimal valid trajectory entry payload dict."""
    return {
        "agent_id": agent_id,
        "action": action,
        "result": result,
        "memory_id": make_memory_id(),
    }
