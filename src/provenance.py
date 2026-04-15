"""Provenance types for actor-aware memory (Phase 6).

Defines the structured types that travel with every memory through the write
pipeline, storage layers, and retrieval path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActorType(str, Enum):
    AGENT = "agent"
    HUMAN = "human"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class SourceTrace:
    """Structured origin context for a memory or derived artifact.

    Stored in Qdrant payloads as a plain dict (via ``to_dict``).
    Callers provide what they have; the system fills in what it knows.
    """

    tool: str = ""
    session_id: str = ""
    upstream_source: str = ""
    parent_memory_id: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {}
        if self.tool:
            d["tool"] = self.tool
        if self.session_id:
            d["session_id"] = self.session_id
        if self.upstream_source:
            d["upstream_source"] = self.upstream_source
        if self.parent_memory_id:
            d["parent_memory_id"] = self.parent_memory_id
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> SourceTrace:
        if not d or not isinstance(d, dict):
            return cls()
        return cls(
            tool=d.get("tool", ""),
            session_id=d.get("session_id", ""),
            upstream_source=d.get("upstream_source", ""),
            parent_memory_id=d.get("parent_memory_id", ""),
            extra=d.get("extra", {}),
        )

    def with_parent(self, parent_memory_id: str) -> SourceTrace:
        """Return a copy with ``parent_memory_id`` set (for derived artifacts)."""
        return SourceTrace(
            tool=self.tool,
            session_id=self.session_id,
            upstream_source=self.upstream_source,
            parent_memory_id=parent_memory_id,
            extra=dict(self.extra),
        )


def default_confidence(actor_type: str, confidence_map: dict[str, float] | None = None) -> float:
    """Look up the default confidence for an actor type.

    Uses the consolidated ``DEFAULT_CONFIDENCE_BY_ACTOR_TYPE`` dict from config
    when *confidence_map* is ``None``.
    """
    if confidence_map is None:
        from config import DEFAULT_CONFIDENCE_BY_ACTOR_TYPE
        confidence_map = DEFAULT_CONFIDENCE_BY_ACTOR_TYPE
    return confidence_map.get(actor_type, confidence_map.get("agent", 0.8))
