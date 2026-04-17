"""Uniform result types for the retrieval pipeline.

Every retrieval source (vector, BM25, literal, registry, graph facts) produces
results that flow through the same pipeline stages. This module defines the
canonical types so no source bypasses ranking with hardcoded scores.
"""

from dataclasses import dataclass, field
from enum import Enum


class RetrievalSource(str, Enum):
    """Where a result originated — used for tracing, not scoring."""
    VECTOR = "vector"
    BM25 = "bm25"
    LITERAL = "literal"
    REGISTRY = "registry"
    GRAPH_FACT = "graph_fact"
    HYDE = "hyde"


@dataclass
class ResultCandidate:
    """Uniform result flowing through all pipeline stages.

    Factory methods normalize source-specific fields to sensible defaults
    so downstream stages (RRF, reranker, synthesis) never hit missing keys.
    """
    id: str = ""
    score: float = 0.0
    text: str = ""
    agent_id: str = ""
    file_path: str = ""
    file_type: str = ""
    date: str = ""
    content_date: str = ""
    indexed_at: str = ""
    team: str = ""
    namespace: str = ""
    chunk_index: int = 0
    parent_id: str | None = None
    is_parent: bool = False
    importance_score: float = 0.5
    retention_class: str = "standard"
    topic: str = ""
    thought_type: str = ""
    l0: str = ""
    l1: str = ""
    source: RetrievalSource = RetrievalSource.VECTOR
    actor_id: str = ""
    actor_type: str = ""
    confidence: float = 1.0
    source_trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to the pipeline dict format expected by all stages."""
        d = {
            "id": self.id,
            "score": self.score,
            "text": self.text,
            "agent_id": self.agent_id,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "date": self.date,
            "content_date": self.content_date,
            "indexed_at": self.indexed_at,
            "team": self.team,
            "namespace": self.namespace,
            "chunk_index": self.chunk_index,
            "parent_id": self.parent_id,
            "is_parent": self.is_parent,
            "importance_score": self.importance_score,
            "retention_class": self.retention_class,
            "topic": self.topic,
            "thought_type": self.thought_type,
            "l0": self.l0,
            "l1": self.l1,
            "retrieval_source": self.source.value,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "confidence": self.confidence,
            "source_trace": self.source_trace,
        }
        return d

    @classmethod
    def from_qdrant_payload(
        cls,
        point_id: str,
        payload: dict,
        score: float = 0.0,
        source: RetrievalSource = RetrievalSource.VECTOR,
    ) -> "ResultCandidate":
        """Build from a raw Qdrant point payload."""
        return cls(
            id=point_id,
            score=score,
            text=payload.get("text", ""),
            agent_id=payload.get("agent_id", ""),
            file_path=payload.get("file_path", ""),
            file_type=payload.get("file_type", ""),
            date=payload.get("date", ""),
            content_date=payload.get("content_date", ""),
            indexed_at=payload.get("indexed_at", ""),
            team=payload.get("team", ""),
            namespace=payload.get("namespace", ""),
            chunk_index=payload.get("chunk_index", 0),
            parent_id=payload.get("parent_id"),
            is_parent=payload.get("is_parent", False),
            importance_score=payload.get("importance_score", 0.5),
            retention_class=payload.get("retention_class", "standard"),
            topic=payload.get("topic", ""),
            thought_type=payload.get("thought_type", ""),
            l0=payload.get("l0", ""),
            l1=payload.get("l1", ""),
            source=source,
            actor_id=payload.get("actor_id", ""),
            actor_type=payload.get("actor_type", ""),
            confidence=payload.get("confidence", 1.0),
            source_trace=payload.get("source_trace") or {},
        )

    @classmethod
    def from_registry_hit(cls, hit: dict) -> "ResultCandidate":
        """Build from a needle registry lookup result.

        Registry hits carry truncated chunk_text (500 chars) and minimal
        metadata. The caller should fetch full payloads from Qdrant and
        update via update_from_payload() before feeding into the pipeline.
        """
        return cls(
            id=hit.get("memory_id", ""),
            score=0.0,
            text=hit.get("chunk_text", ""),
            agent_id=hit.get("agent_id", ""),
            namespace=hit.get("namespace", ""),
            date=hit.get("created_at", "")[:10] if hit.get("created_at") else "",
            file_type="needle_registry",
            source=RetrievalSource.REGISTRY,
            actor_id=hit.get("actor_id", ""),
            actor_type=hit.get("actor_type", ""),
        )

    @classmethod
    def from_bm25_hit(cls, hit: dict) -> "ResultCandidate":
        """Build from a BM25/FTS5 search result dict."""
        return cls(
            id=hit.get("qdrant_id", ""),
            score=abs(hit.get("bm25_rank", 0)),
            text=hit.get("text", ""),
            agent_id=hit.get("agent_id", ""),
            file_path=hit.get("file_path", ""),
            date=hit.get("date", ""),
            namespace=hit.get("namespace", ""),
            chunk_index=hit.get("chunk_index", 0),
            source=RetrievalSource.BM25,
            actor_id=hit.get("actor_id", ""),
            actor_type=hit.get("actor_type", ""),
        )

    def update_from_payload(self, payload: dict) -> None:
        """Refresh fields from a fresh Qdrant payload (stale text fix)."""
        self.text = payload.get("text", self.text)
        self.agent_id = payload.get("agent_id", self.agent_id)
        self.file_path = payload.get("file_path", self.file_path)
        self.file_type = payload.get("file_type", self.file_type)
        self.date = payload.get("date", self.date)
        self.content_date = payload.get("content_date", self.content_date)
        self.team = payload.get("team", self.team)
        self.namespace = payload.get("namespace", self.namespace)
        self.chunk_index = payload.get("chunk_index", self.chunk_index)
        self.parent_id = payload.get("parent_id", self.parent_id)
        self.is_parent = payload.get("is_parent", self.is_parent)
        self.importance_score = payload.get("importance_score", self.importance_score)
        self.retention_class = payload.get("retention_class", self.retention_class)
        self.topic = payload.get("topic", self.topic)
        self.thought_type = payload.get("thought_type", self.thought_type)
        self.l0 = payload.get("l0", self.l0)
        self.l1 = payload.get("l1", self.l1)
        self.actor_id = payload.get("actor_id", self.actor_id)
        self.actor_type = payload.get("actor_type", self.actor_type)
        self.confidence = payload.get("confidence", self.confidence)
        self.source_trace = payload.get("source_trace") or self.source_trace
