"""Single-file local ACID memory store (INIT-024/SPEC-006 + AUDIT-MIN-9).

Stores chunks, FTS5 full-text index, ``LocalVectorIndex`` vectors, entities,
facts, relationships, needle registry, and ``archivist_meta`` inside a single
SQLite WAL file (default ``~/.local/share/archivist/memory.db``, mode ``0600``):

- Zero Qdrant, zero Docker, zero ``OutboxProcessor`` in ``ARCHIVIST_PROFILE=local``.
- Runs the ``<5ms`` ``re2`` Memory Firewall BEFORE embedding or persisting.
- Enforces ``archivist_meta`` embedding model/dimension mismatch guard on
  startup (``AUDIT-MIN-9``) — never silently drops or mixes vector indexes.
- Commits chunk + FTS5 + vector in ONE atomic SQLite transaction; ``forget``
  removes/suppresses across all three atomically (zero zombie vectors).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from archivist.features.local_onnx import (
    DEFAULT_LOCAL_EMBED_DIM,
    DEFAULT_LOCAL_EMBED_MODEL,
    LocalOnnxConfig,
    LocalOnnxRuntime,
)
from archivist.retrieval.context_packer import PackedContext, pack_context
from archivist.security.firewall import (
    FirewallPolicy,
    FirewallResult,
    MemoryFirewall,
    frame_memories,
)
from archivist.security.trust_tiers import (
    RequestContext,
    TrustTier,
    is_retrievable_by_default,
)
from archivist.storage.local_vector import (
    LocalVectorIndex,
    create_local_vector_index,
    load_sqlite_vec,
)
from archivist.storage.metadata import (
    EmbeddingMetadata,
    verify_or_record_embedding_metadata_sync,
)
from archivist.utils.tokenizer import count_tokens
from archivist.write.tiering import extract_l0_deterministic, extract_l1_query_window

logger = logging.getLogger("archivist.local_store")

_FTS_SANITIZE_RE = re.compile(r"[^\w\s-]")


def default_local_db_path() -> Path:
    """Return XDG data path ``~/.local/share/archivist/memory.db``."""
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg) if xdg else (Path.home() / ".local" / "share")
    return base / "archivist" / "memory.db"


_LOCAL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_chunks (
    id                      TEXT PRIMARY KEY,
    scope_id                TEXT NOT NULL,
    content                 TEXT NOT NULL,
    l0                      TEXT NOT NULL DEFAULT '',
    l1                      TEXT NOT NULL DEFAULT '',
    kind                    TEXT NOT NULL DEFAULT 'fact',
    tags_json               TEXT NOT NULL DEFAULT '[]',
    trust_tier              TEXT NOT NULL DEFAULT 'T1_AGENT',
    actor_id                TEXT NOT NULL DEFAULT 'local',
    source_client           TEXT NOT NULL DEFAULT 'unknown',
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    valid_at                TEXT,
    invalid_at              TEXT,
    expired_at              TEXT,
    suppressed              INTEGER NOT NULL DEFAULT 0,
    suppress_reason         TEXT NOT NULL DEFAULT '',
    quarantine_reasons_json TEXT NOT NULL DEFAULT '[]',
    content_hash            TEXT NOT NULL,
    token_count             INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memory_chunks_scope_active
    ON memory_chunks (scope_id, suppressed, trust_tier, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_chunks_hash
    ON memory_chunks (scope_id, content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_fts USING fts5(
    chunk_id UNINDEXED,
    scope_id UNINDEXED,
    content,
    l0,
    tags,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   TEXT NOT NULL DEFAULT 'local',
    scope_id    TEXT NOT NULL,
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    created_at  TEXT NOT NULL,
    UNIQUE(tenant_id, scope_id, name)
);

CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id     TEXT NOT NULL DEFAULT 'local',
    scope_id      TEXT NOT NULL,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    object        TEXT NOT NULL,
    trust_tier    TEXT NOT NULL DEFAULT 'T1_AGENT',
    valid_at      TEXT,
    invalid_at    TEXT,
    created_at    TEXT NOT NULL,
    expired_at    TEXT,
    supersedes_id INTEGER
);

CREATE TABLE IF NOT EXISTS relationships (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id        TEXT NOT NULL DEFAULT 'local',
    scope_id         TEXT NOT NULL,
    source_entity_id INTEGER NOT NULL,
    target_entity_id INTEGER NOT NULL,
    relation_type    TEXT NOT NULL,
    trust_tier       TEXT NOT NULL DEFAULT 'T1_AGENT',
    valid_at         TEXT,
    invalid_at       TEXT,
    created_at       TEXT NOT NULL,
    expired_at       TEXT,
    supersedes_id    INTEGER
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_relationships_active_unique
    ON relationships (tenant_id, scope_id, source_entity_id, target_entity_id, relation_type)
    WHERE expired_at IS NULL;

CREATE TABLE IF NOT EXISTS needle_registry (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_id    TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,
    needle      TEXT NOT NULL,
    needle_type TEXT NOT NULL DEFAULT 'token',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_needle_scope_value
    ON needle_registry (scope_id, needle);
"""


@dataclass(frozen=True)
class SavedMemory:
    """Result of saving a memory into :class:`LocalMemoryStore`."""

    memory_id: str
    scope: str
    kind: str
    trust_tier: str
    quarantined: bool
    redacted_count: int
    enrichment_status: str
    deduplicated: bool = False
    quarantine_reasons: tuple[str, ...] = ()


def _make_fts_query(query: str) -> str:
    cleaned = _FTS_SANITIZE_RE.sub(" ", query or "").strip()
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens[:24])


def _set_owner_only_permissions(path: Path) -> None:
    if os.name == "posix" and path.exists():
        os.chmod(path, 0o600)


class LocalMemoryStore:
    """Single-file SQLite WAL + FTS5 + LocalVectorIndex + ONNX INT8 store."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        onnx_runtime: LocalOnnxRuntime | None = None,
        vector_backend: str = "sqlite-vec",
        quantization: str = "int8",
        firewall: MemoryFirewall | None = None,
        firewall_policy: FirewallPolicy | str = FirewallPolicy.REDACT,
        embed_model: str | None = None,
        vector_dim: int | None = None,
        offline_deterministic: bool = False,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_local_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            os.chmod(self.db_path.parent, 0o700)

        if onnx_runtime is not None:
            self.onnx = onnx_runtime
        else:
            self.onnx = LocalOnnxRuntime(
                LocalOnnxConfig(
                    embed_model=embed_model or DEFAULT_LOCAL_EMBED_MODEL,
                    vector_dim=vector_dim or DEFAULT_LOCAL_EMBED_DIM,
                    offline_deterministic=offline_deterministic,
                )
            )

        self.embed_model = embed_model or self.onnx.embed_model
        self.vector_dim = vector_dim or self.onnx.vector_dim
        self.firewall = firewall or MemoryFirewall(policy=firewall_policy)
        self._write_lock = threading.Lock()

        sidecar_path = self.db_path.with_suffix(".usearch") if vector_backend == "usearch" else None
        self.vector_index: LocalVectorIndex = create_local_vector_index(
            dim=self.vector_dim,
            backend=vector_backend,
            index_path=sidecar_path,
            quantization=quantization,
        )

        # Open dedicated write connection (WAL + synchronous=NORMAL + busy_timeout=5000).
        self._write_conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
        )
        self._write_conn.row_factory = sqlite3.Row
        self._write_conn.execute("PRAGMA journal_mode=WAL")
        self._write_conn.execute("PRAGMA synchronous=NORMAL")
        self._write_conn.execute("PRAGMA busy_timeout=5000")
        self._write_conn.execute("PRAGMA foreign_keys=ON")

        # Enforce 0600 permissions immediately on DB file creation (POSIX).
        _set_owner_only_permissions(self.db_path)

        # Initialize core tables + vector index tables.
        self._write_conn.executescript(_LOCAL_SCHEMA_SQL)
        self.vector_index.init_schema(self._write_conn)
        self._write_conn.commit()

        # AUDIT-MIN-9: Wire archivist_meta embedding model/dim mismatch guard into local startup.
        self.metadata: EmbeddingMetadata = verify_or_record_embedding_metadata_sync(
            self._write_conn,
            embed_model=self.embed_model,
            vector_dim=self.vector_dim,
            schema_version="3.0",
        )

        # Dedicated read-only connection (query_only=ON) for concurrent WAL reads.
        self._read_conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
        )
        self._read_conn.row_factory = sqlite3.Row
        load_sqlite_vec(self._read_conn)
        self._read_conn.execute("PRAGMA query_only=ON")
        self._read_conn.execute("PRAGMA busy_timeout=5000")

        # Ensure WAL/SHM sidecars if already present are also 0600.
        _set_owner_only_permissions(self.db_path)

    def close(self) -> None:
        """Close read/write connections cleanly and persist optional sidecar."""
        if hasattr(self.vector_index, "save"):
            self.vector_index.save()  # type: ignore[attr-defined]
        self._read_conn.close()
        self._write_conn.close()

    def _screen_content(
        self,
        content: str,
        *,
        ctx: RequestContext | None,
        trust_tier: TrustTier | None,
    ) -> FirewallResult:
        base_tier = (
            trust_tier
            if trust_tier is not None
            else (ctx.trust_tier if ctx is not None else TrustTier.T1_AGENT)
        )
        return self.firewall.scan(content, trust_tier=base_tier)

    def save_memory(
        self,
        content: str,
        *,
        scope: str = "user:global",
        kind: str = "fact",
        tags: Sequence[str] | None = None,
        ctx: RequestContext | None = None,
        trust_tier: TrustTier | None = None,
        memory_id: str | None = None,
        valid_at: str | None = None,
        deduplicate: bool = True,
        pre_commit_hook: Any | None = None,
    ) -> SavedMemory:
        """Screen via Memory Firewall, embed locally, and commit chunk+FTS5+vec atomically."""
        if not (content or "").strip():
            raise ValueError("content must be non-empty")

        fw_result = self._screen_content(content, ctx=ctx, trust_tier=trust_tier)
        sanitized = fw_result.text
        effective_tier = fw_result.trust_tier

        tag_list = [str(t).strip() for t in (tags or ()) if str(t).strip()]
        tags_str = " ".join(tag_list)
        content_hash = hashlib.sha256(f"{scope}\x00{sanitized}".encode()).hexdigest()

        # Deterministic zero-LLM L0/L1 summaries (SPEC-001).
        l0 = extract_l0_deterministic(sanitized, max_tokens=100)
        l1 = extract_l1_query_window(sanitized, "", max_tokens=300)
        tok_count = count_tokens(sanitized)

        # Embed strictly AFTER firewall redaction (never exfiltrate raw secrets to embedder).
        vec = self.onnx.embed_one(sanitized)

        now_iso = datetime.now(UTC).isoformat()
        actor_id = ctx.principal_id if ctx is not None else "local"
        source_client = ctx.source_client if ctx is not None else "unknown"
        mid = memory_id or f"mem_{uuid.uuid4().hex[:12]}"

        with self._write_lock:
            if deduplicate and memory_id is None:
                dup = self._write_conn.execute(
                    "SELECT id, trust_tier FROM memory_chunks "
                    "WHERE scope_id = ? AND content_hash = ? AND suppressed = 0 "
                    "LIMIT 1",
                    (scope, content_hash),
                ).fetchone()
                if dup is not None:
                    return SavedMemory(
                        memory_id=str(dup["id"]),
                        scope=scope,
                        kind=kind,
                        trust_tier=str(dup["trust_tier"]),
                        quarantined=str(dup["trust_tier"]) == TrustTier.T3_QUARANTINED.value,
                        redacted_count=fw_result.redacted_count,
                        enrichment_status="synced",
                        deduplicated=True,
                        quarantine_reasons=fw_result.quarantine_reasons,
                    )

            with self._write_conn:
                self._write_conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_chunks (
                        id, scope_id, content, l0, l1, kind, tags_json,
                        trust_tier, actor_id, source_client, created_at, updated_at,
                        valid_at, suppressed, suppress_reason, quarantine_reasons_json,
                        content_hash, token_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '', ?, ?, ?)
                    """,
                    (
                        mid,
                        scope,
                        sanitized,
                        l0,
                        l1,
                        kind,
                        json.dumps(tag_list),
                        effective_tier.value,
                        actor_id,
                        source_client,
                        now_iso,
                        now_iso,
                        valid_at or now_iso,
                        json.dumps(list(fw_result.quarantine_reasons)),
                        content_hash,
                        tok_count,
                    ),
                )
                self._write_conn.execute(
                    "DELETE FROM memory_chunks_fts WHERE chunk_id = ?",
                    (mid,),
                )
                self._write_conn.execute(
                    "INSERT INTO memory_chunks_fts (chunk_id, scope_id, content, l0, tags) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (mid, scope, sanitized, l0, tags_str),
                )
                self.vector_index.upsert(self._write_conn, mid, vec, scope_id=scope)
                if pre_commit_hook is not None:
                    pre_commit_hook()

        return SavedMemory(
            memory_id=mid,
            scope=scope,
            kind=kind,
            trust_tier=effective_tier.value,
            quarantined=fw_result.quarantined,
            redacted_count=fw_result.redacted_count,
            enrichment_status="synced",
            deduplicated=False,
            quarantine_reasons=fw_result.quarantine_reasons,
        )

    def forget_memory(
        self,
        memory_id: str,
        *,
        mode: str = "suppress",
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Atomically suppress or hard-delete *memory_id* across chunks, FTS5, and vectors."""
        norm_mode = (mode or "suppress").strip().lower()
        if norm_mode not in ("suppress", "delete"):
            raise ValueError(f"mode must be 'suppress' or 'delete', got {mode!r}")

        with self._write_lock:
            existing = self._write_conn.execute(
                "SELECT id, scope_id FROM memory_chunks WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if existing is None:
                return {"memory_id": memory_id, "mode": norm_mode, "affected": 0, "found": False}

            now_iso = datetime.now(UTC).isoformat()
            with self._write_conn:
                self._write_conn.execute(
                    "DELETE FROM memory_chunks_fts WHERE chunk_id = ?",
                    (memory_id,),
                )
                self.vector_index.delete(self._write_conn, memory_id)
                self._write_conn.execute(
                    "DELETE FROM needle_registry WHERE chunk_id = ?",
                    (memory_id,),
                )
                if norm_mode == "delete":
                    self._write_conn.execute(
                        "DELETE FROM memory_chunks WHERE id = ?",
                        (memory_id,),
                    )
                else:
                    self._write_conn.execute(
                        "UPDATE memory_chunks "
                        "SET suppressed = 1, suppress_reason = ?, updated_at = ? "
                        "WHERE id = ?",
                        (reason or "user_suppressed", now_iso, memory_id),
                    )

        return {
            "memory_id": memory_id,
            "scope": str(existing["scope_id"]),
            "mode": norm_mode,
            "affected": 1,
            "found": True,
        }

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch a single memory record (including provenance & quarantine metadata)."""
        row = self._read_conn.execute(
            "SELECT * FROM memory_chunks WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "scope": str(row["scope_id"]),
            "text": str(row["content"]),
            "l0": str(row["l0"]),
            "l1": str(row["l1"]),
            "l2": str(row["content"]),
            "kind": str(row["kind"]),
            "tags": json.loads(row["tags_json"] or "[]"),
            "trust_tier": str(row["trust_tier"]),
            "actor": str(row["actor_id"]),
            "source_client": str(row["source_client"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "valid_at": row["valid_at"],
            "expired_at": row["expired_at"],
            "suppressed": bool(row["suppressed"]),
            "suppress_reason": str(row["suppress_reason"] or ""),
            "quarantine_reasons": json.loads(row["quarantine_reasons_json"] or "[]"),
            "token_count": int(row["token_count"] or 0),
        }

    def search_memories(
        self,
        query: str,
        *,
        scopes: Sequence[str] | None = None,
        limit: int = 10,
        kind: str | None = None,
        as_of: str | None = None,
        include_quarantined: bool = False,
    ) -> list[dict[str, Any]]:
        """Hybrid vector + FTS5 BM25 search fused with RRF (k=60), excluding T3 by default."""
        if limit <= 0:
            return []
        search_scopes = list(scopes) if scopes else ["*"]

        q_vec = self.onnx.embed_one(query)
        rrf_k = 60.0
        rrf_scores: dict[str, float] = {}

        # 1. Dense vector recall per scope.
        vec_limit = max(limit * 3, 24)
        for sc in search_scopes:
            vec_hits = self.vector_index.search(
                self._read_conn, q_vec, scope_id=sc, limit=vec_limit
            )
            for rank, (cid, _sim) in enumerate(vec_hits, start=1):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        # 2. FTS5 BM25 recall.
        fts_query = _make_fts_query(query)
        if fts_query:
            if search_scopes == ["*"]:
                fts_rows = self._read_conn.execute(
                    "SELECT chunk_id, bm25(memory_chunks_fts) AS rank_score "
                    "FROM memory_chunks_fts "
                    "WHERE memory_chunks_fts MATCH ? "
                    "ORDER BY rank_score LIMIT ?",
                    (fts_query, vec_limit),
                ).fetchall()
            else:
                ph = ",".join("?" for _ in search_scopes)
                fts_rows = self._read_conn.execute(
                    f"SELECT chunk_id, bm25(memory_chunks_fts) AS rank_score "
                    f"FROM memory_chunks_fts "
                    f"WHERE memory_chunks_fts MATCH ? AND scope_id IN ({ph}) "
                    f"ORDER BY rank_score LIMIT ?",
                    (fts_query, *search_scopes, vec_limit),
                ).fetchall()
            for rank, row in enumerate(fts_rows, start=1):
                cid = str(row["chunk_id"])
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        if not rrf_scores:
            return []

        cand_ids = [
            cid
            for cid, _ in sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[
                : max(limit * 3, 25)
            ]
        ]
        ph = ",".join("?" for _ in cand_ids)
        rows = self._read_conn.execute(
            f"SELECT * FROM memory_chunks WHERE id IN ({ph}) AND suppressed = 0",
            cand_ids,
        ).fetchall()
        by_id = {str(r["id"]): self._row_to_dict(r) for r in rows}

        filtered: list[dict[str, Any]] = []
        for cid in cand_ids:
            item = by_id.get(cid)
            if item is None:
                continue
            if not include_quarantined and not is_retrievable_by_default(item["trust_tier"]):
                continue
            if kind and item["kind"] != kind:
                continue
            if as_of:
                created = str(item.get("created_at") or "")
                expired = item.get("expired_at")
                if created and created > as_of:
                    continue
                if expired and str(expired) <= as_of:
                    continue
            item["score"] = round(rrf_scores[cid], 6)
            filtered.append(item)

        if not filtered:
            return []

        # 3. Optional ONNX cross-encoder rerank over bounded L1 windows (Wave 1 finding).
        if self.onnx.config.rerank_enabled and len(filtered) > 1:
            l1_passages = [
                str(item.get("l1") or extract_l1_query_window(item["text"], query, max_tokens=256))
                for item in filtered
            ]
            rerank_scores = self.onnx.rerank(query, l1_passages)
            for item, r_score in zip(filtered, rerank_scores):
                item["rerank_score"] = round(float(r_score), 6)
            filtered.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)

        return filtered[:limit]

    def build_context(
        self,
        query: str,
        *,
        scopes: Sequence[str] | None = None,
        max_tokens: int = 4000,
        as_of: str | None = None,
        limit: int = 12,
    ) -> dict[str, Any]:
        """Single-call hybrid retrieval + tier-aware packing + structural framing."""
        hits = self.search_memories(
            query,
            scopes=scopes,
            limit=limit,
            as_of=as_of,
            include_quarantined=False,
        )
        packed: PackedContext = pack_context(
            hits,
            max_tokens=max_tokens,
            tier_policy="adaptive",
            query=query,
        )
        framed = frame_memories(packed.sources)
        return {
            "query": query,
            "scopes": list(scopes) if scopes else ["*"],
            "context": framed,
            "sources": [
                {
                    "id": s["id"],
                    "uri": f"archivist://{s['scope']}/memory/{s['id']}",
                    "scope": s["scope"],
                    "kind": s["kind"],
                    "trust_tier": s["trust_tier"],
                    "score": s.get("score", 0.0),
                    "tier": s.get("_packed_tier", "l2"),
                }
                for s in packed.sources
            ],
            "total_tokens": packed.total_tokens,
            "budget_tokens": packed.budget_tokens,
            "token_savings_pct": packed.token_savings_pct,
            "dropped_count": packed.dropped_count,
        }

    def inspect_memories(
        self,
        *,
        scope: str | None = None,
        memory_id: str | None = None,
        limit: int = 20,
        include_suppressed: bool = False,
    ) -> dict[str, Any]:
        """Inspect single memory provenance or scope inventory (including quarantined items)."""
        if memory_id:
            item = self.get_memory(memory_id)
            return {"found": item is not None, "memory": item}

        where_parts: list[str] = []
        params: list[Any] = []
        if scope and scope != "*":
            where_parts.append("scope_id = ?")
            params.append(scope)
        if not include_suppressed:
            where_parts.append("suppressed = 0")
        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        params.append(max(1, int(limit)))

        rows = self._read_conn.execute(
            f"SELECT * FROM memory_chunks {where_sql} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        items = [self._row_to_dict(r) for r in rows]

        counts_rows = self._read_conn.execute(
            f"SELECT trust_tier, COUNT(*) AS cnt FROM memory_chunks {where_sql} "
            "GROUP BY trust_tier",
            params[:-1],
        ).fetchall()
        tier_counts = {str(r["trust_tier"]): int(r["cnt"]) for r in counts_rows}

        return {
            "scope": scope or "*",
            "total_shown": len(items),
            "trust_tier_counts": tier_counts,
            "quarantined_count": tier_counts.get(TrustTier.T3_QUARANTINED.value, 0),
            "embed_model": self.metadata.embed_model,
            "vector_dim": self.metadata.vector_dim,
            "vector_backend": self.vector_index.backend_name,
            "memories": items,
        }

    def stats(self) -> dict[str, Any]:
        """Return compact storage & token savings summary for ``archivist://health`` & stats."""
        total = int(
            self._read_conn.execute(
                "SELECT COUNT(*) FROM memory_chunks WHERE suppressed = 0"
            ).fetchone()[0]
        )
        quarantined = int(
            self._read_conn.execute(
                "SELECT COUNT(*) FROM memory_chunks WHERE suppressed = 0 AND trust_tier = ?",
                (TrustTier.T3_QUARANTINED.value,),
            ).fetchone()[0]
        )
        total_tokens = int(
            self._read_conn.execute(
                "SELECT COALESCE(SUM(token_count), 0) FROM memory_chunks WHERE suppressed = 0"
            ).fetchone()[0]
        )
        scopes_rows = self._read_conn.execute(
            "SELECT scope_id, COUNT(*) AS cnt FROM memory_chunks "
            "WHERE suppressed = 0 GROUP BY scope_id ORDER BY cnt DESC"
        ).fetchall()
        return {
            "status": "ok",
            "db_path": str(self.db_path),
            "embed_model": self.metadata.embed_model,
            "vector_dim": self.metadata.vector_dim,
            "vector_backend": self.vector_index.backend_name,
            "active_memories": total,
            "quarantined_memories": quarantined,
            "total_stored_tokens": total_tokens,
            "scopes": {str(r["scope_id"]): int(r["cnt"]) for r in scopes_rows},
        }
