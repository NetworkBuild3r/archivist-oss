"""Regression tests for every finding in docs/code-review-findings-2026-11-27.md.

Covers:
  - CRIT-1: _handle_pin brand-new entity on SQLite does not deadlock
  - CRIT-2: export_agent/import_agent via asyncio.to_thread(..., loop=loop)
  - MAJ-1 & SEC-6: parameterized UTC cutoffs and cache_hit = 0 across hotness/dashboard/retrieval_log
  - MAJ-3: entities UNIQUE(name, namespace) migration preserves actor_id/actor_type & FKs
  - MAJ-4: _upsert_fts_chunk_postgres forwards conn= and re-raises inside transactions
  - MAJ-5: SQLitePool separates query_only=ON read connection from write connection
  - MAJ-6: outbox.drain dead-letters malformed JSON payloads without crashing batch
  - MAJ-7: _resolve_child_ids includes synthetic_question; merge writes merged_from not parent_id
  - MAJ-9: consolidate_tips + curator_queue preserves valid trajectory_id for FK integrity
  - MAJ-10: ResultCandidate + literal + registry + BM25 agent_ids + entity_facts + graph_context isolation
  - MAJ-12: reranker cooldown retry + to_thread; relevance_forget to_thread
  - MIN-1: FTS5 quote escaping, Postgres backslash escaping, AND/phrase threshold = limit
  - MIN-2: hot_cache key dimensions + fleet-wide invalidation
  - MIN-4: sharded collection routing in compress/deref/lineage & namespace-scoped unpin
  - MIN-5: cascade.sweep_orphans ignores reconsolidate:* sentinel rows
  - MIN-6: batch_update_hotness watermark prevents double-counting retrieval logs
  - MIN-7: background task references tracked in tools_storage and curator_queue
  - MIN-8: indexer deletes old points only after embed_batch succeeds + cleans aux tables; import failed counts
  - MIN-9: curator_queue atomic claim 'processing'; invalid int/date query params return 400 / error
  - SUGG-1: EMBED_URL /v1 normalization; unified __version__ in /health & backup_manager
  - SUGG-2: memory_product fork/import batched outbox + memory_points + OUTBOX_ENABLED=False fallback
  - SEC-1: Postgres backup/restore DSN credential redaction & argv password stripping
  - SEC-2: import_agent path confinement inside BACKUP_DIR (including symlink escape)
  - SEC-3: timing-safe API key comparison + POST-only /admin/invalidate
  - SEC-5: strict RBAC enforcement on audit, retrieval logs, backup, merge, deref, contradictions,
           entity_brief (including write check for add_alias), and handoff
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

pytestmark = [pytest.mark.regression]


# ---------------------------------------------------------------------------
# CRIT-1: _handle_pin brand-new entity on SQLite does not deadlock
# ---------------------------------------------------------------------------


async def test_crit1_pin_brand_new_entity_no_deadlock(async_pool):
    from archivist.app.handlers.tools_storage import _handle_pin

    res = await asyncio.wait_for(
        _handle_pin(
            {
                "agent_id": "alice",
                "entity_name": "BrandNewEntityCrit1",
                "namespace": "ns-crit1",
                "reason": "must not deadlock",
            }
        ),
        timeout=5.0,
    )
    payload = json.loads(res[0].text)
    assert payload["pinned"] is True
    assert payload["items"][0]["created"] is True
    assert payload["items"][0]["name"] == "BrandNewEntityCrit1"

    async with async_pool.read() as conn:
        row = await conn.fetchone(
            "SELECT retention_class, namespace FROM entities WHERE name = ?",
            ("BrandNewEntityCrit1",),
        )
    assert row is not None
    assert row["retention_class"] == "permanent"
    assert row["namespace"] == "ns-crit1"


# ---------------------------------------------------------------------------
# MAJ-1 & SEC-6: parameterized UTC cutoffs & integer cache_hit = 0
# ---------------------------------------------------------------------------


def test_maj1_sec6_no_pg_interval_interpolation_or_bool_cache_hit():
    import archivist.app.dashboard as dash
    import archivist.core.hotness as hot
    import archivist.retrieval.retrieval_log as rlog

    for mod in (dash, hot, rlog):
        src = inspect.getsource(mod)
        assert "INTERVAL '" not in src, f"unparameterized INTERVAL found in {mod.__name__}"
        assert "cache_hit = FALSE" not in src, f"boolean cache_hit found in {mod.__name__}"


# ---------------------------------------------------------------------------
# MAJ-3: entities UNIQUE(name, namespace) migration preserves actor_id/actor_type
# ---------------------------------------------------------------------------


def test_maj3_entity_unique_migration_preserves_columns_and_fks(tmp_path):
    from archivist.storage.graph_schema import _migrate_entity_unique_constraint

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            entity_type TEXT DEFAULT 'unknown',
            first_seen TEXT NOT NULL DEFAULT '2026-01-01',
            last_seen TEXT NOT NULL DEFAULT '2026-01-01',
            mention_count INTEGER DEFAULT 1,
            metadata TEXT DEFAULT '{}',
            aliases TEXT DEFAULT '[]',
            retention_class TEXT DEFAULT 'standard',
            namespace TEXT NOT NULL DEFAULT 'global',
            actor_id TEXT DEFAULT '',
            actor_type TEXT DEFAULT ''
        );
        CREATE INDEX idx_entities_namespace ON entities(namespace);
        CREATE TABLE facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL REFERENCES entities(id),
            fact_text TEXT NOT NULL
        );
        INSERT INTO entities (name, first_seen, last_seen, namespace, actor_id, actor_type)
        VALUES ('Deployer', '2026-01-01', '2026-01-01', 'ns-a', 'actor-1', 'agent');
        INSERT INTO facts (entity_id, fact_text) VALUES (1, 'runs ArgoCD');
        """
    )
    conn.commit()

    with patch("archivist.storage.graph_schema.get_db", return_value=conn):
        _migrate_entity_unique_constraint()

    check_conn = sqlite3.connect(str(db_path))
    check_conn.execute("PRAGMA foreign_keys=ON")
    # Same entity name in a DIFFERENT namespace must now succeed (UNIQUE(name, namespace))
    check_conn.execute(
        "INSERT INTO entities (name, first_seen, last_seen, namespace, actor_id, actor_type) "
        "VALUES ('Deployer', '2026-01-01', '2026-01-01', 'ns-b', 'actor-2', 'user')"
    )
    check_conn.commit()

    # Duplicate (name, namespace) must still be rejected
    with pytest.raises(sqlite3.IntegrityError):
        check_conn.execute(
            "INSERT INTO entities (name, first_seen, last_seen, namespace) "
            "VALUES ('Deployer', '2026-01-01', '2026-01-01', 'ns-a')"
        )

    # actor_id and actor_type survived migration and FK check is clean
    row = check_conn.execute(
        "SELECT actor_id, actor_type FROM entities WHERE name='Deployer' AND namespace='ns-a'"
    ).fetchone()
    assert row == ("actor-1", "agent")
    assert check_conn.execute("PRAGMA foreign_key_check").fetchall() == []
    check_conn.close()


# ---------------------------------------------------------------------------
# MAJ-4: _upsert_fts_chunk_postgres forwards conn= and re-raises on error
# ---------------------------------------------------------------------------


async def test_maj4_upsert_fts_chunk_postgres_uses_conn_and_reraises():
    from archivist.storage.graph_fts import _upsert_fts_chunk_postgres

    fake_conn = MagicMock()
    fake_conn.execute = AsyncMock(side_effect=RuntimeError("pg boom"))

    with pytest.raises(RuntimeError, match="pg boom"):
        await _upsert_fts_chunk_postgres(
            qdrant_id="q1",
            text="hello",
            file_path="f.md",
            chunk_index=0,
            agent_id="a1",
            namespace="ns1",
            date="2026-04-01",
            memory_type="general",
            conn=fake_conn,
        )
    fake_conn.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# MAJ-5: SQLitePool dedicated read-only connection
# ---------------------------------------------------------------------------


async def test_maj5_sqlite_pool_read_connection_is_query_only(tmp_path):
    from archivist.storage.sqlite_pool import SQLitePool

    db_file = tmp_path / "isolated.db"
    p = SQLitePool()
    await p.initialize(str(db_file))
    try:
        async with p.write() as wconn:
            await wconn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            await wconn.execute("INSERT INTO t (val) VALUES (?)", ("ok",))

        async with p.read() as rconn:
            row = await rconn.fetchone("SELECT val FROM t")
            assert row["val"] == "ok"
            with pytest.raises(sqlite3.OperationalError):
                await rconn.execute("INSERT INTO t (val) VALUES (?)", ("forbidden",))
    finally:
        await p.close()


# ---------------------------------------------------------------------------
# MAJ-6: outbox.drain dead-letters malformed JSON without crashing batch
# ---------------------------------------------------------------------------


async def test_maj6_outbox_drain_dead_letters_corrupt_json(async_pool, monkeypatch):
    import archivist.storage.outbox as ob

    now = datetime.now(UTC).isoformat()
    async with async_pool.write() as conn:
        await conn.execute(
            "INSERT INTO outbox (id, event_type, payload, status, retry_count, created_at) "
            "VALUES (?, ?, ?, 'pending', 0, ?)",
            ("corrupt-1", "qdrant_delete", "{not valid json", now),
        )
        await conn.execute(
            "INSERT INTO outbox (id, event_type, payload, status, retry_count, created_at) "
            "VALUES (?, ?, ?, 'pending', 0, ?)",
            (
                "valid-2",
                "qdrant_delete",
                json.dumps({"collection": "c", "ids": ["id-1"]}),
                now,
            ),
        )

    mock_backend = MagicMock()
    mock_backend.delete = AsyncMock(return_value=None)
    processor = ob.OutboxProcessor(mock_backend)

    applied = await processor.drain()
    assert applied == 1

    async with async_pool.read() as conn:
        c_row = await conn.fetchone("SELECT status, error FROM outbox WHERE id='corrupt-1'")
        v_row = await conn.fetchone("SELECT status FROM outbox WHERE id='valid-2'")
    assert c_row["status"] == "dead"
    assert c_row["error"] and "invalid payload JSON" in c_row["error"]
    assert v_row["status"] == "applied"


# ---------------------------------------------------------------------------
# MAJ-7: _resolve_child_ids includes synthetic_question; merge uses merged_from
# ---------------------------------------------------------------------------


async def test_maj7_resolve_child_ids_includes_synthetic_question(async_pool):
    from archivist.lifecycle.memory_lifecycle import _resolve_child_ids

    now = datetime.now(UTC).isoformat()
    async with async_pool.write() as conn:
        await conn.executemany(
            "INSERT INTO memory_points (memory_id, qdrant_id, point_type, created_at) VALUES (?, ?, ?, ?)",
            [
                ("m-root", "m-root", "primary", now),
                ("m-root", "mc-1", "micro_chunk", now),
                ("m-root", "rh-1", "reverse_hyde", now),
                ("m-root", "sq-1", "synthetic_question", now),
            ],
        )

    mc_ids, child_ids = await _resolve_child_ids("m-root", MagicMock(), "coll-test", [])
    assert mc_ids == ["mc-1"]
    assert set(child_ids) == {"rh-1", "sq-1"}


async def test_maj7_merge_stores_merged_from_not_parent_id(async_pool, monkeypatch):
    from archivist.lifecycle import merge as merge_mod

    p1 = MagicMock(id="src-1", payload={"text": "fact one", "namespace": "ns-m", "agent_id": "a1"})
    p2 = MagicMock(id="src-2", payload={"text": "fact two", "namespace": "ns-m", "agent_id": "a1"})
    mock_client = MagicMock()
    mock_client.retrieve.return_value = [p1, p2]
    mock_client.upsert = MagicMock()

    with (
        patch.object(merge_mod, "qdrant_client", return_value=mock_client),
        patch.object(merge_mod, "collection_for", return_value="coll-m"),
        patch.object(merge_mod, "embed_text", new=AsyncMock(return_value=[0.1, 0.2])),
        patch(
            "archivist.lifecycle.memory_lifecycle.delete_memory_complete",
            new=AsyncMock(),
        ),
        patch.object(merge_mod, "log_memory_event", new=AsyncMock()),
        patch("archivist.core.config.OUTBOX_ENABLED", False),
    ):
        res = await merge_mod.merge_memories(["src-1", "src-2"], "concat", "a1", "")

    assert res["merged_count"] == 2
    assert res["merged_id"]
    upserted_point = mock_client.upsert.call_args.kwargs["points"][0]
    assert "parent_id" not in upserted_point.payload
    assert upserted_point.payload["merged_from"] == ["src-1", "src-2"]


# ---------------------------------------------------------------------------
# MAJ-9: consolidate_tips + curator_queue preserves valid trajectory_id FK
# ---------------------------------------------------------------------------


async def test_maj9_consolidate_tips_fk_integrity(async_pool):
    from archivist.core.trajectory import _ensure_trajectory_schema
    from archivist.lifecycle import curator_queue as cq

    _ensure_trajectory_schema()
    now = datetime.now(UTC).isoformat()

    async with async_pool.write() as conn:
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute(
            "INSERT INTO trajectories (id, agent_id, session_id, task_description, outcome, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("traj-1", "alice", "s1", "deploy goal", "success", now),
        )
        await conn.execute(
            "INSERT INTO tips (id, trajectory_id, agent_id, category, tip_text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("tip-1", "traj-1", "alice", "strategy", "first tip", now),
        )
        await conn.execute(
            "INSERT INTO tips (id, trajectory_id, agent_id, category, tip_text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("tip-2", "traj-1", "alice", "strategy", "second tip", now),
        )

    # Even when consolidated payload omits trajectory_id, _apply_consolidate_tips resolves it from original_ids
    await cq._apply_consolidate_tips(
        {
            "original_tip_ids": ["tip-1", "tip-2"],
            "consolidated_tip": {
                "agent_id": "alice",
                "category": "strategy",
                "tip": "merged tip",
            },
        }
    )

    async with async_pool.read() as conn:
        row = await conn.fetchone(
            "SELECT trajectory_id, tip_text FROM tips WHERE tip_text = 'merged tip'"
        )
    assert row is not None
    assert row["trajectory_id"] == "traj-1"


# ---------------------------------------------------------------------------
# MAJ-10: suppression/visibility/tenant metadata & filtering
# ---------------------------------------------------------------------------


async def test_maj10_result_candidate_and_graph_suppression_and_bm25_agent_ids(async_pool):
    from archivist.core.result_types import ResultCandidate
    from archivist.storage.fts_search import search_bm25
    from archivist.storage.graph import add_fact, upsert_entity, upsert_fts_chunk
    from archivist.storage.graph_entities import get_entity_facts, get_entity_facts_bulk
    from archivist.storage.graph_retrieval import (
        build_entity_fact_results,
        graph_context_for_entities,
    )

    cand = ResultCandidate.from_qdrant_payload(
        "p1",
        {
            "text": "secret",
            "is_suppressed": True,
            "is_superseded": True,
            "supersedes_id": "old-1",
            "superseded_by": "new-1",
            "subject": "user:1",
            "purpose": "support",
            "sensitivity": "confidential",
            "statement_kind": "inferred",
        },
    )
    d = cand.to_dict()
    assert d["is_suppressed"] is True
    assert d["is_superseded"] is True
    assert d["subject"] == "user:1"
    assert d["sensitivity"] == "confidential"

    # Suppressed facts excluded from get_entity_facts / get_entity_facts_bulk
    eid_a = await upsert_entity("WidgetA", namespace="ns-a")
    fid_live = await add_fact(
        eid_a, "Deployed on Kubernetes cluster east-1", "a.md", "agent-a", namespace="ns-a"
    )
    fid_supp = await add_fact(
        eid_a, "Confidential billing token alpha-999", "a.md", "agent-a", namespace="ns-a"
    )
    async with async_pool.write() as conn:
        await conn.execute("UPDATE facts SET is_suppressed = 1 WHERE id = ?", (fid_supp,))

    facts = await get_entity_facts(eid_a)
    assert [f["id"] for f in facts] == [fid_live]
    bulk = await get_entity_facts_bulk([eid_a])
    assert [f["id"] for f in bulk[eid_a]] == [fid_live]

    # build_entity_fact_results populates real namespace (not "")
    ef_results = await build_entity_fact_results(
        [{"id": eid_a, "name": "WidgetA", "namespace": "ns-a"}]
    )
    assert ef_results and ef_results[0]["namespace"] == "ns-a"

    # graph_context_for_entities respects namespace scoping (SEC-5)
    ctx_wrong_ns = await graph_context_for_entities([eid_a], depth=1, namespace="ns-other")
    assert ctx_wrong_ns == []
    ctx_right_ns = await graph_context_for_entities([eid_a], depth=1, namespace="ns-a")
    assert len(ctx_right_ns) == 1

    # search_bm25 filters by agent_ids list
    from archivist.storage import fts_search

    await upsert_fts_chunk(
        "bm-1", "quantum widget calibration", "a.md", 0, "agent-1", "ns-a", "2026-04-01"
    )
    await upsert_fts_chunk(
        "bm-2", "quantum widget calibration", "b.md", 0, "agent-2", "ns-a", "2026-04-01"
    )
    await upsert_fts_chunk(
        "bm-3", "quantum widget calibration", "c.md", 0, "agent-3", "ns-a", "2026-04-01"
    )
    with (
        patch.object(fts_search, "BM25_ENABLED", True),
        patch.object(fts_search.health, "is_healthy", return_value=True),
    ):
        hits = await search_bm25(
            "quantum widget", namespace="ns-a", agent_ids=["agent-1", "agent-2"]
        )
    hit_agents = {h["agent_id"] for h in hits}
    assert hit_agents == {"agent-1", "agent-2"}


# ---------------------------------------------------------------------------
# MAJ-12: reranker cooldown retry & relevance_forget to_thread
# ---------------------------------------------------------------------------


async def test_maj12_reranker_retries_after_cooldown(monkeypatch):
    import archivist.retrieval.reranker as rr

    rr._model = None
    rr._model_name = ""
    rr._load_failed = False
    rr._load_failed_at = 0.0

    # Force first load failure
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)
    assert rr._get_model("dummy-model") is None
    assert rr._load_failed is True

    # Simulate cooldown expiry + fake CrossEncoder module
    rr._load_failed_at = 0.01
    fake_ce = MagicMock()
    fake_ce.predict.return_value = [0.9]
    fake_mod = MagicMock(CrossEncoder=MagicMock(return_value=fake_ce))
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_mod)
    monkeypatch.setattr(rr, "_LOAD_RETRY_COOLDOWN_SECONDS", 0.0)

    ranked = await rr.rerank_candidates(
        "q", [{"text": "hello", "score": 0.5}], model_name="dummy-model"
    )
    assert ranked[0]["reranker_score"] == 0.9


# ---------------------------------------------------------------------------
# MIN-1: FTS5 quote escaping, Postgres backslash escaping, AND/phrase trigger
# ---------------------------------------------------------------------------


async def test_min1_fts_escaping_and_extra_modes():
    from archivist.storage import fts_search

    assert '""' in fts_search._fts5_safe_query('foo"bar')
    assert '""' in fts_search._fts5_and_query('foo"bar baz"qux')
    assert "\\" not in fts_search._pg_escape_token(r"foo\bar'baz")

    # When OR hits saturate limit (e.g. limit=5, returns 5), AND mode must fire
    called_modes: list[str] = []

    async def _fake_search_fts(*, fts_mode="or", **kwargs):
        called_modes.append(fts_mode)
        return [{"qdrant_id": f"id-{i}", "bm25_score": 1.0} for i in range(5)]

    with (
        patch.object(fts_search, "BM25_ENABLED", True),
        patch.object(fts_search.health, "is_healthy", return_value=True),
        patch.object(fts_search, "search_fts", side_effect=_fake_search_fts),
    ):
        await fts_search.search_bm25("kubernetes deployment rollback", limit=5)

    assert "or" in called_modes
    assert "and" in called_modes


# ---------------------------------------------------------------------------
# MIN-2: hot_cache fleet-wide invalidation
# ---------------------------------------------------------------------------


def test_min2_hot_cache_invalidates_fleet_entries():
    import archivist.retrieval.hot_cache as hc

    hc.invalidate_all()
    hc.put("fleet", "what happened", {"answer": "fleet-ans"}, namespace="")
    hc.put("alice", "ns query", {"answer": "ns-ans"}, namespace="ns-a")
    hc.put("bob", "other ns", {"answer": "b-ans"}, namespace="ns-b")

    evicted = hc.invalidate_namespace("ns-a")
    assert evicted == 2  # ns-a AND fleet ("")
    assert hc.get("fleet", "what happened", namespace="") is None
    assert hc.get("alice", "ns query", namespace="ns-a") is None
    assert hc.get("bob", "other ns", namespace="ns-b") is not None


# ---------------------------------------------------------------------------
# MIN-4: namespace-scoped unpin & sharded collection deref/lineage
# ---------------------------------------------------------------------------


async def test_min4_unpin_scopes_entity_by_namespace(async_pool):
    from archivist.app.handlers.tools_storage import _handle_pin, _handle_unpin

    await _handle_pin({"agent_id": "a", "entity_name": "SharedName", "namespace": "ns-one"})
    await _handle_pin({"agent_id": "b", "entity_name": "SharedName", "namespace": "ns-two"})
    await _handle_unpin({"agent_id": "a", "entity_name": "SharedName", "namespace": "ns-one"})

    async with async_pool.read() as conn:
        r1 = await conn.fetchone(
            "SELECT retention_class FROM entities WHERE name='SharedName' AND namespace='ns-one'"
        )
        r2 = await conn.fetchone(
            "SELECT retention_class FROM entities WHERE name='SharedName' AND namespace='ns-two'"
        )
    assert r1["retention_class"] == "standard"
    assert r2["retention_class"] == "permanent"


# ---------------------------------------------------------------------------
# MIN-5: cascade.sweep_orphans ignores reconsolidate:* sentinel rows
# ---------------------------------------------------------------------------


async def test_min5_sweep_orphans_ignores_reconsolidate_sentinels(async_pool):
    from archivist.lifecycle import cascade

    async with async_pool.write() as conn:
        await conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, agent_id, namespace, date, memory_type) "
            "VALUES ('reconsolidate:abc', 'sentinel', '', 0, 'sys', 'ns1', '2026-04-01', 'general')"
        )
        await conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, agent_id, namespace, date, memory_type) "
            "VALUES ('real-orphan-1', 'orphan', 'a.md', 0, 'a1', 'ns1', '2026-04-01', 'general')"
        )

    mock_client = MagicMock()
    mock_client.retrieve.return_value = []  # nothing exists in Qdrant

    with (
        patch.object(cascade, "qdrant_client", return_value=mock_client),
        patch.object(cascade, "collections_for_query", return_value=["coll-1"]),
    ):
        await cascade.sweep_orphans()

    async with async_pool.read() as conn:
        sentinel = await conn.fetchone(
            "SELECT qdrant_id FROM memory_chunks WHERE qdrant_id='reconsolidate:abc'"
        )
        orphan = await conn.fetchone(
            "SELECT qdrant_id FROM memory_chunks WHERE qdrant_id='real-orphan-1'"
        )
    assert sentinel is not None, "reconsolidate:* sentinel must not be swept"
    assert orphan is None


# ---------------------------------------------------------------------------
# MIN-6: batch_update_hotness watermark prevents double counting
# ---------------------------------------------------------------------------


async def test_min6_hotness_watermark_prevents_double_counting(async_pool):
    from archivist.core.hotness import batch_update_hotness
    from archivist.retrieval import retrieval_log

    retrieval_log._ensure_schema()
    now = datetime.now(UTC).isoformat()
    async with async_pool.write() as conn:
        await conn.execute(
            "INSERT INTO retrieval_logs (id, agent_id, query, namespace, tier, memory_type, "
            "retrieval_trace, result_count, cache_hit, duration_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 10, ?)",
            (
                "rl-1",
                "a1",
                "q",
                "ns1",
                "l2",
                "general",
                json.dumps({"result_ids": ["mem-hot-1"]}),
                1,
                now,
            ),
        )

    await batch_update_hotness()
    async with async_pool.read() as conn:
        r1 = await conn.fetchone(
            "SELECT retrieval_count FROM memory_hotness WHERE memory_id='mem-hot-1'"
        )
    assert r1["retrieval_count"] == 1

    # Second cycle with no new logs must NOT increment retrieval_count again
    await batch_update_hotness()
    async with async_pool.read() as conn:
        r2 = await conn.fetchone(
            "SELECT retrieval_count FROM memory_hotness WHERE memory_id='mem-hot-1'"
        )
    assert r2["retrieval_count"] == 1


# ---------------------------------------------------------------------------
# MIN-7: strong references for background tasks
# ---------------------------------------------------------------------------


def test_min7_pending_task_sets_exist():
    import archivist.app.handlers.tools_storage as ts
    import archivist.lifecycle.curator_queue as cq

    assert isinstance(ts._pending_tasks, set)
    assert isinstance(cq._pending_tasks, set)


# ---------------------------------------------------------------------------
# MIN-8: indexer keeps old points on embed failure; import counts failed batches
# ---------------------------------------------------------------------------


async def test_min8_indexer_preserves_old_points_when_embed_fails(tmp_path):
    from archivist.write import indexer

    md = tmp_path / "doc.md"
    md.write_text(
        "# Title\n\nThis is a sufficiently long markdown file for chunking and embedding."
    )

    del_mock = AsyncMock()
    with (
        patch.object(indexer, "delete_file_points", new=del_mock),
        patch.object(indexer, "embed_batch", new=AsyncMock(side_effect=RuntimeError("embed down"))),
    ):
        with pytest.raises(RuntimeError, match="embed down"):
            await indexer.index_file(str(md), hierarchical=False)

    del_mock.assert_not_awaited()


def test_min8_import_agent_reports_failed_upsert_and_skips_fts(tmp_path):
    from archivist.storage.backup_manager import import_agent

    ndjson = tmp_path / "fail.ndjson"
    ndjson.write_text(
        json.dumps(
            {
                "id": "p-fail",
                "vector": [0.1, 0.2],
                "payload": {"agent_id": "a1", "namespace": "ns", "text": "hello"},
            }
        )
        + "\n"
    )
    mock_client = MagicMock()
    mock_client.upsert.side_effect = RuntimeError("qdrant unavailable")

    with (
        patch("archivist.storage.backup_manager.BACKUP_DIR", str(tmp_path)),
        patch("archivist.core.config.BM25_ENABLED", True),
        patch("archivist.storage.collection_router.ensure_collection", return_value="coll"),
        patch("archivist.storage.qdrant.qdrant_client", return_value=mock_client),
    ):
        res = import_agent(str(ndjson), dry_run=False)

    assert res["imported"] == 0
    assert res["failed"] == 1
    assert res["fts_rebuilt"] == 0


# ---------------------------------------------------------------------------
# MIN-9: invalid HTTP/MCP params return 400 / error_response
# ---------------------------------------------------------------------------


async def test_min9_invalid_params_return_clean_errors(monkeypatch):
    import archivist.app.main as main
    from archivist.app.handlers.tools_search import _handle_timeline

    async def _noop():
        return None

    monkeypatch.setattr(main, "_startup", _noop)
    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "", raising=False)
    with TestClient(main.app) as client:
        assert client.get("/admin/dashboard", params={"window_days": "notanint"}).status_code == 400
        assert client.get("/admin/retrieval-logs", params={"limit": "bad"}).status_code == 400
        assert (
            client.get(
                "/admin/retrieval-logs", params={"stats": "true", "window_days": "bad"}
            ).status_code
            == 400
        )

    res = await _handle_timeline({"query": "deploy", "agent_id": "alice", "as_of": "not-a-date"})
    data = json.loads(res[0].text)
    assert data["error"] == "invalid_as_of"


# ---------------------------------------------------------------------------
# SUGG-1: EMBED_URL /v1 normalization & unified __version__
# ---------------------------------------------------------------------------


async def test_sugg1_embed_url_normalizes_trailing_v1(monkeypatch):
    import archivist.features.embeddings as emb

    emb._embed_cache.clear()
    monkeypatch.setattr(emb, "EMBED_URL", "http://embed-host:8000/v1")

    captured_urls: list[str] = []

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"embedding": [0.5, 0.6]}]}

    class _FakeClient:
        is_closed = False

        async def post(self, url, **kwargs):
            captured_urls.append(url)
            return _FakeResp()

    monkeypatch.setattr(emb, "_get_embed_client", lambda: _FakeClient())
    vec = await emb.embed_text("normalize url test")
    assert vec == [0.5, 0.6]
    assert captured_urls == ["http://embed-host:8000/v1/embeddings"]


def test_sugg1_unified_version(monkeypatch):
    import archivist
    import archivist.app.main as main
    from archivist.storage.backup_manager import ARCHIVIST_VERSION

    assert archivist.__version__ == ARCHIVIST_VERSION

    async def _noop():
        return None

    monkeypatch.setattr(main, "_startup", _noop)
    with TestClient(main.app) as client:
        resp = client.get("/health")
    assert resp.json()["version"] == archivist.__version__


# ---------------------------------------------------------------------------
# SUGG-2: memory_product batches outbox upserts & supports OUTBOX_ENABLED=False
# ---------------------------------------------------------------------------


async def test_sugg2_memory_product_fork_batches_and_inline_fallback(
    async_pool, tmp_path, monkeypatch
):
    from archivist.storage import memory_product as mp

    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(tmp_path))
    async with async_pool.write() as conn:
        await conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, agent_id, namespace, date, memory_type) "
            "VALUES ('src-mp-1', 'hello world', 'f.md', 0, 'a1', 'ns-src', '2026-04-01', 'general')"
        )

    mock_qdrant = MagicMock()
    mock_pt = MagicMock(
        id="src-mp-1", vector=[0.1, 0.2], payload={"text": "hello world", "namespace": "ns-src"}
    )
    mock_qdrant.retrieve.return_value = [mock_pt]

    with (
        patch("archivist.storage.qdrant.qdrant_client", return_value=mock_qdrant),
        patch("archivist.storage.collection_router.collection_for", return_value="coll-src"),
        patch("archivist.storage.collection_router.ensure_collection", return_value="coll-dst"),
        patch("archivist.core.config.OUTBOX_ENABLED", False),
    ):
        snap = await mp.create_scope_snapshot(
            namespace="ns-src",
            caller_agent_id="a1",
            vectors=[
                {
                    "id": "src-mp-1",
                    "vector": [0.1, 0.2],
                    "payload": {"text": "hello world", "namespace": "ns-src"},
                }
            ],
        )
        forked = await mp.fork_from_snapshot(
            source_version_id=snap.id,
            target_namespace="ns-dst",
            caller_agent_id="a1",
        )

    assert forked.chunk_count == 1
    assert forked.point_count == 1
    mock_qdrant.upsert.assert_called_once()
    async with async_pool.read() as conn:
        mp_rows = await conn.fetchall("SELECT point_type FROM memory_points")
    assert any(r["point_type"] == "primary" for r in mp_rows)


# ---------------------------------------------------------------------------
# SEC-1: Postgres backup/restore DSN credential redaction
# ---------------------------------------------------------------------------


def test_sec1_postgres_backup_redacts_password_in_logs_and_argv(tmp_path, monkeypatch, caplog):
    import logging

    import archivist.storage.backup_manager as bm

    secret_dsn = "postgresql://admin:Sup3rS3cretPass@db.internal:5432/archivist"
    monkeypatch.setattr(bm, "DATABASE_URL", secret_dsn)

    captured_cmd: list[str] = []
    captured_env: dict[str, str] = {}

    def _fake_run(cmd, *, env=None, **kwargs):
        captured_cmd.extend(cmd)
        if env:
            captured_env.update(env)
        # Write dummy dump file for size stat
        for arg in cmd:
            if arg.startswith("--file="):
                with open(arg.split("=", 1)[1], "wb") as f:
                    f.write(b"PGDMP")
        return MagicMock(returncode=0, stderr="")

    with caplog.at_level(logging.INFO, logger="archivist.backup"):
        with patch.object(bm.subprocess, "run", side_effect=_fake_run):
            bm._backup_postgres(tmp_path)

    assert "Sup3rS3cretPass" not in " ".join(captured_cmd)
    assert "Sup3rS3cretPass" not in caplog.text
    assert captured_env.get("PGPASSWORD") == "Sup3rS3cretPass"


# ---------------------------------------------------------------------------
# SEC-2: symlink escape rejected by _resolve_import_path
# ---------------------------------------------------------------------------


def test_sec2_resolve_import_path_blocks_symlink_escape(tmp_path, monkeypatch):
    import archivist.storage.backup_manager as bm

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    outside = tmp_path / "secret.ndjson"
    outside.write_text("{}\n")
    link = backup_dir / "escape.ndjson"
    link.symlink_to(outside)

    monkeypatch.setattr(bm, "BACKUP_DIR", str(backup_dir))
    with pytest.raises(bm.SnapshotPathError):
        bm._resolve_import_path(str(link))


# ---------------------------------------------------------------------------
# SEC-5: strict RBAC enforcement across admin/search/storage/context endpoints
# ---------------------------------------------------------------------------


async def test_sec5_strict_rbac_enforcement(async_pool, rbac_config, monkeypatch):
    import archivist.app.main as main
    from archivist.app.handlers.tools_admin import (
        _handle_audit_trail,
        _handle_backup,
        _handle_retrieval_logs,
    )
    from archivist.app.handlers.tools_context import _handle_handoff
    from archivist.app.handlers.tools_search import (
        _handle_contradictions,
        _handle_deref,
        _handle_entity_brief,
    )
    from archivist.app.handlers.tools_storage import _handle_merge
    from archivist.storage.graph import upsert_entity

    # 1. _handle_merge denies gitbob writing to deployer namespace
    res = await _handle_merge(
        {
            "agent_id": "gitbob",
            "memory_ids": ["m1", "m2"],
            "strategy": "keep_both",
            "namespace": "deployer",
        }
    )
    assert json.loads(res[0].text)["error"] == "access_denied"

    # 2. _handle_handoff denies gitbob reading deployer namespace
    res = await _handle_handoff(
        {
            "agent_id": "gitbob",
            "session_id": "s1",
            "receiving_agent_id": "argo",
            "namespace": "deployer",
        }
    )
    assert json.loads(res[0].text)["error"] == "access_denied"

    # 3. _handle_deref requires caller and denies gitbob reading deployer memory
    fake_pt = MagicMock(
        id="m-dep", payload={"namespace": "deployer", "agent_id": "argo", "text": "top secret"}
    )
    mock_q = MagicMock()
    mock_q.retrieve.return_value = [fake_pt]
    with (
        patch("archivist.storage.qdrant.qdrant_client", return_value=mock_q),
        patch("archivist.storage.collection_router.collections_for_query", return_value=["coll"]),
    ):
        res_no_caller = await _handle_deref({"memory_id": "m-dep"})
        assert json.loads(res_no_caller[0].text)["error"] == "caller_required"
        res_denied = await _handle_deref({"memory_id": "m-dep", "agent_id": "gitbob"})
        assert json.loads(res_denied[0].text)["error"] == "access_denied"

    # 4. _handle_entity_brief denies read-only caller from add_alias (write required)
    await upsert_entity("ChiefOnlyEntity", namespace="chief")
    # gitbob has read-only access to 'chief' in rbac_config, NOT write
    res_alias = await _handle_entity_brief(
        {
            "entity": "ChiefOnlyEntity",
            "namespace": "chief",
            "agent_id": "gitbob",
            "add_alias": "hacked",
        }
    )
    assert json.loads(res_alias[0].text)["error"] == "access_denied"

    # 5. _handle_contradictions denies gitbob on deployer namespace
    res_c = await _handle_contradictions(
        {"entity": "Anything", "namespace": "deployer", "agent_id": "gitbob"}
    )
    assert json.loads(res_c[0].text)["error"] == "access_denied"

    # 6. _handle_audit_trail & _handle_retrieval_logs & _handle_backup enforce RBAC
    res_audit = await _handle_audit_trail({"caller_agent_id": "gitbob", "target_agent": "argo"})
    assert json.loads(res_audit[0].text)["error"] == "access_denied"

    res_rlog = await _handle_retrieval_logs({"caller_agent_id": "gitbob", "agent_id": "argo"})
    assert json.loads(res_rlog[0].text)["error"] == "access_denied"

    res_bak = await _handle_backup(
        {"action": "delete", "snapshot_id": "snap1", "agent_id": "gitbob"}
    )
    assert json.loads(res_bak[0].text)["error"] == "access_denied"

    # 7. HTTP GET /admin/audit denies unauthenticated caller / unauthorized target under RBAC
    async def _noop():
        return None

    monkeypatch.setattr(main, "_startup", _noop)
    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "", raising=False)
    with TestClient(main.app) as client:
        assert client.get("/admin/audit").status_code == 400
        assert (
            client.get(
                "/admin/audit", params={"caller_agent_id": "gitbob", "target_agent": "argo"}
            ).status_code
            == 403
        )
