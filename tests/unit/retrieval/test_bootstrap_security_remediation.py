"""INIT-004/SPEC-007 — bootstrap security remediations (M1/M2/M3).

M1: omitted namespace + agent_id resolves home ns and runs RBAC.
M2: bootstrap never serves stale wake-up cache (live build).
M3: pinned/recent facts SQL is namespace-scoped (no cross-tenant prose).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from archivist.app.handlers import tools_context
from archivist.retrieval.context_api import RelevantContext, get_relevant_context

pytestmark = [pytest.mark.unit, pytest.mark.coach_core]


async def _seed_permanent_fact(
    *,
    name: str,
    fact_text: str,
    agent_id: str,
    namespace: str,
) -> tuple[int, int]:
    from archivist.storage.graph import add_fact, upsert_entity

    eid = await upsert_entity(
        name,
        "person",
        retention_class="permanent",
        namespace=namespace,
    )
    fid = await add_fact(
        eid,
        fact_text,
        "test.md",
        agent_id,
        retention_class="permanent",
        namespace=namespace,
    )
    return eid, fid


async def _ensure_entities_unique_per_namespace() -> None:
    """Force UNIQUE(name, namespace) when init_schema migration was skipped under FKs."""
    from archivist.storage.sqlite_pool import pool

    async with pool.write() as conn:
        row = await conn.fetchone(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='entities'"
        )
        ddl = (row["sql"] if row else "") or ""
        compact = ddl.replace(" ", "").lower()
        if "unique(name,namespace)" in compact:
            return
        await conn.execute("PRAGMA foreign_keys=OFF")
        await conn.execute("DROP TABLE IF EXISTS entities_new")
        await conn.execute(
            """
            CREATE TABLE entities_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL COLLATE NOCASE,
                entity_type TEXT NOT NULL DEFAULT 'unknown',
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                mention_count INTEGER NOT NULL DEFAULT 1,
                metadata TEXT DEFAULT '{}',
                retention_class TEXT NOT NULL DEFAULT 'standard',
                aliases TEXT NOT NULL DEFAULT '[]',
                namespace TEXT NOT NULL DEFAULT 'global',
                actor_id TEXT NOT NULL DEFAULT '',
                actor_type TEXT NOT NULL DEFAULT '',
                UNIQUE(name, namespace)
            )
            """
        )
        await conn.execute(
            """
            INSERT INTO entities_new (
                id, name, entity_type, first_seen, last_seen, mention_count,
                metadata, retention_class, aliases, namespace, actor_id, actor_type
            )
            SELECT id, name, entity_type, first_seen, last_seen, mention_count,
                   metadata, retention_class, aliases, namespace,
                   COALESCE(actor_id, ''), COALESCE(actor_type, '')
            FROM entities
            """
        )
        await conn.execute("DROP TABLE entities")
        await conn.execute("ALTER TABLE entities_new RENAME TO entities")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE)"
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_namespace ON entities(namespace)"
        )
        await conn.execute("PRAGMA foreign_keys=ON")


class TestM3CrossNamespacePinnedFacts:
    async def test_bootstrap_excludes_foreign_namespace_permanent_prose(self, async_pool):
        """Same entity name in ns-a and ns-b — bootstrap for ns-a must not leak ns-b."""
        from archivist.storage.compressed_index import build_wake_up_context, format_wake_up_text

        await _ensure_entities_unique_per_namespace()

        entity = "SharedPerson007"
        ns_a, ns_b = "boot-sec-ns-a-007", "boot-sec-ns-b-007"
        agent = "boot-sec-agent-a"
        secret_b = "FOREIGN_NS_B_PERMANENT_SECRET_PHRASE_007"

        await _seed_permanent_fact(
            name=entity,
            fact_text=f"{entity} prefers oat milk in ns-a",
            agent_id=agent,
            namespace=ns_a,
        )
        await _seed_permanent_fact(
            name=entity,
            fact_text=f"{entity} {secret_b}",
            agent_id="boot-sec-agent-b",
            namespace=ns_b,
        )

        ctx = await build_wake_up_context(ns_a, agent_id=agent)
        l1 = ctx.get("l1_critical") or ""
        assert secret_b not in l1
        assert "oat milk" in l1 or entity in (ctx.get("l0_identity") or "")

        with patch(
            "archivist.core.rbac.list_accessible_namespaces",
            return_value=[{"namespace": ns_a, "can_read": True, "can_write": True}],
        ):
            answer = format_wake_up_text(ctx, agent_id=agent)
        assert secret_b not in answer

        with patch(
            "archivist.core.rbac.list_accessible_namespaces",
            return_value=[{"namespace": ns_a, "can_read": True, "can_write": True}],
        ):
            boot = await get_relevant_context(
                agent_id=agent,
                task_description="session start",
                namespace=ns_a,
                mode="bootstrap",
            )
        assert secret_b not in boot.answer


class TestM2SuppressLiveBootstrap:
    async def test_suppressed_prose_absent_after_live_bootstrap(self, async_pool):
        """M2: suppress then bootstrap — suppressed fact prose must be absent."""
        from archivist.storage.compressed_index import (
            build_wake_up_context,
            cache_wake_up,
            format_wake_up_text,
        )

        ns = "boot-suppress-ns-007"
        agent = "boot-suppress-agent"
        entity = "SuppressPerson007"
        prose = "SUPPRESSED_BOOTSTRAP_PROSE_MUST_VANISH_007"

        _eid, fid = await _seed_permanent_fact(
            name=entity,
            fact_text=f"{entity} {prose}",
            agent_id=agent,
            namespace=ns,
        )

        # Populate stale wake-up cache while fact is still visible
        cached = await cache_wake_up(ns, agent_id=agent)
        assert prose in (cached.get("l1_critical") or "")

        from archivist.storage.sqlite_pool import pool

        async with pool.write() as conn:
            await conn.execute("UPDATE facts SET is_suppressed = 1 WHERE id = ?", (fid,))

        # Live build (what bootstrap uses) must honor suppress even when cache is stale
        live = await build_wake_up_context(ns, agent_id=agent)
        assert prose not in (live.get("l1_critical") or "")

        with patch(
            "archivist.core.rbac.list_accessible_namespaces",
            return_value=[{"namespace": ns, "can_read": True, "can_write": True}],
        ):
            boot = await get_relevant_context(
                agent_id=agent,
                task_description="session start after suppress",
                namespace=ns,
                mode="bootstrap",
            )
        assert prose not in boot.answer
        assert "Bootstrap Context" in boot.answer or "Identity" in boot.answer

        with patch(
            "archivist.core.rbac.list_accessible_namespaces",
            return_value=[{"namespace": ns, "can_read": True, "can_write": True}],
        ):
            rendered = format_wake_up_text(live, agent_id=agent)
        assert prose not in rendered


class TestM1BootstrapNamespaceRbac:
    @pytest.mark.asyncio
    async def test_omitted_namespace_resolves_and_rbac_denies_cross_tenant(self):
        """M1: bootstrap without namespace resolves home ns then RBAC gates it."""
        denied = [MagicMock(text='{"error":"access_denied"}')]
        with (
            patch(
                "archivist.core.rbac.get_namespace_for_agent",
                return_value="home-ns-007",
            ) as resolve_ns,
            patch(
                "archivist.app.handlers.tools_context.require_rbac",
                return_value=denied,
            ) as rbac,
            patch(
                "archivist.retrieval.context_api.get_relevant_context",
                new=AsyncMock(),
            ) as get_ctx,
        ):
            out = await tools_context._handle_get_context(
                {
                    "agent_id": "alice",
                    "task_description": "session start",
                    "mode": "bootstrap",
                    # namespace omitted
                }
            )

        resolve_ns.assert_called_once_with("alice")
        rbac.assert_called_once_with("alice", "read", "home-ns-007")
        get_ctx.assert_not_called()
        assert out is denied

    @pytest.mark.asyncio
    async def test_omitted_namespace_passes_resolved_ns_to_bootstrap(self):
        """M1: when RBAC allows, resolved namespace is passed into get_relevant_context."""
        ctx = RelevantContext(
            answer="## Bootstrap Context\n**Identity:** Namespace: home-ns-007",
            sources=[],
            graph_facts=[],
            tips=[],
            total_tokens=20,
            budget_tokens=400,
            over_budget=False,
            tier_distribution={},
            token_savings_pct=0.0,
            provenance=[],
            pack_policy="bootstrap",
            memories=[],
            mode="bootstrap",
        )
        with (
            patch(
                "archivist.core.rbac.get_namespace_for_agent",
                return_value="home-ns-007",
            ),
            patch(
                "archivist.app.handlers.tools_context.require_rbac",
                return_value=None,
            ) as rbac,
            patch(
                "archivist.retrieval.context_api.get_relevant_context",
                new=AsyncMock(return_value=ctx),
            ) as get_ctx,
        ):
            out = await tools_context._handle_get_context(
                {
                    "agent_id": "alice",
                    "task_description": "session start",
                    "mode": "bootstrap",
                }
            )

        rbac.assert_called_once_with("alice", "read", "home-ns-007")
        assert get_ctx.await_args.kwargs["namespace"] == "home-ns-007"
        assert get_ctx.await_args.kwargs["mode"] == "bootstrap"
        payload = json.loads(out[0].text)
        assert payload["mode"] == "bootstrap"
        assert "error" not in payload

    @pytest.mark.asyncio
    async def test_bootstrap_skips_wake_up_cache(self):
        """M2: get_relevant_context(bootstrap) calls build_wake_up_context, not cache."""
        wake = {
            "l0_identity": "Namespace: ns-live; Agent: alice",
            "l1_critical": "No facts recorded yet.",
            "namespace_toc": "",
            "fleet_tips": [],
            "total_memories": 0,
            "last_activity": "",
            "top_entities": [],
        }
        with (
            patch(
                "archivist.storage.compressed_index.build_wake_up_context",
                new=AsyncMock(return_value=wake),
            ) as build,
            patch(
                "archivist.storage.compressed_index.get_cached_wake_up",
                new=AsyncMock(return_value={"l1_critical": "STALE_CACHE_SHOULD_NOT_APPEAR"}),
            ) as cached,
            patch(
                "archivist.storage.compressed_index.cache_wake_up",
                new=AsyncMock(return_value=wake),
            ) as cache_fn,
            patch(
                "archivist.core.rbac.list_accessible_namespaces",
                return_value=[],
            ),
        ):
            boot = await get_relevant_context(
                agent_id="alice",
                task_description="boot",
                namespace="ns-live",
                mode="bootstrap",
            )

        build.assert_awaited_once_with("ns-live", agent_id="alice")
        cached.assert_not_called()
        cache_fn.assert_not_called()
        assert "STALE_CACHE_SHOULD_NOT_APPEAR" not in boot.answer
