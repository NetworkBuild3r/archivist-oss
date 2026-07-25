"""Regression tests for INIT-022/SPEC-006's H1 fix.

``_apply_merge`` used to be a no-op stub (only a docstring) that ``drain()``
always reported ``status="applied"`` for, since nothing inside it ever
raised -- every enqueued ``"merge_memory"`` op silently dropped its content
instead of merging it. It is now wired to the real
``lifecycle.merge.merge_memories()`` and must raise (so ``drain()`` marks the
op ``"failed"`` instead of falsely ``"applied"``) whenever there is nothing
real to merge.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.lifecycle]


class TestApplyMergeCallsRealLogic:
    async def test_merges_via_merge_memories_when_two_or_more_decisions(self):
        from archivist.lifecycle.curator_queue import _apply_merge

        payload = {
            "agent_id": "agent-1",
            "namespace": "ns-1",
            "decisions": [
                {"decision": "merge", "existing_id": "mem-a"},
                {"decision": "merge", "existing_id": "mem-b"},
                {"decision": "keep", "existing_id": "mem-c"},
            ],
        }
        mock_merge = AsyncMock(return_value={"merged_id": "mem-merged"})
        with patch("archivist.lifecycle.merge.merge_memories", mock_merge):
            await _apply_merge(payload)

        mock_merge.assert_awaited_once_with(["mem-a", "mem-b"], "concat", "agent-1", "ns-1")

    async def test_raises_when_fewer_than_two_mergeable_ids(self):
        """H1: must raise (not silently succeed) so drain() marks the op failed."""
        from archivist.lifecycle.curator_queue import _apply_merge

        payload = {
            "agent_id": "agent-1",
            "namespace": "ns-1",
            "decisions": [{"decision": "merge", "existing_id": "mem-a"}],
        }
        with pytest.raises(ValueError, match="need >= 2"):
            await _apply_merge(payload)

    async def test_raises_when_no_decisions_present(self):
        from archivist.lifecycle.curator_queue import _apply_merge

        with pytest.raises(ValueError, match="need >= 2"):
            await _apply_merge({"agent_id": "agent-1"})

    async def test_raises_when_merge_memories_reports_error(self):
        from archivist.lifecycle.curator_queue import _apply_merge

        payload = {
            "decisions": [
                {"decision": "merge", "existing_id": "mem-a"},
                {"decision": "merge", "existing_id": "mem-b"},
            ],
        }
        mock_merge = AsyncMock(return_value={"error": "simulated merge failure"})
        with (
            patch("archivist.lifecycle.merge.merge_memories", mock_merge),
            pytest.raises(RuntimeError, match="simulated merge failure"),
        ):
            await _apply_merge(payload)


class TestApplyOpAwaitsMerge:
    async def test_apply_op_awaits_apply_merge(self):
        """_apply_op must `await` merge_memory (previously called it unawaited)."""
        from archivist.lifecycle.curator_queue import _apply_op

        mock_apply_merge = AsyncMock()
        with patch("archivist.lifecycle.curator_queue._apply_merge", mock_apply_merge):
            await _apply_op("merge_memory", {"decisions": []})

        mock_apply_merge.assert_awaited_once()
