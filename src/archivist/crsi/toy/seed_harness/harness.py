"""Editable toy harness wiring SeedHarness with interlock.py and prompts/system.md."""

from __future__ import annotations

from pathlib import Path

from archivist.crsi.harness.context import SessionContext
from archivist.crsi.harness.runner import SeedHarness
from archivist.crsi.llm.base import LLMClient


def build_toy_harness(llm_client: LLMClient, interlock_factory: object) -> SeedHarness:
    """Assemble ``SeedHarness`` from the editable prompt template and interlock factory."""
    prompt_path = Path(__file__).resolve().parent / "prompts" / "system.md"
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    context = SessionContext(system_prompt=system_prompt)
    return SeedHarness(
        llm_client=llm_client,
        interlock_factory=interlock_factory,  # type: ignore[arg-type]
        context=context,
    )
