# Archivist 2.0.0 — Release Notes

**Release date:** 2026-04-17
**Git tag:** `v2.0.0`
**Branch:** `feature/v1.12-cascade-tech-debt` (merge target: `main`)

## Summary

Archivist **2.0** is a major milestone: the service codebase is now organized as a proper Python package under `src/archivist/`, with clear boundaries between core configuration, storage, lifecycle, retrieval, write-time enrichment, optional features, and the FastAPI/MCP application layer. Compatibility shims preserve legacy import paths for existing deployments and tests.

This release also publishes **fresh pipeline benchmark results** (Phase 5, semantic chunking) comparing the **`clean_reranker`** and **`vector_plus_synth`** variants on the **small** memory scale. See [`BENCHMARKS.md`](BENCHMARKS.md) for tables and reproduction commands.

## Highlights

### Packaging and maintainability

- **Modular layout** — Modules live under `src/archivist/` in subpackages: `core`, `storage`, `lifecycle`, `retrieval`, `write`, `features`, `utils`, and `app` (including `handlers/`).
- **Compatibility shims** — Top-level modules under `src/` continue to resolve to the new package (runtime `sys.modules` aliasing), so external scripts and tests that import legacy module names keep working.
- **Static analysis** — `mypy` is configured in `pyproject.toml` with `mypy_path = "src"` and `explicit_package_bases = true` so `archivist.*` imports type-check correctly.

### Operations and quality

- **Deletion and cascade** — Retry logic is centralized; async entrypoints offload blocking SQLite/Qdrant work appropriately; curator queue behavior for partial deletes remains explicit and auditable.
- **Version** — API health and startup logs report **2.0.0**; backup manifests record the same `archivist_version` where applicable.

### Benchmarks

- **Phase 5 pipeline run** — Documented in [`BENCHMARKS.md`](BENCHMARKS.md): `clean_reranker` vs `vector_plus_synth`, small corpus, 108 queries per variant. Raw JSON is produced at `.benchmarks/phase5_semantic_chunking.json` when you run the harness locally (that directory remains gitignored).

## Upgrade notes

- No change to the external HTTP/MCP API contract for this release label.
- If you import Archivist Python modules by **legacy bare names** (e.g. `import graph`), shims remain. New code should prefer **`from archivist.storage.graph import …`** (or the appropriate subpackage).
- After pulling, run your usual test suite: `pytest` from the repo root with `pythonpath` as configured in `pyproject.toml`.

## References

- [`CHANGELOG.md`](../CHANGELOG.md) — Full change list.
- [`BENCHMARKS.md`](BENCHMARKS.md) — Pipeline and micro-benchmark results.
- [`README.md`](../README.md) — Quick start and configuration.

---

*Archivist — Memory-as-a-Service for AI agent fleets.*
