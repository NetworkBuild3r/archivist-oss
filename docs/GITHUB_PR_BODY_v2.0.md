## Archivist 2.0.0

### Release

- **Tag:** `v2.0.0` (annotated, pushed with this PR branch)
- **Release notes:** [`docs/RELEASE_NOTES_v2.0.md`](docs/RELEASE_NOTES_v2.0.md)
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md) — section **[2.0.0] - 2026-04-17**

### Benchmarks (Phase 5)

Pipeline evaluation documented in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md):

- Variants: **`clean_reranker`** vs **`vector_plus_synth`**
- Scale: **small**, **108 queries** per variant, **2 variants**, session **~2213s** total
- Reproduce: full JSON at `.benchmarks/phase5_semantic_chunking.json` when running the harness (gitignored locally)

### Summary

- Package layout under `src/archivist/` with compatibility shims
- Mypy configuration for `archivist.*` imports
- Version **2.0.0** in health endpoint, startup log, `archivist.__version__`, backup manifest

---

**Suggested merge:** `main` ← this branch, then confirm the `v2.0.0` tag matches the merge commit (or re-tag if policy requires tags only on `main`).
