# Contributing to Archivist

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest
```

## Tests

From the repository root:

```bash
python -m pytest tests/ -v
```

Tests are designed to run **without** Qdrant, an LLM, or embedding APIs. Optional reranking tests use a nonexistent model name to exercise the graceful fallback path.

## Code layout

- `src/chunking.py` — Pure chunking (shared by indexer and unit tests).
- `src/retrieval_filters.py` — Pure retrieval filters (e.g. score threshold).
- `src/memory_fusion.py` — Dedupe merged hits from multi-agent search.
- `src/rlm_retriever.py` — Vector search pipeline, enrichment, LLM refinement.

When changing chunking or threshold behavior, update the shared modules so tests stay aligned with production code.

## Pull requests

- Run `pytest` before opening a PR.
- For behavioral changes, add or adjust a focused unit test.
- Do not commit secrets, API keys, or internal hostnames; use `.env.example` patterns only.

## License

By contributing, you agree that your contributions are licensed under the same terms as the project (Apache-2.0).
