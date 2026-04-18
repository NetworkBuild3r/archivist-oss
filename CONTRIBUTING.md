# Contributing to Archivist

Thank you for improving Archivist. This project targets production agent-memory workloads: keep changes focused, tested, and documented when behaviour or configuration shifts.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-test.txt
```

Optional extras (not required for the default server image or CI unit tests):

| File | When to install |
|------|-----------------|
| `requirements-rerank.txt` | Cross-encoder reranking (`RERANKER_ENABLED=true`) |
| `requirements-benchmark.txt` | BEIR thin and academic harnesses |

## Checks before you open a PR

Run the same checks CI enforces (adjust paths if you use a different venv):

```bash
ruff check . --fix && ruff format .
python -m mypy src/archivist/ --config-file pyproject.toml
python -m pytest tests/ -q --tb=no
```

**Storage / outbox changes** — also run the focused QA package:

```bash
python -m pytest tests/qa/ -q --tb=no
```

See [`docs/QA.md`](docs/QA.md) and [`tests/qa/README.md`](tests/qa/README.md).

## Code layout

Production code lives under **`src/archivist/`** (`app/`, `storage/`, `lifecycle/`, `retrieval/`, `write/`, `features/`, `core/`, `utils/`). Legacy top-level `src/*.py` shims may still re-export symbols; new code should use the package paths.

When you touch chunking, retrieval thresholds, or the transactional write path, update or add **focused tests** in `tests/` (and `tests/qa/` if the outbox or `MemoryTransaction` contract changes).

## Pull requests

- Run **`pytest`** (and **`pytest tests/qa/`** if you changed storage/outbox code) before requesting review.
- Prefer **small, reviewable diffs** with tests for behaviour changes.
- Do **not** commit secrets, API keys, or internal hostnames; follow [`.env.example`](.env.example) patterns only.
- Document user-visible changes in [`CHANGELOG.md`](CHANGELOG.md) when appropriate.

## License

By contributing, you agree your contributions are licensed under the same terms as the project (**Apache-2.0**).
