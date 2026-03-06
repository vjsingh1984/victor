# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in `victor/` (agents, workflows, providers, toolchains) and is what ships to PyPI. Supporting packages include `victor_cookbook/` and `examples/` for reusable agents, plus `docs/` (MkDocs site) and release tooling in `scripts/` and `docker/`. Tests reside in `tests/`—`unit`, `integration`, `benchmark`, `performance`, and scenario folders in `tests/examples/`. Frontend code lives in `web/` and `ui/`; IDE tooling stays in `vscode-victor/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`: bootstrap an env.
- `make install` or `make install-dev`: install runtime vs. full dev/docs/build extras with pre-commit.
- `make test`, `make test-all`, `make test-quick`: run unit suites, every pytest target, or skip `slow` markers.
- `make lint`, `make format`, `make test-cov`: run Ruff/Black/MyPy, apply formatting, or write coverage to `htmlcov/`.
- `make docs`, `make serve`, `make tui`, `make build`, `make docker`: docs site, HTTP API, TUI, Python packages, or container images.

## Coding Style & Naming Conventions
Python uses 4-space indentation, ~100-character lines, `snake_case` modules/functions, and `PascalCase` classes. Public APIs need type hints and Google-style docstrings, and async/await is mandatory for I/O-heavy helpers. Run `ruff check victor tests`, `black victor tests`, and `mypy victor --ignore-missing-imports` (or `make lint`) before committing, and prefer `respx` for HTTP mocking.

## Testing Guidelines
Pytest powers all suites. Put granular cases in `tests/unit/<feature>/test_<subject>.py`, promote workflow or provider scenarios to `tests/integration/`, and keep opt-in benchmarks in `tests/benchmark/` or `tests/performance/`. Use `@pytest.mark.slow` for long-running flows so `make test-quick` remains fast, and run `make test-cov` (or `pytest --cov=victor --cov-report=html`) before major changes to keep coverage near the 85% bar.

## Commit & Pull Request Guidelines
Follow Conventional Commits (`feat: add workflow guard`, `fix: handle timeout`) and keep each commit scoped; add `Fixes #123` when closing issues. Run `make test` and `make lint`, update docs/CHANGELOG entries, and include logs or screenshots before requesting review. The PR template (description, change list, tests, related issues) plus CI gates—pytest, Black, Ruff, MyPy, Trivy, Docker build, and package build—are mandatory, and at least one maintainer approval is required.

## Security & Configuration Tips
Secrets such as `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or Ollama model paths must stay in environment variables or `~/.victor`; never commit credentials. Follow the hardening guidance in `SECURITY.md` (air-gapped mode, disclosure process, supported versions) whenever handling sensitive code. Discuss networking changes with maintainers before editing `docker/` or `docker-compose*.yml`, since those files define the exposed surface.
