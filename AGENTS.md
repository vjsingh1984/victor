# Repository Guidelines

## Project Structure & Ownership
Core runtime code lives in `victor/`. The main boundaries worth knowing are:
- `victor/framework/`: public framework APIs, agent abstractions, workflows, providers, and other surfaces that usually need docs/tests when changed.
- `victor/agent/`: lower-level orchestration, protocols, planning, runtime, tool-calling, and team internals.
- `victor/tools/`, `victor/providers/`, `victor/verticals/`, `victor/workflows/`: first-class extension surfaces; prefer adding domain behavior here instead of bloating core abstractions.
- `victor/integrations/api/`, `victor/integrations/mcp/`, `victor/ui/`, and `victor/commands/`: CLI, TUI, HTTP API, and integration entry points.
- `victor-sdk/`: separately packaged SDK definitions; it is version-locked to the root package and should stay in sync.
- `rust/`: optional native extensions built with `maturin`; Python must continue to work when native code is absent.
- `docs/`: MkDocs source. `site/` is generated output and should not be hand-edited unless the task is specifically about generated artifacts.
- `ui/`, `web/ui/`, and `vscode-victor/`: separate frontend projects with their own `package.json` and lockfiles.

## Generated Files & Repo Hygiene
Prefer editing source, not generated output:
- Do not hand-edit `site/`, `htmlcov/`, `dist/`, `build/`, `ui/dist/`, `.pytest_cache/`, `.mypy_cache/`, or `.ruff_cache/` unless the task explicitly targets generated artifacts.
- Treat `.victor/`, benchmark outputs, and other local caches as runtime state, not source of truth.
- If you touch docs, workflows, README links, or release metadata, run `python scripts/ci/repo_hygiene_check.py` or `make check-repo-hygiene`.

The hygiene check currently enforces a few non-obvious rules:
- Keep first-party repo URLs pointed at the canonical GitHub repo.
- Use the canonical lowercase `roadmap.md` link target instead of `ROADMAP.md`.
- Do not reintroduce removed legacy paths such as `victor/agent/protocols.py`; `victor/agent/protocols/` is the canonical source.
- Keep the `Makefile` lint target running `mypy victor` without suppressing failures.

## Build, Test, and Development Commands
Python setup:
- `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`: minimal contributor setup.
- `make install-dev`: install dev/docs/build extras plus pre-commit helpers.
- `pre-commit install`: enable the repository hook set if it was not installed automatically.

Core validation:
- `make test`: unit tests with `pytest-xdist`.
- `make test-all`: full pytest suite, including integration tests.
- `make test-quick`: unit tests excluding `slow` markers.
- `make test-cov`: unit coverage report for `victor/`.
- `make test-definition-boundaries`: import-boundary guardrails for definition/protocol surfaces.
- `make lint`: Ruff, Black, MyPy, and repo hygiene checks.
- `make format`: Black plus Ruff autofixes.

Subprojects:
- `make docs` or `make docs-serve`: build or serve MkDocs from `docs/`.
- `npm --prefix ui run build` / `npm --prefix ui run lint`: validate the standalone Vite UI.
- `npm --prefix web/ui run build` / `npm --prefix web/ui run lint`: validate the web UI.
- `npm --prefix vscode-victor run ...`: use npm in the VS Code extension when touching that subtree.
- `cd rust && maturin develop` or `cd rust && cargo test`: build/test native extensions.

## Coding Style & Naming Conventions
Python targets 3.10+ and uses 4-space indentation, ~100-character lines, `snake_case` functions/modules, and `PascalCase` classes. Public or widely reused APIs should have type hints and concise Google-style docstrings. Async code should stay async end-to-end for I/O-heavy paths; avoid introducing sync wrappers around provider, tool, or network flows unless there is already an established sync boundary.

Formatting/linting expectations are defined by repo tooling:
- Black is pinned in `pyproject.toml` and pre-commit.
- Ruff is the primary lint/autofix tool.
- MyPy is enforced on `victor/`.
- `respx` is the preferred HTTP mocking tool in tests.

For frontend work, preserve the existing package manager (`npm`) and keep `package-lock.json` files in sync with dependency changes.

## Testing Guidelines
Pytest powers the Python suites:
- `tests/unit/`: fast, granular behavior and regression coverage.
- `tests/integration/`: provider, API, workflow, and subsystem integration coverage.
- `tests/benchmark/` and `tests/performance/`: opt-in performance/benchmarking coverage.
- `tests/examples/`: example and scenario validation.

Place new tests near the behavior you changed. Use `@pytest.mark.slow` for long-running flows so `make test-quick` remains useful. If you touch public framework APIs, provider contracts, tool registration, workflow compilation, or definition/protocol boundaries, add or update tests in the corresponding unit/integration area.

When changing native-extension behavior, verify both the Rust side and the Python fallback path when practical.

## Docs, Architecture, and Change Scope
Use the smallest layer that fits the change:
- Put framework-wide abstractions in `victor/framework/` only when they are truly reusable.
- Keep domain-specific behavior in verticals, tools, providers, or integrations instead of expanding core APIs.
- Update docs when user-facing commands, config, providers, workflows, or public APIs change.

Large framework-level changes usually need more than code:
- Changes to `victor/framework/` public APIs, protocol definitions, workflow DSL structure, or major architecture patterns likely need a FEP and docs updates.
- For version changes, update `VERSION` and use `python scripts/sync_version.py` so root and `victor-sdk/` stay aligned.

## Commit & Pull Request Guidelines
Follow Conventional Commits such as `feat: add workflow retry guard` or `fix: handle empty tool result`. Keep commits scoped. Reference issues with `Fixes #123` when appropriate.

Before opening a PR, run the relevant subset of:
- `make lint`
- `make test` or `make test-all`
- `make check-repo-hygiene`
- frontend or Rust validation commands for touched subprojects

PRs should include docs or changelog updates when behavior changes, and screenshots/logs when UI or operational behavior changed.

## Security & Configuration Tips
Never commit secrets, tokens, or local credentials. Use environment variables or `~/.victor` for provider configuration. If you touch security-sensitive logic, read `SECURITY.md` first; current security tooling includes secret scanning, Trivy, `pip-audit`, and Bandit, but repo-local scans may still be advisory in some areas.

Networking and container changes have outsized impact. Discuss or review carefully before changing `docker/`, `docker-compose*.yml`, API exposure defaults, sandboxing, or tool/network permissions.
