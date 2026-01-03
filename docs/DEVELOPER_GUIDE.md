# Victor Developer Guide

A focused guide for contributing changes or extensions.

## Setup
```bash
git clone https://github.com/vijayksingh/victor.git
cd victor
pip install -e ".[dev]"
```

## Run Tests
```bash
pytest
ruff check victor tests
mypy victor
```

## Add a Provider (LLM)
1. Create a new provider in `victor/providers/`.
2. Implement the base interface.
3. Register it in the provider registry.
4. Add tests under `tests/unit/`.
5. Update `docs/reference/PROVIDERS.md`.

## Add a Tool
1. Create a tool in `victor/tools/` (use the `@tool` decorator).
2. Add tests under `tests/unit/tools/`.
3. Regenerate the tool catalog if needed: `python scripts/generate_tool_catalog.py`.

## Where to Look
- Framework entrypoints: `victor/framework/`
- Core agent: `victor/agent/`
- Tools: `victor/tools/`
- Providers: `victor/providers/`
- Verticals: `victor/core/verticals/`

## Docs
- Contributing: `CONTRIBUTING.md`
- Tool catalog: `docs/TOOL_CATALOG.md`
- Workflow DSL: `docs/guides/WORKFLOW_DSL.md`
