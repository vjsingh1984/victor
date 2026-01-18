# Development Guide

Contributing to Victor: setup, patterns, and processes.

## Quick Setup

```bash
git clone https://github.com/vijayksingh/victor.git
cd victor
pip install -e ".[dev]"
pytest tests/unit -v
```

## Code Quality

```bash
# Before commits
make lint           # ruff + black --check + mypy
make format         # black + ruff --fix

# Or individually
black victor tests
ruff check --fix victor tests
mypy victor
```

## Testing

```bash
pytest tests/unit -v                    # Unit tests
pytest tests/unit/test_X.py -v          # Single file
pytest -m "not slow" -v                 # Skip slow tests
pytest -m integration                    # Integration tests
pytest --cov=victor --cov-report=html   # Coverage

# Makefile shortcuts
make test           # Unit tests
make test-all       # All tests
make test-cov       # With coverage
```

### Test Markers

- `@pytest.mark.unit` - Fast, isolated tests
- `@pytest.mark.integration` - External dependencies
- `@pytest.mark.slow` - Long-running tests

## Architecture Overview

```
victor/
├── agent/          # Orchestration, tool pipeline
├── providers/      # LLM providers (BaseProvider)
├── tools/          # Tools (BaseTool, CostTier)
├── framework/      # StateGraph DSL, Agent/Task API
├── workflows/      # YAML execution, scheduling
├── coding/         # AST, LSP, code review
├── devops/         # Docker, Terraform, CI
├── rag/            # Embeddings, retrieval
├── dataanalysis/   # Pandas, statistics
├── research/       # Web search, synthesis
└── observability/  # EventBus, metrics
```

## Adding Components

### New Provider

1. Create `victor/providers/your_provider.py`
2. Inherit `BaseProvider`
3. Implement: `chat()`, `stream_chat()`, `supports_tools()`, `name`
4. Register in `ProviderRegistry`
5. Add tests in `tests/unit/providers/`

### New Tool

1. Create `victor/tools/your_tool.py`
2. Inherit `BaseTool`
3. Define: `name`, `description`, `parameters`, `cost_tier`, `execute()`
4. Register in tool registry
5. Add tests, run `python scripts/generate_tool_catalog.py`

### New Vertical

```bash
victor vertical create security --description "Security analysis"
```

Or manually create in `victor/{vertical}/` with:
- `__init__.py` - Vertical class extending `VerticalBase`
- `tools/` - Vertical-specific tools
- `workflows/` - YAML workflow definitions

## Key Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| Provider Protocol | `providers/base.py` | LLM abstraction |
| Tool Protocol | `tools/base.py` | Tool abstraction |
| StateGraph DSL | `framework/graph.py` | Workflow definition |
| Dependency Injection | `agent/service_provider.py` | Component wiring |
| Event Bus | `observability/event_bus.py` | Observability |

## Releasing

See [releasing/publishing.md](releasing/publishing.md) for version bumping and publishing.

## More Details

- [Testing Strategy](testing/strategy.md) - Fixtures and mocking patterns
- [Architecture Deep Dive](architecture/deep-dive.md) - System internals
- [Plugin Guide](extending/plugins.md) - External extensions
- [Vertical Development](extending/verticals.md) - Custom verticals
