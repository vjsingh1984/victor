# Developer Guide

Contribute to Victor AI development.

## Quick Start

```bash
# Clone repository
git clone https://github.com/vjsingh1984/victor
cd victor

# Install for development
pip install -e ".[dev]"

# Run tests
make test

# Run linting
make lint

# Format code
make format
```

## Development Workflow

### 1. Fork and Clone

```bash
# Fork on GitHub
git clone https://github.com/YOUR_USERNAME/victor.git
cd victor
git remote add upstream https://github.com/vjsingh1984/victor.git
```

### 2. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

```bash
# Edit code
vim victor/your_file.py

# Format code
make format

# Run linting
make lint

# Run tests
make test

# Run specific test
pytest tests/unit/path/test_file.py::test_name -v
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add your feature"
```

**Commit message format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Tests
- `chore:` Maintenance

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
# Create PR on GitHub
```

## Project Structure

```
victor/
├── agent/              # Core orchestration
│   ├── coordinators/   # 20 coordinators (app + framework)
│   └── orchestrator.py # Main facade
├── providers/          # 21 LLM providers
├── tools/              # 55+ tools
├── core/               # Foundation services
│   ├── container.py   # Dependency injection
│   └── verticals/      # Vertical base classes
├── framework/          # Workflow engine
├── protocols/          # 98 protocols
└── config/             # YAML configurations
```

## Adding Components

### New Provider

```python
# victor/providers/my_provider.py
from victor.providers.base import BaseProvider

class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    async def chat(self, messages, **kwargs):
        # Implementation
        pass
```

### New Tool

```python
# victor/tools/my_tool.py
from victor.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string"}
        }
    }

    async def execute(self, **kwargs):
        # Implementation
        pass
```

### New Coordinator

```python
# victor/agent/coordinators/my_coordinator.py
from victor.agent.coordinators.base import CoordinatorProtocol

class MyCoordinator(CoordinatorProtocol):
    async def coordinate(self, context):
        # Implementation
        pass
```

## Testing

### Key Fixtures

```python
# tests/conftest.py
@pytest.fixture
def reset_singletons():
    """Reset all singletons between tests."""
    pass

@pytest.fixture
def mock_docker_client():
    """Mock Docker for tests."""
    pass
```

### Test Markers

```python
@pytest.mark.unit        # Unit tests (fast, isolated)
@pytest.mark.integration # Integration tests (external services)
@pytest.mark.slow        # Slow tests (deselect with -m "not slow")
@pytest.mark.workflows   # Workflow tests
```

### Running Tests

```bash
# Unit tests only
make test

# All tests
make test-all

# Specific test
pytest tests/unit/path/test_file.py::test_name -v

# With coverage
pytest --cov=victor --cov-report=html
```

## Code Style

### Formatting

```bash
make format
```

Uses:
- **black**: Line length 100
- **ruff**: Fast linter

### Linting

```bash
make lint
```

Checks:
- **ruff**: Fast Python linter
- **black --check**: Formatting
- **mypy**: Type checking

## Type Safety

Victor uses **98 protocols** for type safety:

```python
from victor.agent.protocols import ToolExecutorProtocol

def my_function(executor: ToolExecutorProtocol):
    # Type-safe usage
    result = await executor.execute_tool(tool, args)
```

## Architecture Patterns

### Protocol-Based Design

Components depend on abstractions (protocols), not concretions:

```python
# Good: Protocol-based
class MyComponent:
    def __init__(self, executor: ToolExecutorProtocol):
        self._executor = executor
```

### Dependency Injection

Use ServiceContainer for dependency management:

```python
from victor.core.container import ServiceContainer

container = ServiceContainer()
executor = container.get(ToolExecutorProtocol)
```

### Two-Layer Coordinators

- **Application Layer** (`victor/agent/coordinators/`): Victor-specific
- **Framework Layer** (`victor/framework/coordinators/`): Domain-agnostic

## Debugging

### Enable Debug Logging

```bash
export VICTOR_LOG_LEVEL=DEBUG
victor chat
```

### pdb Debugging

```python
import pdb; pdb.set_trace()
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Import error | Run `pip install -e ".[dev]"` |
| Test failures | Run `make test` locally first |
| Linting errors | Run `make format` to fix |

## Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Code Review Checklist

- [ ] Tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Formatted (`make format`)
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] No breaking changes (or documented)

## Resources

- [Architecture](../architecture/) - System design
- [Best Practices](../architecture/BEST_PRACTICES.md) - Usage patterns
- [Protocols](../api-reference/protocols.md) - Protocol reference
- [Testing](../development/testing.md) - Test guide

## Getting Help

- [GitHub Issues](https://github.com/vjsingh1984/victor/issues)
- [Discussions](https://github.com/vjsingh1984/victor/discussions)
- [Discord](https://discord.gg/...)

---

<div align="center">

**[← Back to Documentation](../README.md)**

**Developer Guide**

*Contributing to Victor*

</div>
