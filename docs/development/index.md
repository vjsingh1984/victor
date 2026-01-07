# Development Guide

Complete guide for contributing to Victor.

## Overview

This section contains everything you need to contribute to Victor, from development environment setup to architecture, testing, and release processes.

**New to Victor?** Start with [Contributing Guide](contributing/workflow.md).

## Quick Start

```bash
# Clone repository
git clone https://github.com/vjsingh1984/victor.git
cd victor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
make test
pytest tests/unit -v

# Run formatting
make format
```

[Full Setup Guide →](setup/)

## Quick Links

| Topic | Documentation | Description |
|-------|--------------|-------------|
| **Setup** | [Environment](setup/environment.md) | Dev environment setup |
| **Setup** | [IDE Configuration](setup/ide.md) | VS Code, PyCharm setup |
| **Setup** | [Debugging](setup/debugging.md) | Debugging techniques |
| **Architecture** | [Overview](architecture/overview.md) | High-level architecture |
| **Architecture** | [Deep Dive](architecture/deep-dive.md) | Comprehensive internals |
| **Architecture** | [Design Patterns](architecture/design-patterns.md) | SOLID principles |
| **Testing** | [Strategy](testing/strategy.md) | Testing philosophy |
| **Testing** | [Writing Tests](testing/writing-tests.md) | How to write tests |
| **Contributing** | [Workflow](contributing/workflow.md) | Contribution workflow |
| **Contributing** | [Code Style](contributing/code-style.md) | Style guide |
| **Contributing** | [Pull Requests](contributing/pull-requests.md) | PR guidelines |
| **Extending** | [Providers](extending/providers.md) | Adding providers |
| **Extending** | [Tools](extending/tools.md) | Adding tools |
| **Extending** | [Verticals](extending/verticals.md) | Creating verticals |
| **Releasing** | [Versioning](releasing/versioning.md) | Version policy |
| **Releasing** | [Publishing](releasing/publishing.md) | Release process |

## Development Setup

### Environment Setup

**Requirements**:
- Python 3.10 or higher
- Git
- Virtual environment tool (venv, virtualenv, or conda)

**Installation**:
```bash
# Clone repository
git clone https://github.com/vjsingh1984/victor.git
cd victor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Verify installation
victor --version
```

[Full Environment Setup →](setup/environment.md)

### IDE Configuration

**VS Code** (recommended):
```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true
}
```

**PyCharm**:
- Settings → Project → Python Interpreter → Select .venv
- Settings → Tools → Black → Enable
- Settings → Tools → External Tools → Ruff

[Full IDE Setup →](setup/ide.md)

### Debugging

**VS Code Debug Configuration**:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Victor CLI",
      "type": "python",
      "request": "launch",
      "module": "victor",
      "args": ["chat", "your task here"],
      "console": "integratedTerminal"
    }
  ]
}
```

**Debugging Tests**:
```bash
# With VS Code debugger
pytest tests/unit/test_orchestrator.py --pdb

# With IPython debugger
pytest tests/unit/test_orchestrator.py --ipdb
```

[Full Debugging Guide →](setup/debugging.md)

## Architecture

Victor follows a **layered architecture** with clear separation of concerns:

### High-Level Architecture

```
┌─────────────────────────────────────────┐
│   CLIENT LAYER                          │
│   CLI, TUI, HTTP API, MCP Server        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│   ORCHESTRATION LAYER                   │
│   Agent Orchestrator, Conversation      │
│   Controller, Tool Pipeline             │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│   CORE LAYER                            │
│   Providers (21), Tools (55+),          │
│   Workflows, Verticals (5)              │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│   INFRASTRUCTURE LAYER                  │
│   DI Container, Event Bus, Config,      │
│   Storage, Bootstrap                    │
└─────────────────────────────────────────┘
```

### Key Architectural Patterns

**1. Dependency Injection**
- Service container for component wiring
- Protocol-based abstractions
- Lazy initialization

**2. Event-Driven**
- EventBus for cross-component communication
- Event sourcing for observability
- Async event handling

**3. Facade Pattern**
- AgentOrchestrator as single entry point
- Simplified interface for complex operations

**4. Protocol-Based Design**
- BaseProvider for LLM abstraction
- BaseTool for tool implementation
- VerticalBase for domain-specific logic

**5. Plugin System**
- Entry points for external verticals
- Dynamic tool registration
- Provider registry

### Module Structure

**Agent Layer** (`victor/agent/`):
- Orchestrator: Main coordinator
- ConversationController: Message history
- ToolPipeline: Tool execution
- ProviderManager: Provider switching
- StreamingController: Async streaming

**Framework** (`victor/framework/`):
- StateGraph DSL: Workflow definition
- Agent API: Simplified interface
- Task Management: Lifecycle and hooks

**Tools** (`victor/tools/`):
- BaseTool: Tool protocol
- Composition: LCEL-style chains
- Registry: Dynamic registration
- Categories: File, Git, Testing, etc.

**Providers** (`victor/providers/`):
- BaseProvider: LLM abstraction
- 21 provider implementations
- Tool calling support
- Streaming support

**Core** (`victor/core/`):
- DI Container: Service resolution
- Event Bus: Event system
- Configuration: Settings management
- Bootstrap: System initialization

[Full Architecture Documentation →](architecture/)

### Design Patterns

Victor follows **SOLID principles**:

- **S**ingle Responsibility: Each component has one reason to change
- **O**pen/Closed: Open for extension, closed for modification
- **L**iskov Substitution: Protocols define contracts
- **I**nterface Segregation: Focused, client-specific interfaces
- **D**ependency Inversion: Depend on abstractions, not concretions

**Key Patterns**:
- Facade: AgentOrchestrator
- Strategy: Provider switching
- Observer: EventBus
- Repository: Storage abstraction
- Factory: Component creation
- Decorator: ResilientProvider
- Command: Tool execution
- Chain of Responsibility: Tool pipeline

[Full Design Patterns Guide →](architecture/design-patterns.md)

### Data Flow

```
User Input
    ↓
CLI/TUI/HTTP
    ↓
Agent Orchestrator
    ↓
Conversation Controller (history)
    ↓
Provider Manager (select provider)
    ↓
Provider (generate response)
    ↓
Tool Pipeline (execute tools)
    ↓
Event Bus (emit events)
    ↓
Response to User
```

[Full Data Flow Documentation →](architecture/data-flow.md)

## Testing

### Testing Philosophy

**Test Pyramid**:
```
        E2E (5%)
       /         \
    Integration (15%)
   /               \
Unit Tests (80%)
```

**Principles**:
- **Fast**: Unit tests should run in milliseconds
- **Isolated**: No external dependencies
- **Deterministic**: Same inputs → same outputs
- **Readable**: Test names describe behavior

### Test Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── agent/              # Agent component tests
│   ├── core/               # Core system tests
│   ├── providers/          # Provider tests
│   └── tools/              # Tool tests
├── integration/            # External dependency tests
│   ├── agents/             # Multi-agent tests
│   ├── workflows/          # Workflow execution tests
│   └── mcp/                # MCP integration tests
└── conftest.py             # Shared fixtures
```

### Test Markers

```python
import pytest

@pytest.mark.unit
def test_provider_creation():
    """Unit test - no external dependencies"""
    pass

@pytest.mark.integration
def test_provider_with_api():
    """Integration test - requires API keys"""
    pass

@pytest.mark.slow
def test_large_workflow():
    """Slow test - long execution time"""
    pass

@pytest.mark.workflows
def test_workflow_execution():
    """Workflow-specific test"""
    pass
```

**Run specific markers**:
```bash
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests
```

### Writing Tests

**Unit Test Example**:
```python
import pytest
from victor.agent.orchestrator import Orchestrator

@pytest.mark.unit
class TestOrchestrator:
    def test_initialization(self):
        """Test orchestrator initializes correctly"""
        orchestrator = Orchestrator()
        assert orchestrator is not None
        assert orchestrator.conversation is not None

    @pytest.mark.asyncio
    async def test_process_message(self):
        """Test message processing"""
        orchestrator = Orchestrator()
        response = await orchestrator.process("Hello")
        assert response is not None
```

**Integration Test Example**:
```python
import pytest
from victor.providers.anthropic import AnthropicProvider

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"),
                    reason="No API key")
class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_chat_completion(self):
        """Test with actual API"""
        provider = AnthropicProvider()
        response = await provider.chat("Hello")
        assert response.content is not None
```

### Test Fixtures

**Shared fixtures** (`tests/conftest.py`):
```python
import pytest
from victor.agent.orchestrator import Orchestrator

@pytest.fixture
def orchestrator():
    """Provide orchestrator for tests"""
    return Orchestrator()

@pytest.fixture
def mock_provider():
    """Provide mock provider"""
    return MockProvider()
```

**Use in tests**:
```python
def test_with_fixture(orchestrator, mock_provider):
    """Test using fixtures"""
    result = orchestrator.run("test", provider=mock_provider)
    assert result is not None
```

### Running Tests

```bash
# Unit tests only (fast)
make test
pytest tests/unit -v

# All tests including integration
make test-all
pytest -v

# Single test file
pytest tests/unit/test_orchestrator.py -v

# With markers
pytest -m unit tests/unit/agent/ -v
pytest -m integration tests/integration/ -v
pytest -m "not slow" -v  # Skip slow tests

# Coverage
make test-cov
pytest --cov=victor --cov-report=html --cov-report=term-missing

# Run specific test
pytest tests/unit/test_orchestrator.py::TestOrchestrator::test_initialization -v
```

[Full Testing Guide →](testing/)

## Contributing

### Workflow

1. **Fork and clone** the repository
2. **Create a branch** for your feature
3. **Make changes** with tests
4. **Run tests** and linting
5. **Commit** with clear messages
6. **Push** to your fork
7. **Create pull request**

[Full Contribution Workflow →](contributing/workflow.md)

### Code Style

**Formatting**:
```bash
make format
black victor tests
ruff check --fix victor tests
```

**Linting**:
```bash
make lint
ruff check victor
black --check victor
```

**Type Checking**:
```bash
mypy victor  # Strict mode for some modules
```

**Style Guide**:
- **Line length**: 100 characters (Black)
- **Imports**: isort (via Black)
- **Type hints**: Comprehensive
- **Docstrings**: Google style
- **Naming**: snake_case for functions/variables, PascalCase for classes

[Full Code Style Guide →](contributing/code-style.md)

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(orchestrator): add provider switching during conversation

Implement context-preserving provider switching using
ConversationController. Fixes #123

fix(tools): correct file path resolution in ReadTool

Handle both absolute and relative paths correctly.
Fixes #456

docs(readme): update installation instructions

Add Docker installation option and update system requirements.
```

### Pull Requests

**PR Title**: Use conventional commit format

**PR Description**:
```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
- Unit tests added
- Integration tests pass
- Manual testing performed

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Documentation updated
- [ ] Type hints added
```

[Full PR Guidelines →](contributing/pull-requests.md)

## Extending Victor

### Adding Providers

**1. Create provider class**:
```python
# victor/providers/myprovider.py
from victor.providers.base import BaseProvider

class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "myprovider"

    async def chat(self, message: str, **kwargs) -> ChatResponse:
        # Implement chat
        pass
```

**2. Register provider**:
```python
# victor/providers/registry.py
from victor.providers.myprovider import MyProvider

registry.register("myprovider", MyProvider)
```

**3. Add tests**:
```python
# tests/unit/providers/test_myprovider.py
import pytest
from victor.providers.myprovider import MyProvider

@pytest.mark.unit
class TestMyProvider:
    def test_initialization(self):
        provider = MyProvider()
        assert provider.name == "myprovider"
```

[Full Provider Guide →](extending/providers.md)

### Adding Tools

**1. Create tool class**:
```python
# victor/tools/mytool.py
from victor.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"

    parameters = {
        "input": {
            "type": "string",
            "description": "Input to process"
        }
    }

    async def execute(self, **kwargs) -> ToolResult:
        result = process(kwargs["input"])
        return ToolResult(output=result)
```

**2. Register tool**:
```python
# victor/tools/registry.py
from victor.tools.mytool import MyTool

registry.register(MyTool)
```

**3. Add tests**:
```python
# tests/unit/tools/test_mytool.py
import pytest
from victor.tools.mytool import MyTool

@pytest.mark.unit
class TestMyTool:
    @pytest.mark.asyncio
    async def test_execution(self):
        tool = MyTool()
        result = await tool.execute(input="test")
        assert result.success
```

[Full Tool Guide →](extending/tools.md)

### Creating Verticals

**1. Use CLI**:
```bash
victor vertical create security --description "Security analysis"
```

**2. Manual creation**:
```
victor/security/
├── __init__.py           # Vertical class
├── tools/                # Vertical-specific tools
└── workflows/            # YAML workflow definitions
```

**3. Implement vertical**:
```python
# victor/security/__init__.py
from victor.core.verticals import VerticalBase

class SecurityVertical(VerticalBase):
    name = "security"
    description = "Security analysis and auditing"

    def get_tools(self) -> List[BaseTool]:
        return [
            SecurityScanTool(),
            VulnerabilityTool(),
            ComplianceTool()
        ]
```

**4. Register in pyproject.toml**:
```toml
[project.entry-points."victor.verticals"]
security = "victor.security:SecurityVertical"
```

[Full Vertical Guide →](extending/verticals.md)

### Custom Workflows

**1. Create workflow YAML**:
```yaml
# workflows/my_workflow.yaml
workflow: MyWorkflow
description: My custom workflow

nodes:
  - id: step1
    type: agent
    role: You are a helpful assistant.

  - id: step2
    type: compute
    tools: [my_custom_tool]

edges:
  - source: step1
    target: step2
```

**2. Test workflow**:
```bash
victor workflow run my_workflow
```

[Full Workflow Guide →](extending/workflows.md)

## Releasing

### Versioning

Victor follows [Semantic Versioning 2.0.0](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality
- **PATCH**: Backwards-compatible bug fixes

Example: `1.2.3`
- MAJOR: 1
- MINOR: 2
- PATCH: 3

[Full Versioning Guide →](releasing/versioning.md)

### Release Process

**1. Update version**:
```bash
make release VERSION=1.2.3
```

**2. Update CHANGELOG**:
```markdown
## [1.2.3] - 2025-01-07

### Added
- New feature 1
- New feature 2

### Changed
- Updated feature 3

### Fixed
- Bug fix 1
```

**3. Create release commit**:
```bash
git add CHANGELOG.md victor/__init__.py
git commit -m "chore(release): bump version to 1.2.3"
git tag v1.2.3
```

**4. Build and publish**:
```bash
make build
make release
```

[Full Publishing Guide →](releasing/publishing.md)

## Additional Resources

- **Architecture**: [Architecture Deep Dive →](architecture/deep-dive.md)
- **Testing**: [Testing Strategy →](testing/strategy.md)
- **Contributing**: [Contribution Workflow →](contributing/workflow.md)
- **User Guide**: [User Guide →](../user-guide/)
- **Reference**: [Provider Reference →](../reference/providers/)
- **Operations**: [Deployment →](../operations/deployment/)

---

**Next**: [Environment Setup →](setup/environment.md)
