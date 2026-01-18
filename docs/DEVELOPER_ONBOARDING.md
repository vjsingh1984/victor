# Victor AI: Developer Onboarding Guide

**Version**: 0.5.1
**Last Updated**: January 18, 2026
**Audience**: New Developers, Contributors

---

## Table of Contents

1. [Welcome to Victor](#welcome-to-victor)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Development Setup](#development-setup)
5. [Key Concepts](#key-concepts)
6. [Common Workflows](#common-workflows)
7. [Testing Guide](#testing-guide)
8. [Documentation Resources](#documentation-resources)
9. [Contributing](#contributing)
10. [Getting Help](#getting-help)

---

## Welcome to Victor

### What is Victor?

Victor is an **open-source AI coding assistant** supporting:
- **21 LLM providers** (Anthropic, OpenAI, Google, Azure, local models)
- **55 specialized tools** across 5 domain verticals
- **Air-gapped mode** for offline, secure environments
- **YAML workflow DSL** for multi-step automation
- **Multi-agent teams** for complex tasks

### Architecture Highlights

Victor 0.5.1 features a modern, scalable architecture:
- **98 protocols** for loose coupling
- **55+ services** in dependency injection container
- **20 coordinators** with 92.13% test coverage
- **5 event backends** for scalability
- **Vertical template system** for rapid development

### Your First Day Goals

By the end of your first day, you should:
- [ ] Have Victor running locally
- [ ] Understand the high-level architecture
- [ ] Run the test suite successfully
- [ ] Make a small change and verify it works
- [ ] Know where to find help

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Git
- Optional: Docker for containerized development

### Installation

```bash
# Clone repository
git clone https://github.com/vjsingh1984/victor.git
cd victor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Run initial setup
victor init
```

### Your First Run

```bash
# Run with local model (requires Ollama)
ollama pull qwen2.5-coder:7b
victor chat "Hello, Victor!"

# Or run with cloud model
export ANTHROPIC_API_KEY=your_key_here
victor chat --provider anthropic "Hello, Victor!"
```

### Verify Installation

```bash
# Run smoke tests
pytest tests/smoke/ -v

# Check version
victor --version

# List available tools
victor tools --list
```

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────┐
│           CLIENTS                        │
│  CLI │ TUI │ VS Code │ API Server       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      ServiceContainer (DI)               │
│  55+ services (singleton, scoped)        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      AgentOrchestrator (Facade)         │
│  Delegates to coordinators               │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
┌─────────────┐ ┌──────────────┐
│ Coordinators│ │  Services    │
│  (20 total) │ │              │
└─────────────┘ └──────────────┘
```

### Key Components

#### 1. ServiceContainer
**What**: Dependency injection container
**Where**: `victor/core/container.py`
**Why**: Centralized service management

```python
from victor.core.container import ServiceContainer

container = ServiceContainer()
tool_registry = container.get(ToolRegistryProtocol)
```

#### 2. Coordinators
**What**: Specialized orchestration components
**Where**: `victor/agent/coordinators/`, `victor/framework/coordinators/`
**Why**: Single responsibility for complex operations

**Application Layer** (Victor-specific):
- ChatCoordinator - LLM chat operations
- ToolCoordinator - Tool selection and execution
- StateCoordinator - Conversation state management
- ProviderCoordinator - Provider switching

**Framework Layer** (Reusable):
- YAMLWorkflowCoordinator - YAML workflows
- GraphExecutionCoordinator - StateGraph execution
- HITLCoordinator - Human-in-the-loop

#### 3. Protocols
**What**: Interface definitions
**Where**: `victor/protocols/`, `victor/agent/protocols.py`
**Why**: Loose coupling and testability

```python
from typing import Protocol
from typing_extensions import runtime_checkable

@runtime_checkable
class ToolCoordinatorProtocol(Protocol):
    async def select_tools(self, query: str) -> List[BaseTool]:
        ...
```

#### 4. Event Bus
**What**: Pub/sub messaging system
**Where**: `victor/core/events/`
**Why**: Scalable, asynchronous communication

```python
from victor.core.events import create_event_backend, MessagingEvent

backend = create_event_backend(BackendConfig.for_observability())
await backend.publish(
    MessagingEvent(topic="tool.complete", data={"tool": "read_file"})
)
```

#### 5. Verticals
**What**: Domain-specific assistants
**Where**: `victor/coding/`, `victor/research/`, etc.
**Why**: Specialized tools and prompts

**5 Verticals**:
- Coding - Code analysis, refactoring, testing
- DevOps - Infrastructure, CI/CD, deployment
- RAG - Document ingestion, vector search
- DataAnalysis - Pandas, visualization, statistics
- Research - Web search, fact-checking, synthesis

---

## Development Setup

### IDE Configuration

#### VS Code (Recommended)

Install extensions:
- Python
- Pylance
- pytest
- GitLens

Configure settings (`.vscode/settings.json`):
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

#### PyCharm

1. Open project
2. Settings → Project → Python Interpreter
3. Add `.venv` as interpreter
4. Settings → Tools → Python Integrated Tools
5. Set pytest as default runner

### Git Configuration

```bash
# Install pre-commit hooks
pre-commit install

# Configure user (if not already)
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Required: At least one provider API key
```

---

## Key Concepts

### 1. Protocol-Based Design

**Concept**: Define interfaces before implementations

**Why**:
- Loose coupling
- Easy testing (mock protocols)
- Multiple implementations

**Example**:
```python
# Define protocol
@runtime_checkable
class MyServiceProtocol(Protocol):
    async def process(self, data: str) -> dict:
        ...

# Implement protocol
class MyService:
    async def process(self, data: str) -> dict:
        return {"result": data.upper()}

# Use protocol (can mock in tests)
class MyComponent:
    def __init__(self, service: MyServiceProtocol):
        self._service = service
```

### 2. Dependency Injection

**Concept**: Inject dependencies instead of creating them

**Why**:
- Clear dependencies
- Easy testing
- Shared instances

**Example**:
```python
# Register service
container.register(
    MyServiceProtocol,
    lambda c: MyService(),
    ServiceLifetime.SINGLETON,
)

# Inject dependency
class MyComponent:
    def __init__(self, service: MyServiceProtocol):
        self._service = service

# Resolve from container
service = container.get(MyServiceProtocol)
component = MyComponent(service)
```

### 3. Coordinator Pattern

**Concept**: Focused coordinators for complex operations

**Why**:
- Single responsibility
- Testable in isolation
- Reusable

**Example**:
```python
class ToolCoordinator:
    """ONLY coordinates tool selection and execution."""

    def __init__(
        self,
        selector: IToolSelector,
        budget_manager: IBudgetManager,
        executor: ToolExecutorProtocol,
    ):
        self._selector = selector
        self._budget_manager = budget_manager
        self._executor = executor

    async def select_and_execute(self, query: str):
        # Delegate to specialists
        tools = await self._selector.select_tools(query)
        if not self._budget_manager.can_execute(len(tools)):
            raise BudgetExceededError()
        return await self._executor.execute_tools(tools)
```

### 4. Event-Driven Architecture

**Concept**: Publish-subscribe for loose coupling

**Why**:
- Asynchronous communication
- Multiple subscribers
- Scalability

**Example**:
```python
# Publisher
await event_bus.publish(
    MessagingEvent(topic="tool.complete", data={"tool": "read_file"})
)

# Subscriber
await event_bus.subscribe("tool.*", self._on_tool_event)

async def _on_tool_event(self, event: MessagingEvent):
    if event.topic == "tool.complete":
        self._track_completion(event.data)
```

### 5. Vertical Template System

**Concept**: YAML-first vertical configuration

**Why**:
- 65-70% code reduction
- 8x faster vertical creation
- Consistent structure

**Example**:
```yaml
# my_vertical.yaml
metadata:
  name: "my_vertical"
  description: "My custom vertical"

tools:
  - name: read
    cost_tier: FREE
    enabled: true

prompts:
  system: "You are a helpful assistant..."
```

Generate vertical:
```bash
python scripts/generate_vertical.py \
  --template my_vertical.yaml \
  --output victor/my_vertical
```

---

## Common Workflows

### Workflow 1: Adding a New Tool

**Goal**: Add a new tool to Victor

**Steps**:

1. **Create tool file**:
```python
# victor/tools/my_tool.py
from victor.tools.base import BaseTool, ToolCostTier

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    cost_tier = ToolCostTier.FREE

    async def execute(self, **kwargs) -> ToolResult:
        # Implementation
        return ToolResult(output="Result")
```

2. **Register tool**:
```python
# victor/tools/registry.py
from victor.tools.my_tool import MyTool

def register_all_tools():
    tools = [MyTool()]
    for tool in tools:
        tool_registry.register(tool)
```

3. **Add tests**:
```python
# tests/unit/tools/test_my_tool.py
import pytest
from victor.tools.my_tool import MyTool

@pytest.mark.asyncio
async def test_my_tool():
    tool = MyTool()
    result = await tool.execute(param="value")
    assert result.output == "Result"
```

4. **Run tests**:
```bash
pytest tests/unit/tools/test_my_tool.py -v
```

### Workflow 2: Adding a New Coordinator

**Goal**: Add a new coordinator for complex operations

**Steps**:

1. **Define protocol**:
```python
# victor/agent/protocols.py
@runtime_checkable
class MyCoordinatorProtocol(Protocol):
    async def coordinate(self, data: str) -> dict:
        ...
```

2. **Implement coordinator**:
```python
# victor/agent/coordinators/my_coordinator.py
class MyCoordinator:
    def __init__(
        self,
        service_a: ServiceAProtocol,
        service_b: ServiceBProtocol,
    ):
        self._service_a = service_a
        self._service_b = service_b

    async def coordinate(self, data: str) -> dict:
        # Coordination logic
        result_a = await self._service_a.process(data)
        result_b = await self._service_b.process(result_a)
        return result_b
```

3. **Register in DI**:
```python
# victor/agent/service_provider.py
def configure_orchestrator_services(container, settings):
    container.register(
        MyCoordinatorProtocol,
        lambda c: MyCoordinator(
            service_a=c.get(ServiceAProtocol),
            service_b=c.get(ServiceBProtocol),
        ),
        ServiceLifetime.SINGLETON,
    )
```

4. **Add comprehensive tests**:
```python
# tests/unit/agent/coordinators/test_my_coordinator.py
class TestMyCoordinator:
    async def test_coordinate_success(self):
        # Arrange
        mock_service_a = Mock(spec=ServiceAProtocol)
        mock_service_b = Mock(spec=ServiceBProtocol)

        coordinator = MyCoordinator(mock_service_a, mock_service_b)

        # Act
        result = await coordinator.coordinate("test")

        # Assert
        assert result is not None
```

### Workflow 3: Creating a New Vertical

**Goal**: Create a domain-specific assistant

**Steps**:

1. **Create YAML template**:
```bash
cp victor/config/templates/base_vertical_template.yaml my_vertical.yaml
```

2. **Edit template**:
```yaml
# my_vertical.yaml
metadata:
  name: "my_vertical"
  description: "My custom vertical"

tools:
  - name: read
  - name: write
  - name: search

prompts:
  system: "You are a helpful assistant for..."

stages:
  - name: INITIAL
    goal: "Understand request"
    tool_budget: 5
```

3. **Generate vertical**:
```bash
python scripts/generate_vertical.py \
  --template my_vertical.yaml \
  --output victor/my_vertical
```

4. **Test vertical**:
```python
from victor.my_vertical import MyVerticalAssistant

config = MyVerticalAssistant.get_config()
assert len(config.tools) == 3
```

### Workflow 4: Adding Event Publishing

**Goal**: Add event publishing to component

**Steps**:

1. **Inject event bus**:
```python
class MyComponent:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus
```

2. **Publish events**:
```python
async def do_work(self, data: str):
    await self._event_bus.publish(
        MessagingEvent(
            topic="work.start",
            data={"data": data},
            correlation_id=str(uuid.uuid4()),
        )
    )

    result = await self._process(data)

    await self._event_bus.publish(
        MessagingEvent(
            topic="work.complete",
            data={"result": result},
            correlation_id=correlation_id,
        )
    )

    return result
```

3. **Subscribe to events**:
```python
class WorkMonitor:
    def __init__(self, event_bus: IEventBackend):
        event_bus.subscribe("work.*", self._on_work_event)

    async def _on_work_event(self, event: MessagingEvent):
        if event.topic == "work.complete":
            print(f"Work completed: {event.data}")
```

---

## Testing Guide

### Test Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
│   ├── agent/
│   │   └── coordinators/
│   ├── tools/
│   └── providers/
├── integration/       # Integration tests (slower, real components)
└── smoke/            # Smoke tests (quick sanity checks)
```

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run specific test file
pytest tests/unit/tools/test_my_tool.py

# Run specific test
pytest tests/unit/tools/test_my_tool.py::test_my_tool

# Run with coverage
pytest --cov=victor --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

### Writing Tests

#### Unit Test Example

```python
import pytest
from unittest.mock import Mock
from victor.agent.coordinators.my_coordinator import MyCoordinator

class TestMyCoordinator:
    @pytest.mark.asyncio
    async def test_coordinate_success(self):
        # Arrange
        mock_service_a = Mock(spec=ServiceAProtocol)
        mock_service_a.process.return_value = "processed"

        mock_service_b = Mock(spec=ServiceBProtocol)
        mock_service_b.process.return_value = {"result": "done"}

        coordinator = MyCoordinator(mock_service_a, mock_service_b)

        # Act
        result = await coordinator.coordinate("test")

        # Assert
        assert result == {"result": "done"}
        mock_service_a.process.assert_called_once_with("test")
        mock_service_b.process.assert_called_once_with("processed")

    @pytest.mark.asyncio
    async def test_coordinate_with_unicode(self):
        # Edge case: Unicode characters
        coordinator = MyCoordinator(mock_service_a, mock_service_b)
        result = await coordinator.coordinate("测试")
        assert result is not None
```

#### Integration Test Example

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_execution_integration():
    # Test with real components
    from victor.core.bootstrap import bootstrap_container

    container = bootstrap_container()
    tool_executor = container.get(ToolExecutorProtocol)

    result = await tool_executor.execute_tool(
        tool=read_file_tool,
        arguments={"filepath": "/tmp/test.txt"}
    )

    assert result.success
```

### Test Coverage Goals

- **New code**: Aim for 90%+ coverage
- **Coordinators**: Target 90-100% coverage
- **Tools**: Target 80-90% coverage
- **Protocols**: Test all implementations

### Common Test Patterns

#### Mock Protocol

```python
mock_service = Mock(spec=MyServiceProtocol)
mock_service.process.return_value = {"result": "test"}
```

#### Async Test

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_function()
    assert result is not None
```

#### Fixture

```python
@pytest.fixture
async def container():
    container = ServiceContainer()
    configure_services(container)
    yield container
    await container.cleanup()
```

#### Parametrize

```python
@pytest.mark.parametrize("input,expected", [
    ("test", "TEST"),
    ("hello", "HELLO"),
])
def test_uppercase(input, expected):
    assert uppercase(input) == expected
```

---

## Documentation Resources

### Essential Reading

1. **[CLAUDE.md](../CLAUDE.md)** - Project instructions and quick reference
2. **[README.md](../README.md)** - Project overview and features
3. **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System architecture

### Architecture Documentation

1. **[REFACTORING_OVERVIEW.md](architecture/REFACTORING_OVERVIEW.md)** - Refactoring summary
2. **[BEST_PRACTICES.md](architecture/BEST_PRACTICES.md)** - Usage patterns
3. **[PROTOCOLS_REFERENCE.md](architecture/PROTOCOLS_REFERENCE.md)** - Protocol documentation
4. **[MIGRATION_GUIDES.md](architecture/MIGRATION_GUIDES.md)** - Migration instructions

### Specialized Guides

1. **[ISP_MIGRATION_GUIDE.md](../ISP_MIGRATION_GUIDE.md)** - Vertical protocol migration
2. **[vertical_template_guide.md](../vertical_template_guide.md)** - Vertical templates
3. **[vertical_quickstart.md](../vertical_quickstart.md)** - Quick vertical creation

### Testing Documentation

1. **[testing/](testing/)** - Testing guides and patterns
2. **[LOW_PRIORITY_PHASE_COMPLETE.md](../LOW_PRIORITY_PHASE_COMPLETE.md)** - Testing initiative

### Initiative Documentation

1. **[INITIATIVE_EXECUTIVE_SUMMARY.md](../INITIATIVE_EXECUTIVE_SUMMARY.md)** - Business summary
2. **[INITIATIVE_TECHNICAL_SUMMARY.md](../INITIATIVE_TECHNICAL_SUMMARY.md)** - Technical details

---

## Contributing

### Contribution Workflow

1. **Fork repository**
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Make changes**
4. **Write tests**
5. **Run tests**: `pytest`
6. **Format code**: `black victor tests && ruff check --fix victor tests`
7. **Commit**: `git commit -m "feat: add my feature"`
8. **Push**: `git push origin feature/my-feature`
9. **Create PR**

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new tool for X
fix: resolve issue with Y
docs: update documentation for Z
test: add tests for W
refactor: improve code structure
```

### Code Review Checklist

- [ ] Tests pass (pytest)
- [ ] Coverage adequate (90%+ for new code)
- [ ] Type hints added
- [ ] Docstrings included
- [ ] Black formatted
- [ ] Ruff passes
- [ ] Mypy passes
- [ ] Documentation updated

### Pre-commit Hooks

Automatically run on commit:
- Black formatting
- Ruff linting
- Type checking (mypy)
- Test execution

---

## Getting Help

### Ask Questions

1. **GitHub Discussions** - https://github.com/vjsingh1984/victor/discussions
2. **GitHub Issues** - https://github.com/vjsingh1984/victor/issues
3. **Discord** - (link in README)

### Documentation

- **Quick questions**: Check CLAUDE.md first
- **Architecture**: See REFACTORING_OVERVIEW.md
- **Best practices**: See BEST_PRACTICES.md
- **Migration**: See MIGRATION_GUIDES.md

### Debugging Tips

#### Enable Debug Logging

```bash
export VICTOR_LOG_LEVEL=DEBUG
victor chat "test"
```

#### Run with Debugger

```python
import pdb; pdb.set_trace()
# Or use breakpoint() in Python 3.7+
```

#### Check Service Registration

```python
from victor.core.container import ServiceContainer

container = ServiceContainer()
# List all registered services
print(container._services.keys())
```

### Common Issues

**Issue**: Import error
**Solution**: Ensure you installed with `pip install -e ".[dev]"`

**Issue**: Tests fail
**Solution**: Run `pytest --clear-cache` and try again

**Issue**: Protocol not recognized
**Solution**: Ensure `@runtime_checkable` decorator is used

---

## Your First Week Checklist

### Day 1
- [ ] Set up development environment
- [ ] Run Victor locally
- [ ] Read CLAUDE.md
- [ ] Read REFACTORING_OVERVIEW.md

### Day 2-3
- [ ] Read BEST_PRACTICES.md
- [ ] Read MIGRATION_GUIDES.md
- [ ] Run test suite successfully
- [ ] Make a small change (fix typo, add comment)

### Day 4-5
- [ ] Pick a simple issue from GitHub
- [ ] Implement fix/feature
- [ ] Add tests
- [ ] Submit PR

### Week 2-3
- [ ] Review architecture documentation
- [ ] Understand coordinator pattern
- [ ] Understand protocol-based design
- [ ] Contribute to documentation

### Month 1
- [ ] Implement a new tool
- [ ] Implement a new coordinator
- [ ] Create a new vertical (using template system)
- [ ] Review 2-3 PRs

---

## Summary

Victor is a sophisticated AI coding assistant with a modern, scalable architecture. As a new developer:

1. **Start with CLAUDE.md** - Your primary reference
2. **Understand the architecture** - Protocols, DI, coordinators, events
3. **Follow best practices** - SOLID principles, testing, documentation
4. **Ask questions** - GitHub Discussions, Issues, Discord
5. **Contribute** - Start small, think big

**Welcome to the Victor community!** 🚀

---

**Last Updated**: January 18, 2026
**Version**: 0.5.1
**Next Review**: Ongoing

**Additional Resources**:
- [GitHub Repository](https://github.com/vjsingh1984/victor)
- [Documentation Index](index.md)
- [Contributing Guide](../CONTRIBUTING.md)
