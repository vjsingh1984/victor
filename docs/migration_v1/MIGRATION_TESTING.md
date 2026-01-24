# Testing Migration Guide: Victor 0.5.x to 0.5.0

This guide explains how to migrate your tests from Victor 0.5.x to 0.5.0.

## Table of Contents

1. [Overview](#overview)
2. [Test Infrastructure Changes](#test-infrastructure-changes)
3. [Test Fixture Changes](#test-fixture-changes)
4. [Provider Mock Changes](#provider-mock-changes)
5. [Test Migration Examples](#test-migration-examples)
6. [New Testing Patterns](#new-testing-patterns)

---

## Overview

### Key Changes

- **New Test Infrastructure**: Factories, mocks, and fixtures
- **Async Tests**: Most tests now require `asyncio`
- **Protocol Mocks**: Mock protocols instead of concrete classes
- **Improved Fixtures**: Better test isolation and setup

---

## Test Infrastructure Changes

### 1. Test Factories

**New in 0.5.0**:
```python
# tests/factories.py
from victor.testing.factories import (
    OrchestratorFactory,
    ProviderFactory,
    ToolFactory,
    ContextFactory,
)

# Create test orchestrator
orchestrator = OrchestratorFactory.create_test_orchestrator()

# Create mock provider
provider = ProviderFactory.create_mock_provider("openai")

# Create test tool
tool = ToolFactory.create_test_tool(
    name="test_tool",
    description="Test tool"
)

# Create test context
context = ContextFactory.create_test_context()
```

### 2. Mock Providers

**Before (0.5.x)**:
```python
from unittest.mock import Mock
from victor.providers.openai_provider import OpenAIProvider

provider = Mock(spec=OpenAIProvider)
provider.chat.return_value = "Mock response"
```

**After (0.5.0)**:
```python
from tests.mocks import MockProvider
from victor.protocols import BaseProviderProtocol

provider = MockProvider(model="gpt-4")
response = await provider.chat(messages=[...])

# Or use factory
from victor.testing.factories import ProviderFactory

provider = ProviderFactory.create_mock_provider("openai")
```

### 3. Test Utilities

**New in 0.5.0**:
```python
from victor.testing.utils import (
    create_test_orchestrator,
    create_mock_settings,
    create_test_context,
    async_test,
)

# Create test orchestrator with mocks
orchestrator = create_test_orchestrator(
    provider="mock",
    settings=create_mock_settings()
)

# Run async test
@async_test
async def test_something():
    result = await orchestrator.chat("test")
    assert result.content == "expected"
```

---

## Test Fixture Changes

### 1. Orchestrator Fixture

**Before (0.5.x)**:
```python
import pytest
from victor.agent.orchestrator import AgentOrchestrator

@pytest.fixture
def orchestrator():
    provider = OpenAIProvider(api_key="test-key")
    return AgentOrchestrator(provider=provider)
```

**After (0.5.0)**:
```python
import pytest
from victor.testing.fixtures import test_orchestrator

# Use built-in fixture
@pytest.mark.asyncio
async def test_with_orchestrator(test_orchestrator):
    result = await test_orchestrator.chat("test")
    assert result.content == "expected"

# Or create custom fixture
@pytest.fixture
async def custom_orchestrator():
    from victor.testing.factories import OrchestratorFactory
    return await OrchestratorFactory.create_test_orchestrator(
        provider="mock",
        mode="build"
    )
```

### 2. Provider Fixture

**Before (0.5.x)**:
```python
@pytest.fixture
def mock_provider():
    provider = Mock()
    provider.chat.return_value = "response"
    return provider
```

**After (0.5.0)**:
```python
# Use built-in fixture
@pytest.fixture
def mock_provider(mock_openai_provider):
    return mock_openai_provider

# Or use factory
@pytest.fixture
def custom_provider():
    from victor.testing.factories import ProviderFactory
    return ProviderFactory.create_mock_provider(
        name="anthropic",
        model="claude-sonnet-4-5",
        responses=["Response 1", "Response 2"]
    )
```

### 3. Context Fixture

**New in 0.5.0**:
```python
@pytest.fixture
def test_context():
    from victor.testing.factories import ContextFactory
    return ContextFactory.create_test_context(
        query="Test query",
        conversation_stage=Stage.ANALYZING,
        tool_budget=10
    )
```

### 4. Environment Isolation

**Enhanced in 0.5.0**:
```python
# Automatic environment isolation
@pytest.mark.asyncio
async def test_with_isolated_env():
    # Environment variables isolated per test
    # Automatically reset after test
    pass

# Manual isolation
@pytest.fixture
def isolated_env():
    from victor.testing.utils import isolated_environment
    with isolated_environment():
        os.environ["TEST_VAR"] = "test"
        yield
        # Automatically cleaned up
```

---

## Provider Mock Changes

### 1. Mock Provider API

**Before (0.5.x)**:
```python
from unittest.mock import Mock

provider = Mock()
provider.chat.return_value = "response"
provider.stream_chat.return_value = iter([
    Mock(content="chunk1"),
    Mock(content="chunk2")
])
```

**After (0.5.0)**:
```python
from tests.mocks.providers import MockProvider

provider = MockProvider(
    model="gpt-4",
    responses=["Full response"],
    streaming=False
)

# Streaming mock
provider = MockProvider(
    model="gpt-4",
    responses=["chunk1", "chunk2"],
    streaming=True
)

# Use in test
async def test_with_mock():
    response = await provider.chat(messages=[...])
    assert response.content == "Full response"

    async for chunk in provider.stream_chat(messages=[...]):
        print(chunk.content)
```

### 2. Provider Factory

**New in 0.5.0**:
```python
from victor.testing.factories import ProviderFactory

# Create mock provider
provider = ProviderFactory.create_mock_provider(
    name="openai",
    model="gpt-4",
    responses=["Response 1", "Response 2"]
)

# Create with custom behavior
provider = ProviderFactory.create_mock_provider(
    name="anthropic",
    model="claude-sonnet-4-5",
    response_fn=lambda req: f"Response to: {req[-1]['content']}"
)
```

---

## Test Migration Examples

### Example 1: Simple Orchestrator Test

**Before (0.5.x)**:
```python
import pytest
from victor.agent.orchestrator import AgentOrchestrator
from victor.providers.openai_provider import OpenAIProvider

def test_chat():
    provider = OpenAIProvider(api_key="test-key")
    orchestrator = AgentOrchestrator(provider=provider)
    result = orchestrator.chat("Hello")
    assert result == "expected"
```

**After (0.5.0)**:
```python
import pytest
from victor.testing.fixtures import test_orchestrator

@pytest.mark.asyncio
async def test_chat(test_orchestrator):
    result = await test_orchestrator.chat("Hello")
    assert result.content == "expected"
```

### Example 2: Tool Execution Test

**Before (0.5.x)**:
```python
def test_tool_execution():
    orchestrator = AgentOrchestrator(...)
    result = orchestrator.execute_tool(
        tool_name="read_file",
        arguments={"path": "test.txt"}
    )
    assert result["content"] == "file content"
```

**After (0.5.0)**:
```python
@pytest.mark.asyncio
async def test_tool_execution(test_orchestrator):
    from victor.protocols import ToolCoordinatorProtocol

    coordinator = test_orchestrator.container.get(ToolCoordinatorProtocol)
    result = await coordinator.execute_tool(
        tool=tool_registry.get_tool("read_file"),
        arguments={"path": "test.txt"}
    )
    assert result.output["content"] == "file content"
```

### Example 3: Provider Test

**Before (0.5.x)**:
```python
def test_provider():
    provider = OpenAIProvider(api_key="test-key")
    result = provider.chat([{"role": "user", "content": "test"}])
    assert result == "response"
```

**After (0.5.0)**:
```python
@pytest.mark.asyncio
async def test_provider():
    from victor.testing.factories import ProviderFactory

    provider = ProviderFactory.create_mock_provider("openai")
    result = await provider.chat([{"role": "user", "content": "test"}])
    assert result.content == "response"
```

### Example 4: Workflow Test

**Before (0.5.x)**:
```python
def test_workflow():
    executor = WorkflowExecutor(orchestrator)
    result = await executor.execute(workflow, context)
    assert result["status"] == "complete"
```

**After (0.5.0)**:
```python
@pytest.mark.asyncio
async def test_workflow(test_orchestrator):
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

    compiler = UnifiedWorkflowCompiler(test_orchestrator)
    compiled = compiler.compile(workflow)
    result = await compiled.invoke(context)
    assert result["status"] == "complete"
```

---

## New Testing Patterns

### 1. Protocol-Based Testing

**New in 0.5.0**:
```python
from victor.protocols import ToolRegistryProtocol
from unittest.mock import Mock

@pytest.fixture
def mock_tool_registry():
    # Create mock that implements protocol
    mock = Mock(spec=ToolRegistryProtocol)
    mock.get_tools.return_value = []
    mock.get_tool.return_value = mock_tool
    return mock

@pytest.mark.asyncio
async def test_with_protocol_mock(mock_tool_registry):
    tools = await mock_tool_registry.get_tools()
    assert isinstance(tools, list)
```

### 2. Async Test Patterns

**New in 0.5.0**:
```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    # Async test - use await
    result = await orchestrator.chat("test")
    assert result.content == "expected"

@pytest.mark.asyncio
async def test_async_streaming():
    # Streaming test
    chunks = []
    async for chunk in orchestrator.stream_chat("test"):
        chunks.append(chunk.content)
    assert len(chunks) > 0
```

### 3. Integration Test Patterns

**New in 0.5.0**:
```python
@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.asyncio
async def test_integration_with_real_provider():
    # Requires network and real provider
    settings = Settings(provider="openai")
    orchestrator = bootstrap_orchestrator(settings)
    result = await orchestrator.chat("test")
    assert result.content
```

### 4. Property-Based Testing

**New in 0.5.0**:
```python
import hypothesis
from hypothesis import strategies as st

@hypothesis.given(st.text())
@pytest.mark.asyncio
async def test_property_based(text_input):
    """Test that chat handles various inputs."""
    orchestrator = create_test_orchestrator()
    result = await orchestrator.chat(text_input)
    assert result.content is not None
    assert isinstance(result.content, str)
```

---

## Test Markers

### Enhanced Markers

**New in 0.5.0**:
```python
# Test type markers
@pytest.mark.unit
def test_unit_test():
    """Unit test (fast, isolated)"""

@pytest.mark.integration
def test_integration_test():
    """Integration test (requires external services)"""

@pytest.mark.smoke
def test_smoke_test():
    """Smoke test (quick sanity check)"""

# Characteristic markers
@pytest.mark.slow
def test_slow_test():
    """Slow test (deselect with '-m \"not slow\"')"""

@pytest.mark.requires_network
def test_network_test():
    """Test requiring network access"""

@pytest.mark.requires_docker
def test_docker_test():
    """Test requiring Docker daemon"""

# Feature-specific markers
@pytest.mark.workflows
def test_workflow_test():
    """Workflow-related test"""

@pytest.mark.agents
def test_multi_agent_test():
    """Multi-agent test"""

@pytest.mark.benchmark
def test_benchmark():
    """Performance benchmark"""
```

---

## Common Test Patterns

### 1. Orchestrator Test

```python
@pytest.mark.asyncio
async def test_orchestrator_chat(test_orchestrator):
    result = await test_orchestrator.chat("Hello, world!")
    assert result.content
    assert result.metadata.tokens_used > 0
```

### 2. Tool Execution Test

```python
@pytest.mark.asyncio
async def test_tool_execution(test_orchestrator):
    coordinator = test_orchestrator.container.get(ToolCoordinatorProtocol)
    result = await coordinator.execute_tool(
        tool=mock_tool,
        arguments={"path": "test.txt"}
    )
    assert result.success
    assert result.output
```

### 3. Workflow Test

```python
@pytest.mark.asyncio
@pytest.mark.workflows
async def test_workflow_execution(test_orchestrator):
    compiler = UnifiedWorkflowCompiler(test_orchestrator)
    compiled = compiler.compile(workflow_def)
    result = await compiled.invoke(context)
    assert result["status"] == "complete"
```

### 4. Provider Test

```python
@pytest.mark.asyncio
async def test_provider_chat():
    provider = ProviderFactory.create_mock_provider("openai")
    result = await provider.chat(messages=[...])
    assert result.content
```

---

## Test Utilities

### Reset Singletons

**New in 0.5.0**:
```python
@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test."""
    from victor.testing.utils import reset_singletons
    reset_singletons()
    yield
    # Cleanup after test
    reset_singletons()
```

### Auto-Mock Docker

**New in 0.5.0**:
```python
# Automatically applied for orchestrator tests
@pytest.fixture(autouse=True)
def auto_mock_docker():
    from victor.testing.docker import auto_mock_docker_for_orchestrator
    with auto_mock_docker_for_orchestrator():
        yield
```

### Environment Isolation

**New in 0.5.0**:
```python
@pytest.fixture(autouse=True)
def isolate_env():
    """Isolate environment variables."""
    from victor.testing.utils import isolated_environment
    with isolated_environment():
        yield
```

---

## Migration Checklist

Use this checklist to ensure you've migrated all tests:

- [ ] Update orchestrator initialization to use `bootstrap_orchestrator()`
- [ ] Make all async tests use `@pytest.mark.asyncio`
- [ ] Replace direct provider instantiation with `ProviderFactory`
- [ ] Update test fixtures to use built-in fixtures
- [ ] Replace `SharedToolRegistry` with DI container
- [ ] Update event bus usage to use `create_event_backend()`
- [ ] Add environment isolation to tests that need it
- [ ] Update assertions to work with new return types
- [ ] Add appropriate test markers (unit, integration, etc.)
- [ ] Run tests and verify all pass

---

## Additional Resources

- [Main Migration Guide](./MIGRATION_GUIDE.md)
- [API Migration Guide](./MIGRATION_API.md)
- [Testing Best Practices](../architecture/TESTING_BEST_PRACTICES.md)
- [Pytest Documentation](https://docs.pytest.org/)

---

**Last Updated**: 2025-01-21
**Version**: 0.5.0
