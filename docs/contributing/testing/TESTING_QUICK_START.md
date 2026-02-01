# Coordinator Testing Quick Start

**Fast-track guide for testing Victor AI coordinators**

This quick start guide provides ready-to-use templates and common patterns for testing coordinators. Jump straight to what you need without reading the comprehensive guide.

## Table of Contents

- [Test File Templates](#test-file-templates)
- [Common Test Patterns](#common-test-patterns)
- [Mock Templates](#mock-templates)
- [Coverage Commands](#coverage-commands)
- [Coordinator-Specific Patterns](#coordinator-specific-patterns)
- [Troubleshooting](#troubleshooting)

## Test File Templates

### Basic Coordinator Test Template

```python
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 ...

"""Tests for CoordinatorName.

Test Coverage Strategy:
- Test all public methods
- Test async behavior and error handling
- Test edge cases and boundary conditions
"""

import pytest
from unittest.mock import Mock, AsyncMock

from victor.agent.coordinators.coordinator_name import CoordinatorName


class TestCoordinatorNameInitialization:
    """Test suite for initialization."""

    @pytest.fixture
    def coordinator(self) -> CoordinatorName:
        """Create coordinator instance."""
        mock_orch = Mock()
        return CoordinatorName(orchestrator=mock_orch)

    def test_initialization(self, coordinator: CoordinatorName):
        """Test coordinator initializes correctly."""
        assert coordinator is not None


class TestCoordinatorNamePublicAPI:
    """Test suite for public API."""

    @pytest.fixture
    def coordinator(self) -> CoordinatorName:
        """Create coordinator instance."""
        return CoordinatorName(orchestrator=Mock())

    @pytest.mark.asyncio
    async def test_method_success(self, coordinator: CoordinatorName):
        """Test method executes successfully."""
        # Arrange
        input_data = {"key": "value"}

        # Act
        result = await coordinator.method(input_data)

        # Assert
        assert result is not None


class TestCoordinatorNameErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def coordinator(self) -> CoordinatorName:
        """Create coordinator instance."""
        return CoordinatorName(orchestrator=Mock())

    @pytest.mark.asyncio
    async def test_handles_errors(self, coordinator: CoordinatorName):
        """Test handles errors gracefully."""
        # Arrange
        coordinator._dependency = AsyncMock(side_effect=ValueError("Error"))

        # Act & Assert
        with pytest.raises(ValueError):
            await coordinator.method({"invalid": "data"})
```

### Minimal Test Template

```python
"""Tests for CoordinatorName."""

import pytest
from unittest.mock import Mock
from victor.agent.coordinators.coordinator_name import CoordinatorName


@pytest.fixture
def coordinator() -> CoordinatorName:
    """Create coordinator for testing."""
    return CoordinatorName(orchestrator=Mock())


@pytest.mark.asyncio
async def test_basic_functionality(coordinator: CoordinatorName):
    """Test basic functionality."""
    result = await coordinator.method("input")
    assert result is not None
```

## Common Test Patterns

### Pattern 1: Testing Async Methods

```python
@pytest.mark.asyncio
async def test_async_method(coordinator: CoordinatorName):
    """Test async method."""
    # Arrange
    mock_dependency = AsyncMock(return_value="result")
    coordinator._dependency = mock_dependency

    # Act
    result = await coordinator.async_method("input")

    # Assert
    assert result == "expected"
    mock_dependency.assert_called_once_with("input")
```

### Pattern 2: Testing Error Handling

```python
@pytest.mark.asyncio
async def test_error_handling(coordinator: CoordinatorName):
    """Test error handling."""
    # Arrange
    coordinator._dependency = AsyncMock(
        side_effect=ValueError("Expected error")
    )

    # Act & Assert
    with pytest.raises(ValueError, match="Expected error"):
        await coordinator.method("input")
```

### Pattern 3: Testing State Changes

```python
@pytest.mark.asyncio
async def test_state_transition(coordinator: CoordinatorName):
    """Test state changes."""
    # Assert initial state
    assert coordinator.state == "initial"

    # Act
    await coordinator.start()

    # Assert new state
    assert coordinator.state == "running"
```

### Pattern 4: Testing with Return Values

```python
@pytest.mark.asyncio
async def test_return_value(coordinator: CoordinatorName):
    """Test return value processing."""
    # Arrange
    coordinator._dependency = AsyncMock(
        return_value=Mock(data="test", status="success")
    )

    # Act
    result = await coordinator.method("input")

    # Assert
    assert result.status == "success"
    assert result.data == "processed: test"
```

### Pattern 5: Testing Multiple Calls

```python
@pytest.mark.asyncio
async def test_multiple_calls(coordinator: CoordinatorName):
    """Test multiple method calls."""
    # Arrange
    coordinator._dependency = AsyncMock(return_value="result")

    # Act
    results = await asyncio.gather(
        coordinator.method("input1"),
        coordinator.method("input2"),
        coordinator.method("input3"),
    )

    # Assert
    assert len(results) == 3
    assert coordinator._dependency.call_count == 3
```

### Pattern 6: Testing Streaming Responses

```python
@pytest.mark.asyncio
async def test_streaming(coordinator: ChatCoordinator):
    """Test streaming response."""
    # Arrange
    chunks = [
        StreamChunk(content="Hello ", role="assistant"),
        StreamChunk(content="world!", role="assistant"),
    ]

    # Act
    result_chunks = []
    async for chunk in coordinator.stream_chat("Test"):
        result_chunks.append(chunk)

    # Assert
    assert len(result_chunks) == 2
    assert result_chunks[0].content == "Hello "
```

### Pattern 7: Testing Tool Execution

```python
@pytest.mark.asyncio
async def test_tool_execution(coordinator: ToolCoordinator):
    """Test tool execution."""
    # Arrange
    mock_tool = AsyncMock(
        return_value=ToolResult(
            tool_name="read_file",
            success=True,
            output="File content"
        )
    )
    coordinator._pipeline._execute_single_tool = mock_tool

    # Act
    result = await coordinator.execute_tool(
        "read_file",
        {"path": "test.txt"}
    )

    # Assert
    assert result.success is True
    assert result.output == "File content"
    mock_tool.assert_called_once_with("read_file", {"path": "test.txt"})
```

### Pattern 8: Testing Context Management

```python
@pytest.mark.asyncio
async def test_context_compaction(coordinator: ContextCoordinator):
    """Test context compaction."""
    # Arrange
    large_context = {"messages": ["msg"] * 1000}
    coordinator._compactor = AsyncMock(
        return_value=CompactionResult(
            compacted_context={"messages": ["msg"] * 100},
            tokens_saved=9000,
            messages_removed=900,
            strategy_used="truncation"
        )
    )

    # Act
    result = await coordinator.compact(large_context, budget=1000)

    # Assert
    assert result.tokens_saved == 9000
    coordinator._compactor.assert_called_once()
```

### Pattern 9: Testing Provider Switching

```python
@pytest.mark.asyncio
async def test_provider_switch(coordinator: ProviderCoordinator):
    """Test provider switching."""
    # Arrange
    old_provider = coordinator._current_provider
    new_provider = Mock()

    # Act
    await coordinator.switch_provider(new_provider)

    # Assert
    assert coordinator._current_provider == new_provider
    assert old_provider != new_provider
```

### Pattern 10: Testing Budget Enforcement

```python
@pytest.mark.asyncio
async def test_budget_enforcement(coordinator: ToolBudgetCoordinator):
    """Test tool budget enforcement."""
    # Arrange
    coordinator._budget = 5
    coordinator._tools_used = 4

    # Act
    can_execute = await coordinator.can_execute_tool("expensive_tool")

    # Assert
    assert can_execute is True

    # Act - exceed budget
    coordinator._tools_used = 5
    can_execute = await coordinator.can_execute_tool("expensive_tool")

    # Assert
    assert can_execute is False
```

## Mock Templates

### Mock Orchestrator Template

```python
@pytest.fixture
def mock_orchestrator() -> Mock:
    """Create comprehensive mock orchestrator."""
    orch = Mock()

    # Provider
    orch.provider = Mock()
    orch.provider.chat = AsyncMock()
    orch.provider.stream = AsyncMock()
    orch.provider.supports_tools = Mock(return_value=True)

    # Conversation
    orch.conversation = Mock()
    orch.conversation.messages = []
    orch.conversation.message_count = Mock(return_value=0)

    # Settings
    orch.settings = Mock()
    orch.settings.max_iterations = 10

    # Tool selector
    orch.tool_selector = Mock()
    orch.tool_selector.select_tools = AsyncMock(return_value=[])

    # Tool pipeline
    orch.tool_pipeline = Mock()
    orch.tool_pipeline.execute = AsyncMock()

    # Context
    orch._context_compactor = Mock()
    orch._context_compactor.compact = AsyncMock()

    return orch
```

### Mock Provider Template

```python
@pytest.fixture
def mock_provider() -> Mock:
    """Create mock LLM provider."""
    provider = Mock()
    provider.chat = AsyncMock(
        return_value=CompletionResponse(
            content="Test response",
            role="assistant",
            tool_calls=None,
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )
    )

    # Create async stream generator
    async def stream_gen():
        yield StreamChunk(content="Chunk 1", role="assistant")
        yield StreamChunk(content="Chunk 2", role="assistant")

    provider.stream = stream_gen
    provider.supports_tools = Mock(return_value=True)
    return provider
```

### Mock Tool Registry Template

```python
@pytest.fixture
def mock_tool_registry() -> Mock:
    """Create mock tool registry."""
    registry = Mock()
    registry.get_all_tool_names = Mock(
        return_value=["read_file", "write_file", "shell"]
    )
    registry.get_tool = Mock(return_value=Mock(
        name="test_tool",
        description="Test tool",
        parameters={}
    ))
    registry.has_tool = Mock(return_value=True)
    return registry
```

### Mock Tool Pipeline Template

```python
@pytest.fixture
def mock_tool_pipeline() -> Mock:
    """Create mock tool pipeline."""
    pipeline = Mock()
    pipeline.execute_tool_calls = AsyncMock(
        return_value=[
            ToolResult(
                tool_name="read_file",
                success=True,
                output="Content"
            )
        ]
    )
    pipeline._execute_single_tool = AsyncMock(
        return_value=ToolResult(
            tool_name="test",
            success=True,
            output="Result"
        )
    )
    return pipeline
```

### Mock Context Compactor Template

```python
@pytest.fixture
def mock_compactor() -> Mock:
    """Create mock context compactor."""
    compactor = Mock()
    compactor.compact = AsyncMock(
        return_value=CompactionResult(
            compacted_context={"messages": []},
            tokens_saved=5000,
            messages_removed=50,
            strategy_used="truncation"
        )
    )
    compactor.can_compact = Mock(return_value=True)
    return compactor
```

## Coverage Commands

### Run All Coordinator Tests

```bash
# Run all coordinator tests
pytest tests/unit/agent/coordinators/ -v

# Run with coverage
pytest tests/unit/agent/coordinators/ \
    --cov=victor.agent.coordinators \
    --cov-report=term-missing \
    --cov-report=html

# Run specific coordinator test
pytest tests/unit/agent/coordinators/test_chat_coordinator.py -v

# Run specific test
pytest tests/unit/agent/coordinators/test_chat_coordinator.py::TestChatCoordinatorChat::test_chat_with_tools -v
```

### Coverage for Specific Coordinator

```bash
# ChatCoordinator coverage
pytest --cov=victor.agent.coordinators.chat_coordinator \
       --cov-report=term-missing \
       tests/unit/agent/coordinators/test_chat_coordinator.py

# ToolCoordinator coverage
pytest --cov=victor.agent.coordinators.tool_coordinator \
       --cov-report=term-missing \
       tests/unit/agent/coordinators/test_tool_coordinator.py
```

### Generate HTML Coverage Report

```bash
# Generate detailed HTML report
pytest tests/unit/agent/coordinators/ \
    --cov=victor.agent.coordinators \
    --cov-report=html \
    --cov-report=term

# Open report
open htmlcov/index.html
```

### Run Tests in Parallel

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest tests/unit/agent/coordinators/ -n 4

# Run with coverage in parallel
pytest tests/unit/agent/coordinators/ \
    -n 4 \
    --cov=victor.agent.coordinators \
    --cov-report=term-missing
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit tests/unit/agent/coordinators/

# Run only integration tests
pytest -m integration tests/unit/agent/coordinators/

# Run only slow tests
pytest -m slow tests/unit/agent/coordinators/

# Skip slow tests
pytest -m "not slow" tests/unit/agent/coordinators/
```

### Debug Failed Tests

```bash
# Run with verbose output
pytest tests/unit/agent/coordinators/test_chat_coordinator.py -vv

# Run with pdb on failure
pytest tests/unit/agent/coordinators/test_chat_coordinator.py --pdb

# Show local variables on failure
pytest tests/unit/agent/coordinators/test_chat_coordinator.py -l

# Stop on first failure
pytest tests/unit/agent/coordinators/ -x

# Show print statements
pytest tests/unit/agent/coordinators/ -s
```

## Coordinator-Specific Patterns

### ChatCoordinator Patterns

```python
"""ChatCoordinator test patterns."""

@pytest.mark.asyncio
async def test_chat_with_no_tools(coordinator: ChatCoordinator):
    """Test chat without tool calls."""
    # Arrange
    coordinator._orchestrator.provider.chat = AsyncMock(
        return_value=CompletionResponse(
            content="Response",
            role="assistant",
            tool_calls=None
        )
    )

    # Act
    response = await coordinator.chat("Hello")

    # Assert
    assert response.content == "Response"
    assert response.tool_calls is None


@pytest.mark.asyncio
async def test_chat_with_tools(coordinator: ChatCoordinator):
    """Test chat with tool calls."""
    # Arrange
    tool_call = ToolCall(
        id="call_1",
        name="read_file",
        arguments={"path": "test.txt"}
    )

    coordinator._orchestrator.provider.chat = AsyncMock(
        return_value=CompletionResponse(
            content="I'll read the file",
            role="assistant",
            tool_calls=[tool_call]
        )
    )

    coordinator._orchestrator._handle_tool_calls = AsyncMock(
        return_value=[ToolResult(
            tool_name="read_file",
            success=True,
            output="File content"
        )]
    )

    # Act
    response = await coordinator.chat("Read test.txt")

    # Assert
    assert response.content is not None
    coordinator._orchestrator._handle_tool_calls.assert_called_once()
```

### ToolCoordinator Patterns

```python
"""ToolCoordinator test patterns."""

@pytest.mark.asyncio
async def test_execute_single_tool(coordinator: ToolCoordinator):
    """Test single tool execution."""
    # Arrange
    coordinator._pipeline._execute_single_tool = AsyncMock(
        return_value=ToolResult(
            tool_name="read_file",
            success=True,
            output="Content"
        )
    )

    # Act
    result = await coordinator.execute_tool(
        "read_file",
        {"path": "test.txt"}
    )

    # Assert
    assert result.success is True
    assert result.output == "Content"


@pytest.mark.asyncio
async def test_batch_tool_execution(coordinator: ToolCoordinator):
    """Test batch tool execution."""
    # Arrange
    tool_calls = [
        ToolCall(id="1", name="read_file", arguments={"path": "file1.txt"}),
        ToolCall(id="2", name="read_file", arguments={"path": "file2.txt"}),
    ]

    coordinator._pipeline.execute_tool_calls = AsyncMock(
        return_value=[
            ToolResult(tool_name="read_file", success=True, output="Content 1"),
            ToolResult(tool_name="read_file", success=True, output="Content 2"),
        ]
    )

    # Act
    results = await coordinator.execute_batch(tool_calls)

    # Assert
    assert len(results) == 2
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_budget_enforcement(coordinator: ToolCoordinator):
    """Test tool budget is enforced."""
    # Arrange
    coordinator._config.default_budget = 5
    coordinator._tools_used = 5

    # Act
    can_execute = await coordinator.can_execute_tool("any_tool")

    # Assert
    assert can_execute is False
```

### ContextCoordinator Patterns

```python
"""ContextCoordinator test patterns."""

@pytest.mark.asyncio
async def test_context_budget_check(coordinator: ContextCoordinator):
    """Test context budget checking."""
    # Arrange
    context = {"token_count": 8000}
    budget = ContextBudget(max_tokens=10000)

    # Act
    exceeds_budget = await coordinator.exceeds_budget(context, budget)

    # Assert
    assert exceeds_budget is False


@pytest.mark.asyncio
async def test_context_compaction(coordinator: ContextCoordinator):
    """Test context compaction."""
    # Arrange
    large_context = {"messages": ["msg"] * 1000, "token_count": 15000}

    coordinator._compactor = AsyncMock(
        return_value=CompactionResult(
            compacted_context={"messages": ["msg"] * 100},
            tokens_saved=14000,
            messages_removed=900,
            strategy_used="truncation"
        )
    )

    # Act
    result = await coordinator.compact(large_context, budget=1000)

    # Assert
    assert result.tokens_saved == 14000
    assert result.messages_removed == 900


@pytest.mark.asyncio
async def test_semantic_compaction(coordinator: ContextCoordinator):
    """Test semantic-aware compaction."""
    # Arrange
    context = {
        "messages": [
            {"role": "system", "content": "Important system message"},
            {"role": "user", "content": "Old message"},
            {"role": "user", "content": "Recent important message"},
        ]
    }

    coordinator._semantic_compactor = AsyncMock(
        return_value=CompactionResult(
            compacted_context={
                "messages": [
                    {"role": "system", "content": "Important system message"},
                    {"role": "user", "content": "Recent important message"},
                ]
            },
            tokens_saved=1000,
            messages_removed=1,
            strategy_used="semantic"
        )
    )

    # Act
    result = await coordinator.compact_semantic(context, budget=2000)

    # Assert
    assert len(result.compacted_context["messages"]) == 2
```

### ProviderCoordinator Patterns

```python
"""ProviderCoordinator test patterns."""

@pytest.mark.asyncio
async def test_switch_provider(coordinator: ProviderCoordinator):
    """Test switching providers."""
    # Arrange
    old_provider = coordinator._current_provider
    new_provider = Mock(model="new-model")

    # Act
    await coordinator.switch_provider(new_provider)

    # Assert
    assert coordinator._current_provider == new_provider
    assert coordinator._current_provider.model == "new-model"


@pytest.mark.asyncio
async def test_fallback_provider(coordinator: ProviderCoordinator):
    """Test fallback to secondary provider."""
    # Arrange
    primary = AsyncMock(side_effect=ProviderError("Failed"))
    secondary = AsyncMock(
        return_value=CompletionResponse(content="Fallback")
    )

    coordinator._primary_provider = primary
    coordinator._secondary_provider = secondary

    # Act
    result = await coordinator.chat_with_fallback("Test")

    # Assert
    assert result.content == "Fallback"
    primary.assert_called_once()
    secondary.assert_called_once()
```

### AnalyticsCoordinator Patterns

```python
"""AnalyticsCoordinator test patterns."""

@pytest.mark.asyncio
async def test_metrics_collection(coordinator: AnalyticsCoordinator):
    """Test metrics collection."""
    # Arrange
    coordinator._metrics_collector = Mock()

    # Act
    await coordinator.track_event("tool_execution", {
        "tool_name": "read_file",
        "duration": 0.5,
        "success": True
    })

    # Assert
    coordinator._metrics_collector.record.assert_called_once()


@pytest.mark.asyncio
async def test_session_analytics(coordinator: AnalyticsCoordinator):
    """Test session-level analytics."""
    # Act
    analytics = await coordinator.get_session_analytics()

    # Assert
    assert analytics.total_messages == 0
    assert analytics.total_tool_calls == 0
    assert analytics.total_duration > 0
```

### PromptCoordinator Patterns

```python
"""PromptCoordinator test patterns."""

@pytest.mark.asyncio
async def test_prompt_building(coordinator: PromptCoordinator):
    """Test system prompt building."""
    # Arrange
    contributors = [
        Mock(contribute=lambda: "System instruction"),
        Mock(contribute=lambda: "Safety guideline"),
    ]

    # Act
    prompt = await coordinator.build_prompt(contributors)

    # Assert
    assert "System instruction" in prompt
    assert "Safety guideline" in prompt


@pytest.mark.asyncio
async def test_prompt_caching(coordinator: PromptCoordinator):
    """Test prompt caching."""
    # Arrange
    contributors = [Mock(contribute=lambda: "Instruction")]

    # Act - first call
    prompt1 = await coordinator.build_prompt(contributors)

    # Act - second call (should be cached)
    prompt2 = await coordinator.build_prompt(contributors)

    # Assert
    assert prompt1 == prompt2
    # Verify cache was used
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Async tests not running

```bash
# Problem: Tests hang or don't execute
# Solution: Install pytest-asyncio
pip install pytest-asyncio

# Add to pyproject.toml if needed
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

#### Issue: Mock not being called

```python
# Problem: Mock.assert_called() fails
# Solution: Use AsyncMock for async methods

# WRONG
coordinator._dependency = Mock(return_value="result")

# CORRECT
coordinator._dependency = AsyncMock(return_value="result")
```

#### Issue: Coverage report missing lines

```bash
# Problem: Coverage shows 0% or missing lines
# Solution: Ensure source is in PYTHONPATH

export PYTHONPATH=/Users/vijaysingh/code/codingagent:$PYTHONPATH
pytest --cov=victor.agent.coordinators.chat_coordinator tests/
```

#### Issue: Tests pass locally but fail in CI

```bash
# Problem: Environment-specific test behavior
# Solution: Use environment isolation

# Mark tests that require specific environment
@pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
async def test_with_ollama():
    pass
```

#### Issue: Slow test execution

```bash
# Problem: Tests take too long
# Solution: Run tests in parallel

pytest tests/unit/agent/coordinators/ -n auto

# Or skip slow tests
pytest -m "not slow" tests/unit/agent/coordinators/
```

#### Issue: Fixture not found

```python
# Problem: pytestfixture not found error
# Solution: Check fixture scope and location

# Make sure fixtures are in conftest.py or test file
# For shared fixtures, use tests/conftest.py

@pytest.fixture
def shared_fixture():
    return "value"
```

### Debug Commands

```bash
# Run with verbose output
pytest tests/unit/agent/coordinators/test_chat_coordinator.py -vv

# Show print statements
pytest tests/unit/agent/coordinators/ -s

# Stop on first failure and drop into debugger
pytest tests/unit/agent/coordinators/ -x --pdb

# Show local variables on failure
pytest tests/unit/agent/coordinators/ -l

# Run specific test with maximum verbosity
pytest tests/unit/agent/coordinators/test_chat_coordinator.py::TestChatCoordinatorChat::test_chat_with_tools -vvvl -s
```

### Getting Help

- Check [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed guidance
- Review existing test files in `tests/unit/agent/coordinators/`
- Consult pytest documentation: https://docs.pytest.org/
- Check Victor architecture docs: `docs/architecture/overview.md`

## Quick Reference Card

### Test Structure

```python
# 1. Import dependencies
import pytest
from unittest.mock import Mock, AsyncMock

# 2. Import coordinator
from victor.agent.coordinators.coordinator_name import CoordinatorName

# 3. Create fixtures
@pytest.fixture
def coordinator():
    return CoordinatorName(orchestrator=Mock())

# 4. Write tests
@pytest.mark.asyncio
async def test_something(coordinator):
    result = await coordinator.method()
    assert result is not None
```

### Common Assertions

```python
# Equality
assert result == expected

# Boolean
assert result.success is True
assert result.error is None

# Exceptions
with pytest.raises(ValueError):
    coordinator.method()

# Mock calls
mock_method.assert_called_once()
mock_method.assert_called_with(arg1, arg2)

# State
assert coordinator.state == "running"
```

### Common Mocks

```python
# Async mock
AsyncMock(return_value="result")

# Mock with side effect
Mock(side_effect=ValueError("Error"))

# Async generator
async def stream_gen():
    yield StreamChunk(content="chunk")
```

### Running Tests

```bash
# Run all
pytest tests/unit/agent/coordinators/

# Run one file
pytest tests/unit/agent/coordinators/test_chat_coordinator.py

# Run with coverage
pytest --cov=victor.agent.coordinators.chat_coordinator tests/

# Run in parallel
pytest tests/unit/agent/coordinators/ -n 4
```

## Next Steps

1. **Create your test file**: Copy the appropriate template
2. **Write your first test**: Start with a simple success case
3. **Add edge cases**: Test error handling and boundary conditions
4. **Check coverage**: Ensure you meet the >75% target
5. **Run tests locally**: Verify everything passes
6. **Submit PR**: Include coverage report in PR description

For detailed guidance, see [TESTING_GUIDE.md](TESTING_GUIDE.md).

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
