# Testing Quick Start - Part 1

**Part 1 of 3:** Templates, Patterns, and Mocks

---

## Navigation

- **[Part 1: Templates & Patterns](#)** (Current)
- [Part 2: Coverage & Coordinator Patterns](part-2-coverage-coordinators.md)
- [Part 3: Troubleshooting & Reference](part-3-troubleshooting-reference.md)
- [**Complete Guide**](../TESTING_QUICK_START.md)

---
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
