# Testing Guide - Part 1

**Part 1 of 3:** Testing Philosophy, Structure, and Mock Strategies

---

## Navigation

- **[Part 1: Philosophy & Mocks](#)** (Current)
- [Part 2: Async & Coverage](part-2-async-coverage.md)
- [Part 3: Scenarios & Best Practices](part-3-scenarios-best-practices.md)
- [**Complete Guide**](../TESTING_GUIDE.md)

---
# Coordinator Testing Guide

**Comprehensive guide for testing Victor AI coordinators**

This guide provides detailed instructions, patterns,
  and best practices for testing coordinators in the Victor AI codebase. It covers both application-layer and
  framework-layer coordinators,
  with specific guidance for different coordinator types.

## Table of Contents

- [Overview](#overview)
- [Testing Philosophy](#testing-philosophy)
- [Test Structure and Organization](#test-structure-and-organization)
- [Mock Strategies by Coordinator Type](#mock-strategies-by-coordinator-type)
- [Async Testing Patterns](#async-testing-patterns)
- [Coverage Targets and Guidelines](#coverage-targets-and-guidelines)
- [Common Test Scenarios](#common-test-scenarios)
- [Error Handling Testing](#error-handling-testing)
- [Performance Testing](#performance-testing)
- [Integration Testing](#integration-testing)

## Overview

Victor AI uses a two-layer coordinator architecture:

### Application Layer Coordinators (`victor/agent/coordinators/`)

Business logic coordinators that manage the AI agent conversation lifecycle:

- **ChatCoordinator** - LLM chat operations and streaming
- **ToolCoordinator** - Tool validation, execution, and budget enforcement
- **ContextCoordinator** - Context management and compaction strategies
- **AnalyticsCoordinator** - Session metrics and analytics collection
- **PromptCoordinator** - System prompt building from contributors
- **SessionCoordinator** - Conversation session lifecycle
- **ProviderCoordinator** - Provider switching and management
- **ModeCoordinator** - Agent modes (build, plan, explore)
- **ToolSelectionCoordinator** - Semantic tool selection
- **CheckpointCoordinator** - Workflow checkpoint management
- **EvaluationCoordinator** - LLM evaluation and benchmarking
- **MetricsCoordinator** - System metrics collection

### Framework Layer Coordinators (`victor/framework/coordinators/`)

Domain-agnostic workflow infrastructure reusable across all verticals:

- **YAMLWorkflowCoordinator** - YAML workflow loading and execution
- **GraphExecutionCoordinator** - StateGraph/CompiledGraph execution
- **HITLCoordinator** - Human-in-the-loop workflow integration
- **CacheCoordinator** - Workflow caching system

### Testing Statistics

As of the coordinator testing initiative completion:

- **21 test files** created
- **1,149 test cases** implemented
- **23,413 lines of test code** written
- **92.13% average coverage** across 20 coordinators
- **324x speedup** achieved through parallelism

## Testing Philosophy

### Core Principles

1. **Test Isolation**: Each test should be completely independent. Use fixtures and mocks to isolate the coordinator
  under test.
2. **Arrange-Act-Assert (AAA)**: Structure tests clearly with setup, execution, and verification phases.
3. **Single Responsibility**: Each test should verify one specific behavior or edge case.
4. **Descriptive Names**: Test names should clearly describe what is being tested and the expected outcome.
5. **Mock External Dependencies**: Coordinators should be tested in isolation from external services (LLM providers,
  file system, etc.).
6. **Test Public APIs**: Focus on testing the public interface of coordinators, not implementation details.
7. **Async/Await**: Always use async test methods for async coordinator methods.

### Test Pyramid

```
        E2E (5%)
       /        \
    Integration (15%)
   /                \
Unit Tests (80%)
```

- **Unit Tests (80%)**: Test individual coordinator methods in isolation with mocks
- **Integration Tests (15%)**: Test coordinator interactions with real dependencies
- **E2E Tests (5%)**: Test complete workflows through multiple coordinators

## Test Structure and Organization

### File Organization

Test files mirror the source code structure:

```
tests/
├── unit/
│   ├── agent/
│   │   └── coordinators/
│   │       ├── test_chat_coordinator.py
│   │       ├── test_tool_coordinator.py
│   │       ├── test_context_coordinator.py
│   │       └── ...
│   └── coordinators/
│       ├── test_conversation_coordinator.py
│       ├── test_team_coordinator.py
│       └── ...
└── smoke/
    └── test_coordinator_smoke.py
```

### Test File Template

```python
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for CoordinatorName.

This test file provides comprehensive coverage for CoordinatorName.

Test Coverage Strategy:
- Test all public methods
- Test async behavior and error handling
- Test edge cases and boundary conditions
- Test recovery mechanisms
- Test integration with dependencies

CoordinatorName is responsible for [brief description].
Current coverage: X%. Target: >75%.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Any, Dict, List, Optional

from victor.agent.coordinators.coordinator_name import CoordinatorName
from victor.protocols import ICoordinatorProtocol


class TestCoordinatorNameInitialization:
    """Test suite for CoordinatorName initialization and setup."""

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator with all required dependencies."""
        orch = Mock()
        # Configure mock with required attributes
        orch.provider = Mock()
        orch.settings = Mock()
        return orch

    @pytest.fixture
    def coordinator(self, mock_orchestrator: Mock) -> CoordinatorName:
        """Create coordinator instance for testing."""
        return CoordinatorName(orchestrator=mock_orchestrator)

    def test_initialization_with_defaults(self, coordinator: CoordinatorName):
        """Test coordinator initializes with default values."""
        # Assert
        assert coordinator._orchestrator is not None
        assert coordinator._config is not None


class TestCoordinatorNamePublicMethods:
    """Test suite for CoordinatorName public API methods."""

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator."""
        orch = Mock()
        # Setup required mock attributes
        return orch

    @pytest.fixture
    def coordinator(self, mock_orchestrator: Mock) -> CoordinatorName:
        """Create coordinator instance."""
        return CoordinatorName(orchestrator=mock_orchestrator)

    @pytest.mark.asyncio
    async def test_public_method_success(self, coordinator: CoordinatorName):
        """Test public_method handles successful execution."""
        # Arrange
        input_data = {"key": "value"}

        # Act
        result = await coordinator.public_method(input_data)

        # Assert
        assert result is not None
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_public_method_with_error(self, coordinator: CoordinatorName):
        """Test public_method handles errors gracefully."""
        # Arrange
        input_data = {"invalid": "data"}

        # Act & Assert
        with pytest.raises(ValueError, match="Expected error message"):
            await coordinator.public_method(input_data)


class TestCoordinatorNameEdgeCases:
    """Test suite for CoordinatorName edge cases and boundary conditions."""

    @pytest.fixture
    def coordinator(self) -> CoordinatorName:
        """Create coordinator instance."""
        mock_orch = Mock()
        return CoordinatorName(orchestrator=mock_orch)

    @pytest.mark.asyncio
    async def test_handles_empty_input(self, coordinator: CoordinatorName):
        """Test coordinator handles empty input gracefully."""
        # Arrange
        empty_input = {}

        # Act
        result = await coordinator.process(empty_input)

        # Assert
        assert result is not None

    @pytest.mark.asyncio
    async def test_handles_large_input(self, coordinator: CoordinatorName):
        """Test coordinator handles large input efficiently."""
        # Arrange
        large_input = {"data": "x" * 1000000}

        # Act
        result = await coordinator.process(large_input)

        # Assert
        assert result is not None
```

### Test Class Organization

Organize test classes by functionality:

1. **Test Classes**: Group related tests using classes
2. **Test Methods**: Use descriptive method names following `test_{method}_{scenario}_{expected_result}` pattern
3. **Fixtures**: Use pytest fixtures for common setup

Example:

```python
class TestChatCoordinatorInitialization:
    """Tests for ChatCoordinator initialization."""

class TestChatCoordinatorChat:
    """Tests for ChatCoordinator.chat() method."""

class TestChatCoordinatorStreamChat:
    """Tests for ChatCoordinator.stream_chat() method."""

class TestChatCoordinatorErrorHandling:
    """Tests for ChatCoordinator error handling."""
```

## Mock Strategies by Coordinator Type

### Application Layer Coordinators

#### ChatCoordinator

**Dependencies to Mock:**
- `orchestrator.provider` - LLM provider
- `orchestrator.conversation` - Conversation state
- `orchestrator.tool_selector` - Tool selection
- `orchestrator._handle_tool_calls` - Tool execution
- `orchestrator.response_completer` - Response completion

**Mock Example:**

```python
@pytest.fixture
def mock_orchestrator(self) -> Mock:
    """Create mock orchestrator for ChatCoordinator."""
    orch = Mock()
    orch.provider = Mock()
    orch.provider.chat = AsyncMock(return_value=CompletionResponse(
        content="Test response",
        role="assistant",
        tool_calls=None,
        usage={"prompt_tokens": 10, "completion_tokens": 20}
    ))
    orch.provider.stream = self._create_stream_generator()
    orch.provider.supports_tools = Mock(return_value=True)
    orch.conversation = Mock()
    orch.conversation.ensure_system_prompt = Mock()
    orch.conversation.message_count = Mock(return_value=5)
    orch.tool_selector = Mock()
    orch.tool_selector.select_tools = AsyncMock(return_value=[])
    orch._handle_tool_calls = AsyncMock(return_value=[])
    orch.response_completer = Mock()
    orch.response_completer.ensure_response = AsyncMock(
        return_value=Mock(content="Fallback")
    )
    return orch

def _create_stream_generator(self):
    """Create a mock async stream generator."""
    async def stream_gen():
        chunks = [
            StreamChunk(content="Hello", role="assistant"),
            StreamChunk(content=" world", role="assistant"),
        ]
        for chunk in chunks:
            yield chunk
    return stream_gen
```

**Key Test Scenarios:**
- Non-streaming chat with no tools
- Streaming chat with tool calls
- Multi-turn conversation with tool execution
- Error recovery from provider failures
- Context overflow handling
- Token usage tracking
- Rate limit handling

#### ToolCoordinator

**Dependencies to Mock:**
- `tool_pipeline` - Tool execution pipeline
- `tool_registry` - Tool registration
- `tool_selector` - Tool selection (optional)
- `cache` - Tool selection cache (optional)

**Mock Example:**

```python
@pytest.fixture
def mock_pipeline(self) -> Mock:
    """Create mock ToolPipeline."""
    pipeline = Mock()
    pipeline.execute_tool_calls = AsyncMock()
    pipeline._execute_single_tool = AsyncMock(
        return_value=ToolResult(
            tool_name="read_file",
            success=True,
            output="File content"
        )
    )
    return pipeline

@pytest.fixture
def mock_registry(self) -> Mock:
    """Create mock ToolRegistry."""
    registry = Mock()
    registry.get_all_tool_names = Mock(
        return_value=["read_file", "write_file", "shell"]
    )
    registry.get_tool = Mock(return_value=Mock())
    return registry

@pytest.fixture
def coordinator(self, mock_pipeline: Mock, mock_registry: Mock) -> ToolCoordinator:
    """Create ToolCoordinator with mocked dependencies."""
    return ToolCoordinator(
        tool_pipeline=mock_pipeline,
        tool_registry=mock_registry,
    )
```

**Key Test Scenarios:**
- Tool validation and execution
- Budget enforcement
- Tool alias resolution
- Access control
- Batch execution
- Error handling and recovery
- Caching behavior

#### ContextCoordinator

**Dependencies to Mock:**
- `orchestrator.conversation` - Conversation state
- `orchestrator._context_compactor` - Context compaction
- `orchestrator.provider` - For token counting

**Mock Example:**

```python
@pytest.fixture
def mock_orchestrator(self) -> Mock:
    """Create mock orchestrator for ContextCoordinator."""
    orch = Mock()
    orch.conversation = Mock()
    orch.conversation.messages = [
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
    ]
    orch._context_compactor = Mock()
    orch._context_compactor.compact = AsyncMock(
        return_value=CompactionResult(
            compacted_context=[],
            tokens_saved=1000,
            messages_removed=2,
            strategy_used="truncation"
        )
    )
    return orch
```

**Key Test Scenarios:**
- Context budget checking
- Compaction strategy selection
- Token count estimation
- Message prioritization
- Semantic compaction
- Hybrid strategies

### Framework Layer Coordinators

#### YAMLWorkflowCoordinator

**Dependencies to Mock:**
- `workflow_provider` - YAML workflow loader
- `compiler` - Workflow compiler
- `executor` - Workflow executor

**Mock Example:**

```python
@pytest.fixture
def mock_provider(self) -> Mock:
    """Create mock workflow provider."""
    provider = Mock()
    provider.load_workflow = Mock(
        return_value={
            "nodes": [{"id": "start", "type": "agent"}],
            "edges": []
        }
    )
    return provider

@pytest.fixture
def mock_compiler(self) -> Mock:
    """Create mock workflow compiler."""
    compiler = Mock()
    compiler.compile = Mock(return_value=Mock())
    return compiler

@pytest.fixture
def coordinator(self, mock_provider: Mock, mock_compiler: Mock) -> YAMLWorkflowCoordinator:
    """Create YAMLWorkflowCoordinator."""
    return YAMLWorkflowCoordinator(
        provider=mock_provider,
        compiler=mock_compiler,
    )
```

**Key Test Scenarios:**
- YAML loading and validation
- Workflow compilation
- Execution with state management
- Error handling for invalid YAML
- Checkpoint integration
- Caching behavior

#### CacheCoordinator

**Dependencies to Mock:**
- `cache_backend` - Cache storage backend
- `serializer` - State serializer

**Mock Example:**

```python
@pytest.fixture
def mock_backend(self) -> Mock:
    """Create mock cache backend."""
    backend = Mock()
    backend.get = AsyncMock(return_value=None)
    backend.set = AsyncMock()
    backend.delete = AsyncMock()
    backend.exists = AsyncMock(return_value=False)
    return backend

@pytest.fixture
def coordinator(self, mock_backend: Mock) -> CacheCoordinator:
    """Create CacheCoordinator."""
    return CacheCoordinator(backend=mock_backend)
```

**Key Test Scenarios:**
- Cache hit/miss behavior
- TTL expiration
- Cache invalidation
- Serialization/deserialization
- Concurrent access
- Memory management

## Async Testing Patterns

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 8 min
**Last Updated:** February 08, 2026**
