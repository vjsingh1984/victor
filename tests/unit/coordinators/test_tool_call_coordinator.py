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

"""Unit tests for ToolCallCoordinator."""

import pytest
from unittest.mock import Mock, AsyncMock

from victor.agent.coordinators.tool_call_coordinator import (
    ToolCallCoordinator,
    create_tool_call_coordinator,
)
from victor.agent.coordinators.tool_call_protocol import (
    ToolCallContext,
    ToolCallCoordinatorConfig,
)
from victor.agent.tool_calling.base import ToolCall


@pytest.fixture
def mock_tool_executor():
    """Mock tool executor."""
    executor = Mock()
    executor.execute = AsyncMock()
    return executor


@pytest.fixture
def mock_tool_registry():
    """Mock tool registry."""
    registry = Mock()
    registry.is_available = Mock(return_value=True)
    return registry


@pytest.fixture
def mock_sanitizer():
    """Mock tool name sanitizer."""
    sanitizer = Mock()
    sanitizer.is_valid_tool_name = Mock(return_value=True)
    return sanitizer


@pytest.fixture
def coordinator_config():
    """Create coordinator config for testing."""
    return ToolCallCoordinatorConfig(
        max_retries=2,
        retry_delay=0.1,  # Short delay for tests
        retry_backoff_multiplier=2.0,
        parallel_execution=False,
        timeout_seconds=30.0,
        strict_validation=True,
    )


@pytest.fixture
def tool_call_coordinator(
    coordinator_config,
    mock_tool_executor,
    mock_tool_registry,
    mock_sanitizer,
):
    """Create ToolCallCoordinator for testing."""
    return ToolCallCoordinator(
        config=coordinator_config,
        tool_executor=mock_tool_executor,
        tool_registry=mock_tool_registry,
        sanitizer=mock_sanitizer,
    )


@pytest.fixture
def sample_tool_call():
    """Create sample tool call."""
    return ToolCall(
        name="read_file",
        arguments={"path": "test.py"},
        id="call_123",
    )


@pytest.fixture
def execution_context():
    """Create execution context."""
    return ToolCallContext(
        iteration=1,
        max_iterations=10,
        tool_budget=100,
        user_message="Read test.py",
        conversation_stage="initial",
    )


class TestToolCallCoordinator:
    """Test ToolCallCoordinator functionality."""

    def test_initialization(self, tool_call_coordinator, coordinator_config):
        """Test coordinator initialization."""
        assert tool_call_coordinator._config == coordinator_config
        assert tool_call_coordinator._tool_executor is not None
        assert tool_call_coordinator._tool_registry is not None

    def test_create_tool_call_coordinator_factory(
        self,
        mock_tool_executor,
        mock_tool_registry,
        mock_sanitizer,
    ):
        """Test factory function."""
        config = ToolCallCoordinatorConfig()
        coordinator = create_tool_call_coordinator(
            config=config,
            tool_executor=mock_tool_executor,
            tool_registry=mock_tool_registry,
            sanitizer=mock_sanitizer,
        )

        assert isinstance(coordinator, ToolCallCoordinator)
        assert coordinator._config == config

    @pytest.mark.asyncio
    async def test_execute_tool_with_retry_success(
        self,
        tool_call_coordinator,
        mock_tool_executor,
        execution_context,
    ):
        """Test successful tool execution with retry."""
        # Mock successful execution
        mock_result = Mock()
        mock_result.success = True
        mock_result.result = {"content": "print('hello')"}
        mock_tool_executor.execute.return_value = mock_result

        result = await tool_call_coordinator.execute_tool_with_retry(
            tool_name="read_file",
            arguments={"path": "test.py"},
            context=execution_context,
        )

        assert result.success is True
        assert result.output == {"content": "print('hello')"}
        assert result.error is None
        assert result.duration_ms >= 0
        mock_tool_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_with_retry_failure(
        self,
        tool_call_coordinator,
        mock_tool_executor,
        execution_context,
    ):
        """Test tool execution failure."""
        # Mock failed execution
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "File not found"
        mock_tool_executor.execute.return_value = mock_result

        result = await tool_call_coordinator.execute_tool_with_retry(
            tool_name="read_file",
            arguments={"path": "nonexistent.py"},
            context=execution_context,
        )

        assert result.success is False
        assert result.error == "File not found"
        assert result.output is None

    @pytest.mark.asyncio
    async def test_execute_tool_with_retry_transient_error(
        self,
        tool_call_coordinator,
        mock_tool_executor,
        execution_context,
    ):
        """Test tool execution with transient error and retry."""
        # Mock transient error then success
        mock_error = Mock()
        mock_error.success = False
        mock_error.error = "Connection timeout"

        mock_success = Mock()
        mock_success.success = True
        mock_success.result = {"content": "data"}

        mock_tool_executor.execute.side_effect = [mock_error, mock_success]

        result = await tool_call_coordinator.execute_tool_with_retry(
            tool_name="read_file",
            arguments={"path": "test.py"},
            context=execution_context,
        )

        assert result.success is True
        assert mock_tool_executor.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_tool_with_non_retryable_error(
        self,
        tool_call_coordinator,
        mock_tool_executor,
        execution_context,
    ):
        """Test tool execution with non-retryable error."""
        # Mock non-retryable error
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Invalid arguments: Missing required parameter 'path'"
        mock_tool_executor.execute.return_value = mock_result

        result = await tool_call_coordinator.execute_tool_with_retry(
            tool_name="read_file",
            arguments={},
            context=execution_context,
        )

        assert result.success is False
        assert "Invalid arguments" in result.error
        # Should not retry for validation errors
        mock_tool_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_tool_calls_empty(
        self,
        tool_call_coordinator,
        execution_context,
    ):
        """Test handling empty tool calls list."""
        results = await tool_call_coordinator.handle_tool_calls(
            tool_calls=[],
            context=execution_context,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_handle_tool_calls_success(
        self,
        tool_call_coordinator,
        mock_tool_executor,
        sample_tool_call,
        execution_context,
    ):
        """Test successful tool call handling."""
        # Mock successful execution
        mock_result = Mock()
        mock_result.success = True
        mock_result.result = {"content": "data"}
        mock_tool_executor.execute.return_value = mock_result

        results = await tool_call_coordinator.handle_tool_calls(
            tool_calls=[sample_tool_call],
            context=execution_context,
        )

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].tool_name == "read_file"

    @pytest.mark.asyncio
    async def test_handle_tool_calls_validation_error(
        self,
        tool_call_coordinator,
        mock_sanitizer,
        execution_context,
    ):
        """Test tool call handling with validation error."""
        # Mock invalid tool name
        mock_sanitizer.is_valid_tool_name.return_value = False

        tool_call = ToolCall(
            name="invalid_tool_xyz",
            arguments={},
        )

        results = await tool_call_coordinator.handle_tool_calls(
            tool_calls=[tool_call],
            context=execution_context,
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "Invalid tool name" in results[0].error

    @pytest.mark.asyncio
    async def test_handle_tool_calls_missing_name(
        self,
        tool_call_coordinator,
        execution_context,
    ):
        """Test tool call handling with missing name."""
        tool_call = ToolCall(
            name="",
            arguments={},
        )

        results = await tool_call_coordinator.handle_tool_calls(
            tool_calls=[tool_call],
            context=execution_context,
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "missing name" in results[0].error

    @pytest.mark.asyncio
    async def test_handle_tool_calls_budget_exhausted(
        self,
        tool_call_coordinator,
        mock_tool_executor,
        sample_tool_call,
    ):
        """Test tool call handling with exhausted budget."""
        # Create context with zero budget
        context = ToolCallContext(
            iteration=1,
            max_iterations=10,
            tool_budget=0,
            user_message="Read test.py",
        )

        results = await tool_call_coordinator.handle_tool_calls(
            tool_calls=[sample_tool_call],
            context=context,
        )

        # Tool should not execute when budget is exhausted
        assert len(results) == 1
        assert "budget" in results[0].error.lower()
        mock_tool_executor.execute.assert_not_called()

    def test_parse_tool_calls_dict_format(
        self,
        tool_call_coordinator,
    ):
        """Test parsing tool calls in dict format."""
        raw_calls = [
            {
                "name": "read_file",
                "arguments": {"path": "test.py"},
                "id": "call_123",
            },
            {
                "name": "write_file",
                "arguments": {"path": "out.py", "content": "data"},
                "id": "call_456",
            },
        ]

        parsed = tool_call_coordinator.parse_tool_calls(raw_calls)

        assert len(parsed) == 2
        assert parsed[0].name == "read_file"
        assert parsed[0].arguments == {"path": "test.py"}
        assert parsed[1].name == "write_file"
        assert parsed[1].arguments == {"path": "out.py", "content": "data"}

    def test_parse_tool_calls_json_string_arguments(
        self,
        tool_call_coordinator,
    ):
        """Test parsing tool calls with JSON string arguments."""
        raw_calls = [
            {
                "name": "read_file",
                "arguments": '{"path": "test.py"}',
            }
        ]

        parsed = tool_call_coordinator.parse_tool_calls(raw_calls)

        assert len(parsed) == 1
        assert parsed[0].name == "read_file"
        assert parsed[0].arguments == {"path": "test.py"}

    def test_parse_tool_calls_missing_arguments(
        self,
        tool_call_coordinator,
    ):
        """Test parsing tool calls with missing arguments."""
        raw_calls = [
            {
                "name": "read_file",
            }
        ]

        parsed = tool_call_coordinator.parse_tool_calls(raw_calls)

        assert len(parsed) == 1
        assert parsed[0].name == "read_file"
        assert parsed[0].arguments == {}

    def test_parse_tool_calls_invalid_format(
        self,
        tool_call_coordinator,
    ):
        """Test parsing tool calls with invalid format."""
        raw_calls = [
            "not_a_dict",
            {"name": "valid_tool"},
        ]

        parsed = tool_call_coordinator.parse_tool_calls(raw_calls)

        # Should skip invalid entries
        assert len(parsed) == 1
        assert parsed[0].name == "valid_tool"

    def test_validate_tool_calls_success(
        self,
        tool_call_coordinator,
        sample_tool_call,
        execution_context,
    ):
        """Test tool call validation success."""
        errors = tool_call_coordinator.validate_tool_calls(
            tool_calls=[sample_tool_call],
            context=execution_context,
        )

        assert errors == []

    def test_validate_tool_calls_missing_name(
        self,
        tool_call_coordinator,
        execution_context,
    ):
        """Test tool call validation with missing name."""
        tool_call = ToolCall(name="", arguments={})

        errors = tool_call_coordinator.validate_tool_calls(
            tool_calls=[tool_call],
            context=execution_context,
        )

        assert len(errors) > 0
        assert any("missing name" in e.lower() for e in errors)

    def test_validate_tool_calls_invalid_name(
        self,
        tool_call_coordinator,
        mock_sanitizer,
        execution_context,
    ):
        """Test tool call validation with invalid name."""
        mock_sanitizer.is_valid_tool_name.return_value = False

        tool_call = ToolCall(name="invalid$tool", arguments={})

        errors = tool_call_coordinator.validate_tool_calls(
            tool_calls=[tool_call],
            context=execution_context,
        )

        assert len(errors) > 0
        assert any("invalid tool name" in e.lower() for e in errors)

    def test_format_tool_output_dict(
        self,
        tool_call_coordinator,
    ):
        """Test formatting tool output with dict result."""
        output = {"content": "print('hello')", "lines": 1}
        formatted = tool_call_coordinator.format_tool_output(
            tool_name="read_file",
            arguments={"path": "test.py"},
            output=output,
        )

        assert "TOOL_OUTPUT: read_file" in formatted
        assert "ARGUMENTS:" in formatted
        assert "RESULT:" in formatted
        assert "END_TOOL_OUTPUT" in formatted
        assert '"content": "print(\'hello\')"' in formatted

    def test_format_tool_output_string(
        self,
        tool_call_coordinator,
    ):
        """Test formatting tool output with string result."""
        output = "File content here"
        formatted = tool_call_coordinator.format_tool_output(
            tool_name="read_file",
            arguments={"path": "test.py"},
            output=output,
        )

        assert "TOOL_OUTPUT: read_file" in formatted
        assert "File content here" in formatted

    def test_format_tool_output_truncation(
        self,
        tool_call_coordinator,
    ):
        """Test that very long outputs are truncated."""
        long_output = "x" * 15000
        formatted = tool_call_coordinator.format_tool_output(
            tool_name="read_file",
            arguments={"path": "large.py"},
            output=long_output,
        )

        assert "[truncated]" in formatted
        assert len(formatted) < len(long_output)
