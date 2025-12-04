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

"""Tests for ToolPipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.agent.tool_pipeline import (
    ToolPipeline,
    ToolPipelineConfig,
    ToolCallResult,
    PipelineExecutionResult,
)
from victor.agent.tool_executor import ToolExecutionResult


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry."""
    registry = MagicMock()
    registry.is_tool_enabled = MagicMock(return_value=True)
    return registry


@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor."""
    executor = MagicMock()
    executor.execute = AsyncMock(return_value=ToolExecutionResult(
        tool_name="test_tool",
        success=True,
        result={"output": "test result"},
        error=None,
    ))
    return executor


@pytest.fixture
def pipeline(mock_tool_registry, mock_tool_executor):
    """Create a tool pipeline for testing."""
    return ToolPipeline(
        tool_registry=mock_tool_registry,
        tool_executor=mock_tool_executor,
        config=ToolPipelineConfig(tool_budget=10),
    )


class TestToolPipelineConfig:
    """Tests for ToolPipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ToolPipelineConfig()
        assert config.tool_budget == 25
        assert config.enable_caching is True
        assert config.enable_analytics is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ToolPipelineConfig(tool_budget=50, enable_caching=False)
        assert config.tool_budget == 50
        assert config.enable_caching is False


class TestToolCallResult:
    """Tests for ToolCallResult dataclass."""

    def test_success_result(self):
        """Test successful tool call result."""
        result = ToolCallResult(
            tool_name="read_file",
            arguments={"path": "test.py"},
            success=True,
            result="file contents",
        )
        assert result.success is True
        assert result.skipped is False

    def test_skipped_result(self):
        """Test skipped tool call result."""
        result = ToolCallResult(
            tool_name="unknown_tool",
            arguments={},
            success=False,
            skipped=True,
            skip_reason="Unknown tool",
        )
        assert result.skipped is True
        assert result.skip_reason == "Unknown tool"


class TestToolPipeline:
    """Tests for ToolPipeline class."""

    def test_init(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.calls_used == 0
        assert pipeline.calls_remaining == 10

    def test_is_valid_tool_name(self, pipeline):
        """Test tool name validation."""
        # Valid names
        assert pipeline.is_valid_tool_name("read_file") is True
        assert pipeline.is_valid_tool_name("code_search") is True
        assert pipeline.is_valid_tool_name("git") is True

        # Invalid names
        assert pipeline.is_valid_tool_name("") is False
        assert pipeline.is_valid_tool_name(None) is False
        assert pipeline.is_valid_tool_name("Invalid-Name") is False
        assert pipeline.is_valid_tool_name("123_tool") is False
        assert pipeline.is_valid_tool_name("Tool") is False  # Must start lowercase

    def test_is_valid_tool_name_length_limit(self, pipeline):
        """Test tool name length limit."""
        long_name = "a" * 100
        assert pipeline.is_valid_tool_name(long_name) is False

    @pytest.mark.asyncio
    async def test_execute_single_tool_call(self, pipeline, mock_tool_executor):
        """Test executing a single tool call."""
        tool_calls = [{"name": "read_file", "arguments": {"path": "test.py"}}]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.total_calls == 1
        assert result.successful_calls == 1
        assert result.failed_calls == 0
        assert pipeline.calls_used == 1

    @pytest.mark.asyncio
    async def test_execute_multiple_tool_calls(self, pipeline, mock_tool_executor):
        """Test executing multiple tool calls."""
        tool_calls = [
            {"name": "read_file", "arguments": {"path": "test.py"}},
            {"name": "code_search", "arguments": {"query": "test"}},
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.total_calls == 2
        assert result.successful_calls == 2
        assert pipeline.calls_used == 2

    @pytest.mark.asyncio
    async def test_skip_invalid_tool_name(self, pipeline):
        """Test that invalid tool names are skipped."""
        tool_calls = [{"name": "Invalid-Tool", "arguments": {}}]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.skipped_calls == 1
        assert result.results[0].skipped is True
        assert "Invalid tool name" in result.results[0].skip_reason

    @pytest.mark.asyncio
    async def test_skip_unknown_tool(self, pipeline, mock_tool_registry):
        """Test that unknown tools are skipped."""
        mock_tool_registry.is_tool_enabled.return_value = False
        tool_calls = [{"name": "unknown_tool", "arguments": {}}]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.skipped_calls == 1
        assert "Unknown or disabled" in result.results[0].skip_reason

    @pytest.mark.asyncio
    async def test_budget_enforcement(self, pipeline, mock_tool_executor):
        """Test that tool budget is enforced."""
        pipeline.config.tool_budget = 2
        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
            {"name": "tool3", "arguments": {}},  # Should be skipped
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.budget_exhausted is True
        assert pipeline.calls_used == 2
        # Only 2 results because we break after budget exhausted
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_skip_repeated_failures(self, pipeline, mock_tool_executor):
        """Test that repeated failing calls are skipped."""
        # Make the tool fail
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="failing_tool",
            success=False,
            result=None,
            error="Test error",
        )

        tool_calls = [
            {"name": "failing_tool", "arguments": {"x": 1}},
            {"name": "failing_tool", "arguments": {"x": 1}},  # Same signature
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        # First call fails, second is skipped
        assert result.failed_calls == 1
        assert result.skipped_calls == 1
        assert "Repeated failing" in result.results[1].skip_reason

    @pytest.mark.asyncio
    async def test_argument_normalization(self, pipeline, mock_tool_executor):
        """Test that string arguments are normalized."""
        tool_calls = [{"name": "test_tool", "arguments": '{"path": "test.py"}'}]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        # Should succeed with normalized arguments
        assert result.successful_calls == 1
        call_args = mock_tool_executor.execute.call_args
        assert isinstance(call_args.kwargs["arguments"], dict)

    def test_reset(self, pipeline):
        """Test resetting pipeline state."""
        pipeline._calls_used = 5
        pipeline._executed_tools = ["tool1", "tool2"]
        pipeline._failed_signatures.add(("test", "{}"))

        pipeline.reset()

        assert pipeline.calls_used == 0
        assert len(pipeline.executed_tools) == 0
        assert len(pipeline._failed_signatures) == 0

    def test_get_analytics(self, pipeline):
        """Test getting analytics."""
        analytics = pipeline.get_analytics()

        assert "total_calls" in analytics
        assert "budget" in analytics
        assert "remaining" in analytics
        assert "tools" in analytics

    @pytest.mark.asyncio
    async def test_callbacks(self, pipeline, mock_tool_executor):
        """Test that callbacks are invoked."""
        start_called = []
        complete_called = []

        def on_start(name, args):
            start_called.append((name, args))

        def on_complete(result):
            complete_called.append(result)

        pipeline.on_tool_start = on_start
        pipeline.on_tool_complete = on_complete

        tool_calls = [{"name": "test_tool", "arguments": {"x": 1}}]
        await pipeline.execute_tool_calls(tool_calls, {})

        assert len(start_called) == 1
        assert len(complete_called) == 1
        assert start_called[0][0] == "test_tool"

    @pytest.mark.asyncio
    async def test_skip_missing_name(self, pipeline):
        """Test skipping tool call without name."""
        tool_calls = [{"arguments": {"x": 1}}]  # Missing name

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.skipped_calls == 1
        assert "missing name" in result.results[0].skip_reason

    @pytest.mark.asyncio
    async def test_skip_invalid_structure(self, pipeline):
        """Test skipping invalid tool call structure."""
        tool_calls = ["not a dict"]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.skipped_calls == 1
        assert "not a dict" in result.results[0].skip_reason

    def test_executed_tools_tracking(self, pipeline):
        """Test that executed tools are tracked."""
        assert pipeline.executed_tools == []

    def test_clear_failed_signatures(self, pipeline):
        """Test clearing failed signatures."""
        pipeline._failed_signatures.add(("test", "{}"))
        assert len(pipeline._failed_signatures) == 1

        pipeline.clear_failed_signatures()
        assert len(pipeline._failed_signatures) == 0
