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
    executor.execute = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result={"output": "test result"},
            error=None,
        )
    )
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
        assert config.tool_budget == 100
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
        """Test that repeated failing calls are skipped.

        Note: With batch deduplication enabled (default), identical tool calls
        in the same batch are deduplicated first, so the skip reason will be
        'Deduplicated' rather than 'Repeated failing'. The 'Repeated failing'
        behavior still applies to subsequent batches/iterations.
        """
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

        # First call fails, second is skipped (deduplicated in same batch)
        assert result.failed_calls == 1
        assert result.skipped_calls == 1
        # With batch deduplication enabled, duplicates in same batch are deduped first
        assert "Deduplicated" in result.results[1].skip_reason

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


class TestToolPipelineParallelExecution:
    """Tests for parallel tool execution."""

    @pytest.fixture
    def parallel_pipeline(self, mock_tool_registry, mock_tool_executor):
        """Create a pipeline with parallel execution enabled."""
        config = ToolPipelineConfig(
            tool_budget=20,
            enable_parallel_execution=True,
            max_concurrent_tools=5,
        )
        return ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            config=config,
        )

    def test_parallel_executor_property(self, parallel_pipeline):
        """Test that parallel executor is lazily initialized."""
        # First access creates it
        executor = parallel_pipeline.parallel_executor
        assert executor is not None

        # Second access returns same instance
        assert parallel_pipeline.parallel_executor is executor

    @pytest.mark.asyncio
    async def test_parallel_execution_single_tool(self, parallel_pipeline, mock_tool_executor):
        """Test parallel execution with single tool falls back to sequential."""
        tool_calls = [{"name": "test_tool", "arguments": {"x": 1}}]

        result = await parallel_pipeline.execute_tool_calls_parallel(tool_calls, {})

        # Single tool uses sequential execution
        assert result.total_calls == 1
        assert not result.parallel_execution_used

    @pytest.mark.asyncio
    async def test_parallel_execution_disabled(self, mock_tool_registry, mock_tool_executor):
        """Test parallel execution when disabled."""
        config = ToolPipelineConfig(
            tool_budget=20,
            enable_parallel_execution=False,
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            config=config,
        )

        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
        ]

        result = await pipeline.execute_tool_calls_parallel(tool_calls, {})

        # Falls back to sequential
        assert not result.parallel_execution_used

    @pytest.mark.asyncio
    async def test_parallel_skips_invalid_tools(self, parallel_pipeline, mock_tool_registry):
        """Test that parallel execution skips invalid tool names."""

        # Disable one tool
        def is_tool_enabled(name):
            return name != "disabled_tool"

        mock_tool_registry.is_tool_enabled.side_effect = is_tool_enabled

        tool_calls = [
            {"name": "test_tool", "arguments": {}},
            {"name": "Invalid-Name", "arguments": {}},
            {"name": "disabled_tool", "arguments": {}},
        ]

        result = await parallel_pipeline.execute_tool_calls_parallel(
            tool_calls, {}, force_parallel=True
        )

        # Should have skipped results for invalid tools
        assert result.skipped_calls >= 2

    @pytest.mark.asyncio
    async def test_parallel_skips_repeated_failures(self, parallel_pipeline, mock_tool_executor):
        """Test that parallel execution skips repeated failing calls."""
        # Add a failed signature using the actual signature format
        # Compute the signature the same way the pipeline does
        failing_sig = parallel_pipeline._get_call_signature("failing_tool", {"x": 1})
        parallel_pipeline._failed_signatures.add(failing_sig)

        tool_calls = [
            {"name": "test_tool", "arguments": {"a": 1}},
            {"name": "failing_tool", "arguments": {"x": 1}},  # Should be skipped
        ]

        result = await parallel_pipeline.execute_tool_calls_parallel(
            tool_calls, {}, force_parallel=True
        )

        # The failing_tool should be skipped
        skip_reasons = [r.skip_reason for r in result.results if r.skipped]
        assert any("Repeated failing" in (r or "") for r in skip_reasons)

    @pytest.mark.asyncio
    async def test_parallel_budget_enforcement(self, mock_tool_registry, mock_tool_executor):
        """Test that parallel execution enforces budget."""
        config = ToolPipelineConfig(
            tool_budget=2,
            enable_parallel_execution=True,
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            config=config,
        )

        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
            {"name": "tool3", "arguments": {}},
        ]

        result = await pipeline.execute_tool_calls_parallel(tool_calls, {}, force_parallel=True)

        # Should stop after budget exhausted
        assert result.budget_exhausted is True

    def test_parallel_progress_callback(self, parallel_pipeline):
        """Test parallel progress callback."""
        start_calls = []
        parallel_pipeline.on_tool_start = lambda name, args: start_calls.append(name)

        # Call the progress callback directly
        parallel_pipeline._parallel_progress_callback("test_tool", "started", True)

        assert "test_tool" in start_calls

    def test_parallel_progress_callback_not_started(self, parallel_pipeline):
        """Test parallel progress callback for non-started status."""
        start_calls = []
        parallel_pipeline.on_tool_start = lambda name, args: start_calls.append(name)

        # Call with status other than "started"
        parallel_pipeline._parallel_progress_callback("test_tool", "completed", True)

        # Should not call on_tool_start
        assert "test_tool" not in start_calls


class TestToolPipelineNormalization:
    """Tests for argument normalization."""

    @pytest.fixture
    def pipeline(self, mock_tool_registry, mock_tool_executor):
        """Create a tool pipeline for testing."""
        return ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
        )

    def test_normalize_string_json_arguments(self, pipeline):
        """Test normalizing JSON string arguments."""
        args, strategy = pipeline._normalize_arguments("test_tool", '{"path": "file.py"}')
        assert args == {"path": "file.py"}

    def test_normalize_string_python_literal(self, pipeline):
        """Test normalizing Python literal string arguments."""
        args, strategy = pipeline._normalize_arguments("test_tool", "{'path': 'file.py'}")
        assert args == {"path": "file.py"}

    def test_normalize_invalid_string(self, pipeline):
        """Test normalizing invalid string falls back to value wrapper."""
        args, strategy = pipeline._normalize_arguments("test_tool", "just a string")
        assert args == {"value": "just a string"}

    def test_normalize_none_arguments(self, pipeline):
        """Test normalizing None arguments."""
        args, strategy = pipeline._normalize_arguments("test_tool", None)
        assert args == {}

    def test_normalize_dict_arguments(self, pipeline):
        """Test that dict arguments are passed through normalizer."""
        args, strategy = pipeline._normalize_arguments("test_tool", {"x": 1})
        assert "x" in args

    def test_get_call_signature_json(self, pipeline):
        """Test generating call signature."""
        sig = pipeline._get_call_signature("test_tool", {"a": 1, "b": 2})
        # Signature is a string (either hex hash from native or tool:args from fallback)
        assert isinstance(sig, str)
        assert len(sig) > 0
        # Same args should produce same signature
        sig2 = pipeline._get_call_signature("test_tool", {"a": 1, "b": 2})
        assert sig == sig2

    def test_get_call_signature_non_serializable(self, pipeline):
        """Test generating call signature with non-serializable args."""

        class NonSerializable:
            pass

        sig = pipeline._get_call_signature("test_tool", {"obj": NonSerializable()})
        # Signature is a string even with non-serializable args (uses str() fallback)
        assert isinstance(sig, str)
        assert len(sig) > 0


class TestToolPipelineCodeCorrection:
    """Tests for code correction middleware integration."""

    @pytest.fixture
    def mock_correction_middleware(self):
        """Create a mock code correction middleware."""
        middleware = MagicMock()
        middleware.should_validate.return_value = True

        # Create a mock validation result
        validation_result = MagicMock()
        validation_result.valid = True
        validation_result.errors = []

        correction_result = MagicMock()
        correction_result.was_corrected = False
        correction_result.validation = validation_result

        middleware.validate_and_fix.return_value = correction_result
        middleware.apply_correction.return_value = {"path": "corrected.py"}

        return middleware

    @pytest.fixture
    def pipeline_with_correction(
        self, mock_tool_registry, mock_tool_executor, mock_correction_middleware
    ):
        """Create a pipeline with code correction enabled."""
        config = ToolPipelineConfig(
            tool_budget=20,
            enable_code_correction=True,
        )
        return ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            config=config,
            code_correction_middleware=mock_correction_middleware,
        )

    @pytest.mark.asyncio
    async def test_code_correction_applied(
        self, pipeline_with_correction, mock_correction_middleware, mock_tool_executor
    ):
        """Test that code correction is applied when middleware corrects code."""
        # Configure middleware to indicate correction was applied
        validation_result = MagicMock()
        validation_result.valid = True
        validation_result.errors = []

        correction_result = MagicMock()
        correction_result.was_corrected = True
        correction_result.validation = validation_result

        mock_correction_middleware.validate_and_fix.return_value = correction_result

        tool_calls = [{"name": "write_code", "arguments": {"code": "bad code"}}]
        result = await pipeline_with_correction.execute_tool_calls(tool_calls, {})

        assert result.successful_calls == 1
        # Verify apply_correction was called
        mock_correction_middleware.apply_correction.assert_called()

    @pytest.mark.asyncio
    async def test_code_correction_validation_errors(
        self, pipeline_with_correction, mock_correction_middleware, mock_tool_executor
    ):
        """Test that validation errors are collected."""
        # Configure middleware to report validation errors
        validation_result = MagicMock()
        validation_result.valid = False
        validation_result.errors = ["Syntax error", "Missing import"]

        correction_result = MagicMock()
        correction_result.was_corrected = False
        correction_result.validation = validation_result

        mock_correction_middleware.validate_and_fix.return_value = correction_result

        tool_calls = [{"name": "write_code", "arguments": {"code": "invalid"}}]
        result = await pipeline_with_correction.execute_tool_calls(tool_calls, {})

        # Tool still executes, but validation errors are logged
        assert result.total_calls == 1
        # Check that validation errors are tracked
        assert result.results[0].code_validation_errors is not None

    @pytest.mark.asyncio
    async def test_code_correction_middleware_exception(
        self, pipeline_with_correction, mock_correction_middleware, mock_tool_executor
    ):
        """Test that middleware exceptions are handled gracefully."""
        # Use ValueError since we now catch specific exception types
        mock_correction_middleware.validate_and_fix.side_effect = ValueError("Middleware error")

        tool_calls = [{"name": "write_code", "arguments": {"code": "test"}}]
        result = await pipeline_with_correction.execute_tool_calls(tool_calls, {})

        # Should still succeed - middleware errors are logged but don't block
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_code_correction_skipped_for_non_code_tools(
        self, mock_tool_registry, mock_tool_executor, mock_correction_middleware
    ):
        """Test that code correction is skipped for non-code tools."""
        mock_correction_middleware.should_validate.return_value = False

        config = ToolPipelineConfig(enable_code_correction=True)
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            config=config,
            code_correction_middleware=mock_correction_middleware,
        )

        tool_calls = [{"name": "read_file", "arguments": {"path": "test.py"}}]
        await pipeline.execute_tool_calls(tool_calls, {})

        # validate_and_fix should not be called
        mock_correction_middleware.validate_and_fix.assert_not_called()


class TestToolPipelineAnalytics:
    """Tests for analytics tracking."""

    @pytest.fixture
    def pipeline(self, mock_tool_registry, mock_tool_executor):
        """Create a pipeline with analytics enabled."""
        config = ToolPipelineConfig(enable_analytics=True)
        return ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_analytics_updated_on_success(self, pipeline, mock_tool_executor):
        """Test that analytics are updated on successful execution."""
        tool_calls = [{"name": "test_tool", "arguments": {}}]
        await pipeline.execute_tool_calls(tool_calls, {})

        analytics = pipeline.get_analytics()
        assert analytics["tools"]["test_tool"]["calls"] == 1
        assert analytics["tools"]["test_tool"]["successes"] == 1
        assert analytics["tools"]["test_tool"]["failures"] == 0

    @pytest.mark.asyncio
    async def test_analytics_updated_on_failure(self, pipeline, mock_tool_executor):
        """Test that analytics are updated on failed execution."""
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error="Test error",
        )

        tool_calls = [{"name": "test_tool", "arguments": {}}]
        await pipeline.execute_tool_calls(tool_calls, {})

        analytics = pipeline.get_analytics()
        assert analytics["tools"]["test_tool"]["failures"] == 1

    @pytest.mark.asyncio
    async def test_analytics_tracks_time(self, pipeline, mock_tool_executor):
        """Test that analytics track execution time."""
        tool_calls = [{"name": "test_tool", "arguments": {}}]
        await pipeline.execute_tool_calls(tool_calls, {})

        analytics = pipeline.get_analytics()
        assert "total_time_ms" in analytics["tools"]["test_tool"]

    @pytest.mark.asyncio
    async def test_analytics_disabled(self, mock_tool_registry, mock_tool_executor):
        """Test that analytics are not updated when disabled."""
        config = ToolPipelineConfig(enable_analytics=False)
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            config=config,
        )

        tool_calls = [{"name": "test_tool", "arguments": {}}]
        await pipeline.execute_tool_calls(tool_calls, {})

        analytics = pipeline.get_analytics()
        # Tools dict should be empty since analytics disabled
        assert len(analytics["tools"]) == 0


class TestToolPipelineCallbacks:
    """Tests for callback error handling."""

    @pytest.fixture
    def pipeline(self, mock_tool_registry, mock_tool_executor):
        """Create a tool pipeline for testing."""
        return ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
        )

    @pytest.mark.asyncio
    async def test_on_tool_start_exception_handled(self, pipeline, mock_tool_executor):
        """Test that exceptions in on_tool_start are handled."""

        def failing_start(name, args):
            raise Exception("Start callback error")

        pipeline.on_tool_start = failing_start

        tool_calls = [{"name": "test_tool", "arguments": {}}]
        # Should not raise - exception is logged but execution continues
        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_on_tool_complete_exception_handled(self, pipeline, mock_tool_executor):
        """Test that exceptions in on_tool_complete are handled."""

        def failing_complete(result):
            raise Exception("Complete callback error")

        pipeline.on_tool_complete = failing_complete

        tool_calls = [{"name": "test_tool", "arguments": {}}]
        # Should not raise - exception is logged but execution continues
        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.successful_calls == 1


class TestPipelineExecutionResult:
    """Tests for PipelineExecutionResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        from victor.agent.tool_pipeline import PipelineExecutionResult

        result = PipelineExecutionResult()

        assert result.results == []
        assert result.total_calls == 0
        assert result.successful_calls == 0
        assert result.failed_calls == 0
        assert result.skipped_calls == 0
        assert result.budget_exhausted is False
        assert result.parallel_execution_used is False
        assert result.parallel_speedup == 1.0
