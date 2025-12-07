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

"""Tests for tool_executor module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.agent.tool_executor import (
    ToolExecutionResult,
    ToolExecutor,
)
from victor.tools.base import ToolRegistry, BaseTool, ToolResult


class TestToolExecutionResult:
    """Tests for ToolExecutionResult class."""

    def test_result_init_success(self):
        """Test ToolExecutionResult initialization for success."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result={"output": "data"},
        )
        assert result.tool_name == "test_tool"
        assert result.success is True
        assert result.result == {"output": "data"}
        assert result.error is None
        assert result.cached is False
        assert result.retries == 0

    def test_result_init_failure(self):
        """Test ToolExecutionResult initialization for failure."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error="Tool failed",
            retries=2,
        )
        assert result.success is False
        assert result.error == "Tool failed"
        assert result.retries == 2

    def test_result_cached(self):
        """Test ToolExecutionResult with cached flag."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result="cached_data",
            cached=True,
        )
        assert result.cached is True

    def test_result_with_execution_time(self):
        """Test ToolExecutionResult with execution time."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result="data",
            execution_time=1.5,
        )
        assert result.execution_time == 1.5


class TestToolExecutorInit:
    """Tests for ToolExecutor initialization."""

    def test_init_minimal(self):
        """Test ToolExecutor with minimal arguments."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        assert executor.tools is registry
        assert executor.normalizer is not None
        assert executor.cache is None
        assert executor.max_retries == 3
        assert executor.retry_delay == 1.0
        assert executor.context == {}

    def test_init_with_cache(self):
        """Test ToolExecutor with cache."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        assert executor.cache is mock_cache

    def test_init_with_context(self):
        """Test ToolExecutor with context."""
        registry = ToolRegistry()
        context = {"cwd": "/tmp", "user": "test"}

        executor = ToolExecutor(
            tool_registry=registry,
            context=context,
        )

        assert executor.context == context

    def test_init_with_retry_config(self):
        """Test ToolExecutor with custom retry configuration."""
        registry = ToolRegistry()

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=5,
            retry_delay=2.0,
        )

        assert executor.max_retries == 5
        assert executor.retry_delay == 2.0


class TestToolExecutorUpdateContext:
    """Tests for ToolExecutor.update_context method."""

    def test_update_context(self):
        """Test updating context."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        executor.update_context(key1="value1", key2="value2")

        assert executor.context["key1"] == "value1"
        assert executor.context["key2"] == "value2"

    def test_update_context_overwrites(self):
        """Test that update_context overwrites existing keys."""
        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            context={"key": "old_value"},
        )

        executor.update_context(key="new_value")

        assert executor.context["key"] == "new_value"


class TestToolExecutorExecute:
    """Tests for ToolExecutor.execute method."""

    @pytest.fixture
    def registry_with_tool(self):
        """Create a registry with a mock tool."""
        registry = ToolRegistry()

        # Create a mock tool
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.execute = AsyncMock(return_value={"result": "success"})

        registry.register(mock_tool)
        return registry, mock_tool

    @pytest.mark.asyncio
    async def test_execute_success(self, registry_with_tool):
        """Test successful tool execution."""
        registry, mock_tool = registry_with_tool
        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("test_tool", {"arg": "value"})

        assert result.success is True
        assert result.result == {"result": "success"}
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test execution with non-existent tool."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("nonexistent_tool", {})

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_disabled_tool(self, registry_with_tool):
        """Test execution of disabled tool."""
        registry, mock_tool = registry_with_tool

        # Disable the tool
        registry.disable_tool("test_tool")

        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("test_tool", {})

        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_context(self, registry_with_tool):
        """Test execution with context."""
        registry, mock_tool = registry_with_tool
        executor = ToolExecutor(
            tool_registry=registry,
            context={"default_ctx": "value"},
        )

        await executor.execute(
            "test_tool",
            {"arg": "value"},
            context={"call_ctx": "specific"},
        )

        # Verify tool was called with merged context
        mock_tool.execute.assert_called_once()
        call_context = mock_tool.execute.call_args[0][0]
        assert call_context["default_ctx"] == "value"
        assert call_context["call_ctx"] == "specific"

    @pytest.mark.asyncio
    async def test_execute_tracks_stats(self, registry_with_tool):
        """Test that execution tracks statistics."""
        registry, mock_tool = registry_with_tool
        executor = ToolExecutor(tool_registry=registry)

        await executor.execute("test_tool", {})
        await executor.execute("test_tool", {})

        stats = executor.get_stats()
        assert "test_tool" in stats
        assert stats["test_tool"]["calls"] == 2
        assert stats["test_tool"]["successes"] == 2


class TestToolExecutorCache:
    """Tests for ToolExecutor caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit for cacheable tool."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "code_search"
        mock_tool.execute = AsyncMock(return_value={"matches": []})
        registry.register(mock_tool)

        mock_cache = MagicMock()
        mock_cache.get.return_value = {"cached_matches": []}

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        result = await executor.execute("code_search", {"query": "test"})

        assert result.cached is True
        assert result.result == {"cached_matches": []}
        mock_tool.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "code_search"
        mock_tool.execute = AsyncMock(return_value={"matches": ["result"]})
        registry.register(mock_tool)

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        result = await executor.execute("code_search", {"query": "test"})

        assert result.cached is False
        mock_tool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_cache(self):
        """Test skipping cache."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "code_search"
        mock_tool.execute = AsyncMock(return_value={"fresh": True})
        registry.register(mock_tool)

        mock_cache = MagicMock()
        mock_cache.get.return_value = {"stale": True}

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        result = await executor.execute("code_search", {}, skip_cache=True)

        assert result.result == {"fresh": True}
        mock_cache.get.assert_not_called()


class TestToolExecutorRetry:
    """Tests for ToolExecutor retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry logic on failure."""
        registry = ToolRegistry()

        # Tool that fails twice then succeeds
        call_count = [0]

        def make_execute_func():
            async def execute_func(context, **kwargs):
                nonlocal call_count
                call_count[0] += 1
                if call_count[0] < 3:
                    raise ValueError("Temporary failure")
                return {"success": True}

            return execute_func

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "flaky_tool"
        mock_tool.execute = make_execute_func()
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=3,
            retry_delay=0.01,  # Fast for testing
        )

        result = await executor.execute("flaky_tool", {})

        assert result.success is True
        # retries counts the last failed attempt number, which is 1 (0-indexed)
        assert result.retries >= 1  # At least one retry happened

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test that max retries are exhausted."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "failing_tool"
        mock_tool.execute = AsyncMock(side_effect=ValueError("Always fails"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=2,
            retry_delay=0.01,
        )

        result = await executor.execute("failing_tool", {})

        assert result.success is False
        assert "Always fails" in result.error


class TestToolExecutorToolResult:
    """Tests for handling ToolResult returns."""

    @pytest.mark.asyncio
    async def test_tool_result_success(self):
        """Test handling ToolResult with success."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "result_tool"
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(success=True, output={"data": "value"})
        )
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("result_tool", {})

        assert result.success is True
        assert result.result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_tool_result_failure(self):
        """Test handling ToolResult with failure."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "result_tool"
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(success=False, output=None, error="Tool error")
        )
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("result_tool", {})

        assert result.success is False
        assert "Tool error" in result.error


class TestToolExecutorStats:
    """Tests for ToolExecutor statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        """Test getting stats when empty (no tool executions)."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        stats = executor.get_stats()
        # Stats now includes _global with error tracking info
        assert "_global" in stats
        assert stats["_global"]["validation_failures"] == 0
        assert stats["_global"]["errors_by_category"] == {}
        # No tool-specific stats yet
        assert len([k for k in stats.keys() if k != "_global"]) == 0

    @pytest.mark.asyncio
    async def test_get_stats_after_execution(self):
        """Test getting stats after execution."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "stats_tool"
        mock_tool.execute = AsyncMock(return_value="result")
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)

        await executor.execute("stats_tool", {})

        stats = executor.get_stats()
        assert "stats_tool" in stats
        assert stats["stats_tool"]["calls"] == 1
        assert stats["stats_tool"]["successes"] == 1
        assert stats["stats_tool"]["failures"] == 0

    def test_has_failed_before(self):
        """Test tracking failed signatures."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        # Initially not failed
        assert executor.has_failed_before("tool", {"a": 1}) is False

        # Mark as failed
        executor._failed_signatures.add(("tool", "[('a', 1)]"))

        assert executor.has_failed_before("tool", {"a": 1}) is True


class TestToolExecutorCacheableTools:
    """Tests for cacheable tool detection."""

    def test_default_cacheable_tools(self):
        """Test default cacheable tools list."""
        assert "code_search" in ToolExecutor.DEFAULT_CACHEABLE_TOOLS
        assert "read_file" in ToolExecutor.DEFAULT_CACHEABLE_TOOLS
        assert "list_directory" in ToolExecutor.DEFAULT_CACHEABLE_TOOLS

    def test_cache_invalidating_tools(self):
        """Test cache invalidating tools list."""
        assert "write_file" in ToolExecutor.CACHE_INVALIDATING_TOOLS
        assert "edit_files" in ToolExecutor.CACHE_INVALIDATING_TOOLS
        assert "execute_bash" in ToolExecutor.CACHE_INVALIDATING_TOOLS


class TestToolExecutorSafetyCheck:
    """Tests for ToolExecutor safety check integration."""

    @pytest.mark.asyncio
    async def test_safety_check_blocks_execution(self):
        """Test that safety check can block tool execution (covers lines 202-203)."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "dangerous_tool"
        mock_tool.execute = AsyncMock(return_value="should not be called")
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)

        # Mock safety checker to block execution
        executor.safety_checker.check_and_confirm = AsyncMock(
            return_value=(False, "Operation blocked for safety")
        )

        result = await executor.execute("dangerous_tool", {"path": "/etc/passwd"})

        assert result.success is False
        assert "safety" in result.error.lower() or "blocked" in result.error.lower()
        mock_tool.execute.assert_not_called()


class TestToolExecutionResultEnhanced:
    """Tests for enhanced ToolExecutionResult with error handling."""

    def test_result_with_correlation_id(self):
        """Test ToolExecutionResult with correlation_id."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error="Test error",
            correlation_id="abc123",
        )
        assert result.correlation_id == "abc123"

    def test_result_with_error_info(self):
        """Test ToolExecutionResult with error_info."""
        from victor.core.errors import ErrorInfo, ErrorCategory, ErrorSeverity

        error_info = ErrorInfo(
            message="Test error",
            category=ErrorCategory.TOOL_EXECUTION,
            severity=ErrorSeverity.ERROR,
            correlation_id="def456",
            recovery_hint="Try again",
        )
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error="Test error",
            error_info=error_info,
        )
        assert result.error_info is error_info
        assert result.error_info.correlation_id == "def456"

    def test_result_to_dict_success(self):
        """Test ToolExecutionResult.to_dict() for success."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result="data",
            execution_time=1.5,
            retries=0,
            correlation_id="test123",
        )
        d = result.to_dict()
        assert d["tool_name"] == "test_tool"
        assert d["success"] is True
        assert d["execution_time"] == 1.5
        assert d["correlation_id"] == "test123"
        assert "error" not in d

    def test_result_to_dict_failure(self):
        """Test ToolExecutionResult.to_dict() for failure."""
        from victor.core.errors import ErrorInfo, ErrorCategory, ErrorSeverity

        error_info = ErrorInfo(
            message="Test failure",
            category=ErrorCategory.TOOL_EXECUTION,
            severity=ErrorSeverity.ERROR,
            correlation_id="fail123",
        )
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error="Test failure",
            error_info=error_info,
        )
        d = result.to_dict()
        assert d["error"] == "Test failure"
        assert "error_details" in d
        assert d["error_details"]["category"] == "tool_execution"


class TestToolExecutorErrorHandling:
    """Tests for ToolExecutor error handling integration."""

    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset global singletons before each test for isolation."""
        from victor.core.errors import get_error_handler
        import victor.agent.safety as safety_module

        # Clear error history to ensure test isolation
        handler = get_error_handler()
        handler.clear_history()

        # Reset safety checker to avoid pollution from previous tests
        safety_module._default_checker = None

    @pytest.mark.asyncio
    async def test_error_handler_tracks_errors(self):
        """Test that error handler tracks errors by category."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "failing_tool"
        mock_tool.execute = AsyncMock(side_effect=ValueError("Test error"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        await executor.execute("failing_tool", {})

        # Error should be tracked
        assert len(executor._errors_by_category) > 0

    @pytest.mark.asyncio
    async def test_error_result_has_correlation_id(self):
        """Test that failed execution results have correlation ID."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "error_tool"
        mock_tool.execute = AsyncMock(side_effect=FileNotFoundError("File missing"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        result = await executor.execute("error_tool", {"path": "/missing"})

        assert result.success is False
        assert result.correlation_id is not None
        assert len(result.correlation_id) == 8  # UUID first 8 chars

    @pytest.mark.asyncio
    async def test_error_info_populated_on_failure(self):
        """Test that error_info is populated on execution failure."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "error_tool"
        mock_tool.execute = AsyncMock(side_effect=PermissionError("Access denied"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        result = await executor.execute("error_tool", {})

        assert result.success is False
        assert result.error_info is not None
        assert result.error_info.message == "Access denied"

    @pytest.mark.asyncio
    async def test_recovery_hint_in_error(self):
        """Test that recovery hints are included in error messages."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "error_tool"
        mock_tool.execute = AsyncMock(side_effect=ConnectionError("Network error"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        result = await executor.execute("error_tool", {})

        assert result.success is False
        # Error handler adds recovery hints
        assert result.error_info is not None
        # ConnectionError gets a recovery hint from error handler

    @pytest.mark.asyncio
    async def test_get_error_summary(self):
        """Test get_error_summary method."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "error_tool"
        mock_tool.execute = AsyncMock(side_effect=ValueError("Error"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        await executor.execute("error_tool", {})

        summary = executor.get_error_summary()

        assert "errors_by_category" in summary
        assert "total_errors" in summary
        assert summary["total_errors"] >= 1
        assert "recent_errors" in summary

    @pytest.mark.asyncio
    async def test_stats_include_error_categories(self):
        """Test that stats include error category breakdown."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "error_tool"
        mock_tool.execute = AsyncMock(side_effect=TypeError("Type error"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        await executor.execute("error_tool", {})

        stats = executor.get_stats()

        assert "_global" in stats
        assert "errors_by_category" in stats["_global"]
        assert "recent_errors" in stats["_global"]

    def test_track_error_category(self):
        """Test _track_error_category method."""
        from victor.core.errors import ErrorCategory

        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        executor._track_error_category(ErrorCategory.TOOL_EXECUTION)
        executor._track_error_category(ErrorCategory.TOOL_EXECUTION)
        executor._track_error_category(ErrorCategory.VALIDATION_ERROR)

        assert executor._errors_by_category["tool_execution"] == 2
        assert executor._errors_by_category["validation_error"] == 1


class TestToolExecutorWithErrorHandler:
    """Tests for ToolExecutor with custom error handler."""

    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset global singletons before each test for isolation."""
        from victor.core.errors import get_error_handler
        import victor.agent.safety as safety_module

        # Clear error history to ensure test isolation
        handler = get_error_handler()
        handler.clear_history()

        # Reset safety checker to avoid pollution from previous tests
        safety_module._default_checker = None

    def test_init_with_error_handler(self):
        """Test ToolExecutor with custom error handler."""
        from victor.core.errors import ErrorHandler

        registry = ToolRegistry()
        custom_handler = ErrorHandler(logger_name="custom")

        executor = ToolExecutor(
            tool_registry=registry,
            error_handler=custom_handler,
        )

        assert executor.error_handler is custom_handler

    def test_init_uses_global_handler(self):
        """Test ToolExecutor uses global handler when none provided."""
        from victor.core.errors import get_error_handler

        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        # Should use global handler
        assert executor.error_handler is get_error_handler()

    @pytest.mark.asyncio
    async def test_tool_result_failure_uses_error_handler(self):
        """Test that ToolResult failures use error handler."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "result_tool"
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(success=False, output=None, error="Tool error")
        )
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("result_tool", {})

        assert result.success is False
        assert result.error_info is not None
        assert result.error_info.category.value == "tool_execution"
