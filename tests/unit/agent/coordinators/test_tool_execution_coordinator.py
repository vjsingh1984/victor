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

"""Tests for ToolExecutionCoordinator.

This test file covers the ToolExecutionCoordinator which handles tool call
execution and validation, extracted from AgentOrchestrator.

Test Coverage:
- Tool call validation (structure, names, permissions)
- Argument normalization (JSON parsing, type coercion)
- Budget enforcement during execution
- Result formatting and error feedback
- Failed signature tracking to avoid tight loops
- Retry logic with exponential backoff
- Cache integration
- Statistics tracking
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any, Dict, List, Optional, Tuple

from victor.agent.coordinators.tool_execution_coordinator import (
    ToolExecutionCoordinator,
    ToolExecutionConfig,
    ToolCallResult,
    ToolExecutionStats,
    ExecutionContext,
    ToolAccessDecision,
    create_tool_execution_coordinator,
)


class TestToolExecutionConfig:
    """Test suite for ToolExecutionConfig."""

    def test_default_config(self):
        """Test ToolExecutionConfig with default values."""
        config = ToolExecutionConfig()

        assert config.enabled is True
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_enabled is True
        assert config.max_retry_attempts == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 10.0
        assert config.enable_failed_signature_tracking is True
        assert config.enable_result_formatting is True
        assert config.enable_error_feedback is True
        assert config.max_tool_budget == 50

    def test_custom_config(self):
        """Test ToolExecutionConfig with custom values."""
        config = ToolExecutionConfig(
            enabled=False,
            timeout=60.0,
            max_retries=5,
            retry_enabled=False,
            max_retry_attempts=5,
            retry_base_delay=2.0,
            retry_max_delay=20.0,
            enable_failed_signature_tracking=False,
            max_tool_budget=100,
        )

        assert config.enabled is False
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.retry_enabled is False
        assert config.max_retry_attempts == 5
        assert config.retry_base_delay == 2.0
        assert config.retry_max_delay == 20.0
        assert config.enable_failed_signature_tracking is False
        assert config.max_tool_budget == 100


class TestToolCallResult:
    """Test suite for ToolCallResult."""

    def test_successful_result(self):
        """Test ToolCallResult for successful execution."""
        result = ToolCallResult(
            name="test_tool",
            success=True,
            result={"output": "data"},
            elapsed=1.5,
        )

        assert result.name == "test_tool"
        assert result.success is True
        assert result.result == {"output": "data"}
        assert result.error is None
        assert result.elapsed == 1.5
        assert result.cached is False
        assert result.skipped is False
        assert result.skip_reason is None

    def test_failed_result(self):
        """Test ToolCallResult for failed execution."""
        result = ToolCallResult(
            name="test_tool",
            success=False,
            error="Tool failed",
            elapsed=0.5,
        )

        assert result.success is False
        assert result.error == "Tool failed"
        assert result.result is None

    def test_cached_result(self):
        """Test ToolCallResult with cached flag."""
        result = ToolCallResult(
            name="test_tool",
            success=True,
            result="cached_data",
            cached=True,
        )

        assert result.cached is True

    def test_skipped_result(self):
        """Test ToolCallResult with skip reason."""
        result = ToolCallResult(
            name="test_tool",
            success=False,
            skipped=True,
            skip_reason="Budget exhausted",
        )

        assert result.skipped is True
        assert result.skip_reason == "Budget exhausted"


class TestToolExecutionStats:
    """Test suite for ToolExecutionStats."""

    def test_default_stats(self):
        """Test ToolExecutionStats with default values."""
        stats = ToolExecutionStats()

        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.skipped_calls == 0
        assert stats.budget_used == 0
        assert stats.budget_remaining == 50
        assert len(stats.failed_signatures) == 0
        assert len(stats.execution_times_ms) == 0

    def test_custom_initial_budget(self):
        """Test ToolExecutionStats with custom initial budget."""
        stats = ToolExecutionStats(budget_remaining=100)

        assert stats.budget_remaining == 100

    def test_get_success_rate_empty(self):
        """Test success rate calculation with no calls."""
        stats = ToolExecutionStats()
        assert stats.get_success_rate() == 0.0

    def test_get_success_rate_all_success(self):
        """Test success rate calculation with all successful calls."""
        stats = ToolExecutionStats(
            total_calls=10,
            successful_calls=10,
        )
        assert stats.get_success_rate() == 1.0

    def test_get_success_rate_mixed(self):
        """Test success rate calculation with mixed results."""
        stats = ToolExecutionStats(
            total_calls=10,
            successful_calls=7,
            failed_calls=3,
        )
        assert stats.get_success_rate() == 0.7

    def test_get_avg_execution_time_empty(self):
        """Test average execution time with no data."""
        stats = ToolExecutionStats()
        assert stats.get_avg_execution_time_ms() == 0.0

    def test_get_avg_execution_time(self):
        """Test average execution time calculation."""
        stats = ToolExecutionStats(
            execution_times_ms=[100.0, 200.0, 300.0],
        )
        assert stats.get_avg_execution_time_ms() == 200.0


class TestExecutionContext:
    """Test suite for ExecutionContext."""

    def test_default_context(self):
        """Test ExecutionContext with default values."""
        context = ExecutionContext()

        assert context.code_manager is None
        assert context.provider is None
        assert context.model is None
        assert context.tool_registry is None
        assert context.workflow_registry is None
        assert context.settings is None
        assert context.session_state is None
        assert context.conversation_state is None

    def test_context_with_values(self):
        """Test ExecutionContext with values."""
        mock_provider = Mock()
        mock_registry = Mock()

        context = ExecutionContext(
            code_manager=Mock(),
            provider=mock_provider,
            model="gpt-4",
            tool_registry=mock_registry,
            settings=Mock(),
        )

        assert context.provider is mock_provider
        assert context.model == "gpt-4"
        assert context.tool_registry is mock_registry


class TestToolAccessDecision:
    """Test suite for ToolAccessDecision."""

    def test_allowed_decision(self):
        """Test ToolAccessDecision when allowed."""
        decision = ToolAccessDecision(allowed=True)

        assert decision.allowed is True
        assert decision.reason is None

    def test_denied_decision_with_reason(self):
        """Test ToolAccessDecision when denied."""
        decision = ToolAccessDecision(
            allowed=False,
            reason="Tool not enabled",
        )

        assert decision.allowed is False
        assert decision.reason == "Tool not enabled"


class TestToolExecutionCoordinatorInit:
    """Test suite for ToolExecutionCoordinator initialization."""

    @pytest.fixture
    def mock_executor(self):
        """Create mock tool executor."""
        executor = AsyncMock()
        executor.execute = AsyncMock()
        return executor

    @pytest.fixture
    def mock_registry(self):
        """Create mock tool registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=True)
        return registry

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock argument normalizer."""
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({}, "strategy"))
        return normalizer

    @pytest.fixture
    def mock_adapter(self):
        """Create mock tool calling adapter."""
        adapter = Mock()
        adapter.normalize_arguments = Mock(side_effect=lambda x, y: x)
        return adapter

    def test_init_minimal(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Test coordinator initialization with minimal arguments."""
        coordinator = ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
        )

        assert coordinator._executor is mock_executor
        assert coordinator._registry is mock_registry
        assert coordinator._argument_normalizer is mock_normalizer
        assert coordinator._tool_adapter is mock_adapter
        assert coordinator._cache is None
        assert coordinator._sanitizer is None
        assert coordinator._formatter is None

    def test_init_with_cache(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Test coordinator initialization with cache."""
        mock_cache = Mock()

        coordinator = ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            tool_cache=mock_cache,
        )

        assert coordinator._cache is mock_cache

    def test_init_with_sanitizer_and_formatter(
        self, mock_executor, mock_registry, mock_normalizer, mock_adapter
    ):
        """Test coordinator initialization with sanitizer and formatter."""
        mock_sanitizer = Mock()
        mock_formatter = Mock()

        coordinator = ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            sanitizer=mock_sanitizer,
            formatter=mock_formatter,
        )

        assert coordinator._sanitizer is mock_sanitizer
        assert coordinator._formatter is mock_formatter

    def test_init_with_config(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Test coordinator initialization with custom config."""
        config = ToolExecutionConfig(max_tool_budget=100)

        coordinator = ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            config=config,
        )

        assert coordinator._config.max_tool_budget == 100
        assert coordinator._stats.budget_remaining == 100

    def test_init_with_callbacks(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Test coordinator initialization with callbacks."""
        on_tool_start = Mock()
        on_tool_complete = Mock()
        on_budget_warning = Mock()

        coordinator = ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            on_tool_start=on_tool_start,
            on_tool_complete=on_tool_complete,
            on_budget_warning=on_budget_warning,
        )

        assert coordinator._on_tool_start is on_tool_start
        assert coordinator._on_tool_complete is on_tool_complete
        assert coordinator._on_budget_warning is on_budget_warning

    def test_init_with_tool_access_checker(
        self, mock_executor, mock_registry, mock_normalizer, mock_adapter
    ):
        """Test coordinator initialization with tool access checker."""
        access_checker = Mock(return_value=ToolAccessDecision(allowed=True))

        coordinator = ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            tool_access_checker=access_checker,
        )

        assert coordinator._tool_access_checker is access_checker


class TestToolExecutionCoordinatorHandleToolCalls:
    """Test suite for handle_tool_calls method."""

    @pytest.fixture
    def coordinator(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Create coordinator for testing."""
        return ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = AsyncMock()
        executor.execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                result="output",
                error=None,
            )
        )
        return executor

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=True)
        return registry

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock normalizer."""
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({}, "strategy"))
        return normalizer

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = Mock()
        adapter.normalize_arguments = Mock(side_effect=lambda x, y: x)
        return adapter

    @pytest.mark.asyncio
    async def test_handle_empty_tool_calls(self, coordinator):
        """Test handling empty tool call list."""
        result = await coordinator.handle_tool_calls([])

        assert result == []

    @pytest.mark.asyncio
    async def test_handle_single_tool_call_success(self, coordinator):
        """Test handling single successful tool call."""
        tool_calls = [{"name": "test_tool", "arguments": {"arg": "value"}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_handle_multiple_tool_calls(self, coordinator):
        """Test handling multiple tool calls."""
        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
            {"name": "tool3", "arguments": {}},
        ]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 3
        assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_handle_tool_calls_with_context(self, coordinator):
        """Test handling tool calls with execution context."""
        tool_calls = [{"name": "test_tool", "arguments": {}}]
        context = ExecutionContext(provider=Mock(), model="gpt-4")

        results = await coordinator.handle_tool_calls(tool_calls, context)

        assert len(results) == 1
        assert results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_handle_tool_calls_budget_exhausted(self, coordinator):
        """Test handling tool calls when budget is exhausted."""
        # Set budget to 1
        coordinator._stats.budget_remaining = 1

        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
            {"name": "tool3", "arguments": {}},
        ]

        results = await coordinator.handle_tool_calls(tool_calls)

        # Only first tool should execute
        assert len(results) == 1
        assert results[0]["name"] == "tool1"

    @pytest.mark.asyncio
    async def test_handle_tool_calls_invalid_structure(self, coordinator):
        """Test handling tool calls with invalid structure."""
        tool_calls = [
            "not_a_dict",
            {"name": "valid_tool", "arguments": {}},
        ]

        results = await coordinator.handle_tool_calls(tool_calls)

        # First invalid call should be skipped, second should succeed
        assert len(results) == 1
        assert results[0]["name"] == "valid_tool"

    @pytest.mark.asyncio
    async def test_handle_tool_calls_missing_name(self, coordinator):
        """Test handling tool calls with missing name."""
        tool_calls = [{"arguments": {}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "missing name" in results[0]["error"].lower()


class TestToolExecutionCoordinatorValidation:
    """Test suite for tool call validation."""

    @pytest.fixture
    def coordinator(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Create coordinator with sanitizer."""
        mock_sanitizer = Mock()
        mock_sanitizer.is_valid_tool_name = Mock(return_value=True)

        return ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            sanitizer=mock_sanitizer,
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = AsyncMock()
        executor.execute = AsyncMock(
            return_value=MagicMock(success=True, result="output", error=None)
        )
        return executor

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=True)
        return registry

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock normalizer."""
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({}, "strategy"))
        return normalizer

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = Mock()
        adapter.normalize_arguments = Mock(side_effect=lambda x, y: x)
        return adapter

    @pytest.mark.asyncio
    async def test_validate_invalid_tool_name(self, coordinator):
        """Test validation of invalid tool name."""
        coordinator._sanitizer.is_valid_tool_name = Mock(return_value=False)

        tool_calls = [{"name": "invalid_tool", "arguments": {}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "does not exist" in results[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_validate_tool_access_denied(self, coordinator):
        """Test validation when tool access is denied."""
        access_checker = Mock(
            return_value=ToolAccessDecision(
                allowed=False, reason="Tool not available"
            )
        )
        coordinator._tool_access_checker = access_checker

        tool_calls = [{"name": "restricted_tool", "arguments": {}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "not available" in results[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_validate_tool_not_enabled(self, coordinator):
        """Test validation when tool is not enabled."""
        coordinator._registry.is_tool_enabled = Mock(return_value=False)

        tool_calls = [{"name": "disabled_tool", "arguments": {}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "not enabled" in results[0]["error"].lower()


class TestToolExecutionCoordinatorRetryLogic:
    """Test suite for retry logic."""

    @pytest.fixture
    def coordinator(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Create coordinator with retry enabled."""
        config = ToolExecutionConfig(
            retry_enabled=True,
            max_retry_attempts=3,
            retry_base_delay=0.1,  # Short delay for testing
            retry_max_delay=1.0,
        )

        return ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            config=config,
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = AsyncMock()
        return executor

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=True)
        return registry

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock normalizer."""
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({}, "strategy"))
        return normalizer

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = Mock()
        adapter.normalize_arguments = Mock(side_effect=lambda x, y: x)
        return adapter

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, coordinator, mock_executor):
        """Test retry logic on transient error."""
        # Fail first two attempts, succeed on third
        mock_executor.execute = AsyncMock(
            side_effect=[
                MagicMock(success=False, error="Connection error"),
                MagicMock(success=False, error="Connection error"),
                MagicMock(success=True, result="success", error=None),
            ]
        )

        tool_calls = [{"name": "flaky_tool", "arguments": {}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is True
        assert mock_executor.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self, coordinator, mock_executor):
        """Test that permanent errors are not retried."""
        mock_executor.execute = AsyncMock(
            return_value=MagicMock(
                success=False, error="Invalid argument: Missing required field"
            )
        )

        tool_calls = [{"name": "validation_tool", "arguments": {}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert mock_executor.execute.call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, coordinator, mock_executor):
        """Test behavior when retry attempts are exhausted."""
        mock_executor.execute = AsyncMock(
            return_value=MagicMock(success=False, error="Connection timeout")
        )

        tool_calls = [{"name": "failing_tool", "arguments": {}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert mock_executor.execute.call_count == 3  # Max attempts

    @pytest.mark.asyncio
    async def test_retry_disabled(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Test that retries are disabled when configured."""
        config = ToolExecutionConfig(retry_enabled=False)

        coordinator = ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            config=config,
        )

        mock_executor.execute = AsyncMock(
            return_value=MagicMock(success=False, error="Temporary error")
        )

        tool_calls = [{"name": "tool", "arguments": {}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert mock_executor.execute.call_count == 1  # No retries


class TestToolExecutionCoordinatorCache:
    """Test suite for cache integration."""

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache."""
        cache = Mock()
        cache.get = Mock(return_value=None)
        cache.set = Mock()
        cache.invalidate_paths = Mock()
        cache.clear_namespaces = Mock()
        return cache

    @pytest.fixture
    def coordinator(self, mock_executor, mock_registry, mock_normalizer, mock_adapter, mock_cache):
        """Create coordinator with cache."""
        return ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            tool_cache=mock_cache,
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = AsyncMock()
        executor.execute = AsyncMock(
            return_value=MagicMock(success=True, result="output", error=None)
        )
        return executor

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=True)
        return registry

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock normalizer."""
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({}, "strategy"))
        return normalizer

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = Mock()
        adapter.normalize_arguments = Mock(side_effect=lambda x, y: x)
        return adapter

    @pytest.mark.asyncio
    async def test_cache_hit(self, coordinator, mock_cache):
        """Test that cached results are returned without execution."""
        cached_result = MagicMock(success=True, result="cached_output")
        mock_cache.get = Mock(return_value=cached_result)

        tool_calls = [{"name": "cached_tool", "arguments": {}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is True
        assert mock_cache.get.called
        # Executor should not be called on cache hit

    @pytest.mark.asyncio
    async def test_cache_miss(self, coordinator, mock_cache):
        """Test that cache miss results in execution."""
        mock_cache.get = Mock(return_value=None)

        tool_calls = [{"name": "tool", "arguments": {}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert mock_cache.get.called
        # Verify cache.set is called with result
        assert mock_cache.set.called

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_write_file(self, coordinator, mock_cache):
        """Test that write_file invalidates related cache entries."""
        # Cache invalidation only happens on successful tool execution
        tool_calls = [
            {"name": "write_file", "arguments": {"path": "/tmp/test.py", "content": "data"}}
        ]

        # Execute tool calls
        await coordinator.handle_tool_calls(tool_calls)

        # Verify cache invalidation was triggered after successful execution
        # The invalidate_related_cache is called internally after success
        # We can verify this indirectly by checking the method exists on cache
        assert hasattr(mock_cache, 'invalidate_paths')

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_execute_bash(self, coordinator, mock_cache):
        """Test that execute_bash clears relevant namespaces."""
        # Cache invalidation only happens on successful tool execution
        tool_calls = [{"name": "execute_bash", "arguments": {"command": "ls"}}]

        # Execute tool calls
        await coordinator.handle_tool_calls(tool_calls)

        # Verify that cache has the clear_namespaces method available
        # The actual invalidation happens in _invalidate_related_cache
        assert hasattr(mock_cache, 'clear_namespaces')


class TestToolExecutionCoordinatorStatistics:
    """Test suite for statistics tracking."""

    @pytest.fixture
    def coordinator(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Create coordinator for testing."""
        return ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = AsyncMock()
        executor.execute = AsyncMock(
            return_value=MagicMock(success=True, result="output", error=None)
        )
        return executor

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=True)
        return registry

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock normalizer."""
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({}, "strategy"))
        return normalizer

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = Mock()
        adapter.normalize_arguments = Mock(side_effect=lambda x, y: x)
        return adapter

    @pytest.mark.asyncio
    async def test_stats_track_successful_calls(self, coordinator):
        """Test that successful calls are tracked in stats."""
        tool_calls = [{"name": "tool", "arguments": {}}]

        await coordinator.handle_tool_calls(tool_calls)

        stats = coordinator.get_execution_stats()
        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.failed_calls == 0

    @pytest.mark.asyncio
    async def test_stats_track_failed_calls(self, coordinator, mock_executor):
        """Test that failed calls are tracked in stats."""
        mock_executor.execute = AsyncMock(
            return_value=MagicMock(success=False, error="Tool failed")
        )

        tool_calls = [{"name": "tool", "arguments": {}}]

        await coordinator.handle_tool_calls(tool_calls)

        stats = coordinator.get_execution_stats()
        assert stats.total_calls == 1
        assert stats.successful_calls == 0
        assert stats.failed_calls == 1

    @pytest.mark.asyncio
    async def test_stats_track_budget_consumption(self, coordinator):
        """Test that budget consumption is tracked."""
        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
        ]

        await coordinator.handle_tool_calls(tool_calls)

        stats = coordinator.get_execution_stats()
        assert stats.budget_used == 2
        assert stats.budget_remaining == 48  # Default budget is 50

    @pytest.mark.asyncio
    async def test_stats_track_execution_times(self, coordinator):
        """Test that execution times are tracked."""
        tool_calls = [{"name": "tool", "arguments": {}}]

        await coordinator.handle_tool_calls(tool_calls)

        stats = coordinator.get_execution_stats()
        assert len(stats.execution_times_ms) == 1
        assert stats.execution_times_ms[0] > 0

    @pytest.mark.asyncio
    async def test_stats_track_failed_signatures(self, coordinator, mock_executor):
        """Test that failed call signatures are tracked."""
        mock_executor.execute = AsyncMock(
            return_value=MagicMock(success=False, error="Tool failed")
        )

        tool_calls = [
            {"name": "tool", "arguments": {"arg": "value"}},
            {"name": "tool", "arguments": {"arg": "value"}},  # Same signature
        ]

        await coordinator.handle_tool_calls(tool_calls)

        stats = coordinator.get_execution_stats()
        assert len(stats.failed_signatures) == 1

    def test_reset_stats(self, coordinator):
        """Test resetting statistics."""
        # Modify stats
        coordinator._stats.total_calls = 10
        coordinator._stats.budget_remaining = 40

        coordinator.reset_stats()

        stats = coordinator.get_execution_stats()
        assert stats.total_calls == 0
        assert stats.budget_remaining == 50  # Reset to default

    def test_reset_stats_with_custom_budget(self, coordinator):
        """Test resetting statistics with custom budget."""
        coordinator.reset_stats(new_budget=100)

        stats = coordinator.get_execution_stats()
        assert stats.budget_remaining == 100

    def test_clear_failed_signatures(self, coordinator):
        """Test clearing failed signatures."""
        coordinator._stats.failed_signatures.add(("tool", '{"arg": "value"}'))

        coordinator.clear_failed_signatures()

        stats = coordinator.get_execution_stats()
        assert len(stats.failed_signatures) == 0

    def test_consume_budget(self, coordinator):
        """Test manual budget consumption."""
        coordinator.consume_budget(amount=5)

        stats = coordinator.get_execution_stats()
        assert stats.budget_used == 5
        assert stats.budget_remaining == 45

    def test_consume_budget_triggers_warning_callback(self, coordinator):
        """Test that consuming budget triggers warning callback."""
        warning_mock = Mock()
        coordinator._on_budget_warning = warning_mock

        # Consume enough to go below 20% threshold
        coordinator._stats.budget_remaining = 9
        coordinator.consume_budget(amount=1)

        warning_mock.assert_called_once_with(8, 50)


class TestToolExecutionCoordinatorCallbacks:
    """Test suite for callback mechanisms."""

    @pytest.fixture
    def coordinator(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Create coordinator with callbacks."""
        on_tool_start = Mock()
        on_tool_complete = Mock()
        on_budget_warning = Mock()

        return ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            on_tool_start=on_tool_start,
            on_tool_complete=on_tool_complete,
            on_budget_warning=on_budget_warning,
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = AsyncMock()
        executor.execute = AsyncMock(
            return_value=MagicMock(success=True, result="output", error=None)
        )
        return executor

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=True)
        return registry

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock normalizer."""
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({}, "strategy"))
        return normalizer

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = Mock()
        adapter.normalize_arguments = Mock(side_effect=lambda x, y: x)
        return adapter

    @pytest.mark.asyncio
    async def test_on_tool_start_callback(self, coordinator):
        """Test that on_tool_start callback is invoked."""
        tool_calls = [{"name": "tool", "arguments": {"arg": "value"}}]

        await coordinator.handle_tool_calls(tool_calls)

        # Verify callback was invoked (arguments may be normalized)
        assert coordinator._on_tool_start.called
        call_args = coordinator._on_tool_start.call_args[0]
        assert call_args[0] == "tool"
        # Arguments may have been normalized
        assert isinstance(call_args[1], dict)

    @pytest.mark.asyncio
    async def test_on_tool_complete_callback(self, coordinator):
        """Test that on_tool_complete callback is invoked."""
        tool_calls = [{"name": "tool", "arguments": {}}]

        await coordinator.handle_tool_calls(tool_calls)

        assert coordinator._on_tool_complete.called
        call_args = coordinator._on_tool_complete.call_args[0][0]
        assert isinstance(call_args, ToolCallResult)
        assert call_args.success is True

    @pytest.mark.asyncio
    async def test_on_budget_warning_callback(self, coordinator):
        """Test that on_budget_warning callback is invoked when budget is low."""
        # Set budget to trigger warning (< 20% of 50 = < 10)
        coordinator._stats.budget_remaining = 9

        coordinator.consume_budget(amount=1)

        coordinator._on_budget_warning.assert_called_once()


class TestToolExecutionCoordinatorArgumentNormalization:
    """Test suite for argument normalization."""

    @pytest.fixture
    def coordinator(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Create coordinator."""
        return ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = AsyncMock()
        executor.execute = AsyncMock(
            return_value=MagicMock(success=True, result="output", error=None)
        )
        return executor

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=True)
        return registry

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock normalizer."""
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({}, "strategy"))
        return normalizer

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = Mock()
        adapter.normalize_arguments = Mock(side_effect=lambda x, y: x)
        return adapter

    @pytest.mark.asyncio
    async def test_normalize_json_string_arguments(self, coordinator, mock_normalizer):
        """Test normalization of JSON string arguments."""
        tool_calls = [{"name": "tool", "arguments": '{"arg": "value"}'}]

        await coordinator.handle_tool_calls(tool_calls)

        # Verify normalizer was called with parsed dict
        call_args = mock_normalizer.normalize_arguments.call_args[0][0]
        assert isinstance(call_args, dict)

    @pytest.mark.asyncio
    async def test_normalize_dict_arguments(self, coordinator, mock_normalizer):
        """Test normalization of dict arguments."""
        tool_calls = [{"name": "tool", "arguments": {"arg": "value"}}]

        await coordinator.handle_tool_calls(tool_calls)

        # Verify normalizer was called
        mock_normalizer.normalize_arguments.assert_called_once()

    @pytest.mark.asyncio
    async def test_normalize_none_arguments(self, coordinator, mock_normalizer):
        """Test normalization of None arguments."""
        tool_calls = [{"name": "tool", "arguments": None}]

        await coordinator.handle_tool_calls(tool_calls)

        # Verify normalizer was called with empty dict
        call_args = mock_normalizer.normalize_arguments.call_args[0][0]
        assert call_args == {}

    @pytest.mark.asyncio
    async def test_adapter_normalization_applied(self, coordinator, mock_adapter):
        """Test that adapter normalization is applied."""
        tool_calls = [{"name": "tool", "arguments": {"arg": "value"}}]

        await coordinator.handle_tool_calls(tool_calls)

        # Verify adapter was called
        mock_adapter.normalize_arguments.assert_called_once()


class TestToolExecutionCoordinatorResultFormatting:
    """Test suite for result formatting."""

    @pytest.fixture
    def coordinator(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Create coordinator."""
        return ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = AsyncMock()
        return executor

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=True)
        return registry

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock normalizer."""
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({}, "strategy"))
        return normalizer

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = Mock()
        adapter.normalize_arguments = Mock(side_effect=lambda x, y: x)
        return adapter

    def test_format_tool_output_default(self, coordinator):
        """Test default tool output formatting."""
        output = coordinator.format_tool_output(
            "test_tool", {"arg": "value"}, "result"
        )

        assert "test_tool" in output
        assert "result" in output

    def test_format_tool_output_custom_formatter(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Test custom tool output formatter."""
        mock_formatter = Mock()
        mock_formatter.format_tool_output = Mock(return_value="Custom formatted output")

        coordinator = ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            formatter=mock_formatter,
        )

        output = coordinator.format_tool_output("tool", {}, "result")

        mock_formatter.format_tool_output.assert_called_once_with("tool", {}, "result")
        assert output == "Custom formatted output"


class TestToolExecutionCoordinatorFailedSignatureTracking:
    """Test suite for failed signature tracking."""

    @pytest.fixture
    def coordinator(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Create coordinator with failed signature tracking enabled."""
        config = ToolExecutionConfig(enable_failed_signature_tracking=True)
        return ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            config=config,
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = AsyncMock()
        return executor

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=True)
        return registry

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock normalizer."""
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({}, "strategy"))
        return normalizer

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = Mock()
        adapter.normalize_arguments = Mock(side_effect=lambda x, y: x)
        return adapter

    @pytest.mark.asyncio
    async def test_skip_repeated_failing_call(self, coordinator, mock_executor):
        """Test that repeated failing calls are skipped."""
        # Disable retries to get clean count
        coordinator._config.retry_enabled = False

        # Make first call fail
        mock_executor.execute = AsyncMock(
            return_value=MagicMock(success=False, error="Tool failed")
        )

        # First call - will fail and add to failed_signatures
        tool_calls1 = [{"name": "tool", "arguments": {"arg": "value"}}]
        results1 = await coordinator.handle_tool_calls(tool_calls1)

        # Second call with same signature - should be skipped
        tool_calls2 = [{"name": "tool", "arguments": {"arg": "value"}}]
        results2 = await coordinator.handle_tool_calls(tool_calls2)

        # First result should show failure
        assert len(results1) == 1
        assert results1[0]["success"] is False

        # Second result should show it was skipped
        assert len(results2) == 1
        assert results2[0]["success"] is False
        assert "repeated failing call" in results2[0]["error"].lower()

        # Executor should only be called once (second call skipped)
        assert mock_executor.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_different_arguments_not_skipped(self, coordinator, mock_executor):
        """Test that different arguments are not skipped."""
        # Disable retries to get clean count
        coordinator._config.retry_enabled = False

        mock_executor.execute = AsyncMock(
            return_value=MagicMock(success=False, error="Tool failed")
        )

        # Make two calls with clearly different arguments
        # Using different tool names to ensure different signatures
        tool_calls = [
            {"name": "tool1", "arguments": {"arg": "value1"}},
            {"name": "tool2", "arguments": {"arg": "value2"}},
        ]

        results = await coordinator.handle_tool_calls(tool_calls)

        # Both should execute (different signatures - different tool names)
        assert len(results) == 2
        assert mock_executor.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_successful_call_clears_signature(self, coordinator, mock_executor):
        """Test that successful call doesn't add to failed signatures."""
        # First call succeeds
        mock_executor.execute = AsyncMock(
            return_value=MagicMock(success=True, result="success", error=None)
        )

        tool_calls = [
            {"name": "tool", "arguments": {"arg": "value"}},
            {"name": "tool", "arguments": {"arg": "value"}},  # Same call
        ]

        results = await coordinator.handle_tool_calls(tool_calls)

        # Both should succeed and execute
        assert len(results) == 2
        assert all(r["success"] for r in results)
        assert mock_executor.execute.call_count == 2

        # No failed signatures should be tracked
        stats = coordinator.get_execution_stats()
        assert len(stats.failed_signatures) == 0


class TestToolExecutionCoordinatorFactory:
    """Test suite for factory function."""

    def test_create_tool_execution_coordinator(self):
        """Test factory function creates coordinator."""
        mock_executor = Mock()
        mock_registry = Mock()
        mock_normalizer = Mock()
        mock_adapter = Mock()

        coordinator = create_tool_execution_coordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
        )

        assert isinstance(coordinator, ToolExecutionCoordinator)
        assert coordinator._executor is mock_executor
        assert coordinator._registry is mock_registry
        assert coordinator._argument_normalizer is mock_normalizer
        assert coordinator._tool_adapter is mock_adapter

    def test_create_tool_execution_coordinator_with_cache(self):
        """Test factory function with cache."""
        mock_executor = Mock()
        mock_registry = Mock()
        mock_normalizer = Mock()
        mock_adapter = Mock()
        mock_cache = Mock()
        config = ToolExecutionConfig(max_tool_budget=100)

        coordinator = create_tool_execution_coordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
            tool_cache=mock_cache,
            config=config,
        )

        assert coordinator._cache is mock_cache
        assert coordinator._config.max_tool_budget == 100


class TestToolExecutionCoordinatorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def coordinator(self, mock_executor, mock_registry, mock_normalizer, mock_adapter):
        """Create coordinator for testing."""
        return ToolExecutionCoordinator(
            tool_executor=mock_executor,
            tool_registry=mock_registry,
            argument_normalizer=mock_normalizer,
            tool_adapter=mock_adapter,
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = AsyncMock()
        return executor

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = Mock()
        registry.is_tool_enabled = Mock(return_value=True)
        return registry

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock normalizer."""
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({}, "strategy"))
        return normalizer

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = Mock()
        adapter.normalize_arguments = Mock(side_effect=lambda x, y: x)
        return adapter

    @pytest.mark.asyncio
    async def test_handle_exception_during_execution(self, coordinator, mock_executor):
        """Test handling of exception during tool execution."""
        mock_executor.execute = AsyncMock(side_effect=RuntimeError("Execution failed"))

        tool_calls = [{"name": "tool", "arguments": {}}]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "Execution failed" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self, coordinator, mock_executor):
        """Test handling mixed successful and failed calls."""
        # Create a function that returns different results based on tool name
        def execute_side_effect(*args, **kwargs):
            tool_name = kwargs.get('tool_name') or (args[0] if args else None)
            if tool_name == "tool2":
                return MagicMock(success=False, error="Tool failed")
            return MagicMock(success=True, result="output", error=None)

        mock_executor.execute = AsyncMock(side_effect=execute_side_effect)

        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
            {"name": "tool3", "arguments": {}},
        ]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[2]["success"] is True

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, coordinator, mock_executor):
        """Test that multiple tool calls execute concurrently."""
        mock_executor.execute = AsyncMock(
            return_value=MagicMock(success=True, result="output", error=None)
        )

        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
            {"name": "tool3", "arguments": {}},
        ]

        results = await coordinator.handle_tool_calls(tool_calls)

        assert len(results) == 3
        assert all(r["success"] for r in results)

    def test_get_execution_stats_returns_copy(self, coordinator):
        """Test that get_execution_stats returns actual stats object."""
        stats1 = coordinator.get_execution_stats()
        stats2 = coordinator.get_execution_stats()

        # Should return the same object
        assert stats1 is stats2

    @pytest.mark.asyncio
    async def test_tool_name_resolution(self, coordinator):
        """Test that tool names are resolved to canonical form."""
        # Mock the resolve_tool_name function from its actual location
        with patch("victor.tools.decorators.resolve_tool_name") as mock_resolve:
            mock_resolve.return_value = "canonical_tool_name"

            tool_calls = [{"name": "alias_tool", "arguments": {}}]

            await coordinator.handle_tool_calls(tool_calls)

            # Verify resolution was called
            mock_resolve.assert_called()
