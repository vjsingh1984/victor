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

"""Unit tests for ToolRetryCoordinator.

Tests tool execution with retry logic, exponential backoff, and cache integration.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.coordinators.tool_retry_coordinator import (
    ToolRetryCoordinator,
    ToolRetryConfig,
    create_tool_retry_coordinator,
)


@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor."""
    executor = AsyncMock()
    return executor


@pytest.fixture
def mock_tool_cache():
    """Create a mock tool cache."""
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)
    cache.set = MagicMock()
    cache.invalidate_paths = MagicMock()
    cache.clear_namespaces = MagicMock()
    return cache


@pytest.fixture
def mock_task_completion_detector():
    """Create a mock task completion detector."""
    detector = MagicMock()
    detector.record_tool_result = MagicMock()
    return detector


@pytest.fixture
def retry_config():
    """Create a retry configuration for testing."""
    return ToolRetryConfig(
        retry_enabled=True,
        max_attempts=3,
        base_delay=0.1,  # Short delay for tests
        max_delay=1.0,
    )


class TestToolRetryCoordinator:
    """Test suite for ToolRetryCoordinator."""

    def test_initialization(self, mock_tool_executor, retry_config):
        """Test coordinator initialization."""
        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            config=retry_config,
        )

        assert coordinator._tool_executor == mock_tool_executor
        assert coordinator._config == retry_config
        assert coordinator._tool_cache is None

    def test_initialization_with_all_components(
        self, mock_tool_executor, mock_tool_cache, mock_task_completion_detector
    ):
        """Test coordinator initialization with all components."""
        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            tool_cache=mock_tool_cache,
            task_completion_detector=mock_task_completion_detector,
        )

        assert coordinator._tool_executor == mock_tool_executor
        assert coordinator._tool_cache == mock_tool_cache
        assert coordinator._task_completion_detector == mock_task_completion_detector

    @pytest.mark.asyncio
    async def test_successful_tool_execution(self, mock_tool_executor, retry_config):
        """Test successful tool execution on first attempt."""
        # Setup
        mock_result = MagicMock()
        mock_result.success = True
        mock_tool_executor.execute.return_value = mock_result

        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            config=retry_config,
        )

        # Execute
        result = await coordinator.execute_tool(
            tool_name="read_file",
            tool_args={"path": "/tmp/test.txt"},
            context={},
        )

        # Verify
        assert result.success is True
        assert result.result == mock_result
        assert result.error_message is None
        assert result.attempts == 1
        assert result.from_cache is False
        mock_tool_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_execution_with_cache_hit(self, mock_tool_executor, mock_tool_cache):
        """Test tool execution with cache hit."""
        # Setup - cache returns a result
        cached_result = (MagicMock(), True, None)
        mock_tool_cache.get.return_value = cached_result

        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            tool_cache=mock_tool_cache,
            config=ToolRetryConfig(cache_enabled=True),
        )

        # Execute
        result = await coordinator.execute_tool(
            tool_name="read_file",
            tool_args={"path": "/tmp/test.txt"},
            context={},
        )

        # Verify
        assert result.success is True
        assert result.from_cache is True
        assert result.attempts == 1
        mock_tool_executor.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_execution_retry_on_failure(self, mock_tool_executor, retry_config):
        """Test tool execution retries on failure."""
        # Setup - fail twice, then succeed
        mock_failure = MagicMock()
        mock_failure.success = False
        mock_failure.error = "Temporary error"

        mock_success = MagicMock()
        mock_success.success = True

        mock_tool_executor.execute.side_effect = [mock_failure, mock_failure, mock_success]

        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            config=retry_config,
        )

        # Execute
        result = await coordinator.execute_tool(
            tool_name="read_file",
            tool_args={"path": "/tmp/test.txt"},
            context={},
        )

        # Verify
        assert result.success is True
        assert result.attempts == 3
        assert mock_tool_executor.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_tool_execution_non_retryable_error(self, mock_tool_executor, retry_config):
        """Test tool execution fails immediately on non-retryable error."""
        # Setup - validation error (non-retryable)
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Invalid argument: path is required"
        mock_tool_executor.execute.return_value = mock_result

        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            config=retry_config,
        )

        # Execute
        result = await coordinator.execute_tool(
            tool_name="read_file",
            tool_args={},
            context={},
        )

        # Verify
        assert result.success is False
        assert "Invalid argument" in result.error_message
        assert result.attempts == 1
        mock_tool_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_execution_max_attempts_reached(self, mock_tool_executor, retry_config):
        """Test tool execution fails after max attempts."""
        # Setup - always fail
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Service unavailable"
        mock_tool_executor.execute.return_value = mock_result

        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            config=retry_config,
        )

        # Execute
        result = await coordinator.execute_tool(
            tool_name="read_file",
            tool_args={"path": "/tmp/test.txt"},
            context={},
        )

        # Verify
        assert result.success is False
        assert result.attempts == 3
        assert mock_tool_executor.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_tool_execution_timeout_error(self, mock_tool_executor, retry_config):
        """Test tool execution handles timeout errors."""
        # Setup - timeout error
        mock_tool_executor.execute.side_effect = asyncio.TimeoutError("Request timed out")

        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            config=retry_config,
        )

        # Execute
        result = await coordinator.execute_tool(
            tool_name="read_file",
            tool_args={"path": "/tmp/test.txt"},
            context={},
        )

        # Verify
        assert result.success is False
        assert "timed out" in result.error_message.lower()
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_tool_execution_connection_error(self, mock_tool_executor, retry_config):
        """Test tool execution handles connection errors."""
        # Setup - connection error
        mock_tool_executor.execute.side_effect = ConnectionError("Connection refused")

        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            config=retry_config,
        )

        # Execute
        result = await coordinator.execute_tool(
            tool_name="read_file",
            tool_args={"path": "/tmp/test.txt"},
            context={},
        )

        # Verify
        assert result.success is False
        assert "connection" in result.error_message.lower()
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_tool_execution_with_task_completion(
        self, mock_tool_executor, mock_task_completion_detector
    ):
        """Test tool execution records to task completion detector."""
        # Setup
        mock_result = MagicMock()
        mock_result.success = True
        mock_tool_executor.execute.return_value = mock_result

        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            task_completion_detector=mock_task_completion_detector,
            config=ToolRetryConfig(task_completion_enabled=True),
        )

        # Execute
        result = await coordinator.execute_tool(
            tool_name="write_file",
            tool_args={"path": "/tmp/test.txt", "content": "test"},
            context={},
        )

        # Verify
        assert result.success is True
        mock_task_completion_detector.record_tool_result.assert_called_once()
        call_args = mock_task_completion_detector.record_tool_result.call_args
        assert call_args[0][0] == "write_file"
        assert call_args[0][1]["success"] is True
        assert call_args[0][1]["path"] == "/tmp/test.txt"

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_write(self, mock_tool_executor, mock_tool_cache):
        """Test cache is invalidated for write operations."""
        # Setup
        mock_result = MagicMock()
        mock_result.success = True
        mock_tool_executor.execute.return_value = mock_result

        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            tool_cache=mock_tool_cache,
            config=ToolRetryConfig(cache_enabled=True),
        )

        # Execute write_file
        result = await coordinator.execute_tool(
            tool_name="write_file",
            tool_args={"path": "/tmp/test.txt", "content": "test"},
            context={},
        )

        # Verify
        assert result.success is True
        mock_tool_cache.set.assert_called_once()
        mock_tool_cache.invalidate_paths.assert_called_once_with(["/tmp/test.txt"])

    @pytest.mark.asyncio
    async def test_cache_namespace_clear_on_bash(self, mock_tool_executor, mock_tool_cache):
        """Test cache namespaces are cleared for bash operations."""
        # Setup
        mock_result = MagicMock()
        mock_result.success = True
        mock_tool_executor.execute.return_value = mock_result

        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            tool_cache=mock_tool_cache,
            config=ToolRetryConfig(cache_enabled=True),
        )

        # Execute bash without paths
        result = await coordinator.execute_tool(
            tool_name="execute_bash",
            tool_args={"command": "ls -la"},
            context={},
        )

        # Verify
        assert result.success is True
        mock_tool_cache.clear_namespaces.assert_called_once()

    def test_calculate_backoff(self, retry_config):
        """Test exponential backoff calculation."""
        coordinator = ToolRetryCoordinator(
            tool_executor=AsyncMock(),
            config=retry_config,
        )

        # Verify exponential backoff
        assert coordinator._calculate_backoff(0) == 0.1
        assert coordinator._calculate_backoff(1) == 0.2
        assert coordinator._calculate_backoff(2) == 0.4
        # Should cap at max_delay
        assert coordinator._calculate_backoff(10) == 1.0

    def test_get_config(self, retry_config):
        """Test getting current configuration."""
        coordinator = ToolRetryCoordinator(
            tool_executor=AsyncMock(),
            config=retry_config,
        )

        config = coordinator.get_config()
        assert config == retry_config
        assert config.retry_enabled is True
        assert config.max_attempts == 3

    def test_update_config(self, mock_tool_executor):
        """Test updating configuration."""
        coordinator = ToolRetryCoordinator(
            tool_executor=mock_tool_executor,
            config=ToolRetryConfig(max_attempts=2),
        )

        new_config = ToolRetryConfig(max_attempts=5)
        coordinator.update_config(new_config)

        assert coordinator._config == new_config
        assert coordinator._config.max_attempts == 5


class TestCreateToolRetryCoordinator:
    """Test suite for factory function."""

    def test_factory_with_defaults(self, mock_tool_executor):
        """Test factory creates coordinator with defaults."""
        coordinator = create_tool_retry_coordinator(
            tool_executor=mock_tool_executor,
        )

        assert coordinator._tool_executor == mock_tool_executor
        assert coordinator._config.retry_enabled is True
        assert coordinator._config.max_attempts == 3
        assert coordinator._config.base_delay == 1.0

    def test_factory_with_custom_config(self, mock_tool_executor):
        """Test factory creates coordinator with custom config."""
        coordinator = create_tool_retry_coordinator(
            tool_executor=mock_tool_executor,
            max_attempts=5,
            base_delay=2.0,
            retry_enabled=False,
        )

        assert coordinator._config.max_attempts == 5
        assert coordinator._config.base_delay == 2.0
        assert coordinator._config.retry_enabled is False

    def test_factory_with_all_components(
        self, mock_tool_executor, mock_tool_cache, mock_task_completion_detector
    ):
        """Test factory with all optional components."""
        coordinator = create_tool_retry_coordinator(
            tool_executor=mock_tool_executor,
            tool_cache=mock_tool_cache,
            task_completion_detector=mock_task_completion_detector,
        )

        assert coordinator._tool_cache == mock_tool_cache
        assert coordinator._task_completion_detector == mock_task_completion_detector
