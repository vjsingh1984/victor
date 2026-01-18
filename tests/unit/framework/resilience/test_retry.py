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

"""Tests for framework resilience retry handlers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.resilience.retry import (
    DatabaseRetryHandler,
    FRAMEWORK_RETRY_HANDLERS,
    NetworkRetryHandler,
    RateLimitRetryHandler,
    register_framework_retry_handlers,
    retry_with_backoff,
    retry_with_backoff_sync,
    RetryConfig,
    RetryHandler,
    RetryHandlerConfig,
    with_exponential_backoff,
    with_exponential_backoff_sync,
)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter_factor == 0.1

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config = RetryConfig.from_dict(
            {
                "max_retries": 5,
                "base_delay": 2.0,
                "max_delay": 120.0,
            }
        )
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0


class TestRetryHandlerConfig:
    """Tests for RetryHandlerConfig."""

    def test_default_config(self):
        """Test default handler configuration."""
        config = RetryHandlerConfig()
        assert config.fail_fast is False
        assert config.log_attempts is True
        assert config.backoff_strategy == "exponential"


class TestRetryWithBackoff:
    """Tests for standalone retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self):
        """Test successful call without retries."""

        async def mock_func():
            return "success"

        result = await retry_with_backoff(mock_func, max_retries=3)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test retry on connection error."""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await retry_with_backoff(
            failing_func,
            max_retries=5,
            base_delay=0.01,
        )
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        """Test failure after exhausting retries."""

        async def always_failing_func():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            await retry_with_backoff(
                always_failing_func,
                max_retries=2,
                base_delay=0.01,
            )

    @pytest.mark.asyncio
    async def test_with_jitter(self):
        """Test that jitter is applied to delays."""
        delays = []

        async def track_delay():
            # We can't directly measure delay, but we can verify
            # the function completes without error
            raise ConnectionError("Fail")

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = asyncio.sleep(0)
            try:
                await retry_with_backoff(
                    track_delay,
                    max_retries=2,
                    base_delay=1.0,
                    jitter=0.5,
                )
            except ConnectionError:
                pass

            # With max_retries=2, we have 1 initial attempt + 2 retries = 3 attempts
            # Sleep is called after each failed attempt except the last
            # So sleep is called 2 times (after attempt 1 and attempt 2)
            assert mock_sleep.call_count >= 1  # At least one sleep before a retry

    def test_retry_sync_successful(self):
        """Test synchronous retry with success."""
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = retry_with_backoff_sync(
            failing_func,
            max_retries=3,
            base_delay=0.01,
        )
        assert result == "success"


class TestWithExponentialBackoffDecorator:
    """Tests for @with_exponential_backoff decorator."""

    @pytest.mark.asyncio
    async def test_async_decorator_success(self):
        """Test async decorator with successful execution."""

        @with_exponential_backoff(max_retries=3, base_delay=0.01)
        async def test_func():
            return "result"

        result = await test_func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_async_decorator_retry(self):
        """Test async decorator with retry."""
        call_count = 0

        @with_exponential_backoff(max_retries=3, base_delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Fail")
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 2

    def test_sync_decorator_success(self):
        """Test sync decorator with successful execution."""

        @with_exponential_backoff_sync(max_retries=3, base_delay=0.01)
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"


class TestRetryHandler:
    """Tests for RetryHandler workflow compute handler."""

    @pytest.mark.asyncio
    async def test_handler_execution(self):
        """Test handler execution with mock workflow context."""
        handler = RetryHandler()

        # Create mock workflow node
        node = MagicMock()
        node.id = "test_node"
        node.tools = ["test_tool"]
        node.output_key = "result"
        node.constraints = MagicMock()
        node.constraints.allows_tool.return_value = True
        node.constraints.timeout = 30.0
        node.input_mapping = {}
        node.config = {}

        # Create mock context
        context = MagicMock()
        context.data = {}
        context.set = MagicMock()

        # Create mock tool registry
        tool_registry = MagicMock()
        result_mock = MagicMock()
        result_mock.success = True
        result_mock.output = "tool_result"
        tool_registry.execute = AsyncMock(return_value=result_mock)

        # Execute handler
        result = await handler(node, context, tool_registry)

        # Verify result
        assert result.status.value == "completed"
        assert result.output == {"test_tool": "tool_result"}
        tool_registry.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_with_retry(self):
        """Test handler retries on failure."""
        handler = RetryHandler(
            RetryHandlerConfig(
                retry_config=RetryConfig(max_retries=3, base_delay=0.01),
            )
        )

        node = MagicMock()
        node.id = "test_node"
        node.tools = ["test_tool"]
        node.output_key = "result"
        node.constraints = MagicMock()
        node.constraints.allows_tool.return_value = True
        node.constraints.timeout = 30.0
        node.constraints.to_dict.return_value = {}
        node.input_mapping = {}
        node.config = {}

        context = MagicMock()
        context.data = {}
        context.set = MagicMock()

        tool_registry = MagicMock()
        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count < 3:
                result.success = False
                # Use a retryable error message
                result.error = "connection timeout"
            else:
                result.success = True
                result.output = "success"
            return result

        tool_registry.execute = mock_execute

        result = await handler(node, context, tool_registry)

        assert result.status.value == "completed"
        assert call_count == 3


class TestSpecializedHandlers:
    """Tests for specialized retry handlers."""

    def test_network_retry_handler_config(self):
        """Test NetworkRetryHandler has correct defaults."""
        handler = NetworkRetryHandler()
        assert handler._config.retry_config.max_retries == 5
        assert handler._config.retry_config.base_delay == 2.0
        assert handler._config.retry_config.jitter_factor == 0.25

    def test_rate_limit_retry_handler_config(self):
        """Test RateLimitRetryHandler has correct defaults."""
        handler = RateLimitRetryHandler()
        assert handler._config.retry_config.max_retries == 4
        assert handler._config.retry_config.base_delay == 5.0
        assert handler._config.retry_config.exponential_base == 2.5

    def test_database_retry_handler_config(self):
        """Test DatabaseRetryHandler has correct defaults."""
        handler = DatabaseRetryHandler()
        assert handler._config.retry_config.max_retries == 3
        assert handler._config.retry_config.base_delay == 0.5


class TestFrameworkHandlers:
    """Tests for framework handler registry."""

    def test_framework_handlers_dict(self):
        """Test FRAMEWORK_RETRY_HANDLERS contains expected handlers."""
        assert "retry_with_backoff" in FRAMEWORK_RETRY_HANDLERS
        assert "network_retry" in FRAMEWORK_RETRY_HANDLERS
        assert "rate_limit_retry" in FRAMEWORK_RETRY_HANDLERS
        assert "database_retry" in FRAMEWORK_RETRY_HANDLERS

        # Verify all are RetryHandler instances
        for handler in FRAMEWORK_RETRY_HANDLERS.values():
            assert isinstance(handler, RetryHandler)

    @pytest.mark.asyncio
    async def test_register_framework_handlers(self):
        """Test registering framework handlers with executor."""
        # This test verifies the registration function doesn't error
        # Actual registration would require a full workflow executor setup
        register_framework_retry_handlers()  # Should not raise


class TestRetryHandlerErrorClassification:
    """Tests for error classification in retry handler."""

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test that non-retryable exceptions stop immediately."""
        handler = RetryHandler(
            RetryHandlerConfig(
                retry_config=RetryConfig(
                    non_retryable_exceptions=(ValueError,),
                ),
            )
        )

        assert handler._is_retryable_exception(ValueError) is False
        assert handler._is_retryable_exception(ConnectionError) is True

    @pytest.mark.asyncio
    async def test_retryable_error_patterns(self):
        """Test error message pattern matching."""
        handler = RetryHandler()

        # Test patterns that should match
        assert handler._is_retryable_error("timeout occurred") is True
        assert handler._is_retryable_error("connection failed") is True
        assert handler._is_retryable_error("network error") is True

        # Test patterns that shouldn't match
        assert handler._is_retryable_error("validation error") is False
        assert handler._is_retryable_error("authentication failed") is False
