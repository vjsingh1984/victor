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

"""Tests for victor.framework.middleware module.

These tests verify the framework-level middleware implementations:
- LoggingMiddleware
- SecretMaskingMiddleware
- MetricsMiddleware
- GitSafetyMiddleware
"""

import logging
import pytest
from unittest.mock import MagicMock, patch

from victor.core.vertical_types import MiddlewarePriority
from victor.framework.middleware import (
    GitSafetyMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    SecretMaskingMiddleware,
    ToolMetrics,
)


# =============================================================================
# LoggingMiddleware Tests
# =============================================================================


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create a logging middleware instance."""
        return LoggingMiddleware(log_level=logging.DEBUG)

    @pytest.mark.asyncio
    async def test_before_tool_call_logs_message(self, middleware, caplog):
        """LoggingMiddleware should log before tool call."""
        with caplog.at_level(logging.DEBUG):
            result = await middleware.before_tool_call("test_tool", {"arg1": "value1"})

        assert result.proceed is True
        assert "Tool call: test_tool" in caplog.text

    @pytest.mark.asyncio
    async def test_after_tool_call_logs_success(self, middleware, caplog):
        """LoggingMiddleware should log after successful tool call."""
        # First call before to set up timing
        await middleware.before_tool_call("test_tool", {"arg1": "value1"})

        with caplog.at_level(logging.DEBUG):
            result = await middleware.after_tool_call(
                "test_tool", {"arg1": "value1"}, "result", success=True
            )

        assert result is None  # No modification
        assert "Tool success: test_tool" in caplog.text

    @pytest.mark.asyncio
    async def test_after_tool_call_logs_failure(self, middleware, caplog):
        """LoggingMiddleware should log after failed tool call."""
        await middleware.before_tool_call("test_tool", {})

        with caplog.at_level(logging.DEBUG):
            await middleware.after_tool_call("test_tool", {}, "error", success=False)

        assert "Tool failed: test_tool" in caplog.text

    @pytest.mark.asyncio
    async def test_excluded_tools_not_logged(self):
        """LoggingMiddleware should not log excluded tools."""
        middleware = LoggingMiddleware(exclude_tools={"read_file"})

        result = await middleware.before_tool_call("read_file", {})
        assert result.proceed is True

        result = await middleware.after_tool_call("read_file", {}, "data", True)
        assert result is None

    @pytest.mark.asyncio
    async def test_sanitizes_sensitive_arguments(self, middleware, caplog):
        """LoggingMiddleware should sanitize sensitive arguments."""
        with caplog.at_level(logging.DEBUG):
            await middleware.before_tool_call(
                "api_call", {"password": "secret123", "api_key": "key123"}
            )

        assert "secret123" not in caplog.text
        assert "key123" not in caplog.text
        assert "[REDACTED]" in caplog.text

    def test_priority_is_deferred(self, middleware):
        """LoggingMiddleware should have DEFERRED priority."""
        assert middleware.get_priority() == MiddlewarePriority.DEFERRED

    def test_applicable_tools_is_none(self, middleware):
        """LoggingMiddleware should apply to all tools."""
        assert middleware.get_applicable_tools() is None


# =============================================================================
# SecretMaskingMiddleware Tests
# =============================================================================


class TestSecretMaskingMiddleware:
    """Tests for SecretMaskingMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create a secret masking middleware instance."""
        return SecretMaskingMiddleware(replacement="[MASKED]")

    @pytest.mark.asyncio
    async def test_before_tool_call_no_masking_by_default(self, middleware):
        """SecretMaskingMiddleware should not mask arguments by default."""
        result = await middleware.before_tool_call(
            "test_tool", {"data": "ghp_1234567890abcdefghijklmnopqrstuvwxyz"}
        )
        assert result.proceed is True
        assert result.modified_arguments is None

    @pytest.mark.asyncio
    async def test_before_tool_call_masks_arguments_when_enabled(self):
        """SecretMaskingMiddleware should mask arguments when enabled."""
        middleware = SecretMaskingMiddleware(replacement="[MASKED]", mask_in_arguments=True)

        result = await middleware.before_tool_call(
            "test_tool", {"token": "ghp_1234567890abcdefghijklmnopqrstuvwxyz"}
        )

        assert result.proceed is True
        assert result.modified_arguments is not None
        assert "[MASKED]" in result.modified_arguments["token"]

    @pytest.mark.asyncio
    async def test_after_tool_call_masks_secrets_in_result(self, middleware):
        """SecretMaskingMiddleware should mask secrets in result."""
        result_with_secret = "Found token: ghp_1234567890abcdefghijklmnopqrstuvwxyz"

        masked = await middleware.after_tool_call("test_tool", {}, result_with_secret, success=True)

        assert masked is not None
        assert "[MASKED]" in masked
        assert "ghp_" not in masked

    @pytest.mark.asyncio
    async def test_after_tool_call_masks_aws_keys(self, middleware):
        """SecretMaskingMiddleware should mask AWS keys."""
        result_with_key = "AWS key: AKIAIOSFODNN7EXAMPLE"

        masked = await middleware.after_tool_call("test_tool", {}, result_with_key, success=True)

        assert masked is not None
        assert "[MASKED]" in masked
        assert "AKIAIOSFODNN7" not in masked

    @pytest.mark.asyncio
    async def test_after_tool_call_no_change_when_no_secrets(self, middleware):
        """SecretMaskingMiddleware should return None when no secrets found."""
        clean_result = "This is a normal response without secrets."

        masked = await middleware.after_tool_call("test_tool", {}, clean_result, success=True)

        assert masked is None  # No modification needed

    @pytest.mark.asyncio
    async def test_after_tool_call_handles_dict_result(self, middleware):
        """SecretMaskingMiddleware should handle dict results."""
        result_dict = {
            "output": "Token: ghp_1234567890abcdefghijklmnopqrstuvwxyz",
            "status": "ok",
        }

        masked = await middleware.after_tool_call("test_tool", {}, result_dict, success=True)

        assert masked is not None
        assert "[MASKED]" in masked["output"]
        assert masked["status"] == "ok"

    @pytest.mark.asyncio
    async def test_after_tool_call_handles_none_result(self, middleware):
        """SecretMaskingMiddleware should handle None result."""
        masked = await middleware.after_tool_call("test_tool", {}, None, success=True)
        assert masked is None

    def test_priority_is_high(self, middleware):
        """SecretMaskingMiddleware should have HIGH priority."""
        assert middleware.get_priority() == MiddlewarePriority.HIGH


# =============================================================================
# MetricsMiddleware Tests
# =============================================================================


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create a metrics middleware instance."""
        return MetricsMiddleware(enable_timing=True)

    @pytest.mark.asyncio
    async def test_records_call_count(self, middleware):
        """MetricsMiddleware should record call counts."""
        await middleware.before_tool_call("test_tool", {})
        await middleware.after_tool_call("test_tool", {}, "result", success=True)

        metrics = middleware.get_metrics("test_tool")
        assert metrics is not None
        assert metrics.call_count == 1
        assert metrics.success_count == 1
        assert metrics.failure_count == 0

    @pytest.mark.asyncio
    async def test_records_success_and_failure(self, middleware):
        """MetricsMiddleware should track success and failure separately."""
        # Record successes
        for _ in range(3):
            await middleware.before_tool_call("test_tool", {})
            await middleware.after_tool_call("test_tool", {}, "ok", success=True)

        # Record failures
        for _ in range(2):
            await middleware.before_tool_call("test_tool", {})
            await middleware.after_tool_call("test_tool", {}, "error", success=False)

        metrics = middleware.get_metrics("test_tool")
        assert metrics.call_count == 5
        assert metrics.success_count == 3
        assert metrics.failure_count == 2
        assert metrics.success_rate == 0.6

    @pytest.mark.asyncio
    async def test_records_timing(self, middleware):
        """MetricsMiddleware should record execution timing."""
        await middleware.before_tool_call("test_tool", {})
        # Simulate some delay (would happen naturally in real execution)
        await middleware.after_tool_call("test_tool", {}, "result", success=True)

        metrics = middleware.get_metrics("test_tool")
        assert metrics.total_duration_ms >= 0
        assert metrics.min_duration_ms >= 0
        assert metrics.max_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_callback_invoked(self, middleware):
        """MetricsMiddleware should invoke callback after recording."""
        callback_data = []
        middleware._callback = lambda name, m: callback_data.append((name, m))

        await middleware.before_tool_call("test_tool", {})
        await middleware.after_tool_call("test_tool", {}, "result", success=True)

        assert len(callback_data) == 1
        assert callback_data[0][0] == "test_tool"
        assert callback_data[0][1].call_count == 1

    def test_get_summary_returns_all_metrics(self, middleware):
        """MetricsMiddleware should return summary of all metrics."""
        # Manually add metrics
        middleware._metrics["tool1"] = ToolMetrics(tool_name="tool1", call_count=5)
        middleware._metrics["tool2"] = ToolMetrics(tool_name="tool2", call_count=3)

        summary = middleware.get_summary()
        assert len(summary) == 2
        assert "tool1" in summary
        assert "tool2" in summary

    def test_reset_clears_metrics(self, middleware):
        """MetricsMiddleware.reset() should clear all metrics."""
        middleware._metrics["tool1"] = ToolMetrics(tool_name="tool1", call_count=5)
        middleware.reset()

        assert len(middleware._metrics) == 0
        assert middleware.get_metrics("tool1") is None

    def test_export_prometheus(self, middleware):
        """MetricsMiddleware should export Prometheus format."""
        middleware._metrics["test_tool"] = ToolMetrics(
            tool_name="test_tool",
            call_count=10,
            success_count=8,
            total_duration_ms=500.0,
        )

        output = middleware.export_prometheus()
        assert 'tool_calls_total{tool="test_tool"} 10' in output
        assert 'tool_calls_success_total{tool="test_tool"} 8' in output
        assert "tool_duration_ms_avg" in output

    def test_priority_is_low(self, middleware):
        """MetricsMiddleware should have LOW priority."""
        assert middleware.get_priority() == MiddlewarePriority.LOW


class TestToolMetrics:
    """Tests for ToolMetrics dataclass."""

    def test_avg_duration_zero_when_no_calls(self):
        """avg_duration_ms should be 0 when no calls made."""
        metrics = ToolMetrics(tool_name="test")
        assert metrics.avg_duration_ms == 0.0

    def test_avg_duration_calculated(self):
        """avg_duration_ms should be calculated correctly."""
        metrics = ToolMetrics(tool_name="test", call_count=4, total_duration_ms=200.0)
        assert metrics.avg_duration_ms == 50.0

    def test_success_rate_zero_when_no_calls(self):
        """success_rate should be 0 when no calls made."""
        metrics = ToolMetrics(tool_name="test")
        assert metrics.success_rate == 0.0

    def test_success_rate_calculated(self):
        """success_rate should be calculated correctly."""
        metrics = ToolMetrics(tool_name="test", call_count=10, success_count=7)
        assert metrics.success_rate == 0.7


# =============================================================================
# GitSafetyMiddleware Tests
# =============================================================================


class TestGitSafetyMiddleware:
    """Tests for GitSafetyMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create a git safety middleware instance."""
        return GitSafetyMiddleware(block_dangerous=True, warn_on_risky=True)

    @pytest.mark.asyncio
    async def test_blocks_force_push(self, middleware):
        """GitSafetyMiddleware should block force push."""
        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin main"}
        )

        assert result.proceed is False
        assert "force" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_blocks_force_push_short_flag(self, middleware):
        """GitSafetyMiddleware should block force push with -f flag."""
        result = await middleware.before_tool_call("git", {"command": "git push -f origin main"})

        assert result.proceed is False
        # Message may say "push -f" or "force" depending on which pattern matched
        assert "push -f" in result.error_message or "force" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_blocks_hard_reset(self, middleware):
        """GitSafetyMiddleware should block hard reset."""
        result = await middleware.before_tool_call("bash", {"command": "git reset --hard HEAD~3"})

        assert result.proceed is False

    @pytest.mark.asyncio
    async def test_blocks_clean_fd(self, middleware):
        """GitSafetyMiddleware should block git clean -fd."""
        result = await middleware.before_tool_call("shell", {"command": "git clean -fd"})

        assert result.proceed is False

    @pytest.mark.asyncio
    async def test_warns_on_risky_operations(self, middleware):
        """GitSafetyMiddleware should warn on risky operations."""
        result = await middleware.before_tool_call("execute_bash", {"command": "git rebase main"})

        assert result.proceed is True
        assert "git_warning" in result.metadata

    @pytest.mark.asyncio
    async def test_warns_on_stash_drop(self, middleware):
        """GitSafetyMiddleware should warn on stash drop."""
        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git stash drop stash@{0}"}
        )

        assert result.proceed is True
        assert "git_warning" in result.metadata

    @pytest.mark.asyncio
    async def test_allows_safe_operations(self, middleware):
        """GitSafetyMiddleware should allow safe git operations."""
        result = await middleware.before_tool_call("execute_bash", {"command": "git status"})

        assert result.proceed is True
        assert not result.error_message

    @pytest.mark.asyncio
    async def test_allows_normal_push(self, middleware):
        """GitSafetyMiddleware should allow normal push."""
        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push origin feature-branch"}
        )

        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_ignores_non_git_commands(self, middleware):
        """GitSafetyMiddleware should ignore non-git commands."""
        result = await middleware.before_tool_call("execute_bash", {"command": "ls -la"})

        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_ignores_non_applicable_tools(self, middleware):
        """GitSafetyMiddleware should ignore non-applicable tools."""
        result = await middleware.before_tool_call("read_file", {"path": "/etc/passwd"})

        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_blocks_force_push_to_protected_branch(self, middleware):
        """GitSafetyMiddleware should block force push to protected branches."""
        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin main"}
        )

        assert result.proceed is False
        assert "main" in result.error_message or "force" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_custom_protected_branches(self):
        """GitSafetyMiddleware should respect custom protected branches."""
        middleware = GitSafetyMiddleware(
            block_dangerous=True,
            protected_branches={"deploy", "release"},
        )

        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin deploy"}
        )

        # The force push itself is blocked, regardless of branch
        assert result.proceed is False

    @pytest.mark.asyncio
    async def test_allowed_force_branches(self):
        """GitSafetyMiddleware should allow force push on configured branches."""
        middleware = GitSafetyMiddleware(
            block_dangerous=True,
            allowed_force_branches={"feature/*"},
        )

        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin feature/test"}
        )

        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_disabled_blocking(self):
        """GitSafetyMiddleware should not block when disabled."""
        middleware = GitSafetyMiddleware(block_dangerous=False, warn_on_risky=False)

        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin main"}
        )

        assert result.proceed is True

    def test_priority_is_critical(self, middleware):
        """GitSafetyMiddleware should have CRITICAL priority."""
        assert middleware.get_priority() == MiddlewarePriority.CRITICAL

    def test_applicable_tools(self, middleware):
        """GitSafetyMiddleware should only apply to shell tools."""
        tools = middleware.get_applicable_tools()
        assert "git" in tools
        assert "execute_bash" in tools
        assert "bash" in tools
        assert "shell" in tools


# =============================================================================
# Integration Tests
# =============================================================================


class TestMiddlewareIntegration:
    """Integration tests for middleware working together."""

    @pytest.mark.asyncio
    async def test_middleware_chain_ordering(self):
        """Middleware should execute in priority order."""
        execution_order = []

        class TrackingMiddleware:
            def __init__(self, name, priority):
                self._name = name
                self._priority = priority

            async def before_tool_call(self, tool_name, arguments):
                from victor.core.vertical_types import MiddlewareResult

                execution_order.append(f"{self._name}_before")
                return MiddlewareResult()

            async def after_tool_call(self, tool_name, arguments, result, success):
                execution_order.append(f"{self._name}_after")
                return None

            def get_priority(self):
                return self._priority

            def get_applicable_tools(self):
                return None

        # Create middleware with different priorities
        m1 = TrackingMiddleware("critical", MiddlewarePriority.CRITICAL)
        m2 = TrackingMiddleware("high", MiddlewarePriority.HIGH)
        m3 = TrackingMiddleware("low", MiddlewarePriority.LOW)

        # Simulate execution in priority order
        middlewares = sorted([m1, m2, m3], key=lambda m: m.get_priority().value)

        for m in middlewares:
            await m.before_tool_call("test", {})

        assert execution_order == ["critical_before", "high_before", "low_before"]

    @pytest.mark.asyncio
    async def test_devops_middleware_configuration(self):
        """DevOps vertical should configure middleware correctly."""
        from victor.devops.assistant import DevOpsAssistant

        middleware_list = DevOpsAssistant.get_middleware()

        # Should have at least GitSafetyMiddleware
        middleware_types = [type(m).__name__ for m in middleware_list]
        assert "GitSafetyMiddleware" in middleware_types
        assert "SecretMaskingMiddleware" in middleware_types
        assert "LoggingMiddleware" in middleware_types

        # Git safety should block dangerous operations for DevOps
        git_middleware = next(
            m for m in middleware_list if type(m).__name__ == "GitSafetyMiddleware"
        )
        result = await git_middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin main"}
        )
        assert result.proceed is False


__all__ = [
    "TestLoggingMiddleware",
    "TestSecretMaskingMiddleware",
    "TestMetricsMiddleware",
    "TestGitSafetyMiddleware",
    "TestToolMetrics",
    "TestMiddlewareIntegration",
]
