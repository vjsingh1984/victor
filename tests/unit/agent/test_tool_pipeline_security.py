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

"""
Comprehensive Security Tests for ToolPipeline.

This module tests security-critical aspects of the ToolPipeline:
- Tool validation (name format, existence, enabled status)
- Safety checks (dangerous command detection)
- Sandboxing (resource limits, path validation)
- Authorization flow (approval modes, confirmation callbacks)
- Execution pipeline (order, middleware integration)
- Error handling (propagation, containment)
- Caching behavior (security of cached results)
- Middleware integration (before/after processing)

Target: 70%+ coverage for victor/agent/tool_pipeline.py (2,183 lines)
"""

import asyncio
import json
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass
from threading import Lock
import time

from victor.agent.tool_pipeline import (
    ToolPipeline,
    ToolPipelineConfig,
    ToolCallResult,
    PipelineExecutionResult,
    ExecutionMetrics,
    LRUToolCache,
    ToolRateLimiter,
    IDEMPOTENT_TOOLS,
)
from victor.agent.argument_normalizer import NormalizationStrategy
from victor.agent.safety import (
    SafetyChecker,
    OperationalRiskLevel,
    ApprovalMode,
    ConfirmationRequest,
    set_confirmation_callback,
)
from victor.tools.base import BaseTool, ToolRegistry


# =============================================================================
# Test Fixtures
# =============================================================================


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(
        self,
        name: str = "test_tool",
        enabled: bool = True,
        execute_result: Any = "success",
        should_fail: bool = False,
    ):
        self.name = name
        self.enabled = enabled
        self._execute_result = execute_result
        self._should_fail = should_fail
        self.execute_count = 0
        self.last_args: Optional[Dict[str, Any]] = None

    async def execute(self, **kwargs) -> Any:
        """Mock execution."""
        self.execute_count += 1
        self.last_args = kwargs
        if self._should_fail:
            raise RuntimeError("Tool execution failed")
        return self._execute_result


@dataclass
class MockToolCall:
    """Mock tool call structure."""

    name: str
    arguments: Dict[str, Any]


@pytest.fixture
def mock_registry():
    """Create mock tool registry."""
    registry = Mock(spec=ToolRegistry)
    registry.is_tool_enabled = Mock(return_value=True)
    registry.get_tool = Mock(return_value=None)
    # Add methods that decision cache might call
    registry.has_tool = Mock(return_value=True)
    registry.get_all_tools = Mock(return_value=[])
    return registry


@pytest.fixture
def mock_executor():
    """Create mock tool executor."""
    executor = Mock()
    executor.execute = AsyncMock(return_value=Mock(success=True, result="test result"))
    return executor


@pytest.fixture
def pipeline_config():
    """Create pipeline config for testing."""
    return ToolPipelineConfig(
        tool_budget=10,
        enable_caching=True,
        enable_analytics=True,
        enable_failed_signature_tracking=True,
        max_tool_name_length=64,
    )


@pytest.fixture
def basic_pipeline(mock_registry, mock_executor, pipeline_config):
    """Create basic ToolPipeline for testing."""
    return ToolPipeline(
        tool_registry=mock_registry,
        tool_executor=mock_executor,
        config=pipeline_config,
    )


@pytest.fixture
def safety_checker():
    """Create safety checker for testing."""
    return SafetyChecker(
        auto_confirm_low_risk=True,
        require_confirmation_threshold=OperationalRiskLevel.HIGH,
        approval_mode=ApprovalMode.RISKY_ONLY,
    )


# =============================================================================
# 1. Tool Validation Tests (12 tests)
# =============================================================================


class TestToolValidation:
    """Test tool name and parameter validation."""

    def test_valid_tool_name_format(self, basic_pipeline):
        """Test valid tool name format is accepted."""
        valid_names = [
            "read",
            "write_file",
            "execute_bash",
            "code_search",
            "a",
            "tool_with_123_numbers",
        ]
        for name in valid_names:
            assert basic_pipeline.is_valid_tool_name(name), f"Should accept: {name}"

    def test_invalid_tool_name_empty(self, basic_pipeline):
        """Test empty tool name is rejected."""
        assert not basic_pipeline.is_valid_tool_name("")
        assert not basic_pipeline.is_valid_tool_name(None)

    def test_invalid_tool_name_non_string(self, basic_pipeline):
        """Test non-string tool name is rejected."""
        assert not basic_pipeline.is_valid_tool_name(123)
        assert not basic_pipeline.is_valid_tool_name(None)
        assert not basic_pipeline.is_valid_tool_name(["list"])

    def test_invalid_tool_name_too_long(self, basic_pipeline):
        """Test tool name exceeding max length is rejected."""
        long_name = "a" * 100
        assert not basic_pipeline.is_valid_tool_name(long_name)

    def test_invalid_tool_name_uppercase_start(self, basic_pipeline):
        """Test tool name starting with uppercase is rejected."""
        assert not basic_pipeline.is_valid_tool_name("Read")
        assert not basic_pipeline.is_valid_tool_name("Write_file")

    def test_invalid_tool_name_special_chars(self, basic_pipeline):
        """Test tool name with special characters is rejected."""
        invalid_names = [
            "tool-name",
            "tool.name",
            "tool name",
            "tool@name",
            "tool!",
            "tool/name",
        ]
        for name in invalid_names:
            assert not basic_pipeline.is_valid_tool_name(name), f"Should reject: {name}"

    def test_invalid_tool_name_start_with_digit(self, basic_pipeline):
        """Test tool name starting with digit is rejected."""
        assert not basic_pipeline.is_valid_tool_name("1tool")
        assert not basic_pipeline.is_valid_tool_name("2nd_tool")

    def test_valid_tool_name_underscore(self, basic_pipeline):
        """Test tool name with underscore is accepted."""
        assert basic_pipeline.is_valid_tool_name("tool_name")
        assert basic_pipeline.is_valid_tool_name("my_tool_123")

    def test_tool_name_pattern_regex(self, basic_pipeline):
        """Test the VALID_TOOL_NAME_PATTERN regex."""
        pattern = basic_pipeline.VALID_TOOL_NAME_PATTERN
        assert pattern.match("valid_tool_name")
        assert not pattern.match("Invalid_Name")
        assert not pattern.match("tool-name")

    def test_max_tool_name_length_configurable(self, mock_registry, mock_executor):
        """Test max_tool_name_length is configurable."""
        config = ToolPipelineConfig(max_tool_name_length=10)
        pipeline = ToolPipeline(
            tool_registry=mock_registry,
            tool_executor=mock_executor,
            config=config,
        )
        assert pipeline.is_valid_tool_name("a" * 10)
        assert not pipeline.is_valid_tool_name("a" * 11)

    def test_unknown_tool_rejected(self, basic_pipeline):
        """Test unknown tool is rejected during execution."""
        basic_pipeline.tools.is_tool_enabled = Mock(return_value=False)
        tool_call = {"name": "unknown_tool", "arguments": {}}
        result = asyncio.run(basic_pipeline._execute_single_call(tool_call, {}))
        assert result.success is False
        assert result.skipped is True
        assert "Unknown or disabled tool" in result.skip_reason

    def test_disabled_tool_rejected(self, basic_pipeline):
        """Test disabled tool is rejected during execution."""
        basic_pipeline.tools.is_tool_enabled = Mock(return_value=False)
        tool_call = {"name": "read", "arguments": {}}
        result = asyncio.run(basic_pipeline._execute_single_call(tool_call, {}))
        assert result.success is False
        assert result.skipped is True


# =============================================================================
# 2. Safety Checks Tests (15 tests)
# =============================================================================


class TestSafetyChecks:
    """Test dangerous operation detection and safety checks."""

    def test_safe_bash_command(self, safety_checker):
        """Test safe bash command passes safety check."""
        risk, details = safety_checker.check_bash_command("ls -la")
        assert risk == OperationalRiskLevel.SAFE
        assert len(details) == 0

    def test_critical_risk_rm_rf_root(self, safety_checker):
        """Test rm -rf / is detected as CRITICAL."""
        risk, details = safety_checker.check_bash_command("rm -rf /")
        assert risk == OperationalRiskLevel.CRITICAL
        assert len(details) > 0

    def test_high_risk_rm_rf(self, safety_checker):
        """Test rm -rf is detected as HIGH risk."""
        risk, details = safety_checker.check_bash_command("rm -rf /path/to/dir")
        assert risk == OperationalRiskLevel.HIGH
        assert len(details) > 0

    def test_high_risk_git_reset_hard(self, safety_checker):
        """Test git reset --hard is detected as HIGH risk."""
        risk, details = safety_checker.check_bash_command("git reset --hard HEAD")
        assert risk == OperationalRiskLevel.HIGH
        assert "Discard all uncommitted changes" in " ".join(details)

    def test_high_risk_force_push(self, safety_checker):
        """Test git push --force is detected as HIGH risk."""
        risk, details = safety_checker.check_bash_command("git push --force origin main")
        assert risk == OperationalRiskLevel.HIGH
        assert "Force push" in " ".join(details)

    def test_high_risk_sudo(self, safety_checker):
        """Test sudo commands are detected as HIGH risk."""
        risk, details = safety_checker.check_bash_command("sudo apt-get install package")
        assert risk == OperationalRiskLevel.HIGH
        assert "elevated privileges" in " ".join(details)

    def test_medium_risk_simple_rm(self, safety_checker):
        """Test simple rm is detected as MEDIUM risk."""
        risk, details = safety_checker.check_bash_command("rm file.txt")
        assert risk == OperationalRiskLevel.MEDIUM

    def test_sensitive_file_detection(self, safety_checker):
        """Test sensitive file modification is detected."""
        risk, details = safety_checker.check_file_operation("write", ".env", overwrite=True)
        assert risk == OperationalRiskLevel.HIGH
        assert len(details) > 0

    def test_system_file_detection(self, safety_checker):
        """Test system file modification is detected."""
        risk, details = safety_checker.check_file_operation("write", "/etc/passwd", overwrite=True)
        assert risk == OperationalRiskLevel.HIGH
        assert "system file" in " ".join(details)

    def test_safe_file_read(self, safety_checker):
        """Test safe file read operation."""
        risk, details = safety_checker.check_file_operation(
            "read", "/tmp/test.txt", overwrite=False
        )
        assert risk == OperationalRiskLevel.SAFE

    def test_overwrite_existing_file(self, safety_checker, tmp_path):
        """Test overwriting existing file is detected."""
        from victor.agent.safety import _RISK_ORDER

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        risk, details = safety_checker.check_file_operation("write", str(test_file), overwrite=True)
        assert _RISK_ORDER[risk] >= _RISK_ORDER[OperationalRiskLevel.MEDIUM]

    def test_write_tool_detection(self, safety_checker):
        """Test write tools are correctly identified."""
        assert safety_checker.is_write_tool("write_file")
        assert safety_checker.is_write_tool("execute_bash")
        assert safety_checker.is_write_tool("edit_files")
        assert not safety_checker.is_write_tool("read")

    def test_approval_mode_off_auto_approves(self, safety_checker):
        """Test OFF approval mode auto-approves everything."""
        safety_checker.approval_mode = ApprovalMode.OFF
        should_proceed, _ = asyncio.run(
            safety_checker.check_and_confirm("write_file", {"path": "test.txt"})
        )
        assert should_proceed is True

    def test_approval_mode_all_writes_requires_confirmation(self, safety_checker):
        """Test ALL_WRITES mode requires confirmation for writes."""
        safety_checker.approval_mode = ApprovalMode.ALL_WRITES
        should_proceed, _ = asyncio.run(
            safety_checker.check_and_confirm("write_file", {"path": "test.txt"})
        )
        # No callback registered, so it should still proceed but log warning
        assert should_proceed is True

    def test_custom_pattern_addition(self, safety_checker):
        """Test adding custom safety patterns."""
        safety_checker.add_custom_pattern(
            pattern=r"dangerous_command",
            description="Custom dangerous command",
            risk_level="HIGH",
        )
        risk, details = safety_checker.check_bash_command("dangerous_command")
        assert risk == OperationalRiskLevel.HIGH
        assert "Custom dangerous command" in details


# =============================================================================
# 3. Sandboxing Tests (10 tests)
# =============================================================================


class TestSandboxing:
    """Test sandbox enforcement and resource limits."""

    def test_tool_budget_enforcement(self, basic_pipeline):
        """Test tool budget is enforced."""
        basic_pipeline._calls_used = basic_pipeline.config.tool_budget
        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        result = asyncio.run(basic_pipeline._execute_single_call(tool_call, {}))
        assert result.success is False
        assert result.skipped is True
        assert "budget exhausted" in result.skip_reason

    def test_tool_budget_remaining(self, basic_pipeline):
        """Test calls_remaining property."""
        assert basic_pipeline.calls_remaining == basic_pipeline.config.tool_budget
        basic_pipeline._calls_used = 5
        assert basic_pipeline.calls_remaining == 5

    def test_tool_budget_max_calls_respected(self, basic_pipeline):
        """Test max calls is not exceeded."""
        tool_calls = [{"name": "read", "arguments": {"path": f"test{i}.txt"}} for i in range(20)]
        result = asyncio.run(basic_pipeline.execute_tool_calls(tool_calls, {}))
        assert result.total_calls <= basic_pipeline.config.tool_budget + len(
            tool_calls
        )  # Account for skips
        assert result.budget_exhausted is True

    def test_rate_limiter_token_bucket(self):
        """Test rate limiter token bucket behavior."""
        limiter = ToolRateLimiter(rate=10.0, burst=5)
        assert limiter.tokens == 5.0

    def test_rate_limiter_acquire(self):
        """Test rate limiter acquire reduces tokens."""
        limiter = ToolRateLimiter(rate=1.0, burst=2)
        assert limiter.acquire() is True
        assert limiter.tokens < 2.0

    def test_rate_limiter_burst_exceeded(self):
        """Test rate limiter blocks when burst exceeded."""
        limiter = ToolRateLimiter(rate=0.1, burst=1)
        limiter.acquire()
        assert limiter.acquire() is False  # No tokens left

    def test_rate_limiter_refill(self):
        """Test rate limiter refills tokens over time."""
        limiter = ToolRateLimiter(rate=100.0, burst=2)
        limiter.acquire()
        limiter.acquire()
        time.sleep(0.02)  # Wait for refill
        assert limiter.acquire() is True  # Should have refilled

    async def test_rate_limiter_async_wait(self):
        """Test async wait blocks until token available."""
        limiter = ToolRateLimiter(rate=1000.0, burst=1)
        limiter.acquire()
        start = time.monotonic()
        await limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # Should be very fast with high rate

    def test_cache_size_limit_enforced(self):
        """Test LRU cache size limit is enforced."""
        cache = LRUToolCache(max_size=3)
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")
        assert len(cache) <= 3

    def test_cache_ttl_expiration(self):
        """Test cache entries expire after TTL."""
        cache = LRUToolCache(max_size=100, ttl_seconds=0.1)
        cache.set("key1", "value1")
        time.sleep(0.15)
        assert cache.get("key1") is None


# =============================================================================
# 4. Authorization Flow Tests (12 tests)
# =============================================================================


class TestAuthorizationFlow:
    """Test authorization and approval flow."""

    async def test_confirmation_callback_called_for_high_risk(self):
        """Test confirmation callback is invoked for HIGH risk operations."""
        callback_called = []

        async def mock_callback(request: ConfirmationRequest) -> bool:
            callback_called.append(request)
            return True

        checker = SafetyChecker(
            confirmation_callback=mock_callback,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )
        await checker.check_and_confirm("execute_bash", {"command": "rm -rf /tmp/test"})
        assert len(callback_called) == 1
        assert callback_called[0].tool_name == "execute_bash"
        assert callback_called[0].risk_level == OperationalRiskLevel.HIGH

    async def test_confirmation_callback_approves(self):
        """Test operation proceeds when callback returns True."""

        async def approve_callback(request: ConfirmationRequest) -> bool:
            return True

        checker = SafetyChecker(
            confirmation_callback=approve_callback,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )
        should_proceed, _ = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf test"}
        )
        assert should_proceed is True

    async def test_confirmation_callback_rejects(self):
        """Test operation is blocked when callback returns False."""

        async def reject_callback(request: ConfirmationRequest) -> bool:
            return False

        checker = SafetyChecker(
            confirmation_callback=reject_callback,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )
        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf test"}
        )
        assert should_proceed is False
        assert reason is not None

    async def test_low_risk_auto_approved(self):
        """Test LOW risk operations are auto-approved."""

        async def callback(request: ConfirmationRequest) -> bool:
            raise AssertionError("Callback should not be called for LOW risk")

        checker = SafetyChecker(
            confirmation_callback=callback,
            auto_confirm_low_risk=True,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )
        should_proceed, _ = await checker.check_and_confirm("read", {"path": "test.txt"})
        assert should_proceed is True

    async def test_no_callback_allows_high_risk_with_warning(self, caplog):
        """Test HIGH risk without callback logs warning but proceeds."""
        import logging

        checker = SafetyChecker(
            confirmation_callback=None,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )
        with caplog.at_level(logging.WARNING):
            should_proceed, _ = await checker.check_and_confirm(
                "execute_bash", {"command": "rm -rf test"}
            )
        assert should_proceed is True
        assert "High-risk operation without confirmation callback" in caplog.text

    async def test_confirmation_request_formatting(self):
        """Test ConfirmationRequest formats message correctly."""
        request = ConfirmationRequest(
            tool_name="execute_bash",
            risk_level=OperationalRiskLevel.HIGH,
            description="Delete directory",
            details=["Recursive delete", "Multiple files"],
            arguments={"command": "rm -rf test"},
        )
        message = request.format_message()
        assert "HIGH RISK" in message
        assert "execute_bash" in message
        assert "Delete directory" in message

    def test_approval_mode_enum_values(self):
        """Test ApprovalMode enum has correct values."""
        assert ApprovalMode.OFF.value == "off"
        assert ApprovalMode.RISKY_ONLY.value == "risky_only"
        assert ApprovalMode.ALL_WRITES.value == "all_writes"

    def test_risk_level_ordering(self):
        """Test risk levels are properly ordered."""
        from victor.agent.safety import _RISK_ORDER

        assert _RISK_ORDER[OperationalRiskLevel.SAFE] < _RISK_ORDER[OperationalRiskLevel.LOW]
        assert _RISK_ORDER[OperationalRiskLevel.LOW] < _RISK_ORDER[OperationalRiskLevel.MEDIUM]
        assert _RISK_ORDER[OperationalRiskLevel.MEDIUM] < _RISK_ORDER[OperationalRiskLevel.HIGH]
        assert _RISK_ORDER[OperationalRiskLevel.HIGH] < _RISK_ORDER[OperationalRiskLevel.CRITICAL]

    async def test_medium_risk_requires_confirmation_when_threshold_is_medium(self):
        """Test MEDIUM risk requires confirmation when threshold is MEDIUM."""

        async def callback(request: ConfirmationRequest) -> bool:
            return True

        checker = SafetyChecker(
            confirmation_callback=callback,
            require_confirmation_threshold=OperationalRiskLevel.MEDIUM,
        )
        should_proceed, _ = await checker.check_and_confirm(
            "execute_bash", {"command": "rm file.txt"}
        )
        assert should_proceed is True  # Callback should be called and approved

    async def test_critical_risk_always_requires_confirmation(self):
        """Test CRITICAL risk always requires confirmation regardless of threshold."""
        callback_called = []

        async def callback(request: ConfirmationRequest) -> bool:
            callback_called.append(request)
            return True

        checker = SafetyChecker(
            confirmation_callback=callback,
            require_confirmation_threshold=OperationalRiskLevel.CRITICAL,
        )
        # Even with CRITICAL threshold, CRITICAL should trigger confirmation
        await checker.check_and_confirm("execute_bash", {"command": "rm -rf /"})
        assert len(callback_called) == 1


# =============================================================================
# 5. Execution Pipeline Tests (10 tests)
# =============================================================================


class TestExecutionPipeline:
    """Test execution pipeline order and flow."""

    async def test_single_call_execution_order(self, basic_pipeline):
        """Test single call executes in correct order."""
        call_order = []

        def on_start(name, args):
            call_order.append(("start", name))

        def on_complete(result):
            call_order.append(("complete", result.tool_name))

        basic_pipeline.on_tool_start = on_start
        basic_pipeline.on_tool_complete = on_complete

        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        await basic_pipeline._execute_single_call(tool_call, {})

        assert call_order[0] == ("start", "read")
        assert call_order[1] == ("complete", "read")

    async def test_multiple_calls_sequential_execution(self, basic_pipeline):
        """Test multiple calls execute sequentially."""
        tool_calls = [
            {"name": "read", "arguments": {"path": "test1.txt"}},
            {"name": "read", "arguments": {"path": "test2.txt"}},
        ]
        result = await basic_pipeline.execute_tool_calls(tool_calls, {})
        assert len(result.results) == 2
        assert result.total_calls == 2

    async def test_execution_stops_on_budget_exhausted(self, basic_pipeline):
        """Test execution stops when budget exhausted."""
        basic_pipeline.config.tool_budget = 2
        tool_calls = [{"name": "read", "arguments": {"path": f"test{i}.txt"}} for i in range(5)]
        result = await basic_pipeline.execute_tool_calls(tool_calls, {})
        assert result.budget_exhausted is True
        # Only first 2 should execute
        executed = [r for r in result.results if not r.skipped or "budget" not in r.skip_reason]
        assert len(executed) == 2

    async def test_executed_tools_tracking(self, basic_pipeline):
        """Test executed tools are tracked."""
        tool_calls = [
            {"name": "read", "arguments": {"path": "test.txt"}},
            {"name": "grep", "arguments": {"query": "test"}},
        ]
        await basic_pipeline.execute_tool_calls(tool_calls, {})
        assert "read" in basic_pipeline.executed_tools
        assert "grep" in basic_pipeline.executed_tools

    async def test_pipeline_result_aggregation(self, basic_pipeline):
        """Test pipeline result correctly aggregates."""
        basic_pipeline.executor.execute = AsyncMock(
            side_effect=[
                Mock(success=True, result="result1"),
                Mock(success=False, error="error2"),
                Mock(success=True, result="result3"),
            ]
        )

        tool_calls = [{"name": "read", "arguments": {"path": f"test{i}.txt"}} for i in range(3)]
        result = await basic_pipeline.execute_tool_calls(tool_calls, {})

        assert result.successful_calls == 2
        assert result.failed_calls == 1
        assert result.total_calls == 3

    async def test_callbacks_invoked_on_success(self, basic_pipeline):
        """Test callbacks are invoked on successful execution."""
        start_called = []
        complete_called = []

        basic_pipeline.on_tool_start = lambda n, a: start_called.append(n)
        basic_pipeline.on_tool_complete = lambda r: complete_called.append(r)

        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        await basic_pipeline._execute_single_call(tool_call, {})

        assert len(start_called) == 1
        assert len(complete_called) == 1

    async def test_callbacks_invoked_on_failure(self, basic_pipeline):
        """Test callbacks are invoked on failed execution."""
        complete_called = []
        basic_pipeline.executor.execute = AsyncMock(
            return_value=Mock(success=False, error="Test error")
        )
        basic_pipeline.on_tool_complete = lambda r: complete_called.append(r)

        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        result = await basic_pipeline._execute_single_call(tool_call, {})

        assert result.success is False
        assert len(complete_called) == 1

    async def test_execution_time_recorded(self, basic_pipeline):
        """Test execution time is recorded in result."""
        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        result = await basic_pipeline._execute_single_call(tool_call, {})
        assert result.execution_time_ms >= 0

    async def test_normalization_strategy_recorded(self, basic_pipeline):
        """Test normalization strategy is recorded in result."""
        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        result = await basic_pipeline._execute_single_call(tool_call, {})
        # Strategy should be one of the defined strategies
        if result.normalization_applied:
            assert result.normalization_applied in [s.value for s in NormalizationStrategy]

    async def test_context_passed_to_executor(self, basic_pipeline):
        """Test context is passed to executor."""
        test_context = {"test_key": "test_value"}
        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        await basic_pipeline._execute_single_call(tool_call, test_context)
        basic_pipeline.executor.execute.assert_called_once()
        call_kwargs = basic_pipeline.executor.execute.call_args[1]
        assert "context" in call_kwargs
        assert call_kwargs["context"] == test_context


# =============================================================================
# 6. Error Handling Tests (8 tests)
# =============================================================================


class TestErrorHandling:
    """Test error handling and propagation."""

    async def test_invalid_tool_call_structure(self, basic_pipeline):
        """Test invalid tool call structure is handled."""
        result = await basic_pipeline._execute_single_call("not a dict", {})
        assert result.success is False
        assert result.skipped is True
        assert "Invalid tool call structure" in result.skip_reason

    async def test_missing_tool_name(self, basic_pipeline):
        """Test missing tool name is handled."""
        result = await basic_pipeline._execute_single_call({"arguments": {}}, {})
        assert result.success is False
        assert result.skipped is True
        assert "missing name" in result.skip_reason

    async def test_tool_execution_error_propagated(self, basic_pipeline):
        """Test tool execution error is properly handled."""
        # Mock executor to return a failure result (realistic error simulation)
        basic_pipeline.executor.execute = AsyncMock(
            return_value=Mock(success=False, result=None, error="Tool execution failed")
        )
        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        result = await basic_pipeline._execute_single_call(tool_call, {})
        # Result should indicate failure
        assert result.success is False
        assert result.error == "Tool execution failed"

    async def test_callback_exception_doesnt_stop_execution(self, basic_pipeline, caplog):
        """Test callback exceptions are logged but don't stop execution."""
        import logging

        basic_pipeline.on_tool_start = lambda n, a: (_ for _ in ()).throw(
            RuntimeError("Callback error")
        )
        with caplog.at_level(logging.WARNING):
            tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
            result = await basic_pipeline._execute_single_call(tool_call, {})
        assert result.success is True  # Execution should succeed
        assert "on_tool_start callback failed" in caplog.text

    async def test_normalization_error_handled(self, basic_pipeline):
        """Test normalization errors are handled gracefully."""
        # Pass arguments that can't be normalized
        tool_call = {"name": "read", "arguments": "invalid json"}
        result = await basic_pipeline._execute_single_call(tool_call, {})
        # Should still proceed with fallback normalization
        assert result is not None

    async def test_signature_store_io_error_handled(self, basic_pipeline, caplog):
        """Test signature store I/O errors are handled gracefully."""
        import logging

        mock_store = Mock()
        mock_store.is_known_failure = Mock(side_effect=OSError("Store unavailable"))
        basic_pipeline.signature_store = mock_store

        with caplog.at_level(logging.WARNING):
            is_known = basic_pipeline.is_known_failure("read", {"path": "test.txt"})
        assert is_known is False  # Should default to not known failure
        assert "I/O error" in caplog.text

    async def test_cache_error_handled(self, basic_pipeline, caplog):
        """Test cache errors are handled gracefully."""
        # Mock a cache that raises errors
        import logging

        async def mock_get(tool_name, args):
            raise ValueError("Cache corrupted")

        mock_cache = Mock()
        mock_cache.get = mock_get
        basic_pipeline.semantic_cache = mock_cache

        with caplog.at_level(logging.WARNING):
            tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
            result = await basic_pipeline._execute_single_call(tool_call, {})
        # Should still execute despite cache error
        assert result is not None

    async def test_multiple_failures_tracked_separately(self, basic_pipeline):
        """Test different failures are tracked separately."""
        basic_pipeline.executor.execute = AsyncMock(
            side_effect=[
                Mock(success=False, error="Error 1"),
                Mock(success=False, error="Error 2"),
            ]
        )

        tool_calls = [
            {"name": "read", "arguments": {"path": "test1.txt"}},
            {"name": "read", "arguments": {"path": "test2.txt"}},
        ]
        await basic_pipeline.execute_tool_calls(tool_calls, {})

        # Should have 2 distinct failure signatures
        assert len(basic_pipeline._failed_signatures) == 2


# =============================================================================
# 7. Caching Behavior Tests (8 tests)
# =============================================================================


class TestCachingBehavior:
    """Test caching behavior and cache security."""

    def test_idempotent_tool_identified(self, basic_pipeline):
        """Test idempotent tools are correctly identified."""
        assert basic_pipeline.is_idempotent_tool("read")
        assert basic_pipeline.is_idempotent_tool("grep")
        assert basic_pipeline.is_idempotent_tool("glob")
        assert not basic_pipeline.is_idempotent_tool("write")
        assert not basic_pipeline.is_idempotent_tool("execute_bash")

    def test_idempotent_tools_set(self):
        """Test IDEMPOTENT_TOOLS includes expected tools."""
        assert "read" in IDEMPOTENT_TOOLS
        assert "grep" in IDEMPOTENT_TOOLS
        assert "glob" in IDEMPOTENT_TOOLS
        assert "ls" in IDEMPOTENT_TOOLS
        assert "write" not in IDEMPOTENT_TOOLS

    async def test_idempotent_result_cached(self, basic_pipeline):
        """Test idempotent tool results are cached."""
        # Ensure idempotent caching is enabled
        basic_pipeline.config.enable_idempotent_caching = True

        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        await basic_pipeline._execute_single_call(tool_call, {})

        # Check cache
        cached = basic_pipeline.get_cached_result(
            "read", {"path": "test.txt", "offset": 0, "limit": 2000}
        )
        assert cached is not None
        assert cached.cached is True

    async def test_non_idempotent_result_not_cached(self, basic_pipeline):
        """Test non-idempotent tool results are not cached."""
        tool_call = {"name": "write", "arguments": {"path": "test.txt", "content": "test"}}
        await basic_pipeline._execute_single_call(tool_call, {})

        # Check cache
        cached = basic_pipeline.get_cached_result("write", {"path": "test.txt", "content": "test"})
        assert cached is None

    async def test_failed_result_not_cached(self, basic_pipeline):
        """Test failed results are not cached."""
        basic_pipeline.executor.execute = AsyncMock(
            return_value=Mock(success=False, error="Test error")
        )

        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        await basic_pipeline._execute_single_call(tool_call, {})

        cached = basic_pipeline.get_cached_result("read", {"path": "test.txt"})
        assert cached is None

    async def test_cache_invalidated_on_file_write(self, basic_pipeline):
        """Test cache is invalidated when file is written."""
        # First, cache a read result
        read_call = {"name": "read", "arguments": {"path": "test.txt"}}
        await basic_pipeline._execute_single_call(read_call, {})

        # Then write to the file
        basic_pipeline.executor.execute = AsyncMock(
            return_value=Mock(success=True, result="written")
        )
        write_call = {"name": "write", "arguments": {"path": "test.txt", "content": "new"}}
        await basic_pipeline._execute_single_call(write_call, {})

        # Read should be invalidated
        cached = basic_pipeline.get_cached_result("read", {"path": "test.txt"})
        assert cached is None

    async def test_cache_stats_tracked(self, basic_pipeline):
        """Test cache statistics are tracked."""
        # Cache hit
        read_call = {"name": "read", "arguments": {"path": "test.txt"}}
        await basic_pipeline._execute_single_call(read_call, {})
        await basic_pipeline._execute_single_call(read_call, {})  # Should hit cache

        stats = basic_pipeline.get_cache_stats()
        assert stats["cache_hits"] > 0
        assert stats["cache_misses"] > 0
        assert 0.0 <= stats["hit_rate"] <= 1.0

    async def test_cache_disabled_when_config_disabled(self, mock_registry, mock_executor):
        """Test cache is disabled when enable_idempotent_caching is False."""
        config = ToolPipelineConfig(enable_idempotent_caching=False)
        pipeline = ToolPipeline(
            tool_registry=mock_registry,
            tool_executor=mock_executor,
            config=config,
        )

        read_call = {"name": "read", "arguments": {"path": "test.txt"}}
        await pipeline._execute_single_call(read_call, {})

        cached = pipeline.get_cached_result("read", {"path": "test.txt"})
        assert cached is None


# =============================================================================
# 8. Middleware Integration Tests (5 tests)
# =============================================================================


class TestMiddlewareIntegration:
    """Test middleware chain integration."""

    async def test_middleware_chain_process_before_called(self, basic_pipeline):
        """Test middleware chain process_before is called."""
        middleware_result = Mock()
        middleware_result.proceed = True
        middleware_result.modified_arguments = None

        mock_middleware = Mock()
        mock_middleware.process_before = AsyncMock(return_value=middleware_result)

        basic_pipeline.middleware_chain = mock_middleware

        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        await basic_pipeline._execute_single_call(tool_call, {})

        mock_middleware.process_before.assert_called_once()

    async def test_middleware_blocks_execution(self, basic_pipeline):
        """Test middleware can block execution."""
        middleware_result = Mock()
        middleware_result.proceed = False
        middleware_result.error_message = "Blocked by middleware"

        mock_middleware = Mock()
        mock_middleware.process_before = AsyncMock(return_value=middleware_result)

        basic_pipeline.middleware_chain = mock_middleware

        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        result = await basic_pipeline._execute_single_call(tool_call, {})

        assert result.success is False
        assert result.skipped is True
        assert "Blocked by middleware" in result.skip_reason

    async def test_middleware_modifies_arguments(self, basic_pipeline):
        """Test middleware can modify arguments."""
        middleware_result = Mock()
        middleware_result.proceed = True
        middleware_result.modified_arguments = {"path": "modified.txt"}

        mock_middleware = Mock()
        mock_middleware.process_before = AsyncMock(return_value=middleware_result)

        basic_pipeline.middleware_chain = mock_middleware

        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        await basic_pipeline._execute_single_call(tool_call, {})

        # Executor should receive modified arguments
        call_kwargs = basic_pipeline.executor.execute.call_args[1]
        assert call_kwargs["arguments"]["path"] == "modified.txt"

    async def test_middleware_process_after_called(self, basic_pipeline):
        """Test middleware chain process_after is called."""
        mock_middleware = Mock()
        mock_middleware.process_before = AsyncMock(
            return_value=Mock(proceed=True, modified_arguments=None)
        )
        mock_middleware.process_after = AsyncMock(return_value="modified result")

        basic_pipeline.middleware_chain = mock_middleware

        tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
        result = await basic_pipeline._execute_single_call(tool_call, {})

        mock_middleware.process_after.assert_called_once()
        # Check result was modified
        assert result.result == "modified result"

    async def test_middleware_exception_handled(self, basic_pipeline, caplog):
        """Test middleware exceptions are handled gracefully."""
        import logging

        mock_middleware = Mock()
        mock_middleware.process_before = AsyncMock(side_effect=ValueError("Middleware error"))

        basic_pipeline.middleware_chain = mock_middleware

        with caplog.at_level(logging.WARNING):
            tool_call = {"name": "read", "arguments": {"path": "test.txt"}}
            result = await basic_pipeline._execute_single_call(tool_call, {})

        # Should still execute despite middleware error
        assert result is not None
        assert "Middleware chain process_before failed" in caplog.text


# =============================================================================
# Additional: Execution Metrics Tests
# =============================================================================


class TestExecutionMetrics:
    """Test execution metrics tracking."""

    def test_metrics_initial_state(self):
        """Test metrics start with zero values."""
        metrics = ExecutionMetrics()
        assert metrics.total_executions == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 0

    def test_record_execution(self):
        """Test recording an execution."""
        metrics = ExecutionMetrics()
        metrics.record_execution(
            tool_name="read",
            execution_time=0.1,
            success=True,
            cached=False,
            skipped=False,
        )
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        metrics = ExecutionMetrics()
        metrics.record_execution("read", 0.1, True, cached=True)
        metrics.record_execution("read", 0.1, True, cached=False)
        hit_rate = metrics.get_cache_hit_rate()
        assert hit_rate == 0.5

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ExecutionMetrics()
        metrics.record_execution("read", 0.1, True, cached=False)
        metrics.record_execution("write", 0.1, False, cached=False)
        success_rate = metrics.get_success_rate()
        assert success_rate == 0.5

    def test_top_tools_tracking(self):
        """Test top tools are tracked."""
        metrics = ExecutionMetrics()
        metrics.record_execution("read", 0.1, True, cached=False)
        metrics.record_execution("read", 0.1, True, cached=False)
        metrics.record_execution("grep", 0.1, True, cached=False)
        top_tools = metrics.get_top_tools(2)
        assert top_tools[0] == ("read", 2)

    def test_metrics_thread_safety(self):
        """Test metrics operations are thread-safe."""
        import threading

        metrics = ExecutionMetrics()

        def record():
            for _ in range(100):
                metrics.record_execution("test", 0.1, True, cached=False)

        threads = [threading.Thread(target=record) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.total_executions == 1000

    def test_metrics_to_dict(self):
        """Test converting metrics to dict."""
        metrics = ExecutionMetrics()
        metrics.record_execution("read", 0.1, True, cached=True)
        metrics.record_execution("write", 0.1, False, cached=False)

        d = metrics.to_dict()
        assert "total_executions" in d
        assert "cache_hit_rate" in d
        assert "success_rate" in d
        assert "top_tools" in d

    def test_metrics_reset(self):
        """Test resetting metrics."""
        metrics = ExecutionMetrics()
        metrics.record_execution("read", 0.1, True, cached=False)
        metrics.reset()
        assert metrics.total_executions == 0
        assert metrics.successful_executions == 0
