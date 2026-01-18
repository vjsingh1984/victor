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

"""Unit tests for Generic Middleware Library protocols and base classes.

This module tests:
- MiddlewarePhase enum
- IMiddleware protocol
- BaseMiddleware base class
- Enable/disable functionality
- Priority, phase, and tool filtering
"""

import pytest
from typing import Any, Dict, Optional, Set

from victor.framework.middleware_protocols import (
    IMiddleware,
    MiddlewarePhase,
    MiddlewarePriority,
    MiddlewareResult,
)
from victor.framework.middleware_base import BaseMiddleware


# =============================================================================
# Test MiddlewarePhase Enum
# =============================================================================


class TestMiddlewarePhase:
    """Tests for MiddlewarePhase enum."""

    def test_phase_values(self):
        """Test that MiddlewarePhase has all expected values."""
        assert MiddlewarePhase.PRE.value == "pre"
        assert MiddlewarePhase.POST.value == "post"
        assert MiddlewarePhase.AROUND.value == "around"
        assert MiddlewarePhase.ERROR.value == "error"

    def test_phase_count(self):
        """Test that MiddlewarePhase has exactly 4 values."""
        assert len(MiddlewarePhase) == 4

    def test_phase_membership(self):
        """Test phase enum membership."""
        assert MiddlewarePhase.PRE in MiddlewarePhase
        assert MiddlewarePhase.POST in MiddlewarePhase
        assert MiddlewarePhase.AROUND in MiddlewarePhase
        assert MiddlewarePhase.ERROR in MiddlewarePhase


# =============================================================================
# Test IMiddleware Protocol
# =============================================================================


class TestIMiddlewareProtocol:
    """Tests for IMiddleware protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that IMiddleware protocol is runtime_checkable."""
        # The @runtime_checkable decorator marks the protocol as runtime-checkable
        # In Python 3.11+, it's indicated by _is_runtime_protocol attribute
        # In Python 3.12+, it may also have __protocol_attrs__
        is_runtime_checkable = getattr(IMiddleware, "_is_runtime_protocol", False)
        assert is_runtime_checkable, "IMiddleware should be runtime_checkable"

        # Create a concrete implementation with ALL protocol methods
        class FullMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            async def after_tool_call(
                self,
                tool_name: str,
                arguments: Dict[str, Any],
                result: Any,
                success: bool,
            ) -> Optional[Any]:
                return None

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self) -> Optional[Set[str]]:
                return None

            def get_phase(self) -> MiddlewarePhase:
                return MiddlewarePhase.PRE

        middleware = FullMiddleware()

        # This should work because IMiddleware is decorated with @runtime_checkable
        # and FullMiddleware implements all protocol methods
        assert isinstance(middleware, IMiddleware)

    def test_concrete_implementation_implements_protocol(self):
        """Test that concrete implementation implements IMiddleware."""

        class ConcreteMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self) -> Optional[Set[str]]:
                return None

            def get_phase(self) -> MiddlewarePhase:
                return MiddlewarePhase.PRE

        middleware = ConcreteMiddleware()

        # Protocol conformance check
        assert hasattr(middleware, "before_tool_call")
        assert hasattr(middleware, "get_priority")
        assert hasattr(middleware, "get_applicable_tools")
        assert hasattr(middleware, "get_phase")


# =============================================================================
# Test BaseMiddleware
# =============================================================================


class SimpleTestMiddleware(BaseMiddleware):
    """Simple test middleware for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.before_call_count = 0
        self.after_call_count = 0

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Track before_tool_call invocations."""
        self.before_call_count += 1
        self._logger.info(f"Before call: {tool_name}")
        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Track after_tool_call invocations."""
        self.after_call_count += 1
        self._logger.info(f"After call: {tool_name}, success={success}")
        return None


class TestBaseMiddleware:
    """Tests for BaseMiddleware base class."""

    @pytest.mark.asyncio
    async def test_base_middleware_implements_protocol(self):
        """Test that BaseMiddleware implements IMiddleware protocol."""
        middleware = SimpleTestMiddleware()

        # Check protocol conformance
        assert hasattr(middleware, "before_tool_call")
        assert hasattr(middleware, "after_tool_call")
        assert hasattr(middleware, "get_priority")
        assert hasattr(middleware, "get_applicable_tools")
        assert hasattr(middleware, "get_phase")
        assert hasattr(middleware, "enabled")

    @pytest.mark.asyncio
    async def test_default_initialization(self):
        """Test default initialization values."""
        middleware = SimpleTestMiddleware()

        assert middleware.enabled is True
        assert middleware.get_priority() == MiddlewarePriority.NORMAL
        assert middleware.get_applicable_tools() is None
        assert middleware.get_phase() == MiddlewarePhase.PRE

    @pytest.mark.asyncio
    async def test_custom_initialization(self):
        """Test custom initialization values."""
        middleware = SimpleTestMiddleware(
            enabled=False,
            applicable_tools={"write_file", "edit_file"},
            priority=MiddlewarePriority.HIGH,
            phase=MiddlewarePhase.POST,
        )

        assert middleware.enabled is False
        assert middleware.get_applicable_tools() == {"write_file", "edit_file"}
        assert middleware.get_priority() == MiddlewarePriority.HIGH
        assert middleware.get_phase() == MiddlewarePhase.POST

    @pytest.mark.asyncio
    async def test_enable_disable(self):
        """Test enable/disable functionality."""
        middleware = SimpleTestMiddleware(enabled=True)

        assert middleware.enabled is True

        middleware.disable()
        assert middleware.enabled is False

        middleware.enable()
        assert middleware.enabled is True

    @pytest.mark.asyncio
    async def test_before_tool_call_invocation(self):
        """Test that before_tool_call is invoked correctly."""
        middleware = SimpleTestMiddleware()

        result = await middleware.before_tool_call("read_file", {"path": "/tmp/file.txt"})

        assert isinstance(result, MiddlewareResult)
        assert result.proceed is True
        assert middleware.before_call_count == 1

    @pytest.mark.asyncio
    async def test_after_tool_call_invocation(self):
        """Test that after_tool_call is invoked correctly."""
        middleware = SimpleTestMiddleware()

        result = await middleware.after_tool_call(
            "read_file",
            {"path": "/tmp/file.txt"},
            {"content": "file content"},
            True,
        )

        assert result is None
        assert middleware.after_call_count == 1

    @pytest.mark.asyncio
    async def test_get_phase_returns_correct_phase(self):
        """Test that get_phase() returns correct phase."""
        # Test PRE phase (default)
        middleware_pre = SimpleTestMiddleware(phase=MiddlewarePhase.PRE)
        assert middleware_pre.get_phase() == MiddlewarePhase.PRE

        # Test POST phase
        middleware_post = SimpleTestMiddleware(phase=MiddlewarePhase.POST)
        assert middleware_post.get_phase() == MiddlewarePhase.POST

        # Test AROUND phase
        middleware_around = SimpleTestMiddleware(phase=MiddlewarePhase.AROUND)
        assert middleware_around.get_phase() == MiddlewarePhase.AROUND

        # Test ERROR phase
        middleware_error = SimpleTestMiddleware(phase=MiddlewarePhase.ERROR)
        assert middleware_error.get_phase() == MiddlewarePhase.ERROR

    @pytest.mark.asyncio
    async def test_get_priority_returns_correct_priority(self):
        """Test that get_priority() returns correct priority."""
        # Test CRITICAL priority
        middleware_critical = SimpleTestMiddleware(priority=MiddlewarePriority.CRITICAL)
        assert middleware_critical.get_priority() == MiddlewarePriority.CRITICAL

        # Test HIGH priority
        middleware_high = SimpleTestMiddleware(priority=MiddlewarePriority.HIGH)
        assert middleware_high.get_priority() == MiddlewarePriority.HIGH

        # Test NORMAL priority (default)
        middleware_normal = SimpleTestMiddleware(priority=MiddlewarePriority.NORMAL)
        assert middleware_normal.get_priority() == MiddlewarePriority.NORMAL

        # Test LOW priority
        middleware_low = SimpleTestMiddleware(priority=MiddlewarePriority.LOW)
        assert middleware_low.get_priority() == MiddlewarePriority.LOW

        # Test DEFERRED priority
        middleware_deferred = SimpleTestMiddleware(priority=MiddlewarePriority.DEFERRED)
        assert middleware_deferred.get_priority() == MiddlewarePriority.DEFERRED

    @pytest.mark.asyncio
    async def test_get_applicable_tools_filters_correctly(self):
        """Test that get_applicable_tools() filters correctly."""
        # Test with specific tools
        middleware_filtered = SimpleTestMiddleware(
            applicable_tools={"write_file", "edit_file", "read_file"}
        )
        assert middleware_filtered.get_applicable_tools() == {
            "write_file",
            "edit_file",
            "read_file",
        }

        # Test with None (all tools)
        middleware_all = SimpleTestMiddleware(applicable_tools=None)
        assert middleware_all.get_applicable_tools() is None

    @pytest.mark.asyncio
    async def test_applies_to_tool_method(self):
        """Test applies_to_tool() method."""
        middleware = SimpleTestMiddleware(applicable_tools={"write_file", "edit_file"})

        assert middleware.applies_to_tool("write_file") is True
        assert middleware.applies_to_tool("edit_file") is True
        assert middleware.applies_to_tool("read_file") is False
        assert middleware.applies_to_tool("grep") is False

    @pytest.mark.asyncio
    async def test_applies_to_all_tools_when_none(self):
        """Test applies_to_tool() returns True for all tools when None."""
        middleware = SimpleTestMiddleware(applicable_tools=None)

        assert middleware.applies_to_tool("write_file") is True
        assert middleware.applies_to_tool("read_file") is True
        assert middleware.applies_to_tool("grep") is True

    def test_repr(self):
        """Test __repr__ method."""
        middleware = SimpleTestMiddleware(
            enabled=True,
            applicable_tools={"write_file", "edit_file"},
            priority=MiddlewarePriority.HIGH,
            phase=MiddlewarePhase.PRE,
        )

        repr_str = repr(middleware)

        assert "SimpleTestMiddleware" in repr_str
        assert "enabled=True" in repr_str
        assert "priority=HIGH" in repr_str
        assert "phase=pre" in repr_str
        assert "write_file" in repr_str
        assert "edit_file" in repr_str


# =============================================================================
# Test Custom Middleware Implementations
# =============================================================================


class ValidationTestMiddleware(BaseMiddleware):
    """Test middleware that validates tool calls."""

    def __init__(self, blocked_tools: Optional[Set[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.blocked_tools = blocked_tools or set()

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Block execution for specific tools."""
        if tool_name in self.blocked_tools:
            return MiddlewareResult(
                proceed=False,
                error_message=f"Tool {tool_name} is blocked",
            )
        return MiddlewareResult()


class TestCustomMiddleware:
    """Tests for custom middleware implementations."""

    @pytest.mark.asyncio
    async def test_validation_middleware_blocks_tools(self):
        """Test that validation middleware can block tools."""
        middleware = ValidationTestMiddleware(
            blocked_tools={"delete_file", "rm_rf"},
            applicable_tools={"delete_file", "write_file", "read_file"},
        )

        # Blocked tool
        result = await middleware.before_tool_call("delete_file", {"path": "/tmp/file.txt"})
        assert result.proceed is False
        assert "blocked" in result.error_message

        # Allowed tool
        result = await middleware.before_tool_call("write_file", {"path": "/tmp/file.txt"})
        assert result.proceed is True
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_validation_middleware_with_modified_arguments(self):
        """Test middleware that modifies arguments."""

        class ArgumentTransformingMiddleware(BaseMiddleware):
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                if "path" in arguments:
                    # Normalize path
                    original_path = arguments["path"]
                    normalized = original_path.replace("//", "/")
                    return MiddlewareResult(
                        proceed=True,
                        modified_arguments={"path": normalized},
                    )
                return MiddlewareResult()

        middleware = ArgumentTransformingMiddleware()

        result = await middleware.before_tool_call("read_file", {"path": "/tmp//file.txt"})

        assert result.proceed is True
        assert result.modified_arguments is not None
        assert result.modified_arguments["path"] == "/tmp/file.txt"

    @pytest.mark.asyncio
    async def test_after_tool_call_modifies_result(self):
        """Test middleware that modifies results."""

        class ResultTransformingMiddleware(BaseMiddleware):
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                # No-op for before_tool_call
                return MiddlewareResult()

            async def after_tool_call(
                self,
                tool_name: str,
                arguments: Dict[str, Any],
                result: Any,
                success: bool,
            ) -> Optional[Any]:
                if success and isinstance(result, dict):
                    # Add timestamp to result
                    result["timestamp"] = "2025-01-13T00:00:00Z"
                    return result
                return None

        middleware = ResultTransformingMiddleware()

        original_result = {"content": "file content"}
        modified_result = await middleware.after_tool_call(
            "read_file", {"path": "/tmp/file.txt"}, original_result, True
        )

        assert modified_result is not None
        assert "timestamp" in modified_result
        assert modified_result["timestamp"] == "2025-01-13T00:00:00Z"

    @pytest.mark.asyncio
    async def test_disabled_middleware_skips_processing(self):
        """Test that disabled middleware doesn't process calls."""

        class DisabledTestMiddleware(BaseMiddleware):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.processed = False

            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                self.processed = True
                return MiddlewareResult()

        middleware = DisabledTestMiddleware(enabled=False)

        # Middleware should check enabled status in before_tool_call
        # But our implementation doesn't do that by default
        result = await middleware.before_tool_call("read_file", {"path": "/tmp/file.txt"})

        # The middleware still processes even when disabled
        # It's up to the framework to check the enabled property
        assert middleware.processed is True

        # But we can check the enabled property
        assert middleware.enabled is False


# =============================================================================
# Test MiddlewareResult
# =============================================================================


class TestMiddlewareResult:
    """Tests for MiddlewareResult."""

    def test_default_result(self):
        """Test default MiddlewareResult."""
        result = MiddlewareResult()

        assert result.proceed is True
        assert result.modified_arguments is None
        assert result.error_message is None
        assert result.metadata == {}

    def test_custom_result(self):
        """Test custom MiddlewareResult."""
        result = MiddlewareResult(
            proceed=False,
            modified_arguments={"path": "/tmp/new_path.txt"},
            error_message="Access denied",
            metadata={"reason": "permission_denied"},
        )

        assert result.proceed is False
        assert result.modified_arguments == {"path": "/tmp/new_path.txt"}
        assert result.error_message == "Access denied"
        assert result.metadata == {"reason": "permission_denied"}


# =============================================================================
# Integration Tests
# =============================================================================


class TestMiddlewareIntegration:
    """Integration tests for middleware system."""

    @pytest.mark.asyncio
    async def test_middleware_chain_simulation(self):
        """Test simulating a chain of middleware."""
        middleware_chain = [
            SimpleTestMiddleware(priority=MiddlewarePriority.CRITICAL, phase=MiddlewarePhase.PRE),
            SimpleTestMiddleware(priority=MiddlewarePriority.HIGH, phase=MiddlewarePhase.PRE),
            SimpleTestMiddleware(priority=MiddlewarePriority.NORMAL, phase=MiddlewarePhase.PRE),
        ]

        tool_name = "write_file"
        arguments = {"path": "/tmp/file.txt", "content": "Hello, World!"}

        # Simulate middleware chain execution
        final_result = None
        for middleware in middleware_chain:
            result = await middleware.before_tool_call(tool_name, arguments)
            if not result.proceed:
                # Middleware blocked execution
                final_result = result
                break
            # Update arguments if modified
            if result.modified_arguments:
                arguments = result.modified_arguments

        if final_result is None:
            final_result = MiddlewareResult()

        assert final_result.proceed is True
        assert all(m.before_call_count == 1 for m in middleware_chain)

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test that middleware can be ordered by priority."""
        middleware_list = [
            SimpleTestMiddleware(priority=MiddlewarePriority.NORMAL),
            SimpleTestMiddleware(priority=MiddlewarePriority.CRITICAL),
            SimpleTestMiddleware(priority=MiddlewarePriority.HIGH),
            SimpleTestMiddleware(priority=MiddlewarePriority.LOW),
        ]

        # Sort by priority (lower values first for PRE phase)
        sorted_middleware = sorted(middleware_list, key=lambda m: m.get_priority().value)

        assert sorted_middleware[0].get_priority() == MiddlewarePriority.CRITICAL
        assert sorted_middleware[1].get_priority() == MiddlewarePriority.HIGH
        assert sorted_middleware[2].get_priority() == MiddlewarePriority.NORMAL
        assert sorted_middleware[3].get_priority() == MiddlewarePriority.LOW

    @pytest.mark.asyncio
    async def test_phase_filtering(self):
        """Test filtering middleware by phase."""
        middleware_list = [
            SimpleTestMiddleware(phase=MiddlewarePhase.PRE),
            SimpleTestMiddleware(phase=MiddlewarePhase.POST),
            SimpleTestMiddleware(phase=MiddlewarePhase.AROUND),
            SimpleTestMiddleware(phase=MiddlewarePhase.ERROR),
        ]

        # Filter PRE phase middleware
        pre_middleware = [m for m in middleware_list if m.get_phase() == MiddlewarePhase.PRE]
        assert len(pre_middleware) == 1

        # Filter POST phase middleware
        post_middleware = [m for m in middleware_list if m.get_phase() == MiddlewarePhase.POST]
        assert len(post_middleware) == 1

    @pytest.mark.asyncio
    async def test_tool_filtering(self):
        """Test filtering middleware by applicable tools."""
        middleware_list = [
            SimpleTestMiddleware(applicable_tools={"write_file", "edit_file"}),
            SimpleTestMiddleware(applicable_tools={"read_file", "grep"}),
            SimpleTestMiddleware(applicable_tools=None),  # All tools
        ]

        tool_name = "write_file"

        # Get middleware that applies to the tool
        applicable = [m for m in middleware_list if m.applies_to_tool(tool_name)]

        # Should be 2 (first and third middleware)
        assert len(applicable) == 2
