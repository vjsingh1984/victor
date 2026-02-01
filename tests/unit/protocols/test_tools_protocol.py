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

"""Tests for ToolProtocol.

Tests the ToolProtocol interface and conformance.
"""

from typing import Any, Optional

from victor.protocols.tools import ToolProtocol


class MockToolRegistry:
    """Mock tool registry for testing."""

    def __init__(self):
        self.tools = ["read_file", "write_file", "search"]


class MockToolImplementation:
    """Mock implementation of ToolProtocol for testing."""

    def __init__(self):
        self._tool_registry = MockToolRegistry()
        self._allowed_tools = None

    @property
    def tool_registry(self) -> Any:
        """Get the tool registry."""
        return self._tool_registry

    @property
    def allowed_tools(self) -> Optional[list[str]]:
        """Get list of allowed tool names, if restricted."""
        return self._allowed_tools


class TestToolProtocol:
    """Test suite for ToolProtocol."""

    def test_tool_registry_property(self):
        """Test that tool_registry property works correctly."""
        impl = MockToolImplementation()
        registry = impl.tool_registry
        assert isinstance(registry, MockToolRegistry)
        assert registry.tools == ["read_file", "write_file", "search"]

    def test_allowed_tools_property_none(self):
        """Test allowed_tools when not restricted (None)."""
        impl = MockToolImplementation()
        assert impl.allowed_tools is None

    def test_allowed_tools_property_restricted(self):
        """Test allowed_tools when restricted."""
        impl = MockToolImplementation()
        impl._allowed_tools = ["read_file", "search"]
        assert impl.allowed_tools == ["read_file", "search"]

    def test_protocol_conformance(self):
        """Test that mock implements ToolProtocol."""
        impl = MockToolImplementation()
        # This should not raise an error
        assert isinstance(impl, ToolProtocol)

    def test_tool_registry_with_different_tools(self):
        """Test with different tool configurations."""
        impl = MockToolImplementation()
        impl._tool_registry.tools = ["grep", "find", "ls"]
        assert impl.tool_registry.tools == ["grep", "find", "ls"]

    def test_allowed_tools_empty_list(self):
        """Test allowed_tools with empty list (no tools allowed)."""
        impl = MockToolImplementation()
        impl._allowed_tools = []
        assert impl.allowed_tools == []


class TestToolProtocolTypeChecking:
    """Test type checking and protocol compliance."""

    def test_tool_protocol_is_protocol(self):
        """Test that ToolProtocol is a Protocol."""
        from typing import Protocol

        assert issubclass(ToolProtocol, Protocol)

    def test_tool_protocol_has_tool_registry_property(self):
        """Test that ToolProtocol defines tool_registry property."""
        assert hasattr(ToolProtocol, "__annotations__")
        # Check that tool_registry is in the protocol
        assert "tool_registry" in dir(ToolProtocol)

    def test_tool_protocol_has_allowed_tools_property(self):
        """Test that ToolProtocol defines allowed_tools property."""
        assert hasattr(ToolProtocol, "__annotations__")
        assert "allowed_tools" in dir(ToolProtocol)
