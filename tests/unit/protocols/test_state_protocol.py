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

"""Tests for StateProtocol.

Tests the StateProtocol interface and conformance.
"""

import pytest
from typing import List, Set, Tuple

from victor.protocols.state import StateProtocol


class MockStateImplementation:
    """Mock implementation of StateProtocol for testing."""

    def __init__(self):
        self._tool_calls_used = 5
        self._executed_tools = ["read_file", "search", "read_file"]
        self._failed_tool_signatures = {
            ("write_file", "hash1"),
            ("delete_file", "hash2"),
        }
        self._observed_files = {
            "/path/to/file1.py",
            "/path/to/file2.py",
        }

    @property
    def tool_calls_used(self) -> int:
        """Get number of tool calls used in this session."""
        return self._tool_calls_used

    @property
    def executed_tools(self) -> List[str]:
        """Get list of executed tool names in order."""
        return self._executed_tools

    @property
    def failed_tool_signatures(self) -> Set[Tuple[str, str]]:
        """Get set of failed tool call signatures."""
        return self._failed_tool_signatures

    @property
    def observed_files(self) -> Set[str]:
        """Get set of files observed during session."""
        return self._observed_files


class TestStateProtocol:
    """Test suite for StateProtocol."""

    def test_tool_calls_used_property(self):
        """Test that tool_calls_used property works correctly."""
        impl = MockStateImplementation()
        assert impl.tool_calls_used == 5

    def test_executed_tools_property(self):
        """Test that executed_tools property works correctly."""
        impl = MockStateImplementation()
        tools = impl.executed_tools
        assert tools == ["read_file", "search", "read_file"]
        assert len(tools) == 3

    def test_failed_tool_signatures_property(self):
        """Test that failed_tool_signatures property works correctly."""
        impl = MockStateImplementation()
        failed = impl.failed_tool_signatures
        assert len(failed) == 2
        assert ("write_file", "hash1") in failed
        assert ("delete_file", "hash2") in failed

    def test_observed_files_property(self):
        """Test that observed_files property works correctly."""
        impl = MockStateImplementation()
        files = impl.observed_files
        assert len(files) == 2
        assert "/path/to/file1.py" in files
        assert "/path/to/file2.py" in files

    def test_protocol_conformance(self):
        """Test that mock implements StateProtocol."""
        impl = MockStateImplementation()
        # This should not raise an error
        assert isinstance(impl, StateProtocol)

    def test_tool_calls_used_increment(self):
        """Test updating tool_calls_used."""
        impl = MockStateImplementation()
        impl._tool_calls_used = 10
        assert impl.tool_calls_used == 10

    def test_executed_tools_append(self):
        """Test appending to executed_tools."""
        impl = MockStateImplementation()
        impl._executed_tools.append("write_file")
        tools = impl.executed_tools
        assert len(tools) == 4
        assert "write_file" in tools

    def test_failed_tool_signatures_add(self):
        """Test adding to failed_tool_signatures."""
        impl = MockStateImplementation()
        impl._failed_tool_signatures.add(("edit_file", "hash3"))
        failed = impl.failed_tool_signatures
        assert len(failed) == 3
        assert ("edit_file", "hash3") in failed

    def test_observed_files_add(self):
        """Test adding to observed_files."""
        impl = MockStateImplementation()
        impl._observed_files.add("/path/to/file3.py")
        files = impl.observed_files
        assert len(files) == 3
        assert "/path/to/file3.py" in files


class TestStateProtocolTypeChecking:
    """Test type checking and protocol compliance."""

    def test_state_protocol_is_protocol(self):
        """Test that StateProtocol is a Protocol."""
        from typing import Protocol

        assert issubclass(StateProtocol, Protocol)

    def test_state_protocol_has_tool_calls_used_property(self):
        """Test that StateProtocol defines tool_calls_used property."""
        assert hasattr(StateProtocol, "__annotations__")
        # Check that tool_calls_used is in the protocol
        assert "tool_calls_used" in dir(StateProtocol)

    def test_state_protocol_has_executed_tools_property(self):
        """Test that StateProtocol defines executed_tools property."""
        assert hasattr(StateProtocol, "__annotations__")
        assert "executed_tools" in dir(StateProtocol)

    def test_state_protocol_has_failed_tool_signatures_property(self):
        """Test that StateProtocol defines failed_tool_signatures property."""
        assert hasattr(StateProtocol, "__annotations__")
        assert "failed_tool_signatures" in dir(StateProtocol)

    def test_state_protocol_has_observed_files_property(self):
        """Test that StateProtocol defines observed_files property."""
        assert hasattr(StateProtocol, "__annotations__")
        assert "observed_files" in dir(StateProtocol)
