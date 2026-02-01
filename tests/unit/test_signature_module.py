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
Unit tests for the Rust signature computation module.

Tests the fast tool call signature computation for deduplication.
"""

import pytest

victor_native = pytest.importorskip("victor_native")


class TestToolCallSignature:
    """Test suite for tool call signature computation."""

    def test_compute_signature_basic(self):
        """Test basic signature computation."""
        sig = victor_native.compute_tool_call_signature(
            "read_file", {"path": "/tmp/test.txt", "offset": 0}
        )
        assert isinstance(sig, int)
        assert sig > 0

    def test_signature_consistency(self):
        """Test that identical calls produce identical signatures."""
        sig1 = victor_native.compute_tool_call_signature(
            "read_file", {"path": "/tmp/test.txt", "offset": 0}
        )
        sig2 = victor_native.compute_tool_call_signature(
            "read_file", {"offset": 0, "path": "/tmp/test.txt"}  # Different key order
        )
        assert sig1 == sig2, "Signatures should be identical regardless of key order"

    def test_signature_difference(self):
        """Test that different calls produce different signatures."""
        sig1 = victor_native.compute_tool_call_signature("read_file", {"path": "/tmp/test.txt"})
        sig2 = victor_native.compute_tool_call_signature("write_file", {"path": "/tmp/test.txt"})
        assert sig1 != sig2, "Different tools should produce different signatures"

    def test_signature_different_args(self):
        """Test that different arguments produce different signatures."""
        sig1 = victor_native.compute_tool_call_signature("read_file", {"path": "/tmp/test.txt"})
        sig2 = victor_native.compute_tool_call_signature("read_file", {"path": "/tmp/other.txt"})
        assert sig1 != sig2, "Different arguments should produce different signatures"

    def test_batch_compute_signatures(self):
        """Test batch signature computation."""
        tools = ["read_file", "write_file", "search"]
        args = [
            {"path": "a.txt"},
            {"path": "b.txt", "content": "hello"},
            {"query": "test", "limit": 10},
        ]
        sigs = victor_native.batch_compute_tool_call_signatures(tools, args)

        assert len(sigs) == 3
        assert all(isinstance(sig, int) for sig in sigs)
        assert all(sig > 0 for sig in sigs)

        # All signatures should be different
        assert len(set(sigs)) == 3, "All signatures should be unique"

    def test_batch_mismatched_lengths(self):
        """Test that mismatched lengths raise an error."""
        with pytest.raises(ValueError, match="must have same length"):
            victor_native.batch_compute_tool_call_signatures(
                ["read_file", "write_file"], [{"path": "a.txt"}]  # Only one args dict
            )

    def test_tool_call_data_basic(self):
        """Test ToolCallData class."""
        call = victor_native.ToolCallData(
            tool_name="read_file", arguments={"path": "/tmp/test.txt"}
        )
        assert call.tool_name == "read_file"
        assert call.signature is None

        # Compute signature
        sig = call.compute_signature()
        assert isinstance(sig, int)
        assert sig > 0
        assert call.signature == sig

    def test_tool_call_data_with_signature(self):
        """Test ToolCallData with pre-computed signature."""
        call = victor_native.ToolCallData(
            tool_name="read_file", arguments={"path": "/tmp/test.txt"}, signature=12345
        )
        assert call.signature == 12345

        # compute_signature should return existing signature
        sig = call.compute_signature()
        assert sig == 12345

    def test_deduplicate_tool_calls(self):
        """Test tool call deduplication."""

        calls = [
            victor_native.ToolCallData("read_file", {"path": "a.txt"}),
            victor_native.ToolCallData("read_file", {"path": "a.txt"}),  # duplicate
            victor_native.ToolCallData("write_file", {"path": "b.txt"}),
            victor_native.ToolCallData("read_file", {"path": "c.txt"}),
            victor_native.ToolCallData("write_file", {"path": "b.txt"}),  # duplicate
        ]

        unique = victor_native.deduplicate_tool_calls(calls)

        assert len(unique) == 3
        assert unique[0].tool_name == "read_file"
        assert unique[1].tool_name == "write_file"
        assert unique[2].tool_name == "read_file"
        assert unique[0].arguments.get("path") == "a.txt"
        assert unique[2].arguments.get("path") == "c.txt"

    def test_deduplicate_tool_calls_dict(self):
        """Test deduplication with raw Python dicts."""
        calls = [
            {"tool_name": "read_file", "arguments": {"path": "a.txt"}},
            {"tool_name": "read_file", "arguments": {"path": "a.txt"}},  # duplicate
            {"tool_name": "write_file", "arguments": {"path": "b.txt"}},
            {"tool_name": "read_file", "arguments": {"path": "c.txt"}},
            {"tool_name": "write_file", "arguments": {"path": "b.txt"}},  # duplicate
        ]

        unique = victor_native.deduplicate_tool_calls_dict(calls)

        assert len(unique) == 3
        assert unique[0]["tool_name"] == "read_file"
        assert unique[1]["tool_name"] == "write_file"
        assert unique[2]["tool_name"] == "read_file"

    def test_nested_arguments(self):
        """Test signature computation with nested arguments."""
        sig1 = victor_native.compute_tool_call_signature(
            "complex_tool", {"config": {"enabled": True, "level": 5}, "items": ["a", "b", "c"]}
        )
        assert isinstance(sig1, int)
        assert sig1 > 0

    def test_special_characters_in_args(self):
        """Test signature computation with special characters."""
        sig = victor_native.compute_tool_call_signature(
            "read_file", {"path": "/tmp/test with spaces & symbols!.txt"}
        )
        assert isinstance(sig, int)
        assert sig > 0

    def test_empty_arguments(self):
        """Test signature computation with empty arguments."""
        sig = victor_native.compute_tool_call_signature("noop", {})
        assert isinstance(sig, int)
        assert sig > 0

    def test_performance_consistency(self):
        """Test that signatures are computed consistently."""
        # Compute same signature 100 times
        signatures = set()
        for _ in range(100):
            sig = victor_native.compute_tool_call_signature("read_file", {"path": "/tmp/test.txt"})
            signatures.add(sig)

        # All should be identical
        assert len(signatures) == 1, "All signatures should be identical"

    def test_repr(self):
        """Test string representation of ToolCallData."""
        call = victor_native.ToolCallData(
            tool_name="read_file", arguments={"path": "/tmp/test.txt"}
        )
        repr_str = repr(call)
        assert "read_file" in repr_str
        assert "ToolCallData" in repr_str

        call.compute_signature()
        repr_str = repr(call)
        assert "signature=" in repr_str
