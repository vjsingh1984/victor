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
Integration tests for tool call signature computation.

Tests the integration between Rust signature module and Python tool calling.
"""

import pytest

from victor.agent.tool_calling.base import ToolCall
from victor.agent.tool_calling.signature import (
    ToolCallSignatureManager,
    compute_signature,
    deduplicate_tool_calls,
    get_signature_manager,
)

victor_native = pytest.importorskip("victor_native")


class TestSignatureIntegration:
    """Integration tests for signature computation."""

    def test_manager_creation(self):
        """Test signature manager creation."""
        manager = ToolCallSignatureManager()
        assert manager is not None
        # use_rust depends on whether victor_native is available
        assert isinstance(manager.use_rust, bool)

    def test_compute_signature_basic(self):
        """Test basic signature computation."""
        sig = compute_signature("read_file", {"path": "/tmp/test.txt"})
        assert isinstance(sig, int)
        assert sig > 0

    def test_compute_signature_with_tool_call(self):
        """Test signature computation with ToolCall object."""
        call = ToolCall(name="read_file", arguments={"path": "/tmp/test.txt"})
        manager = ToolCallSignatureManager()
        sig = manager.compute_signature_for_tool_call(call)
        assert isinstance(sig, int)
        assert sig > 0

    def test_deduplicate_tool_calls(self):
        """Test tool call deduplication."""
        calls = [
            ToolCall(name="read_file", arguments={"path": "a.txt"}),
            ToolCall(name="read_file", arguments={"path": "a.txt"}),  # duplicate
            ToolCall(name="write_file", arguments={"path": "b.txt"}),
            ToolCall(name="read_file", arguments={"path": "c.txt"}),
            ToolCall(name="write_file", arguments={"path": "b.txt"}),  # duplicate
        ]

        unique = deduplicate_tool_calls(calls)

        assert len(unique) == 3
        assert unique[0].name == "read_file"
        assert unique[1].name == "write_file"
        assert unique[2].name == "read_file"
        assert unique[0].arguments["path"] == "a.txt"
        assert unique[2].arguments["path"] == "c.txt"

    def test_batch_signatures(self):
        """Test batch signature computation."""
        manager = ToolCallSignatureManager()

        tools = ["read_file", "write_file", "search"]
        args = [
            {"path": "a.txt"},
            {"path": "b.txt", "content": "hello"},
            {"query": "test", "limit": 10},
        ]

        sigs = manager.compute_batch_signatures(tools, args)

        assert len(sigs) == 3
        assert all(isinstance(sig, int) for sig in sigs)
        assert all(sig > 0 for sig in sigs)
        assert len(set(sigs)) == 3, "All signatures should be unique"

    def test_loop_detection(self):
        """Test loop detection."""
        manager = ToolCallSignatureManager()

        calls = [
            ToolCall(name="read_file", arguments={"path": "a.txt"}),
            ToolCall(name="write_file", arguments={"path": "b.txt"}),
            ToolCall(name="read_file", arguments={"path": "a.txt"}),
            ToolCall(name="write_file", arguments={"path": "b.txt"}),
            ToolCall(name="read_file", arguments={"path": "a.txt"}),
        ]

        looped = manager.detect_loops(calls, threshold=3)

        assert "read_file" in looped
        assert "write_file" not in looped  # Only appears twice

    def test_signature_consistency(self):
        """Test that identical calls produce identical signatures."""
        sig1 = compute_signature("read_file", {"path": "a.txt", "offset": 0})
        sig2 = compute_signature("read_file", {"offset": 0, "path": "a.txt"})

        assert sig1 == sig2, "Signatures should be identical regardless of key order"

    def test_signature_difference(self):
        """Test that different calls produce different signatures."""
        sig1 = compute_signature("read_file", {"path": "a.txt"})
        sig2 = compute_signature("write_file", {"path": "a.txt"})

        assert sig1 != sig2, "Different tools should produce different signatures"

    def test_global_manager_singleton(self):
        """Test that global manager is a singleton."""
        manager1 = get_signature_manager()
        manager2 = get_signature_manager()

        assert manager1 is manager2

    def test_empty_calls_list(self):
        """Test deduplication with empty list."""
        manager = ToolCallSignatureManager()
        unique = manager.deduplicate_tool_calls([])

        assert unique == []

    def test_preserves_tool_call_metadata(self):
        """Test that deduplication preserves ToolCall metadata."""
        calls = [
            ToolCall(
                name="read_file",
                arguments={"path": "a.txt"},
                id="call_1",
                raw={"original": "data"},
            ),
            ToolCall(
                name="read_file",
                arguments={"path": "a.txt"},
                id="call_2",
                raw={"original": "data"},
            ),
            ToolCall(
                name="write_file",
                arguments={"path": "b.txt"},
                id="call_3",
            ),
        ]

        unique = deduplicate_tool_calls(calls)

        assert len(unique) == 2
        assert unique[0].id == "call_1"  # First occurrence preserved
        assert unique[0].raw == {"original": "data"}

    def test_python_fallback(self):
        """Test Python fallback when Rust is not available."""
        manager = ToolCallSignatureManager(use_rust=False)

        sig = manager.compute_signature("read_file", {"path": "a.txt"})
        assert isinstance(sig, int)
        assert sig > 0

    def test_batch_mismatched_lengths(self):
        """Test that mismatched lengths raise an error."""
        manager = ToolCallSignatureManager()

        with pytest.raises(ValueError, match="must have same length"):
            manager.compute_batch_signatures(
                ["read_file", "write_file"],
                [{"path": "a.txt"}],  # Only one args dict
            )

    def test_nested_arguments(self):
        """Test signature computation with nested arguments."""
        sig = compute_signature(
            "complex_tool",
            {
                "config": {"enabled": True, "level": 5},
                "items": ["a", "b", "c"],
            },
        )
        assert isinstance(sig, int)
        assert sig > 0

    def test_performance_consistency(self):
        """Test that signatures are computed consistently."""
        signatures = set()
        for _ in range(100):
            sig = compute_signature("read_file", {"path": "/tmp/test.txt"})
            signatures.add(sig)

        assert len(signatures) == 1, "All signatures should be identical"
