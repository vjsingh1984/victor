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
Fast tool call signature computation for deduplication.

This module provides a high-performance wrapper around the Rust signature
computation module, achieving 10x speedup over pure Python implementation.

Features:
- Fast signature computation using SeaHash
- Batch processing for efficiency
- Order-preserving deduplication
- Graceful error handling
"""

import logging
from typing import Any, Dict, List, Optional, cast

from victor.agent.tool_calling.base import ToolCall

logger = logging.getLogger(__name__)

# Try to import victor_native, fallback to Python implementation
try:
    from victor import victor_native

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logger.warning(
        "victor_native Rust extension not available. "
        "Using Python fallback (slower). "
        "Build with: cargo build --release && maturin develop --release"
    )
    import hashlib
    import json


class ToolCallSignatureManager:
    """
    Manager for tool call signature computation and deduplication.

    This class provides a high-level interface for computing signatures
    and deduplicating tool calls using the Rust implementation when available.

    Example:
        >>> manager = ToolCallSignatureManager()
        >>> signature = manager.compute_signature("read_file", {"path": "/tmp/test.txt"})
        >>> print(signature)  # 12345678901234567890
    """

    def __init__(self, use_rust: bool = True):
        """
        Initialize the signature manager.

        Args:
            use_rust: Whether to use Rust implementation (default: True).
                     Falls back to Python if Rust is not available.
        """
        self.use_rust = use_rust and RUST_AVAILABLE
        if use_rust and not RUST_AVAILABLE:
            logger.warning("Rust implementation requested but not available, using Python")

    def compute_signature(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> int:
        """
        Compute a signature hash for a tool call.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments as a dictionary

        Returns:
            Signature hash as an integer

        Raises:
            ValueError: If arguments cannot be serialized
        """
        if self.use_rust:
            try:
                result = victor_native.compute_tool_call_signature(tool_name, arguments)
                return cast(int, result)
            except Exception as e:
                logger.error(f"Rust signature computation failed: {e}")
                raise

        # Python fallback
        try:
            sorted_args = json.dumps(arguments, sort_keys=True)
            combined = f"{tool_name}:{sorted_args}"
            hash_bytes = hashlib.sha256(combined.encode()).digest()[:8]
            return int.from_bytes(hash_bytes, byteorder="big")
        except Exception as e:
            logger.error(f"Python signature computation failed: {e}")
            raise ValueError(f"Failed to compute signature: {e}") from e

    def compute_batch_signatures(
        self,
        tool_names: List[str],
        arguments_list: List[Dict[str, Any]],
    ) -> List[int]:
        """
        Compute signatures for multiple tool calls in batch.

        Args:
            tool_names: List of tool names
            arguments_list: List of argument dictionaries

        Returns:
            List of signature hashes

        Raises:
            ValueError: If lengths don't match or serialization fails
        """
        if len(tool_names) != len(arguments_list):
            raise ValueError(
                f"tool_names and arguments_list must have same length: "
                f"got {len(tool_names)} and {len(arguments_list)}"
            )

        if self.use_rust:
            try:
                result = victor_native.batch_compute_tool_call_signatures(
                    tool_names,
                    arguments_list,
                )
                return cast(List[int], result)
            except Exception as e:
                logger.error(f"Rust batch signature computation failed: {e}")
                raise

        # Python fallback
        signatures = []
        for tool_name, arguments in zip(tool_names, arguments_list):
            sig = self.compute_signature(tool_name, arguments)
            signatures.append(sig)
        return signatures

    def deduplicate_tool_calls(
        self,
        tool_calls: List[ToolCall],
    ) -> List[ToolCall]:
        """
        Deduplicate a list of tool calls based on their signatures.

        Preserves the order of first occurrence.

        Args:
            tool_calls: List of ToolCall objects

        Returns:
            List of unique ToolCall objects
        """
        if not tool_calls:
            return []

        if self.use_rust:
            try:
                # Convert to dict format for Rust function
                calls_dict = [
                    {
                        "tool_name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in tool_calls
                ]
                unique_dicts = victor_native.deduplicate_tool_calls_dict(calls_dict)

                # Convert back to ToolCall objects
                return [
                    ToolCall(
                        name=ud["tool_name"],
                        arguments=ud["arguments"],
                        id=tool_calls[calls_dict.index(ud)].id,
                        raw=tool_calls[calls_dict.index(ud)].raw,
                    )
                    for ud in unique_dicts
                ]
            except Exception as e:
                logger.error(f"Rust deduplication failed: {e}")
                # Fall through to Python implementation

        # Python fallback
        seen = set()
        unique_calls = []

        for call in tool_calls:
            sig = self.compute_signature(call.name, call.arguments)
            if sig not in seen:
                seen.add(sig)
                unique_calls.append(call)

        return unique_calls

    def detect_loops(
        self,
        tool_calls: List[ToolCall],
        threshold: int = 3,
    ) -> List[str]:
        """
        Detect potential loops by finding repeated tool calls.

        Args:
            tool_calls: List of tool calls to analyze
            threshold: Number of repetitions before flagging as loop

        Returns:
            List of tool names that appear to be in a loop
        """
        signature_counts: Dict[int, int] = {}
        signature_to_tool: Dict[int, str] = {}

        for call in tool_calls:
            sig = self.compute_signature(call.name, call.arguments)
            signature_counts[sig] = signature_counts.get(sig, 0) + 1
            signature_to_tool[sig] = call.name

        # Find signatures that exceed threshold
        looped_tools = [
            signature_to_tool[sig] for sig, count in signature_counts.items() if count >= threshold
        ]

        return list(set(looped_tools))

    def compute_signature_for_tool_call(self, tool_call: ToolCall) -> int:
        """
        Compute signature for a ToolCall object.

        Convenience method for working with ToolCall objects directly.

        Args:
            tool_call: ToolCall object

        Returns:
            Signature hash
        """
        return self.compute_signature(tool_call.name, tool_call.arguments)


# Global singleton instance
_default_manager: Optional[ToolCallSignatureManager] = None


def get_signature_manager() -> ToolCallSignatureManager:
    """
    Get the global signature manager singleton.

    Returns:
        ToolCallSignatureManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = ToolCallSignatureManager()
    return _default_manager


def compute_signature(tool_name: str, arguments: Dict[str, Any]) -> int:
    """
    Compute a signature hash for a tool call (convenience function).

    Args:
        tool_name: Name of the tool being called
        arguments: Tool arguments as a dictionary

    Returns:
        Signature hash as an integer
    """
    return get_signature_manager().compute_signature(tool_name, arguments)


def deduplicate_tool_calls(tool_calls: List[ToolCall]) -> List[ToolCall]:
    """
    Deduplicate tool calls (convenience function).

    Args:
        tool_calls: List of ToolCall objects

    Returns:
        List of unique ToolCall objects
    """
    return get_signature_manager().deduplicate_tool_calls(tool_calls)
