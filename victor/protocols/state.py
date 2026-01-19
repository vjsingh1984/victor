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

"""State protocol for session state access.

This module defines the StateProtocol for accessing session state
information. This protocol isolates state-related functionality
following the Interface Segregation Principle (ISP).

Design Principles:
    - ISP: Protocol contains only state-related properties
    - DIP: Depend on this abstraction, not concrete implementations
    - OCP: Extend via protocol composition, not modification
"""

from __future__ import annotations

from typing import List, Protocol, Set, Tuple, runtime_checkable


@runtime_checkable
class StateProtocol(Protocol):
    """Session state access interface.

    This protocol defines properties for accessing session state
    information including tool usage, execution history, and
    observed files.

    Implementations:
        - AgentOrchestrator (via IAgentOrchestrator)
        - Mock implementations for testing
    """

    @property
    def tool_calls_used(self) -> int:
        """Get number of tool calls used in this session.

        Returns:
            Count of tool calls made

        Examples:
            >>> count = orchestrator.tool_calls_used
            >>> print(f"Tool calls: {count}")
        """
        ...

    @property
    def executed_tools(self) -> List[str]:
        """Get list of executed tool names in order.

        Returns:
            List of tool names that have been executed

        Examples:
            >>> tools = orchestrator.executed_tools
            >>> print(f"Executed: {tools}")
        """
        ...

    @property
    def failed_tool_signatures(self) -> Set[Tuple[str, str]]:
        """Get set of failed tool call signatures.

        Returns:
            Set of (tool_name, args_hash) for failed calls

        Examples:
            >>> failed = orchestrator.failed_tool_signatures
            >>> if failed:
            ...     print(f"Failed: {failed}")
        """
        ...

    @property
    def observed_files(self) -> Set[str]:
        """Get set of files observed during session.

        Returns:
            Set of file paths that have been read

        Examples:
            >>> files = orchestrator.observed_files
            >>> print(f"Observed: {len(files)} files")
        """
        ...


__all__ = [
    "StateProtocol",
]
