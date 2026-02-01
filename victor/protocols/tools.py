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

"""Tool protocol for tool registry access.

This module defines the ToolProtocol for accessing tool registries
and budget information. This protocol isolates tool-related functionality
following the Interface Segregation Principle (ISP).

Design Principles:
    - ISP: Protocol contains only tool-related properties
    - DIP: Depend on this abstraction, not concrete implementations
    - OCP: Extend via protocol composition, not modification
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class ToolProtocol(Protocol):
    """Tool registry and budget access interface.

    This protocol defines properties for accessing the tool registry
    and tool budget information.

    Implementations:
        - AgentOrchestrator (via IAgentOrchestrator)
        - Mock implementations for testing
    """

    @property
    def tool_registry(self) -> Any:
        """Get the tool registry.

        Returns:
            ToolRegistry instance with available tools

        Examples:
            >>> registry = orchestrator.tool_registry
            >>> tools = registry.list_tools()
        """
        ...

    @property
    def allowed_tools(self) -> Optional[list[str]]:
        """Get list of allowed tool names, if restricted.

        Returns:
            List of allowed tool names, or None if all allowed

        Examples:
            >>> allowed = orchestrator.allowed_tools
            >>> if allowed:
            ...     print(f"Restricted to: {allowed}")
        """
        ...


__all__ = [
    "ToolProtocol",
]
