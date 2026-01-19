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

"""Config protocol for agent configuration access.

This module defines the ConfigProtocol for accessing agent configuration
settings. This protocol isolates configuration-related functionality
following the Interface Segregation Principle (ISP).

Design Principles:
    - ISP: Protocol contains only configuration-related properties
    - DIP: Depend on this abstraction, not concrete implementations
    - OCP: Extend via protocol composition, not modification
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConfigProtocol(Protocol):
    """Agent configuration access interface.

    This protocol defines properties for accessing agent configuration
    including settings, tool budget, and agent mode.

    Implementations:
        - AgentOrchestrator (via IAgentOrchestrator)
        - Mock implementations for testing
    """

    @property
    def settings(self) -> Any:
        """Get configuration settings.

        Returns:
            Settings object with agent configuration

        Examples:
            >>> settings = orchestrator.settings
            >>> print(settings.debug_mode)
        """
        ...

    @property
    def tool_budget(self) -> int:
        """Get the tool budget for this session.

        Returns:
            Maximum number of tool calls allowed

        Examples:
            >>> budget = orchestrator.tool_budget
            >>> print(f"Tool budget: {budget}")
        """
        ...

    @property
    def mode(self) -> Any:
        """Get the current agent mode.

        Returns:
            AgentMode enum value (BUILD, PLAN, EXPLORE)

        Examples:
            >>> mode = orchestrator.mode
            >>> print(f"Mode: {mode.name}")
        """
        ...


__all__ = [
    "ConfigProtocol",
]
