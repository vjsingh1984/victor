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

"""Core tool types for dependency management.

This module defines fundamental tool types used by the core tool dependency
system. These are placed in core to avoid circular imports between
core and verticals.

Note:
    For backward compatibility, these types are also re-exported from
    `victor.core.verticals.protocols`.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Protocol, Set, runtime_checkable


@dataclass
class ToolDependency:
    """Dependency relationship between tools.

    Attributes:
        tool_name: The tool
        depends_on: Tools that should be called before this one
        enables: Tools that are enabled after this one succeeds
        weight: Transition probability weight
    """

    tool_name: str
    depends_on: Set[str] = field(default_factory=set)
    enables: Set[str] = field(default_factory=set)
    weight: float = 1.0


@runtime_checkable
class ToolDependencyProviderProtocol(Protocol):
    """Protocol for providing tool dependency information.

    Enables verticals to define tool execution patterns and
    transition probabilities for intelligent tool selection.

    Example:
        class CodingToolDependencies(ToolDependencyProviderProtocol):
            def get_dependencies(self) -> List[ToolDependency]:
                return [
                    ToolDependency(
                        tool_name="edit_files",
                        depends_on={"read_file"},
                        enables={"run_tests"},
                        weight=0.8,
                    ),
                ]
    """

    @abstractmethod
    def get_dependencies(self) -> List[ToolDependency]:
        """Get tool dependencies for this vertical.

        Returns:
            List of tool dependency definitions
        """
        ...

    def get_tool_sequences(self) -> List[List[str]]:
        """Get common tool call sequences.

        Returns:
            List of tool name sequences representing common workflows
        """
        return []


__all__ = [
    "ToolDependency",
    "ToolDependencyProviderProtocol",
]
