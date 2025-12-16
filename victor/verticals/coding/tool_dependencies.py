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

"""Coding-specific tool dependencies and sequences.

This module defines tool execution patterns and transition probabilities
for intelligent tool selection in software development tasks.
"""

from __future__ import annotations

from typing import List

from victor.verticals.protocols import ToolDependency, ToolDependencyProviderProtocol


# Tool dependencies for coding tasks
CODING_TOOL_DEPENDENCIES: List[ToolDependency] = [
    # Edit should be preceded by read
    ToolDependency(
        tool_name="edit_files",
        depends_on={"read_file"},
        enables={"run_tests", "git"},
        weight=0.9,
    ),
    ToolDependency(
        tool_name="write_file",
        depends_on={"read_file", "list_directory"},
        enables={"run_tests", "git"},
        weight=0.8,
    ),
    # Refactoring depends on understanding code
    ToolDependency(
        tool_name="refactor_rename_symbol",
        depends_on={"read_file", "symbols"},
        enables={"run_tests"},
        weight=0.7,
    ),
    ToolDependency(
        tool_name="refactor_extract_function",
        depends_on={"read_file"},
        enables={"run_tests"},
        weight=0.6,
    ),
    # Git operations
    ToolDependency(
        tool_name="git_commit",
        depends_on={"git_status", "git_diff"},
        enables=set(),
        weight=0.9,
    ),
    ToolDependency(
        tool_name="git_push",
        depends_on={"git_commit", "run_tests"},
        enables=set(),
        weight=0.8,
    ),
    # Testing
    ToolDependency(
        tool_name="run_tests",
        depends_on=set(),
        enables={"git_commit"},
        weight=0.7,
    ),
    # Search enables read
    ToolDependency(
        tool_name="code_search",
        depends_on=set(),
        enables={"read_file"},
        weight=0.8,
    ),
    ToolDependency(
        tool_name="semantic_code_search",
        depends_on=set(),
        enables={"read_file"},
        weight=0.8,
    ),
    # List enables read
    ToolDependency(
        tool_name="list_directory",
        depends_on=set(),
        enables={"read_file"},
        weight=0.7,
    ),
]


# Common tool sequences for coding workflows
CODING_TOOL_SEQUENCES: List[List[str]] = [
    # Exploration workflow
    ["list_directory", "read_file", "code_search"],
    # Edit workflow
    ["read_file", "edit_files", "run_tests"],
    # Create workflow
    ["list_directory", "read_file", "write_file", "run_tests"],
    # Refactor workflow
    ["read_file", "symbols", "refactor_rename_symbol", "run_tests"],
    # Git workflow
    ["git_status", "git_diff", "git_commit"],
    # Debug workflow
    ["read_file", "code_search", "execute_bash"],
    # Architecture analysis
    ["list_directory", "read_file", "semantic_code_search", "read_file"],
    # Test workflow
    ["read_file", "run_tests", "edit_files", "run_tests"],
]


class CodingToolDependencyProvider(ToolDependencyProviderProtocol):
    """Tool dependency provider for coding vertical.

    Provides tool execution patterns and sequences that improve
    intelligent tool selection for software development tasks.
    """

    def __init__(
        self,
        additional_dependencies: List[ToolDependency] | None = None,
        additional_sequences: List[List[str]] | None = None,
    ):
        """Initialize the provider.

        Args:
            additional_dependencies: Additional tool dependencies
            additional_sequences: Additional tool sequences
        """
        self._dependencies = CODING_TOOL_DEPENDENCIES.copy()
        if additional_dependencies:
            self._dependencies.extend(additional_dependencies)

        self._sequences = CODING_TOOL_SEQUENCES.copy()
        if additional_sequences:
            self._sequences.extend(additional_sequences)

    def get_dependencies(self) -> List[ToolDependency]:
        """Get coding tool dependencies.

        Returns:
            List of tool dependency definitions
        """
        return self._dependencies.copy()

    def get_tool_sequences(self) -> List[List[str]]:
        """Get common coding tool sequences.

        Returns:
            List of tool name sequences representing common workflows
        """
        return self._sequences.copy()

    def get_transition_weight(self, from_tool: str, to_tool: str) -> float:
        """Get the transition weight between two tools.

        Higher weight means the transition is more likely to be useful.

        Args:
            from_tool: Previous tool called
            to_tool: Candidate next tool

        Returns:
            Transition weight (0.0 to 1.0)
        """
        # Check if to_tool depends on from_tool
        for dep in self._dependencies:
            if dep.tool_name == to_tool and from_tool in dep.depends_on:
                return dep.weight

        # Check if from_tool enables to_tool
        for dep in self._dependencies:
            if dep.tool_name == from_tool and to_tool in dep.enables:
                return dep.weight * 0.8  # Slightly lower weight for enables

        # Check sequences for adjacency
        for seq in self._sequences:
            for i, tool in enumerate(seq[:-1]):
                if tool == from_tool and seq[i + 1] == to_tool:
                    return 0.6  # Moderate weight for sequence match

        return 0.3  # Default low weight


__all__ = [
    "CodingToolDependencyProvider",
    "CODING_TOOL_DEPENDENCIES",
    "CODING_TOOL_SEQUENCES",
]
