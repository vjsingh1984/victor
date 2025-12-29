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

Extends the core BaseToolDependencyProvider with coding-specific data.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from victor.core.tool_dependency_base import BaseToolDependencyProvider, ToolDependencyConfig
from victor.core.tool_types import ToolDependency


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

# Tool transition probabilities for coding workflows
CODING_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    "read_file": [
        ("edit_files", 0.4),
        ("code_search", 0.3),
        ("write_file", 0.2),
        ("run_tests", 0.1),
    ],
    "code_search": [
        ("read_file", 0.6),
        ("semantic_code_search", 0.2),
        ("list_directory", 0.2),
    ],
    "list_directory": [
        ("read_file", 0.5),
        ("code_search", 0.3),
        ("write_file", 0.2),
    ],
    "edit_files": [
        ("run_tests", 0.5),
        ("read_file", 0.3),
        ("git_status", 0.2),
    ],
    "write_file": [
        ("run_tests", 0.4),
        ("read_file", 0.3),
        ("git_status", 0.3),
    ],
    "run_tests": [
        ("edit_files", 0.4),
        ("git_status", 0.3),
        ("read_file", 0.3),
    ],
    "git_status": [
        ("git_diff", 0.5),
        ("git_commit", 0.3),
        ("read_file", 0.2),
    ],
    "git_diff": [
        ("git_commit", 0.5),
        ("edit_files", 0.3),
        ("git_status", 0.2),
    ],
}

# Tool clusters for coding
CODING_TOOL_CLUSTERS: Dict[str, Set[str]] = {
    "file_operations": {"read_file", "write_file", "edit_files", "list_directory"},
    "search_operations": {"code_search", "semantic_code_search", "symbols"},
    "git_operations": {"git_status", "git_diff", "git_commit", "git_push"},
    "testing": {"run_tests", "execute_bash"},
    "refactoring": {"refactor_rename_symbol", "refactor_extract_function"},
}

# Common tool sequences for coding workflows
CODING_TOOL_SEQUENCES: Dict[str, List[str]] = {
    "exploration": ["list_directory", "read_file", "code_search"],
    "edit": ["read_file", "edit_files", "run_tests"],
    "create": ["list_directory", "read_file", "write_file", "run_tests"],
    "refactor": ["read_file", "symbols", "refactor_rename_symbol", "run_tests"],
    "git": ["git_status", "git_diff", "git_commit"],
    "debug": ["read_file", "code_search", "execute_bash"],
    "architecture": ["list_directory", "read_file", "semantic_code_search", "read_file"],
    "test_fix": ["read_file", "run_tests", "edit_files", "run_tests"],
}

# Required tools for coding vertical
CODING_REQUIRED_TOOLS: Set[str] = {
    "read_file",
    "write_file",
    "edit_files",
    "list_directory",
    "code_search",
}

# Optional tools that enhance coding
CODING_OPTIONAL_TOOLS: Set[str] = {
    "semantic_code_search",
    "symbols",
    "run_tests",
    "git_status",
    "git_diff",
    "git_commit",
    "refactor_rename_symbol",
    "refactor_extract_function",
    "execute_bash",
}


class CodingToolDependencyProvider(BaseToolDependencyProvider):
    """Tool dependency provider for coding vertical.

    Extends BaseToolDependencyProvider with coding-specific tool
    relationships, transitions, and sequences.

    Provides tool execution patterns that improve intelligent tool
    selection for software development tasks.
    """

    def __init__(
        self,
        additional_dependencies: List[ToolDependency] | None = None,
        additional_sequences: Dict[str, List[str]] | None = None,
    ):
        """Initialize the provider.

        Args:
            additional_dependencies: Additional tool dependencies to merge
            additional_sequences: Additional tool sequences to merge
        """
        # Build dependencies list
        dependencies = CODING_TOOL_DEPENDENCIES.copy()
        if additional_dependencies:
            dependencies.extend(additional_dependencies)

        # Build sequences dict
        sequences = CODING_TOOL_SEQUENCES.copy()
        if additional_sequences:
            sequences.update(additional_sequences)

        # Initialize base class with coding-specific config
        super().__init__(
            ToolDependencyConfig(
                dependencies=dependencies,
                transitions=CODING_TOOL_TRANSITIONS.copy(),
                clusters=CODING_TOOL_CLUSTERS.copy(),
                sequences=sequences,
                required_tools=CODING_REQUIRED_TOOLS.copy(),
                optional_tools=CODING_OPTIONAL_TOOLS.copy(),
                default_sequence=["read_file", "edit_files", "run_tests"],
            )
        )


__all__ = [
    "CodingToolDependencyProvider",
    "CODING_TOOL_DEPENDENCIES",
    "CODING_TOOL_SEQUENCES",
    "CODING_TOOL_TRANSITIONS",
    "CODING_TOOL_CLUSTERS",
    "CODING_REQUIRED_TOOLS",
    "CODING_OPTIONAL_TOOLS",
]
