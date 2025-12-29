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

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from victor.core.tool_dependency_base import BaseToolDependencyProvider, ToolDependencyConfig
from victor.core.tool_types import ToolDependency
from victor.framework.tool_naming import ToolNames


# Tool dependencies for coding tasks
# Uses canonical ToolNames constants for consistency
CODING_TOOL_DEPENDENCIES: List[ToolDependency] = [
    # Edit should be preceded by read
    ToolDependency(
        tool_name=ToolNames.EDIT,
        depends_on={ToolNames.READ},
        enables={ToolNames.TEST, ToolNames.GIT},
        weight=0.9,
    ),
    ToolDependency(
        tool_name=ToolNames.WRITE,
        depends_on={ToolNames.READ, ToolNames.LS},
        enables={ToolNames.TEST, ToolNames.GIT},
        weight=0.8,
    ),
    # Refactoring depends on understanding code
    ToolDependency(
        tool_name=ToolNames.RENAME,
        depends_on={ToolNames.READ, ToolNames.SYMBOL},
        enables={ToolNames.TEST},
        weight=0.7,
    ),
    ToolDependency(
        tool_name=ToolNames.EXTRACT,
        depends_on={ToolNames.READ},
        enables={ToolNames.TEST},
        weight=0.6,
    ),
    # Git operations (git_commit, git_diff resolve to "git" canonical)
    ToolDependency(
        tool_name=ToolNames.GIT,  # Unified git tool
        depends_on=set(),  # Git status/diff are operations on the same tool
        enables=set(),
        weight=0.9,
    ),
    # Testing
    ToolDependency(
        tool_name=ToolNames.TEST,
        depends_on=set(),
        enables={ToolNames.GIT},
        weight=0.7,
    ),
    # Search enables read
    ToolDependency(
        tool_name=ToolNames.GREP,  # Keyword code search
        depends_on=set(),
        enables={ToolNames.READ},
        weight=0.8,
    ),
    ToolDependency(
        tool_name=ToolNames.CODE_SEARCH,  # Semantic code search
        depends_on=set(),
        enables={ToolNames.READ},
        weight=0.8,
    ),
    # List enables read
    ToolDependency(
        tool_name=ToolNames.LS,
        depends_on=set(),
        enables={ToolNames.READ},
        weight=0.7,
    ),
]

# Tool transition probabilities for coding workflows
# Uses canonical ToolNames constants for consistency
CODING_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    ToolNames.READ: [
        (ToolNames.EDIT, 0.4),
        (ToolNames.GREP, 0.3),
        (ToolNames.WRITE, 0.2),
        (ToolNames.TEST, 0.1),
    ],
    ToolNames.GREP: [
        (ToolNames.READ, 0.6),
        (ToolNames.CODE_SEARCH, 0.2),
        (ToolNames.LS, 0.2),
    ],
    ToolNames.LS: [
        (ToolNames.READ, 0.5),
        (ToolNames.GREP, 0.3),
        (ToolNames.WRITE, 0.2),
    ],
    ToolNames.EDIT: [
        (ToolNames.TEST, 0.5),
        (ToolNames.READ, 0.3),
        (ToolNames.GIT, 0.2),
    ],
    ToolNames.WRITE: [
        (ToolNames.TEST, 0.4),
        (ToolNames.READ, 0.3),
        (ToolNames.GIT, 0.3),
    ],
    ToolNames.TEST: [
        (ToolNames.EDIT, 0.4),
        (ToolNames.GIT, 0.3),
        (ToolNames.READ, 0.3),
    ],
    ToolNames.GIT: [
        (ToolNames.READ, 0.5),
        (ToolNames.EDIT, 0.3),
        (ToolNames.TEST, 0.2),
    ],
}

# Tool clusters for coding
# Uses canonical ToolNames constants for consistency
CODING_TOOL_CLUSTERS: Dict[str, Set[str]] = {
    "file_operations": {ToolNames.READ, ToolNames.WRITE, ToolNames.EDIT, ToolNames.LS},
    "search_operations": {ToolNames.GREP, ToolNames.CODE_SEARCH, ToolNames.SYMBOL},
    "git_operations": {ToolNames.GIT},  # Unified git tool handles all git operations
    "testing": {ToolNames.TEST, ToolNames.SHELL},
    "refactoring": {ToolNames.RENAME, ToolNames.EXTRACT},
}

# Common tool sequences for coding workflows
# Uses canonical ToolNames constants for consistency
CODING_TOOL_SEQUENCES: Dict[str, List[str]] = {
    "exploration": [ToolNames.LS, ToolNames.READ, ToolNames.GREP],
    "edit": [ToolNames.READ, ToolNames.EDIT, ToolNames.TEST],
    "create": [ToolNames.LS, ToolNames.READ, ToolNames.WRITE, ToolNames.TEST],
    "refactor": [ToolNames.READ, ToolNames.SYMBOL, ToolNames.RENAME, ToolNames.TEST],
    "git": [ToolNames.GIT],  # Unified git tool handles status/diff/commit
    "debug": [ToolNames.READ, ToolNames.GREP, ToolNames.SHELL],
    "architecture": [ToolNames.LS, ToolNames.READ, ToolNames.CODE_SEARCH, ToolNames.READ],
    "test_fix": [ToolNames.READ, ToolNames.TEST, ToolNames.EDIT, ToolNames.TEST],
}

# Required tools for coding vertical
# Uses canonical ToolNames constants for consistency
CODING_REQUIRED_TOOLS: Set[str] = {
    ToolNames.READ,
    ToolNames.WRITE,
    ToolNames.EDIT,
    ToolNames.LS,
    ToolNames.GREP,
}

# Optional tools that enhance coding
# Uses canonical ToolNames constants for consistency
CODING_OPTIONAL_TOOLS: Set[str] = {
    ToolNames.CODE_SEARCH,
    ToolNames.SYMBOL,
    ToolNames.TEST,
    ToolNames.GIT,
    ToolNames.RENAME,
    ToolNames.EXTRACT,
    ToolNames.SHELL,
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
        # Uses canonical ToolNames constants for consistency
        super().__init__(
            ToolDependencyConfig(
                dependencies=dependencies,
                transitions=CODING_TOOL_TRANSITIONS.copy(),
                clusters=CODING_TOOL_CLUSTERS.copy(),
                sequences=sequences,
                required_tools=CODING_REQUIRED_TOOLS.copy(),
                optional_tools=CODING_OPTIONAL_TOOLS.copy(),
                default_sequence=[ToolNames.READ, ToolNames.EDIT, ToolNames.TEST],
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
