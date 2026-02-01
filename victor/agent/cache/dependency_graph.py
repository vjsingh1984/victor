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

"""Tool dependency graph for cascading cache invalidation.

This module implements a directed acyclic graph (DAG) that tracks
dependencies between tools and files, enabling intelligent cache
invalidation strategies.

Design Patterns:
    - Observer Pattern: Track changes and notify dependents
    - Graph Theory: DAG for dependency tracking
    - SRP: Focused only on dependency tracking, not caching logic

Usage:
    graph = ToolDependencyGraph()

    # Tool depends on another tool
    graph.add_tool_dependency("code_search", "read")

    # Tool depends on a file
    graph.add_file_dependency("code_search", "/path/to/file.py")

    # Get all tools affected by a file change
    dependents = graph.get_file_dependents("/path/to/file.py")
    # => {"code_search"}
"""

from __future__ import annotations

from collections import defaultdict


class ToolDependencyGraph:
    """Graph of tool dependencies for cascading invalidation.

    Tracks two types of dependencies:
    1. Tool-to-Tool: When tool A depends on tool B's results
    2. Tool-to-File: When tool A depends on a specific file's contents

    Example Tool-to-Tool dependencies:
        - code_search depends on read
        - test_runner depends on write
        - coverage depends on test_runner

    Example Tool-to-File dependencies:
        - code_search depends on source files
        - linter depends on configuration files
    """

    def __init__(self) -> None:
        """Initialize the dependency graph."""
        # tool -> set of tools it depends on
        self._tool_dependencies: dict[str, set[str]] = defaultdict(set)

        # tool -> set of files it depends on
        self._file_dependencies: dict[str, set[str]] = defaultdict(set)

        # file -> set of tools that depend on it (reverse index)
        self._file_dependents: dict[str, set[str]] = defaultdict(set)

        # tool -> set of tools that depend on it (reverse index)
        self._tool_dependents: dict[str, set[str]] = defaultdict(set)

    def add_tool_dependency(self, tool: str, depends_on: str) -> None:
        """Add a tool-to-tool dependency.

        When `depends_on` is invalidated, `tool` should also be invalidated.

        Args:
            tool: The tool that has a dependency
            depends_on: The tool that `tool` depends on

        Example:
            graph.add_tool_dependency("code_search", "read")
            # When read is invalidated, code_search should also be invalidated
        """
        self._tool_dependencies[tool].add(depends_on)
        self._tool_dependents[depends_on].add(tool)

    def add_file_dependency(self, tool: str, file_path: str) -> None:
        """Add a tool-to-file dependency.

        When `file_path` is modified, `tool` should be invalidated.

        Args:
            tool: The tool that depends on the file
            file_path: Path to the file the tool depends on

        Example:
            graph.add_file_dependency("code_search", "/src/main.py")
            # When /src/main.py is modified, code_search should be invalidated
        """
        self._file_dependencies[tool].add(file_path)
        self._file_dependents[file_path].add(tool)

    def get_dependents(self, tool: str) -> set[str]:
        """Get all tools that depend on the given tool.

        This is used for cascading invalidation. When a tool is
        invalidated, all its dependents should also be invalidated.

        Args:
            tool: The tool to query

        Returns:
            Set of tool names that depend on this tool

        Example:
            graph.add_tool_dependency("code_search", "read")
            graph.add_tool_dependency("ast_analysis", "read")

            dependents = graph.get_dependents("read")
            # => {"code_search", "ast_analysis"}
        """
        return self._tool_dependents.get(tool, set()).copy()

    def get_file_dependents(self, file_path: str) -> set[str]:
        """Get all tools that depend on the given file.

        This is used for file-based cache invalidation. When a file
        is modified, all tools that depend on it should be invalidated.

        Args:
            file_path: The file path to query

        Returns:
            Set of tool names that depend on this file

        Example:
            graph.add_file_dependency("code_search", "/src/main.py")
            graph.add_file_dependency("linter", "/src/main.py")

            dependents = graph.get_file_dependents("/src/main.py")
            # => {"code_search", "linter"}
        """
        return self._file_dependents.get(file_path, set()).copy()

    def get_transitive_dependents(
        self,
        tool: str,
        visited: set[str] | None = None,
        result: set[str] | None = None,
    ) -> set[str]:
        """Get all transitive dependents of a tool.

        This includes not only direct dependents, but also tools
        that depend on those dependents, and so on recursively.

        Args:
            tool: The tool to query
            visited: Internal recursion tracking for cycle detection (do not pass)
            result: Internal accumulation of transitive dependents (do not pass)

        Returns:
            Set of all tool names that transitively depend on this tool

        Example:
            graph.add_tool_dependency("test_runner", "write")
            graph.add_tool_dependency("coverage", "test_runner")

            dependents = graph.get_transitive_dependents("write")
            # => {"test_runner", "coverage"}
        """
        if visited is None:
            visited = set()
        if result is None:
            result = set()

        # Mark current tool as visited to detect cycles
        visited.add(tool)

        # Get direct dependents and recurse
        for dependent in self._tool_dependents.get(tool, set()):
            # Add dependent to result
            result.add(dependent)

            # Recurse if not already visited
            if dependent not in visited:
                self.get_transitive_dependents(dependent, visited, result)

        return result

    def get_dependencies(self, tool: str) -> set[str]:
        """Get all tools that the given tool depends on.

        Args:
            tool: The tool to query

        Returns:
            Set of tool names that this tool depends on

        Example:
            graph.add_tool_dependency("code_search", "read")
            graph.add_tool_dependency("code_search", "grep")

            deps = graph.get_dependencies("code_search")
            # => {"read", "grep"}
        """
        return self._tool_dependencies.get(tool, set()).copy()

    def get_file_dependencies(self, tool: str) -> set[str]:
        """Get all files that the given tool depends on.

        Args:
            tool: The tool to query

        Returns:
            Set of file paths that this tool depends on

        Example:
            graph.add_file_dependency("code_search", "/src/main.py")
            graph.add_file_dependency("code_search", "/src/utils.py")

            deps = graph.get_file_dependencies("code_search")
            # => {"/src/main.py", "/src/utils.py"}
        """
        return self._file_dependencies.get(tool, set()).copy()

    def remove_tool(self, tool: str) -> None:
        """Remove a tool from the graph.

        This removes all dependencies and dependents related to the tool.

        Args:
            tool: The tool to remove
        """
        # Remove from tool dependencies
        if tool in self._tool_dependencies:
            for depends_on in self._tool_dependencies[tool]:
                self._tool_dependents[depends_on].discard(tool)
            del self._tool_dependencies[tool]

        # Remove from tool dependents
        if tool in self._tool_dependents:
            for dependent in self._tool_dependents[tool]:
                self._tool_dependencies[dependent].discard(tool)
            del self._tool_dependents[tool]

        # Remove from file dependencies
        if tool in self._file_dependencies:
            for file_path in self._file_dependencies[tool]:
                self._file_dependents[file_path].discard(tool)
            del self._file_dependencies[tool]

    def remove_file_dependency(self, tool: str, file_path: str) -> None:
        """Remove a specific file dependency.

        Args:
            tool: The tool
            file_path: The file path to remove dependency for
        """
        self._file_dependencies[tool].discard(file_path)
        self._file_dependents[file_path].discard(tool)

    def clear(self) -> None:
        """Clear all dependencies from the graph."""
        self._tool_dependencies.clear()
        self._file_dependencies.clear()
        self._file_dependents.clear()
        self._tool_dependents.clear()

    def get_stats(self) -> dict[str, int]:
        """Get statistics about the dependency graph.

        Returns:
            Dictionary with graph statistics

        Example:
            stats = graph.get_stats()
            # {
            #     "tools": 10,
            #     "files": 25,
            #     "tool_dependencies": 15,
            #     "file_dependencies": 30
            # }
        """
        # Count unique tools across all tool dictionaries (including those with only file dependencies)
        # Note: _file_dependents contains files, not tools, so we don't include it here
        all_tools = (
            set(self._tool_dependencies.keys())
            | set(self._tool_dependents.keys())
            | set(self._file_dependencies.keys())
        )

        return {
            "tools": len(all_tools),
            "files": len(self._file_dependents),
            "tool_dependencies": sum(len(deps) for deps in self._tool_dependencies.values()),
            "file_dependencies": sum(len(deps) for deps in self._file_dependencies.values()),
        }


__all__ = ["ToolDependencyGraph"]
