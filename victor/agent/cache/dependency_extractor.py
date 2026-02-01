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

"""Dependency extractor for automatic file dependency tracking.

This module implements automatic extraction of file dependencies from
tool arguments, enabling intelligent cache invalidation.

Design Patterns:
    - Strategy Pattern: Different extraction strategies for different tools
    - SRP: Focused only on dependency extraction, not cache management

Usage:
    from victor.agent.cache.dependency_extractor import DependencyExtractor

    extractor = DependencyExtractor()

    # Extract dependencies from tool arguments
    deps = extractor.extract_file_dependencies(
        "read_file",
        {"path": "/src/main.py", "encoding": "utf-8"}
    )
    # => {"/src/main.py"}

    deps = extractor.extract_file_dependencies(
        "code_search",
        {"query": "auth", "files": ["/src/auth.py", "/src/login.py"]}
    )
    # => {"/src/auth.py", "/src/login.py"}
"""

from __future__ import annotations

from typing import Any

from victor.protocols import IDependencyExtractor


class DependencyExtractor(IDependencyExtractor):
    """Extract file dependencies from tool arguments.

    Analyzes tool arguments and automatically identifies file paths
    that should be tracked for cache invalidation.

    Supported Argument Patterns:
        - 'path': Single file path
        - 'file': Single file path
        - 'files': List of file paths
        - 'directory': Single directory path
        - 'dir': Single directory path
        - 'dirs': List of directory paths

    Attributes:
        _file_patterns: List of argument keys that indicate file paths
        _dir_patterns: List of argument keys that indicate directory paths
    """

    def __init__(self) -> None:
        """Initialize the dependency extractor."""
        # Common argument patterns for files
        self._file_patterns = [
            "path",
            "file",
            "files",
            "filepath",
            "file_path",
            "source",
            "sources",
            "target",
            "input",
            "inputs",
            "output",
        ]

        # Common argument patterns for directories
        self._dir_patterns = [
            "directory",
            "dir",
            "directories",
            "dirs",
            "folder",
            "folders",
            "root",
            "basedir",
            "base_dir",
            "workspace",
        ]

        # Combine all patterns
        self._all_patterns = set(self._file_patterns + self._dir_patterns)

    def extract_file_dependencies(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> set[str]:
        """Extract file paths from tool arguments.

        Analyzes tool arguments and identifies file paths that should
        be tracked for cache invalidation.

        Args:
            tool_name: Name of the tool being invoked
            arguments: Tool arguments dictionary

        Returns:
            Set of file paths extracted from arguments

        Example:
            >>> extractor = DependencyExtractor()
            >>> deps = extractor.extract_file_dependencies(
            ...     "read_file",
            ...     {"path": "/src/main.py", "encoding": "utf-8"}
            ... )
            >>> deps
            {'/src/main.py'}

            >>> deps = extractor.extract_file_dependencies(
            ...     "code_search",
            ...     {
            ...         "query": "authentication",
            ...         "files": ["/src/auth.py", "/src/login.py"]
            ...     }
            ... )
            >>> deps
            {'/src/auth.py', '/src/login.py'}
        """
        dependencies: set[str] = set()

        # Check all known file/directory argument patterns
        for pattern in self._all_patterns:
            if pattern in arguments:
                value = arguments[pattern]

                # Skip None values
                if value is None:
                    continue

                # Extract paths based on value type
                extracted = self._extract_paths_from_value(value)

                # Filter valid paths
                valid_paths = self._filter_valid_paths(extracted)
                dependencies.update(valid_paths)

        return dependencies

    def _extract_paths_from_value(self, value: Any) -> set[str]:
        """Extract paths from a value.

        Handles strings, lists, and other iterable types.

        Args:
            value: Value to extract paths from

        Returns:
            Set of extracted path strings
        """
        paths: set[str] = set()

        if isinstance(value, str):
            # Single path string
            if value.strip():  # Skip empty strings
                paths.add(value)

        elif isinstance(value, list | tuple | set):
            # Collection of paths
            for item in value:
                if isinstance(item, str) and item.strip():
                    paths.add(item)

        return paths

    def _filter_valid_paths(self, paths: set[str]) -> set[str]:
        """Filter and validate paths.

        Removes empty strings and obvious non-paths.

        Args:
            paths: Set of path strings to filter

        Returns:
            Set of valid path strings
        """
        valid: set[str] = set()

        for path in paths:
            # Skip empty strings
            if not path or not path.strip():
                continue

            # Skip strings that are clearly not paths
            stripped = path.strip()
            if len(stripped) < 1:
                continue

            # Accept the path
            valid.add(stripped)

        return valid


__all__ = ["DependencyExtractor"]
