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

"""Tests for DependencyExtractor implementation using TDD approach.

This test suite validates the dependency extraction functionality that
automatically identifies file dependencies from tool arguments.

Test Coverage:
    - Extract file paths from various argument patterns
    - Handle single files and lists of files
    - Extract directory paths
    - Handle glob patterns
    - Edge cases (empty arguments, missing keys, etc.)
"""

from __future__ import annotations

from typing import Any, Set

import pytest

from victor.protocols import IDependencyExtractor


# =============================================================================
# Test Fixtures
# =============================================================================

# Fixture is provided in conftest.py


# =============================================================================
# Common Argument Pattern Tests
# =============================================================================


class TestCommonArgumentPatterns:
    """Test extraction from common argument patterns."""

    def test_extract_single_file_path(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extracting a single file path from 'path' argument."""
        arguments = {
            "path": "/src/main.py",
            "encoding": "utf-8",
        }

        deps = dependency_extractor.extract_file_dependencies("read_file", arguments)

        assert isinstance(deps, set)
        assert "/src/main.py" in deps

    def test_extract_from_file_argument(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extracting file path from 'file' argument."""
        arguments = {
            "file": "/config/settings.yaml",
            "validate": True,
        }

        deps = dependency_extractor.extract_file_dependencies("load_config", arguments)

        assert isinstance(deps, set)
        assert "/config/settings.yaml" in deps

    def test_extract_from_files_list(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extracting multiple files from 'files' argument."""
        arguments = {
            "files": ["/src/auth.py", "/src/login.py", "/src/user.py"],
            "pattern": "class.*Auth",
        }

        deps = dependency_extractor.extract_file_dependencies("code_search", arguments)

        assert isinstance(deps, set)
        assert len(deps) == 3
        assert "/src/auth.py" in deps
        assert "/src/login.py" in deps
        assert "/src/user.py" in deps

    def test_extract_from_directory_argument(
        self,
        dependency_extractor: IDependencyExtractor,
    ) -> None:
        """Test extracting directory path from 'directory' argument."""
        arguments = {
            "directory": "/src",
            "recursive": True,
        }

        deps = dependency_extractor.extract_file_dependencies("list_files", arguments)

        assert isinstance(deps, set)
        assert "/src" in deps

    def test_extract_from_dir_argument(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extracting directory path from 'dir' argument."""
        arguments = {
            "dir": "/test/fixtures",
        }

        deps = dependency_extractor.extract_file_dependencies("run_tests", arguments)

        assert isinstance(deps, set)
        assert "/test/fixtures" in deps

    def test_extract_from_dirs_list(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extracting multiple directories from 'dirs' argument."""
        arguments = {
            "dirs": ["/src", "/tests", "/docs"],
        }

        deps = dependency_extractor.extract_file_dependencies("analyze", arguments)

        assert isinstance(deps, set)
        assert len(deps) == 3
        assert "/src" in deps
        assert "/tests" in deps
        assert "/docs" in deps


# =============================================================================
# Mixed Arguments Tests
# =============================================================================


class TestMixedArguments:
    """Test extraction from arguments with multiple file-related keys."""

    def test_extract_path_and_files(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extracting from both 'path' and 'files' arguments."""
        arguments = {
            "path": "/src/main.py",
            "files": ["/src/utils.py", "/src/helpers.py"],
            "output": "/tmp/results.json",
        }

        deps = dependency_extractor.extract_file_dependencies("multi_file_op", arguments)

        assert isinstance(deps, set)
        assert "/src/main.py" in deps
        assert "/src/utils.py" in deps
        assert "/src/helpers.py" in deps

    def test_extract_from_all_patterns(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extracting from all common patterns at once."""
        arguments = {
            "path": "/src/main.py",
            "file": "/config.yaml",
            "files": ["/src/auth.py", "/src/user.py"],
            "directory": "/src",
            "dir": "/tests",
            "dirs": ["/docs", "/examples"],
            "other": "ignored",
        }

        deps = dependency_extractor.extract_file_dependencies("complex_tool", arguments)

        assert isinstance(deps, set)
        assert "/src/main.py" in deps
        assert "/config.yaml" in deps
        assert "/src/auth.py" in deps
        assert "/src/user.py" in deps
        assert "/src" in deps
        assert "/tests" in deps
        assert "/docs" in deps
        assert "/examples" in deps
        assert "other" not in deps  # Non-file argument should be ignored


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_arguments(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extraction from empty arguments."""
        arguments: dict[str, Any] = {}

        deps = dependency_extractor.extract_file_dependencies("some_tool", arguments)

        assert isinstance(deps, set)
        assert len(deps) == 0

    def test_no_file_arguments(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test arguments without file-related keys."""
        arguments = {
            "query": "authentication",
            "limit": 10,
            "verbose": True,
        }

        deps = dependency_extractor.extract_file_dependencies("search", arguments)

        assert isinstance(deps, set)
        assert len(deps) == 0

    def test_empty_files_list(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test handling of empty 'files' list."""
        arguments = {
            "files": [],
            "pattern": "test",
        }

        deps = dependency_extractor.extract_file_dependencies("search", arguments)

        assert isinstance(deps, set)
        assert len(deps) == 0

    def test_none_values(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test handling of None values in arguments."""
        arguments = {
            "path": None,
            "files": None,
        }

        deps = dependency_extractor.extract_file_dependencies("some_tool", arguments)

        # Should handle None gracefully (either skip or include)
        assert isinstance(deps, set)

    def test_non_string_values(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test handling of non-string values."""
        arguments = {
            "path": 123,  # Integer instead of string
            "count": 5,
        }

        deps = dependency_extractor.extract_file_dependencies("some_tool", arguments)

        # Should either skip non-string values or convert them
        assert isinstance(deps, set)


# =============================================================================
# Tool-Specific Tests
# =============================================================================


class TestToolSpecificExtraction:
    """Test extraction for specific tool types."""

    def test_read_file_tool(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extraction for read_file tool."""
        arguments = {
            "path": "/src/main.py",
            "encoding": "utf-8",
            "offset": 0,
            "limit": 100,
        }

        deps = dependency_extractor.extract_file_dependencies("read_file", arguments)

        assert "/src/main.py" in deps

    def test_write_file_tool(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extraction for write_file tool."""
        arguments = {
            "path": "/src/new_file.py",
            "content": "print('hello')",
        }

        deps = dependency_extractor.extract_file_dependencies("write_file", arguments)

        assert "/src/new_file.py" in deps

    def test_grep_tool(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extraction for grep tool."""
        arguments = {
            "pattern": "TODO",
            "path": "/src",
            "file_pattern": "*.py",
        }

        deps = dependency_extractor.extract_file_dependencies("grep", arguments)

        assert "/src" in deps

    def test_code_search_tool(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extraction for code_search tool."""
        arguments = {
            "query": "authentication",
            "files": ["/src/auth.py", "/src/login.py"],
            "exclude": ["*/test_*.py"],
        }

        deps = dependency_extractor.extract_file_dependencies("code_search", arguments)

        assert "/src/auth.py" in deps
        assert "/src/login.py" in deps

    def test_glob_tool(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extraction for glob/list_files tool."""
        arguments = {
            "pattern": "*.py",
            "path": "/src",
            "recursive": True,
        }

        deps = dependency_extractor.extract_file_dependencies("glob", arguments)

        assert "/src" in deps


# =============================================================================
# Advanced Scenarios Tests
# =============================================================================


class TestAdvancedScenarios:
    """Test advanced extraction scenarios."""

    def test_relative_paths(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test handling of relative paths."""
        arguments = {
            "path": "./src/main.py",
        }

        deps = dependency_extractor.extract_file_dependencies("read_file", arguments)

        # Should extract the relative path as-is (normalization is optional)
        assert isinstance(deps, set)
        assert len(deps) >= 1

    def test_nested_paths_in_lists(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extraction from nested structures."""
        arguments = {
            "sources": {
                "files": ["/src/a.py", "/src/b.py"],
            },
        }

        deps = dependency_extractor.extract_file_dependencies("complex_tool", arguments)

        # May or may not extract from nested dicts (implementation dependent)
        assert isinstance(deps, set)

    def test_duplicate_paths(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test that duplicate paths are deduplicated."""
        arguments = {
            "path": "/src/main.py",
            "files": ["/src/main.py", "/src/utils.py", "/src/main.py"],
        }

        deps = dependency_extractor.extract_file_dependencies("some_tool", arguments)

        # Should deduplicate duplicates
        assert isinstance(deps, set)
        assert "/src/main.py" in deps
        # Count occurrences (should only appear once in set)
        main_count = sum(1 for p in deps if p == "/src/main.py")
        assert main_count == 1

    def test_mixed_valid_invalid_paths(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test handling of mixed valid and invalid paths."""
        arguments = {
            "files": [
                "/src/valid.py",
                "",  # Empty string
                "/also_valid.py",
                "   ",  # Whitespace
            ],
        }

        deps = dependency_extractor.extract_file_dependencies("some_tool", arguments)

        # Should extract valid paths and handle invalid ones gracefully
        assert isinstance(deps, set)
        assert "/src/valid.py" in deps
        assert "/also_valid.py" in deps


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Test performance characteristics."""

    def test_large_file_list(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extraction with large file list."""
        # Generate a large list of files
        files = [f"/src/file_{i}.py" for i in range(1000)]
        arguments = {
            "files": files,
        }

        deps = dependency_extractor.extract_file_dependencies("batch_process", arguments)

        # Should handle large lists efficiently
        assert isinstance(deps, set)
        assert len(deps) == 1000

    def test_many_arguments(self, dependency_extractor: IDependencyExtractor) -> None:
        """Test extraction with many arguments."""
        arguments = {f"file_{i}": f"/path/{i}.txt" for i in range(100)}
        arguments["other"] = "value"

        deps = dependency_extractor.extract_file_dependencies("multi_tool", arguments)

        # Should handle many arguments efficiently
        assert isinstance(deps, set)
