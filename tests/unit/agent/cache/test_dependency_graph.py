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

"""Tests for ToolDependencyGraph (cache invalidation).

Tests the dependency graph for cascading cache invalidation.
"""

import pytest

from victor.agent.cache.dependency_graph import ToolDependencyGraph


class TestToolDependencyGraph:
    """Tests for ToolDependencyGraph."""

    def test_init_empty(self):
        """Test initialization creates empty graph."""
        graph = ToolDependencyGraph()

        stats = graph.get_stats()
        assert stats["tools"] == 0
        assert stats["files"] == 0
        assert stats["tool_dependencies"] == 0
        assert stats["file_dependencies"] == 0

    def test_add_tool_dependency(self):
        """Test adding tool-to-tool dependency."""
        graph = ToolDependencyGraph()
        graph.add_tool_dependency("code_search", "read")

        stats = graph.get_stats()
        assert stats["tool_dependencies"] == 1
        assert stats["tools"] == 2  # code_search and read

    def test_add_file_dependency(self):
        """Test adding tool-to-file dependency."""
        graph = ToolDependencyGraph()
        graph.add_file_dependency("code_search", "/src/main.py")

        stats = graph.get_stats()
        assert stats["file_dependencies"] == 1
        assert stats["files"] == 1

    def test_get_dependents(self):
        """Test getting tools that depend on a given tool."""
        graph = ToolDependencyGraph()
        graph.add_tool_dependency("code_search", "read")
        graph.add_tool_dependency("ast_analysis", "read")

        dependents = graph.get_dependents("read")
        assert dependents == {"code_search", "ast_analysis"}

    def test_get_dependents_empty(self):
        """Test getting dependents for tool with no dependents."""
        graph = ToolDependencyGraph()
        graph.add_tool_dependency("code_search", "read")

        dependents = graph.get_dependents("code_search")
        assert dependents == set()

    def test_get_file_dependents(self):
        """Test getting tools that depend on a file."""
        graph = ToolDependencyGraph()
        graph.add_file_dependency("code_search", "/src/main.py")
        graph.add_file_dependency("linter", "/src/main.py")

        dependents = graph.get_file_dependents("/src/main.py")
        assert dependents == {"code_search", "linter"}

    def test_get_file_dependents_empty(self):
        """Test getting file dependents for file with no dependents."""
        graph = ToolDependencyGraph()
        graph.add_file_dependency("code_search", "/src/main.py")

        dependents = graph.get_file_dependents("/src/utils.py")
        assert dependents == set()

    def test_get_transitive_dependents(self):
        """Test getting transitive dependents."""
        graph = ToolDependencyGraph()
        graph.add_tool_dependency("coverage", "test_runner")
        graph.add_tool_dependency("test_runner", "write")

        dependents = graph.get_transitive_dependents("write")
        assert dependents == {"test_runner", "coverage"}

    def test_get_transitive_dependents_empty(self):
        """Test transitive dependents for tool with no dependents."""
        graph = ToolDependencyGraph()
        graph.add_tool_dependency("code_search", "read")

        dependents = graph.get_transitive_dependents("code_search")
        assert dependents == set()

    def test_get_dependencies(self):
        """Test getting tools that a given tool depends on."""
        graph = ToolDependencyGraph()
        graph.add_tool_dependency("code_search", "read")
        graph.add_tool_dependency("code_search", "grep")

        dependencies = graph.get_dependencies("code_search")
        assert dependencies == {"read", "grep"}

    def test_get_file_dependencies(self):
        """Test getting files that a tool depends on."""
        graph = ToolDependencyGraph()
        graph.add_file_dependency("code_search", "/src/main.py")
        graph.add_file_dependency("code_search", "/src/utils.py")

        dependencies = graph.get_file_dependencies("code_search")
        assert dependencies == {"/src/main.py", "/src/utils.py"}

    def test_remove_tool(self):
        """Test removing a tool from the graph."""
        graph = ToolDependencyGraph()
        graph.add_tool_dependency("code_search", "read")
        graph.add_tool_dependency("ast_analysis", "read")

        graph.remove_tool("read")

        stats = graph.get_stats()
        assert stats["tools"] == 2  # code_search and ast_analysis
        assert stats["tool_dependencies"] == 0

    def test_remove_file_dependency(self):
        """Test removing a specific file dependency."""
        graph = ToolDependencyGraph()
        graph.add_file_dependency("code_search", "/src/main.py")
        graph.add_file_dependency("code_search", "/src/utils.py")

        graph.remove_file_dependency("code_search", "/src/main.py")

        dependencies = graph.get_file_dependencies("code_search")
        assert dependencies == {"/src/utils.py"}

    def test_clear(self):
        """Test clearing the entire graph."""
        graph = ToolDependencyGraph()
        graph.add_tool_dependency("code_search", "read")
        graph.add_file_dependency("code_search", "/src/main.py")

        graph.clear()

        stats = graph.get_stats()
        assert stats["tools"] == 0
        assert stats["files"] == 0
        assert stats["tool_dependencies"] == 0
        assert stats["file_dependencies"] == 0

    def test_get_stats(self):
        """Test getting graph statistics."""
        graph = ToolDependencyGraph()
        graph.add_tool_dependency("code_search", "read")
        graph.add_tool_dependency("ast_analysis", "read")
        graph.add_file_dependency("code_search", "/src/main.py")
        graph.add_file_dependency("linter", "/src/main.py")

        stats = graph.get_stats()
        assert stats["tools"] == 4  # code_search, ast_analysis, read, linter
        assert stats["files"] == 1
        assert stats["tool_dependencies"] == 2
        assert stats["file_dependencies"] == 2  # code_search has 1, linter has 1

    def test_multiple_dependencies_same_tool(self):
        """Test adding multiple dependencies from the same tool."""
        graph = ToolDependencyGraph()
        graph.add_tool_dependency("code_search", "read")
        graph.add_tool_dependency("code_search", "grep")

        dependencies = graph.get_dependencies("code_search")
        assert len(dependencies) == 2

    def test_complex_dependency_chain(self):
        """Test a complex dependency chain."""
        graph = ToolDependencyGraph()
        # A depends on B, B depends on C, C depends on D
        graph.add_tool_dependency("a", "b")
        graph.add_tool_dependency("b", "c")
        graph.add_tool_dependency("c", "d")

        # Transitive dependents of D should include A, B, C
        dependents = graph.get_transitive_dependents("d")
        assert dependents == {"c", "b", "a"}
