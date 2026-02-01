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

"""Tests for shared tool defaults across verticals.

Tests the shared tool defaults infrastructure that verticals should use
to eliminate code duplication.
"""


from victor.core.tool_types import ToolDependency
from victor.core.verticals.defaults.tool_defaults import (
    COMMON_TOOL_CLUSTERS,
    COMMON_TOOL_DEPENDENCIES,
    COMMON_REQUIRED_TOOLS,
    COMMON_OPTIONAL_TOOLS,
    merge_clusters,
    merge_dependencies,
    merge_transitions,
    merge_required_tools,
)
from victor.framework.tool_naming import ToolNames


class TestCommonToolDefaults:
    """Tests for common tool default values."""

    def test_common_tool_clusters_contains_file_operations(self):
        """Verify COMMON_TOOL_CLUSTERS has file_operations cluster."""
        assert "file_operations" in COMMON_TOOL_CLUSTERS
        assert ToolNames.READ in COMMON_TOOL_CLUSTERS["file_operations"]
        assert ToolNames.WRITE in COMMON_TOOL_CLUSTERS["file_operations"]
        assert ToolNames.EDIT in COMMON_TOOL_CLUSTERS["file_operations"]
        assert ToolNames.LS in COMMON_TOOL_CLUSTERS["file_operations"]

    def test_common_tool_clusters_contains_search_operations(self):
        """Verify COMMON_TOOL_CLUSTERS has search_operations cluster."""
        assert "search_operations" in COMMON_TOOL_CLUSTERS
        assert ToolNames.GREP in COMMON_TOOL_CLUSTERS["search_operations"]
        assert ToolNames.CODE_SEARCH in COMMON_TOOL_CLUSTERS["search_operations"]

    def test_common_tool_clusters_contains_web_operations(self):
        """Verify COMMON_TOOL_CLUSTERS has web_operations cluster."""
        assert "web_operations" in COMMON_TOOL_CLUSTERS
        assert ToolNames.WEB_SEARCH in COMMON_TOOL_CLUSTERS["web_operations"]
        assert ToolNames.WEB_FETCH in COMMON_TOOL_CLUSTERS["web_operations"]

    def test_common_required_tools_contains_basic_tools(self):
        """Verify COMMON_REQUIRED_TOOLS has basic file operations."""
        assert ToolNames.READ in COMMON_REQUIRED_TOOLS
        assert ToolNames.WRITE in COMMON_REQUIRED_TOOLS
        assert ToolNames.EDIT in COMMON_REQUIRED_TOOLS
        assert ToolNames.LS in COMMON_REQUIRED_TOOLS

    def test_common_optional_tools_contains_search_tools(self):
        """Verify COMMON_OPTIONAL_TOOLS has search and analysis tools."""
        assert ToolNames.GREP in COMMON_OPTIONAL_TOOLS
        assert ToolNames.CODE_SEARCH in COMMON_OPTIONAL_TOOLS
        assert ToolNames.SYMBOL in COMMON_OPTIONAL_TOOLS


class TestMergeClusters:
    """Tests for merge_clusters utility."""

    def test_merge_clusters_combines_shared_and_vertical(self):
        """Verify merge_clusters() correctly combines defaults."""
        shared = {"file_operations": {ToolNames.READ, ToolNames.WRITE}}
        vertical = {"devops_operations": ["docker_build"]}

        result = merge_clusters(shared, vertical)

        assert "file_operations" in result
        assert "devops_operations" in result
        assert ToolNames.READ in result["file_operations"]
        assert ToolNames.WRITE in result["file_operations"]
        assert "docker_build" in result["devops_operations"]

    def test_merge_clusters_unions_same_cluster(self):
        """Verify merge_clusters() unions tools for same cluster name."""
        shared = {"file_operations": {ToolNames.READ, ToolNames.WRITE}}
        vertical = {"file_operations": {ToolNames.EDIT, ToolNames.LS}}

        result = merge_clusters(shared, vertical)

        assert len(result["file_operations"]) == 4
        assert ToolNames.READ in result["file_operations"]
        assert ToolNames.WRITE in result["file_operations"]
        assert ToolNames.EDIT in result["file_operations"]
        assert ToolNames.LS in result["file_operations"]


class TestMergeDependencies:
    """Tests for merge_dependencies utility."""

    def test_merge_dependencies_combines_lists(self):
        """Verify merge_dependencies() combines dependency lists."""
        base = COMMON_TOOL_DEPENDENCIES[:2]
        # Get the tool names from base for comparison
        base_tool_names = {dep.tool_name for dep in base}

        override = [
            ToolDependency(
                tool_name="custom_tool",
                depends_on=set(),
                enables=set(),
                weight=0.5,
            ),
        ]

        result = merge_dependencies(base, override)

        assert len(result) == 3
        # Check that base tool names are in result
        result_tool_names = {dep.tool_name for dep in result}
        assert base_tool_names.issubset(result_tool_names)
        assert "custom_tool" in result_tool_names


class TestMergeTransitions:
    """Tests for merge_transitions utility."""

    def test_merge_transitions_combines_dicts(self):
        """Verify merge_transitions() combines transition dicts."""
        base = {ToolNames.READ: [(ToolNames.EDIT, 0.4)]}
        override = {ToolNames.EDIT: [(ToolNames.TEST, 0.5)]}

        result = merge_transitions(base, override)

        assert ToolNames.READ in result
        assert ToolNames.EDIT in result
        assert result[ToolNames.EDIT] == [(ToolNames.TEST, 0.5)]


class TestMergeRequiredTools:
    """Tests for merge_required_tools utility."""

    def test_merge_required_tools_preserves_all(self):
        """Verify merge_required_tools() preserves all unique tools."""
        base = {ToolNames.READ, ToolNames.WRITE, ToolNames.EDIT}
        vertical_tools = [ToolNames.LS, ToolNames.SHELL]

        result = merge_required_tools(base, vertical_tools)

        assert len(result) == 5
        assert ToolNames.READ in result
        assert ToolNames.WRITE in result
        assert ToolNames.EDIT in result
        assert ToolNames.LS in result
        assert ToolNames.SHELL in result

    def test_merge_required_tools_removes_duplicates(self):
        """Verify merge_required_tools() removes duplicate tools."""
        base = {ToolNames.READ, ToolNames.WRITE, ToolNames.EDIT}
        vertical_tools = [ToolNames.READ, ToolNames.LS, ToolNames.SHELL]

        result = merge_required_tools(base, vertical_tools)

        # READ should only appear once
        assert result.count(ToolNames.READ) == 1
        assert len(result) == 5  # Not 6

    def test_merge_required_tools_base_first(self):
        """Verify merge_required_tools() puts base tools first."""
        base = {ToolNames.WRITE, ToolNames.EDIT}  # No READ
        vertical_tools = [ToolNames.LS, ToolNames.READ, ToolNames.SHELL]

        result = merge_required_tools(base, vertical_tools)

        # READ should be at index 2 (after WRITE and EDIT from base)
        read_index = result.index(ToolNames.READ)
        assert read_index >= 2

    def test_merge_required_tools_with_common_defaults(self):
        """Verify merge_required_tools() works with COMMON_REQUIRED_TOOLS."""
        vertical_tools = [ToolNames.SHELL, ToolNames.DOCKER]

        result = merge_required_tools(COMMON_REQUIRED_TOOLS, vertical_tools)

        # Should have all common tools plus vertical ones
        assert ToolNames.READ in result
        assert ToolNames.WRITE in result
        assert ToolNames.EDIT in result
        assert ToolNames.LS in result
        assert ToolNames.SHELL in result
        assert ToolNames.DOCKER in result

    def test_merge_required_tools_empty_vertical(self):
        """Verify merge_required_tools() works with empty vertical list."""
        result = merge_required_tools(COMMON_REQUIRED_TOOLS, [])

        # Should just have common tools
        assert len(result) == len(COMMON_REQUIRED_TOOLS)
        for tool in COMMON_REQUIRED_TOOLS:
            assert tool in result

    def test_merge_required_tools_preserves_order(self):
        """Verify merge_required_tools() preserves vertical tool order."""
        base = {ToolNames.READ, ToolNames.WRITE}
        vertical_tools = [ToolNames.SHELL, ToolNames.DOCKER, ToolNames.GIT]

        result = merge_required_tools(base, vertical_tools)

        # Vertical tools should maintain relative order after base tools
        shell_idx = result.index(ToolNames.SHELL)
        docker_idx = result.index(ToolNames.DOCKER)
        git_idx = result.index(ToolNames.GIT)

        assert shell_idx < docker_idx < git_idx
