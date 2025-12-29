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

"""Tests for victor.tools.tool_graph module."""

import pytest

from victor.tools.tool_graph import (
    ToolNode,
    ToolTransition,
    ToolExecutionGraph,
    ToolGraphRegistry,
)


@pytest.fixture
def graph():
    """Create a fresh graph for each test."""
    return ToolExecutionGraph("test")


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    ToolGraphRegistry.reset_instance()
    return ToolGraphRegistry.get_instance()


class TestToolNode:
    """Tests for ToolNode dataclass."""

    def test_tool_node_defaults(self):
        """ToolNode should have sensible defaults."""
        node = ToolNode(name="test")
        assert node.name == "test"
        assert node.depends_on == set()
        assert node.enables == set()
        assert node.inputs == set()
        assert node.outputs == set()
        assert node.weight == 1.0

    def test_tool_node_with_dependencies(self):
        """ToolNode should store dependencies correctly."""
        node = ToolNode(
            name="edit_files",
            depends_on={"read_file"},
            enables={"run_tests"},
            weight=0.9,
        )
        assert "read_file" in node.depends_on
        assert "run_tests" in node.enables


class TestToolTransition:
    """Tests for ToolTransition dataclass."""

    def test_transition_defaults(self):
        """ToolTransition should have sensible defaults."""
        trans = ToolTransition(from_tool="read", to_tool="edit")
        assert trans.from_tool == "read"
        assert trans.to_tool == "edit"
        assert trans.weight == 0.5
        assert trans.relationship == "transition"


class TestToolExecutionGraph:
    """Tests for ToolExecutionGraph class."""

    def test_add_node(self, graph):
        """add_node should create node."""
        node = graph.add_node("read_file")
        assert node.name == "read_file"
        assert graph.node_count == 1

    def test_add_node_with_dependencies(self, graph):
        """add_node should store dependencies."""
        graph.add_node(
            "edit_files",
            depends_on={"read_file"},
            enables={"run_tests"},
        )
        prereqs = graph.get_prerequisites("edit_files")
        assert "read_file" in prereqs

    def test_add_dependency(self, graph):
        """add_dependency should create dependency."""
        graph.add_dependency(
            "edit_files",
            depends_on={"read_file"},
            enables={"run_tests"},
            weight=0.9,
        )
        prereqs = graph.get_prerequisites("edit_files")
        assert "read_file" in prereqs

    def test_add_dependencies_bulk(self, graph):
        """add_dependencies should add multiple."""
        graph.add_dependencies([
            ("edit", {"read"}, {"test"}, 0.9),
            ("test", {"edit"}, set(), 0.8),
        ])
        assert graph.node_count == 2

    def test_add_transitions_devops_format(self, graph):
        """add_transitions should handle devops format."""
        graph.add_transitions({
            "read_file": [("edit_files", 0.8), ("code_search", 0.6)],
            "edit_files": [("run_tests", 0.9)],
        })
        assert graph.transition_count >= 3

    def test_add_sequence(self, graph):
        """add_sequence should create transitions."""
        graph.add_sequence(["list_directory", "read_file", "edit_files"])
        weight = graph.get_transition_weight("read_file", "edit_files")
        assert weight >= 0.7  # Default sequence weight is 0.7

    def test_add_sequences_bulk(self, graph):
        """add_sequences should add multiple."""
        graph.add_sequences([
            ["read", "edit", "test"],
            ["list", "read", "write"],
        ])
        assert graph.transition_count > 0

    def test_add_io_tool(self, graph):
        """add_io_tool should create IO-based transitions."""
        graph.add_io_tool("read_file", inputs=set(), outputs={"file_content"})
        graph.add_io_tool("edit_files", inputs={"file_content"}, outputs={"modified_file"})
        # Should create transition from read_file to edit_files
        weight = graph.get_transition_weight("read_file", "edit_files")
        assert weight > 0.3

    def test_add_cluster(self, graph):
        """add_cluster should group tools."""
        graph.add_cluster("git", {"git_status", "git_diff", "git_commit"})
        related = graph.get_cluster_tools("git_status")
        assert "git_diff" in related
        assert "git_commit" in related

    def test_get_prerequisites(self, graph):
        """get_prerequisites should return dependencies."""
        graph.add_dependency("edit", depends_on={"read"})
        prereqs = graph.get_prerequisites("edit")
        assert "read" in prereqs

    def test_get_enabled_tools(self, graph):
        """get_enabled_tools should return enabled tools."""
        graph.add_dependency("read", enables={"edit", "search"})
        enabled = graph.get_enabled_tools("read")
        assert "edit" in enabled
        assert "search" in enabled

    def test_get_dependents(self, graph):
        """get_dependents should return tools that depend on given tool."""
        graph.add_dependency("edit", depends_on={"read"})
        graph.add_dependency("write", depends_on={"read"})
        dependents = graph.get_dependents("read")
        assert "edit" in dependents
        assert "write" in dependents

    def test_suggest_next_tools_basic(self, graph):
        """suggest_next_tools should return ranked suggestions."""
        graph.add_dependency("edit", depends_on={"read"}, weight=0.9)
        graph.add_dependency("search", depends_on={"read"}, weight=0.7)
        suggestions = graph.suggest_next_tools("read")
        assert len(suggestions) >= 2
        # First suggestion should be edit (higher weight)
        tools = [s[0] for s in suggestions]
        assert "edit" in tools

    def test_suggest_next_tools_avoids_recent(self, graph):
        """suggest_next_tools should deprioritize recent tools."""
        graph.add_transitions({
            "read": [("edit", 0.9), ("search", 0.8)],
        })
        suggestions = graph.suggest_next_tools("read", history=["edit"])
        # edit should have lower score due to history
        for tool, score in suggestions:
            if tool == "edit":
                assert score < 0.9  # Reduced

    def test_suggest_next_tools_with_available_filter(self, graph):
        """suggest_next_tools should filter by available tools."""
        graph.add_transitions({
            "read": [("edit", 0.9), ("search", 0.8)],
        })
        suggestions = graph.suggest_next_tools(
            "read",
            available_tools={"search"},
        )
        tools = [s[0] for s in suggestions]
        assert "edit" not in tools
        assert "search" in tools

    def test_get_transition_weight_direct(self, graph):
        """get_transition_weight should return direct transition weight."""
        graph.add_transitions({"read": [("edit", 0.85)]})
        weight = graph.get_transition_weight("read", "edit")
        assert weight == 0.85

    def test_get_transition_weight_from_dependency(self, graph):
        """get_transition_weight should check dependencies."""
        graph.add_dependency("edit", depends_on={"read"}, weight=0.9)
        weight = graph.get_transition_weight("read", "edit")
        assert weight == 0.9 * 0.9  # weight * 0.9 for depends_on

    def test_get_transition_weight_from_enables(self, graph):
        """get_transition_weight should check enables."""
        graph.add_dependency("read", enables={"edit"}, weight=0.9)
        weight = graph.get_transition_weight("read", "edit")
        assert weight == 0.9 * 0.8  # weight * 0.8 for enables

    def test_get_transition_weight_from_sequence(self, graph):
        """get_transition_weight should check sequences."""
        graph.add_sequence(["read", "edit", "test"])
        weight = graph.get_transition_weight("read", "edit")
        assert weight == 0.7  # Default sequence weight creates direct transition

    def test_get_transition_weight_default(self, graph):
        """get_transition_weight should return default for unknown."""
        weight = graph.get_transition_weight("unknown1", "unknown2")
        assert weight == 0.3

    def test_plan_for_goal_simple(self, graph):
        """plan_for_goal should return tool sequence."""
        graph.add_dependency("edit", depends_on={"read"})
        graph.add_dependency("test", depends_on={"edit"})
        plan = graph.plan_for_goal({"test"})
        assert "read" in plan
        assert "edit" in plan
        assert plan.index("read") < plan.index("edit")

    def test_plan_for_goal_with_current_state(self, graph):
        """plan_for_goal should skip already executed tools."""
        graph.add_dependency("edit", depends_on={"read"})
        graph.add_dependency("test", depends_on={"edit"})
        plan = graph.plan_for_goal({"test"}, current_state={"read"})
        assert "read" not in plan
        assert "edit" in plan

    def test_validate_execution_satisfied(self, graph):
        """validate_execution should return True when prereqs met."""
        graph.add_dependency("edit", depends_on={"read"})
        is_valid, missing = graph.validate_execution("edit", {"read"})
        assert is_valid is True
        assert missing == []

    def test_validate_execution_missing(self, graph):
        """validate_execution should return False when prereqs missing."""
        graph.add_dependency("edit", depends_on={"read"})
        is_valid, missing = graph.validate_execution("edit", set())
        assert is_valid is False
        assert "read" in missing

    def test_merge_graphs(self, graph):
        """merge should combine two graphs."""
        other = ToolExecutionGraph("other")
        other.add_dependency("special_tool", depends_on={"read"})
        other.add_sequence(["a", "b", "c"])

        graph.add_dependency("edit", depends_on={"read"})
        graph.merge(other)

        assert graph.get_node("special_tool") is not None
        assert graph.get_node("edit") is not None

    def test_to_dict(self, graph):
        """to_dict should export graph correctly."""
        graph.add_dependency("edit", depends_on={"read"}, enables={"test"})
        graph.add_sequence(["a", "b"])
        graph.add_cluster("git", {"status", "diff"})

        d = graph.to_dict()
        assert d["name"] == "test"
        assert "edit" in d["nodes"]
        assert len(d["sequences"]) == 1
        assert "git" in d["clusters"]

    def test_from_dict(self, graph):
        """from_dict should recreate graph."""
        graph.add_dependency("edit", depends_on={"read"})
        d = graph.to_dict()

        recreated = ToolExecutionGraph.from_dict(d)
        assert recreated.name == "test"
        prereqs = recreated.get_prerequisites("edit")
        assert "read" in prereqs

    def test_all_tools_property(self, graph):
        """all_tools should return all tool names."""
        graph.add_node("read")
        graph.add_node("edit")
        graph.add_node("test")
        assert graph.all_tools == {"read", "edit", "test"}


class TestToolGraphRegistry:
    """Tests for ToolGraphRegistry class."""

    def test_singleton_pattern(self, registry):
        """Registry should be a singleton."""
        registry2 = ToolGraphRegistry.get_instance()
        assert registry is registry2

    def test_reset_instance(self, registry):
        """reset_instance should create new instance."""
        ToolGraphRegistry.reset_instance()
        registry2 = ToolGraphRegistry.get_instance()
        assert registry is not registry2

    def test_register_and_get_graph(self, registry):
        """register_graph and get_graph should work."""
        graph = ToolExecutionGraph("coding")
        graph.add_node("read")
        registry.register_graph("coding", graph)

        retrieved = registry.get_graph("coding")
        assert retrieved is graph

    def test_get_graph_fallback(self, registry):
        """get_graph should return default for unknown."""
        graph = registry.get_graph("unknown")
        assert graph is not None  # Returns default graph

    def test_get_merged_graph(self, registry):
        """get_merged_graph should combine graphs."""
        coding = ToolExecutionGraph("coding")
        coding.add_node("edit")
        registry.register_graph("coding", coding)

        devops = ToolExecutionGraph("devops")
        devops.add_node("bash")
        registry.register_graph("devops", devops)

        merged = registry.get_merged_graph(["coding", "devops"])
        assert merged.get_node("edit") is not None
        assert merged.get_node("bash") is not None

    def test_list_graphs(self, registry):
        """list_graphs should return registered names."""
        registry.register_graph("coding", ToolExecutionGraph("coding"))
        registry.register_graph("devops", ToolExecutionGraph("devops"))
        graphs = registry.list_graphs()
        assert "coding" in graphs
        assert "devops" in graphs
