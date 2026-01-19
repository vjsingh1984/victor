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

"""Tests for tool execution graphs."""

import pytest

from victor.agent.tool_graph import (
    CacheStrategy,
    ToolDependency,
    ToolExecutionGraph,
    ToolExecutionNode,
    ValidationRule,
    ValidationRuleType,
)


class TestValidationRule:
    """Tests for ValidationRule."""

    def test_creation(self):
        """Test ValidationRule creation."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.REQUIRED,
            parameter="test_param",
            constraint="required",
        )
        assert rule.rule_type == ValidationRuleType.REQUIRED
        assert rule.parameter == "test_param"
        assert rule.constraint == "required"

    def test_to_dict(self):
        """Test ValidationRule serialization."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.REQUIRED,
            parameter="test_param",
            constraint="required",
            error_message="Test error",
        )
        data = rule.to_dict()
        assert data["rule_type"] == "required"
        assert data["parameter"] == "test_param"
        assert data["constraint"] == "required"
        assert data["error_message"] == "Test error"

    def test_from_dict(self):
        """Test ValidationRule deserialization."""
        data = {
            "rule_type": "required",
            "parameter": "test_param",
            "constraint": "required",
            "error_message": "Test error",
        }
        rule = ValidationRule.from_dict(data)
        assert rule.rule_type == ValidationRuleType.REQUIRED
        assert rule.parameter == "test_param"


class TestToolExecutionNode:
    """Tests for ToolExecutionNode."""

    def test_creation(self):
        """Test node creation."""
        node = ToolExecutionNode(tool_name="test")
        assert node.tool_name == "test"
        assert node.normalization_strategy == "auto"
        assert node.cache_policy == "default"

    def test_hashable(self):
        """Test node is hashable."""
        node = ToolExecutionNode(tool_name="test")
        node_set = {node}
        assert len(node_set) == 1

    def test_equality(self):
        """Test node equality."""
        node1 = ToolExecutionNode(tool_name="test", cache_policy="idempotent")
        node2 = ToolExecutionNode(tool_name="test", cache_policy="idempotent")
        node3 = ToolExecutionNode(tool_name="test", cache_policy="default")

        assert node1 == node2
        assert node1 != node3

    def test_to_dict(self):
        """Test node serialization."""
        node = ToolExecutionNode(
            tool_name="test",
            timeout_seconds=60.0,
            cache_policy="idempotent",
        )
        data = node.to_dict()
        assert data["tool_name"] == "test"
        assert data["timeout_seconds"] == 60.0
        assert data["cache_policy"] == "idempotent"

    def test_from_dict(self):
        """Test node deserialization."""
        data = {
            "tool_name": "test",
            "validation_rules": [],
            "normalization_strategy": "auto",
            "cache_policy": "idempotent",
            "retry_policy": "default",
            "timeout_seconds": 60.0,
            "metadata": {},
        }
        node = ToolExecutionNode.from_dict(data)
        assert node.tool_name == "test"
        assert node.cache_policy == "idempotent"
        assert node.timeout_seconds == 60.0


class TestToolDependency:
    """Tests for ToolDependency."""

    def test_creation(self):
        """Test dependency creation."""
        dep = ToolDependency(from_node="tool1", to_node="tool2")
        assert dep.from_node == "tool1"
        assert dep.to_node == "tool2"
        assert dep.condition is None

    def test_with_condition(self):
        """Test dependency with condition."""
        dep = ToolDependency(from_node="tool1", to_node="tool2", condition="success")
        assert dep.condition == "success"

    def test_to_dict(self):
        """Test dependency serialization."""
        dep = ToolDependency(from_node="tool1", to_node="tool2", condition="success")
        data = dep.to_dict()
        assert data["from"] == "tool1"
        assert data["to"] == "tool2"
        assert data["condition"] == "success"

    def test_from_dict(self):
        """Test dependency deserialization."""
        data = {"from": "tool1", "to": "tool2", "condition": "success"}
        dep = ToolDependency.from_dict(data)
        assert dep.from_node == "tool1"
        assert dep.to_node == "tool2"
        assert dep.condition == "success"


class TestToolExecutionGraph:
    """Tests for ToolExecutionGraph."""

    def test_creation(self):
        """Test graph creation."""
        nodes = [ToolExecutionNode(tool_name="test")]
        graph = ToolExecutionGraph(nodes=nodes)
        assert len(graph.nodes) == 1
        assert graph.cache_strategy == CacheStrategy.ADAPTIVE

    def test_validation_empty_nodes(self):
        """Test graph validation fails with empty nodes."""
        with pytest.raises(ValueError, match="must have at least one node"):
            ToolExecutionGraph(nodes=[])

    def test_get_node(self):
        """Test getting node by name."""
        node1 = ToolExecutionNode(tool_name="tool1")
        node2 = ToolExecutionNode(tool_name="tool2")
        graph = ToolExecutionGraph(nodes=[node1, node2])

        found = graph.get_node("tool1")
        assert found is not None
        assert found.tool_name == "tool1"

        not_found = graph.get_node("tool3")
        assert not_found is None

    def test_get_dependencies(self):
        """Test getting dependencies for a tool."""
        node1 = ToolExecutionNode(tool_name="tool1")
        node2 = ToolExecutionNode(tool_name="tool2")
        edge1 = ToolDependency(from_node="tool1", to_node="tool2")
        edge2 = ToolDependency(from_node="tool1", to_node="tool3")

        graph = ToolExecutionGraph(nodes=[node1, node2], edges=[edge1, edge2])

        deps = graph.get_dependencies("tool1")
        assert len(deps) == 2
        assert deps[0].to_node == "tool2"
        assert deps[1].to_node == "tool3"

    def test_get_dependents(self):
        """Test getting tools that depend on this tool."""
        node1 = ToolExecutionNode(tool_name="tool1")
        node2 = ToolExecutionNode(tool_name="tool2")
        edge1 = ToolDependency(from_node="tool1", to_node="tool2")
        edge2 = ToolDependency(from_node="tool3", to_node="tool2")

        graph = ToolExecutionGraph(nodes=[node1, node2], edges=[edge1, edge2])

        dependents = graph.get_dependents("tool2")
        assert len(dependents) == 2
        assert dependents[0].from_node == "tool1"
        assert dependents[1].from_node == "tool3"

    def test_serialization(self):
        """Test graph serialization round-trip."""
        node = ToolExecutionNode(tool_name="test")
        edge = ToolDependency(from_node="test", to_node="test2")
        graph = ToolExecutionGraph(nodes=[node], edges=[edge])

        data = graph.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert "version" in data
        assert "node_count" in data
        assert "edge_count" in data

        # Round-trip
        restored = ToolExecutionGraph.from_dict(data)
        assert len(restored.nodes) == 1
        assert len(restored.edges) == 1
        assert restored.version == "1.0"

    def test_hash(self):
        """Test graph is hashable."""
        node1 = ToolExecutionNode(tool_name="tool1")
        graph1 = ToolExecutionGraph(nodes=[node1])
        graph2 = ToolExecutionGraph(nodes=[node1])

        # Same nodes should produce same hash
        assert hash(graph1) == hash(graph2)

    def test_equality(self):
        """Test graph equality."""
        node1 = ToolExecutionNode(tool_name="tool1")
        graph1 = ToolExecutionGraph(nodes=[node1])
        graph2 = ToolExecutionGraph(nodes=[node1])

        assert graph1 == graph2

        node2 = ToolExecutionNode(tool_name="tool2")
        graph3 = ToolExecutionGraph(nodes=[node2])

        assert graph1 != graph3
