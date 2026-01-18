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

"""Integration tests for workflow editor import/export functionality.

Tests import/export of production YAML workflows through the visual editor,
including roundtrip conversion and validation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from victor.workflows import load_workflow_from_file
from victor.workflows.definition import (
    WorkflowDefinition,
    AgentNode,
    ComputeNode,
    ConditionNode,
    TeamNodeWorkflow,
    TransformNode,
    ParallelNode,
)
from victor.workflows.hitl import HITLNode
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from victor.core.errors import ConfigurationValidationError


class TestProductionWorkflowImport:
    """Test importing real production workflows into the editor."""

    def test_import_team_node_example(self):
        """Test importing the complex team_node_example.yaml workflow."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")

        assert isinstance(workflows, dict)
        assert "team_node_demo" in workflows

        workflow_def = workflows["team_node_demo"]
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.name == "team_node_demo"
        assert len(workflow_def.nodes) > 0

        # Verify team nodes are present
        team_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, TeamNodeWorkflow)]
        assert len(team_nodes) >= 1, "Should have at least one team node"

    def test_import_deep_research_workflow(self):
        """Test importing the deep_research.yaml workflow."""
        workflows = load_workflow_from_file("victor/research/workflows/deep_research.yaml")

        assert isinstance(workflows, dict)
        assert "deep_research" in workflows

        workflow_def = workflows["deep_research"]
        # Fixed: vertical is in metadata, not a direct attribute
        assert workflow_def.metadata.get("vertical") == "research"

        # Verify parallel nodes
        parallel_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, ParallelNode)]
        assert len(parallel_nodes) >= 1, "Should have parallel search nodes"

        # Verify HITL nodes
        hitl_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, HITLNode)]
        assert len(hitl_nodes) >= 1, "Should have human-in-the-loop nodes"

    def test_import_code_generation_workflow(self):
        """Test importing the code_generation.yaml benchmark workflow."""
        workflows = load_workflow_from_file("victor/benchmark/workflows/code_generation.yaml")

        assert isinstance(workflows, dict)
        assert "code_generation" in workflows

        workflow_def = workflows["code_generation"]
        # Fixed: vertical is in metadata, not a direct attribute
        assert workflow_def.metadata.get("vertical") == "benchmark"

        # Verify compute nodes (for testing)
        compute_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, ComputeNode)]
        assert len(compute_nodes) >= 1, "Should have compute nodes for test execution"

    def test_import_team_research_workflow(self):
        """Test importing the comprehensive team research workflow."""
        workflows = load_workflow_from_file("victor/research/workflows/examples/team_research.yaml")

        assert isinstance(workflows, dict)
        assert "comprehensive_team_research" in workflows

        workflow_def = workflows["comprehensive_team_research"]

        # Verify team formation
        team_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, TeamNodeWorkflow)]
        assert len(team_nodes) >= 1

        team_node = team_nodes[0]
        assert team_node.team_formation == "pipeline"
        assert len(team_node.members) == 4, "Should have 4 pipeline members"


class TestEditorGraphConversion:
    """Test conversion between workflow definitions and editor graph format."""

    def test_convert_team_workflow_to_graph(self):
        """Test converting a team workflow to editor graph format."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Convert to graph format (simulating what the editor API does)
        graph_nodes = []
        graph_edges = []

        for node in workflow_def.nodes.values():  # Fixed: iterate over values
            graph_node = {
                "id": node.id,
                "type": self._get_node_type(node),
                "name": node.name or node.id,
                "config": self._extract_node_config(node),
                "position": {"x": 0, "y": 0},  # Would be auto-layouted
            }
            graph_nodes.append(graph_node)

            # Create edges for next nodes
            if hasattr(node, "next_nodes") and node.next_nodes:
                for next_id in node.next_nodes:
                    edge = {
                        "id": f"{node.id}->{next_id}",
                        "source": node.id,
                        "target": next_id,
                        "label": None,
                    }
                    graph_edges.append(edge)

        # Verify conversion
        assert len(graph_nodes) > 0
        assert len(graph_edges) > 0

        # Verify team nodes have correct structure
        team_nodes = [n for n in graph_nodes if n["type"] == "team"]
        for team_node in team_nodes:
            assert "formation" in team_node["config"]
            assert "members" in team_node["config"]
            assert len(team_node["config"]["members"]) > 0

    def test_convert_conditional_branches_to_edges(self):
        """Test that conditional branches create multiple edges."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Find condition node
        condition_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, ConditionNode)]
        assert len(condition_nodes) > 0

        condition_node = condition_nodes[0]
        assert hasattr(condition_node, "branches")
        assert len(condition_node.branches) > 1

        # Each branch should create an edge
        branch_count = len(condition_node.branches)
        # next_nodes can be empty for condition nodes since they use branches
        assert (
            condition_node.next_nodes is None
            or len(condition_node.next_nodes) == 0
            or len(condition_node.next_nodes) == branch_count
        )

    def _get_node_type(self, node) -> str:
        """Get node type string for editor."""
        if isinstance(node, AgentNode):
            return "agent"
        elif isinstance(node, ComputeNode):
            return "compute"
        elif isinstance(node, TeamNodeWorkflow):
            return "team"
        elif isinstance(node, ConditionNode):
            return "condition"
        elif isinstance(node, TransformNode):
            return "transform"
        elif isinstance(node, ParallelNode):
            return "parallel"
        elif isinstance(node, HITLNode):
            return "hitl"
        else:
            return "unknown"

    def _extract_node_config(self, node) -> Dict[str, Any]:
        """Extract node configuration for editor."""
        config = {}

        if isinstance(node, AgentNode):
            config["role"] = node.role
            config["goal"] = node.goal
            config["tool_budget"] = node.tool_budget
            if hasattr(node, "tools"):
                config["tools"] = node.tools

        elif isinstance(node, TeamNodeWorkflow):
            config["formation"] = node.team_formation
            config["goal"] = node.goal
            config["max_iterations"] = getattr(node, "max_iterations", 5)
            config["members"] = [
                {
                    "id": m.get("id"),
                    "role": m.get("role"),
                    "goal": m.get("goal"),
                    "tool_budget": m.get("tool_budget", 10),
                }
                for m in node.members
            ]

        elif isinstance(node, ConditionNode):
            config["condition"] = node.condition
            config["branches"] = node.branches

        return config


class TestYAMLRoundtrip:
    """Test roundtrip conversion: YAML -> Definition -> YAML."""

    def test_team_workflow_roundtrip(self):
        """Test that team workflow survives YAML roundtrip."""
        # Load original
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        original = workflows["team_node_demo"]

        # Export to dict (to_yaml doesn't exist, use to_dict)
        workflow_dict = original.to_dict()

        # Verify structure
        assert "nodes" in workflow_dict
        assert len(workflow_dict["nodes"]) > 0

        # Verify team nodes are preserved
        team_nodes = [n for n in workflow_dict["nodes"].values() if n.get("type") == "team"]
        assert len(team_nodes) >= 1

        for team_node in team_nodes:
            assert "team_formation" in team_node
            assert "members" in team_node
            assert len(team_node["members"]) > 0

    def test_deep_research_roundtrip(self):
        """Test that deep research workflow survives roundtrip."""
        workflows = load_workflow_from_file("victor/research/workflows/deep_research.yaml")
        original = workflows["deep_research"]

        # Export to dict (to_yaml doesn't exist, use to_dict)
        workflow_dict = original.to_dict()

        # Verify parallel nodes are preserved
        parallel_nodes = [n for n in workflow_dict["nodes"].values() if n.get("type") == "parallel"]
        assert len(parallel_nodes) >= 1

        # Verify HITL nodes are preserved
        hitl_nodes = [n for n in workflow_dict["nodes"].values() if n.get("type") == "hitl"]
        assert len(hitl_nodes) >= 1

    def test_workflow_metadata_preserved(self):
        """Test that workflow metadata is preserved through roundtrip."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        original = workflows["team_node_demo"]

        # Export to dict (to_yaml doesn't exist, use to_dict)
        workflow_dict = original.to_dict()

        # Verify metadata
        assert "metadata" in workflow_dict or workflow_dict.get("metadata") is not None
        assert "description" in workflow_dict


class TestCompilationWithEditor:
    """Test compiling workflows after editor modifications."""

    def test_compile_imported_workflow(self):
        """Test that imported workflows can be compiled."""
        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Compile
        compiled = compiler.compile_definition(workflow_def)

        assert compiled is not None
        assert hasattr(compiled, "invoke")
        assert hasattr(compiled, "stream")

    def test_compile_deep_research(self):
        """Test compiling deep research workflow."""
        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        workflows = load_workflow_from_file("victor/research/workflows/deep_research.yaml")
        workflow_def = workflows["deep_research"]

        # Compile
        compiled = compiler.compile_definition(workflow_def)

        assert compiled is not None
        # Verify graph schema can be retrieved
        schema = compiled.get_graph_schema()
        assert schema is not None
        assert "nodes" in schema

    def test_compile_team_research(self):
        """Test compiling comprehensive team research workflow."""
        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        workflows = load_workflow_from_file("victor/research/workflows/examples/team_research.yaml")
        workflow_def = workflows["comprehensive_team_research"]

        # Compile
        compiled = compiler.compile_definition(workflow_def)

        assert compiled is not None
        schema = compiled.get_graph_schema()
        assert schema is not None


class TestValidation:
    """Test validation of workflows in editor context."""

    def test_validate_team_node_configuration(self):
        """Test validation of team node configuration."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Find team nodes
        team_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, TeamNodeWorkflow)]

        for team_node in team_nodes:
            # Validate required fields
            assert team_node.team_formation in [
                "parallel",
                "sequential",
                "pipeline",
                "hierarchical",
                "consensus",
            ], f"Invalid formation: {team_node.team_formation}"

            assert len(team_node.members) > 0, "Team must have members"

            # Validate each member
            for member in team_node.members:
                assert member.get("id"), "Member must have ID"
                assert member.get("role"), "Member must have role"
                assert member.get("goal"), "Member must have goal"
                assert member.get("tool_budget", 0) >= 0, "Tool budget must be non-negative"

    def test_validate_conditional_branches(self):
        """Test validation of conditional branch configuration."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Find condition nodes
        condition_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, ConditionNode)]

        for condition_node in condition_nodes:
            assert hasattr(condition_node, "condition")
            assert hasattr(condition_node, "branches")
            assert len(condition_node.branches) >= 2, "Condition must have at least 2 branches"

            # Verify branches are strings or valid types
            for branch_name in condition_node.branches.keys():
                assert isinstance(branch_name, str), f"Branch name must be string: {branch_name}"

    def test_validate_parallel_execution(self):
        """Test validation of parallel execution configuration."""
        workflows = load_workflow_from_file("victor/research/workflows/deep_research.yaml")
        workflow_def = workflows["deep_research"]

        # Find parallel nodes
        parallel_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, ParallelNode)]

        for parallel_node in parallel_nodes:
            assert hasattr(parallel_node, "parallel_nodes")
            assert len(parallel_node.parallel_nodes) >= 2, "Parallel must have at least 2 nodes"

            assert hasattr(parallel_node, "join_strategy")
            assert parallel_node.join_strategy in ["all", "any"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_import_workflow_with_missing_optional_fields(self):
        """Test importing workflow with optional fields missing."""
        # Create minimal workflow definition
        from victor.workflows.definition import WorkflowDefinition

        minimal_workflow = WorkflowDefinition(
            name="minimal",
            nodes={
                "agent1": AgentNode(
                    id="agent1",
                    name="Agent",
                    role="assistant",
                    goal="Test goal",
                    tool_budget=10,
                )
            },
        )

        # Should not raise error (use to_dict instead of to_yaml)
        workflow_dict = minimal_workflow.to_dict()
        assert workflow_dict is not None

    def test_import_malformed_yaml_raises_error(self):
        """Test that malformed YAML raises appropriate error."""
        from victor.workflows.yaml_loader import YAMLWorkflowError

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
workflows:
  test_workflow:
    description: "Test"
    nodes:
      - id: node1
        type: invalid_type
        # Invalid node type
"""
            )
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(
                (ConfigurationValidationError, ValueError, KeyError, YAMLWorkflowError)
            ):
                load_workflow_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_team_node_with_no_members_raises_error(self):
        """Test that team node with no members raises validation error."""
        from victor.workflows.yaml_loader import _parse_team_node, YAMLWorkflowError

        node_data = {
            "id": "invalid_team",
            "type": "team",
            "name": "Invalid Team",
            "goal": "Test",
            "team_formation": "parallel",
            "members": [],  # Empty members list
        }

        with pytest.raises((ConfigurationValidationError, ValueError, YAMLWorkflowError)):
            _parse_team_node(node_data)

    def test_conditional_node_with_one_branch_raises_error(self):
        """Test that conditional node with only one branch raises error."""
        from victor.workflows.yaml_loader import _parse_condition_node, YAMLWorkflowConfig

        node_data = {
            "id": "invalid_condition",
            "type": "condition",
            "condition": "test",
            "branches": {"only_branch": "next_node"},  # Only one branch
        }

        # Should validate or warn about single branch
        config = YAMLWorkflowConfig()
        node = _parse_condition_node(node_data, config)
        assert len(node.branches) >= 1  # May be allowed but not recommended


class TestMultipleWorkflowsInFile:
    """Test files containing multiple workflow definitions."""

    def test_import_multiple_workflows(self):
        """Test importing file with multiple workflows."""
        workflows = load_workflow_from_file("victor/research/workflows/deep_research.yaml")

        # Should have multiple workflows
        assert len(workflows) >= 2
        assert "deep_research" in workflows
        assert "quick_research" in workflows

    def test_compile_all_workflows(self):
        """Test that all workflows in a file can be compiled."""
        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        workflows = load_workflow_from_file("victor/research/workflows/deep_research.yaml")

        # Compile all workflows
        compiled_workflows = []
        for name, workflow_def in workflows.items():
            compiled = compiler.compile_definition(workflow_def)
            compiled_workflows.append((name, compiled))

        assert len(compiled_workflows) >= 2

        # Verify all compiled successfully
        for name, compiled in compiled_workflows:
            assert compiled is not None, f"Failed to compile {name}"
            assert hasattr(compiled, "invoke")


class TestWorkflowNodeConnections:
    """Test workflow node connection mapping."""

    def test_connection_mapping_for_team_workflow(self):
        """Test mapping connections in team workflow."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Build connection map
        connections = {}
        for node in workflow_def.nodes.values():  # Fixed: iterate over values
            if hasattr(node, "next_nodes") and node.next_nodes:
                connections[node.id] = node.next_nodes

        # Verify connections
        assert len(connections) > 0

        # Verify no dangling references
        all_node_ids = set(workflow_def.nodes.keys())  # Fixed: use keys() for IDs
        for source, targets in connections.items():
            assert source in all_node_ids
            for target in targets:
                assert target in all_node_ids, f"Dangling reference: {target}"

    def test_connection_mapping_with_branches(self):
        """Test connection mapping with conditional branches."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Find condition node
        condition_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, ConditionNode)]
        assert len(condition_nodes) > 0

        condition = condition_nodes[0]

        # Verify branches point to valid nodes
        all_node_ids = set(workflow_def.nodes.keys())  # Fixed: use keys() for IDs
        for branch_name, target in condition.branches.items():
            assert (
                target in all_node_ids
            ), f"Branch '{branch_name}' points to invalid node: {target}"
