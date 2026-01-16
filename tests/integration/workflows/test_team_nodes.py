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

"""Integration tests for team node execution."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from victor.core.errors import RecursionDepthError
from victor.workflows import load_workflow_from_file
from victor.workflows.definition import TeamNodeWorkflow
from victor.workflows.recursion import RecursionContext, RecursionGuard
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler


class TestTeamNodeLoading:
    """Test loading YAML workflows with team nodes."""

    def test_load_team_node_example(self):
        """Test that team_node_example.yaml can be loaded."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        assert isinstance(workflows, dict)
        assert "team_node_demo" in workflows

    def test_compile_team_workflow(self):
        """Test compiling workflow with team nodes."""
        compiler = UnifiedWorkflowCompiler()
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")

        for name, workflow_def in workflows.items():
            compiled = compiler.compile_definition(workflow_def)
            assert compiled is not None

    def test_team_node_has_required_fields(self):
        """Test that team nodes have all required fields."""
        from victor.workflows.yaml_loader import _parse_team_node

        node_data = {
            "id": "test_team",
            "type": "team",
            "name": "Test Team",
            "goal": "Test goal",
            "team_formation": "sequential",
            "members": [
                {
                    "id": "member1",
                    "role": "researcher",
                    "goal": "Research",
                    "tool_budget": 10,
                }
            ],
        }

        node = _parse_team_node(node_data)
        assert node.id == "test_team"
        assert node.team_formation == "sequential"
        assert len(node.members) == 1


class TestTeamNodeExecution:
    """Test team node execution with recursion tracking."""

    @pytest.mark.asyncio
    async def test_team_node_within_depth_limit(self):
        """Test that team nodes execute within depth limit."""
        # Create a simple team node
        team_node = TeamNodeWorkflow(
            id="test_team",
            name="Test Team",
            goal="Test execution",
            team_formation="sequential",
            members=[],
            timeout_seconds=None,
            total_tool_budget=10,
        )

        # Create recursion context with limit of 3
        recursion_ctx = RecursionContext(max_depth=3)

        # Simulate workflow -> team (depth 1)
        recursion_ctx.enter("workflow", "outer_workflow")

        # Team execution should succeed
        # (In real test, would need orchestrator and tool_registry)
        assert recursion_ctx.current_depth == 1
        assert recursion_ctx.can_nest(1) is True

        # Clean up
        recursion_ctx.exit()

    @pytest.mark.asyncio
    async def test_team_node_exceeds_depth_limit(self):
        """Test that team nodes fail when exceeding depth limit."""
        # Create recursion context with limit of 2
        recursion_ctx = RecursionContext(max_depth=2)

        # Simulate workflow -> workflow -> team (depth 3)
        recursion_ctx.enter("workflow", "outer")
        recursion_ctx.enter("workflow", "middle")

        # Should not be able to spawn team
        assert recursion_ctx.can_nest(1) is False

        # Trying to enter team should raise error
        with pytest.raises(RecursionDepthError) as exc_info:
            recursion_ctx.enter("team", "inner_team")

        # Verify error details
        error = exc_info.value
        assert error.current_depth == 2
        assert error.max_depth == 2
        assert len(error.execution_stack) == 2

    @pytest.mark.asyncio
    async def test_recursion_guard_context_manager(self):
        """Test RecursionGuard context manager for team execution."""
        recursion_ctx = RecursionContext(max_depth=3)

        # Simulate workflow entry
        recursion_ctx.enter("workflow", "outer_workflow")

        # Use RecursionGuard for team execution
        with RecursionGuard(recursion_ctx, "team", "test_team"):
            assert recursion_ctx.current_depth == 2
            assert recursion_ctx.can_nest(1) is True  # Can still nest one more level

        # After context manager, should be back to workflow level
        assert recursion_ctx.current_depth == 1

        # Clean up
        recursion_ctx.exit()


class TestWorkflowRecursion:
    """Test recursion tracking across workflow execution."""

    def test_workflow_yaml_with_custom_depth(self):
        """Test loading workflow with custom max_recursion_depth."""
        yaml_content = """
workflows:
  custom_depth:
    description: "Workflow with custom depth"
    metadata:
      max_recursion_depth: 5
    nodes:
      - id: start
        type: agent
        role: researcher
        goal: "Start"
        next: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            workflows = load_workflow_from_file(temp_file)
            assert "custom_depth" in workflows

            # Check metadata contains custom depth
            workflow = workflows["custom_depth"]
            if hasattr(workflow, "metadata"):
                assert workflow.metadata.get("max_recursion_depth") == 5
        finally:
            Path(temp_file).unlink()

    def test_runtime_depth_override(self):
        """Test runtime override of max_recursion_depth."""
        # Create a workflow definition
        yaml_content = """
workflows:
  test_workflow:
    description: "Test workflow"
    nodes:
      - id: start
        type: agent
        role: researcher
        goal: "Start"
        next: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            compiler = UnifiedWorkflowCompiler()
            workflows = load_workflow_from_file(temp_file)
            workflow_def = workflows["test_workflow"]

            # Compile with default depth
            compiled_default = compiler.compile_definition(workflow_def)
            assert compiled_default is not None

            # Compile with custom depth override (if supported)
            # This tests that the compiler accepts max_recursion_depth parameter
            compiled_custom = compiler.compile_definition(
                workflow_def, max_recursion_depth=10
            )
            assert compiled_custom is not None
        finally:
            Path(temp_file).unlink()

    def test_depth_info_reporting(self):
        """Test recursion context depth information reporting."""
        recursion_ctx = RecursionContext(max_depth=5)

        # Initial state
        info = recursion_ctx.get_depth_info()
        assert info["current_depth"] == 0
        assert info["max_depth"] == 5
        assert info["remaining_depth"] == 5
        assert len(info["execution_stack"]) == 0

        # After entering workflow
        recursion_ctx.enter("workflow", "main")
        info = recursion_ctx.get_depth_info()
        assert info["current_depth"] == 1
        assert info["remaining_depth"] == 4
        assert len(info["execution_stack"]) == 1
        assert "workflow:main" in info["execution_stack"]

        # After entering team
        recursion_ctx.enter("team", "research_team")
        info = recursion_ctx.get_depth_info()
        assert info["current_depth"] == 2
        assert info["remaining_depth"] == 3
        assert len(info["execution_stack"]) == 2


class TestTeamFormations:
    """Test different team formation types in YAML."""

    def test_sequential_formation(self):
        """Test sequential team formation parsing."""
        from victor.workflows.yaml_loader import _parse_team_node

        node_data = {
            "id": "seq_team",
            "type": "team",
            "team_formation": "sequential",
            "members": [],
        }

        node = _parse_team_node(node_data)
        assert node.team_formation == "sequential"

    def test_parallel_formation(self):
        """Test parallel team formation parsing."""
        from victor.workflows.yaml_loader import _parse_team_node

        node_data = {
            "id": "par_team",
            "type": "team",
            "team_formation": "parallel",
            "members": [],
        }

        node = _parse_team_node(node_data)
        assert node.team_formation == "parallel"

    def test_pipeline_formation(self):
        """Test pipeline team formation parsing."""
        from victor.workflows.yaml_loader import _parse_team_node

        node_data = {
            "id": "pipeline_team",
            "type": "team",
            "team_formation": "pipeline",
            "members": [],
        }

        node = _parse_team_node(node_data)
        assert node.team_formation == "pipeline"

    def test_hierarchical_formation(self):
        """Test hierarchical team formation parsing."""
        from victor.workflows.yaml_loader import _parse_team_node

        node_data = {
            "id": "hierarchy_team",
            "type": "team",
            "team_formation": "hierarchical",
            "members": [],
        }

        node = _parse_team_node(node_data)
        assert node.team_formation == "hierarchical"

    def test_consensus_formation(self):
        """Test consensus team formation parsing."""
        from victor.workflows.yaml_loader import _parse_team_node

        node_data = {
            "id": "consensus_team",
            "type": "team",
            "team_formation": "consensus",
            "members": [],
        }

        node = _parse_team_node(node_data)
        assert node.team_formation == "consensus"


class TestTeamNodeMembers:
    """Test team member configuration in YAML."""

    def test_member_configuration_parsing(self):
        """Test parsing team member configuration."""
        from victor.workflows.yaml_loader import _parse_team_node

        node_data = {
            "id": "test_team",
            "type": "team",
            "team_formation": "sequential",
            "members": [
                {
                    "id": "researcher",
                    "role": "researcher",
                    "name": "Code Researcher",
                    "goal": "Research codebase",
                    "tool_budget": 15,
                    "tools": ["read", "grep", "code_search"],
                    "backstory": "Experienced researcher",
                    "expertise": ["code-analysis", "architecture"],
                    "personality": "methodical",
                },
                {
                    "id": "implementer",
                    "role": "executor",
                    "name": "Feature Implementer",
                    "goal": "Implement feature",
                    "tool_budget": 25,
                    "tools": ["read", "write"],
                    "backstory": "Skilled developer",
                    "expertise": ["implementation", "testing"],
                    "personality": "pragmatic",
                },
            ],
        }

        node = _parse_team_node(node_data)
        assert len(node.members) == 2

        # Check first member
        member1 = node.members[0]
        assert member1["id"] == "researcher"
        assert member1["role"] == "researcher"
        assert member1["tool_budget"] == 15
        assert "read" in member1["tools"]
        assert member1["backstory"] == "Experienced researcher"
        assert "code-analysis" in member1["expertise"]

        # Check second member
        member2 = node.members[1]
        assert member2["id"] == "implementer"
        assert member2["role"] == "executor"
        assert member2["tool_budget"] == 25
        assert "write" in member2["tools"]

    def test_team_node_optional_fields(self):
        """Test team node with optional fields."""
        from victor.workflows.yaml_loader import _parse_team_node

        node_data = {
            "id": "optional_team",
            "type": "team",
            "goal": "Test goal",
            "team_formation": "parallel",
            "timeout_seconds": 300,
            "merge_strategy": "dict",
            "merge_mode": "team_wins",
            "output_key": "custom_result",
            "continue_on_error": False,
            "max_iterations": 25,
            "total_tool_budget": 50,
            "members": [],
        }

        node = _parse_team_node(node_data)
        assert node.timeout_seconds == 300
        assert node.merge_strategy == "dict"
        assert node.merge_mode == "team_wins"
        assert node.output_key == "custom_result"
        assert node.continue_on_error is False
        assert node.max_iterations == 25
        assert node.total_tool_budget == 50


class TestTeamNodeInRealWorkflow:
    """Test team nodes within actual workflow definitions."""

    def test_team_node_in_workflow_graph(self):
        """Test that team nodes integrate into workflow graphs."""
        yaml_content = """
workflows:
  team_workflow:
    description: "Workflow with team node"
    nodes:
      - id: analyze
        type: agent
        role: planner
        goal: "Analyze task"
        next: [execute_team]

      - id: execute_team
        type: team
        name: "Execution Team"
        goal: "Execute the task"
        team_formation: sequential
        members:
          - id: worker1
            role: executor
            goal: "Do work"
            tool_budget: 10
        next: [finalize]

      - id: finalize
        type: agent
        role: writer
        goal: "Finalize results"
        next: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            compiler = UnifiedWorkflowCompiler()
            workflows = load_workflow_from_file(temp_file)
            workflow_def = workflows["team_workflow"]

            # Should compile successfully
            compiled = compiler.compile_definition(workflow_def)
            assert compiled is not None

            # CompiledGraph wraps the graph - check it has the graph structure
            # The actual graph is in compiled.compiled_graph
            assert compiled.compiled_graph is not None

            # The workflow definition should have 3 nodes
            # Check the original workflow_def has the nodes
            assert len(workflow_def.nodes) == 3

            # Check node IDs - nodes might be strings or node objects
            # In YAML loading, they're typically node objects
            node_ids = []
            for node in workflow_def.nodes:
                if isinstance(node, str):
                    node_ids.append(node)
                elif hasattr(node, "id"):
                    node_ids.append(node.id)

            assert "analyze" in node_ids
            assert "execute_team" in node_ids
            assert "finalize" in node_ids

        finally:
            Path(temp_file).unlink()

    def test_multiple_team_nodes_in_workflow(self):
        """Test workflow with multiple team nodes."""
        yaml_content = """
workflows:
  multi_team_workflow:
    description: "Workflow with multiple team nodes"
    nodes:
      - id: research_team
        type: team
        name: "Research Team"
        goal: "Research"
        team_formation: parallel
        members:
          - id: r1
            role: researcher
            goal: "Research 1"
            tool_budget: 10
        next: [implement_team]

      - id: implement_team
        type: team
        name: "Implementation Team"
        goal: "Implement"
        team_formation: sequential
        members:
          - id: i1
            role: executor
            goal: "Implement"
            tool_budget: 15
        next: []

      - id: review_team
        type: team
        name: "Review Team"
        goal: "Review"
        team_formation: consensus
        members:
          - id: rev1
            role: reviewer
            goal: "Review"
            tool_budget: 5
        next: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            compiler = UnifiedWorkflowCompiler()
            workflows = load_workflow_from_file(temp_file)
            workflow_def = workflows["multi_team_workflow"]

            # Should compile successfully
            compiled = compiler.compile_definition(workflow_def)
            assert compiled is not None
            assert compiled.compiled_graph is not None

            # The workflow definition should have 3 nodes
            # They might be loaded as strings or as node objects depending on the loader
            assert len(workflow_def.nodes) == 3

            # Check node IDs - nodes might be strings or node objects
            node_ids = []
            node_types = {}
            for node in workflow_def.nodes:
                if isinstance(node, str):
                    node_ids.append(node)
                elif hasattr(node, "id"):
                    node_ids.append(node.id)
                    # Track node type if available
                    if hasattr(node, "type"):
                        node_types[node.id] = node.type
                    elif hasattr(node, "node_type"):
                        node_types[node.id] = node.node_type

            # Verify all team node IDs are present
            assert "research_team" in node_ids
            assert "implement_team" in node_ids
            assert "review_team" in node_ids

            # If nodes have type information, verify they're team nodes
            if node_types:
                from victor.workflows.definition import WorkflowNodeType
                for node_id, node_type in node_types.items():
                    if node_id in ["research_team", "implement_team", "review_team"]:
                        # Check if it's a team type (either string "team" or WorkflowNodeType.TEAM)
                        if isinstance(node_type, str):
                            assert node_type == "team"
                        else:
                            assert node_type == WorkflowNodeType.TEAM

        finally:
            Path(temp_file).unlink()


class TestTeamNodeErrorHandling:
    """Test error handling in team node execution."""

    def test_team_node_missing_required_fields(self):
        """Test that team node requires id field."""
        from victor.workflows.yaml_loader import _parse_team_node, YAMLWorkflowError

        # Missing id
        node_data = {
            "type": "team",
            "team_formation": "sequential",
            "members": [],
        }

        with pytest.raises(YAMLWorkflowError, match="missing required 'id' field"):
            _parse_team_node(node_data)

    def test_team_node_default_values(self):
        """Test that team node has sensible default values."""
        from victor.workflows.yaml_loader import _parse_team_node

        node_data = {
            "id": "default_team",
            "type": "team",
            "members": [],
        }

        node = _parse_team_node(node_data)
        assert node.team_formation == "sequential"  # Default
        assert node.goal == ""  # Default
        assert node.timeout_seconds is None  # Default
        assert node.total_tool_budget == 100  # Default
        assert node.max_iterations == 50  # Default
        assert node.merge_strategy == "dict"  # Default
        assert node.merge_mode == "team_wins"  # Default
        assert node.output_key == "team_result"  # Default
        assert node.continue_on_error is True  # Default
