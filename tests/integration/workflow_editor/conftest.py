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

"""Pytest fixtures for workflow editor integration tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from victor.workflows import load_workflow_from_file
from victor.workflows.definition import (
    WorkflowDefinition,
    AgentNode,
    TeamNodeWorkflow,
)
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler


@pytest.fixture
def sample_team_workflow() -> WorkflowDefinition:
    """Provide a sample team workflow for testing."""
    from victor.workflows.definition import WorkflowDefinition

    return WorkflowDefinition(
        name="sample_team_workflow",
        description="Sample workflow with team node",
        nodes=[
            AgentNode(
                id="analyze",
                name="Analyze Task",
                role="planner",
                goal="Analyze the task",
                tool_budget=10,
                next_nodes=["team_node"],
            ),
            TeamNodeWorkflow(
                id="team_node",
                name="Sample Team",
                goal="Execute as a team",
                team_formation="sequential",
                members=[],
                timeout_seconds=300,
                total_tool_budget=50,
                next_nodes=["finalize"],
            ),
            AgentNode(
                id="finalize",
                name="Finalize",
                role="writer",
                goal="Prepare final report",
                tool_budget=10,
            ),
        ],
    )


@pytest.fixture
def production_workflows():
    """Load all production workflows for testing."""
    workflow_files = {
        "team_node_example": "victor/coding/workflows/team_node_example.yaml",
        "deep_research": "victor/research/workflows/deep_research.yaml",
        "team_research": "victor/research/workflows/examples/team_research.yaml",
        "code_generation": "victor/benchmark/workflows/code_generation.yaml",
    }

    workflows = {}
    for name, path in workflow_files.items():
        try:
            workflows[name] = load_workflow_from_file(path)
        except Exception as e:
            # Some workflows might not exist in all environments
            pytest.skip(f"Could not load {name}: {e}")

    return workflows


@pytest.fixture
def compiler():
    """Provide a workflow compiler instance."""
    return UnifiedWorkflowCompiler(enable_caching=True)


@pytest.fixture
def temp_workflow_file():
    """Create a temporary workflow file for testing."""
    yaml_content = """
workflows:
  test_workflow:
    description: "Test workflow"
    metadata:
      version: "1.0"
      vertical: "test"

    nodes:
      - id: node1
        type: agent
        name: "First Node"
        role: assistant
        goal: "Test goal"
        tool_budget: 10
        next: [node2]

      - id: node2
        type: transform
        name: "Second Node"
        transform: "result = true"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_graph_data():
    """Provide sample graph data for editor testing."""
    return {
        "nodes": [
            {
                "id": "node1",
                "type": "agent",
                "name": "Agent Node",
                "config": {
                    "role": "assistant",
                    "goal": "Test goal",
                    "tool_budget": 10,
                },
                "position": {"x": 100, "y": 100},
            },
            {
                "id": "node2",
                "type": "team",
                "name": "Team Node",
                "config": {
                    "formation": "parallel",
                    "goal": "Team goal",
                    "members": [
                        {
                            "id": "member1",
                            "role": "researcher",
                            "goal": "Research",
                            "tool_budget": 10,
                        }
                    ],
                },
                "position": {"x": 300, "y": 100},
            },
        ],
        "edges": [
            {
                "id": "edge1",
                "source": "node1",
                "target": "node2",
                "label": None,
            }
        ],
        "metadata": {
            "name": "test_graph",
            "description": "Test graph for editor",
        },
    }


@pytest.fixture
def all_formations_yaml():
    """Provide YAML content with all 8 team formations."""
    return """
workflows:
  parallel_formation:
    description: "Parallel formation test"
    nodes:
      - id: parallel_team
        type: team
        name: "Parallel Team"
        goal: "Work in parallel"
        team_formation: parallel
        members:
          - id: m1
            role: assistant
            goal: "Goal 1"
            tool_budget: 10
          - id: m2
            role: assistant
            goal: "Goal 2"
            tool_budget: 10

  sequential_formation:
    description: "Sequential formation test"
    nodes:
      - id: sequential_team
        type: team
        name: "Sequential Team"
        goal: "Work in sequence"
        team_formation: sequential
        members:
          - id: m1
            role: planner
            goal: "Plan"
            tool_budget: 10
            priority: 0
          - id: m2
            role: executor
            goal: "Execute"
            tool_budget: 10
            priority: 1

  pipeline_formation:
    description: "Pipeline formation test"
    nodes:
      - id: pipeline_team
        type: team
        name: "Pipeline Team"
        goal: "Process through pipeline"
        team_formation: pipeline
        members:
          - id: stage1
            role: researcher
            goal: "Research"
            tool_budget: 10
            priority: 0
          - id: stage2
            role: analyst
            goal: "Analyze"
            tool_budget: 10
            priority: 1
          - id: stage3
            role: writer
            goal: "Write"
            tool_budget: 10
            priority: 2

  hierarchical_formation:
    description: "Hierarchical formation test"
    nodes:
      - id: hierarchical_team
        type: team
        name: "Hierarchical Team"
        goal: "Manager coordinates workers"
        team_formation: hierarchical
        members:
          - id: manager
            role: planner
            goal: "Coordinate"
            tool_budget: 15
            is_manager: true
          - id: worker1
            role: executor
            goal: "Execute"
            tool_budget: 10
          - id: worker2
            role: executor
            goal: "Execute"
            tool_budget: 10

  consensus_formation:
    description: "Consensus formation test"
    nodes:
      - id: consensus_team
        type: team
        name: "Consensus Team"
        goal: "Vote on decision"
        team_formation: consensus
        members:
          - id: voter1
            role: reviewer
            goal: "Review"
            tool_budget: 10
          - id: voter2
            role: reviewer
            goal: "Review"
            tool_budget: 10
          - id: voter3
            role: reviewer
            goal: "Review"
            tool_budget: 10

  round_robin_formation:
    description: "Round-robin formation test"
    nodes:
      - id: round_robin_team
        type: team
        name: "Round Robin Team"
        goal: "Rotate through members"
        team_formation: round_robin
        members:
          - id: m1
            role: assistant
            goal: "Turn 1"
            tool_budget: 10
          - id: m2
            role: assistant
            goal: "Turn 2"
            tool_budget: 10

  dynamic_formation:
    description: "Dynamic formation test"
    nodes:
      - id: dynamic_team
        type: team
        name: "Dynamic Team"
        goal: "Adapt to context"
        team_formation: dynamic
        members:
          - id: m1
            role: analyst
            goal: "Adapt"
            tool_budget: 15
          - id: m2
            role: executor
            goal: "Adapt"
            tool_budget: 15

  custom_formation:
    description: "Custom formation test"
    nodes:
      - id: custom_team
        type: team
        name: "Custom Team"
        goal: "Custom logic"
        team_formation: custom
        members:
          - id: m1
            role: coordinator
            goal: "Coordinate"
            tool_budget: 20
          - id: m2
            role: executor
            goal: "Execute"
            tool_budget: 15
"""


@pytest.fixture
def conditional_branches_yaml():
    """Provide YAML with conditional branches for testing."""
    return """
workflows:
  conditional_test:
    description: "Test conditional branches"
    nodes:
      - id: decision
        type: condition
        name: "Decision Point"
        condition: "complexity_check"
        branches:
          simple: simple_path
          medium: medium_path
          complex: complex_path

      - id: simple_path
        type: agent
        name: "Simple Path"
        role: executor
        goal: "Handle simple case"
        tool_budget: 10
        next: [finalize]

      - id: medium_path
        type: agent
        name: "Medium Path"
        role: executor
        goal: "Handle medium case"
        tool_budget: 20
        next: [finalize]

      - id: complex_path
        type: team
        name: "Complex Path"
        goal: "Handle complex case"
        team_formation: pipeline
        members:
          - id: planner
            role: planner
            goal: "Plan"
            tool_budget: 15
          - id: executor
            role: executor
            goal: "Execute"
            tool_budget: 25
        next: [finalize]

      - id: finalize
        type: transform
        transform: "complete = true"
"""


@pytest.fixture
def parallel_execution_yaml():
    """Provide YAML with parallel execution for testing."""
    return """
workflows:
  parallel_test:
    description: "Test parallel execution"
    nodes:
      - id: start
        type: agent
        name: "Start"
        role: assistant
        goal: "Start parallel work"
        tool_budget: 5
        next: [parallel_node]

      - id: parallel_node
        type: parallel
        name: "Parallel Execution"
        parallel_nodes: [branch1, branch2, branch3]
        join_strategy: all
        next: [aggregate]

      - id: branch1
        type: agent
        name: "Branch 1"
        role: researcher
        goal: "Research"
        tool_budget: 15

      - id: branch2
        type: agent
        name: "Branch 2"
        role: analyst
        goal: "Analyze"
        tool_budget: 15

      - id: branch3
        type: agent
        name: "Branch 3"
        role: writer
        goal: "Write"
        tool_budget: 15

      - id: aggregate
        type: transform
        name: "Aggregate Results"
        transform: |
          results = merge(branch1, branch2, branch3)
          complete = true
"""


@pytest.fixture
def recursion_depth_test_yaml():
    """Provide YAML to test recursion depth limits."""
    return """
workflows:
  recursion_test:
    description: "Test recursion depth"
    metadata:
      version: "1.0"

    execution:
      max_recursion_depth: 3
      max_timeout_seconds: 900

    nodes:
      - id: workflow_level_1
        type: team
        name: "Level 1 Team"
        goal: "First level"
        team_formation: sequential
        members:
          - id: member1
            role: assistant
            goal: "Level 1 work"
            tool_budget: 10
        next: [level_2]

      - id: level_2
        type: team
        name: "Level 2 Team"
        goal: "Second level"
        team_formation: sequential
        members:
          - id: member2
            role: assistant
            goal: "Level 2 work"
            tool_budget: 10
        next: [level_3]

      - id: level_3
        type: team
        name: "Level 3 Team"
        goal: "Third level (at limit)"
        team_formation: sequential
        members:
          - id: member3
            role: assistant
            goal: "Level 3 work"
            tool_budget: 10
        next: [complete]

      - id: complete
        type: transform
        transform: "recursion_test_complete = true"
"""
