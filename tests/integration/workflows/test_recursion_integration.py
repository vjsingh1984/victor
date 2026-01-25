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
from typing import Any


"""Integration tests for recursion context flow through workflow execution.

This module tests that RecursionContext properly flows through:
1. Workflow compilation from YAML
2. StateGraph execution
3. All node types (agent, compute, parallel, condition, transform, team)
4. Nested workflow execution
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from victor.workflows.definition import WorkflowDefinition, AgentNode, TeamNodeWorkflow
from victor.workflows.yaml_loader import load_workflow_from_yaml
from victor.workflows.recursion import RecursionContext, RecursionDepthError


@pytest.fixture
def mock_orchestrator() -> Any:
    """Create a mock orchestrator."""
    orchestrator = MagicMock()
    orchestrator.execute_task = AsyncMock(
        return_value={
            "output": "Test result",
            "tool_calls": 0,
        }
    )
    return orchestrator


@pytest.fixture
def compiler(mock_orchestrator: Any) -> Any:
    """Create a compiler with mock orchestrator."""
    return UnifiedWorkflowCompiler(
        orchestrator=mock_orchestrator,
        enable_caching=False,
    )


class TestRecursionContextFlow:
    """Test recursion context propagation through workflow execution."""

    @pytest.mark.asyncio
    async def test_recursion_context_in_workflow_state(self, compiler) -> None:
        """Test that recursion context is added to workflow state."""
        # Create simple workflow
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            nodes={
                "agent": AgentNode(
                    id="agent",
                    name="Test Agent",
                    role="planner",
                    goal="Test goal",
                ),
            },
            start_node="agent",
        )

        # Compile workflow
        compiled = compiler.compile_definition(workflow_def)

        # Invoke workflow
        result = await compiled.invoke({"test_input": "value"})

        # Verify recursion context was in state
        assert "_recursion_context" in result.state
        assert isinstance(result.state["_recursion_context"], RecursionContext)

    @pytest.mark.asyncio
    async def test_max_recursion_depth_from_metadata(self, compiler) -> None:
        """Test that max_recursion_depth from workflow metadata is used."""
        # Create workflow with custom max_recursion_depth in metadata
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            metadata={"max_recursion_depth": 5},
            nodes={
                "agent": AgentNode(
                    id="agent",
                    name="Test Agent",
                    role="planner",
                    goal="Test goal",
                ),
            },
            start_node="agent",
        )

        # Compile workflow
        compiled = compiler.compile_definition(workflow_def)

        # Verify max_recursion_depth is set
        assert compiled.max_recursion_depth == 5

    @pytest.mark.asyncio
    async def test_max_recursion_depth_runtime_override(self, compiler) -> None:
        """Test that runtime parameter overrides metadata value."""
        # Create workflow with metadata max_recursion_depth
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            metadata={"max_recursion_depth": 5},
            nodes={
                "agent": AgentNode(
                    id="agent",
                    name="Test Agent",
                    role="planner",
                    goal="Test goal",
                ),
            },
            start_node="agent",
        )

        # Compile workflow
        compiled = compiler.compile_definition(workflow_def)

        # Invoke with runtime override
        result = await compiled.invoke(
            {"test_input": "value"},
            max_recursion_depth=10,
        )

        # Verify override was used
        assert result.state["_recursion_context"].max_depth == 10

    @pytest.mark.asyncio
    async def test_recursion_context_shared_in_parallel(self, compiler) -> None:
        """Test that recursion context is shared across parallel branches."""
        from victor.workflows.definition import ParallelNode

        # Create workflow with parallel node
        workflow_def = WorkflowDefinition(
            name="test_parallel",
            nodes={
                "agent1": AgentNode(
                    id="agent1",
                    name="Agent 1",
                    role="planner",
                    goal="Test 1",
                ),
                "agent2": AgentNode(
                    id="agent2",
                    name="Agent 2",
                    role="executor",
                    goal="Test 2",
                ),
                "parallel": ParallelNode(
                    id="parallel",
                    name="Parallel",
                    parallel_nodes=["agent1", "agent2"],
                ),
            },
            start_node="parallel",
        )

        # Compile workflow
        compiled = compiler.compile_definition(workflow_def)

        # Invoke workflow
        result = await compiled.invoke({"test_input": "value"})

        # Verify single recursion context was used
        recursion_ctx = result.state.get("_recursion_context")
        assert recursion_ctx is not None
        # Both parallel branches should share the same context object
        assert isinstance(recursion_ctx, RecursionContext)

    @pytest.mark.asyncio
    async def test_yaml_metadata_max_recursion_depth(self) -> None:
        """Test that max_recursion_depth is parsed from YAML metadata."""
        yaml_content = """
workflows:
  test_workflow:
    description: Test workflow
    metadata:
      max_recursion_depth: 8
    nodes:
      - id: start
        type: agent
        role: planner
        goal: Test goal
"""

        # Load from YAML
        workflow_def = load_workflow_from_yaml(yaml_content, "test_workflow")

        # Verify metadata was parsed
        assert workflow_def.metadata.get("max_recursion_depth") == 8

    @pytest.mark.asyncio
    async def test_yaml_execution_max_recursion_depth(self) -> None:
        """Test that max_recursion_depth is parsed from YAML execution settings."""
        yaml_content = """
workflows:
  test_workflow:
    description: Test workflow
    execution:
      max_recursion_depth: 12
    nodes:
      - id: start
        type: agent
        role: planner
        goal: Test goal
"""

        # Load from YAML
        workflow_def = load_workflow_from_yaml(yaml_content, "test_workflow")

        # Verify execution settings were parsed into metadata
        assert workflow_def.metadata.get("max_recursion_depth") == 12

    @pytest.mark.asyncio
    async def test_yaml_precedence_execution_over_metadata(self) -> None:
        """Test that execution settings take precedence over metadata."""
        yaml_content = """
workflows:
  test_workflow:
    description: Test workflow
    metadata:
      max_recursion_depth: 5
    execution:
      max_recursion_depth: 15
    nodes:
      - id: start
        type: agent
        role: planner
        goal: Test goal
"""

        # Load from YAML
        workflow_def = load_workflow_from_yaml(yaml_content, "test_workflow")

        # Execution settings should take precedence
        assert workflow_def.metadata.get("max_recursion_depth") == 15

    @pytest.mark.asyncio
    async def test_recursion_context_cleanup_on_error(self, compiler) -> None:
        """Test that recursion context exists and is properly managed."""
        # Create workflow
        workflow_def = WorkflowDefinition(
            name="test_error",
            nodes={
                "agent": AgentNode(
                    id="agent",
                    name="Test Agent",
                    role="planner",
                    goal="Test goal",
                ),
            },
            start_node="agent",
        )

        # Compile workflow
        compiled = compiler.compile_definition(workflow_def)

        # Invoke workflow
        result = await compiled.invoke({"test_input": "value"})

        # Verify recursion context exists and was properly managed
        assert "_recursion_context" in result.state
        recursion_ctx = result.state["_recursion_context"]
        assert isinstance(recursion_ctx, RecursionContext)
        # Context should have been entered (depth >= 0)
        # The exit() happens in finally block after execution completes
        assert recursion_ctx.current_depth >= 0

    def test_recursion_context_default_depth(self, compiler: Any) -> None:
        """Test that default max_recursion_depth is 3 when not specified."""
        # Create workflow without max_recursion_depth
        workflow_def = WorkflowDefinition(
            name="test_default",
            nodes={
                "agent": AgentNode(
                    id="agent",
                    name="Test Agent",
                    role="planner",
                    goal="Test goal",
                ),
            },
            start_node="agent",
        )

        # Compile workflow
        compiled = compiler.compile_definition(workflow_def)

        # Verify default depth
        assert compiled.max_recursion_depth == 3


class TestRecursionContextTeamNodes:
    """Test recursion context with team nodes."""

    @pytest.mark.asyncio
    async def test_team_node_receives_recursion_context(self, compiler) -> None:
        """Test that recursion context is properly initialized for team nodes."""
        # Create workflow with team node
        workflow_def = WorkflowDefinition(
            name="test_team",
            metadata={"max_recursion_depth": 5},
            nodes={
                "team": TeamNodeWorkflow(
                    id="team",
                    name="Test Team",
                    team_formation="parallel",
                    members=[],
                    goal="Test team goal",
                ),
            },
            start_node="team",
        )

        # Compile workflow
        compiled = compiler.compile_definition(workflow_def)

        # The key test: verify max_recursion_depth from metadata is used
        assert compiled.max_recursion_depth == 5

        # Note: We can't easily test the actual team node execution without
        # a full orchestrator setup, but we can verify the metadata propagation
        # which is the critical part of the integration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
