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

"""Comprehensive end-to-end tests for team node execution with recursion tracking.

Tests the full integration of team nodes in workflows, including:
- Recursion depth tracking across workflow and team boundaries
- All team formation types (sequential, parallel, pipeline, hierarchical, consensus)
- Custom recursion depth configuration via metadata and runtime parameters
- Error handling and recovery
- Member configuration (roles, expertise, personality)
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.core.errors import RecursionDepthError
from victor.teams import TeamFormation, create_coordinator
from victor.teams.protocols import ITeamMember
from victor.teams.types import AgentMessage, MemberResult, TeamResult
from victor.workflows.definition import TeamNodeWorkflow
from victor.workflows.recursion import RecursionContext, RecursionGuard
from victor.framework.teams import TeamMemberSpec
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler


# =============================================================================
# Mock Team Member
# =============================================================================


class MockTeamMember:
    """Mock team member for testing team nodes."""

    def __init__(
        self,
        member_id: str,
        role: str = "test_role",
        response: str = "Task complete",
        should_fail: bool = False,
    ):
        """Initialize mock member.

        Args:
            member_id: Unique member identifier
            role: Member role (e.g., "researcher", "executor")
            response: Response to return from execute_task
            should_fail: Whether to simulate execution failure
        """
        self.id = member_id
        self.role = role
        self._response = response
        self._should_fail = should_fail
        self._messages: list[AgentMessage] = []
        self._execute_count = 0

    async def execute_task(self, task: str, context: dict) -> str:
        """Execute task and return mock response."""
        self._execute_count += 1
        if self._should_fail:
            raise RuntimeError(f"Member {self.id} failed intentionally")
        return self._response

    async def receive_message(self, message: AgentMessage) -> AgentMessage | None:
        """Receive message and return mock response."""
        self._messages.append(message)
        return AgentMessage(
            sender_id=self.id,
            recipient_id=message.sender_id,
            content=f"Received: {message.content}",
            message_type="response",
        )

    @property
    def execute_count(self) -> int:
        """Get number of times execute_task was called."""
        return self._execute_count


# =============================================================================
# Test: Simple Team Node Execution (Within Depth Limit)
# =============================================================================


class TestSimpleTeamNodeExecution:
    """Test simple team node execution within recursion depth limit."""

    @pytest.mark.asyncio
    async def test_team_node_at_depth_zero(self):
        """Test team node execution at top level (depth 0 -> 1)."""
        # Create recursion context with max_depth=3
        recursion_ctx = RecursionContext(max_depth=3)

        # Verify starting state
        assert recursion_ctx.current_depth == 0
        assert recursion_ctx.can_nest(1) is True

        # Simulate entering team node (depth 0 -> 1)
        with RecursionGuard(recursion_ctx, "team", "test_team"):
            assert recursion_ctx.current_depth == 1
            assert recursion_ctx.can_nest(1) is True

        # After context manager, should return to depth 0
        assert recursion_ctx.current_depth == 0

    @pytest.mark.asyncio
    async def test_team_node_within_workflow(self):
        """Test team node execution within a workflow (depth 1 -> 2)."""
        recursion_ctx = RecursionContext(max_depth=3)

        # Simulate workflow entry (depth 0 -> 1)
        recursion_ctx.enter("workflow", "outer_workflow")
        assert recursion_ctx.current_depth == 1

        # Team execution should succeed (depth 1 -> 2)
        with RecursionGuard(recursion_ctx, "team", "inner_team"):
            assert recursion_ctx.current_depth == 2
            assert recursion_ctx.can_nest(1) is True  # Can still nest one more

        # Back to workflow level
        assert recursion_ctx.current_depth == 1

        # Clean up
        recursion_ctx.exit()
        assert recursion_ctx.current_depth == 0

    @pytest.mark.asyncio
    async def test_team_node_with_coordinator(self):
        """Test team node execution via UnifiedTeamCoordinator."""
        # Create lightweight coordinator
        coordinator = create_coordinator(lightweight=True)

        # Add mock members
        member1 = MockTeamMember("member1", "researcher", "Research complete")
        member2 = MockTeamMember("member2", "executor", "Execution complete")
        coordinator.add_member(member1).add_member(member2)

        # Set formation
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        # Execute task (should track recursion internally)
        result = await coordinator.execute_task("Test task", {"team_name": "TestTeam"})

        # Verify execution succeeded
        assert result["success"] is True
        assert result["formation"] == "sequential"
        assert len(result["member_results"]) == 2
        assert member1.execute_count == 1
        assert member2.execute_count == 1

        # Verify recursion tracking
        assert coordinator.get_recursion_depth() == 0
        assert coordinator.can_spawn_nested() is True


# =============================================================================
# Test: Nested Team Nodes (Workflow -> Team -> Team)
# =============================================================================


class TestNestedTeamNodes:
    """Test nested team node execution with proper recursion tracking."""

    @pytest.mark.asyncio
    async def test_workflow_team_team_nesting(self):
        """Test workflow -> team -> team nesting (depth 0 -> 1 -> 2 -> 3)."""
        recursion_ctx = RecursionContext(max_depth=3)

        # Simulate workflow (depth 0 -> 1)
        recursion_ctx.enter("workflow", "outer_workflow")
        assert recursion_ctx.current_depth == 1

        # Simulate first team (depth 1 -> 2)
        with RecursionGuard(recursion_ctx, "team", "first_team"):
            assert recursion_ctx.current_depth == 2

            # Simulate nested team (depth 2 -> 3)
            with RecursionGuard(recursion_ctx, "team", "second_team"):
                assert recursion_ctx.current_depth == 3
                # At max depth, cannot nest further
                assert recursion_ctx.can_nest(1) is False

            # Back to first team
            assert recursion_ctx.current_depth == 2

        # Back to workflow
        assert recursion_ctx.current_depth == 1

        # Clean up
        recursion_ctx.exit()
        assert recursion_ctx.current_depth == 0

    @pytest.mark.asyncio
    async def test_nested_team_coordinators(self):
        """Test nested team coordinators share recursion context."""
        # Create shared recursion context
        shared_ctx = RecursionContext(max_depth=3)

        # Create outer team coordinator with shared context
        outer_coordinator = create_coordinator(lightweight=True, recursion_context=shared_ctx)
        outer_member = MockTeamMember("outer_member", "manager", "Outer complete")
        outer_coordinator.add_member(outer_member)

        # Create inner team coordinator with same context
        inner_coordinator = create_coordinator(lightweight=True, recursion_context=shared_ctx)
        inner_member = MockTeamMember("inner_member", "worker", "Inner complete")
        inner_coordinator.add_member(inner_member)

        # Simulate workflow execution first
        shared_ctx.enter("workflow", "main_workflow")
        assert shared_ctx.current_depth == 1

        # Execute outer team (depth 1 -> 2 during execution)
        outer_result = await outer_coordinator.execute_task(
            "Coordinate work", {"team_name": "OuterTeam"}
        )
        assert outer_result["success"] is True
        # After execution, RecursionGuard exits, so depth returns to 1
        assert outer_coordinator.get_recursion_depth() == 1

        # Execute inner team (depth 1 -> 2 during execution)
        inner_result = await inner_coordinator.execute_task("Do work", {"team_name": "InnerTeam"})
        assert inner_result["success"] is True
        # After execution, depth returns to 1
        assert inner_coordinator.get_recursion_depth() == 1

        # Clean up workflow
        shared_ctx.exit()
        assert shared_ctx.current_depth == 0

    @pytest.mark.asyncio
    async def test_team_spawning_workflow_spawning_team(self):
        """Test team -> workflow -> team nesting pattern."""
        recursion_ctx = RecursionContext(max_depth=4)

        # First team (depth 0 -> 1)
        recursion_ctx.enter("team", "outer_team")
        assert recursion_ctx.current_depth == 1

        # Workflow spawned by team (depth 1 -> 2)
        recursion_ctx.enter("workflow", "nested_workflow")
        assert recursion_ctx.current_depth == 2

        # Team spawned by workflow (depth 2 -> 3)
        with RecursionGuard(recursion_ctx, "team", "inner_team"):
            assert recursion_ctx.current_depth == 3
            assert recursion_ctx.can_nest(1) is True  # Can go to depth 4

        # Back to workflow level
        assert recursion_ctx.current_depth == 2

        # Clean up workflow
        recursion_ctx.exit()

        # Back to outer team
        assert recursion_ctx.current_depth == 1

        # Clean up outer team
        recursion_ctx.exit()
        assert recursion_ctx.current_depth == 0


# =============================================================================
# Test: Recursion Limit Enforcement
# =============================================================================


class TestRecursionLimitEnforcement:
    """Test recursion limit enforcement for team nodes."""

    @pytest.mark.asyncio
    async def test_team_exceeds_max_depth(self):
        """Test that team node fails when exceeding max_depth."""
        recursion_ctx = RecursionContext(max_depth=2)

        # Fill to max depth (depth 0 -> 1 -> 2)
        recursion_ctx.enter("workflow", "outer")
        recursion_ctx.enter("team", "middle")

        assert recursion_ctx.current_depth == 2
        assert recursion_ctx.can_nest(1) is False

        # Trying to enter another team should raise error
        with pytest.raises(RecursionDepthError) as exc_info:
            recursion_ctx.enter("team", "inner_team")

        # Verify error details
        error = exc_info.value
        assert error.current_depth == 2
        assert error.max_depth == 2
        assert len(error.execution_stack) == 2
        assert "workflow:outer" in error.execution_stack
        assert "team:middle" in error.execution_stack

    @pytest.mark.asyncio
    async def test_team_node_prevents_infinite_recursion(self):
        """Test that team nodes prevent infinite recursion chains."""
        recursion_ctx = RecursionContext(max_depth=3)

        # Simulate a recursive pattern that would otherwise be infinite
        # workflow -> team -> workflow (would need to continue)

        # Workflow level (depth 0 -> 1)
        recursion_ctx.enter("workflow", "workflow_0")
        assert recursion_ctx.current_depth == 1

        # Team level (depth 1 -> 2)
        recursion_ctx.enter("team", "team_0")
        assert recursion_ctx.current_depth == 2

        # Workflow level (depth 2 -> 3)
        recursion_ctx.enter("workflow", "workflow_1")
        assert recursion_ctx.current_depth == 3

        # At max depth, cannot continue
        assert recursion_ctx.can_nest(1) is False

        # Next team should fail
        with pytest.raises(RecursionDepthError):
            recursion_ctx.enter("team", "team_1")

    @pytest.mark.asyncio
    async def test_coordinator_respects_recursion_limit(self):
        """Test that coordinator respects recursion context limits."""
        # Create context with low limit
        recursion_ctx = RecursionContext(max_depth=1)

        # Fill to max
        recursion_ctx.enter("workflow", "outer")
        assert recursion_ctx.current_depth == 1

        # Create coordinator
        coordinator = create_coordinator(lightweight=True, recursion_context=recursion_ctx)
        member = MockTeamMember("member1", "worker", "Done")
        coordinator.add_member(member)

        # Even though coordinator has no members checking,
        # the recursion context will prevent nesting
        assert coordinator.can_spawn_nested() is False

        # Clean up
        recursion_ctx.exit()


# =============================================================================
# Test: All Team Formation Types
# =============================================================================


class TestTeamFormationTypes:
    """Test all team formation types with recursion tracking."""

    @pytest.mark.asyncio
    async def test_sequential_formation(self):
        """Test sequential formation with recursion tracking."""
        recursion_ctx = RecursionContext(max_depth=3)
        coordinator = create_coordinator(lightweight=True, recursion_context=recursion_ctx)

        # Add members
        member1 = MockTeamMember("m1", "step1", "Step 1 complete")
        member2 = MockTeamMember("m2", "step2", "Step 2 complete")
        member3 = MockTeamMember("m3", "step3", "Step 3 complete")
        coordinator.add_member(member1).add_member(member2).add_member(member3)

        # Set formation
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        # Execute
        result = await coordinator.execute_task("Execute sequentially", {})

        # Verify sequential execution (each member executed once)
        assert result["success"] is True
        assert result["formation"] == "sequential"
        assert member1.execute_count == 1
        assert member2.execute_count == 1
        assert member3.execute_count == 1

    @pytest.mark.asyncio
    async def test_parallel_formation(self):
        """Test parallel formation with recursion tracking."""
        recursion_ctx = RecursionContext(max_depth=3)
        coordinator = create_coordinator(lightweight=True, recursion_context=recursion_ctx)

        # Add members
        member1 = MockTeamMember("m1", "worker1", "Work 1 done")
        member2 = MockTeamMember("m2", "worker2", "Work 2 done")
        coordinator.add_member(member1).add_member(member2)

        # Set formation
        coordinator.set_formation(TeamFormation.PARALLEL)

        # Execute
        result = await coordinator.execute_task("Execute in parallel", {})

        # Verify parallel execution
        assert result["success"] is True
        assert result["formation"] == "parallel"
        assert member1.execute_count == 1
        assert member2.execute_count == 1

    @pytest.mark.asyncio
    async def test_pipeline_formation(self):
        """Test pipeline formation with recursion tracking."""
        recursion_ctx = RecursionContext(max_depth=3)
        coordinator = create_coordinator(lightweight=True, recursion_context=recursion_ctx)

        # Add members (pipeline passes output from one to next)
        member1 = MockTeamMember("m1", "stage1", "Stage 1 output")
        member2 = MockTeamMember("m2", "stage2", "Stage 2 output")
        member3 = MockTeamMember("m3", "stage3", "Stage 3 output")
        coordinator.add_member(member1).add_member(member2).add_member(member3)

        # Set formation
        coordinator.set_formation(TeamFormation.PIPELINE)

        # Execute
        result = await coordinator.execute_task("Process through pipeline", {})

        # Verify pipeline execution
        assert result["success"] is True
        assert result["formation"] == "pipeline"
        # In pipeline, final_output should be last stage's output
        assert "Stage 3 output" in result["final_output"]

    @pytest.mark.asyncio
    async def test_hierarchical_formation(self):
        """Test hierarchical formation with recursion tracking."""
        recursion_ctx = RecursionContext(max_depth=3)
        coordinator = create_coordinator(lightweight=True, recursion_context=recursion_ctx)

        # Add manager and workers
        manager = MockTeamMember("manager", "manager", "Manager decision")
        worker1 = MockTeamMember("worker1", "worker", "Work complete")
        worker2 = MockTeamMember("worker2", "worker", "Work complete")

        # Set manager first (adds to members if not present)
        coordinator.set_manager(manager).add_member(worker1).add_member(worker2)

        # Set formation
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        # Execute
        result = await coordinator.execute_task("Delegate work", {})

        # Verify hierarchical execution
        assert result["success"] is True
        assert result["formation"] == "hierarchical"

    @pytest.mark.asyncio
    async def test_consensus_formation(self):
        """Test consensus formation with recursion tracking."""
        recursion_ctx = RecursionContext(max_depth=3)
        coordinator = create_coordinator(lightweight=True, recursion_context=recursion_ctx)

        # Add members
        member1 = MockTeamMember("m1", "voter1", "Vote: Approve")
        member2 = MockTeamMember("m2", "voter2", "Vote: Approve")
        member3 = MockTeamMember("m3", "voter3", "Vote: Approve")
        coordinator.add_member(member1).add_member(member2).add_member(member3)

        # Set formation
        coordinator.set_formation(TeamFormation.CONSENSUS)

        # Execute
        result = await coordinator.execute_task("Reach consensus", {})

        # Verify consensus execution
        assert result["success"] is True
        assert result["formation"] == "consensus"
        # Consensus formation adds metadata
        assert "consensus_achieved" in result or "member_results" in result


# =============================================================================
# Test: Custom Max Recursion Depth in Metadata
# =============================================================================


class TestCustomMaxRecursionDepth:
    """Test custom max_recursion_depth configuration via metadata."""

    @pytest.mark.asyncio
    async def test_team_node_with_custom_depth_in_metadata(self):
        """Test team node with custom max_recursion_depth in shared_context."""
        # Create team node with custom recursion depth
        team_node = TeamNodeWorkflow(
            id="custom_depth_team",
            name="Custom Depth Team",
            goal="Execute with custom recursion limit",
            team_formation="sequential",
            members=[],
            shared_context={"max_recursion_depth": 5},  # Custom limit in context
        )

        # Verify shared_context is stored
        assert team_node.shared_context["max_recursion_depth"] == 5

        # In real execution, this shared_context would be used to create
        # a RecursionContext with the custom limit
        custom_ctx = RecursionContext(max_depth=team_node.shared_context["max_recursion_depth"])

        assert custom_ctx.max_depth == 5
        assert custom_ctx.can_nest(5) is True

    @pytest.mark.asyncio
    async def test_team_node_metadata_propagation(self):
        """Test that team node shared_context propagates to execution context."""
        team_node = TeamNodeWorkflow(
            id="metadata_test",
            name="Metadata Test Team",
            goal="Test metadata propagation",
            team_formation="parallel",
            members=[
                {
                    "role": "test",
                    "goal": "Test",
                    "expertise": ["testing"],
                    "personality": "analytical",
                }
            ],
            shared_context={
                "max_recursion_depth": 4,
                "timeout_seconds": 120,
                "custom_flag": True,
            },
        )

        # Verify all shared_context is accessible
        assert team_node.shared_context["max_recursion_depth"] == 4
        assert team_node.shared_context["timeout_seconds"] == 120
        assert team_node.shared_context["custom_flag"] is True

    @pytest.mark.asyncio
    async def test_coordinator_with_custom_context_from_metadata(self):
        """Test coordinator created with custom context from metadata."""
        # Simulate extracting custom depth from node shared_context
        shared_context = {"max_recursion_depth": 10}

        # Create custom context
        custom_ctx = RecursionContext(max_depth=shared_context["max_recursion_depth"])

        # Create coordinator with custom context
        coordinator = create_coordinator(lightweight=True, recursion_context=custom_ctx)

        # Verify coordinator uses custom context
        assert coordinator.get_recursion_depth() == 0
        assert coordinator._recursion_ctx.max_depth == 10
        assert coordinator.can_spawn_nested() is True

        # Enter multiple levels to verify higher limit
        for i in range(9):
            coordinator._recursion_ctx.enter("team", f"level_{i}")
            assert coordinator.can_spawn_nested() is True

        # At depth 9, can still nest to 10
        assert coordinator.get_recursion_depth() == 9
        assert coordinator.can_spawn_nested() is True

        # At depth 10, cannot nest further
        coordinator._recursion_ctx.enter("team", "level_9")
        assert coordinator.get_recursion_depth() == 10
        assert coordinator.can_spawn_nested() is False


# =============================================================================
# Test: Runtime Parameter Override
# =============================================================================


class TestRuntimeParameterOverride:
    """Test runtime parameter override for max_recursion_depth."""

    @pytest.mark.asyncio
    async def test_runtime_override_max_depth(self):
        """Test overriding max_recursion_depth at runtime."""
        # Create context with default limit
        recursion_ctx = RecursionContext(max_depth=3)

        assert recursion_ctx.max_depth == 3

        # Simulate runtime override (e.g., from workflow execution parameters)
        # In real execution, this would be passed as a parameter
        runtime_params = {"max_recursion_depth": 7}

        # Create new context with override (simulating runtime override)
        override_ctx = RecursionContext(max_depth=runtime_params["max_recursion_depth"])

        assert override_ctx.max_depth == 7
        assert override_ctx.can_nest(7) is True

    @pytest.mark.asyncio
    async def test_coordinator_runtime_depth_override(self):
        """Test coordinator respecting runtime depth override."""
        # Create coordinator with default context
        default_ctx = RecursionContext(max_depth=3)
        coordinator = create_coordinator(lightweight=True, recursion_context=default_ctx)

        assert coordinator._recursion_ctx.max_depth == 3

        # Simulate runtime override by creating new context
        # In real system, this would be via execution parameters
        runtime_ctx = RecursionContext(max_depth=5)
        override_coordinator = create_coordinator(lightweight=True, recursion_context=runtime_ctx)

        assert override_coordinator._recursion_ctx.max_depth == 5
        assert override_coordinator.can_spawn_nested() is True

    @pytest.mark.asyncio
    async def test_workflow_execution_with_runtime_params(self):
        """Test workflow execution passing runtime recursion params."""
        from victor.workflows.definition import WorkflowBuilder

        # Build workflow with team node
        workflow = (
            WorkflowBuilder("test_workflow")
            .add_agent(
                "agent1",
                "researcher",
                "Research task",
            )
            .build()
        )

        # Simulate runtime execution parameters
        runtime_params = {
            "max_recursion_depth": 8,
            "team_recursion_limit": 5,
            "custom_timeout": 180,
        }

        # Extract recursion limit from params
        max_depth = runtime_params.get("max_recursion_depth", 3)

        # Create context with runtime override
        runtime_ctx = RecursionContext(max_depth=max_depth)

        # Verify runtime params are applied
        assert runtime_ctx.max_depth == 8
        assert runtime_ctx.can_nest(8) is True


# =============================================================================
# Test: Team Node Error Handling and Recovery
# =============================================================================


class TestTeamNodeErrorHandling:
    """Test team node error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_member_failure_with_continue_on_error(self):
        """Test team continues when member fails with continue_on_error=True."""
        recursion_ctx = RecursionContext(max_depth=3)
        coordinator = create_coordinator(lightweight=True, recursion_context=recursion_ctx)

        # Add mix of failing and successful members
        member1 = MockTeamMember("m1", "worker1", "Success", should_fail=False)
        member2 = MockTeamMember("m2", "worker2", "Fail", should_fail=True)
        member3 = MockTeamMember("m3", "worker3", "Success", should_fail=False)

        coordinator.add_member(member1).add_member(member2).add_member(member3)
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        # Execute (should handle failures internally)
        result = await coordinator.execute_task("Test with failures", {})

        # Result should indicate partial success
        # Actual behavior depends on formation strategy
        assert "member_results" in result

    @pytest.mark.asyncio
    async def test_team_timeout_handling(self):
        """Test team node timeout handling."""
        recursion_ctx = RecursionContext(max_depth=3)
        coordinator = create_coordinator(lightweight=True, recursion_context=recursion_ctx)

        # Add member that simulates long-running task
        async def slow_member(task: str, context: dict) -> str:
            import asyncio

            await asyncio.sleep(0.1)  # Simulate work
            return "Complete"

        # Create mock member with slow execution
        slow_member_obj = MockTeamMember("slow", "worker", "Complete")
        # Monkey patch execute_task to be slow
        original_execute = slow_member_obj.execute_task
        slow_member_obj.execute_task = slow_member  # type: ignore

        coordinator.add_member(slow_member_obj)

        # Execute with timeout (would need actual timeout implementation)
        # For now, just verify it executes
        result = await coordinator.execute_task("Slow task", {})

        assert "member_results" in result

    @pytest.mark.asyncio
    async def test_recursion_error_recovery(self):
        """Test recovery from RecursionDepthError."""
        recursion_ctx = RecursionContext(max_depth=2)

        # Fill to max
        recursion_ctx.enter("workflow", "outer")
        recursion_ctx.enter("team", "inner")

        # Try to exceed limit
        with pytest.raises(RecursionDepthError):
            recursion_ctx.enter("team", "too_deep")

        # Verify state is still consistent after error
        assert recursion_ctx.current_depth == 2

        # Can exit normally
        recursion_ctx.exit()
        assert recursion_ctx.current_depth == 1

        recursion_ctx.exit()
        assert recursion_ctx.current_depth == 0

        # Can start fresh
        recursion_ctx.enter("workflow", "new_workflow")
        assert recursion_ctx.current_depth == 1


# =============================================================================
# Test: Team Node with Member Configuration
# =============================================================================


class TestTeamMemberConfiguration:
    """Test team node with various member configurations."""

    @pytest.mark.asyncio
    async def test_team_with_role_configuration(self):
        """Test team node with explicit role configuration."""
        # Create team with role-based members (using dict format)
        team_node = TeamNodeWorkflow(
            id="role_based_team",
            name="Role-Based Team",
            goal="Execute with role specialization",
            team_formation="sequential",
            members=[
                {
                    "role": "researcher",
                    "goal": "Gather information",
                    "tool_budget": 10,
                },
                {
                    "role": "analyzer",
                    "goal": "Analyze data",
                    "tool_budget": 5,
                },
                {
                    "role": "executor",
                    "goal": "Execute plan",
                    "tool_budget": 15,
                },
            ],
        )

        # Verify member roles are configured
        assert len(team_node.members) == 3
        assert team_node.members[0]["role"] == "researcher"
        assert team_node.members[1]["role"] == "analyzer"
        assert team_node.members[2]["role"] == "executor"

    @pytest.mark.asyncio
    async def test_team_with_expertise_configuration(self):
        """Test team node with expertise tags."""
        team_node = TeamNodeWorkflow(
            id="expertise_team",
            name="Expertise-Based Team",
            goal="Leverage specialized expertise",
            team_formation="parallel",
            members=[
                {
                    "role": "security_reviewer",
                    "goal": "Security review",
                    "expertise": ["security", "cryptography", "penetration_testing"],
                },
                {
                    "role": "performance_reviewer",
                    "goal": "Performance analysis",
                    "expertise": ["performance", "optimization", "profiling"],
                },
                {
                    "role": "quality_reviewer",
                    "goal": "Quality assurance",
                    "expertise": ["testing", "quality", "ci_cd"],
                },
            ],
        )

        # Verify expertise is configured
        security_member = team_node.members[0]
        assert "security" in security_member["expertise"]
        assert "cryptography" in security_member["expertise"]

        performance_member = team_node.members[1]
        assert "performance" in performance_member["expertise"]
        assert "optimization" in performance_member["expertise"]

    @pytest.mark.asyncio
    async def test_team_with_personality_configuration(self):
        """Test team node with personality configuration."""
        team_node = TeamNodeWorkflow(
            id="personality_team",
            name="Personality-Configured Team",
            goal="Test personality-based behavior",
            team_formation="consensus",
            members=[
                {
                    "role": "supporter",
                    "goal": "Provide positive feedback",
                    "personality": "optimistic",
                    "backstory": "You are always supportive and find opportunities in challenges.",
                },
                {
                    "role": "critic",
                    "goal": "Identify potential issues",
                    "personality": "critical",
                    "backstory": "You are analytical and identify risks and edge cases.",
                },
                {
                    "role": "moderator",
                    "goal": "Balance perspectives",
                    "personality": "balanced",
                    "backstory": "You weigh all perspectives objectively.",
                },
            ],
        )

        # Verify personality configuration
        assert team_node.members[0]["personality"] == "optimistic"
        assert team_node.members[1]["personality"] == "critical"
        assert team_node.members[2]["personality"] == "balanced"

        # Verify personas (backstory field)
        assert "supportive" in team_node.members[0]["backstory"].lower()
        assert "analytical" in team_node.members[1]["backstory"].lower()
        assert "objectively" in team_node.members[2]["backstory"].lower()

    @pytest.mark.asyncio
    async def test_team_with_mixed_configuration(self):
        """Test team node with mixed role, expertise, and personality."""
        team_node = TeamNodeWorkflow(
            id="mixed_config_team",
            name="Mixed Configuration Team",
            goal="Test all configuration options together",
            team_formation="hierarchical",
            members=[
                {
                    "role": "team_lead",
                    "goal": "Coordinate team",
                    "expertise": ["leadership", "planning"],
                    "personality": "decisive",
                    "backstory": "You make clear decisions and delegate effectively.",
                    "tool_budget": 20,
                },
                {
                    "role": "domain_expert",
                    "goal": "Provide domain expertise",
                    "expertise": ["domain_knowledge", "analysis"],
                    "personality": "analytical",
                    "backstory": "You provide deep technical insights.",
                    "tool_budget": 15,
                },
                {
                    "role": "executor",
                    "goal": "Implement solutions",
                    "expertise": ["implementation", "testing"],
                    "personality": "pragmatic",
                    "backstory": "You focus on practical, working solutions.",
                    "tool_budget": 25,
                },
            ],
            shared_context={"max_recursion_depth": 4},
        )

        # Verify all configurations
        lead = team_node.members[0]
        assert lead["role"] == "team_lead"
        assert "leadership" in lead["expertise"]
        assert lead["personality"] == "decisive"
        assert lead["tool_budget"] == 20

        specialist = team_node.members[1]
        assert specialist["role"] == "domain_expert"
        assert "domain_knowledge" in specialist["expertise"]
        assert specialist["personality"] == "analytical"

        implementer = team_node.members[2]
        assert implementer["role"] == "executor"
        assert "implementation" in implementer["expertise"]
        assert implementer["personality"] == "pragmatic"

        # Verify team-level shared_context
        assert team_node.shared_context["max_recursion_depth"] == 4


# =============================================================================
# Test: Integration with Workflow Compiler
# =============================================================================


class TestWorkflowCompilerIntegration:
    """Test team node integration with UnifiedWorkflowCompiler."""

    @pytest.mark.asyncio
    async def test_compile_workflow_with_team_node(self):
        """Test that team nodes can be created and serialized."""
        # Create team node
        team_node = TeamNodeWorkflow(
            id="review_team",
            name="Review Team",
            goal="Conduct comprehensive review",
            team_formation="parallel",
            members=[
                {
                    "role": "security",
                    "goal": "Security review",
                },
                {
                    "role": "quality",
                    "goal": "Quality review",
                },
            ],
        )

        # Verify team node structure
        assert team_node.id == "review_team"
        assert team_node.name == "Review Team"
        assert team_node.goal == "Conduct comprehensive review"
        assert team_node.team_formation == "parallel"
        assert len(team_node.members) == 2

        # Verify serialization
        team_dict = team_node.to_dict()
        assert team_dict["id"] == "review_team"
        assert team_dict["type"] == "team"
        assert team_dict["goal"] == team_node.goal
        assert team_dict["team_formation"] == "parallel"

    @pytest.mark.asyncio
    async def test_team_node_in_yaml_workflow(self):
        """Test loading team node from YAML workflow definition."""
        from victor.workflows import load_workflow_from_file

        # Try to load team node example (if it exists)
        try:
            workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")

            # Verify workflow loaded
            assert isinstance(workflows, dict)

            # Check for team nodes in loaded workflows
            for name, workflow_def in workflows.items():
                if hasattr(workflow_def, "nodes"):
                    for node in workflow_def.nodes:
                        if hasattr(node, "node_type"):
                            if node.node_type.value == "team":
                                # Verify team node structure
                                assert hasattr(node, "members")
                                assert hasattr(node, "team_formation")
        except (FileNotFoundError, ImportError):
            # Skip if file doesn't exist
            pytest.skip("team_node_example.yaml not found")


# =============================================================================
# Test: Recursion Context State Management
# =============================================================================


class TestRecursionContextStateManagement:
    """Test recursion context state management across team executions."""

    @pytest.mark.asyncio
    async def test_context_isolation_between_teams(self):
        """Test that different teams have isolated recursion contexts."""
        # Create separate contexts for different teams
        ctx1 = RecursionContext(max_depth=3)
        ctx2 = RecursionContext(max_depth=5)

        # Enter teams in different contexts
        ctx1.enter("team", "team1")
        ctx2.enter("team", "team2")

        # Verify isolation
        assert ctx1.current_depth == 1
        assert ctx2.current_depth == 1
        assert ctx1.max_depth == 3
        assert ctx2.max_depth == 5

        # Enter nested teams
        ctx1.enter("workflow", "workflow1")
        ctx2.enter("workflow", "workflow2")

        assert ctx1.current_depth == 2
        assert ctx2.current_depth == 2

        # Verify different limits
        assert ctx1.can_nest(1) is True  # Can go to 3
        assert ctx2.can_nest(3) is True  # Can go to 5
        assert ctx1.can_nest(2) is False  # Cannot go to 4

    @pytest.mark.asyncio
    async def test_context_cleanup_after_team_execution(self):
        """Test that recursion context is properly cleaned up after execution."""
        recursion_ctx = RecursionContext(max_depth=3)

        # Simulate team execution
        initial_depth = recursion_ctx.current_depth
        assert initial_depth == 0

        with RecursionGuard(recursion_ctx, "team", "test_team"):
            # Inside team execution
            assert recursion_ctx.current_depth == 1

        # After context manager exits, depth should be restored
        assert recursion_ctx.current_depth == initial_depth

        # Verify stack is clean
        assert len(recursion_ctx.execution_stack) == 0

    @pytest.mark.asyncio
    async def test_multiple_sequential_team_executions(self):
        """Test multiple team executions in sequence with same context."""
        recursion_ctx = RecursionContext(max_depth=3)

        # Execute first team
        with RecursionGuard(recursion_ctx, "team", "team1"):
            assert recursion_ctx.current_depth == 1

        assert recursion_ctx.current_depth == 0

        # Execute second team
        with RecursionGuard(recursion_ctx, "team", "team2"):
            assert recursion_ctx.current_depth == 1

        assert recursion_ctx.current_depth == 0

        # Execute third team
        with RecursionGuard(recursion_ctx, "team", "team3"):
            assert recursion_ctx.current_depth == 1

        # All should succeed with independent depth tracking
        assert recursion_ctx.current_depth == 0
        assert len(recursion_ctx.execution_stack) == 0


# =============================================================================
# Test: Edge Cases and Corner Cases
# =============================================================================


class TestTeamNodeEdgeCases:
    """Test edge cases and corner cases for team node execution."""

    @pytest.mark.asyncio
    async def test_empty_team_execution(self):
        """Test behavior when team has no members."""
        recursion_ctx = RecursionContext(max_depth=3)
        coordinator = create_coordinator(lightweight=True, recursion_context=recursion_ctx)

        # Don't add any members

        # Execute with empty team
        result = await coordinator.execute_task("Test empty team", {})

        # Should handle gracefully
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_single_member_team(self):
        """Test team with only one member."""
        recursion_ctx = RecursionContext(max_depth=3)
        coordinator = create_coordinator(lightweight=True, recursion_context=recursion_ctx)

        # Add single member
        member = MockTeamMember("sole_member", "solo_worker", "Solo complete")
        coordinator.add_member(member)

        # Execute
        result = await coordinator.execute_task("Single member task", {})

        # Should work fine
        assert result["success"] is True
        assert member.execute_count == 1

    @pytest.mark.asyncio
    async def test_team_formation_switching(self):
        """Test switching formations between executions."""
        recursion_ctx = RecursionContext(max_depth=3)
        coordinator = create_coordinator(lightweight=True, recursion_context=recursion_ctx)

        member1 = MockTeamMember("m1", "worker", "Done")
        member2 = MockTeamMember("m2", "worker", "Done")
        coordinator.add_member(member1).add_member(member2)

        # Execute with sequential
        coordinator.set_formation(TeamFormation.SEQUENTIAL)
        result1 = await coordinator.execute_task("Task 1", {})
        assert result1["formation"] == "sequential"

        # Execute with parallel
        coordinator.set_formation(TeamFormation.PARALLEL)
        result2 = await coordinator.execute_task("Task 2", {})
        assert result2["formation"] == "parallel"

        # Execute with pipeline
        coordinator.set_formation(TeamFormation.PIPELINE)
        result3 = await coordinator.execute_task("Task 3", {})
        assert result3["formation"] == "pipeline"

    @pytest.mark.asyncio
    async def test_max_recursion_depth_zero(self):
        """Test behavior when max_recursion_depth is set to 0."""
        # Edge case: max_depth = 0 means no nesting allowed
        recursion_ctx = RecursionContext(max_depth=0)

        # Cannot nest at all
        assert recursion_ctx.can_nest(1) is False

        # Trying to enter should raise error
        with pytest.raises(RecursionDepthError):
            recursion_ctx.enter("team", "test_team")

    @pytest.mark.asyncio
    async def test_very_large_max_recursion_depth(self):
        """Test behavior with very large max_recursion_depth."""
        # Large limit for deep nesting scenarios
        recursion_ctx = RecursionContext(max_depth=1000)

        assert recursion_ctx.max_depth == 1000
        assert recursion_ctx.can_nest(500) is True
        assert recursion_ctx.can_nest(1000) is True
        assert recursion_ctx.can_nest(1001) is False

    @pytest.mark.asyncio
    async def test_recursion_depth_info_reporting(self):
        """Test recursion depth information reporting."""
        recursion_ctx = RecursionContext(max_depth=5)

        # Get initial info
        info = recursion_ctx.get_depth_info()
        assert info["current_depth"] == 0
        assert info["max_depth"] == 5
        assert info["remaining_depth"] == 5
        assert len(info["execution_stack"]) == 0

        # Enter some levels
        recursion_ctx.enter("workflow", "w1")
        recursion_ctx.enter("team", "t1")

        # Get updated info
        info = recursion_ctx.get_depth_info()
        assert info["current_depth"] == 2
        assert info["max_depth"] == 5
        assert info["remaining_depth"] == 3
        assert len(info["execution_stack"]) == 2
        assert "workflow:w1" in info["execution_stack"]
        assert "team:t1" in info["execution_stack"]
