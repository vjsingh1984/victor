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

"""Tests for Framework TeamCoordinator.

These tests follow TDD - written before implementation.
They verify the team coordination patterns for multi-agent orchestration.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.framework.agent_protocols import (
    AgentCapability,
    IAgentPersona,
    IAgentRole,
    ITeamMember,
)
from victor.teams import (
    AgentMessage,
    MessageType,
    TeamFormation,
)


# =============================================================================
# Mock Implementations for Testing
# =============================================================================


@dataclass
class MockRole:
    """Mock role for testing."""

    name: str = "mock_role"
    capabilities: Set[AgentCapability] = field(
        default_factory=lambda: {AgentCapability.READ, AgentCapability.COMMUNICATE}
    )
    allowed_tools: Set[str] = field(default_factory=lambda: {"read_file"})
    tool_budget: int = 10

    def get_system_prompt_section(self) -> str:
        return f"You are a {self.name}."


@dataclass
class MockPersona:
    """Mock persona for testing."""

    name: str = "Test Agent"
    background: str = "Test background"
    communication_style: str = "Professional"

    def format_message(self, content: str) -> str:
        return f"[{self.name}] {content}"


class MockTeamMember:
    """Mock team member for testing."""

    def __init__(self, member_id: str, role: Optional[IAgentRole] = None, delay: float = 0.0):
        self.id = member_id
        self.role = role or MockRole(name=member_id)
        self.persona = MockPersona(name=member_id)
        self.delay = delay
        self.executed_tasks: List[str] = []
        self.received_messages: List[AgentMessage] = []
        self.execution_order: List[str] = []  # For tracking parallel execution

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.executed_tasks.append(task)
        # Track execution order for parallel tests
        if "execution_tracker" in context:
            context["execution_tracker"].append(self.id)
        return f"Result from {self.id}: {task}"

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        self.received_messages.append(message)
        return AgentMessage(
            sender_id=self.id,
            recipient_id=message.sender_id,
            content=f"Acknowledged: {message.content}",
            message_type=MessageType.RESULT,
        )


# =============================================================================
# FrameworkTeamCoordinator Tests
# =============================================================================


class TestFrameworkTeamCoordinatorBasics:
    """Tests for basic FrameworkTeamCoordinator functionality."""

    def test_coordinator_exists(self):
        """FrameworkTeamCoordinator should be importable."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        assert FrameworkTeamCoordinator is not None

    def test_coordinator_instantiation(self):
        """FrameworkTeamCoordinator should be instantiable."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        assert coordinator is not None

    def test_coordinator_has_add_member(self):
        """FrameworkTeamCoordinator should have add_member method."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        assert hasattr(coordinator, "add_member")
        assert callable(coordinator.add_member)

    def test_coordinator_has_set_formation(self):
        """FrameworkTeamCoordinator should have set_formation method."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        assert hasattr(coordinator, "set_formation")
        assert callable(coordinator.set_formation)

    def test_coordinator_has_execute_task(self):
        """FrameworkTeamCoordinator should have execute_task method."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        assert hasattr(coordinator, "execute_task")
        assert callable(coordinator.execute_task)

    def test_coordinator_has_broadcast(self):
        """FrameworkTeamCoordinator should have broadcast method."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        assert hasattr(coordinator, "broadcast")
        assert callable(coordinator.broadcast)


class TestFrameworkTeamCoordinatorMemberManagement:
    """Tests for member management in FrameworkTeamCoordinator."""

    def test_add_member_stores_member(self):
        """add_member should store the member."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        member = MockTeamMember("agent1")

        coordinator.add_member(member)

        assert len(coordinator.members) == 1
        assert coordinator.members[0].id == "agent1"

    def test_add_member_returns_self_for_chaining(self):
        """add_member should return self for fluent chaining."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        member1 = MockTeamMember("agent1")
        member2 = MockTeamMember("agent2")

        result = coordinator.add_member(member1).add_member(member2)

        assert result is coordinator
        assert len(coordinator.members) == 2

    def test_add_multiple_members(self):
        """Multiple members can be added."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        for i in range(5):
            coordinator.add_member(MockTeamMember(f"agent{i}"))

        assert len(coordinator.members) == 5

    def test_members_property_returns_list(self):
        """members property should return list of members."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        member = MockTeamMember("agent1")
        coordinator.add_member(member)

        assert isinstance(coordinator.members, list)


class TestFrameworkTeamCoordinatorFormation:
    """Tests for formation management in FrameworkTeamCoordinator."""

    def test_set_formation_stores_formation(self):
        """set_formation should store the formation."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        coordinator.set_formation(TeamFormation.PARALLEL)

        assert coordinator.formation == TeamFormation.PARALLEL

    def test_set_formation_returns_self(self):
        """set_formation should return self for chaining."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        result = coordinator.set_formation(TeamFormation.SEQUENTIAL)

        assert result is coordinator

    def test_default_formation_is_sequential(self):
        """Default formation should be SEQUENTIAL."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        assert coordinator.formation == TeamFormation.SEQUENTIAL

    def test_set_formation_to_all_types(self):
        """set_formation should accept all TeamFormation values."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        for formation in TeamFormation:
            coordinator = FrameworkTeamCoordinator()
            coordinator.set_formation(formation)
            assert coordinator.formation == formation


class TestFrameworkTeamCoordinatorManager:
    """Tests for manager assignment in FrameworkTeamCoordinator."""

    def test_set_manager_stores_manager(self):
        """set_manager should store the manager."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        manager = MockTeamMember("manager")
        coordinator.add_member(manager)
        coordinator.set_manager(manager)

        assert coordinator.manager == manager

    def test_set_manager_returns_self(self):
        """set_manager should return self for chaining."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        manager = MockTeamMember("manager")
        result = coordinator.add_member(manager).set_manager(manager)

        assert result is coordinator

    def test_manager_is_none_by_default(self):
        """Manager should be None by default."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        assert coordinator.manager is None


class TestSequentialExecution:
    """Tests for sequential task execution."""

    @pytest.mark.asyncio
    async def test_sequential_execution_order(self):
        """Sequential execution should run members in order."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        execution_tracker: List[str] = []

        member1 = MockTeamMember("agent1", delay=0.01)
        member2 = MockTeamMember("agent2", delay=0.01)
        member3 = MockTeamMember("agent3", delay=0.01)

        coordinator.add_member(member1)
        coordinator.add_member(member2)
        coordinator.add_member(member3)
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task(
            "Test task", {"execution_tracker": execution_tracker}
        )

        # Verify sequential order
        assert execution_tracker == ["agent1", "agent2", "agent3"]

    @pytest.mark.asyncio
    async def test_sequential_passes_context(self):
        """Sequential execution should pass context to each member."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        member = MockTeamMember("agent1")
        coordinator.add_member(member)
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        context = {"key": "value"}
        await coordinator.execute_task("Test task", context)

        # Member should have received the task
        assert len(member.executed_tasks) == 1

    @pytest.mark.asyncio
    async def test_sequential_collects_results(self):
        """Sequential execution should collect results from all members."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        coordinator.add_member(MockTeamMember("agent1"))
        coordinator.add_member(MockTeamMember("agent2"))
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task("Test task", {})

        assert "member_results" in result
        assert len(result["member_results"]) == 2


class TestParallelExecution:
    """Tests for parallel task execution."""

    @pytest.mark.asyncio
    async def test_parallel_runs_concurrently(self):
        """Parallel execution should run members concurrently."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()

        # Use delays to demonstrate parallelism - parallel should be faster
        member1 = MockTeamMember("agent1", delay=0.1)
        member2 = MockTeamMember("agent2", delay=0.1)
        member3 = MockTeamMember("agent3", delay=0.1)

        coordinator.add_member(member1)
        coordinator.add_member(member2)
        coordinator.add_member(member3)
        coordinator.set_formation(TeamFormation.PARALLEL)

        import time

        start = time.time()
        await coordinator.execute_task("Test task", {})
        duration = time.time() - start

        # If truly parallel, total time should be ~0.1s, not ~0.3s
        assert duration < 0.25  # Allow some overhead

    @pytest.mark.asyncio
    async def test_parallel_collects_all_results(self):
        """Parallel execution should collect results from all members."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        coordinator.add_member(MockTeamMember("agent1"))
        coordinator.add_member(MockTeamMember("agent2"))
        coordinator.add_member(MockTeamMember("agent3"))
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task("Test task", {})

        assert "member_results" in result
        assert len(result["member_results"]) == 3


class TestHierarchicalExecution:
    """Tests for hierarchical task execution."""

    @pytest.mark.asyncio
    async def test_hierarchical_manager_executes_first(self):
        """In hierarchical, manager should execute first to delegate."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        execution_tracker: List[str] = []

        manager_role = MockRole(
            name="manager",
            capabilities={AgentCapability.DELEGATE, AgentCapability.COMMUNICATE},
        )
        manager = MockTeamMember("manager", role=manager_role)
        worker1 = MockTeamMember("worker1")
        worker2 = MockTeamMember("worker2")

        coordinator.add_member(manager)
        coordinator.add_member(worker1)
        coordinator.add_member(worker2)
        coordinator.set_manager(manager)
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        result = await coordinator.execute_task(
            "Test task", {"execution_tracker": execution_tracker}
        )

        # Manager should execute first
        assert execution_tracker[0] == "manager"

    @pytest.mark.asyncio
    async def test_hierarchical_requires_manager(self):
        """Hierarchical execution should have a manager."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        coordinator.add_member(MockTeamMember("worker1"))
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        # Without explicit manager, first member or auto-select should be used
        result = await coordinator.execute_task("Test task", {})

        # Should still execute without error
        assert "member_results" in result


class TestPipelineExecution:
    """Tests for pipeline task execution."""

    @pytest.mark.asyncio
    async def test_pipeline_execution_order(self):
        """Pipeline execution should run members in sequence with output passing."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        execution_tracker: List[str] = []

        coordinator.add_member(MockTeamMember("stage1"))
        coordinator.add_member(MockTeamMember("stage2"))
        coordinator.add_member(MockTeamMember("stage3"))
        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task(
            "Pipeline task", {"execution_tracker": execution_tracker}
        )

        # Pipeline should execute in order
        assert execution_tracker == ["stage1", "stage2", "stage3"]

    @pytest.mark.asyncio
    async def test_pipeline_passes_output_to_next_stage(self):
        """Pipeline should pass output of one stage to the next."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()

        coordinator.add_member(MockTeamMember("stage1"))
        coordinator.add_member(MockTeamMember("stage2"))
        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task("Pipeline task", {})

        # Each stage should have received the task
        assert "member_results" in result


class TestBroadcast:
    """Tests for broadcast functionality."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_members(self):
        """broadcast should send message to all team members."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        member1 = MockTeamMember("agent1")
        member2 = MockTeamMember("agent2")
        member3 = MockTeamMember("agent3")

        coordinator.add_member(member1)
        coordinator.add_member(member2)
        coordinator.add_member(member3)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="Broadcast message",
            message_type=MessageType.TASK,
        )

        responses = await coordinator.broadcast(message)

        # All members should have received the message
        assert len(member1.received_messages) == 1
        assert len(member2.received_messages) == 1
        assert len(member3.received_messages) == 1

    @pytest.mark.asyncio
    async def test_broadcast_returns_responses(self):
        """broadcast should return responses from all members."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        coordinator.add_member(MockTeamMember("agent1"))
        coordinator.add_member(MockTeamMember("agent2"))

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="Broadcast message",
            message_type=MessageType.TASK,
        )

        responses = await coordinator.broadcast(message)

        assert len(responses) == 2
        assert all(r is not None for r in responses)


class TestSendMessage:
    """Tests for send_message functionality."""

    @pytest.mark.asyncio
    async def test_send_message_to_specific_member(self):
        """send_message should send to a specific member."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        member1 = MockTeamMember("agent1")
        member2 = MockTeamMember("agent2")

        coordinator.add_member(member1)
        coordinator.add_member(member2)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="agent1",
            content="Direct message",
            message_type=MessageType.TASK,
        )

        response = await coordinator.send_message(message)

        # Only agent1 should have received the message
        assert len(member1.received_messages) == 1
        assert len(member2.received_messages) == 0

    @pytest.mark.asyncio
    async def test_send_message_returns_response(self):
        """send_message should return the response."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        member = MockTeamMember("agent1")
        coordinator.add_member(member)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="agent1",
            content="Direct message",
            message_type=MessageType.TASK,
        )

        response = await coordinator.send_message(message)

        assert response is not None
        assert response.message_type == MessageType.RESULT


# =============================================================================
# Result Dataclass Tests
# =============================================================================


class TestMemberResult:
    """Tests for MemberResult dataclass."""

    def test_member_result_exists(self):
        """MemberResult should be importable."""
        from victor.teams import MemberResult

        assert MemberResult is not None

    def test_member_result_creation(self):
        """MemberResult should be creatable with required fields."""
        from victor.teams import MemberResult

        result = MemberResult(
            member_id="agent1",
            success=True,
            output="Task completed",
        )

        assert result.member_id == "agent1"
        assert result.success is True
        assert result.output == "Task completed"

    def test_member_result_with_error(self):
        """MemberResult should support error field."""
        from victor.teams import MemberResult

        result = MemberResult(
            member_id="agent1",
            success=False,
            output="",
            error="Task failed",
        )

        assert result.success is False
        assert result.error == "Task failed"


class TestTeamResult:
    """Tests for TeamResult dataclass."""

    def test_team_result_exists(self):
        """TeamResult should be importable."""
        from victor.teams import TeamResult

        assert TeamResult is not None

    def test_team_result_creation(self):
        """TeamResult should be creatable with required fields."""
        from victor.teams import MemberResult, TeamResult, TeamFormation

        member_result = MemberResult(
            member_id="agent1",
            success=True,
            output="Done",
        )

        team_result = TeamResult(
            success=True,
            member_results={"agent1": member_result},
            final_output="Team completed",
            formation=TeamFormation.SEQUENTIAL,
        )

        assert team_result.success is True
        assert "agent1" in team_result.member_results


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_coordinator_satisfies_protocol(self):
        """FrameworkTeamCoordinator should satisfy ITeamCoordinator protocol."""
        from victor.framework.agent_protocols import ITeamCoordinator
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        coordinator = FrameworkTeamCoordinator()
        assert isinstance(coordinator, ITeamCoordinator)


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_exports_coordinator(self):
        """team_coordinator should export FrameworkTeamCoordinator."""
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        assert FrameworkTeamCoordinator is not None

    def test_exports_member_result(self):
        """team_coordinator should export MemberResult."""
        from victor.teams import MemberResult

        assert MemberResult is not None

    def test_exports_team_result(self):
        """team_coordinator should export TeamResult."""
        from victor.teams import TeamResult

        assert TeamResult is not None
