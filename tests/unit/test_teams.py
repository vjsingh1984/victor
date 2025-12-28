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

"""Unit tests for the agent teams module."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.teams import (
    TeamFormation,
    TeamMember,
    TeamConfig,
    TeamResult,
    TeamCoordinator,
    MessageType,
    AgentMessage,
    TeamMessageBus,
    TeamSharedMemory,
)
from victor.agent.teams.team import MemberResult, MemberStatus
from victor.agent.subagents import SubAgentRole


class TestTeamFormation:
    """Test TeamFormation enum."""

    def test_all_formations_defined(self):
        """All expected formations are defined."""
        assert TeamFormation.SEQUENTIAL.value == "sequential"
        assert TeamFormation.PARALLEL.value == "parallel"
        assert TeamFormation.HIERARCHICAL.value == "hierarchical"
        assert TeamFormation.PIPELINE.value == "pipeline"

    def test_formations_are_iterable(self):
        """All formations can be iterated."""
        formations = list(TeamFormation)
        assert len(formations) == 4


class TestMemberStatus:
    """Test MemberStatus enum."""

    def test_all_statuses_defined(self):
        """All expected statuses are defined."""
        assert MemberStatus.IDLE.value == "idle"
        assert MemberStatus.WORKING.value == "working"
        assert MemberStatus.COMPLETED.value == "completed"
        assert MemberStatus.FAILED.value == "failed"


class TestTeamMember:
    """Test TeamMember dataclass."""

    def test_minimal_member(self):
        """Create member with minimal fields."""
        member = TeamMember(
            id="test",
            role=SubAgentRole.RESEARCHER,
            name="Test Researcher",
            goal="Find something",
        )
        assert member.id == "test"
        assert member.role == SubAgentRole.RESEARCHER
        assert member.name == "Test Researcher"
        assert member.goal == "Find something"

    def test_default_values(self):
        """Default values are set correctly."""
        member = TeamMember(
            id="test",
            role=SubAgentRole.EXECUTOR,
            name="Executor",
            goal="Do something",
        )
        assert member.tool_budget == 15
        assert member.allowed_tools is None
        assert member.can_delegate is False
        assert member.reports_to is None
        assert member.is_manager is False
        assert member.priority == 0

    def test_all_fields_can_be_set(self):
        """All fields can be explicitly set."""
        member = TeamMember(
            id="mgr",
            role=SubAgentRole.PLANNER,
            name="Manager",
            goal="Manage team",
            tool_budget=20,
            allowed_tools=["read", "ls"],
            can_delegate=True,
            is_manager=True,
            priority=1,
        )
        assert member.tool_budget == 20
        assert member.allowed_tools == ["read", "ls"]
        assert member.can_delegate is True
        assert member.is_manager is True
        assert member.priority == 1


class TestTeamConfig:
    """Test TeamConfig dataclass."""

    def test_minimal_config(self):
        """Create config with minimal fields."""
        member = TeamMember(
            id="worker",
            role=SubAgentRole.EXECUTOR,
            name="Worker",
            goal="Work",
        )
        config = TeamConfig(
            name="Test Team",
            goal="Test goal",
            members=[member],
        )
        assert config.name == "Test Team"
        assert config.goal == "Test goal"
        assert len(config.members) == 1
        assert config.formation == TeamFormation.SEQUENTIAL

    def test_default_values(self):
        """Default values are set correctly."""
        member = TeamMember(
            id="worker",
            role=SubAgentRole.EXECUTOR,
            name="Worker",
            goal="Work",
        )
        config = TeamConfig(
            name="Test",
            goal="Goal",
            members=[member],
        )
        assert config.max_iterations == 50
        assert config.total_tool_budget == 100
        assert config.shared_context == {}
        assert config.allow_dynamic_membership is False
        assert config.timeout_seconds == 600

    def test_empty_members_raises(self):
        """Empty members list raises ValueError."""
        with pytest.raises(ValueError, match="at least one member"):
            TeamConfig(name="Test", goal="Goal", members=[])

    def test_duplicate_member_ids_raises(self):
        """Duplicate member IDs raise ValueError."""
        members = [
            TeamMember(id="dup", role=SubAgentRole.RESEARCHER, name="R1", goal="G1"),
            TeamMember(id="dup", role=SubAgentRole.EXECUTOR, name="E1", goal="G2"),
        ]
        with pytest.raises(ValueError, match="unique"):
            TeamConfig(name="Test", goal="Goal", members=members)

    def test_hierarchical_requires_manager(self):
        """Hierarchical formation requires exactly one manager."""
        members = [
            TeamMember(id="w1", role=SubAgentRole.EXECUTOR, name="W1", goal="G1"),
            TeamMember(id="w2", role=SubAgentRole.EXECUTOR, name="W2", goal="G2"),
        ]
        with pytest.raises(ValueError, match="exactly one manager"):
            TeamConfig(
                name="Test",
                goal="Goal",
                members=members,
                formation=TeamFormation.HIERARCHICAL,
            )

    def test_get_member(self):
        """get_member returns correct member."""
        members = [
            TeamMember(id="r1", role=SubAgentRole.RESEARCHER, name="R1", goal="G1"),
            TeamMember(id="e1", role=SubAgentRole.EXECUTOR, name="E1", goal="G2"),
        ]
        config = TeamConfig(name="Test", goal="Goal", members=members)

        assert config.get_member("r1").name == "R1"
        assert config.get_member("e1").name == "E1"
        assert config.get_member("unknown") is None

    def test_get_manager(self):
        """get_manager returns the manager."""
        members = [
            TeamMember(id="mgr", role=SubAgentRole.PLANNER, name="Mgr", goal="Manage", is_manager=True),
            TeamMember(id="w1", role=SubAgentRole.EXECUTOR, name="W1", goal="Work"),
        ]
        config = TeamConfig(
            name="Test",
            goal="Goal",
            members=members,
            formation=TeamFormation.HIERARCHICAL,
        )

        manager = config.get_manager()
        assert manager is not None
        assert manager.id == "mgr"

    def test_get_workers(self):
        """get_workers returns non-manager members."""
        members = [
            TeamMember(id="mgr", role=SubAgentRole.PLANNER, name="Mgr", goal="Manage", is_manager=True),
            TeamMember(id="w1", role=SubAgentRole.EXECUTOR, name="W1", goal="Work1"),
            TeamMember(id="w2", role=SubAgentRole.EXECUTOR, name="W2", goal="Work2"),
        ]
        config = TeamConfig(
            name="Test",
            goal="Goal",
            members=members,
            formation=TeamFormation.HIERARCHICAL,
        )

        workers = config.get_workers()
        assert len(workers) == 2
        assert all(not w.is_manager for w in workers)

    def test_to_dict(self):
        """to_dict serializes correctly."""
        member = TeamMember(
            id="test",
            role=SubAgentRole.EXECUTOR,
            name="Test",
            goal="Goal",
        )
        config = TeamConfig(name="Team", goal="Goal", members=[member])

        d = config.to_dict()
        assert d["name"] == "Team"
        assert d["goal"] == "Goal"
        assert len(d["members"]) == 1
        assert d["formation"] == "sequential"


class TestMemberResult:
    """Test MemberResult dataclass."""

    def test_success_result(self):
        """Create successful member result."""
        result = MemberResult(
            member_id="test",
            success=True,
            output="Done!",
            tool_calls_used=5,
            duration_seconds=10.5,
        )
        assert result.success is True
        assert result.output == "Done!"
        assert result.error is None

    def test_failure_result(self):
        """Create failed member result."""
        result = MemberResult(
            member_id="test",
            success=False,
            output="",
            tool_calls_used=3,
            duration_seconds=5.0,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"


class TestTeamResult:
    """Test TeamResult dataclass."""

    def test_success_result(self):
        """Create successful team result."""
        member_result = MemberResult(
            member_id="test",
            success=True,
            output="Done",
            tool_calls_used=5,
            duration_seconds=10.0,
        )
        result = TeamResult(
            success=True,
            final_output="Team completed!",
            member_results={"test": member_result},
            total_tool_calls=5,
            total_duration=10.0,
        )
        assert result.success is True
        assert result.final_output == "Team completed!"
        assert len(result.member_results) == 1

    def test_to_dict(self):
        """to_dict serializes correctly."""
        member_result = MemberResult(
            member_id="test",
            success=True,
            output="Done",
            tool_calls_used=5,
            duration_seconds=10.0,
        )
        result = TeamResult(
            success=True,
            final_output="Done",
            member_results={"test": member_result},
            total_tool_calls=5,
            total_duration=10.0,
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["total_tool_calls"] == 5
        assert "test" in d["member_results"]


class TestMessageType:
    """Test MessageType enum."""

    def test_all_types_defined(self):
        """All expected message types are defined."""
        assert MessageType.DISCOVERY.value == "discovery"
        assert MessageType.REQUEST.value == "request"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.STATUS.value == "status"
        assert MessageType.ALERT.value == "alert"
        assert MessageType.HANDOFF.value == "handoff"
        assert MessageType.RESULT.value == "result"


class TestAgentMessage:
    """Test AgentMessage dataclass."""

    def test_minimal_message(self):
        """Create message with minimal fields."""
        msg = AgentMessage(
            type=MessageType.DISCOVERY,
            from_agent="researcher",
            content="Found something",
        )
        assert msg.type == MessageType.DISCOVERY
        assert msg.from_agent == "researcher"
        assert msg.content == "Found something"
        assert msg.to_agent is None  # Broadcast
        assert msg.id is not None  # Auto-generated

    def test_directed_message(self):
        """Create directed message."""
        msg = AgentMessage(
            type=MessageType.REQUEST,
            from_agent="manager",
            to_agent="worker",
            content="Do this task",
        )
        assert msg.to_agent == "worker"

    def test_to_dict(self):
        """to_dict serializes correctly."""
        msg = AgentMessage(
            type=MessageType.STATUS,
            from_agent="worker",
            content="50% done",
        )
        d = msg.to_dict()
        assert d["type"] == "status"
        assert d["from_agent"] == "worker"
        assert d["content"] == "50% done"

    def test_to_context_string(self):
        """to_context_string formats correctly."""
        msg = AgentMessage(
            type=MessageType.DISCOVERY,
            from_agent="researcher",
            content="Found 5 endpoints",
        )
        s = msg.to_context_string()
        assert "DISCOVERY" in s
        assert "researcher" in s
        assert "Found 5 endpoints" in s


class TestTeamMessageBus:
    """Test TeamMessageBus class."""

    def test_register_agent(self):
        """Agents can be registered."""
        bus = TeamMessageBus("test_team")
        bus.register_agent("agent1")
        bus.register_agent("agent2")
        # No exception = success
        assert True

    @pytest.mark.asyncio
    async def test_send_broadcast(self):
        """Broadcast message is delivered to all except sender."""
        bus = TeamMessageBus("test_team")
        bus.register_agent("sender")
        bus.register_agent("receiver1")
        bus.register_agent("receiver2")

        msg = AgentMessage(
            type=MessageType.DISCOVERY,
            from_agent="sender",
            content="Found something",
        )
        await bus.send(msg)

        # Receivers should have the message
        r1_msg = await bus.receive("receiver1")
        r2_msg = await bus.receive("receiver2")
        assert r1_msg is not None
        assert r2_msg is not None
        assert r1_msg.content == "Found something"

        # Sender should not have the message
        s_msg = await bus.receive("sender", timeout=0)
        assert s_msg is None

    @pytest.mark.asyncio
    async def test_send_directed(self):
        """Directed message is delivered only to recipient."""
        bus = TeamMessageBus("test_team")
        bus.register_agent("sender")
        bus.register_agent("target")
        bus.register_agent("other")

        msg = AgentMessage(
            type=MessageType.REQUEST,
            from_agent="sender",
            to_agent="target",
            content="Do this",
        )
        await bus.send(msg)

        # Target should have the message
        t_msg = await bus.receive("target")
        assert t_msg is not None
        assert t_msg.content == "Do this"

        # Other should not have the message
        o_msg = await bus.receive("other", timeout=0)
        assert o_msg is None

    def test_get_message_log(self):
        """Message log is maintained."""
        bus = TeamMessageBus("test_team")
        bus.register_agent("agent1")

        # Messages are added synchronously to log
        # (send is async, but log is immediate)
        assert len(bus.get_message_log()) == 0

    def test_get_discoveries(self):
        """get_discoveries filters by type."""
        bus = TeamMessageBus("test_team")
        # Initially empty
        discoveries = bus.get_discoveries()
        assert len(discoveries) == 0


class TestTeamSharedMemory:
    """Test TeamSharedMemory class."""

    def test_set_and_get(self):
        """Values can be set and retrieved."""
        memory = TeamSharedMemory()
        memory.set("key", "value", "agent1")
        assert memory.get("key") == "value"

    def test_get_default(self):
        """get returns default for missing keys."""
        memory = TeamSharedMemory()
        assert memory.get("missing") is None
        assert memory.get("missing", "default") == "default"

    def test_append(self):
        """Values can be appended to lists."""
        memory = TeamSharedMemory()
        memory.append("list", "item1", "agent1")
        memory.append("list", "item2", "agent2")
        assert memory.get("list") == ["item1", "item2"]

    def test_append_to_non_list_raises(self):
        """Appending to non-list raises TypeError."""
        memory = TeamSharedMemory()
        memory.set("key", "string", "agent1")
        with pytest.raises(TypeError):
            memory.append("key", "item", "agent2")

    def test_update(self):
        """Dictionary values can be updated."""
        memory = TeamSharedMemory()
        memory.update("dict", {"a": 1}, "agent1")
        memory.update("dict", {"b": 2}, "agent2")
        assert memory.get("dict") == {"a": 1, "b": 2}

    def test_has(self):
        """has checks key existence."""
        memory = TeamSharedMemory()
        assert memory.has("key") is False
        memory.set("key", "value", "agent1")
        assert memory.has("key") is True

    def test_keys(self):
        """keys returns all keys."""
        memory = TeamSharedMemory()
        memory.set("a", 1, "agent1")
        memory.set("b", 2, "agent2")
        assert set(memory.keys()) == {"a", "b"}

    def test_get_contributors(self):
        """get_contributors tracks who set values."""
        memory = TeamSharedMemory()
        memory.set("key", "v1", "agent1")
        memory.set("key", "v2", "agent2")  # Overwrites
        contributors = memory.get_contributors("key")
        assert "agent1" in contributors
        assert "agent2" in contributors

    def test_get_all(self):
        """get_all returns snapshot of all data."""
        memory = TeamSharedMemory()
        memory.set("a", 1, "agent1")
        memory.set("b", 2, "agent2")
        all_data = memory.get_all()
        assert all_data == {"a": 1, "b": 2}

    def test_get_summary(self):
        """get_summary returns formatted string."""
        memory = TeamSharedMemory()
        memory.set("findings", ["item1", "item2"], "researcher")
        summary = memory.get_summary()
        assert "findings" in summary
        assert "researcher" in summary

    def test_clear(self):
        """clear removes all data."""
        memory = TeamSharedMemory()
        memory.set("key", "value", "agent1")
        memory.clear()
        assert memory.get("key") is None
        assert len(memory.keys()) == 0


class TestTeamCoordinatorIntegration:
    """Integration tests for TeamCoordinator."""

    def test_initialization(self):
        """TeamCoordinator can be initialized with mock orchestrator."""
        mock_orchestrator = MagicMock()
        coordinator = TeamCoordinator(mock_orchestrator)
        assert coordinator.orchestrator == mock_orchestrator
        assert coordinator.sub_agents is not None

    def test_get_active_teams(self):
        """get_active_teams returns empty list initially."""
        mock_orchestrator = MagicMock()
        coordinator = TeamCoordinator(mock_orchestrator)
        assert coordinator.get_active_teams() == []


class TestModuleExports:
    """Test module exports are correct."""

    def test_teams_init_exports(self):
        """Teams __init__ exports all expected symbols."""
        from victor.agent.teams import (
            TeamFormation,
            TeamMember,
            TeamConfig,
            TeamResult,
            TeamCoordinator,
            MessageType,
            AgentMessage,
            TeamMessageBus,
            TeamSharedMemory,
        )
        # If we get here without ImportError, all exports work
        assert True
