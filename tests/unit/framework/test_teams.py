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

"""Tests for victor.framework.teams module.

These tests verify the high-level Teams API that exposes the existing
multi-agent infrastructure via the framework.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.subagents.base import SubAgentRole
from victor.agent.teams.team import TeamFormation, TeamMember, TeamConfig
from victor.framework.teams import (
    AgentTeam,
    TeamMemberSpec,
    TeamEvent,
    TeamEventType,
    ROLE_MAPPING,
    team_start_event,
    team_complete_event,
    member_start_event,
    member_complete_event,
)


# =============================================================================
# TeamMemberSpec Tests
# =============================================================================


class TestTeamMemberSpec:
    """Tests for TeamMemberSpec dataclass."""

    def test_spec_defaults(self):
        """TeamMemberSpec should have sensible defaults."""
        spec = TeamMemberSpec(role="researcher", goal="Find patterns")
        assert spec.role == "researcher"
        assert spec.goal == "Find patterns"
        assert spec.name is None
        assert spec.tool_budget is None
        assert spec.is_manager is False
        assert spec.priority == 0

    def test_spec_with_all_fields(self):
        """TeamMemberSpec should store all fields."""
        spec = TeamMemberSpec(
            role="planner",
            goal="Design implementation",
            name="Lead Planner",
            tool_budget=20,
            is_manager=True,
            priority=5,
        )
        assert spec.name == "Lead Planner"
        assert spec.tool_budget == 20
        assert spec.is_manager is True
        assert spec.priority == 5

    def test_to_team_member_basic(self):
        """to_team_member should convert spec to TeamMember."""
        spec = TeamMemberSpec(role="researcher", goal="Find code")
        member = spec.to_team_member()

        assert member.role == SubAgentRole.RESEARCHER
        assert member.goal == "Find code"
        assert member.name == "Researcher Agent"
        assert member.tool_budget == 20  # Default for researcher

    def test_to_team_member_with_custom_name(self):
        """to_team_member should use custom name if provided."""
        spec = TeamMemberSpec(role="executor", goal="Write code", name="Code Writer")
        member = spec.to_team_member()

        assert member.name == "Code Writer"

    def test_to_team_member_with_custom_budget(self):
        """to_team_member should use custom budget if provided."""
        spec = TeamMemberSpec(
            role="planner",
            goal="Design",
            tool_budget=30,
        )
        member = spec.to_team_member()

        assert member.tool_budget == 30

    def test_to_team_member_priority(self):
        """to_team_member should set priority correctly."""
        spec = TeamMemberSpec(role="reviewer", goal="Review", priority=3)
        member = spec.to_team_member(index=1)

        assert member.priority == 3  # Uses spec priority, not index

    def test_to_team_member_uses_index_as_default_priority(self):
        """to_team_member should use index as priority if not specified."""
        spec = TeamMemberSpec(role="executor", goal="Execute")
        member = spec.to_team_member(index=2)

        assert member.priority == 2  # Uses index as default

    def test_to_team_member_is_manager(self):
        """to_team_member should preserve is_manager flag."""
        spec = TeamMemberSpec(
            role="planner",
            goal="Coordinate",
            is_manager=True,
        )
        member = spec.to_team_member()

        assert member.is_manager is True


class TestTeamMemberSpecPersona:
    """Tests for TeamMemberSpec persona attributes (CrewAI-compatible)."""

    def test_persona_defaults(self):
        """TeamMemberSpec persona attributes should have sensible defaults."""
        spec = TeamMemberSpec(role="researcher", goal="Find patterns")

        assert spec.backstory == ""
        assert spec.memory is False
        assert spec.cache is True
        assert spec.verbose is False
        assert spec.max_iterations is None

    def test_persona_with_backstory(self):
        """TeamMemberSpec should store backstory."""
        spec = TeamMemberSpec(
            role="researcher",
            goal="Find patterns",
            backstory="You are a security expert with 10 years experience.",
        )

        assert spec.backstory == "You are a security expert with 10 years experience."

    def test_persona_with_memory(self):
        """TeamMemberSpec should store memory flag."""
        spec = TeamMemberSpec(
            role="researcher",
            goal="Research",
            memory=True,
        )

        assert spec.memory is True

    def test_persona_with_cache_disabled(self):
        """TeamMemberSpec should allow disabling cache."""
        spec = TeamMemberSpec(
            role="executor",
            goal="Execute",
            cache=False,
        )

        assert spec.cache is False

    def test_persona_with_verbose(self):
        """TeamMemberSpec should store verbose flag."""
        spec = TeamMemberSpec(
            role="reviewer",
            goal="Review",
            verbose=True,
        )

        assert spec.verbose is True

    def test_persona_with_max_iterations(self):
        """TeamMemberSpec should store max_iterations."""
        spec = TeamMemberSpec(
            role="executor",
            goal="Execute",
            max_iterations=25,
        )

        assert spec.max_iterations == 25

    def test_persona_all_fields(self):
        """TeamMemberSpec should store all persona fields together."""
        spec = TeamMemberSpec(
            role="researcher",
            goal="Find authentication code",
            name="Auth Researcher",
            backstory="You are a meticulous code archaeologist.",
            memory=True,
            cache=True,
            verbose=True,
            max_iterations=30,
        )

        assert spec.backstory == "You are a meticulous code archaeologist."
        assert spec.memory is True
        assert spec.cache is True
        assert spec.verbose is True
        assert spec.max_iterations == 30

    def test_to_team_member_passes_backstory(self):
        """to_team_member should pass through backstory."""
        spec = TeamMemberSpec(
            role="researcher",
            goal="Research",
            backstory="Expert in security analysis.",
        )
        member = spec.to_team_member()

        assert member.backstory == "Expert in security analysis."

    def test_to_team_member_passes_memory(self):
        """to_team_member should pass through memory flag."""
        spec = TeamMemberSpec(
            role="researcher",
            goal="Research",
            memory=True,
        )
        member = spec.to_team_member()

        assert member.memory is True

    def test_to_team_member_passes_cache(self):
        """to_team_member should pass through cache flag."""
        spec = TeamMemberSpec(
            role="executor",
            goal="Execute",
            cache=False,
        )
        member = spec.to_team_member()

        assert member.cache is False

    def test_to_team_member_passes_verbose(self):
        """to_team_member should pass through verbose flag."""
        spec = TeamMemberSpec(
            role="researcher",
            goal="Debug",
            verbose=True,
        )
        member = spec.to_team_member()

        assert member.verbose is True

    def test_to_team_member_passes_max_iterations(self):
        """to_team_member should pass through max_iterations."""
        spec = TeamMemberSpec(
            role="executor",
            goal="Execute",
            max_iterations=50,
        )
        member = spec.to_team_member()

        assert member.max_iterations == 50

    def test_to_team_member_passes_all_persona_fields(self):
        """to_team_member should pass through all persona fields."""
        spec = TeamMemberSpec(
            role="researcher",
            goal="Research auth patterns",
            backstory="Security expert with deep knowledge.",
            memory=True,
            cache=False,
            verbose=True,
            max_iterations=20,
        )
        member = spec.to_team_member()

        assert member.backstory == "Security expert with deep knowledge."
        assert member.memory is True
        assert member.cache is False
        assert member.verbose is True
        assert member.max_iterations == 20


class TestRoleMapping:
    """Tests for role string to SubAgentRole mapping."""

    def test_researcher_mappings(self):
        """Researcher role aliases should map correctly."""
        assert ROLE_MAPPING["researcher"] == SubAgentRole.RESEARCHER
        assert ROLE_MAPPING["research"] == SubAgentRole.RESEARCHER
        assert ROLE_MAPPING["analyzer"] == SubAgentRole.RESEARCHER

    def test_planner_mappings(self):
        """Planner role aliases should map correctly."""
        assert ROLE_MAPPING["planner"] == SubAgentRole.PLANNER
        assert ROLE_MAPPING["plan"] == SubAgentRole.PLANNER

    def test_executor_mappings(self):
        """Executor role aliases should map correctly."""
        assert ROLE_MAPPING["executor"] == SubAgentRole.EXECUTOR
        assert ROLE_MAPPING["execute"] == SubAgentRole.EXECUTOR
        assert ROLE_MAPPING["impl"] == SubAgentRole.EXECUTOR
        assert ROLE_MAPPING["implementer"] == SubAgentRole.EXECUTOR
        assert ROLE_MAPPING["writer"] == SubAgentRole.EXECUTOR

    def test_reviewer_mappings(self):
        """Reviewer role aliases should map correctly."""
        assert ROLE_MAPPING["reviewer"] == SubAgentRole.REVIEWER
        assert ROLE_MAPPING["review"] == SubAgentRole.REVIEWER
        assert ROLE_MAPPING["critic"] == SubAgentRole.REVIEWER
        assert ROLE_MAPPING["verifier"] == SubAgentRole.REVIEWER

    def test_unknown_role_defaults_to_executor(self):
        """Unknown roles should default to executor."""
        spec = TeamMemberSpec(role="unknown_role", goal="Do something")
        member = spec.to_team_member()

        assert member.role == SubAgentRole.EXECUTOR


# =============================================================================
# TeamEvent Tests
# =============================================================================


class TestTeamEvent:
    """Tests for TeamEvent dataclass."""

    def test_event_defaults(self):
        """TeamEvent should have sensible defaults."""
        event = TeamEvent(
            type=TeamEventType.TEAM_START,
            team_name="Test Team",
        )
        assert event.team_name == "Test Team"
        assert event.member_id is None
        assert event.member_name is None
        assert event.progress == 0.0
        assert event.message == ""
        assert event.error is None

    def test_is_member_event(self):
        """is_member_event should identify member events."""
        assert TeamEvent(type=TeamEventType.MEMBER_START, team_name="T").is_member_event is True

        assert TeamEvent(type=TeamEventType.MEMBER_COMPLETE, team_name="T").is_member_event is True

        assert TeamEvent(type=TeamEventType.TEAM_START, team_name="T").is_member_event is False

    def test_is_lifecycle_event(self):
        """is_lifecycle_event should identify lifecycle events."""
        assert TeamEvent(type=TeamEventType.TEAM_START, team_name="T").is_lifecycle_event is True

        assert TeamEvent(type=TeamEventType.TEAM_COMPLETE, team_name="T").is_lifecycle_event is True

        assert TeamEvent(type=TeamEventType.MEMBER_START, team_name="T").is_lifecycle_event is False

    def test_to_dict(self):
        """to_dict should serialize event correctly."""
        event = TeamEvent(
            type=TeamEventType.MEMBER_COMPLETE,
            team_name="Test",
            member_id="m1",
            member_name="Researcher",
            progress=50.0,
            message="Done",
        )
        d = event.to_dict()

        assert d["type"] == "member_complete"
        assert d["team_name"] == "Test"
        assert d["member_id"] == "m1"
        assert d["progress"] == 50.0


class TestEventConstructors:
    """Tests for event convenience constructors."""

    def test_team_start_event(self):
        """team_start_event should create correct event."""
        event = team_start_event("My Team", message="Starting up")

        assert event.type == TeamEventType.TEAM_START
        assert event.team_name == "My Team"
        assert event.message == "Starting up"

    def test_team_complete_event_success(self):
        """team_complete_event should create success event."""
        event = team_complete_event("My Team", success=True)

        assert event.type == TeamEventType.TEAM_COMPLETE
        assert event.team_name == "My Team"

    def test_team_complete_event_failure(self):
        """team_complete_event should create error event on failure."""
        event = team_complete_event("My Team", success=False)

        assert event.type == TeamEventType.TEAM_ERROR
        assert event.team_name == "My Team"

    def test_member_start_event(self):
        """member_start_event should create correct event."""
        event = member_start_event(
            team_name="Team",
            member_id="m1",
            member_name="Researcher",
        )

        assert event.type == TeamEventType.MEMBER_START
        assert event.member_id == "m1"
        assert event.member_name == "Researcher"


# =============================================================================
# AgentTeam Tests
# =============================================================================


class TestAgentTeamConfig:
    """Tests for AgentTeam configuration building."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orch = MagicMock()
        orch.provider = MagicMock()
        orch.provider.name = "mock"
        return orch

    @pytest.fixture
    def basic_members(self):
        """Create basic member specs."""
        return [
            TeamMemberSpec(role="researcher", goal="Find code"),
            TeamMemberSpec(role="executor", goal="Write code"),
        ]

    @pytest.mark.asyncio
    async def test_create_builds_team_config(self, mock_orchestrator, basic_members):
        """create should build TeamConfig correctly."""
        team = await AgentTeam.create(
            orchestrator=mock_orchestrator,
            name="Test Team",
            goal="Test goal",
            members=basic_members,
        )

        assert team.name == "Test Team"
        assert team.goal == "Test goal"
        assert len(team.members) == 2
        assert team.formation == TeamFormation.SEQUENTIAL

    @pytest.mark.asyncio
    async def test_create_with_formation(self, mock_orchestrator, basic_members):
        """create should set formation correctly."""
        team = await AgentTeam.create(
            orchestrator=mock_orchestrator,
            name="Team",
            goal="Goal",
            members=basic_members,
            formation=TeamFormation.PIPELINE,
        )

        assert team.formation == TeamFormation.PIPELINE

    @pytest.mark.asyncio
    async def test_create_with_budgets(self, mock_orchestrator, basic_members):
        """create should set budgets correctly."""
        team = await AgentTeam.create(
            orchestrator=mock_orchestrator,
            name="Team",
            goal="Goal",
            members=basic_members,
            total_tool_budget=200,
            max_iterations=100,
            timeout_seconds=1200,
        )

        assert team.config.total_tool_budget == 200
        assert team.config.max_iterations == 100
        assert team.config.timeout_seconds == 1200

    @pytest.mark.asyncio
    async def test_create_hierarchical_auto_manager(self, mock_orchestrator):
        """create should auto-assign manager for hierarchical if missing."""
        members = [
            TeamMemberSpec(role="planner", goal="Coordinate"),
            TeamMemberSpec(role="executor", goal="Execute"),
        ]

        team = await AgentTeam.create(
            orchestrator=mock_orchestrator,
            name="Team",
            goal="Goal",
            members=members,
            formation=TeamFormation.HIERARCHICAL,
        )

        # First member should be manager
        assert team.members[0].is_manager is True

    @pytest.mark.asyncio
    async def test_create_hierarchical_preserves_explicit_manager(self, mock_orchestrator):
        """create should preserve explicitly set manager."""
        members = [
            TeamMemberSpec(role="researcher", goal="Research"),
            TeamMemberSpec(role="planner", goal="Coordinate", is_manager=True),
            TeamMemberSpec(role="executor", goal="Execute"),
        ]

        team = await AgentTeam.create(
            orchestrator=mock_orchestrator,
            name="Team",
            goal="Goal",
            members=members,
            formation=TeamFormation.HIERARCHICAL,
        )

        # Second member (planner) should be manager
        manager = next(m for m in team.members if m.is_manager)
        assert manager.role == SubAgentRole.PLANNER

    @pytest.mark.asyncio
    async def test_create_with_shared_context(self, mock_orchestrator, basic_members):
        """create should include shared context."""
        context = {"key": "value", "number": 42}

        team = await AgentTeam.create(
            orchestrator=mock_orchestrator,
            name="Team",
            goal="Goal",
            members=basic_members,
            shared_context=context,
        )

        assert team.config.shared_context == context


class TestAgentTeamProperties:
    """Tests for AgentTeam properties."""

    @pytest.fixture
    def mock_team(self):
        """Create a mock team."""
        config = TeamConfig(
            name="Test Team",
            goal="Test goal",
            members=[
                TeamMember(
                    id="m1",
                    role=SubAgentRole.RESEARCHER,
                    name="Researcher",
                    goal="Research",
                ),
                TeamMember(
                    id="m2",
                    role=SubAgentRole.EXECUTOR,
                    name="Executor",
                    goal="Execute",
                ),
            ],
            formation=TeamFormation.PIPELINE,
        )
        coordinator = MagicMock()
        return AgentTeam(coordinator, config)

    def test_name_property(self, mock_team):
        """name should return team name."""
        assert mock_team.name == "Test Team"

    def test_goal_property(self, mock_team):
        """goal should return team goal."""
        assert mock_team.goal == "Test goal"

    def test_formation_property(self, mock_team):
        """formation should return team formation."""
        assert mock_team.formation == TeamFormation.PIPELINE

    def test_members_property(self, mock_team):
        """members should return team members."""
        assert len(mock_team.members) == 2
        assert mock_team.members[0].name == "Researcher"

    def test_config_property(self, mock_team):
        """config should return TeamConfig."""
        assert mock_team.config.name == "Test Team"

    def test_result_initially_none(self, mock_team):
        """result should be None before run."""
        assert mock_team.result is None

    def test_repr(self, mock_team):
        """__repr__ should be informative."""
        r = repr(mock_team)
        assert "Test Team" in r
        assert "pipeline" in r
        assert "2" in r  # 2 members


class TestAgentTeamExecution:
    """Tests for AgentTeam execution methods."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a mock coordinator with execute_team."""
        from victor.agent.teams.team import TeamResult, MemberResult

        coordinator = MagicMock()
        coordinator.execute_team = AsyncMock(
            return_value=TeamResult(
                success=True,
                final_output="Team completed",
                member_results={
                    "m1": MemberResult(
                        member_id="m1",
                        success=True,
                        output="Done",
                        tool_calls_used=5,
                        duration_seconds=1.0,
                    ),
                },
                total_tool_calls=5,
                total_duration=2.0,
            )
        )
        return coordinator

    @pytest.fixture
    def team_with_mock(self, mock_coordinator):
        """Create team with mock coordinator."""
        config = TeamConfig(
            name="Test",
            goal="Goal",
            members=[
                TeamMember(
                    id="m1",
                    role=SubAgentRole.EXECUTOR,
                    name="Executor",
                    goal="Execute",
                ),
            ],
        )
        return AgentTeam(mock_coordinator, config)

    @pytest.mark.asyncio
    async def test_run_executes_team(self, team_with_mock, mock_coordinator):
        """run should call coordinator.execute_team."""
        result = await team_with_mock.run()

        mock_coordinator.execute_team.assert_called_once()
        assert result.success is True
        assert result.final_output == "Team completed"

    @pytest.mark.asyncio
    async def test_run_stores_result(self, team_with_mock):
        """run should store result in team."""
        await team_with_mock.run()

        assert team_with_mock.result is not None
        assert team_with_mock.result.success is True

    @pytest.mark.asyncio
    async def test_get_member_result(self, team_with_mock):
        """get_member_result should return result for member."""
        await team_with_mock.run()

        result = team_with_mock.get_member_result("m1")
        assert result is not None
        assert result.success is True
        assert result.tool_calls_used == 5

    @pytest.mark.asyncio
    async def test_get_member_result_not_found(self, team_with_mock):
        """get_member_result should return None for unknown member."""
        await team_with_mock.run()

        result = team_with_mock.get_member_result("unknown")
        assert result is None


class TestAgentTeamFromAgent:
    """Tests for creating teams from Agent instances."""

    @pytest.mark.asyncio
    async def test_from_agent_creates_team(self):
        """from_agent should create team from Agent."""
        mock_agent = MagicMock()
        mock_orchestrator = MagicMock()
        mock_agent.get_orchestrator.return_value = mock_orchestrator

        members = [
            TeamMemberSpec(role="researcher", goal="Research"),
        ]

        team = await AgentTeam.from_agent(
            mock_agent,
            name="Test",
            goal="Goal",
            members=members,
        )

        assert team.name == "Test"
        mock_agent.get_orchestrator.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestFrameworkExports:
    """Tests for framework module exports."""

    def test_teams_exported_from_framework(self):
        """Teams should be exported from victor.framework."""
        from victor.framework import (
            AgentTeam,
            TeamMemberSpec,
            TeamEvent,
            TeamEventType,
            TeamFormation,
        )

        assert AgentTeam is not None
        assert TeamMemberSpec is not None
        assert TeamEvent is not None
        assert TeamEventType is not None
        assert TeamFormation is not None

    def test_team_event_types(self):
        """All team event types should be available."""
        from victor.framework import TeamEventType

        assert TeamEventType.TEAM_START.value == "team_start"
        assert TeamEventType.TEAM_COMPLETE.value == "team_complete"
        assert TeamEventType.TEAM_ERROR.value == "team_error"
        assert TeamEventType.MEMBER_START.value == "member_start"
        assert TeamEventType.MEMBER_PROGRESS.value == "member_progress"
        assert TeamEventType.MEMBER_COMPLETE.value == "member_complete"
        assert TeamEventType.MEMBER_ERROR.value == "member_error"
        assert TeamEventType.MESSAGE_SENT.value == "message_sent"
        assert TeamEventType.HANDOFF.value == "handoff"

    def test_team_formations(self):
        """All team formations should be available."""
        from victor.framework import TeamFormation

        assert TeamFormation.SEQUENTIAL.value == "sequential"
        assert TeamFormation.PARALLEL.value == "parallel"
        assert TeamFormation.HIERARCHICAL.value == "hierarchical"
        assert TeamFormation.PIPELINE.value == "pipeline"
