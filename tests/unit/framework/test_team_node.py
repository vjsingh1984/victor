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

"""Unit tests for TeamNode."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from victor.framework.workflows.nodes import TeamNode, TeamNodeConfig
from victor.framework.state_merging import MergeMode, StateMergeError
from victor.teams.types import TeamFormation, TeamMember
from victor.agent.subagents import SubAgentRole


class TestTeamNodeConfig:
    """Tests for TeamNodeConfig dataclass."""

    def test_default_config(self):
        """Test creating default config."""
        config = TeamNodeConfig()

        assert config.timeout_seconds is None
        assert config.merge_strategy == "dict"
        assert config.merge_mode == MergeMode.TEAM_WINS
        assert config.output_key == "team_result"
        assert config.continue_on_error is True
        assert config.validate_before_merge is False
        assert config.required_keys is None

    def test_custom_config(self):
        """Test creating custom config."""
        config = TeamNodeConfig(
            timeout_seconds=300,
            merge_strategy="list",
            merge_mode=MergeMode.GRAPH_WINS,
            output_key="custom_result",
            continue_on_error=False,
            validate_before_merge=True,
            required_keys=["key1", "key2"],
        )

        assert config.timeout_seconds == 300
        assert config.merge_strategy == "list"
        assert config.merge_mode == MergeMode.GRAPH_WINS
        assert config.output_key == "custom_result"
        assert config.continue_on_error is False
        assert config.validate_before_merge is True
        assert config.required_keys == ["key1", "key2"]


class TestTeamNode:
    """Tests for TeamNode class."""

    @pytest.fixture
    def sample_members(self):
        """Create sample team members."""
        return [
            TeamMember(
                id="researcher",
                role=SubAgentRole.RESEARCHER,
                name="Researcher",
                goal="Find information",
                tool_budget=10,
            ),
            TeamMember(
                id="reviewer",
                role=SubAgentRole.REVIEWER,
                name="Reviewer",
                goal="Review and refine report",
                tool_budget=15,
            ),
        ]

    @pytest.fixture
    def sample_config(self):
        """Create sample team node config."""
        return TeamNodeConfig(
            timeout_seconds=300,
            merge_strategy="dict",
            merge_mode=MergeMode.TEAM_WINS,
            output_key="team_result",
        )

    @pytest.fixture
    def sample_team_node(self, sample_members, sample_config):
        """Create a sample TeamNode."""
        return TeamNode(
            id="test_team",
            name="Test Team",
            goal="Test goal",
            team_formation=TeamFormation.SEQUENTIAL,
            members=sample_members,
            config=sample_config,
            max_iterations=30,
            total_tool_budget=50,
        )

    def test_team_node_creation(self, sample_team_node):
        """Test creating a TeamNode."""
        assert sample_team_node.id == "test_team"
        assert sample_team_node.name == "Test Team"
        assert sample_team_node.goal == "Test goal"
        assert sample_team_node.team_formation == TeamFormation.SEQUENTIAL
        assert len(sample_team_node.members) == 2
        assert sample_team_node.max_iterations == 30
        assert sample_team_node.total_tool_budget == 50

    def test_team_node_to_dict(self, sample_team_node):
        """Test serializing TeamNode to dictionary."""
        data = sample_team_node.to_dict()

        assert data["id"] == "test_team"
        assert data["name"] == "Test Team"
        assert data["type"] == "team"
        assert data["goal"] == "Test goal"
        assert data["team_formation"] == "sequential"
        assert len(data["members"]) == 2
        assert data["max_iterations"] == 30
        assert data["total_tool_budget"] == 50

    def test_team_node_from_dict(self):
        """Test deserializing TeamNode from dictionary."""
        data = {
            "id": "test_team",
            "name": "Test Team",
            "goal": "Test goal",
            "team_formation": "sequential",
            "members": [
                {
                    "id": "researcher",
                    "role": "researcher",
                    "name": "Researcher",
                    "goal": "Find information",
                    "tool_budget": 10,
                },
                {
                    "id": "reviewer",
                    "role": "reviewer",
                    "name": "Reviewer",
                    "goal": "Review and refine report",
                    "tool_budget": 15,
                },
            ],
            "config": {
                "timeout_seconds": 300,
                "merge_strategy": "dict",
                "merge_mode": "team_wins",
                "output_key": "team_result",
            },
            "max_iterations": 30,
            "total_tool_budget": 50,
        }

        node = TeamNode.from_dict(data)

        assert node.id == "test_team"
        assert node.name == "Test Team"
        assert node.goal == "Test goal"
        assert node.team_formation == TeamFormation.SEQUENTIAL
        assert len(node.members) == 2
        assert node.members[0].role == SubAgentRole.RESEARCHER
        assert node.members[1].role == SubAgentRole.REVIEWER
        assert node.max_iterations == 30
        assert node.total_tool_budget == 50

    def test_build_goal_with_substitution(self, sample_members, sample_config):
        """Test building goal with context variable substitution."""
        # Create a node with goal containing placeholders
        node = TeamNode(
            id="test_team",
            name="Test Team",
            goal="Work on: ${user_task} with priority: ${priority}",
            team_formation=TeamFormation.SEQUENTIAL,
            members=sample_members,
            config=sample_config,
            max_iterations=30,
            total_tool_budget=50,
        )

        graph_state = {
            "user_task": "Implement feature X",
            "priority": "high",
        }

        goal = node._build_goal(graph_state)

        assert "user_task" not in goal  # Should be substituted
        assert "Implement feature X" in goal
        assert "high" in goal

    def test_extract_context(self, sample_team_node):
        """Test extracting context from graph state."""
        graph_state = {
            "user_task": "Implement X",
            "_internal": "secret",
            "_task": "task",
        }

        context = sample_team_node._extract_context(graph_state)

        # Should include user_task
        assert "user_task" in context
        # Should exclude _internal
        assert "_internal" not in context
        # Should include _task (exception)
        assert "_task" in context

    @pytest.mark.asyncio
    async def test_execute_async_success(self, sample_team_node):
        """Test successful team execution."""
        # Mock the team coordinator
        mock_coordinator = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_output = "Team completed successfully"
        mock_result.formation = TeamFormation.SEQUENTIAL
        mock_result.total_duration = 10.0
        mock_result.total_tool_calls = 25
        mock_result.member_results = {}
        mock_result.shared_context = {}

        with patch(
            "victor.teams.create_coordinator",
            return_value=mock_coordinator,
        ):
            with patch.object(
                mock_coordinator,
                "run",
            ) as mock_run:
                mock_run.return_value = mock_result

                graph_state = {"user_task": "Test task"}
                result = await sample_team_node.execute_async(None, graph_state)

                # Should have team result
                assert "team_result" in result
                assert result["team_result"]["success"] is True
                # Note: actual output format may differ, just check success

    @pytest.mark.asyncio
    async def test_execute_async_timeout(self, sample_team_node):
        """Test team execution with timeout."""

        # Mock that times out
        async def timeout_execute(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return MagicMock()

        with patch.object(
            sample_team_node,
            "_execute_team",
            side_effect=timeout_execute,
        ):
            # Set short timeout
            sample_team_node.config.timeout_seconds = 0.1

            graph_state = {"user_task": "Test task"}
            result = await sample_team_node.execute_async(None, graph_state)

            # Should have error but continue (continue_on_error=True)
            assert "_error" in result
            assert "_timeout" in result
            assert "timed out" in result["_error"].lower()

    @pytest.mark.asyncio
    async def test_execute_async_exception_continue(self, sample_team_node):
        """Test team execution with exception and continue_on_error."""
        with patch.object(
            sample_team_node,
            "_execute_team",
            side_effect=Exception("Team execution failed"),
        ):
            sample_team_node.config.continue_on_error = True
            graph_state = {"user_task": "Test task"}
            result = await sample_team_node.execute_async(None, graph_state)

            # Should have error but continue
            assert "_error" in result
            assert "Team execution failed" in result["_error"]

    @pytest.mark.asyncio
    async def test_execute_async_exception_raise(self, sample_team_node):
        """Test team execution with exception and continue_on_error=False."""
        with patch.object(
            sample_team_node,
            "_execute_team",
            side_effect=Exception("Team execution failed"),
        ):
            sample_team_node.config.continue_on_error = False
            graph_state = {"user_task": "Test task"}

            with pytest.raises(Exception, match="Team execution failed"):
                await sample_team_node.execute_async(None, graph_state)

    def test_merge_team_result_team_wins(self, sample_team_node):
        """Test merging team result with team_wins mode."""

        graph_state = {"key": "graph_value"}
        team_result = MagicMock()
        team_result.success = True
        team_result.final_output = "Team output"
        team_result.shared_context = {"key": "team_value"}
        team_result.formation = TeamFormation.SEQUENTIAL

        result = sample_team_node._merge_team_result(graph_state, team_result)

        # Team wins on conflict
        assert result["key"] == "team_value"

    def test_merge_team_result_graph_wins(self, sample_team_node):
        """Test merging team result with graph_wins mode."""

        sample_team_node.config.merge_mode = MergeMode.GRAPH_WINS

        graph_state = {"key": "graph_value"}
        team_result = MagicMock()
        team_result.success = True
        team_result.final_output = "Team output"
        team_result.shared_context = {"key": "team_value"}
        team_result.formation = TeamFormation.SEQUENTIAL

        result = sample_team_node._merge_team_result(graph_state, team_result)

        # Graph wins on conflict
        assert result["key"] == "graph_value"

    def test_merge_team_result_error_mode(self, sample_team_node):
        """Test merging team result with error mode raises on conflict."""

        sample_team_node.config.merge_mode = MergeMode.ERROR

        graph_state = {"key": "graph_value"}
        team_result = MagicMock()
        team_result.success = True
        team_result.final_output = "Team output"
        team_result.shared_context = {"key": "team_value"}
        team_result.formation = TeamFormation.SEQUENTIAL

        with pytest.raises(StateMergeError):
            sample_team_node._merge_team_result(graph_state, team_result)


class TestTeamNodeFormations:
    """Tests for different team formations."""

    @pytest.fixture
    def hierarchical_members(self):
        """Create hierarchical team members."""
        return [
            TeamMember(
                id="manager",
                role=SubAgentRole.PLANNER,
                name="Manager",
                goal="Coordinate work",
                is_manager=True,
                can_delegate=True,
                tool_budget=10,
            ),
            TeamMember(
                id="worker1",
                role=SubAgentRole.EXECUTOR,
                name="Worker 1",
                goal="Execute tasks",
                reports_to="manager",
                tool_budget=20,
            ),
            TeamMember(
                id="worker2",
                role=SubAgentRole.EXECUTOR,
                name="Worker 2",
                goal="Execute tasks",
                reports_to="manager",
                tool_budget=20,
            ),
        ]

    def test_sequential_formation(self, hierarchical_members):
        """Test creating sequential team node."""
        node = TeamNode(
            id="sequential_team",
            name="Sequential Team",
            goal="Execute sequentially",
            team_formation=TeamFormation.SEQUENTIAL,
            members=hierarchical_members[:2],
        )

        assert node.team_formation == TeamFormation.SEQUENTIAL

    def test_parallel_formation(self, hierarchical_members):
        """Test creating parallel team node."""
        node = TeamNode(
            id="parallel_team",
            name="Parallel Team",
            goal="Execute in parallel",
            team_formation=TeamFormation.PARALLEL,
            members=hierarchical_members,
        )

        assert node.team_formation == TeamFormation.PARALLEL

    def test_hierarchical_formation(self, hierarchical_members):
        """Test creating hierarchical team node."""
        node = TeamNode(
            id="hierarchical_team",
            name="Hierarchical Team",
            goal="Execute with hierarchy",
            team_formation=TeamFormation.HIERARCHICAL,
            members=hierarchical_members,
        )

        assert node.team_formation == TeamFormation.HIERARCHICAL

    def test_pipeline_formation(self, hierarchical_members):
        """Test creating pipeline team node."""
        node = TeamNode(
            id="pipeline_team",
            name="Pipeline Team",
            goal="Execute as pipeline",
            team_formation=TeamFormation.PIPELINE,
            members=hierarchical_members[:2],
        )

        assert node.team_formation == TeamFormation.PIPELINE

    def test_consensus_formation(self, hierarchical_members):
        """Test creating consensus team node."""
        node = TeamNode(
            id="consensus_team",
            name="Consensus Team",
            goal="Execute with consensus",
            team_formation=TeamFormation.CONSENSUS,
            members=hierarchical_members[:3],
        )

        assert node.team_formation == TeamFormation.CONSENSUS


class TestTeamNodeSyncExecute:
    """Tests for synchronous execute method."""

    @pytest.fixture
    def simple_team_node(self):
        """Create a simple team node for sync tests."""
        return TeamNode(
            id="test_team",
            name="Test Team",
            goal="Test goal",
            team_formation=TeamFormation.SEQUENTIAL,
            members=[
                TeamMember(
                    id="member1",
                    role=SubAgentRole.EXECUTOR,
                    name="Member 1",
                    goal="Execute",
                    tool_budget=10,
                )
            ],
        )

    def test_sync_execute_wraps_async(self, simple_team_node):
        """Test that sync execute properly wraps async execute."""
        with patch.object(
            simple_team_node,
            "execute_async",
            return_value=asyncio.sleep(0),
        ) as mock_execute:
            graph_state = {"key": "value"}

            # This should work without raising
            result = simple_team_node.execute(None, graph_state)

            # Verify async was called
            mock_execute.assert_called_once()


class TestTeamNodeRichPersona:
    """Tests for rich persona support in TeamNode."""

    def test_team_member_with_rich_persona(self):
        """Test creating team member with rich persona attributes."""
        member = TeamMember(
            id="expert",
            role=SubAgentRole.RESEARCHER,
            name="Expert Researcher",
            goal="Conduct deep research",
            backstory="20 years of experience in the field",
            expertise=["research", "analysis", "writing"],
            personality="methodical and thorough",
            max_delegation_depth=2,
        )

        assert member.backstory == "20 years of experience in the field"
        assert member.expertise == ["research", "analysis", "writing"]
        assert member.personality == "methodical and thorough"
        assert member.max_delegation_depth == 2

    def test_team_member_to_system_prompt(self):
        """Test generating system prompt from rich persona."""
        member = TeamMember(
            id="expert",
            role=SubAgentRole.RESEARCHER,
            name="Expert Researcher",
            goal="Conduct research",
            backstory="10 years experience",
            expertise=["analysis"],
            personality="thorough",
        )

        prompt = member.to_system_prompt()

        assert "# Role: Expert Researcher" in prompt
        assert "## Goal" in prompt
        assert "Conduct research" in prompt
        assert "## Background" in prompt
        assert "10 years experience" in prompt
        assert "## Expertise" in prompt
        assert "analysis" in prompt
        assert "## Communication Style" in prompt
        assert "thorough" in prompt
