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

"""Unit tests for Agent Swarming."""

import pytest
from victor.teams.swarming import (
    AgentSwarm,
    SwarmConfig,
    ConsensusStrategy,
    VotingStrategy,
    ConflictResolution,
    AgentVote,
    ConsensusResult,
)


class TestAgentSwarm:
    """Test suite for AgentSwarm."""

    @pytest.fixture
    def swarm(self):
        """Create AgentSwarm instance."""
        config = SwarmConfig(
            agent_count=5,
            consensus_strategy=ConsensusStrategy.MAJORITY_VOTE,
            voting_strategy=VotingStrategy.PLURALITY,
            agreement_threshold=0.67,
        )
        return AgentSwarm(orchestrator=None, config=config)

    def test_initialization(self, swarm):
        """Test swarm initialization."""
        assert swarm.config is not None
        assert swarm.config.agent_count == 5
        assert swarm.config.consensus_strategy == ConsensusStrategy.MAJORITY_VOTE
        assert swarm.config.agreement_threshold == 0.67

    @pytest.mark.asyncio
    async def test_majority_consensus(self, swarm):
        """Test majority consensus."""
        votes = [
            AgentVote(agent_id="agent_1", choice="option_a", confidence=0.9),
            AgentVote(agent_id="agent_2", choice="option_a", confidence=0.8),
            AgentVote(agent_id="agent_3", choice="option_b", confidence=0.7),
        ]

        result = await swarm._majority_consensus(votes)

        assert result.achieved is True
        assert result.decision == "option_a"
        assert result.agreement_score == 2 / 3

    @pytest.mark.asyncio
    async def test_majority_consensus_no_majority(self, swarm):
        """Test majority consensus with no majority."""
        votes = [
            AgentVote(agent_id="agent_1", choice="option_a", confidence=0.9),
            AgentVote(agent_id="agent_2", choice="option_b", confidence=0.8),
            AgentVote(agent_id="agent_3", choice="option_c", confidence=0.7),
        ]

        result = await swarm._majority_consensus(votes)

        assert result.achieved is False
        assert result.agreement_score == 1 / 3

    @pytest.mark.asyncio
    async def test_unanimous_consensus_success(self, swarm):
        """Test unanimous consensus with agreement."""
        votes = [
            AgentVote(agent_id="agent_1", choice="option_a", confidence=0.9),
            AgentVote(agent_id="agent_2", choice="option_a", confidence=0.8),
            AgentVote(agent_id="agent_3", choice="option_a", confidence=0.7),
        ]

        result = await swarm._unanimous_consensus(votes)

        assert result.achieved is True
        assert result.decision == "option_a"
        assert result.agreement_score == 1.0

    @pytest.mark.asyncio
    async def test_unanimous_consensus_failure(self, swarm):
        """Test unanimous consensus with disagreement."""
        votes = [
            AgentVote(agent_id="agent_1", choice="option_a", confidence=0.9),
            AgentVote(agent_id="agent_2", choice="option_a", confidence=0.8),
            AgentVote(agent_id="agent_3", choice="option_b", confidence=0.7),
        ]

        result = await swarm._unanimous_consensus(votes)

        assert result.achieved is False
        assert result.agreement_score == 0.0

    @pytest.mark.asyncio
    async def test_supermajority_consensus(self, swarm):
        """Test supermajority consensus (2/3)."""
        votes = [
            AgentVote(agent_id="agent_1", choice="option_a", confidence=0.9),
            AgentVote(agent_id="agent_2", choice="option_a", confidence=0.8),
            AgentVote(agent_id="agent_3", choice="option_a", confidence=0.7),
        ]

        result = await swarm._supermajority_consensus(votes)

        assert result.achieved is True
        assert result.decision == "option_a"
        assert result.agreement_score == 1.0

    def test_calculate_agreement(self, swarm):
        """Test agreement calculation."""
        votes = [
            AgentVote(agent_id="agent_1", choice="option_a", confidence=0.9),
            AgentVote(agent_id="agent_2", choice="option_a", confidence=0.8),
            AgentVote(agent_id="agent_3", choice="option_b", confidence=0.7),
        ]

        agreement = swarm._calculate_agreement(votes)

        assert agreement == 2 / 3

    def test_check_convergence(self, swarm):
        """Test convergence check."""
        votes = [
            AgentVote(agent_id="agent_1", choice="option_a", confidence=0.9),
            AgentVote(agent_id="agent_2", choice="option_a", confidence=0.8),
            AgentVote(agent_id="agent_3", choice="option_a", confidence=0.7),
        ]

        converged = swarm._check_convergence(votes)

        assert converged is True

    def test_check_convergence_not_met(self, swarm):
        """Test convergence check when not met."""
        votes = [
            AgentVote(agent_id="agent_1", choice="option_a", confidence=0.9),
            AgentVote(agent_id="agent_2", choice="option_b", confidence=0.8),
            AgentVote(agent_id="agent_3", choice="option_c", confidence=0.7),
        ]

        converged = swarm._check_convergence(votes)

        assert converged is False

    @pytest.mark.asyncio
    async def test_plurality_vote(self, swarm):
        """Test plurality voting."""
        options = ["option_a", "option_b", "option_c"]
        agents = ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]

        winner, votes = await swarm._plurality_vote(options, agents)

        assert winner in options
        assert len(votes) == len(agents)

    def test_merge_proposals(self, swarm):
        """Test proposal merging."""
        proposals = {
            "agent_1": {"suggestion": "A", "reason": "reason 1"},
            "agent_2": {"suggestion": "B", "reason": "reason 2"},
        }

        merged = swarm._merge_proposals(proposals)

        assert "suggestion" in merged or "reason" in merged

    def test_find_compromise(self, swarm):
        """Test finding compromise."""
        conflict = {
            "options": ["left", "middle", "right"],
        }

        compromise = swarm._find_compromise(conflict)

        assert compromise == "middle"


class TestAgentVote:
    """Test suite for AgentVote."""

    def test_vote_creation(self):
        """Test creating vote."""
        vote = AgentVote(
            agent_id="agent_1",
            choice="option_a",
            confidence=0.9,
            reasoning="This is the best option",
            weight=1.5,
        )

        assert vote.agent_id == "agent_1"
        assert vote.choice == "option_a"
        assert vote.confidence == 0.9
        assert vote.reasoning == "This is the best option"
        assert vote.weight == 1.5


class TestConsensusResult:
    """Test suite for ConsensusResult."""

    def test_result_creation(self):
        """Test creating consensus result."""
        result = ConsensusResult(
            achieved=True,
            decision="option_a",
            votes=[],
            agreement_score=0.8,
            iterations=2,
        )

        assert result.achieved is True
        assert result.decision == "option_a"
        assert result.agreement_score == 0.8
        assert result.iterations == 2


class TestSwarmConfig:
    """Test suite for SwarmConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SwarmConfig()

        assert config.agent_count == 5
        assert config.consensus_strategy == ConsensusStrategy.MAJORITY_VOTE
        assert config.voting_strategy == VotingStrategy.PLURALITY
        assert config.agreement_threshold == 0.67

    def test_custom_config(self):
        """Test custom configuration."""
        config = SwarmConfig(
            agent_count=10,
            consensus_strategy=ConsensusStrategy.SUPERMAJORITY,
            agreement_threshold=0.75,
        )

        assert config.agent_count == 10
        assert config.consensus_strategy == ConsensusStrategy.SUPERMAJORITY
        assert config.agreement_threshold == 0.75
