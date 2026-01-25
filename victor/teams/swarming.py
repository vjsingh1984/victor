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

"""Agent swarming with advanced coordination patterns.

This module implements sophisticated multi-agent coordination including:
- Swarm coordination patterns
- Consensus mechanisms
- Voting strategies
- Conflict resolution
- Dynamic swarm composition
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


class ConsensusStrategy(str, Enum):
    """Consensus building strategies."""

    MAJORITY_VOTE = "majority_vote"  # Simple majority
    UNANIMOUS = "unanimous"  # All must agree
    WEIGHTED = "weighted"  # Weighted by agent expertise
    SUPERMAJORITY = "supermajority"  # 2/3 majority
    BFT = "bft"  # Byzantine fault tolerant
    DELPHI = "delphi"  # Iterative refinement


class VotingStrategy(str, Enum):
    """Voting strategies."""

    PLURALITY = "plurality"  # Most votes wins
    BORDA = "borda"  # Borda count
    APPROVAL = "approval"  # Approve multiple options
    RANKED = "ranked"  # Ranked choice voting
    QUADRATIC = "quadratic"  # Quadratic voting


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""

    MERGE = "merge"  # Merge conflicting proposals
    VOTE = "vote"  # Vote on conflicts
    ARBITRATE = "arbitrate"  # Use arbitrator agent
    DEFER = "defer"  # Defer to human
    COMPROMISE = "compromise"  # Find middle ground


@dataclass
class AgentVote:
    """Vote from an agent.

    Attributes:
        agent_id: Agent identifier
        choice: Selected choice/option
        confidence: Confidence in choice (0-1)
        reasoning: Reasoning for choice
        weight: Vote weight (for weighted voting)
    """

    agent_id: str
    choice: Any
    confidence: float
    reasoning: str = ""
    weight: float = 1.0


@dataclass
class ConsensusResult:
    """Result from consensus process.

    Attributes:
        achieved: Whether consensus was achieved
        decision: Final decision
        votes: All votes cast
        agreement_score: Degree of agreement (0-1)
        iterations: Number of iterations needed
        conflicts: Any conflicts identified
    """

    achieved: bool
    decision: Any
    votes: List[AgentVote] = field(default_factory=list)
    agreement_score: float = 0.0
    iterations: int = 0
    conflicts: List[str] = field(default_factory=list)


@dataclass
class SwarmConfig:
    """Configuration for agent swarm.

    Attributes:
        agent_count: Number of agents in swarm
        consensus_strategy: How to build consensus
        voting_strategy: How to vote
        conflict_resolution: How to resolve conflicts
        max_iterations: Maximum consensus iterations
        agreement_threshold: Minimum agreement for consensus
        diversity_penalty: Penalty for similar opinions
    """

    agent_count: int = 5
    consensus_strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY_VOTE
    voting_strategy: VotingStrategy = VotingStrategy.PLURALITY
    conflict_resolution: ConflictResolution = ConflictResolution.VOTE
    max_iterations: int = 3
    agreement_threshold: float = 0.67  # 2/3 supermajority
    diversity_penalty: float = 0.1


class AgentSwarm:
    """Agent swarm with advanced coordination.

    Example:
        from victor.teams import AgentSwarm, SwarmConfig
        from victor.teams.types import TeamFormation

        # Create swarm
        swarm = AgentSwarm(
            orchestrator=orchestrator,
            config=SwarmConfig(
                agent_count=7,
                consensus_strategy=ConsensusStrategy.SUPERMAJORITY
            )
        )

        # Execute task with swarm
        result = await swarm.execute_task(
            task="Design a REST API",
            options=["approach_a", "approach_b", "approach_c"]
        )

        # Build consensus
        consensus = await swarm.build_consensus(
            proposals={
                "agent_1": proposal_1,
                "agent_2": proposal_2,
            }
        )
    """

    def __init__(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        config: Optional[SwarmConfig] = None,
    ):
        """Initialize agent swarm.

        Args:
            orchestrator: Agent orchestrator for spawning agents
            config: Swarm configuration
        """
        from victor.teams import create_coordinator

        self.orchestrator = orchestrator
        self.config = config or SwarmConfig()
        self.coordinator = create_coordinator(orchestrator) if orchestrator else None

    async def execute_task(
        self,
        task: str,
        options: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConsensusResult:
        """Execute task with swarm consensus.

        Args:
            task: Task description
            options: Optional list of options to choose from
            context: Additional context

        Returns:
            ConsensusResult with final decision
        """
        if not self.coordinator:
            return ConsensusResult(achieved=False, decision=None, conflicts=["No coordinator configured"])

        # Stage 1: Generate proposals
        proposals = await self._generate_proposals(task, context)

        # Stage 2: Build consensus
        consensus = await self.build_consensus(proposals, options)

        return consensus

    async def _generate_proposals(
        self, task: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate proposals from agents."""
        if not self.coordinator:
            return {}

        # For now, use parallel formation
        from victor.teams.types import TeamFormation

        self.coordinator.set_formation(TeamFormation.PARALLEL)

        # Execute with all agents
        result = await self.coordinator.execute_task(task, context or {})

        # Extract proposals from member results
        proposals = {}
        member_results = result.get("member_results", {})
        if isinstance(member_results, dict):  # type: ignore[attr-defined]
            for member_id, member_result in member_results.items():
                if member_result.get("success", False):
                    proposals[member_id] = member_result.get("result")

        return proposals

    async def build_consensus(
        self,
        proposals: Dict[str, Any],
        options: Optional[List[Any]] = None,
    ) -> ConsensusResult:
        """Build consensus from agent proposals.

        Args:
            proposals: Dict of agent_id -> proposal
            options: Optional list of options

        Returns:
            ConsensusResult
        """
        votes = []

        # Convert proposals to votes
        for agent_id, proposal in proposals.items():
            if isinstance(proposal, dict):
                choice = proposal.get("choice", proposal)
                confidence = proposal.get("confidence", 0.8)
                reasoning = proposal.get("reasoning", "")
            else:
                choice = proposal
                confidence = 0.8
                reasoning = ""

            votes.append(
                AgentVote(
                    agent_id=agent_id,
                    choice=choice,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )

        # Apply consensus strategy
        if self.config.consensus_strategy == ConsensusStrategy.MAJORITY_VOTE:
            return await self._majority_consensus(votes)
        elif self.config.consensus_strategy == ConsensusStrategy.UNANIMOUS:
            return await self._unanimous_consensus(votes)
        elif self.config.consensus_strategy == ConsensusStrategy.WEIGHTED:
            return await self._weighted_consensus(votes)
        elif self.config.consensus_strategy == ConsensusStrategy.SUPERMAJORITY:
            return await self._supermajority_consensus(votes)
        elif self.config.consensus_strategy == ConsensusStrategy.DELPHI:
            return await self._delphi_consensus(votes, proposals)
        else:
            return await self._majority_consensus(votes)

    async def _majority_consensus(self, votes: List[AgentVote]) -> ConsensusResult:
        """Simple majority consensus."""
        if not votes:
            return ConsensusResult(achieved=False, decision=None)

        # Count votes
        vote_counts: Dict[Any, int] = {}
        for vote in votes:
            choice = vote.choice
            vote_counts[choice] = vote_counts.get(choice, 0) + 1

        # Find majority
        total_votes = len(votes)
        majority_threshold = total_votes / 2

        for choice, count in vote_counts.items():
            if count > majority_threshold:
                # Calculate agreement score
                agreement = count / total_votes

                return ConsensusResult(
                    achieved=True,
                    decision=choice,
                    votes=votes,
                    agreement_score=agreement,
                    iterations=1,
                )

        # No majority
        best_choice = max(vote_counts.items(), key=lambda x: x[1])[0]
        return ConsensusResult(
            achieved=False,
            decision=best_choice,
            votes=votes,
            agreement_score=max(vote_counts.values()) / total_votes,
            iterations=1,
        )

    async def _unanimous_consensus(self, votes: List[AgentVote]) -> ConsensusResult:
        """Unanimous consensus."""
        if not votes:
            return ConsensusResult(achieved=False, decision=None)

        # Check if all votes are the same
        first_choice = votes[0].choice
        all_same = all(vote.choice == first_choice for vote in votes)

        if all_same:
            return ConsensusResult(
                achieved=True,
                decision=first_choice,
                votes=votes,
                agreement_score=1.0,
                iterations=1,
            )
        else:
            return ConsensusResult(
                achieved=False,
                decision=first_choice,
                votes=votes,
                agreement_score=0.0,
                iterations=1,
                conflicts=["Agents could not reach unanimous agreement"],
            )

    async def _weighted_consensus(self, votes: List[AgentVote]) -> ConsensusResult:
        """Weighted consensus based on agent expertise."""
        if not votes:
            return ConsensusResult(achieved=False, decision=None)

        # Calculate weighted scores
        weighted_scores: Dict[Any, float] = {}
        for vote in votes:
            choice = vote.choice
            # Combine weight and confidence
            score = vote.weight * vote.confidence
            weighted_scores[choice] = weighted_scores.get(choice, 0) + score

        # Find highest weighted choice
        best_choice = max(weighted_scores.items(), key=lambda x: x[1])[0]
        best_score = weighted_scores[best_choice]
        total_score = sum(weighted_scores.values())

        return ConsensusResult(
            achieved=True,
            decision=best_choice,
            votes=votes,
            agreement_score=best_score / total_score if total_score > 0 else 0,
            iterations=1,
        )

    async def _supermajority_consensus(self, votes: List[AgentVote]) -> ConsensusResult:
        """Supermajority consensus (2/3)."""
        if not votes:
            return ConsensusResult(achieved=False, decision=None)

        # Count votes
        vote_counts: Dict[Any, int] = {}
        for vote in votes:
            choice = vote.choice
            vote_counts[choice] = vote_counts.get(choice, 0) + 1

        # Find supermajority
        total_votes = len(votes)
        supermajority_threshold = total_votes * self.config.agreement_threshold

        for choice, count in vote_counts.items():
            if count >= supermajority_threshold:
                agreement = count / total_votes

                return ConsensusResult(
                    achieved=True,
                    decision=choice,
                    votes=votes,
                    agreement_score=agreement,
                    iterations=1,
                )

        # No supermajority
        best_choice = max(vote_counts.items(), key=lambda x: x[1])[0]
        return ConsensusResult(
            achieved=False,
            decision=best_choice,
            votes=votes,
            agreement_score=max(vote_counts.values()) / total_votes,
            iterations=1,
        )

    async def _delphi_consensus(
        self, initial_votes: List[AgentVote], proposals: Dict[str, Any]
    ) -> ConsensusResult:
        """Delphi method - iterative consensus building."""
        votes = initial_votes
        iterations = 0
        max_iterations = self.config.max_iterations

        while iterations < max_iterations:
            iterations += 1

            # Check for convergence
            if self._check_convergence(votes):
                return ConsensusResult(
                    achieved=True,
                    decision=votes[0].choice,
                    votes=votes,
                    agreement_score=self._calculate_agreement(votes),
                    iterations=iterations,
                )

            # Refine votes based on group feedback
            votes = await self._refine_votes(votes, proposals)

        # No convergence after max iterations
        return ConsensusResult(
            achieved=False,
            decision=votes[0].choice if votes else None,
            votes=votes,
            agreement_score=self._calculate_agreement(votes) if votes else 0,
            iterations=iterations,
            conflicts=["Failed to converge after max iterations"],
        )

    def _check_convergence(self, votes: List[AgentVote]) -> bool:
        """Check if votes have converged."""
        if not votes:
            return False

        # Check if agreement threshold is met
        vote_counts: Dict[Any, int] = {}
        for vote in votes:
            choice = vote.choice
            vote_counts[choice] = vote_counts.get(choice, 0) + 1

        max_count = max(vote_counts.values()) if vote_counts else 0
        agreement_ratio = max_count / len(votes) if votes else 0

        return agreement_ratio >= self.config.agreement_threshold

    def _calculate_agreement(self, votes: List[AgentVote]) -> float:
        """Calculate agreement score."""
        if not votes:
            return 0.0

        vote_counts: Dict[Any, int] = {}
        for vote in votes:
            choice = vote.choice
            vote_counts[choice] = vote_counts.get(choice, 0) + 1

        max_count = max(vote_counts.values()) if vote_counts else 0
        return max_count / len(votes)

    async def _refine_votes(
        self, votes: List[AgentVote], proposals: Dict[str, Any]
    ) -> List[AgentVote]:
        """Refine votes based on group feedback."""
        # In a real implementation, this would:
        # 1. Share reasoning and confidence with all agents
        # 2. Allow agents to update their choices
        # 3. Encourage convergence toward consensus

        # For now, return unchanged votes
        return votes

    async def vote(
        self,
        options: List[Any],
        agents: List[str],
        strategy: Optional[VotingStrategy] = None,
    ) -> Tuple[Any, List[AgentVote]]:
        """Conduct vote among agents.

        Args:
            options: List of options to vote on
            agents: List of agent IDs
            strategy: Voting strategy

        Returns:
            Tuple of (winner, votes)
        """
        strategy = strategy or self.config.voting_strategy

        if strategy == VotingStrategy.PLURALITY:
            return await self._plurality_vote(options, agents)
        elif strategy == VotingStrategy.BORDA:
            return await self._borda_vote(options, agents)
        elif strategy == VotingStrategy.APPROVAL:
            return await self._approval_vote(options, agents)
        else:
            return await self._plurality_vote(options, agents)

    async def _plurality_vote(
        self, options: List[Any], agents: List[str]
    ) -> Tuple[Any, List[AgentVote]]:
        """Plurality voting (most votes wins)."""
        # Simulate votes (in real implementation, query agents)
        votes = [
            AgentVote(
                agent_id=agent_id,
                choice=options[i % len(options)],
                confidence=0.8,
            )
            for i, agent_id in enumerate(agents)
        ]

        # Count votes
        vote_counts: Dict[Any, int] = {}
        for vote in votes:
            choice = vote.choice
            vote_counts[choice] = vote_counts.get(choice, 0) + 1

        winner = max(vote_counts.items(), key=lambda x: x[1])[0] if vote_counts else None

        return winner, votes

    async def _borda_vote(
        self, options: List[Any], agents: List[str]
    ) -> Tuple[Any, List[AgentVote]]:
        """Borda count voting."""
        # In Borda count, voters rank options
        # Points awarded: n-1 for first, n-2 for second, etc.

        scores: Dict[Any, int] = dict.fromkeys(options, 0)

        # Simulate ranked votes
        for agent_id in agents:
            # Shuffle options for each agent
            ranked_options = options.copy()
            # In real implementation, get ranking from agent

            for i, option in enumerate(ranked_options):
                points = len(options) - i - 1
                scores[option] += points

        winner = max(scores.items(), key=lambda x: x[1])[0] if scores else None

        # Create vote objects
        votes = [AgentVote(agent_id=aid, choice=winner, confidence=0.8) for aid in agents]

        return winner, votes

    async def _approval_vote(
        self, options: List[Any], agents: List[str]
    ) -> Tuple[Any, List[AgentVote]]:
        """Approval voting (approve multiple options)."""
        # In approval voting, agents can approve any number of options

        approvals: Dict[Any, int] = dict.fromkeys(options, 0)

        # Simulate approvals
        for agent_id in agents:
            # Each agent approves random subset
            # In real implementation, get approvals from agent
            for option in options[: len(options) // 2 + 1]:
                approvals[option] += 1

        winner = max(approvals.items(), key=lambda x: x[1])[0] if approvals else None

        votes = [AgentVote(agent_id=aid, choice=winner, confidence=0.8) for aid in agents]

        return winner, votes

    async def resolve_conflict(
        self,
        conflict: Dict[str, Any],
        resolution: Optional[ConflictResolution] = None,
    ) -> Any:
        """Resolve conflict between agents.

        Args:
            conflict: Conflict description
            resolution: Resolution strategy

        Returns:
            Resolved decision
        """
        resolution = resolution or self.config.conflict_resolution

        if resolution == ConflictResolution.VOTE:
            # Vote on resolution
            options = conflict.get("options", [])
            agents = conflict.get("agents", [])
            winner, _ = await self._plurality_vote(options, agents)
            return winner

        elif resolution == ConflictResolution.MERGE:
            # Merge conflicting proposals
            proposals = conflict.get("proposals", {})
            return self._merge_proposals(proposals)

        elif resolution == ConflictResolution.COMPROMISE:
            # Find middle ground
            return self._find_compromise(conflict)

        else:
            # Default: return first proposal
            proposals = conflict.get("proposals", {})
            return list(proposals.values())[0] if proposals else None

    def _merge_proposals(self, proposals: Dict[str, Any]) -> Dict[str, Any]:
        """Merge conflicting proposals."""
        merged = {}

        for proposal in proposals.values():
            if isinstance(proposal, dict):
                merged.update(proposal)

        return merged

    def _find_compromise(self, conflict: Dict[str, Any]) -> Any:
        """Find compromise position."""
        options = conflict.get("options", [])
        if not options:
            return None

        # Return middle option
        return options[len(options) // 2]
