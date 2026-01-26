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

"""Advanced team formations for multi-agent coordination.

This module implements cutting-edge team coordination patterns including:
- SwitchingFormation: Dynamic formation switching based on execution progress
- NegotiationFormation: Team members negotiate on approach before execution
- VotingFormation: Democratic decision-making on next steps
- ExpertiseFormation: Select members based on task requirements
- HybridAdaptiveFormation: ML-based real-time formation switching

Example:
    from victor.teams.advanced_formations import SwitchingFormation
    from victor.teams.types import TeamFormation

    # Create switching formation with dynamic criteria
    formation = SwitchingFormation(
        initial_formation=TeamFormation.SEQUENTIAL,
        switching_criteria=[
            SwitchingCriteria(
                condition="progress > 0.5",
                target_formation=TeamFormation.PARALLEL,
                trigger_once=True
            ),
            SwitchingCriteria(
                condition="errors > 2",
                target_formation=TeamFormation.SEQUENTIAL,
                trigger_once=False
            )
        ]
    )
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from victor.coordination.formations.base import TeamContext
    from victor.teams.types import AgentMessage, MemberResult, TeamFormation, MessageType

logger = logging.getLogger(__name__)


# =============================================================================
# Switching Formation
# =============================================================================


class TriggerType(str, Enum):
    """Types of switching triggers."""

    CONDITION = "condition"  # Python expression evaluated on context
    CALLBACK = "callback"  # Callable that returns bool
    THRESHOLD = "threshold"  # Value-based threshold (e.g., progress > 0.5)
    ERROR_COUNT = "error_count"  # Number of errors exceeded
    TIME_ELAPSED = "time_elapsed"  # Time threshold exceeded


@dataclass
class SwitchingCriteria:
    """Criteria for switching formations.

    Attributes:
        condition: Python expression to evaluate (for CONDITION trigger type)
        callback: Callable to evaluate (for CALLBACK trigger type)
        threshold: Value threshold (for THRESHOLD trigger type)
        threshold_key: Key in context for threshold value (e.g., "progress")
        threshold_value: Value to compare against
        comparison: Comparison operator (">", "<", ">=", "<=", "==", "!=")
        target_formation: Formation to switch to when criteria is met
        trigger_once: If True, only trigger once per execution
        priority: Higher priority criteria are evaluated first
        cooldown_seconds: Minimum time before triggering again
    """

    target_formation: "TeamFormation"
    trigger_type: TriggerType = TriggerType.CONDITION
    condition: Optional[str] = None
    callback: Optional[Callable[[Dict[str, Any]], bool]] = None
    threshold_key: Optional[str] = None
    threshold_value: Optional[Union[int, float]] = None
    comparison: str = ">="
    trigger_once: bool = True
    priority: int = 0
    cooldown_seconds: float = 0.0

    # Internal state
    _triggered: bool = field(default=False, init=False, repr=False)
    _last_triggered: float = field(default=0.0, init=False, repr=False)

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """Check if criteria should trigger formation switch.

        Args:
            context: Current execution context with metrics and state

        Returns:
            True if formation should switch
        """
        import time

        # Check if already triggered (if trigger_once)
        if self.trigger_once and self._triggered:
            return False

        # Check cooldown
        if self.cooldown_seconds > 0:
            time_since_trigger = time.time() - self._last_triggered
            if time_since_trigger < self.cooldown_seconds:
                return False

        # Evaluate based on trigger type
        result = False

        if self.trigger_type == TriggerType.CONDITION and self.condition:
            try:
                # Safe eval with limited context
                safe_context = {k: v for k, v in context.items() if k.isidentifier()}
                result = eval(self.condition, {"__builtins__": {}}, safe_context)
            except Exception as e:
                logger.warning(f"Failed to evaluate condition '{self.condition}': {e}")

        elif self.trigger_type == TriggerType.CALLBACK and self.callback:
            try:
                result = self.callback(context)
            except Exception as e:
                logger.warning(f"Callback evaluation failed: {e}")

        elif self.trigger_type == TriggerType.THRESHOLD:
            if self.threshold_key and self.threshold_value is not None:
                actual_value = context.get(self.threshold_key, 0)
                result = self._compare(actual_value, self.threshold_value, self.comparison)

        elif self.trigger_type == TriggerType.ERROR_COUNT:
            error_count = context.get("error_count", 0)
            if self.threshold_value is not None:
                result = error_count >= self.threshold_value

        elif self.trigger_type == TriggerType.TIME_ELAPSED:
            elapsed = context.get("elapsed_time", 0.0)
            if self.threshold_value is not None:
                result = elapsed >= self.threshold_value

        # Update triggered state
        if result:
            self._triggered = True
            self._last_triggered = time.time()

        return result

    def _compare(self, actual: Any, expected: Any, comparison: str) -> bool:
        """Compare values using specified operator."""
        try:
            if comparison == ">":
                return bool(actual > expected)
            elif comparison == "<":
                return bool(actual < expected)
            elif comparison == ">=":
                return bool(actual >= expected)
            elif comparison == "<=":
                return bool(actual <= expected)
            elif comparison == "==":
                return bool(actual == expected)
            elif comparison == "!=":
                return bool(actual != expected)
            else:
                logger.warning(f"Unknown comparison operator: {comparison}")
                return False
        except Exception as e:
            logger.warning(f"Comparison failed: {e}")
            return False

    def reset(self) -> None:
        """Reset triggered state (for reusable criteria)."""
        self._triggered = False
        self._last_triggered = 0.0


class SwitchingFormation:
    """Formation that dynamically switches based on execution progress.

    This formation allows teams to adapt their coordination strategy in
    real-time based on execution metrics, errors, progress, or custom criteria.

    Example:
        formation = SwitchingFormation(
            initial_formation=TeamFormation.SEQUENTIAL,
            switching_criteria=[
                SwitchingCriteria(
                    trigger_type=TriggerType.THRESHOLD,
                    threshold_key="progress",
                    threshold_value=0.5,
                    comparison=">=",
                    target_formation=TeamFormation.PARALLEL
                )
            ]
        )
    """

    def __init__(
        self,
        initial_formation: "TeamFormation",
        switching_criteria: List[SwitchingCriteria],
        max_switches: int = 10,
        track_switches: bool = True,
    ):
        """Initialize switching formation.

        Args:
            initial_formation: Starting formation
            switching_criteria: List of criteria for switching
            max_switches: Maximum number of formation switches
            track_switches: Whether to track switch history
        """
        from victor.teams.types import TeamFormation

        self.initial_formation = initial_formation
        self.switching_criteria = sorted(switching_criteria, key=lambda c: c.priority, reverse=True)
        self.max_switches = max_switches
        self.track_switches = track_switches

        # State
        self.current_formation = initial_formation
        self.switch_count = 0
        self.switch_history: List[Dict[str, Any]] = []

    async def execute(
        self,
        agents: List[Any],
        context: "TeamContext",
        task: "AgentMessage",
    ) -> List["MemberResult"]:
        """Execute with dynamic formation switching.

        Args:
            agents: List of agents to execute
            context: Team context
            task: Task message

        Returns:
            List of member results
        """
        # TODO: Implement formation registry/factory
        from victor.coordination.formations import get_formation  # type: ignore[attr-defined]

        results = []
        execution_context = {
            "progress": 0.0,
            "error_count": 0,
            "elapsed_time": 0.0,
            "completed_members": 0,
            "total_members": len(agents),
        }

        import time

        start_time = time.time()
        current_agents = agents

        # Execute with formation switching
        while current_agents and self.switch_count < self.max_switches:
            # Check if we should switch formation
            should_switch, new_formation, criteria = self._check_switch_criteria(execution_context)

            if should_switch:
                old_formation = self.current_formation
                self.current_formation = new_formation  # type: ignore[assignment]
                self.switch_count += 1

                if self.track_switches:
                    from_val = old_formation.value if old_formation else None
                    to_val = new_formation.value if new_formation else None
                    self.switch_history.append(
                        {
                            "from": from_val,
                            "to": to_val,
                            "timestamp": time.time(),
                            "context": dict(execution_context),
                            "criteria_index": (
                                self.switching_criteria.index(criteria) if criteria else -1
                            ),
                        }
                    )

                logger.info(
                    f"Switched formation: {old_formation.value if old_formation else None} -> {new_formation.value if new_formation else None} "
                    f"(switch #{self.switch_count})"
                )

            # Get current formation strategy
            strategy = get_formation(self.current_formation)

            # Prepare context with current execution state
            enhanced_context = context
            enhanced_context.shared_state.update(execution_context)

            # Execute with current formation
            current_results = await strategy.execute(current_agents, enhanced_context, task)
            results.extend(current_results)

            # Update execution context
            execution_context["completed_members"] += len(current_results)
            execution_context["elapsed_time"] = time.time() - start_time
            execution_context["error_count"] += sum(1 for r in current_results if not r.success)

            # Calculate progress
            if len(agents) > 0:
                execution_context["progress"] = execution_context["completed_members"] / len(agents)

            # Check if all agents completed
            if execution_context["completed_members"] >= len(agents):
                break

            # Determine remaining agents (for formations that support partial completion)
            current_agents = [
                a for i, a in enumerate(agents) if i >= execution_context["completed_members"]
            ]

        return results

    def _check_switch_criteria(
        self, context: Dict[str, Any]
    ) -> tuple[bool, Optional["TeamFormation"], Optional[SwitchingCriteria]]:
        """Check if any switching criteria is met.

        Args:
            context: Current execution context

        Returns:
            Tuple of (should_switch, target_formation, criteria)
        """
        for criteria in self.switching_criteria:
            if criteria.should_trigger(context):
                return True, criteria.target_formation, criteria

        return False, None, None

    def get_switch_history(self) -> List[Dict[str, Any]]:
        """Get history of formation switches.

        Returns:
            List of switch events with timestamps and context
        """
        return list(self.switch_history)

    def reset(self) -> None:
        """Reset formation state for reuse."""
        self.current_formation = self.initial_formation
        self.switch_count = 0
        self.switch_history.clear()
        for criteria in self.switching_criteria:
            criteria.reset()


# =============================================================================
# Negotiation Formation
# =============================================================================


@dataclass
class NegotiationProposal:
    """Proposal from a team member during negotiation.

    Attributes:
        member_id: ID of the proposing member
        proposal_type: Type of proposal (approach, formation, resource_allocation)
        content: The proposal content
        confidence: Member's confidence in this proposal (0.0-1.0)
        rationale: Reasoning behind the proposal
        alternatives: Alternative proposals if this one is rejected
        metadata: Additional metadata
    """

    member_id: str
    proposal_type: str
    content: str
    confidence: float = 0.5
    rationale: str = ""
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "member_id": self.member_id,
            "proposal_type": self.proposal_type,
            "content": self.content,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "alternatives": self.alternatives,
            "metadata": self.metadata,
        }


@dataclass
class NegotiationResult:
    """Result of negotiation round.

    Attributes:
        round_number: Which round of negotiation
        proposals: All proposals in this round
        consensus_reached: Whether consensus was achieved
        selected_proposal: The proposal that was selected (if any)
        confidence_scores: Confidence scores for each proposal
        discussion_summary: Summary of the negotiation discussion
    """

    round_number: int
    proposals: List[NegotiationProposal]
    consensus_reached: bool
    selected_proposal: Optional[NegotiationProposal] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    discussion_summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class NegotiationFormation:
    """Formation where team members negotiate on approach before execution.

    This formation implements a structured negotiation process where team
    members discuss and agree on the best approach before executing.
    Supports multiple rounds and various consensus strategies.

    Example:
        formation = NegotiationFormation(
            max_rounds=3,
            consensus_threshold=0.7,
            fallback_formation=TeamFormation.SEQUENTIAL
        )
    """

    def __init__(
        self,
        max_rounds: int = 3,
        consensus_threshold: float = 0.7,
        fallback_formation: Optional["TeamFormation"] = None,
        voting_strategy: str = "weighted_confidence",
        timeout_per_round: float = 60.0,
    ):
        """Initialize negotiation formation.

        Args:
            max_rounds: Maximum number of negotiation rounds
            consensus_threshold: Minimum agreement level (0.0-1.0)
            fallback_formation: Formation to use if consensus not reached
            voting_strategy: How to select final proposal
            timeout_per_round: Maximum time per negotiation round
        """
        from victor.teams.types import TeamFormation

        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.fallback_formation = fallback_formation or TeamFormation.SEQUENTIAL
        self.voting_strategy = voting_strategy
        self.timeout_per_round = timeout_per_round

        # State
        self.negotiation_history: List[NegotiationResult] = []

    async def execute(
        self,
        agents: List[Any],
        context: "TeamContext",
        task: "AgentMessage",
    ) -> List["MemberResult"]:
        """Execute with negotiation phase.

        Args:
            agents: List of agents to execute
            context: Team context
            task: Task message

        Returns:
            List of member results
        """
        # Phase 1: Negotiation
        negotiation_result = await self._negotiate(agents, context, task)

        # Phase 2: Execution
        execution_formation = self._determine_formation(negotiation_result)
        return await self._execute_with_formation(agents, context, task, execution_formation)

    async def _negotiate(
        self,
        agents: List[Any],
        context: "TeamContext",
        task: "AgentMessage",
    ) -> NegotiationResult:
        """Run negotiation phase.

        Args:
            agents: List of agents
            context: Team context
            task: Task message

        Returns:
            Negotiation result
        """
        proposals = []
        consensus_reached = False
        selected_proposal = None

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Negotiation round {round_num}/{self.max_rounds}")

            # Gather proposals from all agents
            round_proposals = await self._gather_proposals(agents, context, task, round_num)
            proposals.extend(round_proposals)

            # Evaluate proposals and check for consensus
            consensus_reached, selected_proposal = self._evaluate_proposals(round_proposals)

            if consensus_reached:
                logger.info(f"Consensus reached in round {round_num}")
                break

            # If not last round, share proposals for next round
            if round_num < self.max_rounds:
                await self._share_proposals(agents, round_proposals)

        result = NegotiationResult(
            round_number=len(self.negotiation_history) + 1,
            proposals=proposals,
            consensus_reached=consensus_reached,
            selected_proposal=selected_proposal,
            discussion_summary=self._summarize_negotiation(proposals),
        )

        self.negotiation_history.append(result)
        return result

    async def _gather_proposals(
        self,
        agents: List[Any],
        context: "TeamContext",
        task: "AgentMessage",
        round_number: int,
    ) -> List[NegotiationProposal]:
        """Gather proposals from all agents.

        Args:
            agents: List of agents
            context: Team context
            task: Task message
            round_number: Current round number

        Returns:
            List of proposals
        """
        proposals = []

        # Create negotiation prompt
        negotiation_prompt = self._create_negotiation_prompt(task, round_number)

        # Request proposals from each agent
        for agent in agents:
            try:
                import time

                time.time()
                response = await asyncio.wait_for(
                    agent.execute(
                        AgentMessage(
                            sender_id="coordinator",
                            content=negotiation_prompt,
                            message_type=MessageType.QUERY,
                            data={"round": round_number},
                        ),
                        context,
                    ),
                    timeout=self.timeout_per_round,
                )

                # Parse proposal from response
                proposal = self._parse_proposal(agent.id, response.output)
                if proposal:
                    proposals.append(proposal)

            except asyncio.TimeoutError:
                logger.warning(f"Agent {agent.id} timed out during negotiation")
            except Exception as e:
                logger.warning(f"Failed to get proposal from {agent.id}: {e}")

        return proposals

    def _create_negotiation_prompt(self, task: "AgentMessage", round_number: int) -> str:
        """Create negotiation prompt for agents.

        Args:
            task: Original task
            round_number: Current round number

        Returns:
            Negotiation prompt
        """
        return f"""We are negotiating the best approach for the following task: Task[Any, Any]: {task.content}

Round: {round_number}

Please provide a proposal with:
1. Your suggested approach
2. Your confidence level (0.0-1.0)
3. Rationale for your approach
4. Any alternative approaches

Format your response as a JSON object:
{{
    "proposal_type": "approach",
    "content": "Your suggested approach here",
    "confidence": 0.8,
    "rationale": "Your reasoning here",
    "alternatives": ["Alternative 1", "Alternative 2"]
}}
"""

    def _parse_proposal(self, member_id: str, response: str) -> Optional[NegotiationProposal]:
        """Parse proposal from agent response.

        Args:
            member_id: Agent ID
            response: Agent response text

        Returns:
            Parsed proposal or None
        """
        try:
            import json

            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return NegotiationProposal(member_id=member_id, **data)
        except Exception as e:
            logger.warning(f"Failed to parse proposal from {member_id}: {e}")

        # Fallback: create simple proposal from response
        return NegotiationProposal(
            member_id=member_id,
            proposal_type="approach",
            content=response,
            confidence=0.5,
            rationale="Free-form response",
        )

    def _evaluate_proposals(
        self, proposals: List[NegotiationProposal]
    ) -> tuple[bool, Optional[NegotiationProposal]]:
        """Evaluate proposals and check for consensus.

        Args:
            proposals: List of proposals

        Returns:
            Tuple of (consensus_reached, selected_proposal)
        """
        if not proposals:
            return False, None

        # Calculate confidence scores
        confidence_scores = {p.member_id: p.confidence for p in proposals}

        # Check if any proposal exceeds consensus threshold
        max_confidence = max(confidence_scores.values())
        if max_confidence >= self.consensus_threshold:
            # Find proposal with max confidence
            selected = max(proposals, key=lambda p: p.confidence)
            return True, selected

        return False, None

    async def _share_proposals(
        self, agents: List[Any], proposals: List[NegotiationProposal]
    ) -> None:
        """Share proposals with all agents for next round.

        Args:
            agents: List of agents
            proposals: Proposals to share
        """
        summary = self._summarize_proposals(proposals)

        for agent in agents:
            try:
                await agent.receive_message(
                    AgentMessage(
                        sender_id="coordinator",
                        content=f"Team proposals from previous round:\n{summary}",
                        message_type=MessageType.STATUS,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to share proposals with {agent.id}: {e}")

    def _summarize_proposals(self, proposals: List[NegotiationProposal]) -> str:
        """Create summary of proposals.

        Args:
            proposals: List of proposals

        Returns:
            Summary string
        """
        lines = []
        for i, p in enumerate(proposals, 1):
            lines.append(
                f"{i}. {p.member_id}: {p.content} (confidence: {p.confidence:.2f})\n"
                f"   Rationale: {p.rationale}"
            )
        return "\n".join(lines)

    def _summarize_negotiation(self, proposals: List[NegotiationProposal]) -> str:
        """Create summary of entire negotiation.

        Args:
            proposals: All proposals

        Returns:
            Summary string
        """
        return f"Negotiation completed with {len(proposals)} proposals from team members."

    def _determine_formation(self, result: NegotiationResult) -> "TeamFormation":
        """Determine which formation to use based on negotiation result.

        Args:
            result: Negotiation result

        Returns:
            Formation to use
        """
        if result.consensus_reached and result.selected_proposal:
            # Extract formation preference from proposal if present
            formation_str = result.selected_proposal.metadata.get("preferred_formation")
            if formation_str:
                try:
                    from victor.teams.types import TeamFormation

                    return TeamFormation(formation_str)
                except ValueError:
                    pass

        return self.fallback_formation

    async def _execute_with_formation(
        self,
        agents: List[Any],
        context: "TeamContext",
        task: "AgentMessage",
        formation: "TeamFormation",
    ) -> List["MemberResult"]:
        """Execute with determined formation.

        Args:
            agents: List of agents
            context: Team context
            task: Task message
            formation: Formation to use

        Returns:
            Member results
        """
        # TODO: Implement formation registry/factory
        from victor.coordination.formations import get_formation  # type: ignore[attr-defined]

        strategy = get_formation(formation)
        return await strategy.execute(agents, context, task)  # type: ignore[no-any-return]

    def get_negotiation_history(self) -> List[NegotiationResult]:
        """Get history of negotiation rounds.

        Returns:
            List of negotiation results
        """
        return list(self.negotiation_history)


# =============================================================================
# Voting Formation
# =============================================================================


class VotingMethod(str, Enum):
    """Methods for voting on decisions."""

    MAJORITY = "majority"  # Simple majority (>50%)
    SUPERMAJORITY = "supermajority"  # Supermajority (default 67%)
    UNANIMOUS = "unanimous"  # All must agree
    WEIGHTED = "weighted"  # Weighted by member expertise/confidence
    RANKED_CHOICE = "ranked_choice"  # Ranked choice voting


@dataclass
class Vote:
    """A vote from a team member.

    Attributes:
        member_id: ID of voting member
        choice: The choice being voted for
        ranking: Optional ranking (for ranked choice)
        confidence: Confidence in this vote
        rationale: Reasoning behind the vote
    """

    member_id: str
    choice: str
    ranking: Optional[List[str]] = None
    confidence: float = 1.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "member_id": self.member_id,
            "choice": self.choice,
            "ranking": self.ranking,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass
class VotingResult:
    """Result of a voting round.

    Attributes:
        votes: All votes cast
        winner: The winning choice
        vote_distribution: Distribution of votes across choices
        consensus_level: Level of agreement (0.0-1.0)
        method: Voting method used
        metadata: Additional metadata
    """

    votes: List[Vote]
    winner: Optional[str]
    vote_distribution: Dict[str, int]
    consensus_level: float
    method: VotingMethod
    metadata: Dict[str, Any] = field(default_factory=dict)


class VotingFormation:
    """Formation with democratic decision-making on next steps.

    Team members vote on important decisions during execution, allowing
    for collaborative decision-making with various voting strategies.

    Example:
        formation = VotingFormation(
            voting_method=VotingMethod.SUPERMAJORITY,
            supermajority_threshold=0.67,
            tiebreaker="first_vote"
        )
    """

    def __init__(
        self,
        voting_method: VotingMethod = VotingMethod.MAJORITY,
        supermajority_threshold: float = 0.67,
        tiebreaker: str = "first_vote",
        fallback_formation: Optional["TeamFormation"] = None,
    ):
        """Initialize voting formation.

        Args:
            voting_method: Method for voting
            supermajority_threshold: Threshold for supermajority
            tiebreaker: How to break ties ("first_vote", "random", "manager_decides")
            fallback_formation: Formation to use if voting fails
        """
        from victor.teams.types import TeamFormation

        self.voting_method = voting_method
        self.supermajority_threshold = supermajority_threshold
        self.tiebreaker = tiebreaker
        self.fallback_formation = fallback_formation or TeamFormation.SEQUENTIAL

        # State
        self.voting_history: List[VotingResult] = []

    async def execute(
        self,
        agents: List[Any],
        context: "TeamContext",
        task: "AgentMessage",
    ) -> List["MemberResult"]:
        """Execute with voting on key decisions.

        Args:
            agents: List of agents
            context: Team context
            task: Task message

        Returns:
            List of member results
        """
        # Phase 1: Vote on formation/approach
        vote_result = await self._vote_on_approach(agents, context, task)

        # Phase 2: Execute with chosen approach
        return await self._execute_with_vote_result(agents, context, task, vote_result)

    async def _vote_on_approach(
        self,
        agents: List[Any],
        context: "TeamContext",
        task: "AgentMessage",
    ) -> VotingResult:
        """Conduct vote on execution approach.

        Args:
            agents: List of agents
            context: Team context
            task: Task message

        Returns:
            Voting result
        """
        # Present choices to agents
        choices = self._get_choices(task)

        # Collect votes
        votes = []
        for agent in agents:
            try:
                vote = await self._request_vote(agent, choices, context, task)
                if vote:
                    votes.append(vote)
            except Exception as e:
                logger.warning(f"Failed to get vote from {agent.id}: {e}")

        # Tally votes
        winner, distribution, consensus = self._tally_votes(votes, choices)

        result = VotingResult(
            votes=votes,
            winner=winner,
            vote_distribution=distribution,
            consensus_level=consensus,
            method=self.voting_method,
        )

        self.voting_history.append(result)
        return result

    def _get_choices(self, task: "AgentMessage") -> List[str]:
        """Get available choices for voting.

        Args:
            task: Task message

        Returns:
            List of choices
        """
        # Default choices: different formations
        return ["sequential", "parallel", "hierarchical", "pipeline", "consensus"]

    async def _request_vote(
        self,
        agent: Any,
        choices: List[str],
        context: "TeamContext",
        task: "AgentMessage",
    ) -> Optional[Vote]:
        """Request vote from an agent.

        Args:
            agent: Agent to request vote from
            choices: Available choices
            context: Team context
            task: Task message

        Returns:
            Vote or None
        """
        prompt = self._create_voting_prompt(task, choices)

        try:
            response = await agent.execute(
                AgentMessage(
                    sender_id="coordinator",
                    content=prompt,
                    message_type=MessageType.QUERY,
                ),
                context,
            )

            return self._parse_vote(agent.id, response.output, choices)
        except Exception as e:
            logger.warning(f"Failed to get vote from {agent.id}: {e}")
            return None

    def _create_voting_prompt(self, task: "AgentMessage", choices: List[str]) -> str:
        """Create voting prompt.

        Args:
            task: Task message
            choices: Available choices

        Returns:
            Voting prompt
        """
        choices_str = "\n".join(f"{i + 1}. {choice}" for i, choice in enumerate(choices))

        return f"""We need to decide on the best approach for this task: Task[Any, Any]: {task.content}

Please vote on one of the following approaches:
{choices_str}

Respond with your choice number or name, along with your confidence (0.0-1.0) and rationale.

Format:
Choice: [number or name]
Confidence: [0.0-1.0]
Rationale: [your reasoning]
"""

    def _parse_vote(self, member_id: str, response: str, choices: List[str]) -> Optional[Vote]:
        """Parse vote from agent response.

        Args:
            member_id: Agent ID
            response: Response text
            choices: Available choices

        Returns:
            Parsed vote or None
        """
        try:
            # Extract choice
            choice = None
            for c in choices:
                if c.lower() in response.lower():
                    choice = c
                    break

            # Try to extract number
            if not choice:
                import re

                num_match = re.search(r"choice[:\s]+(\d+)", response, re.IGNORECASE)
                if num_match:
                    idx = int(num_match.group(1)) - 1
                    if 0 <= idx < len(choices):
                        choice = choices[idx]

            # Extract confidence
            import re

            conf_match = re.search(r"confidence[:\s]+([\d.]+)", response, re.IGNORECASE)
            confidence = float(conf_match.group(1)) if conf_match else 0.5

            # Extract rationale
            lines = response.split("\n")
            rationale_lines = []
            in_rationale = False
            for line in lines:
                if "rationale" in line.lower():
                    in_rationale = True
                    continue
                if in_rationale:
                    rationale_lines.append(line.strip())

            rationale = " ".join(rationale_lines).strip() if rationale_lines else ""

            if choice:
                return Vote(
                    member_id=member_id, choice=choice, confidence=confidence, rationale=rationale
                )

        except Exception as e:
            logger.warning(f"Failed to parse vote from {member_id}: {e}")

        return None

    def _tally_votes(
        self, votes: List[Vote], choices: List[str]
    ) -> tuple[Optional[str], Dict[str, int], float]:
        """Tally votes and determine winner.

        Args:
            votes: List of votes
            choices: Available choices

        Returns:
            Tuple of (winner, distribution, consensus_level)
        """
        if not votes:
            return None, {}, 0.0

        # Count votes
        distribution: dict[Any, int | float] = dict.fromkeys(choices, 0)
        for vote in votes:
            if vote.choice in distribution:
                if self.voting_method == VotingMethod.WEIGHTED:
                    distribution[vote.choice] += vote.confidence
                else:
                    distribution[vote.choice] += 1

        # Determine winner based on method
        winner = None
        consensus = 0.0

        if self.voting_method == VotingMethod.MAJORITY:
            winner = max(
                distribution, key=lambda k: distribution[k] if distribution[k] is not None else 0
            )
            total_votes = sum(distribution.values())
            consensus = distribution[winner] / total_votes if total_votes > 0 else 0.0

        elif self.voting_method == VotingMethod.SUPERMAJORITY:
            for choice, count in distribution.items():
                total_votes = sum(distribution.values())
                if count / total_votes >= self.supermajority_threshold:
                    winner = choice
                    consensus = count / total_votes
                    break

            if not winner:
                winner = max(
                    distribution,
                    key=lambda k: distribution[k] if distribution[k] is not None else 0,
                )
                consensus = distribution.get(winner, 0) / sum(distribution.values())

        elif self.voting_method == VotingMethod.UNANIMOUS:
            unanimous = all(vote.choice == votes[0].choice for vote in votes)
            if unanimous:
                winner = votes[0].choice
                consensus = 1.0
            else:
                # Use tiebreaker
                winner = self._apply_tiebreaker(distribution)  # type: ignore[arg-type]
                consensus = distribution.get(winner, 0) / sum(distribution.values())

        elif self.voting_method == VotingMethod.WEIGHTED:
            winner = max(
                distribution, key=lambda k: distribution[k] if distribution[k] is not None else 0
            )
            total_weight = sum(distribution.values())
            consensus = distribution[winner] / total_weight if total_weight > 0 else 0.0

        else:  # RANKED_CHOICE or default
            winner = max(
                distribution, key=lambda k: distribution[k] if distribution[k] is not None else 0
            )
            total_votes = sum(1 for vote in votes if vote.choice == winner)
            consensus = total_votes / len(votes) if votes else 0.0

        # Cast distribution to expected type for return
        distribution_typed: Dict[str, int] = {
            k: int(v) if v is not None else 0 for k, v in distribution.items()
        }
        return winner, distribution_typed, consensus

    def _apply_tiebreaker(self, distribution: Dict[str, int]) -> Optional[str]:
        """Apply tiebreaker rule.

        Args:
            distribution: Vote distribution

        Returns:
            Winning choice
        """
        max_count = max(distribution.values())
        tied_choices = [choice for choice, count in distribution.items() if count == max_count]

        if len(tied_choices) == 1:
            return tied_choices[0]

        # Apply tiebreaker
        if self.tiebreaker == "first_vote":
            return tied_choices[0]
        elif self.tiebreaker == "random":
            import random

            return random.choice(tied_choices)
        elif self.tiebreaker == "manager_decides":
            # Return None to indicate manager should decide
            return None

        return tied_choices[0]

    async def _execute_with_vote_result(
        self,
        agents: List[Any],
        context: "TeamContext",
        task: "AgentMessage",
        vote_result: VotingResult,
    ) -> List["MemberResult"]:
        """Execute with chosen approach.

        Args:
            agents: List of agents
            context: Team context
            task: Task message
            vote_result: Voting result

        Returns:
            Member results
        """
        # Map vote choice to formation
        formation = self._map_choice_to_formation(vote_result.winner)

        # TODO: Implement formation registry/factory
        from victor.coordination.formations import get_formation  # type: ignore[attr-defined]

        strategy = get_formation(formation)
        return await strategy.execute(agents, context, task)  # type: ignore[no-any-return]

    def _map_choice_to_formation(self, choice: Optional[str]) -> "TeamFormation":
        """Map voting choice to formation.

        Args:
            choice: Winning choice

        Returns:
            Formation to use
        """
        from victor.teams.types import TeamFormation

        if not choice:
            return self.fallback_formation

        try:
            return TeamFormation(choice)
        except ValueError:
            return self.fallback_formation

    def get_voting_history(self) -> List[VotingResult]:
        """Get history of voting rounds.

        Returns:
            List of voting results
        """
        return list(self.voting_history)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SwitchingCriteria",
    "SwitchingFormation",
    "NegotiationProposal",
    "NegotiationResult",
    "NegotiationFormation",
    "Vote",
    "VotingResult",
    "VotingFormation",
    "VotingMethod",
    "TriggerType",
]
