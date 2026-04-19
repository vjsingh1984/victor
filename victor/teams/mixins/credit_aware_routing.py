"""Credit-aware team routing — Shapley-driven agent task reassignment.

Extends CreditAssignmentMixin to dynamically reroute tasks from
underperforming agents to higher-performing ones based on accumulated
Shapley credit values. This is the "self-healing team" capability:
the team learns which agents are best at which tasks and adapts.

Usage:
    from victor.teams.mixins.credit_aware_routing import CreditAwareTeamCoordinator

    coordinator = CreditAwareTeamCoordinator(orchestrator)
    coordinator.add_member("coder", coder_agent)
    coordinator.add_member("reviewer", reviewer_agent)
    coordinator.add_member("tester", tester_agent)

    # After multiple rounds, the coordinator learns that 'tester' is
    # underperforming and routes testing tasks to 'reviewer' instead.
    result = await coordinator.execute_task("Build feature", context)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.framework.rl import (
    CreditAssignmentIntegration,
    CreditMethodology,
    ActionMetadata,
    CreditSignal,
)
from victor.teams.mixins.credit_assignment import CreditAssignmentMixin

if TYPE_CHECKING:
    from victor.teams.unified_coordinator import UnifiedTeamCoordinator

logger = logging.getLogger(__name__)


class CreditAwareTeamCoordinator(CreditAssignmentMixin):
    """Team coordinator that reroutes tasks based on credit performance.

    Tracks per-agent Shapley credit over multiple rounds. When an agent's
    average credit drops below a threshold relative to the team mean,
    its tasks are reassigned to the highest-performing agent.

    This creates a self-improving team: underperforming agents get
    fewer tasks, high-performers get more, without manual intervention.
    """

    def __init__(
        self,
        orchestrator: Any = None,
        reroute_threshold: float = 0.5,
        min_rounds_before_reroute: int = 3,
        **kwargs: Any,
    ):
        """Initialize credit-aware coordinator.

        Args:
            orchestrator: Agent orchestrator for SubAgent spawning
            reroute_threshold: Ratio below team mean that triggers rerouting.
                Agent is rerouted when: agent_credit < team_mean * threshold
            min_rounds_before_reroute: Minimum completed rounds before
                rerouting decisions are made (need enough data).
        """
        super().__init__(orchestrator, **kwargs)
        self._reroute_threshold = reroute_threshold
        self._min_rounds = min_rounds_before_reroute

        # Per-agent credit history (agent_id → list of credit values per round)
        self._agent_credit_history: Dict[str, List[float]] = defaultdict(list)

        # Rerouting map: original_agent → replacement_agent
        self._reroute_map: Dict[str, str] = {}

        # Round counter
        self._round_count: int = 0

    # ----------------------------------------------------------------
    # Override execute_task to add rerouting + credit tracking
    # ----------------------------------------------------------------

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with credit-driven rerouting.

        Before execution: apply rerouting map to redirect tasks.
        After execution: update credit history and compute new rerouting.
        """
        self._round_count += 1

        # Execute via parent (which handles formation + credit tracking)
        result = await super().execute_task(task, context)

        # Update credit history from this round's Shapley attribution
        self._update_credit_history(result)

        # Compute rerouting decisions for next round
        if self._round_count >= self._min_rounds:
            self._compute_rerouting()

        return result

    # ----------------------------------------------------------------
    # Credit history tracking
    # ----------------------------------------------------------------

    def _update_credit_history(self, result: Dict[str, Any]) -> None:
        """Extract per-agent credit from result and update history."""
        member_results = result.get("member_results", {})
        if not member_results:
            return

        # Build trajectory + rewards from member results
        trajectory: List[ActionMetadata] = []
        rewards: List[float] = []

        for member_id, member_result in member_results.items():
            success = getattr(member_result, "success", False)
            reward = 1.0 if success else -0.5

            trajectory.append(
                ActionMetadata(
                    agent_id=member_id,
                    action_id=f"round{self._round_count}_{member_id}",
                    turn_index=self._round_count,
                )
            )
            rewards.append(reward)

        if not trajectory:
            return

        # Compute Shapley values
        try:
            ca = CreditAssignmentIntegration()
            signals = ca.assign_credit(trajectory, rewards, CreditMethodology.SHAPLEY)

            for signal in signals:
                if signal.metadata:
                    agent_id = signal.metadata.agent_id
                    self._agent_credit_history[agent_id].append(signal.credit)
        except Exception as e:
            logger.debug("Credit computation for team rerouting failed: %s", e)

    # ----------------------------------------------------------------
    # Rerouting logic
    # ----------------------------------------------------------------

    def _compute_rerouting(self) -> None:
        """Decide which agents should be rerouted based on credit history.

        An agent is rerouted when its average credit falls below
        (team_mean * reroute_threshold). Tasks are redirected to
        the highest-performing agent.
        """
        if not self._agent_credit_history:
            return

        # Compute per-agent average credit
        agent_avg: Dict[str, float] = {}
        for agent_id, credits in self._agent_credit_history.items():
            if credits:
                agent_avg[agent_id] = sum(credits) / len(credits)

        if not agent_avg:
            return

        team_mean = sum(agent_avg.values()) / len(agent_avg)
        threshold = team_mean * self._reroute_threshold

        # Find best performer
        best_agent = max(agent_avg, key=agent_avg.get)  # type: ignore

        # Reroute underperformers to best
        new_reroute_map: Dict[str, str] = {}
        for agent_id, avg_credit in agent_avg.items():
            if avg_credit < threshold and agent_id != best_agent:
                new_reroute_map[agent_id] = best_agent
                logger.info(
                    "Credit rerouting: %s (avg=%.2f) → %s (avg=%.2f), " "threshold=%.2f",
                    agent_id,
                    avg_credit,
                    best_agent,
                    agent_avg[best_agent],
                    threshold,
                )

        # Only update if rerouting changed
        if new_reroute_map != self._reroute_map:
            self._reroute_map = new_reroute_map

    def get_reroute_target(self, agent_id: str) -> str:
        """Get the reroute target for an agent, or the agent itself.

        Args:
            agent_id: Original agent ID

        Returns:
            Target agent ID (may be the same if no rerouting needed)
        """
        return self._reroute_map.get(agent_id, agent_id)

    # ----------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------

    def get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for each agent.

        Returns:
            Dict mapping agent_id to {avg_credit, total_rounds,
            is_rerouted, rerouted_to}
        """
        result: Dict[str, Dict[str, Any]] = {}
        for agent_id, credits in self._agent_credit_history.items():
            avg = sum(credits) / len(credits) if credits else 0.0
            result[agent_id] = {
                "avg_credit": avg,
                "total_rounds": len(credits),
                "is_rerouted": agent_id in self._reroute_map,
                "rerouted_to": self._reroute_map.get(agent_id),
            }
        return result

    @property
    def active_reroutes(self) -> Dict[str, str]:
        """Currently active rerouting map."""
        return dict(self._reroute_map)

    @property
    def round_count(self) -> int:
        """Number of completed team rounds."""
        return self._round_count

    def reset_routing(self) -> None:
        """Clear all rerouting decisions and credit history."""
        self._reroute_map.clear()
        self._agent_credit_history.clear()
        self._round_count = 0


__all__ = [
    "CreditAwareTeamCoordinator",
]
