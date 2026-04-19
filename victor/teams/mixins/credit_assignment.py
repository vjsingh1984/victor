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

"""
Credit Assignment Mixin for Teams.

Provides automatic credit tracking for multi-agent team coordination.
Integrates with UnifiedTeamCoordinator to assign credit across team members.

Usage:
    from victor.teams.mixins.credit_assignment import CreditAssignmentMixin
    from victor.teams import UnifiedTeamCoordinator

    class MyTeamCoordinator(CreditAssignmentMixin, UnifiedTeamCoordinator):
        pass

    coordinator = MyTeamCoordinator(orchestrator)
    result = await coordinator.execute_task("Build feature", context)

    # Get credit attribution
    credit = coordinator.get_team_credit_attribution()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.framework.rl import (
    ActionMetadata,
    CreditAssignmentConfig,
    CreditAssignmentIntegration,
    CreditMethodology,
    CreditSignal,
)

if TYPE_CHECKING:
    from victor.teams.types import TeamResult

logger = logging.getLogger(__name__)


# ============================================================================
# Team Execution Trace
# ============================================================================


@dataclass
class TeamExecutionStep:
    """Single step in team execution for credit tracking."""

    step_id: str
    member_id: str
    action: str
    timestamp: float
    input_context: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    reward: float = 0.0
    duration_ms: int = 0
    error: Optional[str] = None


@dataclass
class TeamExecutionTrace:
    """Complete trace of team execution with credit data."""

    trace_id: str
    team_id: str
    task: str
    start_time: float
    end_time: float
    steps: List[TeamExecutionStep]
    success: bool
    final_result: Optional["TeamResult"] = None
    credit_signals: List[CreditSignal] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def total_reward(self) -> float:
        return sum(step.reward for step in self.steps)

    @property
    def member_count(self) -> int:
        return len(set(step.member_id for step in self.steps))


# ============================================================================
# Credit Assignment Mixin
# ============================================================================


class CreditAssignmentMixin:
    """Mixin for team coordinators to add automatic credit tracking.

    This mixin:
    - Tracks each member's execution steps
    - Extracts rewards from member results
    - Assigns credit using configured methodology
    - Provides attribution queries

    Example:
        class MyCoordinator(CreditAssignmentMixin, UnifiedTeamCoordinator):
            pass

        coordinator = MyCoordinator(orchestrator)
        coordinator.enable_credit_tracking(methodology=CreditMethodology.SHAPLEY)
        result = await coordinator.execute_task("Build feature", context)

        # Get attribution
        attribution = coordinator.get_team_credit_attribution()
        print(f"Member contributions: {attribution}")
    """

    def __init__(self, *args, **kwargs):
        # Initialize mixin state
        self._credit_tracking_enabled = False
        self._credit_trace: Optional[TeamExecutionTrace] = None
        self._credit_config = CreditAssignmentConfig()
        self._credit_methodology = CreditMethodology.GAE
        self._credit_integration: Optional[CreditAssignmentIntegration] = None

        # Call parent __init__ if this is used with multiple inheritance
        super().__init__(*args, **kwargs)

    def enable_credit_tracking(
        self,
        methodology: CreditMethodology = CreditMethodology.SHAPLEY,
        config: Optional[CreditAssignmentConfig] = None,
    ) -> None:
        """Enable credit tracking for team execution.

        Args:
            methodology: Credit assignment methodology (SHAPLEY recommended for teams)
            config: Optional credit configuration
        """
        self._credit_tracking_enabled = True
        self._credit_methodology = methodology
        self._credit_config = config or CreditAssignmentConfig()
        self._credit_integration = CreditAssignmentIntegration(default_config=self._credit_config)
        logger.info(f"Credit tracking enabled with methodology: {methodology.value}")

    def disable_credit_tracking(self) -> None:
        """Disable credit tracking."""
        self._credit_tracking_enabled = False
        logger.info("Credit tracking disabled")

    def _start_credit_trace(self, task: str, team_id: str) -> None:
        """Start a new credit trace.

        Called at the beginning of team execution.
        """
        if not self._credit_tracking_enabled:
            return

        self._credit_trace = TeamExecutionTrace(
            trace_id=f"team_trace_{datetime.now().timestamp()}",
            team_id=team_id,
            task=task,
            start_time=datetime.now().timestamp(),
            end_time=0.0,
            steps=[],
            success=False,
        )
        logger.debug(f"Started credit trace: {self._credit_trace.trace_id}")

    def _record_credit_step(
        self,
        member_id: str,
        action: str,
        input_context: Dict[str, Any],
        output: Any,
        reward: float = 0.0,
        duration_ms: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Record a step in team execution.

        Called after each member completes their action.

        Args:
            member_id: ID of the team member
            action: Action performed
            input_context: Input to the member
            output: Output from the member
            reward: Reward for this step
            duration_ms: Execution duration
            error: Error if execution failed
        """
        if not self._credit_tracking_enabled or self._credit_trace is None:
            return

        step = TeamExecutionStep(
            step_id=f"{member_id}_{action}_{datetime.now().timestamp()}",
            member_id=member_id,
            action=action,
            timestamp=datetime.now().timestamp(),
            input_context=input_context,
            output=output,
            reward=reward,
            duration_ms=duration_ms,
            error=error,
        )

        self._credit_trace.steps.append(step)
        logger.debug(f"Recorded credit step: {step.step_id}")

    def _finalize_credit_trace(
        self,
        success: bool,
        final_result: Optional["TeamResult"] = None,
    ) -> None:
        """Finalize credit trace and assign credit.

        Called at the end of team execution.

        Args:
            success: Whether team execution succeeded
            final_result: Final team result
        """
        if not self._credit_tracking_enabled or self._credit_trace is None:
            return

        self._credit_trace.end_time = datetime.now().timestamp()
        self._credit_trace.success = success
        self._credit_trace.final_result = final_result

        # Assign credit
        self._assign_credit_to_team()

        logger.debug(
            f"Finalized credit trace: {self._credit_trace.trace_id} "
            f"(reward={self._credit_trace.total_reward:.3f})"
        )

    def _assign_credit_to_team(self) -> None:
        """Assign credit to team members based on trace."""
        if self._credit_trace is None or self._credit_integration is None:
            return

        # Build trajectory from steps
        trajectory = []
        rewards = []

        for step in self._credit_trace.steps:
            metadata = ActionMetadata(
                agent_id=step.member_id,
                action_id=step.step_id,
                method_name=step.action,
                timestamp=step.timestamp,
                duration_ms=step.duration_ms,
            )
            trajectory.append(metadata)
            rewards.append(step.reward)

        # Assign credit
        signals = self._credit_integration.assign_credit(
            trajectory,
            rewards,
            methodology=self._credit_methodology,
        )

        self._credit_trace.credit_signals = signals

    def get_team_credit_attribution(self) -> Dict[str, Any]:
        """Get credit attribution for all team members.

        Returns:
            Dictionary with:
            - trace_id: ID of the execution trace
            - total_reward: Total reward for the execution
            - member_attribution: Per-member credit attribution
            - signals: All credit signals
        """
        if self._credit_trace is None:
            return {
                "trace_id": None,
                "total_reward": 0.0,
                "member_attribution": {},
                "signals": [],
            }

        # Group credit by member
        member_attribution: Dict[str, Dict[str, float]] = {}

        for signal in self._credit_trace.credit_signals:
            if signal.metadata:
                member_id = signal.metadata.agent_id
                if member_id not in member_attribution:
                    member_attribution[member_id] = {}

                # Add direct credit
                member_attribution[member_id][member_id] = (
                    member_attribution[member_id].get(member_id, 0.0) + signal.credit
                )

                # Add attribution to other members
                for contributor, amount in signal.attribution.items():
                    member_attribution[member_id][contributor] = (
                        member_attribution[member_id].get(contributor, 0.0) + amount
                    )

        return {
            "trace_id": self._credit_trace.trace_id,
            "team_id": self._credit_trace.team_id,
            "task": self._credit_trace.task,
            "duration": self._credit_trace.duration,
            "total_reward": self._credit_trace.total_reward,
            "success": self._credit_trace.success,
            "member_count": self._credit_trace.member_count,
            "step_count": len(self._credit_trace.steps),
            "member_attribution": member_attribution,
            "signals": [s.to_dict() for s in self._credit_trace.credit_signals],
        }

    def get_member_credit(self, member_id: str) -> Dict[str, float]:
        """Get credit attribution for a specific team member.

        Args:
            member_id: ID of the team member

        Returns:
            Dictionary mapping contributors to credit amounts
        """
        team_attribution = self.get_team_credit_attribution()
        return team_attribution["member_attribution"].get(member_id, {})

    def get_credit_trace(self) -> Optional[TeamExecutionTrace]:
        """Get the complete credit trace.

        Returns:
            TeamExecutionTrace if available, None otherwise
        """
        return self._credit_trace

    def export_credit_report(
        self,
        output_path: str,
        format: str = "html",
    ) -> None:
        """Export credit assignment report.

        Args:
            output_path: Path to save report
            format: Report format (html, md, json)
        """
        from victor.framework.rl import export_credit_report, compute_credit_metrics

        if self._credit_trace is None:
            logger.warning("No credit trace to export")
            return

        attribution = self.get_team_credit_attribution()
        metrics = compute_credit_metrics(self._credit_trace.credit_signals)

        export_credit_report(
            signals=self._credit_trace.credit_signals,
            metrics=metrics,
            attribution=attribution["member_attribution"],
            output_path=output_path,
            format=format,
        )

        logger.info(f"Exported credit report to: {output_path}")


# ============================================================================
# Reward Extraction Helpers
# ============================================================================


def extract_reward_from_member_result(result: Any) -> float:
    """Extract reward from team member result.

    Looks for reward in common locations:
    - result.reward
    - result.score
    - result.success (1.0 if True, 0.0 if False)
    - result.get("reward") if dict
    - 0.0 if not found

    Args:
        result: Member result

    Returns:
        Extracted reward value
    """
    if result is None:
        return 0.0

    if isinstance(result, dict):
        if "reward" in result:
            return float(result["reward"])
        if "score" in result:
            return float(result["score"])
        if "success" in result:
            return 1.0 if result["success"] else 0.0
        return 0.0

    if hasattr(result, "reward"):
        return float(result.reward)
    if hasattr(result, "score"):
        return float(result.score)
    if hasattr(result, "success"):
        return 1.0 if result.success else 0.0

    return 0.0


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    "CreditAssignmentMixin",
    "TeamExecutionStep",
    "TeamExecutionTrace",
    "extract_reward_from_member_result",
]
