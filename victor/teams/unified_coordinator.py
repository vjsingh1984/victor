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

"""Unified team coordinator implementing ITeamCoordinator.

This is the production-ready coordinator that combines:
- ITeamCoordinator protocol compliance (LSP)
- All 5 formations including CONSENSUS
- EventBus observability via ObservabilityMixin
- RL integration via RLMixin
- Message bus and shared memory support
- Recursion depth tracking for nested team/workflow execution

Example:
    from victor.teams import UnifiedTeamCoordinator, TeamFormation

    coordinator = UnifiedTeamCoordinator(orchestrator)
    coordinator.set_execution_context(task_type="feature", complexity="high")

    coordinator.add_member(researcher).add_member(executor)
    coordinator.set_formation(TeamFormation.PIPELINE)

    result = await coordinator.execute_task("Implement authentication", {})

    # Check recursion depth for nested execution
    depth = coordinator.get_recursion_depth()
    can_spawn = coordinator.can_spawn_nested()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.mixins.observability import ObservabilityMixin
from victor.teams.mixins.rl import RLMixin
from victor.teams.types import (
    AgentMessage,
    MemberResult,
    MessageType,
    TeamFormation,
    TeamResult,
)
from victor.workflows.recursion import RecursionContext, RecursionGuard

if TYPE_CHECKING:
    from victor.coordination.formations import (
        ConsensusFormation,
        HierarchicalFormation,
        ParallelFormation,
        PipelineFormation,
        SequentialFormation,
    )
    from victor.teams.protocols import ITeamMember

logger = logging.getLogger(__name__)


# =============================================================================
# Team Member Adapter
# =============================================================================


class _TeamMemberAdapter:
    """Adapter to bridge ITeamMember to formation strategy agent interface.

    Formation strategies expect agents with execute(task, context) -> MemberResult
    but ITeamMember uses execute_task(task, context) -> str.
    """

    def __init__(self, member: "ITeamMember", coordinator_context: Dict[str, Any]):
        self._member = member
        self._context = coordinator_context
        self.id = member.id

    @property
    def role(self) -> Optional[str]:
        """Expose role from underlying member for formation strategy detection."""
        return getattr(self._member, "role", None)

    async def execute(self, task: AgentMessage, context: TeamContext) -> MemberResult:
        """Execute task using ITeamMember interface."""
        import time

        start_time = time.time()

        try:
            # Merge TeamContext.shared_state into coordinator context
            merged_context = {**self._context, **context.shared_state}

            # Call ITeamMember's execute_task
            output = await self._member.execute_task(task.content, merged_context)

            duration = time.time() - start_time

            return MemberResult(
                member_id=self._member.id,
                success=True,
                output=output,
                duration_seconds=duration,
                metadata={"task": task.content},
            )
        except Exception as e:
            duration = time.time() - start_time
            return MemberResult(
                member_id=self._member.id,
                success=False,
                output="",
                error=str(e),
                duration_seconds=duration,
                metadata={"task": task.content},
            )


class UnifiedTeamCoordinator(ObservabilityMixin, RLMixin):
    """Production-ready team coordinator implementing ITeamCoordinator.

    This coordinator unifies the framework and agent-layer implementations,
    providing full protocol compliance with all production features.

    Features:
        - ITeamCoordinator protocol compliance
        - 5 formation patterns (SEQUENTIAL, PARALLEL, HIERARCHICAL, PIPELINE, CONSENSUS)
        - EventBus observability integration
        - RL integration for team composition learning
        - Message bus for inter-agent communication
        - Shared memory for team context
        - Recursion depth tracking for nested team/workflow execution

    Attributes:
        _orchestrator: Agent orchestrator (optional, for SubAgent spawning)
        _members: List of team members
        _formation: Current team formation
        _manager: Manager member for HIERARCHICAL formation
        _message_history: Log of inter-agent messages
        _shared_context: Shared context dictionary
        _recursion_ctx: Recursion context for tracking nested execution
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        *,
        enable_observability: bool = True,
        enable_rl: bool = True,
        lightweight_mode: bool = False,
        recursion_context: Optional[RecursionContext] = None,
    ) -> None:
        """Initialize the unified coordinator.

        Args:
            orchestrator: Optional agent orchestrator for SubAgent spawning
            enable_observability: Enable EventBus integration
            enable_rl: Enable RL integration
            lightweight_mode: If True, disable mixins (for testing without dependencies)
            recursion_context: Optional recursion context for tracking nested team execution
        """
        # Initialize mixins conditionally based on lightweight_mode
        if not lightweight_mode:
            ObservabilityMixin.__init__(self, enable_observability=enable_observability)
            RLMixin.__init__(self, enable_rl=enable_rl)
            self._enable_observability = enable_observability
            self._enable_rl = enable_rl
        else:
            # In lightweight mode, skip mixin initialization
            self._enable_observability = False
            self._enable_rl = False

        # Core state
        self._orchestrator = orchestrator
        self._members: List[ITeamMember] = []
        self._formation = TeamFormation.SEQUENTIAL
        self._manager: Optional[ITeamMember] = None
        self._lightweight_mode = lightweight_mode

        # Communication
        self._message_history: List[AgentMessage] = []
        self._shared_context: Dict[str, Any] = {}

        # Formation strategies (lazy loaded to avoid circular imports)
        self._formations: Optional[Dict[TeamFormation, BaseFormationStrategy]] = None

        # Recursion tracking
        self._recursion_ctx = recursion_context or RecursionContext()

    # =========================================================================
    # ITeamCoordinator Protocol Methods
    # =========================================================================

    def add_member(self, member: "ITeamMember") -> "UnifiedTeamCoordinator":
        """Add a member to the team.

        Args:
            member: Team member implementing ITeamMember protocol

        Returns:
            Self for fluent chaining
        """
        self._members.append(member)
        return self

    def set_formation(self, formation: TeamFormation) -> "UnifiedTeamCoordinator":
        """Set the team formation pattern.

        Args:
            formation: Formation to use

        Returns:
            Self for fluent chaining
        """
        self._formation = formation
        return self

    def set_manager(self, manager: "ITeamMember") -> "UnifiedTeamCoordinator":
        """Set the manager for HIERARCHICAL formation.

        Args:
            manager: Manager member

        Returns:
            Self for fluent chaining
        """
        self._manager = manager
        if manager not in self._members:
            self._members.insert(0, manager)
        return self

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with the team.

        Dispatches to the appropriate formation executor based on
        the current formation setting.

        Args:
            task: Task description
            context: Execution context with shared state

        Returns:
            Result dictionary with:
                - success: Whether execution succeeded
                - member_results: Results from each member
                - final_output: Synthesized final output
                - formation: Formation used
        """
        from victor.core.errors import RecursionDepthError

        if not self._members:
            return {
                "success": False,
                "error": "No team members added",
                "member_results": {},
                "final_output": "",
                "formation": self._formation.value,
            }

        # Initialize shared context
        self._shared_context = dict(context)
        self._message_history = []
        start_time = time.time()

        # Track recursion depth when team executes
        team_name = context.get("team_name", "UnifiedTeam")
        with RecursionGuard(self._recursion_ctx, "team", team_name):
            try:
                # Emit start event
                self._emit_team_event(
                    "started",
                    {
                        "task": task,
                        "formation": self._formation.value,
                        "member_count": len(self._members),
                        "recursion_depth": self._recursion_ctx.current_depth,
                    },
                )

                # Dispatch to formation executor
                result = await self._execute_formation(task, context)

                # Record RL outcome
                duration = time.time() - start_time
                failed_count = sum(
                    1 for r in result.get("member_results", {}).values() if not r.success
                )
                quality = self._compute_quality_score(
                    success=result.get("success", False),
                    member_count=len(self._members),
                    total_tool_calls=result.get("total_tool_calls", 0),
                    duration_seconds=duration,
                    failed_members=failed_count,
                )

                self._record_team_rl_outcome(
                    team_name=team_name,
                    formation=self._formation.value,
                    success=result.get("success", False),
                    quality_score=quality,
                    metadata={
                        "member_count": len(self._members),
                        "duration": duration,
                    },
                )

                # Emit completion event
                self._emit_team_event(
                    "completed",
                    {
                        "success": result.get("success", False),
                        "duration": duration,
                    },
                )

                return result

            except RecursionDepthError:
                # Re-raise recursion errors as-is
                raise
            except Exception as e:
                self._emit_team_event("error", {"error": str(e)})
                logger.error(f"Team execution failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "member_results": {},
                    "final_output": "",
                    "formation": self._formation.value,
                }

    async def broadcast(self, message: AgentMessage) -> List[Optional[AgentMessage]]:
        """Broadcast a message to all team members.

        Args:
            message: Message to broadcast (recipient_id should be None)

        Returns:
            List of responses from members
        """
        self._message_history.append(message)
        responses: List[Optional[AgentMessage]] = []

        for member in self._members:
            if member.id != message.sender_id:  # Don't send to sender
                try:
                    response = await member.receive_message(message)
                    if response:
                        self._message_history.append(response)
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Member {member.id} failed to receive: {e}")
                    responses.append(None)

        return responses

    async def send_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Send a message to a specific team member.

        Args:
            message: Message with recipient_id set

        Returns:
            Response from the recipient, or None if not found
        """
        if not message.recipient_id:
            return None

        self._message_history.append(message)

        # Find recipient
        for member in self._members:
            if member.id == message.recipient_id:
                try:
                    response = await member.receive_message(message)
                    if response:
                        self._message_history.append(response)
                    return response
                except Exception as e:
                    logger.warning(f"Member {member.id} failed to receive: {e}")
                    return None

        return None

    # =========================================================================
    # IMessageBusProvider / ISharedMemoryProvider
    # =========================================================================

    @property
    def message_bus(self) -> Optional[Any]:
        """Get message bus (self provides message routing)."""
        return self

    @property
    def shared_memory(self) -> Dict[str, Any]:
        """Get shared memory context."""
        return self._shared_context

    # =========================================================================
    # Formation Executors
    # =========================================================================

    async def _execute_formation(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using formation strategies.

        This replaces the old per-formation methods with a single method
        that delegates to the appropriate formation strategy.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Result dictionary
        """
        # Get the formation strategy
        strategy = self._get_formations()[self._formation]

        # Wrap team members with adapters
        adapted_members = [_TeamMemberAdapter(m, context) for m in self._members]

        # Create TeamContext
        # Add explicit manager ID if set (for hierarchical formation)
        shared_state_with_manager = dict(self._shared_context)
        if self._manager is not None:
            shared_state_with_manager["explicit_manager_id"] = self._manager.id

        team_context = TeamContext(
            team_id=context.get("team_name", "UnifiedTeam"),
            formation=self._formation.value,
            shared_state=shared_state_with_manager,
            **context,
        )

        # Create AgentMessage for the task
        agent_task = AgentMessage(
            sender_id="coordinator",
            content=task,
            message_type=MessageType.TASK,
            data=context,
        )

        # Execute using formation strategy
        member_results_list = await strategy.execute(adapted_members, team_context, agent_task)

        # Convert list of MemberResults to dict
        member_results: Dict[str, MemberResult] = {r.member_id: r for r in member_results_list}

        # Build final output
        success = all(r.success for r in member_results_list) if member_results_list else False
        final_outputs = [r.output for r in member_results_list if r.success]
        total_tool_calls = sum(r.tool_calls_used for r in member_results_list)

        # Determine final output based on formation
        # For pipeline, use only the last stage's output
        # For other formations, join all outputs
        if self._formation == TeamFormation.PIPELINE and final_outputs:
            final_output = final_outputs[-1]  # Last stage's output only
        else:
            final_output = "\n\n".join(final_outputs)

        # Extract consensus metadata if present (from ConsensusFormation)
        result_dict = {
            "success": success,
            "member_results": member_results,
            "final_output": final_output,
            "formation": self._formation.value,
            "total_tool_calls": total_tool_calls,
            "communication_log": self._message_history,
            "shared_context": self._shared_context,
        }

        # Add consensus metadata if any member result has it
        if member_results_list:
            first_metadata = member_results_list[0].metadata
            if "consensus_achieved" in first_metadata:
                result_dict["consensus_achieved"] = first_metadata["consensus_achieved"]
            if "consensus_rounds" in first_metadata:
                result_dict["consensus_rounds"] = first_metadata["consensus_rounds"]

        return result_dict

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear(self) -> "UnifiedTeamCoordinator":
        """Clear all members and reset state.

        Returns:
            Self for fluent chaining
        """
        self._members.clear()
        self._manager = None
        self._message_history.clear()
        self._shared_context.clear()
        return self

    @property
    def members(self) -> List["ITeamMember"]:
        """Get list of team members."""
        return list(self._members)

    @property
    def formation(self) -> TeamFormation:
        """Get current formation."""
        return self._formation

    @property
    def manager(self) -> Optional["ITeamMember"]:
        """Get team manager (for hierarchical formation)."""
        return self._manager

    def _get_formations(self) -> Dict[TeamFormation, BaseFormationStrategy]:
        """Get formation strategies, lazy loading to avoid circular imports."""
        if self._formations is None:
            # Lazy import to avoid circular dependency
            from victor.coordination.formations import (
                ConsensusFormation,
                HierarchicalFormation,
                ParallelFormation,
                PipelineFormation,
                SequentialFormation,
            )

            self._formations = {
                TeamFormation.SEQUENTIAL: SequentialFormation(),
                TeamFormation.PARALLEL: ParallelFormation(),
                TeamFormation.HIERARCHICAL: HierarchicalFormation(),
                TeamFormation.PIPELINE: PipelineFormation(),
                TeamFormation.CONSENSUS: ConsensusFormation(),
            }
        return self._formations

    # =========================================================================
    # Recursion Tracking Helper Methods
    # =========================================================================

    def get_recursion_depth(self) -> int:
        """Get current recursion depth.

        Returns:
            Current recursion depth level
        """
        return self._recursion_ctx.current_depth

    def can_spawn_nested(self) -> bool:
        """Check if team can spawn nested workflows/teams.

        Returns:
            True if nesting is allowed, False otherwise
        """
        return self._recursion_ctx.can_nest(1)
