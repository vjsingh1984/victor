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

Example:
    from victor.teams import UnifiedTeamCoordinator, TeamFormation

    coordinator = UnifiedTeamCoordinator(orchestrator)
    coordinator.set_execution_context(task_type="feature", complexity="high")

    coordinator.add_member(researcher).add_member(executor)
    coordinator.set_formation(TeamFormation.PIPELINE)

    result = await coordinator.execute_task("Implement authentication", {})
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.teams.mixins.observability import ObservabilityMixin
from victor.teams.mixins.rl import RLMixin
from victor.teams.types import (
    AgentMessage,
    MemberResult,
    MessageType,
    TeamFormation,
    TeamResult,
)

if TYPE_CHECKING:
    from victor.teams.protocols import ITeamMember

logger = logging.getLogger(__name__)


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

    Attributes:
        _orchestrator: Agent orchestrator (optional, for SubAgent spawning)
        _members: List of team members
        _formation: Current team formation
        _manager: Manager member for HIERARCHICAL formation
        _message_history: Log of inter-agent messages
        _shared_context: Shared context dictionary
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        *,
        enable_observability: bool = True,
        enable_rl: bool = True,
    ) -> None:
        """Initialize the unified coordinator.

        Args:
            orchestrator: Optional agent orchestrator for SubAgent spawning
            enable_observability: Enable EventBus integration
            enable_rl: Enable RL integration
        """
        # Initialize mixins
        ObservabilityMixin.__init__(self, enable_observability=enable_observability)
        RLMixin.__init__(self, enable_rl=enable_rl)

        # Core state
        self._orchestrator = orchestrator
        self._members: List[ITeamMember] = []
        self._formation = TeamFormation.SEQUENTIAL
        self._manager: Optional[ITeamMember] = None

        # Communication
        self._message_history: List[AgentMessage] = []
        self._shared_context: Dict[str, Any] = {}

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

        # Emit start event
        self._emit_team_event(
            "started",
            {
                "task": task,
                "formation": self._formation.value,
                "member_count": len(self._members),
            },
        )

        try:
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
                team_name=context.get("team_name", "UnifiedTeam"),
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
        """Dispatch to appropriate formation executor.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Result dictionary
        """
        executors = {
            TeamFormation.SEQUENTIAL: self._execute_sequential,
            TeamFormation.PARALLEL: self._execute_parallel,
            TeamFormation.HIERARCHICAL: self._execute_hierarchical,
            TeamFormation.PIPELINE: self._execute_pipeline,
            TeamFormation.CONSENSUS: self._execute_consensus,
        }
        return await executors[self._formation](task, context)

    async def _execute_sequential(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute members sequentially with context chaining.

        Each member receives the accumulated context from previous members.
        """
        member_results: Dict[str, MemberResult] = {}
        accumulated_context = dict(context)
        final_outputs: List[str] = []
        total_tool_calls = 0
        success = True

        for i, member in enumerate(self._members):
            self._report_progress(member.id, "executing", (i + 1) / len(self._members))

            try:
                start = time.time()
                output = await member.execute_task(task, accumulated_context)
                duration = time.time() - start

                result = MemberResult(
                    member_id=member.id,
                    success=True,
                    output=output,
                    duration_seconds=duration,
                )
                member_results[member.id] = result
                final_outputs.append(output)

                # Update context for next member
                accumulated_context[f"member_{member.id}_output"] = output
                self._shared_context[member.id] = output

            except Exception as e:
                logger.warning(f"Member {member.id} failed: {e}")
                member_results[member.id] = MemberResult(
                    member_id=member.id,
                    success=False,
                    output="",
                    error=str(e),
                )
                success = False

        return {
            "success": success,
            "member_results": member_results,
            "final_output": "\n\n".join(final_outputs),
            "formation": self._formation.value,
            "total_tool_calls": total_tool_calls,
            "communication_log": self._message_history,
            "shared_context": self._shared_context,
        }

    async def _execute_parallel(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all members in parallel.

        All members work simultaneously on the same task.
        """
        member_results: Dict[str, MemberResult] = {}

        async def run_member(member: "ITeamMember") -> MemberResult:
            try:
                start = time.time()
                output = await member.execute_task(task, context)
                duration = time.time() - start
                return MemberResult(
                    member_id=member.id,
                    success=True,
                    output=output,
                    duration_seconds=duration,
                )
            except Exception as e:
                return MemberResult(
                    member_id=member.id,
                    success=False,
                    output="",
                    error=str(e),
                )

        # Execute all in parallel
        results = await asyncio.gather(
            *[run_member(m) for m in self._members],
            return_exceptions=False,
        )

        for result in results:
            member_results[result.member_id] = result

        success = all(r.success for r in results)
        final_outputs = [r.output for r in results if r.success]

        return {
            "success": success,
            "member_results": member_results,
            "final_output": "\n\n".join(final_outputs),
            "formation": self._formation.value,
            "communication_log": self._message_history,
            "shared_context": self._shared_context,
        }

    async def _execute_hierarchical(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with manager-worker hierarchy.

        1. Manager plans and delegates
        2. Workers execute in parallel
        3. Manager synthesizes results
        """
        member_results: Dict[str, MemberResult] = {}

        # Determine manager
        manager = self._manager or (self._members[0] if self._members else None)
        if not manager:
            return {
                "success": False,
                "error": "No manager available",
                "member_results": {},
                "final_output": "",
                "formation": self._formation.value,
            }

        workers = [m for m in self._members if m != manager]

        # Phase 1: Manager planning
        self._report_progress(manager.id, "planning", 0.1)
        planning_context = {
            **context,
            "role": "manager",
            "phase": "planning",
            "worker_count": len(workers),
        }

        try:
            plan_output = await manager.execute_task(
                f"Plan delegation for: {task}", planning_context
            )
            member_results[manager.id] = MemberResult(
                member_id=manager.id,
                success=True,
                output=f"[Plan] {plan_output}",
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Manager planning failed: {e}",
                "member_results": member_results,
                "final_output": "",
                "formation": self._formation.value,
            }

        # Phase 2: Worker execution (parallel)
        if workers:
            worker_context = {
                **context,
                "role": "worker",
                "manager_plan": plan_output,
            }

            async def run_worker(worker: "ITeamMember") -> MemberResult:
                try:
                    output = await worker.execute_task(task, worker_context)
                    return MemberResult(
                        member_id=worker.id,
                        success=True,
                        output=output,
                    )
                except Exception as e:
                    return MemberResult(
                        member_id=worker.id,
                        success=False,
                        output="",
                        error=str(e),
                    )

            worker_results = await asyncio.gather(*[run_worker(w) for w in workers])

            for result in worker_results:
                member_results[result.member_id] = result

        # Phase 3: Manager synthesis
        self._report_progress(manager.id, "synthesizing", 0.9)
        worker_outputs = [r.output for r in member_results.values() if r.success]

        synthesis_context = {
            **context,
            "role": "manager",
            "phase": "synthesis",
            "worker_outputs": worker_outputs,
        }

        try:
            final_output = await manager.execute_task(
                f"Synthesize worker results for: {task}", synthesis_context
            )
        except Exception as e:
            final_output = f"Synthesis failed: {e}. Results: {worker_outputs}"

        success = all(r.success for r in member_results.values())

        return {
            "success": success,
            "member_results": member_results,
            "final_output": final_output,
            "formation": self._formation.value,
            "communication_log": self._message_history,
            "shared_context": self._shared_context,
        }

    async def _execute_pipeline(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute as a pipeline where output feeds to next stage.

        Each member's output becomes the input for the next member.
        """
        member_results: Dict[str, MemberResult] = {}
        current_input = task
        current_context = dict(context)
        success = True

        for i, member in enumerate(self._members):
            self._report_progress(member.id, "processing", (i + 1) / len(self._members))

            # Build pipeline context
            pipeline_context = {
                **current_context,
                "pipeline_stage": i,
                "pipeline_input": current_input,
            }

            try:
                start = time.time()
                output = await member.execute_task(current_input, pipeline_context)
                duration = time.time() - start

                member_results[member.id] = MemberResult(
                    member_id=member.id,
                    success=True,
                    output=output,
                    duration_seconds=duration,
                )

                # Output becomes input for next stage
                current_input = output
                current_context[f"stage_{i}_output"] = output

                # Send handoff message
                if i < len(self._members) - 1:
                    handoff = AgentMessage(
                        sender_id=member.id,
                        recipient_id=self._members[i + 1].id,
                        content=output,
                        message_type=MessageType.HANDOFF,
                    )
                    self._message_history.append(handoff)

            except Exception as e:
                logger.warning(f"Pipeline stage {i} failed: {e}")
                member_results[member.id] = MemberResult(
                    member_id=member.id,
                    success=False,
                    output="",
                    error=str(e),
                )
                success = False
                break  # Pipeline breaks on failure

        return {
            "success": success,
            "member_results": member_results,
            "final_output": current_input,  # Last stage output
            "formation": self._formation.value,
            "communication_log": self._message_history,
            "shared_context": current_context,
        }

    async def _execute_consensus(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with consensus - all members must agree.

        Members execute in parallel, and success requires consensus
        (configurable agreement threshold). If not reached, re-execute
        with shared context until agreement or max rounds.

        Args:
            task: Task description
            context: Context with optional:
                - max_consensus_rounds: Max attempts (default: 3)
                - agreement_threshold: Required agreement (default: 1.0)
        """
        max_rounds = context.get("max_consensus_rounds", 3)
        threshold = context.get("agreement_threshold", 1.0)

        for round_num in range(max_rounds):
            # Add round context
            round_context = {
                **context,
                "consensus_round": round_num + 1,
                "max_rounds": max_rounds,
            }

            if round_num > 0:
                # Add disagreement context for retry rounds
                round_context["previous_results"] = self._shared_context.get("previous_results", [])

            # Execute parallel
            result = await self._execute_parallel(task, round_context)

            # Check consensus
            member_results = result.get("member_results", {})
            success_rate = (
                sum(1 for r in member_results.values() if r.success) / len(member_results)
                if member_results
                else 0
            )

            if success_rate >= threshold:
                result["consensus_achieved"] = True
                result["consensus_rounds"] = round_num + 1
                return result

            # Store for next round
            self._shared_context["previous_results"] = [
                {"member": mid, "output": r.output} for mid, r in member_results.items()
            ]

        # Max rounds exceeded
        return {
            "success": False,
            "error": f"Consensus not reached after {max_rounds} rounds",
            "member_results": member_results if "member_results" in locals() else {},
            "final_output": "",
            "formation": self._formation.value,
            "consensus_achieved": False,
            "consensus_rounds": max_rounds,
        }

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
