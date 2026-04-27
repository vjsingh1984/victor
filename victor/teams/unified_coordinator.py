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
import copy
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.coordination.formations import (
    SequentialFormation,
    ParallelFormation,
    HierarchicalFormation,
    PipelineFormation,
    ConsensusFormation,
)
from victor.teams.mixins.observability import ObservabilityMixin
from victor.teams.mixins.rl import RLMixin
from victor.teams.types import (
    AgentMessage,
    MemberResult,
    MessageType,
    TeamFormation,
    TeamResult,
)
from victor.teams.merge_analyzer import MergeAnalyzer
from victor.teams.worktree_runtime import WorktreeExecutionPlan, WorktreeIsolationPlanner

if TYPE_CHECKING:
    from victor.protocols.team import ITeamMember

logger = logging.getLogger(__name__)


# =============================================================================
# Team Member Adapter
# =============================================================================


class _TeamMemberAdapter:
    """Adapter to bridge ITeamMember to formation strategy agent interface.

    Formation strategies expect agents with execute(task, context) -> MemberResult
    but ITeamMember uses execute_task(task, context) -> str.
    """

    def __init__(
        self,
        member: "ITeamMember",
        coordinator_context: Dict[str, Any],
        member_context: Optional[Dict[str, Any]] = None,
    ):
        self._member = member
        self._context = coordinator_context
        self._member_context = dict(member_context or {})
        self.id = member.id

    @property
    def role(self):
        """Expose role from underlying member for formation strategy detection."""
        return getattr(self._member, "role", None)

    async def execute(self, task: AgentMessage, context: TeamContext) -> MemberResult:
        """Execute task using ITeamMember interface."""
        start_time = time.time()

        try:
            # Merge TeamContext.shared_state into coordinator context
            merged_context = {**self._context, **context.shared_state, **self._member_context}

            # Preserve caller-owned mutable context objects when present so
            # external observers can track execution side effects without
            # losing formation-managed shared-state overlays like
            # ``previous_output``.
            for key, value in self._context.items():
                if (
                    key in context.shared_state
                    and context.shared_state[key] is not value
                    and isinstance(value, (list, dict, set))
                ):
                    merged_context[key] = value

            # Call ITeamMember's execute_task
            raw_output = await self._member.execute_task(task.content, merged_context)

            duration = time.time() - start_time
            (
                success,
                output,
                error,
                metadata,
                tool_calls_used,
                discoveries,
                duration_override,
            ) = self._normalize_execution_result(raw_output)
            metadata.setdefault("task", task.content)
            if "worktree_assignment" in self._member_context and "worktree_assignment" not in metadata:
                metadata["worktree_assignment"] = self._member_context["worktree_assignment"]
            if "claimed_paths" in self._member_context and "claimed_paths" not in metadata:
                metadata["claimed_paths"] = list(self._member_context["claimed_paths"])
            if "readonly_paths" in self._member_context and "readonly_paths" not in metadata:
                metadata["readonly_paths"] = list(self._member_context["readonly_paths"])

            return MemberResult(
                member_id=self._member.id,
                success=success,
                output=output,
                error=error,
                duration_seconds=duration_override if duration_override is not None else duration,
                metadata=metadata,
                tool_calls_used=tool_calls_used,
                discoveries=discoveries,
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

    @staticmethod
    def _normalize_execution_result(
        raw_output: Any,
    ) -> tuple[bool, str, Optional[str], Dict[str, Any], int, List[str], Optional[float]]:
        """Normalize string or mapping member outputs into MemberResult fields."""
        if not isinstance(raw_output, Mapping):
            return True, "" if raw_output is None else str(raw_output), None, {}, 0, [], None

        success = bool(raw_output.get("success", True))
        output_value = (
            raw_output.get("output")
            or raw_output.get("final_output")
            or raw_output.get("content")
            or ""
        )
        error = str(raw_output.get("error")) if raw_output.get("error") is not None else None
        metadata = dict(raw_output.get("metadata", {}) or {})
        for key in ("changed_files", "files_touched", "modified_files", "claimed_paths", "readonly_paths"):
            if raw_output.get(key) is not None and key not in metadata:
                metadata[key] = raw_output.get(key)

        discoveries = list(raw_output.get("discoveries") or [])
        tool_calls_raw = raw_output.get("tool_calls_used", raw_output.get("tool_calls", 0))
        try:
            tool_calls_used = int(tool_calls_raw or 0)
        except (TypeError, ValueError):
            tool_calls_used = 0

        duration_raw = raw_output.get("duration_seconds")
        try:
            duration_override = float(duration_raw) if duration_raw is not None else None
        except (TypeError, ValueError):
            duration_override = None

        return success, str(output_value), error, metadata, tool_calls_used, discoveries, duration_override


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
        lightweight_mode: bool = False,
        worktree_planner: Optional[Any] = None,
        merge_analyzer: Optional[Any] = None,
    ) -> None:
        """Initialize the unified coordinator.

        Args:
            orchestrator: Optional agent orchestrator for SubAgent spawning
            enable_observability: Enable EventBus integration
            enable_rl: Enable RL integration
            lightweight_mode: If True, disable mixins (for testing without dependencies)
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
        self._message_lock = asyncio.Lock()
        self._shared_context: Dict[str, Any] = {}
        self._worktree_planner = worktree_planner or WorktreeIsolationPlanner()
        self._merge_analyzer = merge_analyzer or MergeAnalyzer()

        # LSP capability for language intelligence
        self._lsp: Optional[Any] = None

        # Formation strategies (composition over inheritance)
        self._formations: Dict[TeamFormation, BaseFormationStrategy] = {
            TeamFormation.SEQUENTIAL: SequentialFormation(),
            TeamFormation.PARALLEL: ParallelFormation(),
            TeamFormation.HIERARCHICAL: HierarchicalFormation(),
            TeamFormation.PIPELINE: PipelineFormation(),
            TeamFormation.CONSENSUS: ConsensusFormation(),
        }

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

        # Initialize shared context (deep copy for isolation)
        self._shared_context = copy.deepcopy(dict(context))
        self._message_history = []
        start_time = time.time()
        effective_formation = self._resolve_effective_formation(context)

        # Emit start event
        self._emit_team_event(
            "started",
            {
                "task": task,
                "formation": effective_formation.value,
                "member_count": len(self._members),
            },
        )

        try:
            # Dispatch to formation executor
            result = await self._execute_formation(
                task,
                context,
                formation_override=effective_formation,
            )

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
                formation=effective_formation.value,
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
                "formation": effective_formation.value,
            }

    async def broadcast(self, message: AgentMessage) -> List[Optional[AgentMessage]]:
        """Broadcast a message to all team members.

        Args:
            message: Message to broadcast (recipient_id should be None)

        Returns:
            List of responses from members
        """
        async with self._message_lock:
            self._message_history.append(message)
        responses: List[Optional[AgentMessage]] = []

        for member in self._members:
            if member.id != message.sender_id:  # Don't send to sender
                try:
                    response = await member.receive_message(message)
                    if response:
                        async with self._message_lock:
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

        async with self._message_lock:
            self._message_history.append(message)

        # Find recipient
        for member in self._members:
            if member.id == message.recipient_id:
                try:
                    response = await member.receive_message(message)
                    if response:
                        async with self._message_lock:
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

    @property
    def lsp(self) -> Optional[Any]:
        """Get the LSP capability for code intelligence in team coordination.

        Returns:
            LSPCapability instance or None
        """
        return self._lsp

    def set_lsp(self, lsp_capability: Any) -> None:
        """Set the LSP capability for team coordination.

        Enables language intelligence features for code-related team
        operations and member coordination.

        Args:
            lsp_capability: LSPCapability instance
        """
        self._lsp = lsp_capability

    # =========================================================================
    # Formation Executors
    # =========================================================================

    async def _execute_formation(
        self,
        task: str,
        context: Dict[str, Any],
        formation_override: Optional[TeamFormation] = None,
    ) -> Dict[str, Any]:
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
        active_formation = formation_override or self._formation
        strategy = self._formations[active_formation]

        # Wrap team members with adapters
        shared_state_with_manager = dict(self._shared_context)
        if self._manager is not None:
            shared_state_with_manager["explicit_manager_id"] = self._manager.id

        max_workers = self._extract_max_workers(context, shared_state_with_manager)
        execution_members = self._limit_execution_members(
            self._members,
            active_formation,
            max_workers,
        )
        worktree_plan = self._plan_worktree_execution(
            execution_members,
            context=context,
            formation=active_formation,
        )
        member_context_overrides = (
            {
                assignment.member_id: assignment.to_context_overrides()
                for assignment in worktree_plan.assignments
            }
            if worktree_plan is not None
            else {}
        )
        adapted_members = [
            _TeamMemberAdapter(m, context, member_context_overrides.get(m.id))
            for m in execution_members
        ]

        # Create TeamContext
        if max_workers is not None:
            shared_state_with_manager["max_workers"] = max_workers
        shared_state_with_manager["effective_formation"] = active_formation.value
        if worktree_plan is not None:
            shared_state_with_manager["worktree_plan"] = worktree_plan.to_dict()

        team_context = TeamContext(
            team_id=context.get("team_name", "UnifiedTeam"),
            formation=active_formation.value,
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
            "formation": active_formation.value,
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
        if worktree_plan is not None:
            result_dict["worktree_plan"] = worktree_plan.to_dict()
        merge_analysis = self._analyze_merge(member_results, worktree_plan=worktree_plan)
        if merge_analysis is not None:
            result_dict["merge_analysis"] = merge_analysis.to_dict()
            result_dict["merge_risk_level"] = merge_analysis.risk_level.value

        return result_dict

    def _plan_worktree_execution(
        self,
        members: List["ITeamMember"],
        *,
        context: Dict[str, Any],
        formation: TeamFormation,
    ) -> Optional[WorktreeExecutionPlan]:
        planner = getattr(self._worktree_planner, "plan", None)
        if not callable(planner):
            return None
        try:
            return planner(members, context=context, formation=formation)
        except Exception as exc:
            logger.debug("Worktree planning failed; continuing without isolation: %s", exc)
            return None

    def _analyze_merge(
        self,
        member_results: Dict[str, MemberResult],
        *,
        worktree_plan: Optional[WorktreeExecutionPlan],
    ) -> Optional[Any]:
        analyzer = getattr(self._merge_analyzer, "analyze", None)
        if not callable(analyzer):
            return None
        try:
            return analyzer(member_results, worktree_plan=worktree_plan)
        except Exception as exc:
            logger.debug("Merge analysis failed; continuing without merge metadata: %s", exc)
            return None

    def _resolve_effective_formation(self, context: Dict[str, Any]) -> TeamFormation:
        """Resolve formation override hints without mutating the default formation."""
        raw_hint = context.get("formation_hint") or context.get("topology_formation_hint")
        if not raw_hint:
            return self._formation

        normalized = str(raw_hint).strip().lower()
        for formation in TeamFormation:
            if formation.value == normalized:
                return formation
        return self._formation

    @staticmethod
    def _extract_max_workers(
        context: Dict[str, Any],
        shared_state: Dict[str, Any],
    ) -> Optional[int]:
        """Extract max worker hint from the execution context."""
        raw_value = context.get("max_workers", shared_state.get("max_workers"))
        if raw_value is None:
            return None
        try:
            max_workers = int(raw_value)
        except (TypeError, ValueError):
            return None
        return max_workers if max_workers > 0 else None

    def _limit_execution_members(
        self,
        members: List["ITeamMember"],
        formation: TeamFormation,
        max_workers: Optional[int],
    ) -> List["ITeamMember"]:
        """Limit members for a single execution while preserving a hierarchical manager."""
        if max_workers is None or max_workers >= len(members):
            return list(members)

        if formation == TeamFormation.HIERARCHICAL and self._manager in members:
            selected: List["ITeamMember"] = [self._manager]
            for member in members:
                if member is self._manager:
                    continue
                selected.append(member)
                if len(selected) >= max_workers:
                    break
            return selected

        return list(members[:max_workers])

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
