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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional

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
from victor.teams.worktree_runtime import (
    GitWorktreeRuntime,
    WorktreeExecutionPlan,
    WorktreeIsolationPlanner,
    WorktreeMaterializationSession,
)

if TYPE_CHECKING:
    from victor.protocols.team import ITeamMember

logger = logging.getLogger(__name__)


# =============================================================================
# StateGraph Node Configuration
# =============================================================================


@dataclass(frozen=True)
class StateGraphNodeConfig:
    """Configuration for ``UnifiedTeamCoordinator`` used as a StateGraph node.

    The default values match the historical behaviour of ``__call__``: the
    node reads the task from ``state["task"]`` (falling back to
    ``state["query"]``) and writes the team result under ``"result"`` /
    ``"team_output"`` / ``"error"``. Override any field to map the node onto
    an existing graph schema without renaming state keys.

    ``formation_strategy`` is an optional callable that receives the original
    state and returns a ``TeamFormation``. Its result is injected into the
    per-call context as ``formation_hint`` so the coordinator's existing
    ``_resolve_effective_formation`` does the work — ``self._formation`` is
    never mutated, which keeps concurrent ``__call__`` invocations safe.
    """

    task_key: str = "task"
    query_key: str = "query"
    result_key: str = "result"
    output_key: str = "team_output"
    error_key: str = "error"
    formation_strategy: Optional[Callable[[Any], TeamFormation]] = None


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
            if (
                "worktree_assignment" in self._member_context
                and "worktree_assignment" not in metadata
            ):
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
        for key in (
            "changed_files",
            "files_touched",
            "modified_files",
            "claimed_paths",
            "readonly_paths",
        ):
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

        return (
            success,
            str(output_value),
            error,
            metadata,
            tool_calls_used,
            discoveries,
            duration_override,
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
        worktree_runtime: Optional[Any] = None,
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
        self._worktree_runtime = worktree_runtime or GitWorktreeRuntime()

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

        # StateGraph node config (used when the coordinator is invoked as a
        # graph node via ``__call__``). Default preserves historical keys.
        self._state_graph_config: StateGraphNodeConfig = StateGraphNodeConfig()

        # Lock that serialises ``execute_team_config`` invocations on a
        # shared coordinator instance. The implementation temporarily swaps
        # ``_formation`` / ``_members`` / ``_manager`` / ``_shared_context``;
        # serialising avoids cross-call corruption while letting callers
        # safely use ``asyncio.gather`` against the same coordinator.
        self._exec_team_config_lock: asyncio.Lock = asyncio.Lock()

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

    async def execute_team_config(
        self,
        config: Any,  # TeamConfig from victor.teams.types
        *,
        members: Optional[List[Any]] = None,
    ) -> Any:  # TeamResult from victor.teams.types
        """Execute a team with a config without mutating coordinator state.

        This method enables TeamNode and other callers to execute teams
        with a TeamConfig without first mutating self._formation or
        self._members. The coordinator's internal state remains unchanged,
        making this method safe for concurrent calls on a shared coordinator.

        Args:
            config: TeamConfig with name, goal, formation, members
            members: Optional member override (uses config.members if not provided)

        Returns:
            TeamResult with success, final_output, member_results, formation

        Raises:
            ValueError: If no members provided and config.members requires
                an orchestrator to resolve (but self._orchestrator is None)
        """
        from victor.teams.types import TeamResult, TeamFormation as TF

        # Resolve members: override parameter > config.members
        if members is None:
            # Try to use config.members - these are TeamMember configs,
            # not executable IAgent instances, so we need to adapt them
            if hasattr(config, "members") and config.members:
                # Check if we have an orchestrator to resolve TeamMember to IAgent
                if self._orchestrator is None:
                    raise ValueError(
                        "Cannot execute TeamConfig with members: "
                        "coordinator has no orchestrator to resolve TeamMember configs. "
                        "Pass executable members via the members= parameter, or "
                        "create the coordinator with an orchestrator."
                    )
                # For now, require explicit members= override
                # (Future: implement TeamMember -> IAgent resolution via orchestrator)
                raise ValueError(
                    "Cannot execute TeamConfig with members without explicit members= override. "
                    "Pass executable IAgent instances via the members= parameter."
                )
            else:
                raise ValueError(
                    "No members to execute: either config.members must be non-empty "
                    "or members= override must be provided."
                )

        if not members:
            return TeamResult(
                success=False,
                final_output="",
                member_results={},
                formation=TF.SEQUENTIAL,
                error="No members to execute",
            )

        # Resolve formation from config (use SEQUENTIAL as fallback)
        config_formation = getattr(config, "formation", TF.SEQUENTIAL)
        if isinstance(config_formation, str):
            config_formation = TF(config_formation)

        # Save current state
        saved_members = self._members
        saved_formation = self._formation

        try:
            # Temporarily set members and formation for execution
            self._members = members
            self._formation = config_formation

            # Execute the task
            task = getattr(config, "goal", "Execute team task")
            context = getattr(config, "shared_context", {})

            result_dict = await self.execute_task(task, context)

            # Convert dict result to TeamResult
            from victor.teams.types import MemberResult

            member_results: Dict[str, MemberResult] = {}
            for member_id, mr_dict in result_dict.get("member_results", {}).items():
                # Convert dict to MemberResult if needed
                if isinstance(mr_dict, dict):
                    member_results[member_id] = MemberResult(
                        member_id=member_id,
                        success=mr_dict.get("success", False),
                        output=mr_dict.get("output", ""),
                        error=mr_dict.get("error"),
                        tool_calls=mr_dict.get("tool_calls", []),
                        duration_seconds=mr_dict.get("duration_seconds", 0.0),
                    )
                else:
                    member_results[member_id] = mr_dict

            return TeamResult(
                success=result_dict.get("success", False),
                final_output=result_dict.get("final_output", ""),
                member_results=member_results,
                formation=config_formation,
                total_tool_calls=result_dict.get("total_tool_calls", 0),
                total_duration=result_dict.get("total_duration", 0.0),
                error=result_dict.get("error"),
            )
        finally:
            # Restore coordinator state (crucial for concurrent safety)
            self._members = saved_members
            self._formation = saved_formation

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
        worktree_session = self._materialize_worktree_plan(worktree_plan, context=context)
        worktree_overrides_source = (
            worktree_session.assignments
            if worktree_session
            else (worktree_plan.assignments if worktree_plan is not None else ())
        )
        member_context_overrides = (
            {
                assignment.member_id: assignment.to_context_overrides()
                for assignment in worktree_overrides_source
            }
            if worktree_overrides_source
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
        if worktree_session is not None:
            shared_state_with_manager["worktree_session"] = worktree_session.to_dict()

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

        result_dict: Optional[Dict[str, Any]] = None
        try:
            # Execute using formation strategy
            member_results_list = await strategy.execute(adapted_members, team_context, agent_task)

            # Convert list of MemberResults to dict
            member_results: Dict[str, MemberResult] = {r.member_id: r for r in member_results_list}
            self._inject_worktree_changed_files(
                member_results,
                worktree_session=worktree_session,
            )

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
            if worktree_session is not None:
                result_dict["worktree_session"] = worktree_session.to_dict()
            merge_analysis = self._analyze_merge(member_results, worktree_plan=worktree_plan)
            if merge_analysis is not None:
                result_dict["merge_analysis"] = merge_analysis.to_dict()
                result_dict["merge_risk_level"] = merge_analysis.risk_level.value
                if worktree_session is not None:
                    merge_orchestration = self._build_merge_orchestration(
                        worktree_session,
                        merge_analysis=merge_analysis.to_dict(),
                    )
                    if merge_orchestration is not None:
                        result_dict["merge_orchestration"] = merge_orchestration

            return result_dict
        finally:
            if worktree_session is not None and self._should_cleanup_worktrees(context):
                cleanup_summary = self._cleanup_worktree_session(worktree_session)
                if result_dict is not None:
                    result_dict["worktree_cleanup"] = cleanup_summary

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

    @staticmethod
    def _coerce_context_flag(
        context: Dict[str, Any],
        key: str,
        *,
        default: bool = False,
    ) -> bool:
        raw_value = context.get(key)
        if raw_value is None:
            return default
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

    def _materialize_worktree_plan(
        self,
        worktree_plan: Optional[WorktreeExecutionPlan],
        *,
        context: Dict[str, Any],
    ) -> Optional[WorktreeMaterializationSession]:
        if worktree_plan is None:
            return None
        materialize = self._coerce_context_flag(context, "materialize_worktrees", default=False)
        dry_run = self._coerce_context_flag(context, "dry_run_worktrees", default=False)
        if not materialize and not dry_run:
            return None
        runtime = getattr(self._worktree_runtime, "materialize", None)
        if not callable(runtime):
            return None
        try:
            return runtime(worktree_plan, dry_run=dry_run)
        except Exception as exc:
            logger.debug("Worktree materialization failed; continuing with planned paths: %s", exc)
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

    def _inject_worktree_changed_files(
        self,
        member_results: Dict[str, MemberResult],
        *,
        worktree_session: Optional[WorktreeMaterializationSession],
    ) -> None:
        if worktree_session is None:
            return
        collector = getattr(self._worktree_runtime, "collect_changed_files", None)
        if not callable(collector):
            return
        for member_id, result in list(member_results.items()):
            metadata = dict(result.metadata or {})
            if any(
                metadata.get(key) for key in ("changed_files", "files_touched", "modified_files")
            ):
                continue
            try:
                changed_files = list(collector(worktree_session, member_id))
            except Exception as exc:
                logger.debug("Failed to collect changed files for %s: %s", member_id, exc)
                continue
            if not changed_files:
                continue
            metadata["changed_files"] = changed_files
            member_results[member_id] = MemberResult(
                member_id=result.member_id,
                success=result.success,
                output=result.output,
                error=result.error,
                metadata=metadata,
                tool_calls_used=result.tool_calls_used,
                duration_seconds=result.duration_seconds,
                discoveries=list(result.discoveries),
            )

    def _build_merge_orchestration(
        self,
        worktree_session: WorktreeMaterializationSession,
        *,
        merge_analysis: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        builder = getattr(self._worktree_runtime, "build_merge_orchestration", None)
        if not callable(builder):
            return None
        try:
            return builder(worktree_session, merge_analysis=merge_analysis)
        except Exception as exc:
            logger.debug("Merge orchestration build failed: %s", exc)
            return None

    def _should_cleanup_worktrees(self, context: Dict[str, Any]) -> bool:
        return self._coerce_context_flag(context, "cleanup_worktrees", default=True)

    def _cleanup_worktree_session(
        self,
        worktree_session: WorktreeMaterializationSession,
    ) -> Dict[str, Any]:
        cleaner = getattr(self._worktree_runtime, "cleanup", None)
        if not callable(cleaner):
            return {"removed": [], "skipped": [], "errors": []}
        try:
            return cleaner(worktree_session, force=True)
        except Exception as exc:
            logger.debug("Worktree cleanup failed: %s", exc)
            return {"removed": [], "skipped": [], "errors": [str(exc)]}

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

    # =========================================================================
    # Parameterised execution & TeamConfig adapter
    # =========================================================================

    async def _execute_with(
        self,
        task: str,
        context: Dict[str, Any],
        *,
        formation: TeamFormation,
        members: List["ITeamMember"],
        manager: Optional["ITeamMember"] = None,
    ) -> Dict[str, Any]:
        """Concurrency-safe parameterised execution.

        Temporarily applies the given formation/members/manager, delegates
        to ``execute_task``, then restores the coordinator's prior state.
        Serialised via ``self._exec_team_config_lock`` so concurrent
        invocations on a shared coordinator instance don't corrupt each
        other's view of ``_formation`` / ``_members``.
        """
        async with self._exec_team_config_lock:
            saved_formation = self._formation
            saved_members = list(self._members)
            saved_manager = self._manager
            try:
                self._formation = formation
                self._members = list(members)
                self._manager = manager
                return await self.execute_task(task, context)
            finally:
                self._formation = saved_formation
                self._members = saved_members
                self._manager = saved_manager

    def _adapt_team_members(
        self, members: List[Any]
    ) -> List["ITeamMember"]:
        """Adapt ``TeamMember`` dataclasses to ``ITeamMember`` adapters.

        Uses ``self._orchestrator`` to build a ``SubAgentOrchestrator`` that
        actually executes each member. Raises ``ValueError`` if no
        orchestrator is configured — callers can bypass this requirement by
        passing pre-built ``ITeamMember`` instances via the ``members=``
        parameter on ``execute_team_config``.
        """
        if self._orchestrator is None:
            raise ValueError(
                "UnifiedTeamCoordinator requires an orchestrator to adapt "
                "TeamMember dataclasses into ITeamMember instances. Either "
                "construct the coordinator with an orchestrator, or pass "
                "pre-built members via execute_team_config(config, members=...)."
            )

        from victor.teams.types import TeamMemberAdapter
        from victor.agent.subagents.orchestrator import SubAgentOrchestrator

        sub_orchestrator = SubAgentOrchestrator(self._orchestrator)

        def _make_executor(team_member):
            async def executor(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
                spawn_result = await sub_orchestrator.spawn(
                    role=team_member.role,
                    task=task,
                    tool_budget=team_member.tool_budget,
                    allowed_tools=team_member.allowed_tools,
                )
                return {
                    "success": getattr(spawn_result, "success", False),
                    "output": getattr(spawn_result, "summary", "") or "",
                    "error": getattr(spawn_result, "error", None),
                    "tool_calls_used": getattr(spawn_result, "tool_calls_used", 0),
                    "duration_seconds": getattr(spawn_result, "duration_seconds", 0.0),
                }

            return executor

        return [
            TeamMemberAdapter(member=m, executor=_make_executor(m)) for m in members
        ]

    def _dict_result_to_team_result(
        self,
        result: Dict[str, Any],
        *,
        formation: TeamFormation,
    ) -> TeamResult:
        """Convert ``execute_task``'s dict result into a ``TeamResult``."""
        return TeamResult(
            success=bool(result.get("success", False)),
            final_output=str(result.get("final_output", "")),
            member_results=dict(result.get("member_results", {})),
            formation=formation,
            total_tool_calls=int(result.get("total_tool_calls", 0)),
            total_duration=float(result.get("total_duration", 0.0)),
            communication_log=list(result.get("communication_log", [])),
            shared_context=dict(result.get("shared_context", {})),
            consensus_achieved=result.get("consensus_achieved"),
            consensus_rounds=result.get("consensus_rounds"),
            error=result.get("error"),
        )

    async def execute_team_config(
        self,
        config: Any,
        *,
        members: Optional[List["ITeamMember"]] = None,
    ) -> TeamResult:
        """Execute a ``TeamConfig`` and return a ``TeamResult``.

        This is the entry point used by ``TeamNode`` in
        ``victor.framework.workflows.nodes`` and any other caller that
        already has a ``TeamConfig`` in hand.

        Args:
            config: A ``TeamConfig`` describing the run (goal, formation,
                shared context, members).
            members: Optional pre-built ``ITeamMember`` instances. If
                omitted, the coordinator builds adapters from
                ``config.members`` using its orchestrator (raises
                ``ValueError`` if no orchestrator is configured).

        Returns:
            A ``TeamResult`` matching the formation in the config.
        """
        if members is None:
            members = self._adapt_team_members(list(config.members))

        result = await self._execute_with(
            task=config.goal,
            context=dict(config.shared_context or {}),
            formation=config.formation,
            members=members,
        )
        return self._dict_result_to_team_result(result, formation=config.formation)

    # =========================================================================
    # StateGraph Integration
    # =========================================================================

    def with_state_graph_config(
        self, config: StateGraphNodeConfig
    ) -> "UnifiedTeamCoordinator":
        """Configure how this coordinator behaves when used as a StateGraph node.

        ``__call__`` reads/writes the input/output keys named in ``config``.
        The default config preserves historical behaviour (``task`` / ``query``
        in, ``result`` / ``team_output`` / ``error`` out).

        Returns ``self`` for fluent chaining.
        """
        self._state_graph_config = config
        return self

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute as a StateGraph node.

        This makes the coordinator directly usable in StateGraph:
            graph.add_node("team", coordinator)

        The coordinator reads the task from ``state[task_key]`` (with a
        fallback to ``state[query_key]``), executes the team with the
        configured formation, and returns a *new* state dict with the
        results. The caller's input dict is never mutated.

        Use ``with_state_graph_config(...)`` to map the node onto a graph
        schema with different key names. For declarative workflow YAML with
        timeout/retry/merge-strategy configuration, use ``TeamNode`` from
        ``victor.framework.workflows.nodes`` instead — ``__call__`` is the
        lean programmatic path.

        Args:
            state: Current graph state. The task is read from
                ``state[config.task_key]`` (default ``"task"``), falling
                back to ``state[config.query_key]`` (default ``"query"``).
                All other keys are passed through as execution context.

        Returns:
            New state dict with:
                - ``config.result_key`` (default ``"result"``): final output
                  string when execution succeeded.
                - ``config.output_key`` (default ``"team_output"``): full
                  team execution result dict.
                - ``config.error_key`` (default ``"error"``): error message
                  when execution failed.

        Example::

            from victor.framework import StateGraph
            from victor.teams import (
                UnifiedTeamCoordinator,
                TeamFormation,
                StateGraphNodeConfig,
            )

            coordinator = UnifiedTeamCoordinator(orchestrator)
            coordinator.set_formation(TeamFormation.PARALLEL)
            coordinator.add_member(agent1).add_member(agent2)
            coordinator.with_state_graph_config(
                StateGraphNodeConfig(task_key="instruction", result_key="answer")
            )

            graph = StateGraph(AgentState)
            graph.add_node("research_team", coordinator)
        """
        config = self._state_graph_config

        # Extract task using configured keys (task takes precedence over query)
        task = state.get(config.task_key, state.get(config.query_key, ""))

        # Build context excluding task/query keys
        context = {
            k: v
            for k, v in state.items()
            if k not in (config.task_key, config.query_key)
        }

        # Execute team task
        result = await self.execute_task(task, context)

        # Build the return state as a shallow copy so the caller's dict is
        # never mutated. StateGraph executors already isolate state via
        # CopyOnWriteState, but well-behaved nodes still avoid in-place writes
        # so the function is safe to use programmatically too.
        new_state = dict(state)
        if result.get("success"):
            new_state[config.result_key] = result.get("final_output", "")
            new_state[config.output_key] = result
        else:
            new_state[config.error_key] = result.get("error", "Unknown error")
            new_state[config.output_key] = result

        return new_state
