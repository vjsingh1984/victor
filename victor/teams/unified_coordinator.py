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
from contextvars import ContextVar
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

    ``formation_strategy`` is an optional synchronous callable that receives
    the original state and returns a ``TeamFormation``. Its result is injected
    into the per-call context as ``formation_hint`` so the coordinator's
    existing ``_resolve_effective_formation`` does the work —
    ``self._formation`` is never mutated, which keeps concurrent ``__call__``
    invocations safe. Async strategies are intentionally not supported here.
    """

    task_key: str = "task"
    query_key: str = "query"
    result_key: str = "result"
    output_key: str = "team_output"
    error_key: str = "error"
    formation_strategy: Optional[Callable[[Any], TeamFormation]] = None


@dataclass
class _CoordinatorExecutionState:
    """Per-call execution state for concurrent-safe team runs."""

    members: List["ITeamMember"]
    formation: TeamFormation
    manager: Optional["ITeamMember"]
    shared_context: Dict[str, Any]
    message_history: List[AgentMessage] = field(default_factory=list)


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

        # Per-task execution state keeps parameterised runs concurrency-safe
        # without mutating coordinator defaults like ``_formation`` or
        # ``_members``.
        self._execution_state: ContextVar[Optional[_CoordinatorExecutionState]] = ContextVar(
            "unified_team_execution_state",
            default=None,
        )

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
        return await self._execute_with(
            task=task,
            context=context,
            formation=self._formation,
            members=list(self._members),
            manager=self._manager,
            persist_execution_state=True,
        )

    async def broadcast(self, message: AgentMessage) -> List[Optional[AgentMessage]]:
        """Broadcast a message to all team members.

        Args:
            message: Message to broadcast (recipient_id should be None)

        Returns:
            List of responses from members
        """
        async with self._message_lock:
            self._active_message_history().append(message)
        responses: List[Optional[AgentMessage]] = []

        for member in self._active_members():
            if member.id != message.sender_id:  # Don't send to sender
                try:
                    response = await member.receive_message(message)
                    if response:
                        async with self._message_lock:
                            self._active_message_history().append(response)
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
            self._active_message_history().append(message)

        # Find recipient
        for member in self._active_members():
            if member.id == message.recipient_id:
                try:
                    response = await member.receive_message(message)
                    if response:
                        async with self._message_lock:
                            self._active_message_history().append(response)
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
        return self._active_shared_context()

    def _current_execution_state(self) -> Optional[_CoordinatorExecutionState]:
        """Return the active per-call execution state, if any."""
        return self._execution_state.get()

    def _active_members(self) -> List["ITeamMember"]:
        state = self._current_execution_state()
        if state is not None:
            return list(state.members)
        return list(self._members)

    def _active_formation(self) -> TeamFormation:
        state = self._current_execution_state()
        if state is not None:
            return state.formation
        return self._formation

    def _active_manager(self) -> Optional["ITeamMember"]:
        state = self._current_execution_state()
        if state is not None:
            return state.manager
        return self._manager

    def _active_shared_context(self) -> Dict[str, Any]:
        state = self._current_execution_state()
        if state is not None:
            return state.shared_context
        return self._shared_context

    def _active_message_history(self) -> List[AgentMessage]:
        state = self._current_execution_state()
        if state is not None:
            return state.message_history
        return self._message_history

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
        active_formation = formation_override or self._active_formation()
        strategy = self._formations[active_formation]

        # Wrap team members with adapters
        shared_state_with_manager = self._active_shared_context()
        active_manager = self._active_manager()
        if active_manager is not None:
            shared_state_with_manager["explicit_manager_id"] = active_manager.id

        max_workers = self._extract_max_workers(context, shared_state_with_manager)
        execution_members = self._limit_execution_members(
            self._active_members(),
            active_formation,
            max_workers,
            manager=active_manager,
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
            if active_formation == TeamFormation.PIPELINE and final_outputs:
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
                "communication_log": list(self._active_message_history()),
                "shared_context": dict(shared_state_with_manager),
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
                    if self._should_execute_merge_orchestration(context):
                        merge_execution = self._execute_merge_orchestration(
                            worktree_session,
                            merge_analysis=merge_analysis.to_dict(),
                            context=context,
                        )
                        if merge_execution is not None:
                            result_dict["merge_execution"] = merge_execution

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

    def _should_execute_merge_orchestration(self, context: Dict[str, Any]) -> bool:
        return self._coerce_context_flag(context, "auto_merge_worktrees", default=False)

    def _execute_merge_orchestration(
        self,
        worktree_session: WorktreeMaterializationSession,
        *,
        merge_analysis: Optional[Dict[str, Any]] = None,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        executor = getattr(self._worktree_runtime, "execute_merge_orchestration", None)
        if not callable(executor):
            return None
        try:
            return executor(
                worktree_session,
                merge_analysis=merge_analysis,
                allow_risky=self._coerce_context_flag(
                    context,
                    "allow_risky_worktree_merge",
                    default=False,
                ),
                preserve_artifacts=self._coerce_context_flag(
                    context,
                    "preserve_merge_workspace",
                    default=False,
                ),
            )
        except Exception as exc:
            logger.debug("Merge orchestration execution failed: %s", exc)
            return {
                "status": "error",
                "executed": False,
                "blocked_reason": "merge_execution_failed",
                "error": str(exc),
            }

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

    def _resolve_effective_formation(
        self,
        context: Dict[str, Any],
        *,
        default_formation: Optional[TeamFormation] = None,
    ) -> TeamFormation:
        """Resolve formation override hints without mutating the default formation."""
        fallback_formation = default_formation or self._active_formation()
        raw_hint = context.get("formation_hint") or context.get("topology_formation_hint")
        if not raw_hint:
            return fallback_formation

        normalized = str(raw_hint).strip().lower()
        for formation in TeamFormation:
            if formation.value == normalized:
                return formation
        return fallback_formation

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
        *,
        manager: Optional["ITeamMember"] = None,
    ) -> List["ITeamMember"]:
        """Limit members for a single execution while preserving a hierarchical manager."""
        if max_workers is None or max_workers >= len(members):
            return list(members)

        if formation == TeamFormation.HIERARCHICAL and manager in members:
            selected: List["ITeamMember"] = [manager]
            for member in members:
                if member is manager:
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
        persist_execution_state: bool = False,
    ) -> Dict[str, Any]:
        """Execute a team run with per-call members/formation overrides.

        The coordinator's defaults remain unchanged. Per-call state lives in a
        task-local ``ContextVar`` so concurrent invocations can execute against
        the same coordinator instance without serialisation or instance swaps.
        """
        execution_members = list(members)
        if not execution_members:
            return {
                "success": False,
                "error": "No team members added",
                "member_results": {},
                "final_output": "",
                "formation": formation.value,
            }

        execution_state = _CoordinatorExecutionState(
            members=execution_members,
            formation=formation,
            manager=manager,
            shared_context=copy.deepcopy(dict(context)),
        )
        token = self._execution_state.set(execution_state)
        try:
            start_time = time.time()
            effective_formation = self._resolve_effective_formation(
                context,
                default_formation=formation,
            )

            self._emit_team_event(
                "started",
                {
                    "task": task,
                    "formation": effective_formation.value,
                    "member_count": len(execution_members),
                },
            )
            try:
                result = await self._execute_formation(
                    task,
                    context,
                    formation_override=effective_formation,
                )

                duration = time.time() - start_time
                failed_count = sum(
                    1 for r in result.get("member_results", {}).values() if not r.success
                )
                quality = self._compute_quality_score(
                    success=result.get("success", False),
                    member_count=len(execution_members),
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
                        "member_count": len(execution_members),
                        "duration": duration,
                    },
                )

                self._emit_team_event(
                    "completed",
                    {
                        "success": result.get("success", False),
                        "duration": duration,
                    },
                )
                if persist_execution_state:
                    self._shared_context = dict(
                        result.get("shared_context", execution_state.shared_context)
                    )
                    self._message_history = list(execution_state.message_history)
                return result

            except Exception as e:
                self._emit_team_event("error", {"error": str(e)})
                logger.error(f"Team execution failed: {e}")
                if persist_execution_state:
                    self._shared_context = dict(execution_state.shared_context)
                    self._message_history = list(execution_state.message_history)
                return {
                    "success": False,
                    "error": str(e),
                    "member_results": {},
                    "final_output": "",
                    "formation": effective_formation.value,
                }
        finally:
            self._execution_state.reset(token)

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

        This is the entry point used by the declarative team-step adapter in
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

    @staticmethod
    def _classify_state(state: Any) -> str:
        """Classify the incoming graph state.

        Returns one of ``"cow"``, ``"pydantic"``, or ``"dict"``. The
        coordinator branches on this when reading task/context and when
        writing back the team result, so the integration works against any
        of the state schemas StateGraph supports.
        """
        # Lazy imports to keep victor/teams from depending on pydantic at
        # module load time.
        from victor.framework.graph import CopyOnWriteState

        if isinstance(state, CopyOnWriteState):
            return "cow"

        try:
            from pydantic import BaseModel
        except ImportError:  # pragma: no cover - pydantic is a hard dep here
            return "dict"
        if isinstance(state, BaseModel):
            return "pydantic"
        return "dict"

    @staticmethod
    def _read_state(state: Any, kind: str, key: str, default: Any = None) -> Any:
        if kind == "pydantic":
            return getattr(state, key, default)
        if kind == "cow":
            return state.get(key, default)
        return state.get(key, default)

    @staticmethod
    def _build_call_context(
        state: Any, kind: str, *, exclude: set
    ) -> Dict[str, Any]:
        if kind == "pydantic":
            dump = state.model_dump()
            return {k: v for k, v in dump.items() if k not in exclude}
        if kind == "cow":
            return {k: state[k] for k in state.keys() if k not in exclude}
        return {k: v for k, v in state.items() if k not in exclude}

    @staticmethod
    def _apply_updates(state: Any, kind: str, updates: Dict[str, Any]) -> Any:
        """Write ``updates`` back into the original state container.

        - ``dict``: returns a new dict (caller's input is not mutated).
        - ``pydantic``: returns ``state.model_copy(update=updates)``. If the
          model rejects unknown fields, raises ``ValueError`` naming the
          offending key so users know to either widen ``model_config`` with
          ``extra='allow'`` or remap keys via ``StateGraphNodeConfig``.
        - ``cow``: assigns each key via ``__setitem__`` and returns the
          same wrapper, so the StateGraph executor sees the mutation.
        """
        if kind == "pydantic":
            # ``model_copy(update=...)`` in Pydantic v2 does not run validation,
            # so a strict (extra='forbid') model would silently drop unknown
            # keys. Detect that case up front and raise a clear error so users
            # know to either remap keys or widen ``model_config``.
            extra_policy = getattr(state.model_config, "get", lambda *_: None)("extra")
            if extra_policy is None and isinstance(state.model_config, dict):
                extra_policy = state.model_config.get("extra")
            declared = set(getattr(type(state), "model_fields", {}).keys())
            unknown = [k for k in updates if k not in declared]
            if unknown and extra_policy not in ("allow",):
                hint = (
                    "Either remap the keys via StateGraphNodeConfig "
                    "(e.g. result_key='context') or set "
                    "model_config = {'extra': 'allow'} on your state model."
                )
                raise ValueError(
                    f"Cannot write team result into Pydantic state of type "
                    f"{type(state).__name__}: fields {unknown} are not declared "
                    f"and the model does not allow extras. {hint}"
                )
            try:
                return state.model_copy(update=updates)
            except Exception as exc:  # pydantic.ValidationError on assignment
                raise ValueError(
                    f"Failed to write team result into Pydantic state of type "
                    f"{type(state).__name__}: {exc}. Consider remapping keys "
                    f"via StateGraphNodeConfig."
                ) from exc
        if kind == "cow":
            for k, v in updates.items():
                state[k] = v
            return state
        # dict
        new_state = dict(state)
        new_state.update(updates)
        return new_state

    async def __call__(self, state: Any) -> Any:
        """Execute as a StateGraph node.

        This makes the coordinator directly usable in StateGraph:
            graph.add_node("team", coordinator)

        The coordinator reads the task from ``state[task_key]`` (with a
        fallback to ``state[query_key]``), executes the team with the
        configured formation, and returns a *new* state dict with the
        results. The caller's input dict is never mutated.

        Use ``with_state_graph_config(...)`` to map the node onto a graph
        schema with different key names. For declarative workflow YAML with
        timeout/retry/merge-strategy configuration, use ``TeamStep`` from
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
        kind = self._classify_state(state)

        # Extract task using configured keys (task takes precedence over query).
        # ``None`` from a missing pydantic optional is treated like missing.
        task = self._read_state(state, kind, config.task_key, default=None)
        if not task:
            task = self._read_state(state, kind, config.query_key, default="")

        # Build context excluding task/query keys.
        context = self._build_call_context(
            state, kind, exclude={config.task_key, config.query_key}
        )
        strategy_state = state.get_state() if kind == "cow" else state
        if callable(config.formation_strategy):
            try:
                selected_formation = config.formation_strategy(strategy_state)
            except Exception as exc:
                logger.debug("Formation strategy failed; keeping default formation: %s", exc)
            else:
                if asyncio.iscoroutine(selected_formation):
                    close_coro = getattr(selected_formation, "close", None)
                    if callable(close_coro):
                        close_coro()
                    raise TypeError(
                        "StateGraphNodeConfig.formation_strategy must return a "
                        "TeamFormation synchronously; async strategies are not supported."
                    )
                if isinstance(selected_formation, TeamFormation):
                    context["formation_hint"] = selected_formation.value
                elif selected_formation is not None:
                    context["formation_hint"] = str(selected_formation)

        # Execute team task.
        result = await self.execute_task(task, context)

        # Compose updates and apply via the type-aware writer. dict inputs
        # are returned as a new dict (caller's state is not mutated), Pydantic
        # inputs are returned via model_copy, CoW wrappers are mutated via
        # __setitem__ so the executor sees the change.
        updates: Dict[str, Any] = {}
        if result.get("success"):
            updates[config.result_key] = result.get("final_output", "")
            updates[config.output_key] = result
        else:
            updates[config.error_key] = result.get("error", "Unknown error")
            updates[config.output_key] = result

        return self._apply_updates(state, kind, updates)
