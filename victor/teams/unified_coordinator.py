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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence

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
    MaterializedWorktreeAssignment,
    WorktreeAssignment,
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
            "task_summary",
            "summary",
            "result_summary",
            "validation_run",
            "validation_status",
            "validation_summary",
            "validation_command",
            "test_command",
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
        delegate_reentry_contract = self._extract_delegate_reentry_contract(context)
        effective_context = self._apply_delegate_reentry_context(
            context,
            delegate_reentry_contract=delegate_reentry_contract,
        )

        # Wrap team members with adapters
        shared_state_with_manager = self._active_shared_context()
        active_manager = self._active_manager()
        if active_manager is not None:
            shared_state_with_manager["explicit_manager_id"] = active_manager.id

        max_workers = self._extract_max_workers(effective_context, shared_state_with_manager)
        candidate_members = self._filter_execution_members(
            self._active_members(),
            member_ids=self._extract_delegate_reentry_member_ids(delegate_reentry_contract),
        )
        execution_members = self._limit_execution_members(
            candidate_members,
            active_formation,
            max_workers,
            manager=active_manager,
        )
        member_context_overrides = self._extract_delegate_reentry_member_context_overrides(
            delegate_reentry_contract
        )
        worktree_plan = None
        worktree_session = None
        if not member_context_overrides:
            worktree_plan = self._plan_worktree_execution(
                execution_members,
                context=effective_context,
                formation=active_formation,
            )
            worktree_session = self._materialize_worktree_plan(
                worktree_plan,
                context=effective_context,
            )
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
            _TeamMemberAdapter(m, effective_context, member_context_overrides.get(m.id))
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
            team_id=effective_context.get("team_name", "UnifiedTeam"),
            formation=active_formation.value,
            shared_state=shared_state_with_manager,
            **effective_context,
        )

        # Create AgentMessage for the task
        agent_task = AgentMessage(
            sender_id="coordinator",
            content=task,
            message_type=MessageType.TASK,
            data=effective_context,
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
            worker_return_contracts = self._build_worker_return_contracts(
                member_results,
                merge_analysis=merge_analysis,
            )
            if worker_return_contracts:
                result_dict["worker_return_contracts"] = worker_return_contracts
            merge_orchestration: Optional[Dict[str, Any]] = None
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
                    if self._should_execute_merge_orchestration(
                        effective_context,
                        merge_orchestration=merge_orchestration,
                    ):
                        merge_execution = self._execute_merge_orchestration(
                            worktree_session,
                            merge_analysis=merge_analysis.to_dict(),
                            context=effective_context,
                        )
                        if merge_execution is not None:
                            result_dict["merge_execution"] = merge_execution
            if worker_return_contracts:
                merge_review_contract = self._build_merge_review_contract(
                    worker_return_contracts,
                    merge_analysis=merge_analysis,
                    merge_orchestration=merge_orchestration,
                )
                if merge_review_contract:
                    result_dict["merge_review_contract"] = merge_review_contract
                    delegate_follow_up_contract = self._build_delegate_follow_up_contract(
                        worker_return_contracts,
                        merge_review_contract=merge_review_contract,
                        worktree_session=worktree_session,
                        merge_execution=result_dict.get("merge_execution"),
                        merge_analysis=result_dict.get("merge_analysis"),
                        merge_orchestration=merge_orchestration,
                        preserve_merge_follow_up=self._resolve_context_mode(effective_context)
                        == "delegate",
                    )
                    if delegate_follow_up_contract:
                        result_dict["delegate_follow_up_contract"] = delegate_follow_up_contract

            return result_dict
        finally:
            if worktree_session is not None:
                if self._should_cleanup_worktrees(effective_context, result_dict=result_dict):
                    cleanup_summary = self._cleanup_worktree_session(worktree_session)
                else:
                    cleanup_summary = self._build_preserved_worktree_cleanup_summary(
                        worktree_session,
                        reason="preserved_for_follow_up",
                    )
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

    @classmethod
    def _resolve_context_mode(cls, context: Dict[str, Any]) -> Optional[str]:
        for key in ("mode", "current_mode", "active_mode"):
            raw_value = context.get(key)
            if raw_value is None:
                continue
            value = str(raw_value).strip().lower()
            if value:
                return value
        return None

    def _materialize_worktree_plan(
        self,
        worktree_plan: Optional[WorktreeExecutionPlan],
        *,
        context: Dict[str, Any],
    ) -> Optional[WorktreeMaterializationSession]:
        if worktree_plan is None:
            return None
        if "materialize_worktrees" in context:
            materialize = self._coerce_context_flag(
                context,
                "materialize_worktrees",
                default=False,
            )
        else:
            materialize = bool(
                self._resolve_context_mode(context) == "delegate"
                and self._coerce_context_flag(context, "worktree_isolation", default=False)
            )
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

    @staticmethod
    def _coerce_optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _normalize_path_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        normalized: list[str] = []
        for item in list(value or []):
            text = cls._coerce_optional_text(item)
            if text is None:
                continue
            normalized.append(text.rstrip("/"))
        return list(dict.fromkeys(normalized))

    @classmethod
    def _normalize_path_map(cls, value: Any) -> Dict[str, List[str]]:
        if not isinstance(value, Mapping):
            return {}
        normalized: Dict[str, List[str]] = {}
        for member_id, paths in value.items():
            key = cls._coerce_optional_text(member_id)
            if key is None:
                continue
            normalized[key] = cls._normalize_path_list(paths)
        return normalized

    @classmethod
    def _normalize_member_id_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        normalized: list[str] = []
        for item in list(value or []):
            text = cls._coerce_optional_text(item)
            if text is None:
                continue
            normalized.append(text)
        return list(dict.fromkeys(normalized))

    @classmethod
    def _extract_delegate_reentry_contract(cls, context: Mapping[str, Any]) -> Dict[str, Any]:
        raw_value = context.get("delegate_reentry_contract")
        return dict(raw_value) if isinstance(raw_value, Mapping) else {}

    @classmethod
    def _extract_delegate_merge_contract(cls, context: Mapping[str, Any]) -> Dict[str, Any]:
        raw_value = context.get("delegate_merge_contract")
        return dict(raw_value) if isinstance(raw_value, Mapping) else {}

    @classmethod
    def _extract_delegate_follow_up_contract(cls, context: Mapping[str, Any]) -> Dict[str, Any]:
        raw_value = context.get("delegate_follow_up_contract")
        return dict(raw_value) if isinstance(raw_value, Mapping) else {}

    @classmethod
    def _extract_delegate_approval_contract(cls, context: Mapping[str, Any]) -> Dict[str, Any]:
        raw_value = context.get("delegate_approval_contract")
        if isinstance(raw_value, Mapping):
            return dict(raw_value)
        follow_up_contract = cls._extract_delegate_follow_up_contract(context)
        nested = follow_up_contract.get("approval_contract")
        return dict(nested) if isinstance(nested, Mapping) else {}

    @classmethod
    def _normalize_delegate_next_steps(cls, value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        normalized: List[Dict[str, Any]] = []
        seen_step_ids: Dict[str, int] = {}
        for index, item in enumerate(list(value), start=1):
            if not isinstance(item, Mapping):
                continue
            payload = dict(item)
            base_step_id = cls._coerce_optional_text(payload.get("step_id")) or cls._coerce_optional_text(
                payload.get("step")
            ) or f"delegate_step_{index}"
            occurrence = seen_step_ids.get(base_step_id, 0) + 1
            seen_step_ids[base_step_id] = occurrence
            payload["step_id"] = base_step_id if occurrence == 1 else f"{base_step_id}_{occurrence}"
            normalized.append(payload)
        return normalized

    @classmethod
    def _resolve_delegate_next_step_by_id(
        cls,
        context: Mapping[str, Any],
        *,
        step_id: str,
    ) -> Dict[str, Any]:
        approval_contract = cls._extract_delegate_approval_contract(context)
        next_steps = cls._normalize_delegate_next_steps(approval_contract.get("next_steps"))
        normalized_step_id = cls._coerce_optional_text(step_id)
        if normalized_step_id is None:
            return {}
        for step in next_steps:
            candidate_id = cls._coerce_optional_text(step.get("step_id"))
            candidate_step = cls._coerce_optional_text(step.get("step"))
            if normalized_step_id in {candidate_id, candidate_step}:
                return step
        return {}

    @classmethod
    def _extract_delegate_next_step(cls, context: Mapping[str, Any]) -> Dict[str, Any]:
        raw_value = context.get("delegate_next_step")
        if isinstance(raw_value, Mapping):
            return dict(raw_value)
        step_id = cls._coerce_optional_text(raw_value)
        if step_id is None:
            step_id = cls._coerce_optional_text(context.get("delegate_next_step_id"))
        if step_id is None:
            return {}
        return cls._resolve_delegate_next_step_by_id(context, step_id=step_id)

    @classmethod
    def _apply_delegate_reentry_context(
        cls,
        context: Dict[str, Any],
        *,
        delegate_reentry_contract: Mapping[str, Any],
    ) -> Dict[str, Any]:
        context_overrides = (
            dict(delegate_reentry_contract.get("context_overrides") or {})
            if isinstance(delegate_reentry_contract.get("context_overrides"), Mapping)
            else {}
        )
        effective_context = dict(context_overrides)
        effective_context.update(context)
        return effective_context

    @classmethod
    def _apply_delegate_next_step_context(
        cls,
        context: Dict[str, Any],
        *,
        delegate_next_step: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not delegate_next_step:
            return dict(context)
        selected_step = dict(context)
        step_name = cls._coerce_optional_text(delegate_next_step.get("step"))
        instruction = cls._coerce_optional_text(delegate_next_step.get("instruction"))
        if step_name is not None:
            selected_step.setdefault("delegate_selected_step", step_name)
        if instruction is not None:
            selected_step.setdefault("delegate_selected_instruction", instruction)

        resume_context = (
            dict(delegate_next_step.get("resume_context") or {})
            if isinstance(delegate_next_step.get("resume_context"), Mapping)
            else {}
        )
        execution_context = (
            dict(delegate_next_step.get("execution_context") or {})
            if isinstance(delegate_next_step.get("execution_context"), Mapping)
            else {}
        )
        effective_context = dict(resume_context)
        effective_context.update(execution_context)
        effective_context.update(selected_step)
        return effective_context

    @classmethod
    def _apply_delegate_merge_context(
        cls,
        context: Dict[str, Any],
        *,
        delegate_merge_contract: Mapping[str, Any],
    ) -> Dict[str, Any]:
        context_overrides = (
            dict(delegate_merge_contract.get("context_overrides") or {})
            if isinstance(delegate_merge_contract.get("context_overrides"), Mapping)
            else {}
        )
        effective_context = dict(context_overrides)
        effective_context.update(context)
        return effective_context

    @classmethod
    def _extract_delegate_reentry_member_ids(
        cls,
        delegate_reentry_contract: Mapping[str, Any],
    ) -> List[str]:
        member_ids = cls._normalize_member_id_list(delegate_reentry_contract.get("retry_member_ids"))
        if member_ids:
            return member_ids
        overrides = delegate_reentry_contract.get("resume_member_context_overrides")
        if isinstance(overrides, Mapping):
            return cls._normalize_member_id_list(overrides.keys())
        return []

    @classmethod
    def _extract_delegate_reentry_member_context_overrides(
        cls,
        delegate_reentry_contract: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        overrides = delegate_reentry_contract.get("resume_member_context_overrides")
        if not isinstance(overrides, Mapping):
            return {}
        next_action = cls._coerce_optional_text(delegate_reentry_contract.get("next_action"))
        retry_tasks_by_member = (
            dict(delegate_reentry_contract.get("retry_tasks_by_member") or {})
            if isinstance(delegate_reentry_contract.get("retry_tasks_by_member"), Mapping)
            else {}
        )
        normalized: Dict[str, Dict[str, Any]] = {}
        for member_id, payload in overrides.items():
            key = cls._coerce_optional_text(member_id)
            if key is None or not isinstance(payload, Mapping):
                continue
            override = dict(payload)
            if next_action is not None:
                override["delegate_reentry_next_action"] = next_action
            follow_up_task_brief = cls._coerce_optional_text(retry_tasks_by_member.get(key))
            if follow_up_task_brief is not None:
                override["follow_up_task_brief"] = follow_up_task_brief
            normalized[key] = override
        return normalized

    @classmethod
    def _filter_execution_members(
        cls,
        members: List["ITeamMember"],
        *,
        member_ids: List[str],
    ) -> List["ITeamMember"]:
        if not member_ids:
            return list(members)
        selected_ids = set(member_ids)
        filtered = [member for member in members if member.id in selected_ids]
        return filtered or list(members)

    @classmethod
    def _summarize_member_output(cls, output: str) -> Optional[str]:
        text = cls._coerce_optional_text(output)
        if text is None:
            return None
        compact = " ".join(text.split())
        if len(compact) <= 280:
            return compact
        return compact[:277].rstrip() + "..."

    @classmethod
    def _normalize_validation_run(cls, metadata: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        raw_validation = metadata.get("validation_run")
        payload = dict(raw_validation) if isinstance(raw_validation, Mapping) else {}
        status = cls._coerce_optional_text(
            payload.get("status")
            or payload.get("result")
            or payload.get("outcome")
            or metadata.get("validation_status")
        )
        command = cls._coerce_optional_text(
            payload.get("command")
            or payload.get("test_command")
            or metadata.get("validation_command")
            or metadata.get("test_command")
        )
        summary = cls._coerce_optional_text(
            payload.get("summary")
            or payload.get("output")
            or payload.get("result_summary")
            or metadata.get("validation_summary")
        )
        normalized: Dict[str, Any] = {}
        if status is not None:
            normalized["status"] = status
        if command is not None:
            normalized["command"] = command
        if summary is not None:
            normalized["summary"] = summary
        return normalized or None

    @classmethod
    def _build_worker_return_contracts(
        cls,
        member_results: Dict[str, MemberResult],
        *,
        merge_analysis: Optional[Any],
    ) -> Dict[str, Dict[str, Any]]:
        if hasattr(merge_analysis, "to_dict"):
            merge_payload = merge_analysis.to_dict()
        elif isinstance(merge_analysis, Mapping):
            merge_payload = dict(merge_analysis)
        else:
            merge_payload = {}

        member_changed_files = cls._normalize_path_map(merge_payload.get("member_changed_files"))
        out_of_scope_writes = cls._normalize_path_map(merge_payload.get("out_of_scope_writes"))
        readonly_violations = cls._normalize_path_map(merge_payload.get("readonly_violations"))
        overlapping_files_by_member: Dict[str, List[str]] = {}
        for conflict in list(merge_payload.get("overlapping_files") or []):
            if not isinstance(conflict, Mapping):
                continue
            path = cls._coerce_optional_text(conflict.get("path"))
            if path is None:
                continue
            for member_id in list(conflict.get("members") or []):
                normalized_member_id = cls._coerce_optional_text(member_id)
                if normalized_member_id is None:
                    continue
                overlapping_files_by_member.setdefault(normalized_member_id, []).append(path)

        contracts: Dict[str, Dict[str, Any]] = {}
        for member_id, result in member_results.items():
            normalized_member_id = cls._coerce_optional_text(member_id)
            if normalized_member_id is None:
                continue
            metadata = dict(result.metadata or {})
            changed_files = cls._normalize_path_list(
                metadata.get("changed_files")
                or metadata.get("files_touched")
                or metadata.get("modified_files")
                or member_changed_files.get(normalized_member_id)
            )
            overlap_paths = cls._normalize_path_list(
                overlapping_files_by_member.get(normalized_member_id)
            )
            out_of_scope_paths = cls._normalize_path_list(
                out_of_scope_writes.get(normalized_member_id)
            )
            readonly_paths = cls._normalize_path_list(readonly_violations.get(normalized_member_id))
            reasons: list[str] = []
            if overlap_paths:
                reasons.append("overlapping_files")
            if readonly_paths:
                reasons.append("readonly_violations")
            if out_of_scope_paths:
                reasons.append("out_of_scope_writes")
            risk_level = "low"
            if overlap_paths or readonly_paths:
                risk_level = "high"
            elif out_of_scope_paths:
                risk_level = "medium"

            task_summary = cls._coerce_optional_text(
                metadata.get("task_summary")
                or metadata.get("summary")
                or metadata.get("result_summary")
            ) or cls._summarize_member_output(result.output)

            contracts[normalized_member_id] = {
                "member_id": normalized_member_id,
                "success": bool(result.success),
                "task_summary": task_summary,
                "changed_files": changed_files,
                "validation_run": cls._normalize_validation_run(metadata),
                "merge_risk": {
                    "level": risk_level,
                    "reasons": reasons,
                    "overlapping_files": overlap_paths,
                    "out_of_scope_writes": out_of_scope_paths,
                    "readonly_violations": readonly_paths,
                },
            }

        return contracts

    @classmethod
    def _build_merge_review_contract(
        cls,
        worker_return_contracts: Mapping[str, Mapping[str, Any]],
        *,
        merge_analysis: Optional[Any],
        merge_orchestration: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if hasattr(merge_analysis, "to_dict"):
            merge_payload = merge_analysis.to_dict()
        elif isinstance(merge_analysis, Mapping):
            merge_payload = dict(merge_analysis)
        else:
            merge_payload = {}

        orchestration_payload = dict(merge_orchestration or {})
        review_required_members: list[str] = []
        validation_failed_members: list[str] = []
        blocking_issues: list[Dict[str, Any]] = []

        def add_review_member(member_id: Optional[str]) -> None:
            normalized = cls._coerce_optional_text(member_id)
            if normalized is None or normalized in review_required_members:
                return
            review_required_members.append(normalized)

        for member_id, contract in worker_return_contracts.items():
            normalized_member_id = cls._coerce_optional_text(member_id)
            if normalized_member_id is None:
                continue
            validation_run = (
                dict(contract.get("validation_run") or {})
                if isinstance(contract.get("validation_run"), Mapping)
                else {}
            )
            validation_status = cls._coerce_optional_text(validation_run.get("status"))
            normalized_status = validation_status.lower() if validation_status is not None else None
            if normalized_status not in (None, "passed", "pass", "success", "succeeded", "ok"):
                validation_failed_members.append(normalized_member_id)
                add_review_member(normalized_member_id)
                issue: Dict[str, Any] = {
                    "type": "validation_failed",
                    "member_id": normalized_member_id,
                }
                if validation_status is not None:
                    issue["status"] = validation_status
                if cls._coerce_optional_text(validation_run.get("summary")) is not None:
                    issue["summary"] = str(validation_run.get("summary"))
                if cls._coerce_optional_text(validation_run.get("command")) is not None:
                    issue["command"] = str(validation_run.get("command"))
                blocking_issues.append(issue)

            merge_risk = (
                dict(contract.get("merge_risk") or {})
                if isinstance(contract.get("merge_risk"), Mapping)
                else {}
            )
            risk_level = cls._coerce_optional_text(merge_risk.get("level")) or "low"
            if risk_level in {"medium", "high"}:
                add_review_member(normalized_member_id)
                blocking_issues.append(
                    {
                        "type": f"merge_risk_{risk_level}",
                        "member_id": normalized_member_id,
                        "reasons": list(merge_risk.get("reasons") or []),
                    }
                )

        recommended_merge_order = (
            list(orchestration_payload.get("recommended_merge_order") or [])
            or list(merge_payload.get("recommended_merge_order") or [])
            or list(worker_return_contracts.keys())
        )
        merge_risk_level = cls._coerce_optional_text(
            orchestration_payload.get("merge_risk_level") or merge_payload.get("risk_level")
        )
        if "merge_execution_eligible" in orchestration_payload:
            merge_execution_eligible = bool(orchestration_payload.get("merge_execution_eligible"))
        else:
            merge_execution_eligible = merge_risk_level in (None, "low")
        merge_ready = bool(merge_execution_eligible and not blocking_issues)
        if (
            not merge_ready
            and not review_required_members
            and (
                not merge_execution_eligible
                or merge_risk_level not in (None, "low")
            )
        ):
            for member_id in recommended_merge_order:
                add_review_member(member_id)
        if merge_ready:
            next_action = "merge"
        elif validation_failed_members:
            next_action = "fix_validation"
        elif review_required_members:
            next_action = "review"
        else:
            next_action = "inspect"

        return {
            "merge_ready": merge_ready,
            "review_required": bool(review_required_members) or not merge_ready,
            "recommended_merge_order": recommended_merge_order,
            "review_required_members": review_required_members,
            "validation_failed_members": validation_failed_members,
            "blocking_issues": blocking_issues,
            "merge_risk_level": merge_risk_level,
            "merge_execution_eligible": merge_execution_eligible,
            "recommended_mode": orchestration_payload.get("recommended_mode"),
            "next_action": next_action,
        }

    @classmethod
    def _build_delegate_follow_up_contract(
        cls,
        worker_return_contracts: Mapping[str, Mapping[str, Any]],
        *,
        merge_review_contract: Mapping[str, Any],
        worktree_session: Optional[WorktreeMaterializationSession],
        merge_execution: Optional[Mapping[str, Any]] = None,
        merge_analysis: Optional[Mapping[str, Any]] = None,
        merge_orchestration: Optional[Mapping[str, Any]] = None,
        preserve_merge_follow_up: bool = False,
    ) -> Dict[str, Any]:
        next_action = cls._coerce_optional_text(merge_review_contract.get("next_action")) or "inspect"
        merge_execution_payload = (
            dict(merge_execution or {}) if isinstance(merge_execution, Mapping) else {}
        )
        merge_executed = bool(merge_execution_payload.get("executed", False))
        validation_failed_members = cls._normalize_path_list(
            merge_review_contract.get("validation_failed_members")
        )
        review_required_members = cls._normalize_path_list(
            merge_review_contract.get("review_required_members")
        )

        fix_validation_queue: list[Dict[str, Any]] = []
        for member_id in validation_failed_members:
            contract = (
                dict(worker_return_contracts.get(member_id) or {})
                if isinstance(worker_return_contracts.get(member_id), Mapping)
                else {}
            )
            validation_run = (
                dict(contract.get("validation_run") or {})
                if isinstance(contract.get("validation_run"), Mapping)
                else {}
            )
            fix_validation_queue.append(
                {
                    "member_id": member_id,
                    "validation_command": cls._coerce_optional_text(validation_run.get("command")),
                    "validation_summary": cls._coerce_optional_text(validation_run.get("summary")),
                    "changed_files": cls._normalize_path_list(contract.get("changed_files")),
                }
            )

        review_queue: list[Dict[str, Any]] = []
        for member_id in review_required_members:
            contract = (
                dict(worker_return_contracts.get(member_id) or {})
                if isinstance(worker_return_contracts.get(member_id), Mapping)
                else {}
            )
            merge_risk = (
                dict(contract.get("merge_risk") or {})
                if isinstance(contract.get("merge_risk"), Mapping)
                else {}
            )
            review_queue.append(
                {
                    "member_id": member_id,
                    "merge_risk_level": cls._coerce_optional_text(merge_risk.get("level")) or "low",
                    "merge_risk_reasons": list(merge_risk.get("reasons") or []),
                    "changed_files": cls._normalize_path_list(contract.get("changed_files")),
                    "task_summary": cls._coerce_optional_text(contract.get("task_summary")),
                }
            )

        preserve_worktrees = bool(
            worktree_session is not None
            and (
                next_action in {"fix_validation", "review", "inspect"}
                or (preserve_merge_follow_up and next_action == "merge" and not merge_executed)
            )
        )
        preserved_worktree_paths: list[str] = []
        if preserve_worktrees and worktree_session is not None:
            assignments = getattr(worktree_session, "assignments", [])
            for assignment in list(assignments or []):
                path = cls._coerce_optional_text(getattr(assignment, "worktree_path", None))
                if path is not None:
                    preserved_worktree_paths.append(path)

        reentry_contract: Optional[Dict[str, Any]] = None
        merge_execution_contract: Optional[Dict[str, Any]] = None
        if preserve_worktrees and worktree_session is not None:
            if next_action == "merge" and not merge_executed:
                merge_execution_contract = cls._build_delegate_merge_execution_contract(
                    worktree_session=worktree_session,
                    merge_analysis=merge_analysis,
                    merge_orchestration=merge_orchestration,
                    merge_review_contract=merge_review_contract,
                    worker_return_contracts=worker_return_contracts,
                )
            else:
                retry_member_ids = cls._resolve_delegate_reentry_member_ids(
                    next_action=next_action,
                    validation_failed_members=validation_failed_members,
                    review_required_members=review_required_members,
                )
                retry_tasks_by_member = cls._build_delegate_reentry_retry_tasks(
                    next_action=next_action,
                    fix_validation_queue=fix_validation_queue,
                    review_queue=review_queue,
                )
                resume_member_context_overrides: Dict[str, Dict[str, Any]] = {}
                resume_worktree_paths: Dict[str, str] = {}
                for member_id in retry_member_ids:
                    assignment = worktree_session.assignment_for(member_id)
                    if assignment is None:
                        continue
                    override = cls._build_delegate_reentry_member_context_override(assignment)
                    if not override:
                        continue
                    resume_member_context_overrides[member_id] = override
                    path = cls._coerce_optional_text(override.get("worktree_path"))
                    if path is not None:
                        resume_worktree_paths[member_id] = path
                if resume_member_context_overrides:
                    reentry_contract = {
                        "mode": "delegate",
                        "next_action": next_action,
                        "retry_member_ids": retry_member_ids,
                        "resume_worktree_paths": resume_worktree_paths,
                        "retry_tasks_by_member": retry_tasks_by_member,
                        "resume_member_context_overrides": resume_member_context_overrides,
                        "context_overrides": {
                            "mode": "delegate",
                            "worktree_isolation": True,
                            "materialize_worktrees": False,
                            "cleanup_worktrees": False,
                        },
                    }

        contract = {
            "next_action": next_action,
            "preserve_worktrees": preserve_worktrees,
            "fix_validation_queue": fix_validation_queue,
            "review_queue": review_queue,
            "review_required_members": review_required_members,
            "validation_failed_members": validation_failed_members,
            "preserved_worktree_paths": preserved_worktree_paths,
            "approval_contract": cls._build_delegate_approval_contract(
                next_action=next_action,
                merge_review_contract=merge_review_contract,
                reentry_contract=reentry_contract,
                merge_execution_contract=merge_execution_contract,
                fix_validation_queue=fix_validation_queue,
                review_queue=review_queue,
                merge_execution=merge_execution,
            ),
        }
        if reentry_contract is not None:
            contract["reentry_contract"] = reentry_contract
        if merge_execution_contract is not None:
            contract["merge_execution_contract"] = merge_execution_contract
        return contract

    @classmethod
    def _build_delegate_approval_contract(
        cls,
        *,
        next_action: str,
        merge_review_contract: Mapping[str, Any],
        reentry_contract: Optional[Mapping[str, Any]],
        merge_execution_contract: Optional[Mapping[str, Any]],
        fix_validation_queue: List[Dict[str, Any]],
        review_queue: List[Dict[str, Any]],
        merge_execution: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        merge_execution_payload = (
            dict(merge_execution or {}) if isinstance(merge_execution, Mapping) else {}
        )
        reentry_payload = dict(reentry_contract or {}) if isinstance(reentry_contract, Mapping) else {}
        retry_member_ids = cls._normalize_member_id_list(reentry_payload.get("retry_member_ids"))
        recommended_merge_order = cls._normalize_member_id_list(
            merge_review_contract.get("recommended_merge_order")
        )
        validation_target_ids = cls._normalize_member_id_list(
            item.get("member_id") for item in fix_validation_queue if isinstance(item, Mapping)
        )
        review_target_ids = cls._normalize_member_id_list(
            item.get("member_id") for item in review_queue if isinstance(item, Mapping)
        )
        merge_executed = bool(merge_execution_payload.get("executed"))
        if not merge_executed:
            merge_status = cls._coerce_optional_text(merge_execution_payload.get("status"))
            if merge_status is not None:
                merge_executed = merge_status.lower() in {
                    "success",
                    "succeeded",
                    "merged",
                    "completed",
                }
        recommended_mode = cls._coerce_optional_text(merge_review_contract.get("recommended_mode"))
        if recommended_mode is None:
            if merge_executed or bool(merge_review_contract.get("merge_execution_eligible")):
                recommended_mode = "auto_apply_safe"
            elif next_action in {"fix_validation", "review", "inspect"}:
                recommended_mode = "manual_review"

        if merge_executed:
            target_member_ids = recommended_merge_order or retry_member_ids
            return cls._finalize_delegate_approval_contract(
                {
                    "required": False,
                    "reason": "merge_executed",
                    "recommended_action": "merged",
                    "recommended_mode": recommended_mode,
                    "resume_ready": False,
                    "auto_retry_eligible": False,
                    "merge_executed": True,
                    "target_member_ids": target_member_ids,
                    "summary": cls._build_delegate_approval_summary(
                        "Merge orchestration already executed for",
                        target_member_ids=target_member_ids,
                    ),
                },
                reentry_payload=reentry_payload,
                merge_execution_contract=merge_execution_contract,
            )

        if next_action == "fix_validation":
            target_member_ids = retry_member_ids or validation_target_ids
            resume_ready = bool(reentry_payload) and bool(target_member_ids)
            return cls._finalize_delegate_approval_contract(
                {
                    "required": not resume_ready,
                    "reason": "validation_failed",
                    "recommended_action": "retry" if resume_ready else "approve_retry",
                    "recommended_mode": recommended_mode,
                    "resume_ready": resume_ready,
                    "auto_retry_eligible": resume_ready,
                    "merge_executed": False,
                    "target_member_ids": target_member_ids,
                    "summary": cls._build_delegate_approval_summary(
                        (
                            "Resume preserved worktrees to fix failing validation for"
                            if resume_ready
                            else "Approve a validation retry for"
                        ),
                        target_member_ids=target_member_ids,
                    ),
                },
                reentry_payload=reentry_payload,
                merge_execution_contract=merge_execution_contract,
            )

        if next_action == "review":
            target_member_ids = retry_member_ids or review_target_ids
            return cls._finalize_delegate_approval_contract(
                {
                    "required": True,
                    "reason": "review_required",
                    "recommended_action": "review_then_retry",
                    "recommended_mode": recommended_mode,
                    "resume_ready": bool(reentry_payload),
                    "auto_retry_eligible": False,
                    "merge_executed": False,
                    "target_member_ids": target_member_ids,
                    "summary": cls._build_delegate_approval_summary(
                        "Review merge risks before retrying preserved worktrees for",
                        target_member_ids=target_member_ids,
                    ),
                },
                reentry_payload=reentry_payload,
                merge_execution_contract=merge_execution_contract,
            )

        if next_action == "merge":
            target_member_ids = recommended_merge_order or retry_member_ids
            return cls._finalize_delegate_approval_contract(
                {
                    "required": True,
                    "reason": "merge_ready",
                    "recommended_action": "approve_merge",
                    "recommended_mode": recommended_mode,
                    "resume_ready": False,
                    "auto_retry_eligible": False,
                    "merge_executed": False,
                    "target_member_ids": target_member_ids,
                    "summary": cls._build_delegate_approval_summary(
                        "Review and approve merge execution for",
                        target_member_ids=target_member_ids,
                    ),
                },
                reentry_payload=reentry_payload,
                merge_execution_contract=merge_execution_contract,
            )

        target_member_ids = retry_member_ids or list(
            dict.fromkeys([*validation_target_ids, *review_target_ids])
        )
        return cls._finalize_delegate_approval_contract(
            {
                "required": True,
                "reason": "inspect_required",
                "recommended_action": "inspect_worktrees",
                "recommended_mode": recommended_mode,
                "resume_ready": bool(reentry_payload),
                "auto_retry_eligible": False,
                "merge_executed": False,
                "target_member_ids": target_member_ids,
                "summary": cls._build_delegate_approval_summary(
                    "Inspect preserved worktrees before retrying work for",
                    target_member_ids=target_member_ids,
                ),
            },
            reentry_payload=reentry_payload,
            merge_execution_contract=merge_execution_contract,
        )

    @classmethod
    def _finalize_delegate_approval_contract(
        cls,
        contract: Mapping[str, Any],
        *,
        reentry_payload: Mapping[str, Any],
        merge_execution_contract: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        approval_contract = dict(contract)
        target_member_ids = cls._normalize_member_id_list(approval_contract.get("target_member_ids"))
        resume_context = cls._build_delegate_approval_resume_context(
            reentry_payload,
            target_member_ids=target_member_ids,
        )
        if resume_context is not None:
            approval_contract["resume_context"] = resume_context
        task_briefs = cls._build_delegate_approval_task_briefs(
            reentry_payload,
            target_member_ids=target_member_ids,
        )
        if task_briefs:
            approval_contract["task_briefs_by_member"] = task_briefs
        if "next_steps" not in approval_contract:
            next_steps = cls._build_delegate_approval_next_steps(
                approval_contract,
                target_member_ids=target_member_ids,
                resume_context=resume_context,
                task_briefs=task_briefs,
                merge_execution_contract=merge_execution_contract,
            )
            if next_steps:
                approval_contract["next_steps"] = next_steps
        normalized_next_steps = cls._normalize_delegate_next_steps(approval_contract.get("next_steps"))
        if normalized_next_steps:
            approval_contract["next_steps"] = normalized_next_steps
        return approval_contract

    @classmethod
    def _build_delegate_approval_resume_context(
        cls,
        reentry_payload: Mapping[str, Any],
        *,
        target_member_ids: List[str],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(reentry_payload, Mapping) or not reentry_payload:
            return None
        normalized_payload = dict(reentry_payload)
        retry_member_ids = cls._normalize_member_id_list(normalized_payload.get("retry_member_ids"))
        resume_overrides = normalized_payload.get("resume_member_context_overrides")
        resume_paths = normalized_payload.get("resume_worktree_paths")
        has_resume_details = bool(retry_member_ids)
        if isinstance(resume_overrides, Mapping):
            has_resume_details = has_resume_details or bool(resume_overrides)
        if isinstance(resume_paths, Mapping):
            has_resume_details = has_resume_details or bool(resume_paths)
        if not has_resume_details:
            return None
        if not retry_member_ids and target_member_ids:
            normalized_payload["retry_member_ids"] = list(target_member_ids)
        return {
            "mode": "delegate",
            "delegate_reentry_contract": normalized_payload,
        }

    @classmethod
    def _build_delegate_approval_task_briefs(
        cls,
        reentry_payload: Mapping[str, Any],
        *,
        target_member_ids: List[str],
    ) -> Dict[str, str]:
        if not isinstance(reentry_payload, Mapping):
            return {}
        raw_briefs = reentry_payload.get("retry_tasks_by_member")
        if not isinstance(raw_briefs, Mapping):
            return {}
        prioritized_ids = cls._normalize_member_id_list(target_member_ids)
        if not prioritized_ids:
            prioritized_ids = cls._normalize_member_id_list(raw_briefs.keys())
        task_briefs: Dict[str, str] = {}
        for member_id in prioritized_ids:
            task_brief = cls._coerce_optional_text(raw_briefs.get(member_id))
            if task_brief is None:
                continue
            task_briefs[member_id] = task_brief
        return task_briefs

    @classmethod
    def _build_delegate_approval_summary(
        cls,
        prefix: str,
        *,
        target_member_ids: List[str],
    ) -> str:
        normalized_targets = cls._normalize_member_id_list(target_member_ids)
        if normalized_targets:
            return f"{prefix}: {', '.join(normalized_targets)}."
        return f"{prefix} the delegate worktree set."

    @classmethod
    def _build_delegate_approval_next_steps(
        cls,
        approval_contract: Mapping[str, Any],
        *,
        target_member_ids: List[str],
        resume_context: Optional[Mapping[str, Any]],
        task_briefs: Mapping[str, str],
        merge_execution_contract: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        recommended_action = cls._coerce_optional_text(approval_contract.get("recommended_action"))
        summary = cls._coerce_optional_text(approval_contract.get("summary"))
        requires_approval = bool(approval_contract.get("required", False))
        normalized_targets = cls._normalize_member_id_list(target_member_ids)

        def build_step(
            step: str,
            instruction: Optional[str],
            *,
            step_requires_approval: bool,
            include_resume: bool = False,
        ) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "step": step,
                "instruction": instruction
                or cls._build_delegate_approval_summary(
                    "Continue delegate follow-up for",
                    target_member_ids=normalized_targets,
                ),
                "target_member_ids": list(normalized_targets),
                "requires_approval": step_requires_approval,
            }
            if include_resume and resume_context:
                payload["resume_context"] = dict(resume_context)
            if include_resume and task_briefs:
                payload["task_briefs_by_member"] = dict(task_briefs)
            return payload

        if recommended_action == "merged" or bool(approval_contract.get("merge_executed", False)):
            return [
                build_step(
                    "status_merged",
                    summary,
                    step_requires_approval=False,
                )
            ]
        if recommended_action == "retry":
            return [
                build_step(
                    "resume_delegate_retry",
                    summary,
                    step_requires_approval=False,
                    include_resume=True,
                )
            ]
        if recommended_action == "approve_retry":
            return [
                build_step(
                    "approve_delegate_retry",
                    summary,
                    step_requires_approval=True,
                )
            ]
        if recommended_action == "review_then_retry":
            steps = [
                build_step(
                    "review_worktrees",
                    summary,
                    step_requires_approval=True,
                )
            ]
            if resume_context:
                steps.append(
                    build_step(
                        "resume_delegate_retry",
                        cls._build_delegate_approval_summary(
                            "Resume preserved worktrees after review for",
                            target_member_ids=normalized_targets,
                        ),
                        step_requires_approval=requires_approval,
                        include_resume=True,
                    )
                )
            return steps
        if recommended_action == "approve_merge":
            step = build_step(
                "approve_merge_execution",
                summary,
                step_requires_approval=True,
            )
            if merge_execution_contract:
                step["execution_context"] = {
                    "mode": "delegate",
                    "delegate_merge_contract": dict(merge_execution_contract),
                }
            return [step]
        if recommended_action == "inspect_worktrees":
            steps = [
                build_step(
                    "inspect_worktrees",
                    summary,
                    step_requires_approval=True,
                )
            ]
            if resume_context:
                steps.append(
                    build_step(
                        "resume_delegate_retry",
                        cls._build_delegate_approval_summary(
                            "Resume preserved worktrees after inspection for",
                            target_member_ids=normalized_targets,
                        ),
                        step_requires_approval=requires_approval,
                        include_resume=True,
                    )
                )
            return steps
        return []

    @classmethod
    def _resolve_delegate_reentry_member_ids(
        cls,
        *,
        next_action: str,
        validation_failed_members: List[str],
        review_required_members: List[str],
    ) -> List[str]:
        if next_action == "fix_validation":
            return list(validation_failed_members)
        if next_action == "review":
            return list(review_required_members)
        combined = list(dict.fromkeys([*validation_failed_members, *review_required_members]))
        return combined

    @classmethod
    def _build_delegate_reentry_retry_tasks(
        cls,
        *,
        next_action: str,
        fix_validation_queue: List[Dict[str, Any]],
        review_queue: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        retry_tasks: Dict[str, str] = {}
        for item in fix_validation_queue:
            if not isinstance(item, Mapping):
                continue
            member_id = cls._coerce_optional_text(item.get("member_id"))
            if member_id is None:
                continue
            command = cls._coerce_optional_text(item.get("validation_command"))
            summary = cls._coerce_optional_text(item.get("validation_summary"))
            changed_files = cls._normalize_path_list(item.get("changed_files"))
            parts = [f"Fix the failing validation run for {member_id}."]
            if command is not None:
                parts.append(f"Re-run `{command}`.")
            if summary is not None:
                parts.append(f"Last result: {summary}.")
            if changed_files:
                parts.append(f"Focus on: {', '.join(changed_files)}.")
            retry_tasks[member_id] = " ".join(parts)

        for item in review_queue:
            if not isinstance(item, Mapping):
                continue
            member_id = cls._coerce_optional_text(item.get("member_id"))
            if member_id is None or member_id in retry_tasks:
                continue
            risk_level = cls._coerce_optional_text(item.get("merge_risk_level")) or "low"
            reasons = cls._normalize_path_list(item.get("merge_risk_reasons"))
            changed_files = cls._normalize_path_list(item.get("changed_files"))
            task_summary = cls._coerce_optional_text(item.get("task_summary"))
            parts = [f"Review the pending merge risk for {member_id} ({risk_level})."]
            if reasons:
                parts.append(f"Address: {', '.join(reasons)}.")
            if changed_files:
                parts.append(f"Inspect: {', '.join(changed_files)}.")
            if task_summary is not None:
                parts.append(f"Prior output: {task_summary}.")
            retry_tasks[member_id] = " ".join(parts)

        if not retry_tasks and next_action == "inspect":
            for item in review_queue:
                if not isinstance(item, Mapping):
                    continue
                member_id = cls._coerce_optional_text(item.get("member_id"))
                if member_id is None:
                    continue
                retry_tasks[member_id] = f"Inspect the preserved worktree state for {member_id}."
        return retry_tasks

    @classmethod
    def _build_delegate_reentry_member_context_override(
        cls,
        assignment: Any,
    ) -> Dict[str, Any]:
        override = {}
        to_context_overrides = getattr(assignment, "to_context_overrides", None)
        if callable(to_context_overrides):
            raw_override = to_context_overrides()
            if isinstance(raw_override, Mapping):
                override.update(dict(raw_override))

        worktree_path = cls._coerce_optional_text(getattr(assignment, "worktree_path", None))
        branch_name = cls._coerce_optional_text(getattr(assignment, "branch_name", None))
        if worktree_path is not None:
            override.setdefault("workspace_root", worktree_path)
            override["worktree_path"] = worktree_path
        if branch_name is not None:
            override["branch_name"] = branch_name

        assignment_payload: Dict[str, Any] = {}
        to_dict = getattr(assignment, "to_dict", None)
        if callable(to_dict):
            raw_payload = to_dict()
            if isinstance(raw_payload, Mapping):
                assignment_payload.update(dict(raw_payload))
        if not assignment_payload:
            member_id = cls._coerce_optional_text(getattr(assignment, "member_id", None))
            if member_id is not None:
                assignment_payload["member_id"] = member_id
            if branch_name is not None:
                assignment_payload["branch_name"] = branch_name
            if worktree_path is not None:
                assignment_payload["worktree_path"] = worktree_path
        if assignment_payload:
            override["worktree_assignment"] = assignment_payload

        return override

    @classmethod
    def _rebuild_worktree_assignment(
        cls,
        payload: Mapping[str, Any],
    ) -> Optional[WorktreeAssignment]:
        member_id = cls._coerce_optional_text(payload.get("member_id"))
        branch_name = cls._coerce_optional_text(payload.get("branch_name"))
        worktree_name = cls._coerce_optional_text(payload.get("worktree_name"))
        worktree_path = cls._coerce_optional_text(payload.get("worktree_path"))
        if None in (member_id, branch_name, worktree_name, worktree_path):
            return None
        return WorktreeAssignment(
            member_id=member_id,
            branch_name=branch_name,
            worktree_name=worktree_name,
            worktree_path=worktree_path,
            claimed_paths=tuple(cls._normalize_path_list(payload.get("claimed_paths"))),
            readonly_paths=tuple(cls._normalize_path_list(payload.get("readonly_paths"))),
            merge_priority=int(payload.get("merge_priority") or 0),
            metadata=dict(payload.get("metadata") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {},
        )

    @classmethod
    def _rebuild_worktree_plan(
        cls,
        payload: Mapping[str, Any],
    ) -> Optional[WorktreeExecutionPlan]:
        team_name = cls._coerce_optional_text(payload.get("team_name"))
        repo_root = cls._coerce_optional_text(payload.get("repo_root"))
        parent_dir = cls._coerce_optional_text(payload.get("parent_dir"))
        base_ref = cls._coerce_optional_text(payload.get("base_ref"))
        branch_prefix = cls._coerce_optional_text(payload.get("branch_prefix"))
        formation_name = cls._coerce_optional_text(payload.get("formation")) or TeamFormation.SEQUENTIAL.value
        if None in (team_name, repo_root, parent_dir, base_ref, branch_prefix):
            return None
        try:
            formation = TeamFormation(formation_name)
        except ValueError:
            formation = TeamFormation.SEQUENTIAL
        assignments = tuple(
            assignment
            for assignment in (
                cls._rebuild_worktree_assignment(item)
                for item in list(payload.get("assignments") or [])
                if isinstance(item, Mapping)
            )
            if assignment is not None
        )
        return WorktreeExecutionPlan(
            team_name=team_name,
            repo_root=repo_root,
            parent_dir=parent_dir,
            base_ref=base_ref,
            branch_prefix=branch_prefix,
            formation=formation,
            assignments=assignments,
            merge_order=tuple(cls._normalize_member_id_list(payload.get("merge_order"))),
            shared_readonly_paths=tuple(cls._normalize_path_list(payload.get("shared_readonly_paths"))),
            rationale=cls._coerce_optional_text(payload.get("rationale")),
            metadata=dict(payload.get("metadata") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {},
        )

    @classmethod
    def _rebuild_materialized_worktree_assignment(
        cls,
        payload: Mapping[str, Any],
    ) -> Optional[MaterializedWorktreeAssignment]:
        assignment = cls._rebuild_worktree_assignment(payload)
        if assignment is None:
            return None
        runtime_metadata = payload.get("runtime_metadata")
        return MaterializedWorktreeAssignment(
            assignment=assignment,
            materialized=bool(payload.get("materialized", False)),
            cleanup_required=bool(payload.get("cleanup_required", False)),
            metadata=dict(runtime_metadata or {}) if isinstance(runtime_metadata, Mapping) else {},
        )

    @classmethod
    def _rebuild_worktree_session(
        cls,
        payload: Mapping[str, Any],
    ) -> Optional[WorktreeMaterializationSession]:
        plan = cls._rebuild_worktree_plan(
            dict(payload.get("plan") or {}) if isinstance(payload.get("plan"), Mapping) else {}
        )
        if plan is None:
            return None
        assignments = tuple(
            assignment
            for assignment in (
                cls._rebuild_materialized_worktree_assignment(item)
                for item in list(payload.get("assignments") or [])
                if isinstance(item, Mapping)
            )
            if assignment is not None
        )
        metadata = payload.get("metadata")
        return WorktreeMaterializationSession(
            plan=plan,
            assignments=assignments,
            materialized=bool(payload.get("materialized", False)),
            dry_run=bool(payload.get("dry_run", False)),
            metadata=dict(metadata or {}) if isinstance(metadata, Mapping) else {},
        )

    @classmethod
    def _build_delegate_merge_execution_contract(
        cls,
        *,
        worktree_session: WorktreeMaterializationSession,
        merge_analysis: Optional[Mapping[str, Any]],
        merge_orchestration: Optional[Mapping[str, Any]],
        merge_review_contract: Mapping[str, Any],
        worker_return_contracts: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "mode": "delegate",
            "next_action": "merge",
            "worktree_session": worktree_session.to_dict(),
            "merge_analysis": dict(merge_analysis or {}),
            "merge_orchestration": dict(merge_orchestration or {}),
            "merge_review_contract": dict(merge_review_contract or {}),
            "worker_return_contracts": {
                member_id: dict(contract)
                for member_id, contract in worker_return_contracts.items()
                if isinstance(contract, Mapping)
            },
            "context_overrides": {
                "mode": "delegate",
                "worktree_isolation": True,
                "materialize_worktrees": False,
                "cleanup_worktrees": True,
            },
        }

    def _execute_delegate_merge_contract(
        self,
        task: str,
        context: Dict[str, Any],
        *,
        formation: TeamFormation,
        delegate_merge_contract: Mapping[str, Any],
    ) -> Dict[str, Any]:
        effective_context = self._apply_delegate_merge_context(
            context,
            delegate_merge_contract=delegate_merge_contract,
        )
        session_payload = (
            dict(delegate_merge_contract.get("worktree_session") or {})
            if isinstance(delegate_merge_contract.get("worktree_session"), Mapping)
            else {}
        )
        worktree_session = self._rebuild_worktree_session(session_payload)
        if worktree_session is None:
            return {
                "success": False,
                "error": "Invalid delegate merge contract",
                "member_results": {},
                "final_output": "",
                "formation": formation.value,
            }

        merge_analysis = (
            dict(delegate_merge_contract.get("merge_analysis") or {})
            if isinstance(delegate_merge_contract.get("merge_analysis"), Mapping)
            else {}
        )
        merge_orchestration = (
            dict(delegate_merge_contract.get("merge_orchestration") or {})
            if isinstance(delegate_merge_contract.get("merge_orchestration"), Mapping)
            else {}
        )
        merge_review_contract = (
            dict(delegate_merge_contract.get("merge_review_contract") or {})
            if isinstance(delegate_merge_contract.get("merge_review_contract"), Mapping)
            else {}
        )
        worker_return_contracts = {
            member_id: dict(contract)
            for member_id, contract in (
                dict(delegate_merge_contract.get("worker_return_contracts") or {})
                if isinstance(delegate_merge_contract.get("worker_return_contracts"), Mapping)
                else {}
            ).items()
            if isinstance(contract, Mapping)
        }
        merge_execution = self._execute_merge_orchestration(
            worktree_session,
            merge_analysis=merge_analysis or None,
            context=effective_context,
        ) or {
            "status": "blocked",
            "executed": False,
            "blocked_reason": "merge_execution_unavailable",
        }

        result: Dict[str, Any] = {
            "success": bool(merge_execution.get("executed", False)),
            "member_results": {},
            "final_output": (
                "Approved merge orchestration executed."
                if merge_execution.get("executed", False)
                else "Approved merge orchestration did not execute."
            ),
            "formation": formation.value,
            "total_tool_calls": 0,
            "communication_log": list(self._active_message_history()),
            "shared_context": dict(self._active_shared_context()),
            "worktree_session": worktree_session.to_dict(),
            "merge_execution": merge_execution,
        }
        if merge_analysis:
            result["merge_analysis"] = dict(merge_analysis)
        if merge_orchestration:
            result["merge_orchestration"] = dict(merge_orchestration)
        else:
            built_orchestration = self._build_merge_orchestration(
                worktree_session,
                merge_analysis=merge_analysis or None,
            )
            if built_orchestration is not None:
                merge_orchestration = built_orchestration
                result["merge_orchestration"] = built_orchestration
        if merge_review_contract:
            result["merge_review_contract"] = dict(merge_review_contract)
        if worker_return_contracts:
            result["worker_return_contracts"] = dict(worker_return_contracts)
        if worker_return_contracts and merge_review_contract:
            delegate_follow_up_contract = self._build_delegate_follow_up_contract(
                worker_return_contracts,
                merge_review_contract=merge_review_contract,
                worktree_session=worktree_session,
                merge_execution=merge_execution,
                merge_analysis=merge_analysis,
                merge_orchestration=merge_orchestration,
                preserve_merge_follow_up=self._resolve_context_mode(effective_context) == "delegate",
            )
            if delegate_follow_up_contract:
                result["delegate_follow_up_contract"] = delegate_follow_up_contract

        if self._should_cleanup_worktrees(effective_context, result_dict=result):
            cleanup_summary = self._cleanup_worktree_session(worktree_session)
        else:
            cleanup_summary = self._build_preserved_worktree_cleanup_summary(
                worktree_session,
                reason="preserved_for_follow_up",
            )
        result["worktree_cleanup"] = cleanup_summary
        return result

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

    def _should_execute_merge_orchestration(
        self,
        context: Dict[str, Any],
        *,
        merge_orchestration: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if "auto_merge_worktrees" in context:
            return self._coerce_context_flag(context, "auto_merge_worktrees", default=False)
        if self._resolve_context_mode(context) != "delegate":
            return False
        if not self._coerce_context_flag(context, "worktree_isolation", default=False):
            return False
        orchestration_payload = dict(merge_orchestration or {})
        return bool(orchestration_payload.get("merge_execution_eligible"))

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

    def _should_cleanup_worktrees(
        self,
        context: Dict[str, Any],
        *,
        result_dict: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if "cleanup_worktrees" in context:
            return self._coerce_context_flag(context, "cleanup_worktrees", default=True)
        follow_up_contract = (
            dict(result_dict.get("delegate_follow_up_contract") or {})
            if isinstance(result_dict, Mapping)
            else {}
        )
        if bool(follow_up_contract.get("preserve_worktrees")):
            return False
        return True

    @classmethod
    def _build_preserved_worktree_cleanup_summary(
        cls,
        worktree_session: WorktreeMaterializationSession,
        *,
        reason: str,
    ) -> Dict[str, Any]:
        skipped: list[str] = []
        assignments = getattr(worktree_session, "assignments", [])
        for assignment in list(assignments or []):
            path = cls._coerce_optional_text(getattr(assignment, "worktree_path", None))
            if path is not None:
                skipped.append(path)
        return {
            "removed": [],
            "skipped": skipped,
            "errors": [],
            "reason": reason,
        }

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
        delegate_next_step = self._extract_delegate_next_step(context)
        effective_context = self._apply_delegate_next_step_context(
            context,
            delegate_next_step=delegate_next_step,
        )
        execution_members = list(members)
        delegate_merge_contract = self._extract_delegate_merge_contract(effective_context)
        if not execution_members and not delegate_merge_contract:
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
            shared_context=copy.deepcopy(dict(effective_context)),
        )
        token = self._execution_state.set(execution_state)
        try:
            start_time = time.time()
            effective_formation = self._resolve_effective_formation(
                effective_context,
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
                if delegate_merge_contract:
                    result = self._execute_delegate_merge_contract(
                        task,
                        effective_context,
                        formation=effective_formation,
                        delegate_merge_contract=delegate_merge_contract,
                    )
                else:
                    result = await self._execute_formation(
                        task,
                        effective_context,
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
                        "delegate_selected_step": effective_context.get("delegate_selected_step"),
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

    def _adapt_team_members(self, members: List[Any]) -> List["ITeamMember"]:
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

        return [TeamMemberAdapter(member=m, executor=_make_executor(m)) for m in members]

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

    def with_state_graph_config(self, config: StateGraphNodeConfig) -> "UnifiedTeamCoordinator":
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
    def _build_call_context(state: Any, kind: str, *, exclude: set) -> Dict[str, Any]:
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
        context = self._build_call_context(state, kind, exclude={config.task_key, config.query_key})
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
