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

"""Team coordinator for orchestrating multi-agent execution.

This module provides TeamCoordinator, the central class for executing
agent teams with different coordination patterns (formations).

Formations:
- SEQUENTIAL: One agent after another, passing context
- PARALLEL: All agents simultaneously on independent aspects
- HIERARCHICAL: Manager delegates to workers
- PIPELINE: Output chains from one agent to the next

Design Principles:
- Built on existing SubAgentOrchestrator infrastructure
- Non-blocking async execution
- Progress tracking via events
- Graceful error handling with partial results

Example:
    from victor.agent.teams import TeamCoordinator, TeamConfig, TeamFormation

    coordinator = TeamCoordinator(orchestrator)
    result = await coordinator.execute_team(config)

    if result.success:
        print(result.final_output)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from victor.observability.team_metrics import (
    record_team_spawned,
    record_team_completed,
)
from victor.agent.subagents.base import SubAgentConfig, SubAgentResult
from victor.agent.subagents.orchestrator import (
    ROLE_DEFAULT_BUDGETS,
    ROLE_DEFAULT_CONTEXT,
    ROLE_DEFAULT_TOOLS,
    SubAgentOrchestrator,
    SubAgentTask,
)
# Import canonical types from victor.teams
from victor.teams import (
    TeamFormation,
    MessageType,
    MemberResult,
    TeamResult,
)
from victor.agent.teams.communication import (
    AgentMessage,  # Local AgentMessage for TeamMessageBus compatibility
    TeamMessageBus,
    TeamSharedMemory,
)
from victor.agent.teams.team import (
    MemberStatus,
    TeamConfig,
    TeamMember,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


class TeamExecution:
    """Tracks state of an active team execution.

    Attributes:
        team_id: Unique identifier for this execution
        config: Team configuration
        status: Current execution status
        member_statuses: Status of each member
        start_time: When execution started
    """

    def __init__(self, config: TeamConfig):
        self.team_id = uuid.uuid4().hex[:12]
        self.config = config
        self.status: str = "pending"
        self.member_statuses: Dict[str, MemberStatus] = {
            m.id: MemberStatus.IDLE for m in config.members
        }
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.message_bus = TeamMessageBus(self.team_id)
        self.shared_memory = TeamSharedMemory()

        # Register all members on the message bus
        for member in config.members:
            self.message_bus.register_agent(member.id)


class TeamCoordinator:
    """Coordinates execution of agent teams.

    Provides high-level API for executing teams with different formation
    patterns. Handles member lifecycle, communication, and result aggregation.

    Attributes:
        orchestrator: Parent orchestrator for spawning sub-agents
        sub_agent_orchestrator: Underlying sub-agent infrastructure
        active_teams: Currently executing teams

    Example:
        coordinator = TeamCoordinator(orchestrator)

        # Execute a sequential team
        result = await coordinator.execute_team(config)

        # Execute in background
        team_id = await coordinator.start_team(config)
        status = coordinator.get_team_status(team_id)
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        sub_agent_orchestrator: Optional[SubAgentOrchestrator] = None,
    ):
        """Initialize team coordinator.

        Args:
            orchestrator: Parent orchestrator
            sub_agent_orchestrator: Optional sub-agent orchestrator (created if not provided)
        """
        self.orchestrator = orchestrator
        self.sub_agents = sub_agent_orchestrator or SubAgentOrchestrator(orchestrator)
        self._active_teams: Dict[str, TeamExecution] = {}
        self._on_progress: Optional[Callable[[str, str, float], None]] = None

        # Execution context for observability
        self._task_type: str = "unknown"
        self._complexity: str = "medium"
        self._vertical_name: str = "coding"
        self._trigger: str = "auto"  # auto, manual, suggestion
        self._rl_coordinator: Optional[Any] = None

        logger.info("TeamCoordinator initialized")

    def set_execution_context(
        self,
        task_type: str = "unknown",
        complexity: str = "medium",
        vertical: str = "coding",
        trigger: str = "auto",
    ) -> None:
        """Set execution context for observability and RL.

        Args:
            task_type: Type of task being executed
            complexity: Complexity level (low, medium, high, extreme)
            vertical: Vertical name (coding, devops, etc.)
            trigger: What triggered the execution (auto, manual, suggestion)
        """
        self._task_type = task_type
        self._complexity = complexity
        self._vertical_name = vertical
        self._trigger = trigger

    def set_rl_coordinator(self, rl_coordinator: Any) -> None:
        """Set the RL coordinator for recording outcomes.

        Args:
            rl_coordinator: RLCoordinator instance
        """
        self._rl_coordinator = rl_coordinator

    def set_progress_callback(
        self,
        callback: Callable[[str, str, float], None],
    ) -> None:
        """Set callback for progress updates.

        Callback receives (team_id, status_message, progress_percent).

        Args:
            callback: Progress callback function
        """
        self._on_progress = callback

    async def execute_team(
        self,
        config: TeamConfig,
        on_member_complete: Optional[Callable[[str, MemberResult], None]] = None,
    ) -> TeamResult:
        """Execute a team synchronously.

        Blocks until team execution completes or times out.

        Args:
            config: Team configuration
            on_member_complete: Optional callback when each member finishes

        Returns:
            TeamResult with execution outcome

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout
        """
        execution = TeamExecution(config)
        self._active_teams[execution.team_id] = execution
        execution.status = "running"
        execution.start_time = time.time()

        logger.info(
            f"Starting team '{config.name}' with {len(config.members)} members "
            f"({config.formation.value} formation)"
        )

        # Record team spawned for observability
        record_team_spawned(
            team_name=config.name,
            vertical=self._vertical_name,
            task_type=self._task_type,
            complexity=self._complexity,
            trigger=self._trigger,
        )

        try:
            # Execute based on formation
            if config.formation == TeamFormation.SEQUENTIAL:
                result = await self._execute_sequential(execution, on_member_complete)
            elif config.formation == TeamFormation.PARALLEL:
                result = await self._execute_parallel(execution, on_member_complete)
            elif config.formation == TeamFormation.HIERARCHICAL:
                result = await self._execute_hierarchical(execution, on_member_complete)
            elif config.formation == TeamFormation.PIPELINE:
                result = await self._execute_pipeline(execution, on_member_complete)
            else:
                raise ValueError(f"Unknown formation: {config.formation}")

            execution.status = "completed" if result.success else "failed"
            execution.end_time = time.time()
            duration_seconds = result.total_duration

            logger.info(
                f"Team '{config.name}' completed: "
                f"success={result.success}, "
                f"tool_calls={result.total_tool_calls}, "
                f"duration={result.total_duration:.1f}s"
            )

            # Record team completed for observability
            record_team_completed(
                team_name=config.name,
                success=result.success,
                duration_seconds=duration_seconds,
                tool_calls=result.total_tool_calls,
                formation=config.formation.value,
                member_count=len(config.members),
            )

            # Record RL outcome for team composition learning
            self._record_team_rl_outcome(config, result)

            return result

        except Exception as e:
            execution.status = "error"
            execution.end_time = time.time()
            duration_seconds = time.time() - execution.start_time
            logger.error(f"Team '{config.name}' failed: {e}", exc_info=True)

            # Record failed team for observability
            record_team_completed(
                team_name=config.name,
                success=False,
                duration_seconds=duration_seconds,
                tool_calls=0,
                formation=config.formation.value,
                member_count=len(config.members),
            )

            return TeamResult(
                success=False,
                final_output=f"Team execution failed: {str(e)}",
                member_results={},
                total_tool_calls=0,
                total_duration=duration_seconds,
                formation=config.formation,
            )

        finally:
            # Cleanup
            self._active_teams.pop(execution.team_id, None)

    def _record_team_rl_outcome(
        self,
        config: TeamConfig,
        result: TeamResult,
    ) -> None:
        """Record RL outcome for team composition learning.

        Args:
            config: Team configuration used
            result: Execution result
        """
        if not self._rl_coordinator:
            return

        try:
            from victor.agent.rl.base import RLOutcome

            # Compute quality score based on success and efficiency
            quality_score = 0.5  # Baseline
            if result.success:
                quality_score = 0.7
                # Bonus for efficient tool usage
                if result.total_tool_calls < len(config.members) * 10:
                    quality_score += 0.1
                # Bonus for fast execution
                if result.total_duration < 60:
                    quality_score += 0.1
                # Penalty for many failures
                failed_members = sum(1 for r in result.member_results.values() if not r.success)
                if failed_members > 0:
                    quality_score -= 0.1 * failed_members

            quality_score = min(1.0, max(0.0, quality_score))

            outcome = RLOutcome(
                provider="team_coordinator",
                model=config.formation.value,
                task_type=self._task_type,
                success=result.success,
                quality_score=quality_score,
                metadata={
                    "team_name": config.name,
                    "member_count": len(config.members),
                    "tool_calls": result.total_tool_calls,
                    "duration_seconds": result.total_duration,
                    "formation": config.formation.value,
                },
                vertical=self._vertical_name,
            )

            self._rl_coordinator.record_outcome(
                "team_composition",
                outcome,
                vertical=self._vertical_name,
            )

            # Also emit via RL hooks for event-driven tracking
            self._emit_team_completed_event(config, result, quality_score)

            logger.debug(
                f"Recorded RL outcome for team '{config.name}': "
                f"quality={quality_score:.2f}, success={result.success}"
            )

        except ImportError:
            # RL module not available
            pass
        except Exception as e:
            logger.warning(f"Failed to record team RL outcome: {e}")

    def _emit_team_completed_event(
        self,
        config: "TeamConfig",
        result: "TeamResult",
        quality_score: float,
    ) -> None:
        """Emit RL event for team completion.

        This activates the team_composition learner via the event system.

        Args:
            config: Team configuration
            result: Execution result
            quality_score: Computed quality score
        """
        try:
            from victor.agent.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            event = RLEvent(
                type=RLEventType.TEAM_COMPLETED,
                team_id=config.name,
                team_formation=config.formation.value,
                success=result.success,
                quality_score=quality_score,
                task_type=self._task_type,
                vertical=self._vertical_name,
                metadata={
                    "member_count": len(config.members),
                    "tool_calls": result.total_tool_calls,
                    "duration_seconds": result.total_duration,
                },
            )

            hooks.emit(event)

        except Exception as e:
            logger.debug(f"Team completed event emission failed: {e}")

    async def _execute_sequential(
        self,
        execution: TeamExecution,
        on_member_complete: Optional[Callable[[str, MemberResult], None]],
    ) -> TeamResult:
        """Execute members one after another, passing context.

        Each member receives the shared context including results from
        previous members.

        Args:
            execution: Team execution state
            on_member_complete: Member completion callback

        Returns:
            TeamResult with all member results
        """
        config = execution.config
        member_results: Dict[str, MemberResult] = {}
        total_tool_calls = 0

        # Sort by priority
        sorted_members = sorted(config.members, key=lambda m: m.priority)

        for i, member in enumerate(sorted_members):
            execution.member_statuses[member.id] = MemberStatus.WORKING
            self._report_progress(
                execution.team_id,
                f"Running {member.name}...",
                (i / len(sorted_members)) * 100,
            )

            # Build context with previous results
            context = self._build_member_context(
                member,
                config,
                execution.shared_memory,
                member_results,
            )

            # Execute member
            result = await self._execute_member(member, context, execution)
            member_results[member.id] = result
            total_tool_calls += result.tool_calls_used

            # Update shared memory with discoveries
            for discovery in result.discoveries:
                execution.shared_memory.append("discoveries", discovery, member.id)

            # Store result in shared memory
            execution.shared_memory.set(
                f"result_{member.id}",
                result.output,
                member.id,
            )

            # Callback
            if on_member_complete:
                on_member_complete(member.id, result)

            # Update status
            execution.member_statuses[member.id] = (
                MemberStatus.COMPLETED if result.success else MemberStatus.FAILED
            )

            # Notify team via message bus
            await execution.message_bus.send(
                AgentMessage(
                    type=MessageType.RESULT,
                    from_agent=member.id,
                    content=result.output[:500],
                    data={"success": result.success, "tool_calls": result.tool_calls_used},
                )
            )

        # Synthesize final output
        final_output = self._synthesize_results(member_results, config)

        return TeamResult(
            success=all(r.success for r in member_results.values()),
            final_output=final_output,
            member_results=member_results,
            total_tool_calls=total_tool_calls,
            total_duration=time.time() - execution.start_time,
            communication_log=[m.to_dict() for m in execution.message_bus.get_message_log()],
            shared_context=execution.shared_memory.get_all(),
            formation=TeamFormation.SEQUENTIAL,
        )

    async def _execute_parallel(
        self,
        execution: TeamExecution,
        on_member_complete: Optional[Callable[[str, MemberResult], None]],
    ) -> TeamResult:
        """Execute all members simultaneously.

        All members work in parallel on independent aspects. Good for
        research and exploration tasks.

        Args:
            execution: Team execution state
            on_member_complete: Member completion callback

        Returns:
            TeamResult with all member results
        """
        config = execution.config

        # Mark all as working
        for member in config.members:
            execution.member_statuses[member.id] = MemberStatus.WORKING

        self._report_progress(execution.team_id, "Running all members in parallel...", 10)

        # Create tasks for parallel execution
        tasks = []
        for member in config.members:
            context = self._build_member_context(
                member,
                config,
                execution.shared_memory,
                {},
            )
            tasks.append(
                SubAgentTask(
                    role=member.role,
                    task=context,
                    tool_budget=member.tool_budget,
                    allowed_tools=member.allowed_tools,
                )
            )

        # Execute in parallel using fan_out
        fan_out_result = await self.sub_agents.fan_out(
            tasks,
            max_concurrent=len(tasks),
        )

        # Process results
        member_results: Dict[str, MemberResult] = {}
        for i, (member, sub_result) in enumerate(zip(config.members, fan_out_result.results)):
            result = self._convert_subagent_result(member.id, sub_result)
            member_results[member.id] = result

            execution.member_statuses[member.id] = (
                MemberStatus.COMPLETED if result.success else MemberStatus.FAILED
            )

            if on_member_complete:
                on_member_complete(member.id, result)

        self._report_progress(execution.team_id, "All members complete", 100)

        # Synthesize results
        final_output = self._synthesize_results(member_results, config)

        return TeamResult(
            success=fan_out_result.all_success,
            final_output=final_output,
            member_results=member_results,
            total_tool_calls=fan_out_result.total_tool_calls,
            total_duration=fan_out_result.total_duration,
            shared_context=execution.shared_memory.get_all(),
            formation=TeamFormation.PARALLEL,
        )

    async def _execute_hierarchical(
        self,
        execution: TeamExecution,
        on_member_complete: Optional[Callable[[str, MemberResult], None]],
    ) -> TeamResult:
        """Execute with manager delegating to workers.

        Manager analyzes the goal, delegates sub-tasks to workers,
        and synthesizes their results.

        Args:
            execution: Team execution state
            on_member_complete: Member completion callback

        Returns:
            TeamResult with all member results
        """
        config = execution.config
        manager = config.get_manager()
        workers = config.get_workers()
        member_results: Dict[str, MemberResult] = {}
        total_tool_calls = 0

        if not manager:
            raise ValueError("Hierarchical formation requires a manager")

        # Phase 1: Manager analyzes and plans
        self._report_progress(execution.team_id, f"{manager.name} planning...", 10)
        execution.member_statuses[manager.id] = MemberStatus.WORKING

        manager_planning_prompt = f"""You are the team manager. Your goal: {config.goal}

You have the following team members to delegate to:
{self._format_worker_list(workers)}

Analyze the goal and create a delegation plan. For each worker, specify what they should do.
Output your plan in a structured format."""

        manager_result = await self._execute_member(
            manager,
            manager_planning_prompt,
            execution,
        )
        member_results[manager.id] = manager_result
        total_tool_calls += manager_result.tool_calls_used

        if not manager_result.success:
            return TeamResult(
                success=False,
                final_output=f"Manager failed to plan: {manager_result.error}",
                member_results=member_results,
                total_tool_calls=total_tool_calls,
                total_duration=time.time() - execution.start_time,
                formation=TeamFormation.HIERARCHICAL,
            )

        execution.member_statuses[manager.id] = MemberStatus.DELEGATING

        # Phase 2: Workers execute in parallel
        self._report_progress(execution.team_id, "Delegating to workers...", 30)

        worker_tasks = []
        for worker in workers:
            execution.member_statuses[worker.id] = MemberStatus.WORKING
            worker_context = f"""You are {worker.name}, a {worker.role.value} agent.

Team Goal: {config.goal}

Manager's Instructions:
{manager_result.output}

Your specific goal: {worker.goal}

Execute your assigned tasks and report your findings."""

            worker_tasks.append(
                SubAgentTask(
                    role=worker.role,
                    task=worker_context,
                    tool_budget=worker.tool_budget,
                    allowed_tools=worker.allowed_tools,
                )
            )

        worker_fan_out = await self.sub_agents.fan_out(
            worker_tasks,
            max_concurrent=min(len(worker_tasks), 4),
        )

        for worker, sub_result in zip(workers, worker_fan_out.results):
            result = self._convert_subagent_result(worker.id, sub_result)
            member_results[worker.id] = result
            total_tool_calls += result.tool_calls_used

            execution.member_statuses[worker.id] = (
                MemberStatus.COMPLETED if result.success else MemberStatus.FAILED
            )

            if on_member_complete:
                on_member_complete(worker.id, result)

        # Phase 3: Manager synthesizes
        self._report_progress(execution.team_id, f"{manager.name} synthesizing...", 80)
        execution.member_statuses[manager.id] = MemberStatus.WORKING

        worker_reports = "\n\n".join(
            f"**{w.name}**:\n{member_results[w.id].output}" for w in workers
        )

        synthesis_prompt = f"""You are the team manager completing the team's work.

Original Goal: {config.goal}

Worker Reports:
{worker_reports}

Synthesize these reports into a final comprehensive result that addresses the original goal."""

        synthesis_result = await self._execute_member(
            manager,
            synthesis_prompt,
            execution,
        )
        total_tool_calls += synthesis_result.tool_calls_used

        execution.member_statuses[manager.id] = MemberStatus.COMPLETED

        if on_member_complete:
            on_member_complete(manager.id, synthesis_result)

        return TeamResult(
            success=worker_fan_out.all_success and synthesis_result.success,
            final_output=synthesis_result.output,
            member_results=member_results,
            total_tool_calls=total_tool_calls,
            total_duration=time.time() - execution.start_time,
            shared_context=execution.shared_memory.get_all(),
            formation=TeamFormation.HIERARCHICAL,
        )

    async def _execute_pipeline(
        self,
        execution: TeamExecution,
        on_member_complete: Optional[Callable[[str, MemberResult], None]],
    ) -> TeamResult:
        """Execute as a pipeline where output chains to next member.

        Each member's output becomes the primary input for the next member.
        Good for multi-stage processing (research → plan → execute → review).

        Args:
            execution: Team execution state
            on_member_complete: Member completion callback

        Returns:
            TeamResult with all member results
        """
        config = execution.config
        member_results: Dict[str, MemberResult] = {}
        total_tool_calls = 0

        # Sort by priority for pipeline order
        sorted_members = sorted(config.members, key=lambda m: m.priority)

        previous_output: Optional[str] = None

        for i, member in enumerate(sorted_members):
            execution.member_statuses[member.id] = MemberStatus.WORKING
            self._report_progress(
                execution.team_id,
                f"Pipeline stage {i + 1}/{len(sorted_members)}: {member.name}",
                (i / len(sorted_members)) * 100,
            )

            # Build pipeline context
            if previous_output:
                context = f"""You are {member.name}, a {member.role.value} agent in a processing pipeline.

Team Goal: {config.goal}
Your Goal: {member.goal}

## Input from Previous Stage:
{previous_output}

Process this input according to your goal and produce output for the next stage."""
            else:
                context = f"""You are {member.name}, the first agent in a processing pipeline.

Team Goal: {config.goal}
Your Goal: {member.goal}

Start the pipeline by {member.goal.lower()}. Your output will be passed to the next stage."""

            # Execute member
            result = await self._execute_member(member, context, execution)
            member_results[member.id] = result
            total_tool_calls += result.tool_calls_used
            previous_output = result.output

            # Update status
            execution.member_statuses[member.id] = (
                MemberStatus.COMPLETED if result.success else MemberStatus.FAILED
            )

            if on_member_complete:
                on_member_complete(member.id, result)

            # Send handoff message
            if i < len(sorted_members) - 1:
                next_member = sorted_members[i + 1]
                await execution.message_bus.send(
                    AgentMessage(
                        type=MessageType.HANDOFF,
                        from_agent=member.id,
                        to_agent=next_member.id,
                        content=f"Pipeline stage complete. Passing to {next_member.name}.",
                    )
                )

            # Stop pipeline on failure
            if not result.success:
                logger.warning(f"Pipeline stopped at {member.name}: {result.error}")
                break

        # Final output is from last member
        final_member_id = sorted_members[-1].id
        final_result = member_results.get(final_member_id)

        return TeamResult(
            success=all(r.success for r in member_results.values()),
            final_output=final_result.output if final_result else "Pipeline failed",
            member_results=member_results,
            total_tool_calls=total_tool_calls,
            total_duration=time.time() - execution.start_time,
            communication_log=[m.to_dict() for m in execution.message_bus.get_message_log()],
            shared_context=execution.shared_memory.get_all(),
            formation=TeamFormation.PIPELINE,
        )

    async def _execute_member(
        self,
        member: TeamMember,
        context: str,
        execution: TeamExecution,
    ) -> MemberResult:
        """Execute a single team member.

        Args:
            member: Team member to execute
            context: Full context/prompt for the member
            execution: Team execution state

        Returns:
            MemberResult with execution outcome
        """
        start_time = time.time()

        try:
            result = await self.sub_agents.spawn(
                role=member.role,
                task=context,
                tool_budget=member.tool_budget,
                allowed_tools=member.allowed_tools,
                timeout_seconds=execution.config.timeout_seconds // len(execution.config.members),
            )

            return MemberResult(
                member_id=member.id,
                success=result.success,
                output=result.summary if result.summary else "",
                tool_calls_used=result.tool_calls_used,
                duration_seconds=result.duration_seconds,
                discoveries=self._extract_discoveries(result),
                error=result.error,
            )

        except Exception as e:
            logger.error(f"Member {member.id} execution failed: {e}")
            return MemberResult(
                member_id=member.id,
                success=False,
                output="",
                tool_calls_used=0,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    def _build_member_context(
        self,
        member: TeamMember,
        config: TeamConfig,
        shared_memory: TeamSharedMemory,
        previous_results: Dict[str, MemberResult],
    ) -> str:
        """Build context string for a member.

        Incorporates all persona attributes (backstory, expertise, personality)
        to create a rich context that guides agent behavior.

        Args:
            member: Member to build context for
            config: Team configuration
            shared_memory: Shared memory state
            previous_results: Results from previous members

        Returns:
            Full context string for the member
        """
        lines = [
            f"You are {member.name}, a {member.role.value} agent on a team.",
            "",
            "## Team Goal",
            config.goal,
            "",
            "## Your Specific Goal",
            member.goal,
        ]

        # Add backstory/persona if provided (CrewAI-compatible)
        if member.backstory:
            lines.extend(
                [
                    "",
                    "## Background",
                    member.backstory,
                ]
            )

        # Add expertise domains if specified
        if member.expertise:
            lines.extend(
                [
                    "",
                    "## Expertise",
                    f"Your areas of expertise: {', '.join(member.expertise)}",
                    "Leverage this expertise when analyzing problems and making recommendations.",
                ]
            )

        # Add personality/communication style if specified
        if member.personality:
            lines.extend(
                [
                    "",
                    "## Communication Style",
                    member.personality,
                ]
            )

        # Add delegation info if applicable
        if member.can_delegate:
            depth_info = ""
            if member.max_delegation_depth > 0:
                depth_info = f" (maximum {member.max_delegation_depth} levels deep)"
            if member.delegation_targets:
                lines.extend(
                    [
                        "",
                        "## Delegation",
                        f"You can delegate tasks to: {', '.join(member.delegation_targets)}{depth_info}",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        "## Delegation",
                        f"You can delegate tasks to other team members when appropriate{depth_info}.",
                    ]
                )

        # Add shared context
        if config.shared_context:
            lines.extend(
                [
                    "",
                    "## Initial Context",
                    str(config.shared_context),
                ]
            )

        # Add shared memory
        memory_summary = shared_memory.get_summary()
        if "No shared data yet" not in memory_summary:
            lines.extend(["", memory_summary])

        # Add previous results
        if previous_results:
            lines.extend(["", "## Results from Previous Team Members"])
            for member_id, result in previous_results.items():
                lines.extend(
                    [
                        f"### {member_id}",
                        result.output[:1000] if result.output else "(no output)",
                        "",
                    ]
                )

        return "\n".join(lines)

    def _convert_subagent_result(
        self,
        member_id: str,
        result: SubAgentResult,
    ) -> MemberResult:
        """Convert SubAgentResult to MemberResult.

        Args:
            member_id: ID of the member
            result: SubAgent result

        Returns:
            Converted MemberResult
        """
        return MemberResult(
            member_id=member_id,
            success=result.success,
            output=result.summary if result.summary else "",
            tool_calls_used=result.tool_calls_used,
            duration_seconds=result.duration_seconds,
            discoveries=self._extract_discoveries(result),
            error=result.error,
        )

    def _extract_discoveries(self, result: SubAgentResult) -> List[str]:
        """Extract discovery statements from a result.

        Args:
            result: SubAgent result

        Returns:
            List of discovery strings
        """
        discoveries = []
        if result.success and result.summary:
            # Simple extraction: lines starting with "Found", "Discovered", etc.
            for line in result.summary.split("\n"):
                line = line.strip()
                if any(
                    line.lower().startswith(prefix)
                    for prefix in ["found", "discovered", "identified", "located", "detected"]
                ):
                    discoveries.append(line)
        return discoveries

    def _synthesize_results(
        self,
        member_results: Dict[str, MemberResult],
        config: TeamConfig,
    ) -> str:
        """Synthesize final output from member results.

        Args:
            member_results: All member results
            config: Team configuration

        Returns:
            Synthesized final output
        """
        lines = [f"# Team '{config.name}' Results\n"]

        for member in config.members:
            result = member_results.get(member.id)
            if result:
                status = "✓" if result.success else "✗"
                lines.append(f"## {status} {member.name}")
                lines.append(result.output if result.output else "(no output)")
                lines.append("")

        return "\n".join(lines)

    def _format_worker_list(self, workers: List[TeamMember]) -> str:
        """Format worker list for manager context.

        Includes expertise information to help managers make informed
        delegation decisions.

        Args:
            workers: List of worker members

        Returns:
            Formatted worker list with roles, goals, and expertise
        """
        lines = []
        for worker in workers:
            expertise_info = ""
            if worker.expertise:
                expertise_info = f" [expertise: {', '.join(worker.expertise[:5])}]"
            lines.append(
                f"- **{worker.name}** ({worker.role.value}): {worker.goal}{expertise_info}"
            )
        return "\n".join(lines)

    def _report_progress(self, team_id: str, message: str, percent: float) -> None:
        """Report execution progress.

        Args:
            team_id: Team identifier
            message: Status message
            percent: Progress percentage (0-100)
        """
        if self._on_progress:
            try:
                self._on_progress(team_id, message, percent)
            except Exception as e:
                logger.debug(f"Progress callback failed: {e}")

    def get_team_status(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active team.

        Args:
            team_id: Team identifier

        Returns:
            Status dictionary or None if team not found
        """
        execution = self._active_teams.get(team_id)
        if not execution:
            return None

        return {
            "team_id": team_id,
            "name": execution.config.name,
            "status": execution.status,
            "formation": execution.config.formation.value,
            "member_statuses": {
                mid: status.value for mid, status in execution.member_statuses.items()
            },
            "duration": time.time() - execution.start_time if execution.start_time else 0,
        }

    def get_active_teams(self) -> List[str]:
        """Get IDs of all active teams.

        Returns:
            List of active team IDs
        """
        return list(self._active_teams.keys())


__all__ = [
    "TeamCoordinator",
    "TeamExecution",
]
