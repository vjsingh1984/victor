"""Planning-to-team execution adapter.

This module keeps planning mode wired to the reusable team formation runtime
instead of creating a separate multi-agent execution path.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

from victor.agent.planning.base import ExecutionPlan, PlanStep, StepResult, StepType
from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
from victor.agent.runtime.naming import build_display_name
from victor.agent.subagents import SubAgentOrchestrator, SubAgentRole
from victor.teams.types import TeamFormation, TeamMember, TeamMemberAdapter, TeamResult


class PlanningTeamExecutionAdapter:
    """Execute complex planned work through ``UnifiedTeamCoordinator`` formations."""

    def __init__(
        self,
        orchestrator: Any,
        sub_agent_orchestrator: Optional[SubAgentOrchestrator] = None,
        coordinator_factory: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.sub_agent_orchestrator = sub_agent_orchestrator
        self._coordinator_factory = coordinator_factory

    def should_use_team(self, plan: ReadableTaskPlan) -> bool:
        """Return whether this plan benefits from team formation execution."""
        if plan.complexity == TaskComplexity.COMPLEX:
            return True

        exploratory_steps = 0
        for step_data in plan.steps:
            if len(step_data) < 2:
                continue
            step_type = str(step_data[1]).lower()
            if step_type in {"research", "analyze", "analysis", "review", "doc"}:
                exploratory_steps += 1
        return exploratory_steps >= 2

    async def execute_step(
        self,
        *,
        plan: ReadableTaskPlan,
        execution_plan: ExecutionPlan,
        step: PlanStep,
        root_session_id: Optional[str] = None,
    ) -> StepResult:
        """Execute a plan step as a hierarchical team run."""
        coordinator = self._create_coordinator()
        team_id = self._team_id(execution_plan.id, step.id)
        root_session_id = root_session_id or self._root_session_id()
        members = self._build_members(execution_plan, team_id, current_step=step)
        manager = members["plan_manager"]

        coordinator.set_formation(TeamFormation.HIERARCHICAL)
        coordinator.set_manager(manager)
        for member_id, member in members.items():
            if member_id != manager.id:
                coordinator.add_member(member)

        context = {
            "root_session_id": root_session_id,
            "parent_session_id": root_session_id,
            "plan_id": execution_plan.id,
            "plan_step_id": step.id,
            "team_id": team_id,
            "formation": TeamFormation.HIERARCHICAL.value,
            "plan_name": plan.name,
            "plan_complexity": plan.complexity.value,
            "task_description": plan.desc,
            "step_description": step.description,
        }
        result = await coordinator.execute_task(step.description, context)
        return self._team_result_to_step_result(result)

    def _create_coordinator(self) -> Any:
        if self._coordinator_factory is not None:
            return self._coordinator_factory(self.orchestrator)

        from victor.teams import UnifiedTeamCoordinator

        return UnifiedTeamCoordinator(self.orchestrator)

    def _build_members(
        self,
        execution_plan: ExecutionPlan,
        team_id: str,
        *,
        current_step: Optional[PlanStep] = None,
    ) -> Dict[str, Any]:
        members: Dict[str, Any] = {}
        manager = TeamMember(
            id="plan_manager",
            role=SubAgentRole.PLANNER,
            name="Plan Manager",
            goal=f"Coordinate and synthesize plan {execution_plan.id}",
            tool_budget=10,
            is_manager=True,
            can_delegate=True,
        )
        members[manager.id] = self._adapt_member(manager, team_id, execution_plan.id)

        step_members = [current_step] if current_step is not None else execution_plan.steps
        for step in step_members:
            role = self._role_for_step(step)
            member_id = f"step_{self._slug(step.id)}_{role.value}"
            member = TeamMember(
                id=member_id,
                role=role,
                name=self._display_name(step, role),
                goal=step.description,
                tool_budget=step.estimated_tool_calls or 10,
                reports_to=manager.id,
            )
            members[member.id] = self._adapt_member(
                member,
                team_id,
                execution_plan.id,
                plan_step_id=step.id,
            )

        return members

    def _adapt_member(
        self,
        member: TeamMember,
        team_id: str,
        plan_id: str,
        *,
        plan_step_id: Optional[str] = None,
    ) -> TeamMemberAdapter:
        async def _execute(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
            subagents = self._subagents()
            parent_session_id = context.get("parent_session_id") or context.get("root_session_id")
            child_session_id = self._child_session_id(parent_session_id, team_id, member.id)
            agent_id = f"{team_id}_{member.id}"
            result = await subagents.spawn(
                role=member.role,
                task=task,
                tool_budget=member.tool_budget,
                allowed_tools=member.allowed_tools,
                member_id=member.id,
                agent_id=agent_id,
                display_name=member.name,
                team_id=team_id,
                plan_id=plan_id,
                plan_step_id=plan_step_id or context.get("plan_step_id"),
                parent_session_id=parent_session_id,
                child_session_id=child_session_id,
            )
            return {
                "success": result.success,
                "output": result.summary,
                "error": result.error,
                "metadata": dict(result.details),
                "tool_calls_used": result.tool_calls_used,
                "duration_seconds": result.duration_seconds,
            }

        return TeamMemberAdapter(member=member, executor=_execute)

    def _subagents(self) -> SubAgentOrchestrator:
        if self.sub_agent_orchestrator is None:
            self.sub_agent_orchestrator = SubAgentOrchestrator(self.orchestrator)
        return self.sub_agent_orchestrator

    @staticmethod
    def _team_result_to_step_result(result: Any) -> StepResult:
        if isinstance(result, TeamResult):
            success = result.success
            output = result.final_output
            error = result.error
            tool_calls = result.total_tool_calls
        else:
            success = bool(result.get("success"))
            output = str(result.get("final_output", ""))
            error = result.get("error")
            tool_calls = int(result.get("total_tool_calls", 0) or 0)

        return StepResult(
            success=success,
            output=output,
            error=error,
            tool_calls_used=tool_calls,
            artifacts=[],
        )

    @staticmethod
    def _role_for_step(step: PlanStep) -> SubAgentRole:
        role = (step.sub_agent_role or "").lower()
        role_map = {
            "researcher": SubAgentRole.RESEARCHER,
            "planner": SubAgentRole.PLANNER,
            "executor": SubAgentRole.EXECUTOR,
            "reviewer": SubAgentRole.REVIEWER,
            "tester": SubAgentRole.TESTER,
        }
        if role in role_map:
            return role_map[role]
        step_type_map = {
            StepType.RESEARCH: SubAgentRole.RESEARCHER,
            StepType.PLANNING: SubAgentRole.PLANNER,
            StepType.IMPLEMENTATION: SubAgentRole.EXECUTOR,
            StepType.TESTING: SubAgentRole.TESTER,
            StepType.REVIEW: SubAgentRole.REVIEWER,
            StepType.DEPLOYMENT: SubAgentRole.EXECUTOR,
        }
        return step_type_map.get(step.step_type, SubAgentRole.EXECUTOR)

    @staticmethod
    def _display_name(step: PlanStep, role: SubAgentRole) -> str:
        return build_display_name(role, task=step.description, ordinal=step.id)

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
        return slug or "step"

    @classmethod
    def _team_id(cls, plan_id: str, step_id: str) -> str:
        return f"team_{cls._slug(plan_id)}_{cls._slug(step_id)}"

    @staticmethod
    def _child_session_id(parent_session_id: Optional[str], team_id: str, member_id: str) -> str:
        parent = parent_session_id or "session"
        return f"{parent}:{team_id}:{member_id}"

    def _root_session_id(self) -> Optional[str]:
        return (
            getattr(self.orchestrator, "active_session_id", None)
            or getattr(self.orchestrator, "session_id", None)
            or getattr(self.orchestrator, "_memory_session_id", None)
        )
