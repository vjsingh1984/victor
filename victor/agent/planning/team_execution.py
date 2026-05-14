"""Planning-to-team execution adapter.

This module keeps planning mode wired to the reusable team formation runtime
instead of creating a separate multi-agent execution path.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

from victor.agent.planning.base import (
    ExecutionPlan,
    PlanStep,
    StepResult,
    StepType,
    get_step_allowed_tools,
)
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
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Dispatch a plan step to the appropriate execution node type.

        Dispatch order:
          1. compute  — deterministic function, no model call
          2. tool     — deterministic tool-only subagent, no reasoning overhead
          3. loop     — iterate over plan_state collection, one subagent per item
          4. team     — UnifiedTeamCoordinator formation
          5. agent    — default single-worker path
        """
        # 1. Compute node — no model, no spawn
        compute_node = self._compute_node_for_step(step)
        if compute_node is not None:
            return self._execute_compute_node(step, compute_node)

        execution = (step.execution or step.context.get("execution", "")).lower()

        team_id = self._team_id(execution_plan.id, step.id)
        root_session_id = root_session_id or self._root_session_id()
        context = self._step_context(
            plan=plan,
            execution_plan=execution_plan,
            step=step,
            team_id=team_id,
            root_session_id=root_session_id,
        )
        resolved_plan_state = plan_state or {}

        # 2. Tool node — single subagent, tool-only
        if execution == "tool":
            return await self._execute_tool_node(step, execution_plan, team_id, context)

        # 3. Loop node — iterate over a plan-state collection
        if execution == "loop":
            return await self._execute_loop_node(
                step, execution_plan, team_id, context, resolved_plan_state
            )

        # 4. Team node — explicit multi-agent formation
        if execution == "team":
            return await self._execute_team_node(step, execution_plan, team_id, context)

        # 5. Default: single worker (agent)
        if self._should_execute_step_directly(execution_plan, step):
            members = self._build_members(execution_plan, team_id, current_step=step)
            worker = next(
                member for member_id, member in members.items() if member_id != "plan_manager"
            )
            payload = await worker.execute_task(step.description, context)
            return self._member_payload_to_step_result(payload, worker.id)

        coordinator = self._create_coordinator()
        members = self._build_members(execution_plan, team_id, current_step=step)
        manager = members["plan_manager"]

        coordinator.set_formation(TeamFormation.HIERARCHICAL)
        coordinator.set_manager(manager)
        for member_id, member in members.items():
            if member_id != manager.id:
                coordinator.add_member(member)

        result = await coordinator.execute_task(step.description, context)
        return self._team_result_to_step_result(result)

    async def _execute_tool_node(
        self,
        step: PlanStep,
        execution_plan: "ExecutionPlan",
        team_id: str,
        context: Dict[str, Any],
    ) -> StepResult:
        """Run a deterministic tool-only step via a single subagent worker."""
        members = self._build_members(execution_plan, team_id, current_step=step)
        worker = next(m for mid, m in members.items() if mid != "plan_manager")
        payload = await worker.execute_task(step.description, context)
        result = self._member_payload_to_step_result(payload, worker.id)
        result.metadata["execution_mode"] = "tool_node"
        return result

    async def _execute_team_node(
        self,
        step: PlanStep,
        execution_plan: "ExecutionPlan",
        team_id: str,
        context: Dict[str, Any],
    ) -> StepResult:
        """Run a step as an explicit team formation."""
        coordinator = self._create_coordinator()
        members = self._build_members(execution_plan, team_id, current_step=step)
        manager = members["plan_manager"]
        coordinator.set_formation(TeamFormation.HIERARCHICAL)
        coordinator.set_manager(manager)
        for member_id, member in members.items():
            if member_id != manager.id:
                coordinator.add_member(member)
        result = await coordinator.execute_task(step.description, context)
        step_result = self._team_result_to_step_result(result)
        step_result.metadata["execution_mode"] = "team_node"
        return step_result

    async def _execute_loop_node(
        self,
        step: PlanStep,
        execution_plan: "ExecutionPlan",
        team_id: str,
        context: Dict[str, Any],
        plan_state: Dict[str, Any],
    ) -> StepResult:
        """Iterate over a plan-state collection, spawning one subagent per item.

        Items are resolved from (in priority order):
        1. ``step.context["items"]`` — static list embedded in the step
        2. ``plan_state[step.context["loop_over"]]`` — dynamic list from a prior step

        Execution is sequential by default to avoid exhausting provider retry budget.
        Set ``step.context["parallel"] = True`` to use concurrent execution.
        """
        items = self._resolve_loop_items(step, plan_state)
        if not items:
            return StepResult(
                success=True,
                output=f"Loop '{step.description}': no items to iterate — skipping.",
                tool_calls_used=0,
                metadata={
                    "execution_mode": "loop_node",
                    "loop_items_count": 0,
                    "loop_over": step.context.get("loop_over", ""),
                },
            )

        parallel = bool(step.context.get("parallel", False))
        subagents = self._subagents()
        parent_session_id = context.get("parent_session_id") or context.get("root_session_id")

        async def _run_item(index: int, item: str) -> tuple[str, StepResult]:
            item_task = f"{step.description} — [{item}]"
            agent_id = f"{team_id}_{self._slug(step.id)}_loop_{index}"
            child_session_id = self._child_session_id(parent_session_id, team_id, f"{step.id}_loop_{index}")
            spawn_result = await subagents.spawn(
                role=self._role_for_step(step),
                task=item_task,
                tool_budget=step.estimated_tool_calls,
                allowed_tools=get_step_allowed_tools(step),
                member_id=f"loop_{self._slug(step.id)}_{index}",
                agent_id=agent_id,
                display_name=f"Loop {index + 1}/{len(items)}: {item}",
                team_id=team_id,
                plan_id=execution_plan.id,
                plan_step_id=step.id,
                parent_session_id=parent_session_id,
                child_session_id=child_session_id,
            )
            item_result = StepResult(
                success=spawn_result.success,
                output=spawn_result.summary or "",
                error=spawn_result.error,
                tool_calls_used=spawn_result.tool_calls_used,
                duration_seconds=spawn_result.duration_seconds,
                metadata=dict(spawn_result.details),
            )
            return item, item_result

        if parallel:
            import asyncio as _asyncio
            pairs = await _asyncio.gather(*[_run_item(i, item) for i, item in enumerate(items)])
        else:
            pairs = []
            for i, item in enumerate(items):
                pairs.append(await _run_item(i, item))

        outputs: list[str] = []
        total_tool_calls = 0
        failed_items: list[str] = []
        for item, item_result in pairs:
            total_tool_calls += item_result.tool_calls_used
            if item_result.success:
                outputs.append(f"[{item}]\n{item_result.output}")
            else:
                failed_items.append(item)
                outputs.append(f"[{item}] FAILED: {item_result.error}")

        success = len(failed_items) == 0
        return StepResult(
            success=success,
            output="\n\n".join(outputs),
            error=f"Failed items: {failed_items}" if failed_items else None,
            tool_calls_used=total_tool_calls,
            metadata={
                "execution_mode": "loop_node",
                "loop_items": items,
                "loop_items_count": len(items),
                "failed_items": failed_items,
                "parallel": parallel,
            },
        )

    @staticmethod
    def _resolve_loop_items(step: PlanStep, plan_state: Dict[str, Any]) -> list[str]:
        """Resolve the item collection for a loop node."""
        static_items = step.context.get("items", [])
        if static_items:
            return [str(i) for i in static_items]
        loop_over = step.context.get("loop_over", "")
        if loop_over and plan_state:
            raw = plan_state.get(loop_over)
            if isinstance(raw, list):
                return [str(i) for i in raw if str(i).strip()]
            if isinstance(raw, str):
                return [line.strip() for line in raw.splitlines() if line.strip()]
        return []

    @staticmethod
    def _should_execute_step_directly(execution_plan: ExecutionPlan, step: PlanStep) -> bool:
        """Avoid a manager/worker hierarchy when a step has exactly one owner."""
        return step in execution_plan.steps

    @staticmethod
    def _step_context(
        *,
        plan: ReadableTaskPlan,
        execution_plan: ExecutionPlan,
        step: PlanStep,
        team_id: str,
        root_session_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "root_session_id": root_session_id,
            "parent_session_id": root_session_id,
            "plan_id": execution_plan.id,
            "plan_step_id": step.id,
            "team_id": team_id,
            "formation": "direct_step_worker",
            "plan_name": plan.name,
            "plan_complexity": plan.complexity.value,
            "task_description": plan.desc,
            "step_description": step.description,
        }

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
        manager_allowed_tools = (
            get_step_allowed_tools(current_step) if current_step is not None else None
        )
        manager = TeamMember(
            id="plan_manager",
            role=SubAgentRole.PLANNER,
            name="Plan Manager",
            goal=f"Coordinate and synthesize plan {execution_plan.id}",
            tool_budget=10,
            allowed_tools=manager_allowed_tools,
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
                allowed_tools=get_step_allowed_tools(step),
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

    # ---------------------------------------------------------------------------
    # Compute-node registry
    # ---------------------------------------------------------------------------

    #: Named deterministic compute nodes.  Each entry maps a node name to a
    #: callable that receives the PlanStep and returns a StepResult.  New
    #: deterministic plan actions should be added here rather than as heuristics.
    _COMPUTE_NODES: Dict[str, Callable[["PlanStep"], StepResult]] = {}

    #: No built-in language-specific nodes.  Language/domain checklists belong in
    #: verticals (e.g. victor-coding) and are registered via register_compute_node()
    #: inside VictorPlugin.register(context) at plugin init.
    _BUILTIN_NODES: frozenset = frozenset()

    @classmethod
    def register_compute_node(
        cls, name: str, fn: Callable[["PlanStep"], StepResult]
    ) -> None:
        """Register a named deterministic compute node."""
        cls._COMPUTE_NODES[name] = fn

    # ---------------------------------------------------------------------------
    # Execution dispatch helpers
    # ---------------------------------------------------------------------------

    @classmethod
    def _compute_node_for_step(cls, step: PlanStep) -> Optional[str]:
        """Return a deterministic compute node name when one applies.

        Checks in order:
        1. Explicit ``step.execution == "compute"`` — use ``step.context["node"]``
           or fall through to the description-based registry lookup.
        2. Registered compute node whose name appears in the step description.
        3. Legacy heuristic for checklist steps (backward compat).
        """
        execution = (step.execution or step.context.get("execution", "")).lower()
        if execution == "compute":
            explicit_node = step.context.get("node", "")
            if explicit_node and (
                explicit_node in cls._COMPUTE_NODES or explicit_node in cls._BUILTIN_NODES
            ):
                return explicit_node
            # Find a registered node whose name matches the description
            desc = (step.description or "").lower()
            for node_name in (*cls._COMPUTE_NODES, *cls._BUILTIN_NODES):
                if node_name.replace("_", " ") in desc or node_name in desc:
                    return node_name
            # Fallback: any compute step gets a generic no-op node
            return "_generic_compute"

        # Legacy heuristic for checklist steps (no explicit exec field)
        if execution == "" and cls._is_checklist_step(step):
            return "rust_best_practices_checklist"

        return None

    @staticmethod
    def _is_checklist_step(step: PlanStep) -> bool:
        description = (step.description or "").lower()
        return "checklist" in description and (
            "create" in description
            or "build" in description
            or "present" in description
            or "finalized" in description
        )

    @classmethod
    def _execute_compute_node(cls, step: PlanStep, compute_node: str) -> StepResult:
        """Dispatch to a registered compute node or return a generic placeholder.

        Language/domain-specific nodes must be registered by verticals via
        register_compute_node().  Unknown node names produce a generic result
        so deterministic steps never block execution on missing domain content.
        """
        if compute_node in cls._COMPUTE_NODES:
            return cls._COMPUTE_NODES[compute_node](step)
        # Generic fallback — no domain content; the agent step that follows can
        # elaborate if needed, or the vertical can register a named handler.
        description = (step.description or "").strip()
        output = f"Compute step: {description}"
        if "present" in description.lower():
            output = f"Step ready for review: {description}"
        return StepResult(
            success=True,
            output=output,
            tool_calls_used=0,
            metadata={
                "execution_mode": "compute_node",
                "compute_node": compute_node,
                "node_type": "deterministic_planning_step",
                "registered": compute_node in cls._COMPUTE_NODES,
            },
        )

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
            metadata = {
                "execution_mode": "hierarchical_team",
                "formation": getattr(result.formation, "value", str(result.formation)),
                "member_results": {
                    member_id: {
                        "success": member.success,
                        "error": member.error,
                        "tool_calls_used": member.tool_calls_used,
                    }
                    for member_id, member in result.member_results.items()
                },
            }
            fallback = PlanningTeamExecutionAdapter._successful_worker_fallback(
                result.member_results
            )
            if not success and fallback is not None:
                success = True
                output = fallback
                error = None
                metadata["used_successful_worker_fallback"] = True
        else:
            success = bool(result.get("success"))
            output = str(result.get("final_output", ""))
            error = result.get("error")
            tool_calls = int(result.get("total_tool_calls", 0) or 0)
            metadata = {
                "execution_mode": "hierarchical_team",
                "member_results": result.get("member_results", {}),
            }
            fallback = PlanningTeamExecutionAdapter._successful_worker_fallback(
                result.get("member_results", {})
            )
            if not success and fallback is not None:
                success = True
                output = fallback
                error = None
                metadata["used_successful_worker_fallback"] = True

        return StepResult(
            success=success,
            output=output,
            error=error,
            tool_calls_used=tool_calls,
            artifacts=[],
            metadata=metadata,
        )

    @staticmethod
    def _member_payload_to_step_result(payload: Any, member_id: str) -> StepResult:
        if not isinstance(payload, dict):
            return StepResult(
                success=True,
                output=str(payload),
                metadata={"execution_mode": "direct_step_worker", "member_id": member_id},
            )

        metadata = dict(payload.get("metadata") or {})
        metadata.setdefault("execution_mode", "direct_step_worker")
        metadata.setdefault("member_id", member_id)
        artifacts = metadata.get("artifacts") or metadata.get("changed_files") or []
        if not isinstance(artifacts, list):
            artifacts = [str(artifacts)]
        return StepResult(
            success=bool(payload.get("success")),
            output=str(payload.get("output", "") or ""),
            error=payload.get("error"),
            tool_calls_used=int(payload.get("tool_calls_used", 0) or 0),
            duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
            artifacts=[str(artifact) for artifact in artifacts],
            metadata=metadata,
        )

    @staticmethod
    def _successful_worker_fallback(member_results: Any) -> Optional[str]:
        """Return worker output when manager failed but every worker succeeded."""
        if not member_results:
            return None
        items = dict(member_results).items()
        worker_results = [
            member_result for member_id, member_result in items if member_id != "plan_manager"
        ]
        if not worker_results:
            return None
        if not all(
            PlanningTeamExecutionAdapter._member_success(member) for member in worker_results
        ):
            return None
        return "\n\n".join(
            output
            for output in (
                PlanningTeamExecutionAdapter._member_output(member) for member in worker_results
            )
            if output
        )

    @staticmethod
    def _member_success(member_result: Any) -> bool:
        if isinstance(member_result, dict):
            return bool(member_result.get("success"))
        return bool(getattr(member_result, "success", False))

    @staticmethod
    def _member_output(member_result: Any) -> str:
        if isinstance(member_result, dict):
            return str(member_result.get("output", "") or "")
        return str(getattr(member_result, "output", "") or "")

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
