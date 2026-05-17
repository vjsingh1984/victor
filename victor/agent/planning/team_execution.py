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
        approval_callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.sub_agent_orchestrator = sub_agent_orchestrator
        self._coordinator_factory = coordinator_factory
        # Optional async callback ``(step, context) -> (approved: bool, feedback: str)``.
        # When None, approval nodes auto-approve and record a checkpoint marker.
        self._approval_callback = approval_callback

    # Node execution types that require the team-adapter path.
    # AutonomousPlanner (the fallback) has no awareness of these types.
    _TEAM_REQUIRED_EXECUTION_TYPES: frozenset = frozenset(
        {"compute", "tool", "loop", "conditional", "approval", "checkpoint"}
    )

    def should_use_team(self, plan: ReadableTaskPlan) -> bool:
        """Return whether this plan requires team-adapter execution.

        Always True for COMPLEX plans.  Also True when any step declares an
        advanced execution type (compute, loop, conditional, approval) because
        AutonomousPlanner — the fallback — has no dispatch logic for these.
        Falls back to exploratory-step counting for SIMPLE/MODERATE plans that
        use only the default agent execution path.
        """
        if plan.complexity == TaskComplexity.COMPLEX:
            return True

        exploratory_steps = 0
        for step_data in plan.steps:
            if isinstance(step_data, dict):
                exec_type = str(step_data.get("exec", step_data.get("execution", ""))).lower()
                step_type = str(step_data.get("type", "")).lower()
            elif len(step_data) >= 2:
                exec_type = str(step_data[5]).lower() if len(step_data) > 5 else ""
                step_type = str(step_data[1]).lower()
            else:
                continue
            if exec_type in self._TEAM_REQUIRED_EXECUTION_TYPES:
                return True
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
          1. compute      — deterministic function, no model call
          2. tool         — deterministic tool-only subagent, no reasoning overhead
          3. loop         — iterate over plan_state collection, one subagent per item
          4. conditional  — evaluate condition on plan_state, branch route downstream steps
          5. approval     — user checkpoint before continuing
          6. team         — UnifiedTeamCoordinator formation
          7. agent        — default single-worker path
        """
        resolved_plan_state = plan_state or {}

        # 1. Compute node — no model, no spawn; receives plan_state so it can
        # read outputs produced by earlier tool/compute steps.
        compute_node = self._compute_node_for_step(step)
        if compute_node is not None:
            return self._execute_compute_node(step, compute_node, resolved_plan_state)

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

        # 2. Tool node — single subagent, tool-only
        if execution == "tool":
            return await self._execute_tool_node(
                step, execution_plan, team_id, context, resolved_plan_state
            )

        # 3. Loop node — iterate over a plan-state collection
        if execution == "loop":
            return await self._execute_loop_node(
                step, execution_plan, team_id, context, resolved_plan_state
            )

        # 4. Conditional node — evaluate condition, record branch decision in metadata
        if execution == "conditional":
            return self._execute_conditional_node(step, resolved_plan_state)

        # 5. Approval node — user checkpoint
        if execution in ("approval", "checkpoint"):
            return await self._execute_approval_node(step, context)

        # 6. Team node — explicit multi-agent formation
        if execution == "team":
            return await self._execute_team_node(
                step, execution_plan, team_id, context, resolved_plan_state
            )

        # 7. Default: single worker (agent)
        task = self._task_description_for_step(step, resolved_plan_state)
        if self._should_execute_step_directly(execution_plan, step):
            members = self._build_members(execution_plan, team_id, current_step=step)
            worker = next(
                member for member_id, member in members.items() if member_id != "plan_manager"
            )
            payload = await worker.execute_task(task, context)
            return self._member_payload_to_step_result(payload, worker.id)

        coordinator = self._create_coordinator()
        members = self._build_members(execution_plan, team_id, current_step=step)
        manager = members["plan_manager"]

        coordinator.set_formation(TeamFormation.HIERARCHICAL)
        coordinator.set_manager(manager)
        for member_id, member in members.items():
            if member_id != manager.id:
                coordinator.add_member(member)

        result = await coordinator.execute_task(task, context)
        return self._team_result_to_step_result(result)

    async def _execute_tool_node(
        self,
        step: PlanStep,
        execution_plan: "ExecutionPlan",
        team_id: str,
        context: Dict[str, Any],
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Run a deterministic tool-only step via a single subagent worker."""
        members = self._build_members(execution_plan, team_id, current_step=step)
        worker = next(m for mid, m in members.items() if mid != "plan_manager")
        task = self._task_description_for_step(step, plan_state)
        payload = await worker.execute_task(task, context)
        result = self._member_payload_to_step_result(payload, worker.id)
        result.metadata["execution_mode"] = "tool_node"
        return result

    async def _execute_team_node(
        self,
        step: PlanStep,
        execution_plan: "ExecutionPlan",
        team_id: str,
        context: Dict[str, Any],
        plan_state: Optional[Dict[str, Any]] = None,
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
        task = self._task_description_for_step(step, plan_state)
        result = await coordinator.execute_task(task, context)
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
            child_session_id = self._child_session_id(
                parent_session_id, team_id, f"{step.id}_loop_{index}"
            )
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
            early_stopped = False
            for i, item in enumerate(items):
                pair = await _run_item(i, item)
                pairs.append(pair)
                # Exit criteria check after each iteration (sequential only)
                if step.exit_criteria:
                    accumulated = "\n".join(p[1].output for p in pairs)
                    if self._exit_criteria_met(step.exit_criteria, accumulated):
                        early_stopped = True
                        break
            else:
                early_stopped = False

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
        meta: Dict[str, Any] = {
            "execution_mode": "loop_node",
            "loop_items": items,
            "loop_items_count": len(items),
            "items_executed": len(pairs),
            "failed_items": failed_items,
            "parallel": parallel,
        }
        if not parallel:
            meta["early_stopped"] = early_stopped
        return StepResult(
            success=success,
            output="\n\n".join(outputs),
            error=f"Failed items: {failed_items}" if failed_items else None,
            tool_calls_used=total_tool_calls,
            metadata=meta,
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

            # Fallback: find the plan_state key whose name overlaps most with
            # loop_over, then try to fall back to any non-empty list value.
            if raw is None:
                lo_words = {
                    w.rstrip("s") for w in loop_over.replace("_", " ").split() if len(w) > 3
                }
                best_val = None
                best_score = 0
                for k, v in plan_state.items():
                    if k.startswith("step_") or not isinstance(v, list) or not v:
                        continue
                    k_words = {w.rstrip("s") for w in k.replace("_", " ").split()}
                    score = len(lo_words & k_words)
                    if score > best_score:
                        best_score = score
                        best_val = v
                if best_val is not None:
                    return [str(i) for i in best_val if str(i).strip()]
        return []

    @staticmethod
    def _exit_criteria_met(criteria: list[str], output: str) -> bool:
        """Return True when all exit criteria appear in accumulated output."""
        output_lower = output.lower()
        return all(c.lower() in output_lower for c in criteria)

    # ---------------------------------------------------------------------------
    # Conditional node
    # ---------------------------------------------------------------------------

    def _execute_conditional_node(
        self,
        step: PlanStep,
        plan_state: Dict[str, Any],
    ) -> StepResult:
        """Evaluate a plan-state condition and record which branch to skip.

        The ``skip_step_ids`` list in the result metadata is read by the runtime
        to mark inactive branch steps as SKIPPED before the next iteration.

        Context keys:
          ``condition_on`` — plan_state key to evaluate
          ``condition``    — ``"non_empty"`` (default), ``"multiple"``,
                             ``"single"``, ``"empty"``, or ``"truthy"``
          ``produces``     — optional plan_state key to store the bool result
          ``branches``     — ``{"true": [step_ids], "false": [step_ids]}``
        """
        import logging as _logging

        _log = _logging.getLogger(__name__)

        condition_on = step.context.get("condition_on", "")
        condition = step.context.get("condition", "non_empty")
        branches: Dict[str, Any] = step.context.get("branches", {})
        value = plan_state.get(condition_on) if condition_on else None
        value_source = (
            f"plan_state['{condition_on}']" if condition_on and value is not None else "none"
        )

        # Fallback A: key name mismatch — find a plan_state key whose words
        # overlap with condition_on (handles LLM naming variation).
        if condition_on and value is None:
            _log.info(
                "Conditional step %s: key '%s' missing from plan_state %s; trying word-overlap fallback",
                step.id,
                condition_on,
                [k for k in plan_state if not k.startswith("step_")],
            )
            cond_words = {
                w.rstrip("s") for w in condition_on.replace("_", " ").split() if len(w) > 3
            }
            for k, v in plan_state.items():
                if not k.startswith("step_") and isinstance(v, list) and v:
                    key_words = {w.rstrip("s") for w in k.replace("_", " ").split()}
                    if cond_words & key_words:
                        value = v
                        value_source = f"fallback_word_overlap['{k}']"
                        _log.info(
                            "Conditional step %s: resolved '%s' via word-overlap key '%s' (%d items)",
                            step.id,
                            condition_on,
                            k,
                            len(v),
                        )
                        break

        # Fallback B: condition_on is empty or still unresolved — pick the first
        # non-empty list in plan_state that isn't a raw step dump (step_N keys).
        # This handles planning schemas where the LLM never sets condition_on.
        if value is None:
            for k, v in plan_state.items():
                if not k.startswith("step_") and isinstance(v, list) and v:
                    value = v
                    value_source = f"fallback_first_list['{k}']"
                    _log.info(
                        "Conditional step %s: condition_on='%s' unresolved; "
                        "using first plan_state list '%s' (%d items) as fallback",
                        step.id,
                        condition_on or "<empty>",
                        k,
                        len(v),
                    )
                    break

        result = self._evaluate_condition(condition, value)

        inactive = "false" if result else "true"
        skip_ids = [str(s) for s in (branches.get(inactive) or [])]

        produces = step.context.get("produces", "")
        if produces:
            plan_state[produces] = result

        label = f"'{condition}' on '{condition_on}'" if condition_on else f"'{condition}'"
        _log.info(
            "Conditional step %s: %s = %s (value_source=%s, value=%r) → active=%s, skip=%s",
            step.id,
            label,
            result,
            value_source,
            value,
            "true" if result else "false",
            skip_ids,
        )
        return StepResult(
            success=True,
            output=f"Condition {label}: {result} — skipping {skip_ids if skip_ids else 'nothing'}.",
            tool_calls_used=0,
            metadata={
                "execution_mode": "conditional_node",
                "condition": condition,
                "condition_on": condition_on,
                "condition_result": result,
                "value_source": value_source,
                "skip_step_ids": skip_ids,
                "active_branch": "true" if result else "false",
            },
        )

    @staticmethod
    def _evaluate_condition(condition: str, value: Any) -> bool:
        """Evaluate a named condition against a plan-state value."""
        if condition == "non_empty" or condition == "truthy":
            return bool(value)
        if condition == "empty":
            return not bool(value)
        if condition == "multiple":
            return isinstance(value, (list, tuple)) and len(value) > 1
        if condition == "single":
            return isinstance(value, (list, tuple)) and len(value) == 1
        return bool(value)

    # ---------------------------------------------------------------------------
    # Approval node
    # ---------------------------------------------------------------------------

    async def _execute_approval_node(
        self,
        step: PlanStep,
        context: Dict[str, Any],
    ) -> StepResult:
        """Pause execution for a user approval checkpoint.

        When ``_approval_callback`` is set (injected at construction), it is called
        as ``await callback(step, context)`` and must return ``(approved: bool,
        feedback: str)``.  When unset, the checkpoint auto-approves so the plan
        continues without interruption — suitable for non-interactive contexts.
        """
        if self._approval_callback is not None:
            try:
                approved, feedback = await self._approval_callback(step, context)
            except Exception as exc:
                return StepResult(
                    success=False,
                    output="",
                    error=f"Approval callback raised: {exc}",
                    metadata={"execution_mode": "approval_node", "approved": False},
                )
            return StepResult(
                success=approved,
                output=feedback
                or (
                    f"Approved: {step.description}" if approved else f"Rejected: {step.description}"
                ),
                error=None if approved else f"User rejected: {feedback}",
                tool_calls_used=0,
                metadata={
                    "execution_mode": "approval_node",
                    "approved": approved,
                    "feedback": feedback,
                },
            )

        # No callback — auto-approve with a checkpoint marker
        return StepResult(
            success=True,
            output=f"Approval checkpoint (auto-approved): {step.description}",
            tool_calls_used=0,
            metadata={
                "execution_mode": "approval_node",
                "approved": True,
                "auto_approved": True,
            },
        )

    _SYNTHESIS_KEYWORDS_RE = re.compile(
        r"\b(synthesize|summarize|compile|report|document|write up|present)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_synthesis_step(step: PlanStep) -> bool:
        """Return True when this step assembles/writes a report from prior findings."""
        if "write" in (step.allowed_tools or []):
            return True
        return bool(
            PlanningTeamExecutionAdapter._SYNTHESIS_KEYWORDS_RE.search(
                step.description or ""
            )
        )

    @staticmethod
    def _task_description_for_step(
        step: PlanStep,
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the task string for a worker, augmenting with prior-step context
        and output-format instructions when the step must produce a named list.

        Injecting plan_state ensures downstream steps know exactly which paths/items
        were produced by earlier steps.  Named outputs (non-step_N keys) are always
        injected.  For synthesis/write steps, truncated raw step outputs (step_N keys)
        are also included so the sub-agent can synthesize without re-reading every file.
        """
        task = step.description
        is_synthesis = PlanningTeamExecutionAdapter._is_synthesis_step(step)

        # Inject prior-step context.
        if plan_state:
            context_lines: list[str] = []
            for key, value in plan_state.items():
                is_step_key = key.startswith("step_")
                # Named outputs always injected; step_N raw dumps only for synthesis steps.
                if is_step_key and not is_synthesis:
                    continue
                if isinstance(value, list) and value:
                    items_str = ", ".join(str(v) for v in value[:20])
                    context_lines.append(f"- {key}: {items_str}")
                elif isinstance(value, (str, bool, int, float)) and str(value).strip():
                    short = str(value).strip()
                    # Truncate raw step dumps to avoid overwhelming the context.
                    short = short[:600] if is_step_key else short[:200]
                    context_lines.append(f"- {key}: {short}")
            if context_lines:
                task = task + "\n\nContext from prior steps:\n" + "\n".join(context_lines)

        produces_key = step.context.get("produces", "")
        if produces_key:
            task = (
                f"{task}\n\n"
                f"OUTPUT FORMAT (required): respond with a plain list — one item "
                f"per line, no bullet symbols, no prose introduction, no trailing "
                f"commentary. Each line must be a single '{produces_key}' item "
                f"(e.g. a file path, identifier, or short label). If you have "
                f"nothing to list, output exactly: (none)"
            )
        return task

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

    # ---------------------------------------------------------------------------
    # Built-in compute node: workspace member discovery
    # ---------------------------------------------------------------------------

    @staticmethod
    def _builtin_parse_workspace_members(
        step: "PlanStep", plan_state: Dict[str, Any]
    ) -> "StepResult":
        """Deterministically parse workspace member paths from a Cargo.toml manifest.

        LLM-backed sub-agents reliably fail this purely mechanical parsing task —
        they echo the filename instead of extracting the ``members = [...]`` array.
        This node is registered as the canonical handler for any step that
        produces ``workspace_members`` and mentions cargo.toml or workspace members.
        """
        import re as _re
        from pathlib import Path

        desc = step.description or ""
        # Prefer an explicit path embedded in the step description
        cargo_match = _re.search(r"([\w./\\-]+[Cc]argo\.toml)", desc)
        candidates: list[str] = []
        if cargo_match:
            candidates.append(cargo_match.group(1))
        candidates.extend(["rust/Cargo.toml", "Cargo.toml"])

        cargo_path: Optional[Path] = None
        for candidate in candidates:
            p = Path(candidate)
            if p.exists():
                cargo_path = p
                break

        if cargo_path is None:
            return StepResult(success=True, output="(none)", tool_calls_used=0)

        try:
            content = cargo_path.read_text()
            # Match the [workspace] members = [...] array (handles multi-line)
            match = _re.search(
                r"\[workspace\].*?members\s*=\s*\[(.*?)\]", content, _re.DOTALL
            )
            if not match:
                return StepResult(success=True, output="(none)", tool_calls_used=0)

            workspace_dir = str(cargo_path.parent)
            members = _re.findall(r'"([^"]+)"', match.group(1))
            if workspace_dir and workspace_dir != ".":
                members = [f"{workspace_dir}/{m}" for m in members]

            output = "\n".join(members) if members else "(none)"
            return StepResult(
                success=True,
                output=output,
                tool_calls_used=0,
                metadata={
                    "execution_mode": "builtin_compute",
                    "compute_node": "_workspace_members",
                    "cargo_path": str(cargo_path),
                    "member_count": len(members),
                },
            )
        except Exception as e:
            return StepResult(
                success=False,
                output="(none)",
                error=str(e),
                tool_calls_used=0,
            )

    @classmethod
    def register_compute_node(
        cls,
        name: str,
        fn: "Callable[..., StepResult]",
    ) -> None:
        """Register a named deterministic compute node.

        ``fn`` receives ``(step: PlanStep, plan_state: Dict[str, Any])`` so it
        can read outputs from prior tool or compute steps.  Single-argument
        callables ``fn(step)`` are also accepted for backward compatibility.
        """
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
        3. Auto-detect workspace member discovery steps: when a step produces
           ``workspace_members`` and mentions Cargo.toml or "workspace member(s)",
           use the built-in deterministic parser regardless of execution type.
           LLM-backed agents reliably echo the filename rather than parsing members.
        4. Legacy heuristic for checklist steps (backward compat).
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
            # Compute steps that must produce structured output (list/data) for downstream
            # steps cannot use the generic no-op placeholder — they need real tool execution.
            # Return None to let the step fall through to the agent execution path.
            if step.context.get("produces", ""):
                return None
            # Fallback: non-producing compute steps use the generic placeholder
            return "_generic_compute"

        # Auto-detect workspace member discovery regardless of declared execution type.
        # This is a purely mechanical parsing task: LLMs (especially small budget models)
        # echo the Cargo.toml filename instead of extracting the members = [...] array.
        # Only triggers when the description explicitly mentions "cargo.toml" so that
        # generic "workspace members" steps for other languages fall through to the LLM.
        produces_key = step.context.get("produces", "")
        if produces_key == "workspace_members" and "_workspace_members" in cls._COMPUTE_NODES:
            if "cargo.toml" in (step.description or "").lower():
                return "_workspace_members"

        return None

    @classmethod
    def _execute_compute_node(
        cls,
        step: PlanStep,
        compute_node: str,
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Dispatch to a registered compute node or return a generic placeholder.

        Language/domain-specific nodes must be registered by verticals via
        register_compute_node().  Unknown node names produce a generic result
        so deterministic steps never block execution on missing domain content.

        ``plan_state`` is forwarded to the registered function so compute nodes
        can read outputs produced by earlier tool or compute steps.  The
        registered function signature should be ``fn(step, plan_state) -> StepResult``
        but callables that only accept ``step`` are also supported (plan_state
        is passed as a keyword argument and ignored if the function doesn't
        declare it).
        """
        if compute_node in cls._COMPUTE_NODES:
            fn = cls._COMPUTE_NODES[compute_node]
            try:
                return fn(step, plan_state or {})
            except TypeError:
                # Registered function uses the old single-argument signature.
                return fn(step)
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


# Register the built-in workspace member discovery node.
# This must happen after the class definition so the static method reference resolves.
PlanningTeamExecutionAdapter.register_compute_node(
    "_workspace_members",
    PlanningTeamExecutionAdapter._builtin_parse_workspace_members,
)
