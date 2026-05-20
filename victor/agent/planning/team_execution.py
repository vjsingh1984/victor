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
from victor.tools.core_tool_aliases import canonicalize_core_tool_name


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
            evidence_guidance = self._evidence_guidance_for_step(step)
            if evidence_guidance:
                item_task = f"{item_task}\n\n{evidence_guidance}"
            # Append per-step exit criteria so the sub-agent has explicit completion
            # requirements and cannot self-terminate after a shallow scan.
            if step.exit_criteria:
                criteria_str = "; ".join(step.exit_criteria)
                item_task = f"{item_task}\n\nVerification criteria (must satisfy before completing): {criteria_str}"
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
    _ARTIFACT_PRODUCES_RE = re.compile(
        r"(?:^|_)(?:checklist|report|summary|findings)(?:_|$)",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_synthesis_step(step: PlanStep) -> bool:
        """Return True when this step assembles/writes a report from prior findings."""
        if "write" in (step.allowed_tools or []):
            return True
        return bool(
            PlanningTeamExecutionAdapter._SYNTHESIS_KEYWORDS_RE.search(step.description or "")
        )

    @staticmethod
    def _produces_output_contract(produces_key: str) -> str:
        """Return the final-answer contract for a step that writes plan_state.

        List-like producers feed loops and conditionals, so they need compact
        line-oriented items.  Artifact producers such as checklist/report/findings
        feed humans or synthesis steps and must return the actual content, not a
        progress note or a single label.
        """
        key = str(produces_key or "").strip()
        if PlanningTeamExecutionAdapter._ARTIFACT_PRODUCES_RE.search(key):
            return (
                "OUTPUT FORMAT (required): return the concrete artifact for "
                f"plan_state key '{key}' as Markdown. Include clear section headings "
                "and bullet or numbered items with enough detail for the next step to "
                "use directly. Do not return a status update, progress note, intent "
                "statement, or a single-line placeholder. If there is genuinely "
                "nothing to report, output exactly: (none)"
            )
        return (
            "OUTPUT FORMAT (required): respond with a plain list — one item "
            "per line, no bullet symbols, no prose introduction, no trailing "
            f"commentary. Each line must be a single '{key}' item "
            "(e.g. a file path, identifier, or short label). If you have "
            "nothing to list, output exactly: (none)"
        )

    @staticmethod
    def _task_description_for_step(
        step: PlanStep,
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the task string for a worker, augmenting with prior-step context
        and output-format instructions when the step must produce a named list.

        Context injection priority:
          1. step.inputs declared — inject only those specific named outputs (precise)
          2. Synthesis/write step — inject all named outputs + truncated step_N dumps
          3. Default — inject all named outputs (non-step_N keys) only
        """
        task = step.description
        is_synthesis = PlanningTeamExecutionAdapter._is_synthesis_step(step)

        if plan_state:
            declared_inputs: list[str] = getattr(step, "inputs", []) or []
            context_lines: list[str] = []

            if declared_inputs:
                # Precise injection: only what this step declared it needs.
                for key in declared_inputs:
                    value = plan_state.get(key)
                    if value is None:
                        continue
                    if isinstance(value, list) and value:
                        items_str = ", ".join(str(v) for v in value[:20])
                        context_lines.append(f"- {key}: {items_str}")
                    elif str(value).strip():
                        context_lines.append(f"- {key}: {str(value).strip()[:400]}")
            else:
                # Fallback: inject all named outputs; for synthesis steps also include
                # truncated step_N content so the sub-agent can synthesize findings.
                for key, value in plan_state.items():
                    is_step_key = key.startswith("step_")
                    if is_step_key and not is_synthesis:
                        continue
                    if isinstance(value, list) and value:
                        items_str = ", ".join(str(v) for v in value[:20])
                        context_lines.append(f"- {key}: {items_str}")
                    elif isinstance(value, (str, bool, int, float)) and str(value).strip():
                        short = str(value).strip()
                        short = short[:600] if is_step_key else short[:200]
                        context_lines.append(f"- {key}: {short}")

            if context_lines:
                task = task + "\n\nContext from prior steps:\n" + "\n".join(context_lines)

        produces_key = step.context.get("produces", "")
        if produces_key:
            contract = PlanningTeamExecutionAdapter._produces_output_contract(produces_key)
            task = f"{task}\n\n{contract}"

        # Compute steps (execution="compute") that produce named output are dispatched
        # to the agent path when no registered handler exists.  Without guidance,
        # the model sometimes hallucinates tool names ("FirstResponderTool") or enters
        # a halted state instead of generating the structured content from knowledge.
        # Adding an explicit note suppresses that behaviour.
        step_execution = (step.execution or step.context.get("execution", "")).lower()
        if step_execution == "compute" and produces_key:
            task = (
                f"{task}\n\n"
                f"NOTE: This is a knowledge generation step — no tool calls are required. "
                f"Produce the content directly from your knowledge and reasoning. "
                f"Do not attempt to call any tools."
            )

        evidence_guidance = PlanningTeamExecutionAdapter._evidence_guidance_for_step(step)
        if evidence_guidance:
            task = f"{task}\n\n{evidence_guidance}"

        # Append exit criteria so the sub-agent has explicit completion requirements
        # and cannot self-terminate before satisfying them.  Loop iterations receive
        # these via _execute_loop_node; all other steps get them here.
        exit_criteria = getattr(step, "exit_criteria", []) or []
        if exit_criteria and step_execution != "loop":
            criteria_str = "; ".join(exit_criteria)
            task = (
                f"{task}\n\n"
                f"Verification criteria (must satisfy before completing): {criteria_str}"
            )

        return task

    @staticmethod
    def _evidence_guidance_for_step(step: PlanStep) -> str:
        """Return pre-run evidence guidance for analysis/review sub-agents.

        This mirrors the post-run evidence contract so agents know the success
        bar before they start. It intentionally avoids inventory/discovery steps,
        where listing files is the work.
        """
        description = str(step.description or "")
        desc = description.lower()
        step_type = getattr(step, "step_type", None)
        execution = (step.execution or step.context.get("execution", "")).lower()
        if execution in {"approval", "compute", "conditional", "tool", "checkpoint"}:
            return ""
        if step_type not in {StepType.RESEARCH, StepType.REVIEW}:
            return ""
        if re.search(r"\b(map|inventory|discover|enumerate|list|present|show)\b", desc):
            return ""
        if not re.search(r"\b(review|analy[sz]e|analysis|audit|inspect|scan)\b", desc):
            return ""

        minimum = (
            "at least 3 concrete findings"
            if re.search(r"\b(crate|workspace|cross-crate|codebase|module)\b", desc)
            else "concrete findings"
        )
        return (
            "Evidence requirements (must satisfy before completing): "
            "read the relevant source files, not only directory listings; "
            f"return {minimum} with file:line references when findings exist; "
            "if there are no findings, state `No findings` and name the files inspected; "
            "do not stop after inventory, planning, or an intent statement."
        )

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
        manager_allowed_tools = self._runtime_allowed_tools(current_step)
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
                allowed_tools=self._runtime_allowed_tools(step),
                reports_to=manager.id,
            )
            members[member.id] = self._adapt_member(
                member,
                team_id,
                execution_plan.id,
                plan_step_id=step.id,
            )

        return members

    def _runtime_allowed_tools(self, step: Optional[PlanStep]) -> Optional[list[str]]:
        """Return step tools filtered to tools the parent runtime actually exposes.

        Plan generation may include useful-but-optional tools such as ``shell`` even
        when the active profile or runtime has not registered them.  Passing such
        tools into a constrained sub-agent advertises an unusable tool schema and
        causes ``Unknown or disabled tool`` skips.  When we can inspect the parent
        runtime, keep the step's order but drop unavailable tools.  If runtime
        availability cannot be inspected (tests, lightweight adapters), preserve the
        original plan hints.
        """
        if step is None:
            return None

        requested = get_step_allowed_tools(step)
        if not requested:
            return None

        available = self._available_parent_tools()
        if available is None:
            return requested

        filtered: list[str] = []
        dropped: list[str] = []
        for tool_name in requested:
            canonical = canonicalize_core_tool_name(str(tool_name))
            if canonical in available:
                if canonical not in filtered:
                    filtered.append(canonical)
            else:
                dropped.append(str(tool_name))

        if dropped:
            import logging as _logging

            _logging.getLogger(__name__).info(
                "Planning step %s: dropped unavailable tool hint(s): %s " "(available=%s)",
                getattr(step, "id", "?"),
                dropped,
                sorted(available)[:12],
            )

        return filtered

    def _available_parent_tools(self) -> Optional[set[str]]:
        """Best-effort snapshot of tool names enabled in the parent runtime."""
        candidates: list[Any] = []

        getter = getattr(self.orchestrator, "get_enabled_tools", None)
        if callable(getter):
            try:
                enabled = getter()
                if enabled:
                    return {canonicalize_core_tool_name(str(name)) for name in enabled}
            except Exception:
                pass

        tools = getattr(self.orchestrator, "tools", None) or getattr(
            self.orchestrator, "tool_registry", None
        )
        if tools is not None:
            candidates.append(tools)

        for registry in candidates:
            try:
                if hasattr(registry, "list_tools"):
                    names: set[str] = set()
                    for item in registry.list_tools():
                        name = item if isinstance(item, str) else getattr(item, "name", None)
                        if name:
                            names.add(canonicalize_core_tool_name(str(name)))
                    if names:
                        return names
                if hasattr(registry, "get_tool_names"):
                    names = registry.get_tool_names()
                    if names:
                        return {canonicalize_core_tool_name(str(name)) for name in names}
            except Exception:
                continue

        return None

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
            match = _re.search(r"\[workspace\].*?members\s*=\s*\[(.*?)\]", content, _re.DOTALL)
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

    @staticmethod
    def _builtin_checklist_artifact(step: "PlanStep", plan_state: Dict[str, Any]) -> "StepResult":
        """Generate a deterministic planning checklist artifact.

        Checklist steps are scaffolding for later review work.  Sending them to a
        general sub-agent wastes provider time and has repeatedly timed out under
        slower models.  This node turns the step description into a concrete
        Markdown checklist without reading repository files.
        """
        del plan_state

        description = (step.description or "").strip()
        topic = description or "Review checklist"
        categories = PlanningTeamExecutionAdapter._extract_checklist_categories(description)
        if not categories:
            categories = [
                "Shared ownership and Arc/Rc selection",
                "Clone elimination and borrowing",
                "Immutable bindings and interior mutability",
                "Zero-copy data flow",
                "Concurrency and async correctness",
                "Error handling and unwrap hygiene",
                "Allocation and resource efficiency",
                "Trait design and dynamic dispatch",
                "Unsafe code and FFI boundaries",
                "Dependency and feature hygiene",
            ]

        lines = [
            f"# {topic}",
            "",
            "Use this checklist as the evaluation framework for downstream analysis.",
            "Record file:line evidence for every actionable finding.",
            "",
        ]
        for index, category in enumerate(categories, start=1):
            clean = category.strip(" .")
            if not clean:
                continue
            lines.extend(
                [
                    f"## {index}. {clean}",
                    f"- Check whether {clean.lower()} is relevant in each crate.",
                    "- Prefer borrowing, immutable data, and scoped ownership unless sharing is required.",
                    "- Record any performance, resource, correctness, or maintainability risk.",
                    "- Note concrete file paths and suggested fixes for actionable items.",
                    "",
                ]
            )

        return StepResult(
            success=True,
            output="\n".join(lines).strip(),
            tool_calls_used=0,
            metadata={
                "execution_mode": "builtin_compute",
                "compute_node": "_checklist_artifact",
                "node_type": "deterministic_planning_step",
                "category_count": len(categories),
            },
        )

    @staticmethod
    def _extract_checklist_categories(description: str) -> list[str]:
        """Extract comma-separated checklist categories from a plan-step description."""
        if not description:
            return []
        match = re.search(r"\bcovering:\s*(.+)$", description, flags=re.IGNORECASE)
        raw = match.group(1) if match else description
        raw = re.sub(r"^create\s+(?:comprehensive\s+)?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\b(?:best practices|checklist)\b", "", raw, flags=re.IGNORECASE)
        parts = re.split(r",|\band\b", raw)
        categories: list[str] = []
        for part in parts:
            item = re.sub(r"\([^)]*\)", "", part).strip(" -.;:")
            item = re.sub(r"\s+", " ", item)
            if len(item) < 4:
                continue
            if item.lower() not in {c.lower() for c in categories}:
                categories.append(item)
        return categories[:18]

    @staticmethod
    def _builtin_cross_crate_findings(step: "PlanStep", plan_state: Dict[str, Any]) -> "StepResult":
        """Generate deterministic cross-crate findings from local workspace files."""
        from collections import Counter, defaultdict
        from pathlib import Path

        del step

        members = PlanningTeamExecutionAdapter._workspace_members_from_state(plan_state)
        cargo_root = Path("rust") / "Cargo.toml"
        if not members and cargo_root.exists():
            parsed = PlanningTeamExecutionAdapter._builtin_parse_workspace_members(
                PlanStep(
                    id="_workspace_members",
                    description="Inventory workspace members from rust/Cargo.toml",
                ),
                {},
            )
            members = [line.strip() for line in parsed.output.splitlines() if line.strip()]
        if not members:
            members = sorted(str(path.parent) for path in Path("rust").glob("crates/*/Cargo.toml"))

        dep_versions: dict[str, list[str]] = defaultdict(list)
        path_edges: list[tuple[str, str, str]] = []
        pattern_counts: dict[str, Counter[str]] = {}
        public_arc_files: list[str] = []

        for member in members:
            crate_dir = Path(member)
            crate_name = crate_dir.name
            manifest = crate_dir / "Cargo.toml"
            if manifest.exists():
                for dep_name, dep_spec in PlanningTeamExecutionAdapter._parse_manifest_dependencies(
                    manifest.read_text(errors="ignore")
                ).items():
                    dep_versions[dep_name].append(dep_spec)
                    if "path" in dep_spec:
                        path_edges.append((crate_name, dep_name, dep_spec))

            counts: Counter[str] = Counter()
            for source in sorted((crate_dir / "src").rglob("*.rs")):
                text = source.read_text(errors="ignore")
                for pattern in ("Arc<", "Arc::clone", ".clone()", "Mutex<", "RwLock<", "Cow<"):
                    counts[pattern] += text.count(pattern)
                if "pub " in text and "Arc<" in text:
                    public_arc_files.append(str(source))
            pattern_counts[crate_name] = counts

        duplicate_deps = {
            dep: sorted(set(specs)) for dep, specs in dep_versions.items() if len(set(specs)) > 1
        }

        lines = [
            "# Cross-Crate Rust Findings",
            "",
            "## Dependency And Ownership Boundaries",
        ]
        if path_edges:
            for crate, dep, spec in path_edges:
                lines.append(
                    f"- `{crate}` depends on workspace crate `{dep}` via `{spec}`; "
                    "review API ownership contracts at this boundary for avoidable clones."
                )
        else:
            lines.append("- No path-based workspace dependency edges were found in manifests.")

        lines.extend(["", "## Shared Ownership And Clone Patterns"])
        emitted_pattern = False
        for crate, counts in sorted(pattern_counts.items()):
            interesting = {key: value for key, value in counts.items() if value and key != "Cow<"}
            if interesting:
                emitted_pattern = True
                summary = ", ".join(f"{key}={value}" for key, value in sorted(interesting.items()))
                lines.append(f"- `{crate}` contains shared ownership or clone signals: {summary}.")
        if not emitted_pattern:
            lines.append("- No Arc/Mutex/RwLock/clone pattern signals were found in Rust sources.")

        lines.extend(["", "## Public API Surface"])
        if public_arc_files:
            for file_path in public_arc_files[:20]:
                lines.append(
                    f"- `{file_path}` contains public items near `Arc<T>` usage; verify callers "
                    "need thread-safe shared ownership instead of borrowing, `Rc<T>`, or owned data."
                )
        else:
            lines.append("- No public Rust source file combines `pub` items with `Arc<T>` usage.")

        lines.extend(["", "## Dependency Version Consistency"])
        if duplicate_deps:
            for dep, specs in sorted(duplicate_deps.items()):
                lines.append(
                    f"- Dependency `{dep}` uses multiple specs across crates: {', '.join(specs)}."
                )
        else:
            lines.append("- Workspace crate dependency specs appear consistent across manifests.")

        per_crate_findings = plan_state.get("per_crate_findings")
        if per_crate_findings:
            lines.extend(
                [
                    "",
                    "## Downstream Synthesis Inputs",
                    f"- Per-crate review produced {len(per_crate_findings)} crate finding group(s); "
                    "synthesize these with the cross-crate dependency and ownership signals above.",
                ]
            )

        return StepResult(
            success=True,
            output="\n".join(lines).strip(),
            tool_calls_used=0,
            metadata={
                "execution_mode": "builtin_compute",
                "compute_node": "_cross_crate_findings",
                "node_type": "deterministic_planning_step",
                "crate_count": len(members),
                "dependency_edge_count": len(path_edges),
                "public_arc_file_count": len(public_arc_files),
            },
        )

    @staticmethod
    def _workspace_members_from_state(plan_state: Dict[str, Any]) -> list[str]:
        from pathlib import Path

        raw = plan_state.get("workspace_members") or plan_state.get("crate_file_inventory")
        if isinstance(raw, list):
            candidates = [str(item).strip() for item in raw]
        elif isinstance(raw, str):
            candidates = [line.strip() for line in raw.splitlines()]
        else:
            candidates = []
        members: list[str] = []
        for candidate in candidates:
            if not candidate or candidate.startswith("["):
                continue
            path = candidate.split(":", 1)[0].strip()
            if "/src/" in path:
                path = path.split("/src/", 1)[0]
            if path.endswith("/Cargo.toml"):
                path = path.rsplit("/", 1)[0]
            if path and path not in members and Path(path).joinpath("Cargo.toml").exists():
                members.append(path)
        return members

    @staticmethod
    def _parse_manifest_dependencies(manifest_text: str) -> dict[str, str]:
        deps: dict[str, str] = {}
        current_section = ""
        for raw_line in manifest_text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line.strip("[]")
                continue
            if current_section not in {
                "dependencies",
                "dev-dependencies",
                "build-dependencies",
            }:
                continue
            if "=" not in line:
                continue
            name, spec = line.split("=", 1)
            deps[name.strip()] = spec.strip()
        return deps

    @staticmethod
    def _builtin_rust_hotspot_scan(step: "PlanStep", plan_state: Dict[str, Any]) -> "StepResult":
        """Count Rust hotspot patterns across workspace source files."""
        from collections import Counter, defaultdict
        from pathlib import Path

        del step, plan_state

        patterns = {
            "Arc::new": "Arc::new",
            ".clone()": ".clone()",
            "Arc<Mutex>": "Arc<Mutex",
            "Arc<RwLock>": "Arc<RwLock",
            "to_owned()": ".to_owned()",
            "to_string()": ".to_string()",
            "format!": "format!",
        }
        files = sorted(Path("rust").glob("**/*.rs"))
        crate_counts: dict[str, Counter[str]] = defaultdict(Counter)
        total_counts: Counter[str] = Counter()
        top_files: list[tuple[int, str, Counter[str]]] = []

        for file_path in files:
            text = file_path.read_text(errors="ignore")
            file_counts = Counter({label: text.count(needle) for label, needle in patterns.items()})
            file_counts = Counter({label: count for label, count in file_counts.items() if count})
            if not file_counts:
                continue
            crate = PlanningTeamExecutionAdapter._crate_name_for_rust_file(file_path)
            crate_counts[crate].update(file_counts)
            total_counts.update(file_counts)
            top_files.append((sum(file_counts.values()), str(file_path), file_counts))

        top_files.sort(reverse=True)
        lines = [
            "# Rust Quantitative Hotspot Scan",
            "",
            f"Scanned {len(files)} Rust source file(s) under `rust/`.",
            "",
            "## Pattern Totals",
        ]
        for label in patterns:
            lines.append(f"- `{label}`: {total_counts.get(label, 0)}")

        lines.extend(["", "## Crate Hotspot Severity"])
        if crate_counts:
            for crate, counts in sorted(
                crate_counts.items(),
                key=lambda item: sum(item[1].values()),
                reverse=True,
            ):
                total = sum(counts.values())
                severity = "high" if total >= 25 else "medium" if total >= 8 else "low"
                summary = ", ".join(f"{label}={counts.get(label, 0)}" for label in patterns)
                lines.append(f"- `{crate}`: {severity} ({total} signal(s)); {summary}")
        else:
            lines.append("- No hotspot patterns were found.")

        lines.extend(["", "## Top Files"])
        for total, file_path, counts in top_files[:15]:
            summary = ", ".join(f"{label}={count}" for label, count in counts.items())
            lines.append(f"- `{file_path}`: {total} signal(s); {summary}")

        return StepResult(
            success=True,
            output="\n".join(lines).strip(),
            tool_calls_used=0,
            metadata={
                "execution_mode": "builtin_compute",
                "compute_node": "_rust_hotspot_scan",
                "node_type": "deterministic_planning_step",
                "file_count": len(files),
                "total_signal_count": sum(total_counts.values()),
            },
        )

    @staticmethod
    def _builtin_rust_crate_review(step: "PlanStep", plan_state: Dict[str, Any]) -> "StepResult":
        """Generate deterministic file:line Rust review findings for one crate.

        Per-crate Rust review steps are narrow source scans. Running a general
        sub-agent for each crate has proven flaky with smaller provider budgets:
        workers often list files and stop before producing findings. This scanner
        gives downstream synthesis stable evidence while keeping deeper semantic
        interpretation as a follow-up human/model task.
        """
        from collections import Counter
        from pathlib import Path

        members = PlanningTeamExecutionAdapter._workspace_members_from_state(plan_state)
        if not members:
            members = sorted(str(path.parent) for path in Path("rust").glob("crates/*/Cargo.toml"))

        crate_dir = PlanningTeamExecutionAdapter._rust_crate_dir_for_step(step, members)
        desc = (step.description or "").lower()
        review_all_crates = crate_dir is None and re.search(
            r"\b(each|every|all)\b.*\b(crate|workspace member|workspace)\b|"
            r"\bper-crate\b|\bworkspace member\b",
            desc,
        )
        crate_dirs = [crate_dir] if crate_dir is not None else []
        if review_all_crates:
            crate_dirs = [
                Path(member) for member in members if Path(member).joinpath("Cargo.toml").exists()
            ]
            if not crate_dirs:
                crate_dirs = sorted(
                    path.parent for path in Path("rust").glob("crates/*/Cargo.toml")
                )

        if not crate_dirs:
            return StepResult(
                success=False,
                output="(none)",
                error="Could not resolve Rust crate directory for review step",
                tool_calls_used=0,
                metadata={
                    "execution_mode": "builtin_compute",
                    "compute_node": "_rust_crate_review",
                    "resolved": False,
                },
            )

        patterns = {
            "Arc::new": "Arc allocation",
            "Arc<": "thread-safe shared ownership",
            "Arc::clone": "shared ownership clone",
            ".clone()": "owned clone",
            "RwLock<": "read/write lock",
            "Mutex<": "exclusive lock",
            "Cow<": "clone-on-write type",
            ".to_string()": "string allocation",
            "format!": "formatted string allocation",
            "unsafe": "unsafe block or marker",
        }
        total_counts: Counter[str] = Counter()
        total_files = 0
        total_findings = 0
        lines = [
            (
                "# Rust Crate Review: workspace"
                if len(crate_dirs) > 1
                else f"# Rust Crate Review: {crate_dirs[0].name}"
            ),
            "",
        ]

        for crate_dir in crate_dirs:
            source_dir = crate_dir / "src"
            files = sorted(source_dir.rglob("*.rs")) if source_dir.exists() else []
            counts: Counter[str] = Counter()
            findings: list[tuple[str, int, str, str]] = []

            for file_path in files:
                try:
                    source_lines = file_path.read_text(errors="ignore").splitlines()
                except OSError:
                    continue
                for line_no, line in enumerate(source_lines, start=1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("//"):
                        continue
                    for pattern, label in patterns.items():
                        occurrences = line.count(pattern)
                        if occurrences <= 0:
                            continue
                        counts[pattern] += occurrences
                        if len(findings) < 60:
                            findings.append((str(file_path), line_no, pattern, label))

            total_counts.update(counts)
            total_files += len(files)
            total_findings += len(findings)
            crate_name = crate_dir.name
            lines.extend(
                [
                    f"## {crate_name}",
                    "",
                    f"Scanned {len(files)} Rust source file(s) under `{source_dir}`.",
                    "",
                    "### Signal Counts",
                ]
            )
            if counts:
                for pattern in patterns:
                    if counts.get(pattern, 0):
                        lines.append(f"- `{pattern}`: {counts[pattern]}")
            else:
                lines.append("- No Arc/clone/lock/allocation/unsafe review signals were found.")

            lines.extend(["", "### File-Line Findings"])
            if findings:
                for file_path, line_no, pattern, label in findings:
                    recommendation = PlanningTeamExecutionAdapter._rust_pattern_recommendation(
                        pattern
                    )
                    lines.append(
                        f"- `{file_path}:{line_no}`: `{pattern}` ({label}); {recommendation}"
                    )
            else:
                lines.append(
                    "- No file-line findings from the deterministic scan. "
                    "A semantic review may still be useful for API design."
                )
            lines.append("")

        lines.extend(["", "## Review Notes"])
        if total_counts.get("Arc<", 0) or total_counts.get("Arc::new", 0):
            lines.append(
                "- Verify each `Arc` crosses a thread/task boundary; prefer owned values, "
                "borrows, or `Rc` for single-threaded sharing."
            )
        if total_counts.get(".clone()", 0):
            lines.append(
                "- Audit clones at API boundaries first; owned key insertion and PyO3 "
                "boundary conversions may be intentional."
            )
        if total_counts.get(".to_string()", 0) or total_counts.get("format!", 0):
            lines.append(
                "- Treat string allocation counts as hotspot signals, not automatic bugs; "
                "prioritize loops and repeated conversions."
            )
        if not any(total_counts.values()):
            lines.append("- No immediate high-signal ownership or allocation hotspot was detected.")

        return StepResult(
            success=True,
            output="\n".join(lines).strip(),
            tool_calls_used=0,
            metadata={
                "execution_mode": "builtin_compute",
                "compute_node": "_rust_crate_review",
                "node_type": "deterministic_planning_step",
                "crate": crate_dirs[0].name if len(crate_dirs) == 1 else "workspace",
                "crate_count": len(crate_dirs),
                "file_count": total_files,
                "finding_count": total_findings,
                "total_signal_count": sum(total_counts.values()),
            },
        )

    @staticmethod
    def _builtin_rust_prioritized_report(
        step: "PlanStep", plan_state: Dict[str, Any]
    ) -> "StepResult":
        """Synthesize Rust review findings from plan_state into a deterministic report."""
        del step

        per_crate = PlanningTeamExecutionAdapter._plan_state_markdown(
            plan_state.get("per_crate_findings")
            or plan_state.get("crate_findings")
            or plan_state.get("rust_crate_findings")
        )
        cross_crate = PlanningTeamExecutionAdapter._plan_state_markdown(
            plan_state.get("cross_crate_findings")
        )
        hotspot_scan = PlanningTeamExecutionAdapter._plan_state_markdown(
            plan_state.get("rust_hotspot_scan")
            or plan_state.get("hotspot_scan")
            or plan_state.get("quantitative_scan")
        )
        checklist = PlanningTeamExecutionAdapter._plan_state_markdown(
            plan_state.get("best_practices_checklist")
        )

        evidence_sections = [
            value for value in (per_crate, cross_crate, hotspot_scan) if value.strip()
        ]
        if not evidence_sections:
            output = (
                "# Rust Best Practices Report\n\n"
                "No Rust review findings were available in plan_state. Re-run the per-crate "
                "and cross-crate analysis steps before synthesizing a final report."
            )
            return StepResult(
                success=True,
                output=output,
                tool_calls_used=0,
                metadata={
                    "execution_mode": "builtin_compute",
                    "compute_node": "_rust_prioritized_report",
                    "node_type": "deterministic_planning_step",
                    "source_section_count": 0,
                },
            )

        combined = "\n".join(evidence_sections).lower()
        recommendations: list[str] = []
        if "arc::new" in combined or "arc<" in combined or "arc::clone" in combined:
            recommendations.append(
                "High impact: audit `Arc` usage first. Keep `Arc` only where ownership crosses "
                "thread/task boundaries; otherwise prefer borrows, owned values, or `Rc`."
            )
        if ".clone()" in combined:
            recommendations.append(
                "High impact: review clone hotspots at public API and loop boundaries. Replace "
                "avoidable clones with borrowed parameters or ownership transfer."
            )
        if "rwlock<" in combined or "mutex<" in combined:
            recommendations.append(
                "Medium impact: inspect lock granularity and contention. Prefer immutable snapshots "
                "or narrower critical sections where shared state is read-heavy."
            )
        if ".to_string()" in combined or "format!" in combined:
            recommendations.append(
                "Medium impact: treat string-allocation counts as hotspot signals. Optimize only "
                "repeated conversions or hot loops, not one-off formatting."
            )
        if "unsafe" in combined:
            recommendations.append(
                "Correctness: audit unsafe blocks and document invariants required for soundness."
            )
        if not recommendations:
            recommendations.append(
                "No high-signal Arc/clone/lock/allocation pattern was found. Prioritize semantic "
                "API review and tests before refactoring."
            )

        lines = [
            "# Rust Best Practices Report",
            "",
            "## Executive Summary",
            "- Per-crate and cross-crate Rust evidence was synthesized from prior plan outputs.",
            "- Recommendations below are ranked by expected impact and implementation risk.",
            "",
            "## Prioritized Recommendations",
        ]
        for index, recommendation in enumerate(recommendations, start=1):
            lines.append(f"{index}. {recommendation}")

        if per_crate:
            lines.extend(["", "## Per-Crate Evidence", per_crate])
        if cross_crate:
            lines.extend(["", "## Cross-Crate Evidence", cross_crate])
        if hotspot_scan:
            lines.extend(["", "## Quantitative Hotspot Evidence", hotspot_scan])
        if checklist:
            lines.extend(["", "## Checklist Used", checklist[:3000]])

        return StepResult(
            success=True,
            output="\n".join(lines).strip(),
            tool_calls_used=0,
            metadata={
                "execution_mode": "builtin_compute",
                "compute_node": "_rust_prioritized_report",
                "node_type": "deterministic_planning_step",
                "source_section_count": len(evidence_sections),
                "recommendation_count": len(recommendations),
            },
        )

    @staticmethod
    def _plan_state_markdown(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            if not parts:
                return ""
            if len(parts) == 1:
                return parts[0]
            return "\n".join(f"- {part}" for part in parts)
        return str(value).strip()

    @staticmethod
    def _rust_crate_dir_for_step(step: "PlanStep", members: list[str]) -> Any:
        from pathlib import Path

        desc = (step.description or "").lower()
        candidates = [Path(member) for member in members]
        if not candidates:
            candidates = sorted(path.parent for path in Path("rust").glob("crates/*/Cargo.toml"))

        def aliases(crate_dir: Path) -> set[str]:
            names = {crate_dir.name.lower(), crate_dir.name.replace("-", " ").lower()}
            manifest = crate_dir / "Cargo.toml"
            if manifest.exists():
                match = re.search(
                    r'(?m)^\s*name\s*=\s*"([^"]+)"',
                    manifest.read_text(errors="ignore"),
                )
                if match:
                    package = match.group(1).lower()
                    names.add(package)
                    names.add(package.replace("-", " "))
            return names

        for crate_dir in candidates:
            if any(name and name in desc for name in aliases(crate_dir)):
                return crate_dir
        return None

    @staticmethod
    def _rust_pattern_recommendation(pattern: str) -> str:
        if pattern in {"Arc::new", "Arc<", "Arc::clone"}:
            return "confirm shared ownership is required and not replacing a simple borrow"
        if pattern == ".clone()":
            return "check whether the callee can borrow or consume the existing value"
        if pattern in {"RwLock<", "Mutex<"}:
            return "confirm lock granularity and contention behavior are appropriate"
        if pattern == "Cow<":
            return "good zero-copy candidate; verify it avoids clone-heavy call paths"
        if pattern in {".to_string()", "format!"}:
            return "review if this occurs in a loop or hot path before optimizing"
        if pattern == "unsafe":
            return "audit invariants and document why safe Rust cannot express this"
        return "review for ownership and resource efficiency"

    @staticmethod
    def _crate_name_for_rust_file(file_path: Any) -> str:
        parts = list(getattr(file_path, "parts", ()))
        if "crates" in parts:
            index = parts.index("crates")
            if index + 1 < len(parts):
                return str(parts[index + 1])
        return "workspace"

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
        produces_key = step.context.get("produces", "")
        desc = (step.description or "").lower()
        if execution == "compute":
            explicit_node = step.context.get("node", "")
            if explicit_node and (
                explicit_node in cls._COMPUTE_NODES or explicit_node in cls._BUILTIN_NODES
            ):
                return explicit_node
            if "_checklist_artifact" in cls._COMPUTE_NODES and (
                "checklist" in produces_key.lower() or "checklist" in desc
            ):
                return "_checklist_artifact"
            # Find a registered node whose name matches the description
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
        if produces_key == "workspace_members" and "_workspace_members" in cls._COMPUTE_NODES:
            if "cargo.toml" in (step.description or "").lower():
                return "_workspace_members"
        if "_checklist_artifact" in cls._COMPUTE_NODES and (
            "checklist" in produces_key.lower() or "checklist" in desc
        ):
            return "_checklist_artifact"
        if "_cross_crate_findings" in cls._COMPUTE_NODES and (
            produces_key == "cross_crate_findings" or "cross-crate" in desc
        ):
            return "_cross_crate_findings"
        if "_rust_crate_review" in cls._COMPUTE_NODES and re.search(
            r"\breview\b.*\bcrate\b|\bcrate\b.*\breview\b",
            desc,
        ):
            if "rust" in desc or "arc" in desc or "clone" in desc or "ownership" in desc:
                return "_rust_crate_review"
        if "_rust_hotspot_scan" in cls._COMPUTE_NODES and (
            "quantitative scan" in desc
            or "hotspot" in desc
            or "arc::new" in desc
            or "to_owned" in desc
        ):
            return "_rust_hotspot_scan"
        if "_rust_prioritized_report" in cls._COMPUTE_NODES and (
            produces_key
            in {"final_report", "prioritized_report", "ranked_findings", "rust_report"}
            or (
                re.search(r"\b(synthesize|summarize|compile|write)\b", desc)
                and "report" in desc
                and ("rust" in desc or "arc" in desc or "findings" in desc)
            )
        ):
            return "_rust_prioritized_report"

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
PlanningTeamExecutionAdapter.register_compute_node(
    "_checklist_artifact",
    PlanningTeamExecutionAdapter._builtin_checklist_artifact,
)
PlanningTeamExecutionAdapter.register_compute_node(
    "_cross_crate_findings",
    PlanningTeamExecutionAdapter._builtin_cross_crate_findings,
)
PlanningTeamExecutionAdapter.register_compute_node(
    "_rust_hotspot_scan",
    PlanningTeamExecutionAdapter._builtin_rust_hotspot_scan,
)
PlanningTeamExecutionAdapter.register_compute_node(
    "_rust_crate_review",
    PlanningTeamExecutionAdapter._builtin_rust_crate_review,
)
PlanningTeamExecutionAdapter.register_compute_node(
    "_rust_prioritized_report",
    PlanningTeamExecutionAdapter._builtin_rust_prioritized_report,
)
