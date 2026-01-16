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

"""Autonomous planner for goal-oriented task execution.

This module provides the AutonomousPlanner class that enables Victor to
accept high-level goals and autonomously:
1. Generate an execution plan
2. Execute steps sequentially or in parallel
3. Handle failures and adapt the plan
4. Report progress to the user

Design Principles:
- Goal-oriented: Accept natural language goals, produce actionable plans
- Transparent: Plans are shown to users before execution
- Safe: Critical steps require explicit approval
- Resumable: Plans can be paused and continued

Example Usage:
    planner = AutonomousPlanner(orchestrator)

    # Generate a plan for a goal
    plan = await planner.plan_for_goal("Add user authentication with JWT")

    # Show plan to user
    print(plan.to_markdown())

    # Execute with auto-approval for low-risk steps
    result = await planner.execute_plan(plan, auto_approve=True)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from victor.agent.planning.base import (
    ExecutionPlan,
    PlanResult,
    PlanStep,
    StepResult,
    StepStatus,
    StepType,
)

if TYPE_CHECKING:
    # Use protocol for type hint to avoid circular dependency (DIP compliance)
    from victor.protocols.agent import IAgentOrchestrator
    from victor.agent.subagents import SubAgentOrchestrator, SubAgentRole

logger = logging.getLogger(__name__)


# System prompt for plan generation
PLANNING_SYSTEM_PROMPT = """You are a planning assistant that breaks down complex goals into actionable steps.

When given a goal, create a structured plan with discrete steps. For each step:
1. Provide a clear, specific description
2. Estimate the number of tool calls needed
3. Identify dependencies on other steps
4. Determine the step type (research, implementation, testing, etc.)
5. Decide if user approval is needed (for destructive or risky actions)

Output your plan as a JSON array with this structure:
```json
[
  {
    "id": "1",
    "description": "Step description",
    "step_type": "research|planning|implementation|testing|review|deployment",
    "depends_on": [],
    "estimated_tool_calls": 10,
    "requires_approval": false,
    "sub_agent_role": "researcher|planner|executor|reviewer|tester|null"
  }
]
```

Guidelines:
- Break complex tasks into 3-10 steps
- Research/exploration steps should come before implementation
- Include testing steps when code changes are involved
- Mark deployment/destructive steps as requiring approval
- Use sub_agent_role for parallelizable research tasks
"""


class AutonomousPlanner:
    """Autonomous planner for goal-oriented execution.

    The planner takes high-level goals and produces structured execution plans.
    It can optionally execute plans automatically with configurable approval
    requirements.

    Attributes:
        orchestrator: Parent orchestrator for executing steps
        sub_agent_orchestrator: Optional sub-agent orchestrator for delegation
        active_plan: Currently active plan (if any)
        approval_callback: Callback for requesting user approval
    """

    def __init__(
        self,
        orchestrator: "IAgentOrchestrator",
        sub_agent_orchestrator: Optional["SubAgentOrchestrator"] = None,
        approval_callback: Optional[Callable[[str], bool]] = None,
    ):
        """Initialize the autonomous planner.

        Args:
            orchestrator: Parent orchestrator (via IAgentOrchestrator protocol) for LLM calls and tool execution
            sub_agent_orchestrator: Optional orchestrator for delegating to sub-agents
            approval_callback: Optional callback for user approval prompts
        """
        self.orchestrator = orchestrator
        self.sub_agent_orchestrator = sub_agent_orchestrator
        self.approval_callback = approval_callback or self._default_approval
        self.active_plan: Optional[ExecutionPlan] = None
        self._plan_history: List[ExecutionPlan] = []

        logger.info("AutonomousPlanner initialized")

    def _default_approval(self, message: str) -> bool:
        """Default approval callback (always returns False for safety)."""
        logger.warning(f"Step requires approval but no callback set: {message[:100]}...")
        return False

    async def plan_for_goal(
        self,
        goal: str,
        context: Optional[str] = None,
        max_steps: int = 10,
    ) -> ExecutionPlan:
        """Generate an execution plan for a goal.

        Uses the LLM to analyze the goal and generate a structured plan
        with discrete, actionable steps.

        Args:
            goal: High-level goal to plan for
            context: Optional additional context about the codebase/project
            max_steps: Maximum number of steps to generate

        Returns:
            ExecutionPlan with steps to achieve the goal

        Example:
            plan = await planner.plan_for_goal(
                "Implement user authentication with JWT",
                context="We're using FastAPI with SQLAlchemy"
            )
        """
        logger.info(f"Planning for goal: {goal[:100]}...")

        # Build the planning prompt
        prompt = self._build_planning_prompt(goal, context, max_steps)

        # Call LLM to generate plan
        plan_json = await self._generate_plan_json(prompt)

        # Parse and validate the plan
        plan = self._parse_plan_json(goal, plan_json)

        # Store in history
        self._plan_history.append(plan)

        logger.info(
            f"Generated plan with {len(plan.steps)} steps, "
            f"~{plan.total_estimated_tool_calls()} tool calls"
        )

        return plan

    def _build_planning_prompt(
        self,
        goal: str,
        context: Optional[str],
        max_steps: int,
    ) -> str:
        """Build the prompt for plan generation."""
        prompt_parts = [
            f"Goal: {goal}",
            "",
        ]

        if context:
            prompt_parts.extend(
                [
                    "Context:",
                    context,
                    "",
                ]
            )

        prompt_parts.extend(
            [
                f"Generate a plan with at most {max_steps} steps.",
                "Output only the JSON array, no additional text.",
            ]
        )

        return "\n".join(prompt_parts)

    async def _generate_plan_json(self, prompt: str) -> str:
        """Call LLM to generate plan JSON."""
        # Store original system prompt
        original_prompt = getattr(self.orchestrator, "_system_prompt_override", None)

        try:
            # Use planning system prompt
            self.orchestrator.set_system_prompt(PLANNING_SYSTEM_PROMPT)

            # Call the orchestrator
            response = await self.orchestrator.chat(prompt)
            return response.content if hasattr(response, "content") else str(response)

        finally:
            # Restore original prompt
            if original_prompt:
                self.orchestrator.set_system_prompt(original_prompt)

    def _parse_plan_json(self, goal: str, json_str: str) -> ExecutionPlan:
        """Parse plan JSON into ExecutionPlan."""
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        try:
            # Extract JSON from response (may have markdown code blocks)
            json_str = json_str.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            steps_data = json.loads(json_str)

            if not isinstance(steps_data, list):
                raise ValueError("Plan must be a JSON array")

            steps = []
            for i, step_data in enumerate(steps_data):
                step_type_str = step_data.get("step_type", "implementation")
                try:
                    step_type = StepType(step_type_str)
                except ValueError:
                    step_type = StepType.IMPLEMENTATION

                step = PlanStep(
                    id=step_data.get("id", str(i + 1)),
                    description=step_data.get("description", f"Step {i + 1}"),
                    step_type=step_type,
                    depends_on=step_data.get("depends_on", []),
                    estimated_tool_calls=step_data.get("estimated_tool_calls", 10),
                    requires_approval=step_data.get("requires_approval", False),
                    sub_agent_role=step_data.get("sub_agent_role"),
                )
                steps.append(step)

            return ExecutionPlan(
                id=plan_id,
                goal=goal,
                steps=steps,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse plan JSON: {e}. Creating simple plan.")
            # Fallback: create a simple single-step plan
            return ExecutionPlan(
                id=plan_id,
                goal=goal,
                steps=[
                    PlanStep(
                        id="1",
                        description=goal,
                        step_type=StepType.IMPLEMENTATION,
                        estimated_tool_calls=20,
                    )
                ],
            )

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        auto_approve: bool = False,
        parallel: bool = False,
        max_concurrent: int = 3,
        progress_callback: Optional[Callable[[PlanStep, StepStatus], None]] = None,
    ) -> PlanResult:
        """Execute a plan step by step.

        Executes each step in the plan, respecting dependencies.
        Steps can optionally run in parallel using sub-agents.

        Args:
            plan: The execution plan to run
            auto_approve: Auto-approve low-risk steps (default: False)
            parallel: Use sub-agents for parallel execution (default: False)
            max_concurrent: Max concurrent sub-agents (default: 3)
            progress_callback: Callback for progress updates

        Returns:
            PlanResult with execution summary
        """
        logger.info(f"Executing plan {plan.id} ({len(plan.steps)} steps)")

        self.active_plan = plan
        start_time = time.time()

        result = PlanResult(
            plan_id=plan.id,
            success=True,
        )

        try:
            if parallel and self.sub_agent_orchestrator:
                await self._execute_parallel(plan, result, max_concurrent, progress_callback)
            else:
                await self._execute_sequential(plan, result, auto_approve, progress_callback)

        except Exception as e:
            logger.error(f"Plan execution failed: {e}", exc_info=True)
            result.success = False
            result.final_output = f"Plan execution failed: {str(e)}"

        finally:
            result.total_duration = time.time() - start_time
            result.steps_completed = len(plan.get_completed_steps())
            result.steps_failed = len(plan.get_failed_steps())
            result.success = result.steps_failed == 0 and result.steps_completed == len(plan.steps)

            self.active_plan = None

        logger.info(
            f"Plan {plan.id} finished: {result.steps_completed}/{len(plan.steps)} steps, "
            f"success={result.success}, duration={result.total_duration:.1f}s"
        )

        return result

    async def _execute_sequential(
        self,
        plan: ExecutionPlan,
        result: PlanResult,
        auto_approve: bool,
        progress_callback: Optional[Callable[[PlanStep, StepStatus], None]],
    ) -> None:
        """Execute plan steps sequentially."""
        while not plan.is_complete() and not plan.is_failed():
            ready_steps = plan.get_ready_steps()

            if not ready_steps:
                # Check if blocked
                pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                if pending:
                    logger.warning("No ready steps but plan not complete - deadlock?")
                    for step in pending:
                        step.status = StepStatus.BLOCKED
                break

            # Execute the first ready step
            step = ready_steps[0]

            # Check approval
            if step.requires_approval and not auto_approve:
                if not self.approval_callback(f"Execute step: {step.description}?"):
                    step.status = StepStatus.BLOCKED
                    if progress_callback:
                        progress_callback(step, StepStatus.BLOCKED)
                    continue

            # Execute step
            step.status = StepStatus.IN_PROGRESS
            if progress_callback:
                progress_callback(step, StepStatus.IN_PROGRESS)

            step_result = await self._execute_step(step)
            step.result = step_result
            step.status = StepStatus.COMPLETED if step_result.success else StepStatus.FAILED

            result.step_results[step.id] = step_result
            result.total_tool_calls += step_result.tool_calls_used

            if progress_callback:
                progress_callback(step, step.status)

            # Mark dependent steps as skipped if this step failed
            if not step_result.success:
                self._skip_dependents(plan, step.id)

    async def _execute_parallel(
        self,
        plan: ExecutionPlan,
        result: PlanResult,
        max_concurrent: int,
        progress_callback: Optional[Callable[[PlanStep, StepStatus], None]],
    ) -> None:
        """Execute plan steps in parallel using sub-agents."""
        from victor.agent.subagents import SubAgentTask, SubAgentRole

        while not plan.is_complete() and not plan.is_failed():
            ready_steps = plan.get_ready_steps()

            if not ready_steps:
                break

            # Group steps by whether they can use sub-agents
            subagent_steps = [s for s in ready_steps if s.sub_agent_role]
            main_steps = [s for s in ready_steps if not s.sub_agent_role]

            # Execute sub-agent steps in parallel
            if subagent_steps and self.sub_agent_orchestrator:
                tasks = []
                for step in subagent_steps[:max_concurrent]:
                    step.status = StepStatus.IN_PROGRESS
                    if progress_callback:
                        progress_callback(step, StepStatus.IN_PROGRESS)

                    role = self._map_role_string(step.sub_agent_role)
                    tasks.append(
                        SubAgentTask(
                            role=role,
                            task=step.description,
                            tool_budget=step.estimated_tool_calls,
                        )
                    )

                fan_out_result = await self.sub_agent_orchestrator.fan_out(tasks, max_concurrent)

                # Process results
                for step, subagent_result in zip(
                    subagent_steps[:max_concurrent], fan_out_result.results
                ):
                    step_result = StepResult(
                        success=subagent_result.success,
                        output=subagent_result.summary,
                        error=subagent_result.error,
                        tool_calls_used=subagent_result.tool_calls_used,
                        duration_seconds=subagent_result.duration_seconds,
                    )
                    step.result = step_result
                    step.status = StepStatus.COMPLETED if step_result.success else StepStatus.FAILED
                    result.step_results[step.id] = step_result
                    result.total_tool_calls += step_result.tool_calls_used

                    if progress_callback:
                        progress_callback(step, step.status)

                    if not step_result.success:
                        self._skip_dependents(plan, step.id)

            # Execute main steps sequentially
            for step in main_steps:
                step.status = StepStatus.IN_PROGRESS
                if progress_callback:
                    progress_callback(step, StepStatus.IN_PROGRESS)

                step_result = await self._execute_step(step)
                step.result = step_result
                step.status = StepStatus.COMPLETED if step_result.success else StepStatus.FAILED
                result.step_results[step.id] = step_result
                result.total_tool_calls += step_result.tool_calls_used

                if progress_callback:
                    progress_callback(step, step.status)

                if not step_result.success:
                    self._skip_dependents(plan, step.id)
                    break

    def _map_role_string(self, role_str: Optional[str]) -> "SubAgentRole":
        """Map role string to SubAgentRole enum."""
        from victor.agent.subagents import SubAgentRole

        mapping = {
            "researcher": SubAgentRole.RESEARCHER,
            "planner": SubAgentRole.PLANNER,
            "executor": SubAgentRole.EXECUTOR,
            "reviewer": SubAgentRole.REVIEWER,
            "tester": SubAgentRole.TESTER,
        }
        return mapping.get(role_str or "", SubAgentRole.EXECUTOR)

    async def _execute_step(self, step: PlanStep) -> StepResult:
        """Execute a single step using the orchestrator."""
        start_time = time.time()
        tool_calls_before = getattr(self.orchestrator, "tool_calls_used", 0)

        try:
            # Build step prompt
            prompt = self._build_step_prompt(step)

            # Execute via orchestrator
            response = await self.orchestrator.chat(prompt)
            output = response.content if hasattr(response, "content") else str(response)

            tool_calls = getattr(self.orchestrator, "tool_calls_used", 0) - tool_calls_before

            return StepResult(
                success=True,
                output=output[:1000],  # Truncate long outputs
                tool_calls_used=tool_calls,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Step {step.id} failed: {e}", exc_info=True)
            return StepResult(
                success=False,
                output="",
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _build_step_prompt(self, step: PlanStep) -> str:
        """Build the prompt for executing a step."""
        parts = [
            f"Execute this task: {step.description}",
            "",
        ]

        if step.context:
            parts.extend(
                [
                    "Context from previous steps:",
                    json.dumps(step.context, indent=2),
                    "",
                ]
            )

        type_instructions = {
            StepType.RESEARCH: "Focus on reading and understanding. Do not make changes.",
            StepType.PLANNING: "Create a detailed plan or specification.",
            StepType.IMPLEMENTATION: "Implement the required code changes.",
            StepType.TESTING: "Write and run tests to verify correctness.",
            StepType.REVIEW: "Review the changes for quality and correctness.",
            StepType.DEPLOYMENT: "Deploy or apply the changes.",
        }

        if step.step_type in type_instructions:
            parts.append(f"Note: {type_instructions[step.step_type]}")

        return "\n".join(parts)

    def _skip_dependents(self, plan: ExecutionPlan, failed_step_id: str) -> None:
        """Skip all steps that depend on a failed step."""
        for step in plan.steps:
            if failed_step_id in step.depends_on and step.status == StepStatus.PENDING:
                step.status = StepStatus.SKIPPED
                logger.info(f"Skipping step {step.id} due to failed dependency {failed_step_id}")

    def get_active_plan(self) -> Optional[ExecutionPlan]:
        """Get the currently active plan, if any."""
        return self.active_plan

    def get_plan_history(self) -> List[ExecutionPlan]:
        """Get all plans created in this session."""
        return self._plan_history.copy()


__all__ = [
    "AutonomousPlanner",
    "PLANNING_SYSTEM_PROMPT",
]
