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

"""Planning coordinator for integrating chat with autonomous planning.

This module provides the PlanningCoordinator that bridges the gap between
ChatCoordinator's chat flow and AutonomousPlanner's structured planning system.

Key Features:
- Auto-detects complex multi-step tasks that benefit from planning
- Generates structured plans using ReadableTaskPlan schema
- Executes plans step-by-step with context-aware tools
- Provides progress reporting and plan state management
- Falls back to regular chat for simple tasks

Design:
--------
The coordinator uses a complexity threshold to decide when to use planning:
- SIMPLE tasks: Direct chat (fast, minimal overhead)
- MODERATE/COMPLEX tasks: Plan-based execution (structured, reliable)

For plan-based tasks:
1. Generate plan using generate_task_plan() with token-efficient schema
2. Show plan to user
3. Execute steps sequentially with StepAwareToolSelector
4. Report progress and handle failures
5. Complete with summary

Example:
    coordinator = PlanningCoordinator(orchestrator)

    # Will auto-detect and plan if complex
    response = await coordinator.chat_with_planning(
        "Analyze the Victor codebase architecture and provide SOLID evaluation"
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from victor.agent.planning.constants import (
    COMPLEXITY_KEYWORDS,
    DEFAULT_MIN_PLANNING_COMPLEXITY,
    DEFAULT_MIN_STEPS_THRESHOLD,
    DEFAULT_MIN_KEYWORD_MATCHES,
    STEP_INDICATORS,
)
from victor.agent.planning.readable_schema import (
    ReadableTaskPlan,
    TaskComplexity as PlanningTaskComplexity,
    generate_task_plan,
)
from victor.agent.task_analyzer import TaskAnalysis
from victor.framework.task import TaskComplexity
from victor.providers.base import CompletionResponse

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.planning.base import ExecutionPlan, PlanResult

logger = logging.getLogger(__name__)


class PlanningMode(Enum):
    """Planning mode for chat requests."""

    AUTO = "auto"  # Automatically decide based on task complexity
    ALWAYS = "always"  # Always use planning
    NEVER = "never"  # Never use planning (direct chat only)


@dataclass
class PlanningConfig:
    """Configuration for planning behavior."""

    # Minimum complexity level to trigger planning
    # Use framework TaskComplexity (simple/medium/complex)
    min_planning_complexity: TaskComplexity = TaskComplexity.MEDIUM

    # Thresholds for detecting multi-step tasks
    min_steps_threshold: int = DEFAULT_MIN_STEPS_THRESHOLD
    min_keyword_matches: int = DEFAULT_MIN_KEYWORD_MATCHES
    complexity_keywords: List[str] = field(default_factory=lambda: list(COMPLEXITY_KEYWORDS))
    step_indicators: List[str] = field(default_factory=lambda: list(STEP_INDICATORS))

    # Planning behavior
    show_plan_before_execution: bool = True  # Require user to see plan first
    allow_plan_modification: bool = False  # Allow user to modify plan (future)
    max_parallel_steps: int = 1  # Max steps to execute in parallel (future)

    # Fallback behavior
    fallback_on_planning_failure: bool = True  # Fall back to direct chat if planning fails
    max_planning_retries: int = 1  # Number of retries for plan generation


@dataclass
class PlanningResult:
    """Result from planning-based chat execution."""

    mode: str  # "planned" or "direct"
    plan: Optional[ReadableTaskPlan] = None
    execution_result: Optional["PlanResult"] = None
    response: Optional[CompletionResponse] = None
    steps_completed: int = 0
    steps_total: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        if self.mode == "direct":
            return self.response is not None
        return self.execution_result is not None and self.execution_result.success


class PlanningCoordinator:
    """Coordinator for integrating chat with autonomous planning.

    This coordinator sits between ChatCoordinator and AutonomousPlanner,
    automatically deciding when to use structured planning vs direct chat.

    Attributes:
        orchestrator: Parent orchestrator for LLM calls and tool execution
        config: Configuration for planning behavior
        active_plan: Currently active plan (if any)
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        config: Optional[PlanningConfig] = None,
    ):
        """Initialize the planning coordinator.

        Args:
            orchestrator: Parent orchestrator
            config: Optional configuration (uses defaults if not provided)
        """
        self.orchestrator = orchestrator
        self.config = config or PlanningConfig()
        self.active_plan: Optional[ReadableTaskPlan] = None
        self._planning_mode = PlanningMode.AUTO

        logger.info(
            f"PlanningCoordinator initialized with "
            f"min_complexity={self.config.min_planning_complexity.value}"
        )

    async def chat_with_planning(
        self,
        user_message: str,
        task_analysis: Optional[TaskAnalysis] = None,
        mode: PlanningMode = PlanningMode.AUTO,
    ) -> CompletionResponse:
        """Chat with automatic planning for complex tasks.

        This is the main entry point. It analyzes the task and decides:
        - Simple tasks → direct chat (fast)
        - Complex tasks → plan → execute → summarize (reliable)

        Args:
            user_message: User's message
            task_analysis: Optional pre-computed task analysis
            mode: Planning mode (auto/always/never)

        Returns:
            CompletionResponse from the model
        """
        self._planning_mode = mode

        # Step 1: Analyze if planning is needed
        should_plan = self._should_use_planning(user_message, task_analysis)

        if not should_plan or mode == PlanningMode.NEVER:
            logger.info("Using direct chat mode")
            return await self._direct_chat(user_message)

        # Step 2: Generate plan
        logger.info(f"Using planning mode for: {user_message[:100]}...")
        try:
            # Compact context before plan generation to avoid overflow
            self._compact_context_if_needed()
            plan = await self._generate_plan(user_message, task_analysis)
            self.active_plan = plan
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            if self.config.fallback_on_planning_failure:
                logger.info("Falling back to direct chat")
                return await self._direct_chat(user_message)
            raise

        # Step 3: Show plan and potentially wait for approval
        if self.config.show_plan_before_execution:
            self._show_plan_to_user(plan)

        # Step 4: Execute plan
        result = await self._execute_plan(plan)

        # Step 5: Generate final response
        # Compact context before generating final summary to avoid overflow
        self._compact_context_if_needed()
        response = await self._generate_final_response(plan, result)

        return response

    def _map_complexity(self, framework_complexity: TaskComplexity) -> PlanningTaskComplexity:
        """Map framework TaskComplexity to planning TaskComplexity.

        Args:
            framework_complexity: Framework complexity level

        Returns:
            Planning TaskComplexity
        """
        # Map framework complexity to planning complexity
        # Framework: simple/medium/complex/generation/action/analysis
        # Planning: simple/moderate/complex
        complexity_map = {
            TaskComplexity.SIMPLE: PlanningTaskComplexity.SIMPLE,
            TaskComplexity.MEDIUM: PlanningTaskComplexity.MODERATE,
            TaskComplexity.COMPLEX: PlanningTaskComplexity.COMPLEX,
            TaskComplexity.GENERATION: PlanningTaskComplexity.COMPLEX,
            TaskComplexity.ACTION: PlanningTaskComplexity.MODERATE,
            TaskComplexity.ANALYSIS: PlanningTaskComplexity.COMPLEX,
        }
        return complexity_map.get(framework_complexity, PlanningTaskComplexity.MODERATE)

    def _should_use_planning(
        self,
        user_message: str,
        task_analysis: Optional[TaskAnalysis] = None,
    ) -> bool:
        """Determine if planning should be used for this task.

        Uses multiple signals:
        1. Task complexity from TaskAnalyzer
        2. Keyword analysis for multi-step indicators
        3. User's explicit planning mode

        Args:
            user_message: User's message
            task_analysis: Optional task analysis

        Returns:
            True if planning should be used
        """
        if self._planning_mode == PlanningMode.ALWAYS:
            return True

        if self._planning_mode == PlanningMode.NEVER:
            return False

        # Check task complexity if available
        if task_analysis:
            # Map framework complexity to planning complexity for threshold comparison
            planning_complexity = self._map_complexity(task_analysis.complexity)
            min_planning_complexity = self._map_complexity(self.config.min_planning_complexity)

            if planning_complexity.value >= min_planning_complexity.value:
                logger.info(f"Planning triggered by complexity: {planning_complexity.value}")
                return True

        # Check for multi-step keywords
        message_lower = user_message.lower()
        keyword_matches = sum(1 for kw in self.config.complexity_keywords if kw in message_lower)
        if keyword_matches >= self.config.min_keyword_matches:
            logger.info(f"Planning triggered by keywords: {keyword_matches} matches")
            return True

        # Check for explicit "step" language
        step_count = sum(
            1 for indicator in self.config.step_indicators if indicator in message_lower
        )
        if step_count >= self.config.min_steps_threshold:
            logger.info(f"Planning triggered by step indicators: {step_count} matches")
            return True

        return False

    async def _generate_plan(
        self,
        user_message: str,
        task_analysis: Optional[TaskAnalysis] = None,
    ) -> ReadableTaskPlan:
        """Generate a structured plan for the task.

        Args:
            user_message: User's message
            task_analysis: Optional task analysis

        Returns:
            ReadableTaskPlan with structured steps
        """
        # Determine complexity
        if task_analysis:
            complexity = self._map_complexity(task_analysis.complexity)
        else:
            complexity = PlanningTaskComplexity.MODERATE

        # Check for planning-specific provider/model override (priority: CLI > profile > default)
        planning_provider = self.orchestrator.provider
        planning_model = self.orchestrator.model

        # Try to get CLI override first (stored in orchestrator if set)
        if hasattr(self.orchestrator, '_planning_model_override'):
            cli_planning_model = self.orchestrator._planning_model_override
            if cli_planning_model:
                logger.info(f"Using CLI planning model override: {cli_planning_model}")
                # Parse planning model (format: "model" or "provider:model")
                if ':' in cli_planning_model:
                    planning_provider_name, planning_model = cli_planning_model.split(':', 1)
                    from victor.providers.provider_factory import get_provider
                    try:
                        planning_provider = get_provider(planning_provider_name)
                        logger.info(f"Using planning provider from CLI override: {planning_provider_name}")
                    except Exception as e:
                        logger.warning(f"Failed to get planning provider {planning_provider_name}: {e}")
                else:
                    planning_model = cli_planning_model

        # Try profile override if no CLI override
        elif hasattr(self.orchestrator, '_profile') and self.orchestrator._profile:
            profile = self.orchestrator._profile
            planning_provider_override = getattr(profile, 'planning_provider', None)
            planning_model_override = getattr(profile, 'planning_model', None)

            if planning_provider_override:
                logger.info(f"Using planning provider override from profile: {planning_provider_override}")
                from victor.providers.provider_factory import get_provider
                try:
                    planning_provider = get_provider(planning_provider_override)
                except Exception as e:
                    logger.warning(f"Failed to get planning provider {planning_provider_override}: {e}")
                    logger.info("Falling back to default provider for planning")

            if planning_model_override:
                logger.info(f"Using planning model override from profile: {planning_model_override}")
                planning_model = planning_model_override

        # Generate plan using readable schema
        plan = await generate_task_plan(
            provider=planning_provider,
            user_request=user_message,
            complexity=complexity,
            model=planning_model,
        )

        logger.info(
            f"Generated plan: {plan.name} with {len(plan.steps)} steps, "
            f"complexity={plan.complexity.value}"
        )

        return plan

    def _show_plan_to_user(self, plan: ReadableTaskPlan) -> None:
        """Display the plan to the user.

        Args:
            plan: Plan to display
        """
        # TODO: Integrate with UI for interactive plan display
        # For now, log the plan structure
        logger.info(f"Plan: {plan.name}")
        logger.info(f"Complexity: {plan.complexity.value}")
        logger.info(f"Steps: {len(plan.steps)}")

        for i, step in enumerate(plan.steps):
            step_id = step[0]
            step_type = step[1]
            step_desc = step[2]
            logger.info(f"  {step_id}. [{step_type}] {step_desc}")

    async def _execute_plan(self, plan: ReadableTaskPlan) -> "PlanResult":
        """Execute the plan step by step.

        Args:
            plan: Plan to execute

        Returns:
            PlanResult with execution summary
        """
        # Import here to avoid circular dependency
        from victor.agent.planning.autonomous import AutonomousPlanner

        planner = AutonomousPlanner(self.orchestrator)

        # Convert to execution plan
        execution_plan = plan.to_execution_plan()

        # Execute with auto-approval (can be made configurable)
        result = await planner.execute_plan(
            execution_plan,
            auto_approve=True,  # TODO: Make this configurable
        )

        logger.info(
            f"Plan execution: success={result.success}, "
            f"steps_completed={result.steps_completed}/{result.total_steps}"
        )

        return result

    async def _generate_final_response(
        self,
        plan: ReadableTaskPlan,
        result: "PlanResult",
    ) -> CompletionResponse:
        """Generate final response summarizing plan execution.

        Args:
            plan: Original plan
            result: Execution result

        Returns:
            CompletionResponse with summary
        """
        # Build summary prompt
        summary_prompt = self._build_summary_prompt(plan, result)

        # Get response from provider
        response = await self.orchestrator.provider.chat(
            messages=[{"role": "user", "content": summary_prompt}],
            model=self.orchestrator.model,
            temperature=0.7,  # Lower temperature for summaries
            max_tokens=self.orchestrator.max_tokens,
        )

        return response

    def _build_summary_prompt(
        self,
        plan: ReadableTaskPlan,
        result: "PlanResult",
    ) -> str:
        """Build a prompt for summarizing plan execution.

        Args:
            plan: Original plan
            result: Execution result

        Returns:
            Summary prompt
        """
        parts = [
            "I've completed the requested analysis task. Here's what I did:\n",
            f"Task: {plan.name}",
            f"Complexity: {plan.complexity.value}",
            f"Steps completed: {result.steps_completed}/{result.total_steps}",
            "",
            "Steps executed:",
        ]

        for i, step in enumerate(plan.steps):
            step_id = step[0]
            step_type = step[1]
            step_desc = step[2]
            status = "✓" if i < result.steps_completed else "✗"
            parts.append(f"  {status} {step_id}. [{step_type}] {step_desc}")

        if result.error_message:
            parts.extend(
                [
                    "",
                    f"Error: {result.error_message}",
                ]
            )

        parts.extend(
            [
                "",
                "Please provide a comprehensive summary of the results, "
                "addressing all the steps that were completed. "
                "Format the response clearly with sections matching the steps above.",
            ]
        )

        return "\n".join(parts)

    async def _direct_chat(self, user_message: str) -> CompletionResponse:
        """Fallback to direct chat without planning.

        Args:
            user_message: User's message

        Returns:
            CompletionResponse from chat
        """
        # Delegate to orchestrator's chat
        return await self.orchestrator.chat(user_message)

    def set_planning_mode(self, mode: PlanningMode) -> None:
        """Set the planning mode.

        Args:
            mode: Planning mode to use
        """
        self._planning_mode = mode
        logger.info(f"Planning mode set to: {mode.value}")

    def get_active_plan(self) -> Optional[ReadableTaskPlan]:
        """Get the currently active plan.

        Returns:
            Active plan or None
        """
        return self.active_plan

    def clear_active_plan(self) -> None:
        """Clear the active plan."""
        self.active_plan = None
        logger.info("Active plan cleared")

    def _compact_context_if_needed(self) -> None:
        """Compact conversation context if it's getting too large.

        This helps prevent context overflow during long planning sessions.
        """
        orch = self.orchestrator
        if hasattr(orch, "_context_compactor") and orch._context_compactor:
            try:
                # Get current query (user message)
                current_query = (
                    orch._conversation_history.get_latest_user_message()
                    if hasattr(orch, "_conversation_history")
                    else ""
                )

                # Check and compact
                compaction_result = orch._context_compactor.check_and_compact(
                    current_query=current_query,
                    force=False,
                    tool_call_count=orch.tool_calls_used,
                    task_complexity="complex",  # Planning tasks are complex
                )

                if compaction_result.action_taken:
                    logger.info(
                        f"Context compacted: {compaction_result.messages_removed} messages removed, "
                        f"{compaction_result.tokens_freed} tokens freed"
                    )
            except Exception as e:
                logger.warning(f"Context compaction failed: {e}")
