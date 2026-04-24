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

"""Canonical service-owned planning runtime implementation.

This module provides `PlanningCoordinator` as the active implementation for
structured planning during chat execution. The historical
`victor.agent.coordinators.planning_coordinator` module is now only a
compatibility import path that re-exports these definitions.

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
    from victor.agent.services.protocols.chat_runtime import PlanningContextProtocol
    from victor.agent.planning.base import ExecutionPlan, PlanResult
    from victor.ui.rendering.protocol import StreamRenderer

logger = logging.getLogger(__name__)

__all__ = [
    "PlanningCoordinator",
    "PlanningConfig",
    "PlanningMode",
    "PlanningResult",
]


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
    auto_approve: bool = False  # Require user confirmation before executing plans (safer default)
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
    """Service-owned runtime helper for integrating chat with autonomous planning.

    This helper sits between chat execution and the AutonomousPlanner,
    automatically deciding when to use structured planning vs direct chat.

    Attributes:
        orchestrator: Parent orchestrator for LLM calls and tool execution
        config: Configuration for planning behavior
        active_plan: Currently active plan (if any)
    """

    def __init__(
        self,
        orchestrator: "PlanningContextProtocol",
        config: Optional[PlanningConfig] = None,
        renderer: Optional["StreamRenderer"] = None,  # NEW: Inject renderer for consistent UI
    ):
        """Initialize the planning coordinator.

        Args:
            orchestrator: Any object satisfying PlanningContextProtocol
            config: Optional configuration (uses defaults if not provided)
            renderer: Optional renderer for consistent plan display (falls back to Rich console)
        """
        self.orchestrator = orchestrator
        self.config = config or PlanningConfig()
        self.renderer = renderer  # NEW: Store renderer for consistent plan display
        self.active_plan: Optional[ReadableTaskPlan] = None
        self._planning_mode = PlanningMode.AUTO

        logger.info(
            f"PlanningCoordinator initialized with "
            f"min_complexity={self.config.min_planning_complexity.value}, "
            f"renderer={'injected' if renderer else 'console fallback'}"
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
            approved = self._show_plan_to_user(plan)
            if not approved:
                # User rejected the plan
                logger.info("Plan rejected by user")
                # Return a response explaining the plan was rejected
                return await self._generate_plan_rejected_response(plan)

        # Step 4: Execute plan (only if approved)
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

        # Try to get CLI override first (public attribute, set by CLI arg parsing)
        cli_planning_model = getattr(self.orchestrator, "planning_model_override", None)
        if cli_planning_model:
            logger.info(f"Using CLI planning model override: {cli_planning_model}")
            # Parse planning model (format: "model" or "provider:model")
            if ":" in cli_planning_model:
                planning_provider_name, planning_model = cli_planning_model.split(":", 1)
                from victor.providers.provider_factory import get_provider

                try:
                    planning_provider = get_provider(planning_provider_name)
                    logger.info(
                        f"Using planning provider from CLI override: {planning_provider_name}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get planning provider {planning_provider_name}: {e}")
            else:
                planning_model = cli_planning_model
        else:
            # Try profile override if no CLI override
            profile = getattr(self.orchestrator, "profile", None)
            if profile:
                planning_provider_override = getattr(profile, "planning_provider", None)
                planning_model_override = getattr(profile, "planning_model", None)

                if planning_provider_override:
                    logger.info(
                        f"Using planning provider override from profile: {planning_provider_override}"
                    )
                    from victor.providers.provider_factory import get_provider

                    try:
                        planning_provider = get_provider(planning_provider_override)
                    except Exception as e:
                        logger.warning(
                            f"Failed to get planning provider {planning_provider_override}: {e}"
                        )
                        logger.info("Falling back to default provider for planning")

                if planning_model_override:
                    logger.info(
                        f"Using planning model override from profile: {planning_model_override}"
                    )
                    planning_model = planning_model_override

        # Build skill-aware prompt if skills are available
        enriched_request = user_message
        try:
            matcher = getattr(self.orchestrator, "skill_matcher", None)
            if (
                matcher
                and getattr(matcher, "initialized", False)
                and getattr(matcher, "skills", None)
            ):
                from victor.framework.skill_planner import build_skill_aware_plan_prompt

                skills = getattr(matcher, "skills", [])
                enriched_request = build_skill_aware_plan_prompt(user_message, skills)
                logger.debug("Plan generation enriched with %d skills", len(skills))
        except Exception:
            logger.debug("Skill-aware planning enrichment skipped", exc_info=True)

        # Generate plan using readable schema
        plan = await generate_task_plan(
            provider=planning_provider,
            user_request=enriched_request,
            complexity=complexity,
            model=planning_model,
        )

        logger.info(
            f"Generated plan: {plan.name} with {len(plan.steps)} steps, "
            f"complexity={plan.complexity.value}"
        )

        return plan

    def _show_plan_to_user(self, plan: ReadableTaskPlan) -> bool:
        """Display the plan to user and request approval.

        Args:
            plan: Plan to display

        Returns:
            True if user approved the plan, False otherwise
        """
        # CRITICAL FIX: Use injected renderer for consistent display
        if self.renderer:
            return self._show_plan_with_renderer(plan)
        else:
            return self._show_plan_with_console(plan)

    def _show_plan_with_renderer(self, plan: ReadableTaskPlan) -> bool:
        """Display plan using injected renderer (consistent UI).

        Args:
            plan: Plan to display

        Returns:
            True if user approved the plan, False otherwise
        """
        from rich.table import Table

        # Pause renderer to show plan
        self.renderer.pause()

        try:
            # Build plan table
            table = Table(
                title=f"\U0001f4cb {plan.name} ({plan.complexity.value})",
                show_lines=False,
            )
            table.add_column("#", style="dim", width=3)
            table.add_column("Type", style="cyan", width=12)
            table.add_column("Description")
            table.add_column("Tools", style="dim")

            for step in plan.steps:
                step_id = str(step[0]) if len(step) > 0 else ""
                step_type = str(step[1]) if len(step) > 1 else ""
                step_desc = str(step[2]) if len(step) > 2 else ""
                step_tools = str(step[3]) if len(step) > 3 else ""
                table.add_row(step_id, step_type, step_desc, step_tools)

            # Get console from renderer (if available)
            console = getattr(self.renderer, "console", None)
            if console:
                console.print(table)
                if plan.duration:
                    console.print(f"[dim]Estimated: {plan.duration}[/]")

            # Save plan to disk
            self._save_plan_to_disk(plan, console)

            # Request approval if not auto-approving
            if not self.config.auto_approve:
                return self._request_plan_approval(plan, console)
            else:
                if console:
                    console.print("[dim yellow]Auto-approving plan (auto_approve=True)[/]")
                logger.info("Auto-approving plan (auto_approve=True)")
                return True

        finally:
            # Always resume renderer
            self.renderer.resume()

    def _show_plan_with_console(self, plan: ReadableTaskPlan) -> bool:
        """Fallback: Display plan using separate Rich console.

        Args:
            plan: Plan to display

        Returns:
            True if user approved the plan, False otherwise
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(
            title=f"\U0001f4cb {plan.name} ({plan.complexity.value})",
            show_lines=False,
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Type", style="cyan", width=12)
        table.add_column("Description")
        table.add_column("Tools", style="dim")

        for step in plan.steps:
            step_id = str(step[0]) if len(step) > 0 else ""
            step_type = str(step[1]) if len(step) > 1 else ""
            step_desc = str(step[2]) if len(step) > 2 else ""
            step_tools = str(step[3]) if len(step) > 3 else ""
            table.add_row(step_id, step_type, step_desc, step_tools)

        console.print(table)

        if plan.duration:
            console.print(f"[dim]Estimated: {plan.duration}[/]")

        # Save plan to disk
        self._save_plan_to_disk(plan, console)

        # Request approval if not auto-approving
        if not self.config.auto_approve:
            return self._request_plan_approval(plan, console)
        else:
            console.print("[dim yellow]Auto-approving plan (auto_approve=True)[/]")
            logger.info("Auto-approving plan (auto_approve=True)")
            return True

    def _request_plan_approval(self, plan: ReadableTaskPlan, console) -> bool:
        """Request user approval for plan execution.

        Args:
            plan: Plan to approve
            console: Rich console for prompt

        Returns:
            True if approved, False otherwise
        """
        from rich.prompt import Confirm

        console.print()
        approved = Confirm.ask("[bold yellow]Execute this plan?[/]", default=False, console=console)

        if approved:
            console.print("[green]✓ Plan approved. Executing...[/]")
            logger.info(f"Plan approved by user: {plan.name}")
        else:
            console.print("[red]✗ Plan rejected.[/]")
            logger.info(f"Plan rejected by user: {plan.name}")

        return approved

    def _save_plan_to_disk(self, plan: ReadableTaskPlan, console) -> None:
        """Save plan to .victor/plans/ directory.

        Args:
            plan: Plan to save
            console: Rich console for status messages
        """
        import os
        import json
        from datetime import datetime

        try:
            # Ensure plans directory exists
            plans_dir = os.path.expanduser("~/.victor/plans")
            os.makedirs(plans_dir, exist_ok=True)

            # Generate filename from plan name and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in plan.name
            )
            safe_name = safe_name[:50]  # Truncate long names
            filename = f"{timestamp}_{safe_name}.json"
            filepath = os.path.join(plans_dir, filename)

            # Convert plan to dict for JSON serialization
            plan_dict = {
                "name": plan.name,
                "complexity": plan.complexity.value,
                "description": plan.description if hasattr(plan, "description") else "",
                "duration": plan.duration if hasattr(plan, "duration") else "",
                "steps": [
                    {
                        "id": step[0],
                        "type": step[1],
                        "description": step[2] if len(step) > 2 else "",
                        "tools": step[3] if len(step) > 3 else [],
                    }
                    for step in plan.steps
                ],
                "generated_at": timestamp,
                "step_count": len(plan.steps),
            }

            # Save to file
            with open(filepath, "w") as f:
                json.dump(plan_dict, f, indent=2)

            logger.info(f"Plan saved to {filepath}")
            if console:
                console.print(f"[dim]Plan saved to: {filepath}[/]")

        except Exception as e:
            logger.warning(f"Failed to save plan to disk: {e}")
            if console:
                console.print(f"[dim yellow]⚠ Failed to save plan: {e}[/]")

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
            auto_approve=self.config.auto_approve,
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

    async def _generate_plan_rejected_response(
        self,
        plan: ReadableTaskPlan,
    ) -> CompletionResponse:
        """Generate response when user rejects plan.

        Args:
            plan: The rejected plan

        Returns:
            CompletionResponse explaining the rejection
        """
        rejection_prompt = f"""The user has rejected the following plan:

Plan: {plan.name}
Complexity: {plan.complexity.value}
Steps: {len(plan.steps)}

Please acknowledge that the plan was rejected and ask if the user would like to:
1. Modify the plan
2. Try a different approach
3. Answer the question without a structured plan

Keep your response concise and helpful.
"""

        response = await self.orchestrator.provider.chat(
            messages=[{"role": "user", "content": rejection_prompt}],
            model=self.orchestrator.model,
            temperature=0.7,
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
        if orch.has_capability("context_compactor") and orch.get_capability_value(
            "context_compactor"
        ):
            try:
                # Get current query from public conversation API
                current_query = ""
                if hasattr(orch, "conversation") and orch.conversation:
                    current_query = orch.conversation.get_latest_user_message() or ""

                # Check and compact
                compactor = orch.get_capability_value("context_compactor")
                compaction_result = compactor.check_and_compact(
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
