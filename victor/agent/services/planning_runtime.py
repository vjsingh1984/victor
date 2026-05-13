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

This module provides `PlanningRuntimeService` as the active implementation for
structured planning during chat execution. `PlanningCoordinator` remains as a
compatibility alias for older imports and attributes.

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
    service = PlanningRuntimeService(orchestrator)

    # Will auto-detect and plan if complex
    response = await service.chat_with_planning(
        "Analyze the Victor codebase architecture and provide SOLID evaluation"
    )
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
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
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter
    from victor.ui.rendering.protocol import StreamRenderer

logger = logging.getLogger(__name__)

__all__ = [
    "PlanningRuntimeService",
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
    max_parallel_steps: int = 3  # Max independent plan steps to execute concurrently

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


@dataclass(frozen=True)
class PlanApprovalDecision:
    """Decision after displaying a plan to the user."""

    proceed: bool
    user_approved_execution: bool = False
    reason: str = ""


class PlanningRuntimeService:
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
            f"PlanningRuntimeService initialized with "
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
            await self._compact_context_if_needed()
            plan = await self._generate_plan(user_message, task_analysis)
            self.active_plan = plan
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            if self.config.fallback_on_planning_failure:
                logger.info("Falling back to direct chat")
                return await self._direct_chat(user_message)
            raise

        # Step 3: Show plan and potentially wait for approval
        user_approved = True
        if self.config.show_plan_before_execution:
            approval = await self._show_plan_to_user(plan)
            if not approval.proceed:
                # User rejected the plan
                logger.info("Plan rejected by user")
                # Return a response explaining the plan was rejected
                return await self._generate_plan_rejected_response(plan)
            user_approved = approval.user_approved_execution

        # Step 4: Execute plan (only if approved)
        result = await self._execute_plan(plan, user_approved=user_approved)

        # Step 5: Generate final response
        # Compact context before generating final summary to avoid overflow
        await self._compact_context_if_needed()
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

        from victor.agent.planning.intent import is_explicit_planning_request

        if is_explicit_planning_request(user_message):
            logger.info("Planning triggered by explicit planning/checklist request")
            return True

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

        # Build skill-aware prompt from the shared framework skill matcher/capability.
        enriched_request = user_message
        try:
            matcher = getattr(self.orchestrator, "_skill_matcher", None) or getattr(
                self.orchestrator, "skill_matcher", None
            )
            if matcher and getattr(matcher, "initialized", getattr(matcher, "_initialized", False)):
                from victor.framework.skill_planner import (
                    build_skill_aware_plan_prompt,
                    build_skill_decomposition,
                    coerce_skill_catalog,
                )

                skills = coerce_skill_catalog(getattr(matcher, "skills", None))
                if skills:
                    decomposition = build_skill_decomposition(user_message, matcher)
                    enriched_request = build_skill_aware_plan_prompt(
                        user_message,
                        skills,
                        selected_skills=decomposition.skills if decomposition else None,
                        decomposition_confidence=(
                            decomposition.confidence if decomposition else None
                        ),
                    )
                    logger.debug(
                        "Plan generation enriched with %d skills%s",
                        len(skills),
                        (
                            f" and decomposition {' -> '.join(decomposition.skills)}"
                            if decomposition and decomposition.skills
                            else ""
                        ),
                    )
        except Exception:
            logger.debug("Skill-aware planning enrichment skipped", exc_info=True)

        # Extract repository and conversation context so the plan is grounded
        # in actual project structure instead of language-specific assumptions.
        prior_context = self._build_plan_generation_context()

        # Generate plan using readable schema
        plan = await generate_task_plan(
            provider=planning_provider,
            user_request=enriched_request,
            complexity=complexity,
            model=planning_model,
            conversation_context=prior_context or None,
        )

        logger.info(
            f"Generated plan: {plan.name} with {len(plan.steps)} steps, "
            f"complexity={plan.complexity.value}"
        )

        return plan

    def _build_plan_generation_context(self) -> str:
        """Build compact context for plan generation."""
        parts = []
        repository_context = self._extract_repository_profile_context()
        if repository_context:
            parts.append(repository_context)
        prior_context = self._extract_prior_context()
        if prior_context:
            parts.append(
                "Prior assistant context:\n"
                f"{prior_context}"
            )
        return "\n\n".join(parts)

    def _extract_repository_profile_context(self) -> str:
        """Return language-aware repository inventory guidance for the planner."""
        try:
            from pathlib import Path

            from victor.agent.planning.repository_profile import detect_repository_profile
            from victor.config.settings import get_project_paths

            root = Path(get_project_paths().project_root)
            profile = detect_repository_profile(root)
            return profile.to_planning_context()
        except Exception as exc:
            logger.debug("_extract_repository_profile_context failed: %s", exc)
            return ""

    def _extract_prior_context(self) -> str:
        """Return the most recent substantive assistant response for plan seeding.

        Walks the conversation history in reverse and returns the first assistant
        message with meaningful content (>200 chars), capped at 3000 chars so the
        plan-generation prompt stays within token budget.  Returns "" when no
        suitable message is found.
        """
        orch = self.orchestrator
        if not (hasattr(orch, "conversation") and orch.conversation):
            return ""
        try:
            messages = orch.conversation.messages
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.content and len(msg.content) > 200:
                    return msg.content[:3000]
        except Exception as exc:
            logger.debug("_extract_prior_context failed: %s", exc)
        return ""

    async def _show_plan_to_user(self, plan: ReadableTaskPlan) -> PlanApprovalDecision:
        """Display the plan to user and request approval when execution is effectful.

        Args:
            plan: Plan to display

        Returns:
            Plan approval decision.
        """
        # CRITICAL FIX: Use injected renderer for consistent display
        if self.renderer:
            return await self._show_plan_with_renderer(plan)
        else:
            return await self._show_plan_with_console(plan)

    async def _show_plan_with_renderer(self, plan: ReadableTaskPlan) -> PlanApprovalDecision:
        """Display plan using injected renderer (consistent UI).

        Args:
            plan: Plan to display

        Returns:
            Plan approval decision.
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

            if not self._plan_requires_execution_approval(plan):
                if console:
                    console.print(
                        "[dim green]Read-only exploration plan: continuing without approval.[/]"
                    )
                logger.info("Plan auto-continued without approval: read-only exploration")
                return PlanApprovalDecision(
                    proceed=True,
                    user_approved_execution=False,
                    reason="read_only_exploration",
                )

            # Request approval if not auto-approving
            if not self.config.auto_approve:
                approved = await self._request_plan_approval(plan, console)
                return PlanApprovalDecision(
                    proceed=approved,
                    user_approved_execution=approved,
                    reason="user_prompt",
                )
            else:
                if console:
                    console.print("[dim yellow]Auto-approving plan (auto_approve=True)[/]")
                logger.info("Auto-approving plan (auto_approve=True)")
                return PlanApprovalDecision(
                    proceed=True,
                    user_approved_execution=True,
                    reason="config_auto_approve",
                )

        finally:
            # Always resume renderer
            self.renderer.resume()

    async def _show_plan_with_console(self, plan: ReadableTaskPlan) -> PlanApprovalDecision:
        """Fallback: Display plan using separate Rich console.

        Args:
            plan: Plan to display

        Returns:
            Plan approval decision.
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

        if not self._plan_requires_execution_approval(plan):
            console.print("[dim green]Read-only exploration plan: continuing without approval.[/]")
            logger.info("Plan auto-continued without approval: read-only exploration")
            return PlanApprovalDecision(
                proceed=True,
                user_approved_execution=False,
                reason="read_only_exploration",
            )

        # Request approval if not auto-approving
        if not self.config.auto_approve:
            approved = await self._request_plan_approval(plan, console)
            return PlanApprovalDecision(
                proceed=approved,
                user_approved_execution=approved,
                reason="user_prompt",
            )
        else:
            console.print("[dim yellow]Auto-approving plan (auto_approve=True)[/]")
            logger.info("Auto-approving plan (auto_approve=True)")
            return PlanApprovalDecision(
                proceed=True,
                user_approved_execution=True,
                reason="config_auto_approve",
            )

    def _plan_requires_execution_approval(self, plan: ReadableTaskPlan) -> bool:
        """Return True when a displayed plan includes effectful execution."""
        if bool(getattr(plan, "approval", False)):
            return True

        for step in plan.steps:
            step_type = str(step[1]).strip().lower() if len(step) > 1 else ""
            step_desc = str(step[2]).strip().lower() if len(step) > 2 else ""
            step_tools = self._extract_plan_step_tools(step)
            if self._step_requires_execution_approval(step_type, step_desc, step_tools):
                return True
        return False

    @staticmethod
    def _extract_plan_step_tools(step: List[Any]) -> set[str]:
        """Extract normalized tool names from readable step data."""
        if len(step) <= 3:
            return set()
        raw_tools = step[3]
        if isinstance(raw_tools, str):
            return {
                tool.strip().lower()
                for tool in raw_tools.replace(";", ",").split(",")
                if tool.strip()
            }
        if isinstance(raw_tools, list):
            return {str(tool).strip().lower() for tool in raw_tools if str(tool).strip()}
        return set()

    @classmethod
    def _step_requires_execution_approval(
        cls,
        step_type: str,
        step_desc: str,
        tools: set[str],
    ) -> bool:
        """Classify whether a plan step can mutate state or run commands."""
        effectful_types = {
            "bug",
            "bugfix",
            "deploy",
            "deployment",
            "feature",
            "implementation",
            "implement",
            "refactor",
            "test",
            "testing",
        }
        if step_type in effectful_types:
            return True

        effectful_tools = {
            "apply_patch",
            "bash",
            "command",
            "edit",
            "execute",
            "patch",
            "run",
            "shell",
            "test",
            "write",
        }
        if tools & effectful_tools:
            return True

        effectful_desc_markers = (
            "apply patch",
            "create file",
            "edit file",
            "modify file",
            "run command",
            "run tests",
            "shell command",
            "write file",
        )
        return any(marker in step_desc for marker in effectful_desc_markers)

    async def _request_plan_approval(self, plan: ReadableTaskPlan, console) -> bool:
        """Request user approval for plan execution.

        Runs the blocking Rich prompt in a thread so the asyncio event loop
        is not stalled while waiting for stdin — avoids conflicts with the
        victor CLI's own input handler.

        Default answer is YES so that pressing Enter (the natural "looks good,
        proceed" gesture) executes the plan rather than silently rejecting it.

        Falls back to auto-approve when stdin is non-interactive (e.g. pipes,
        EOF on Ctrl-D) so the agent never hangs in headless mode.

        Args:
            plan: Plan to approve
            console: Rich console for prompt

        Returns:
            True if approved, False otherwise
        """
        from rich.prompt import Confirm

        console.print()

        # Non-interactive stdin (pipe / redirect / headless): auto-approve and proceed.
        if not sys.stdin.isatty():
            console.print("[dim yellow]Non-interactive stdin: auto-approving plan[/]")
            logger.info("Plan auto-approved (non-interactive stdin)")
            return True

        try:
            # Off-load blocking stdin read to a thread pool so the event loop
            # can continue processing (e.g. CLI keep-alive tasks, signal handlers).
            console.print(
                "[yellow]This plan includes file changes, shell commands, tests, "
                "deployment, or another effectful step.[/]"
            )
            console.print(
                "[dim]Press Enter to execute, or type y then Enter. "
                "Type n then Enter to reject.[/]"
            )
            approved = await asyncio.to_thread(
                Confirm.ask,
                "[bold yellow]Execute this plan? [Y/n][/]",
                default=True,  # Enter = yes; matches user intent when asking to "plan then implement"
                console=console,
            )
        except EOFError:
            # Ctrl-D or stdin closed: treat as implicit approval to avoid silent failure.
            console.print("[dim yellow]stdin closed: auto-approving plan[/]")
            logger.info("Plan auto-approved (stdin EOF)")
            approved = True
        except Exception as exc:
            logger.warning("Plan approval prompt failed (%s): auto-approving", exc)
            approved = True

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

    async def _execute_plan(
        self, plan: ReadableTaskPlan, user_approved: bool = False
    ) -> "PlanResult":
        """Execute the plan step by step.

        Args:
            plan: Plan to execute
            user_approved: True if user explicitly approved this plan (should auto-approve steps)

        Returns:
            PlanResult with execution summary
        """
        # Import here to avoid circular dependency
        from victor.agent.planning.autonomous import AutonomousPlanner
        from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

        team_adapter = PlanningTeamExecutionAdapter(self.orchestrator)
        if team_adapter.should_use_team(plan):
            return await self._execute_plan_via_team_adapter(plan, team_adapter)

        planner = AutonomousPlanner(self.orchestrator)

        # Convert to execution plan
        execution_plan = plan.to_execution_plan()

        # If user explicitly approved the plan, auto-approve all steps
        # Otherwise, use config setting (which defaults to False for safety)
        auto_approve = user_approved or self.config.auto_approve

        # Execute with auto-approval based on user's plan approval
        result = await planner.execute_plan(
            execution_plan,
            auto_approve=auto_approve,
        )

        logger.info(
            f"Plan execution: success={result.success}, "
            f"steps_completed={result.steps_completed}/{result.total_steps}"
        )

        return result

    async def _execute_plan_via_team_adapter(
        self,
        plan: ReadableTaskPlan,
        team_adapter: "PlanningTeamExecutionAdapter",
    ) -> "PlanResult":
        """Execute complex planned work through reusable team formations."""
        from victor.agent.planning.base import PlanResult, StepStatus

        execution_plan = plan.to_execution_plan()
        max_concurrent = self._effective_team_plan_concurrency()
        root_session_id = (
            getattr(self.orchestrator, "active_session_id", None)
            or getattr(self.orchestrator, "session_id", None)
            or getattr(self.orchestrator, "_memory_session_id", None)
        )
        result = PlanResult(
            plan_id=execution_plan.id,
            success=True,
            total_steps=len(execution_plan.steps),
        )

        while not execution_plan.is_complete() and not execution_plan.is_failed():
            ready_steps = execution_plan.get_ready_steps()
            if not ready_steps:
                pending_steps = [
                    step
                    for step in execution_plan.steps
                    if step.status == StepStatus.PENDING
                ]
                for step in pending_steps:
                    step.status = StepStatus.BLOCKED
                break

            batch = ready_steps[:max_concurrent]
            for step in batch:
                step.status = StepStatus.IN_PROGRESS

            step_results = await asyncio.gather(
                *[
                    team_adapter.execute_step(
                        plan=plan,
                        execution_plan=execution_plan,
                        step=step,
                        root_session_id=root_session_id,
                    )
                    for step in batch
                ],
                return_exceptions=True,
            )

            failed_step_ids: list[str] = []
            for step, step_result in zip(batch, step_results):
                if isinstance(step_result, Exception):
                    from victor.agent.planning.base import StepResult

                    step_result = StepResult(
                        success=False,
                        output="",
                        error=f"{type(step_result).__name__}: {step_result}",
                    )

                step.result = step_result
                step.status = StepStatus.COMPLETED if step_result.success else StepStatus.FAILED
                result.step_results[step.id] = step_result
                result.total_tool_calls += step_result.tool_calls_used
                if not step_result.success:
                    failed_step_ids.append(step.id)

            if failed_step_ids:
                self._skip_team_plan_dependents(execution_plan, failed_step_ids)
                break

        result.steps_completed = sum(
            1 for step in execution_plan.steps if step.status == StepStatus.COMPLETED
        )
        result.steps_failed = sum(
            1 for step in execution_plan.steps if step.status == StepStatus.FAILED
        )
        result.success = result.steps_failed == 0 and result.steps_completed == len(
            execution_plan.steps
        )
        result.final_output = "\n\n".join(
            step_result.output for step_result in result.step_results.values() if step_result.output
        )

        logger.info(
            "Team plan execution: success=%s, steps_completed=%s/%s",
            result.success,
            result.steps_completed,
            result.total_steps,
        )
        return result

    def _effective_team_plan_concurrency(self) -> int:
        """Return the bounded concurrency for independent team-plan steps."""
        env_value = os.getenv("VICTOR_PLAN_MAX_CONCURRENT_AGENTS") or os.getenv(
            "VICTOR_PLAN_MAX_PARALLEL_STEPS"
        )
        if env_value:
            try:
                return max(1, int(env_value))
            except ValueError:
                logger.warning(
                    "Ignoring invalid plan concurrency override: %s",
                    env_value,
                )
        return max(1, int(self.config.max_parallel_steps or 1))

    @staticmethod
    def _skip_team_plan_dependents(
        execution_plan: "ExecutionPlan",
        failed_step_ids: list[str],
    ) -> None:
        """Mark pending transitive dependents skipped after a failed team-plan step."""
        from victor.agent.planning.base import StepStatus

        failed = set(failed_step_ids)
        changed = True
        while changed:
            changed = False
            for step in execution_plan.steps:
                if step.status != StepStatus.PENDING:
                    continue
                if any(dep in failed for dep in step.depends_on):
                    step.status = StepStatus.SKIPPED
                    failed.add(step.id)
                    changed = True

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
            "Summarize the planned task execution using only the evidence below.",
            f"Task: {plan.name}",
            f"Complexity: {plan.complexity.value}",
            f"Steps completed: {result.steps_completed}/{result.total_steps}",
            f"Overall success: {result.success}",
            "",
            "Steps executed:",
        ]

        for i, step in enumerate(plan.steps):
            step_id = step[0]
            step_type = step[1]
            step_desc = step[2]
            step_result = result.step_results.get(str(step_id)) or result.step_results.get(step_id)
            if step_result is not None:
                status = "completed" if step_result.success else "failed"
                parts.append(f"  - {step_id}. [{step_type}] {status}: {step_desc}")
                if step_result.output:
                    parts.append(f"    Evidence: {step_result.output[:2000]}")
                if step_result.error:
                    parts.append(f"    Error: {step_result.error}")
            else:
                status = "not run" if i >= result.steps_completed else "unknown"
                parts.append(f"  - {step_id}. [{step_type}] {status}: {step_desc}")

        error_message = getattr(result, "error_message", "")
        if error_message:
            parts.extend(
                [
                    "",
                    f"Error: {error_message}",
                ]
            )
        if result.final_output:
            parts.extend(
                [
                    "",
                    "Aggregated step output:",
                    result.final_output[:6000],
                ]
            )

        parts.extend(
            [
                "",
                "Produce a concise user-facing summary. If a step failed or lacks evidence, "
                "state that plainly and do not invent repository findings.",
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

    async def _compact_context_if_needed(self) -> None:
        """Compact conversation context if it's getting too large.

        This helps prevent context overflow during long planning sessions.
        """
        orch = self.orchestrator
        if await self._compact_with_context_service():
            return

        has_capability = getattr(orch, "has_capability", None)
        get_capability_value = getattr(orch, "get_capability_value", None)
        if (
            callable(has_capability)
            and callable(get_capability_value)
            and has_capability("context_compactor")
            and get_capability_value("context_compactor")
        ):
            try:
                # Get current query from public conversation API
                current_query = ""
                if hasattr(orch, "conversation") and orch.conversation:
                    current_query = orch.conversation.get_latest_user_message() or ""

                # Check and compact
                compactor = get_capability_value("context_compactor")
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

    async def _compact_with_context_service(self) -> bool:
        """Use the canonical context service before legacy compactor fallback."""
        context_service = getattr(self.orchestrator, "_context_service", None)
        if context_service is None:
            return False

        recommendation_getter = getattr(context_service, "get_compaction_recommendation", None)
        compact_context = getattr(context_service, "compact_context", None)
        if not callable(recommendation_getter) or not callable(compact_context):
            return False

        try:
            recommendation = recommendation_getter()
            if inspect.isawaitable(recommendation):
                recommendation = await recommendation
            if not isinstance(recommendation, dict):
                return True
            if not recommendation.get("should_compact", False):
                return True

            removed = compact_context(
                strategy=self._context_compaction_strategy(),
                min_messages=6,
            )
            if inspect.isawaitable(removed):
                removed = await removed
            removed_count = int(removed or 0)
            if removed_count > 0:
                logger.info(
                    "ContextService compacted planning context: %s messages removed",
                    removed_count,
                )
            return True
        except Exception as e:
            logger.warning(f"Context service compaction failed: {e}")
            return True

    def _context_compaction_strategy(self) -> str:
        settings = getattr(self.orchestrator, "settings", None)
        return str(getattr(settings, "context_compaction_strategy", "tiered") or "tiered")


# Compatibility alias. New service-owned call sites should import
# PlanningRuntimeService; legacy coordinator paths continue to resolve.
PlanningCoordinator = PlanningRuntimeService
