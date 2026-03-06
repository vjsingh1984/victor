"""Planning workflow extracted from ChatCoordinator.

Manages plan creation, execution, and the decision of when to
enter planning mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    """A structured plan for task execution."""

    steps: list[PlanStep] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    estimated_budget: int = 0


@dataclass
class PlanStep:
    """A single step in a plan."""

    description: str = ""
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    result: Optional[Any] = None


@dataclass
class StepResult:
    """Result from executing a plan step."""

    success: bool = True
    output: Any = None
    error: Optional[str] = None


class PlanningWorkflow:
    """Manages planning workflow: creation, execution, and entry decisions.

    Extracted from ChatCoordinator to isolate planning logic.
    """

    # Patterns that suggest planning would be beneficial
    PLANNING_TRIGGERS = [
        "implement",
        "refactor",
        "redesign",
        "migrate",
        "build",
        "create a",
        "add feature",
        "multi-step",
        "complex",
    ]

    def __init__(self, planning_threshold: int = 3):
        self._planning_threshold = planning_threshold

    async def create_plan(self, task: str, context: dict[str, Any] = None) -> Plan:
        """Create a plan for the given task.

        Args:
            task: The task description.
            context: Additional context (file paths, constraints, etc.)

        Returns:
            A Plan with ordered steps.
        """
        plan = Plan(context=context or {})
        # Planning is delegated to the LLM via the chat coordinator;
        # this class structures the result
        return plan

    async def execute_plan_step(self, step: PlanStep) -> StepResult:
        """Execute a single plan step.

        Args:
            step: The step to execute.

        Returns:
            StepResult with output or error.
        """
        try:
            step.completed = True
            return StepResult(success=True, output=step.result)
        except Exception as e:
            return StepResult(success=False, error=str(e))

    def should_enter_planning(self, message: str, state: Any = None) -> bool:
        """Determine if the message warrants planning mode.

        Args:
            message: The user's message.
            state: Current conversation state.

        Returns:
            True if planning mode should be activated.
        """
        message_lower = message.lower()
        trigger_count = sum(1 for t in self.PLANNING_TRIGGERS if t in message_lower)
        return trigger_count >= self._planning_threshold or len(message) > 500
