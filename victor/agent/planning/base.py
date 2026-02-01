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

"""Base data structures for autonomous planning.

This module provides the core data structures for execution plans:
- PlanStep: Individual step in an execution plan
- ExecutionPlan: Collection of steps with dependencies
- StepResult: Result of executing a single step
- PlanResult: Overall result of plan execution

Design Principles:
- Plans are declarative: describe what to achieve, not how
- Steps can have dependencies on prior steps
- Plans can be serialized for review/approval
- Execution is resumable after interruption

Example Usage:
    plan = ExecutionPlan(
        goal="Implement JWT authentication",
        steps=[
            PlanStep(id="1", description="Research existing auth patterns", ...),
            PlanStep(id="2", description="Create auth middleware", depends_on=["1"], ...),
            PlanStep(id="3", description="Add JWT validation", depends_on=["2"], ...),
        ]
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol

logger = logging.getLogger(__name__)


def _get_step_status_icon(
    status: "StepStatus",
    presentation: Optional["PresentationProtocol"] = None,
) -> str:
    """Get icon for a step status using presentation adapter.

    Args:
        status: The step status
        presentation: Optional presentation adapter (creates default if None)

    Returns:
        Icon string for the status
    """
    if presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        presentation = create_presentation_adapter()

    # Map step statuses to icon names
    icon_map = {
        StepStatus.PENDING: "pending",
        StepStatus.IN_PROGRESS: "refresh",
        StepStatus.COMPLETED: "success",
        StepStatus.FAILED: "error",
        StepStatus.SKIPPED: "skipped",
        StepStatus.BLOCKED: "blocked",
    }

    icon_name = icon_map.get(status, "unknown")
    return presentation.icon(icon_name, with_color=False)


class StepStatus(Enum):
    """Status of a plan step."""

    PENDING = "pending"  # Not yet started
    IN_PROGRESS = "in_progress"  # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed with error
    SKIPPED = "skipped"  # Skipped due to dependency failure
    BLOCKED = "blocked"  # Blocked by user approval


class StepType(Enum):
    """Type of step indicating what kind of work it does."""

    RESEARCH = "research"  # Information gathering (read-only)
    PLANNING = "planning"  # Sub-planning or task breakdown
    IMPLEMENTATION = "implementation"  # Code changes
    TESTING = "testing"  # Running tests
    REVIEW = "review"  # Code review or validation
    DEPLOYMENT = "deployment"  # Deployment actions


@dataclass
class PlanStep:
    """A single step in an execution plan.

    Steps are the atomic units of work in a plan. Each step has:
    - A unique ID for tracking
    - A description of what needs to be done
    - A type indicating what kind of work
    - Optional dependencies on other steps
    - Optional sub-agent role for delegation

    Attributes:
        id: Unique identifier for this step
        description: Human-readable description of what to do
        step_type: Type of work (research, implementation, etc.)
        depends_on: List of step IDs that must complete first
        estimated_tool_calls: Estimated number of tool calls needed
        requires_approval: Whether user approval needed before execution
        sub_agent_role: If set, delegate to this sub-agent role
        context: Additional context/data for the step
        status: Current execution status
        result: Result after execution (if completed)
    """

    id: str
    description: str
    step_type: StepType = StepType.IMPLEMENTATION
    depends_on: list[str] = field(default_factory=list)
    estimated_tool_calls: int = 10
    requires_approval: bool = False
    sub_agent_role: Optional[str] = None  # "researcher", "executor", etc.
    context: dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Optional["StepResult"] = None

    def is_ready(self, completed_steps: set[str]) -> bool:
        """Check if this step is ready to execute.

        A step is ready when:
        - It's in PENDING status
        - All dependencies are completed
        - It doesn't require approval (or approval was granted)

        Args:
            completed_steps: Set of step IDs that have completed

        Returns:
            True if step is ready to execute
        """
        if self.status != StepStatus.PENDING:
            return False
        return all(dep in completed_steps for dep in self.depends_on)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "step_type": self.step_type.value,
            "depends_on": self.depends_on,
            "estimated_tool_calls": self.estimated_tool_calls,
            "requires_approval": self.requires_approval,
            "sub_agent_role": self.sub_agent_role,
            "context": self.context,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanStep":
        """Create from dictionary."""
        result_data = data.get("result")
        result = StepResult.from_dict(result_data) if result_data else None

        return cls(
            id=data["id"],
            description=data["description"],
            step_type=StepType(data.get("step_type", "implementation")),
            depends_on=data.get("depends_on", []),
            estimated_tool_calls=data.get("estimated_tool_calls", 10),
            requires_approval=data.get("requires_approval", False),
            sub_agent_role=data.get("sub_agent_role"),
            context=data.get("context", {}),
            status=StepStatus(data.get("status", "pending")),
            result=result,
        )


@dataclass
class StepResult:
    """Result of executing a single step.

    Captures both success/failure and detailed metrics about execution.

    Attributes:
        success: Whether the step completed successfully
        output: Primary output/result of the step
        error: Error message if failed
        tool_calls_used: Number of tool calls made
        duration_seconds: Execution time
        artifacts: Any files created/modified
    """

    success: bool
    output: str
    error: Optional[str] = None
    tool_calls_used: int = 0
    duration_seconds: float = 0.0
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "tool_calls_used": self.tool_calls_used,
            "duration_seconds": self.duration_seconds,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepResult":
        """Create from dictionary."""
        return cls(
            success=data["success"],
            output=data["output"],
            error=data.get("error"),
            tool_calls_used=data.get("tool_calls_used", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            artifacts=data.get("artifacts", []),
        )


@dataclass
class ExecutionPlan:
    """A complete execution plan for achieving a goal.

    An ExecutionPlan is a directed acyclic graph (DAG) of steps, where
    dependencies define the execution order. Steps without dependencies
    can execute in parallel (if using sub-agents).

    Attributes:
        id: Unique identifier for this plan
        goal: High-level goal the plan achieves
        steps: List of steps in the plan
        created_at: Timestamp when plan was created
        approved: Whether the plan has been approved for execution
        metadata: Additional metadata about the plan
    """

    id: str
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    approved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_ready_steps(self) -> list[PlanStep]:
        """Get all steps that are ready to execute.

        Returns:
            List of steps ready to execute (pending with satisfied dependencies)
        """
        completed = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        return [s for s in self.steps if s.is_ready(completed)]

    def get_completed_steps(self) -> list[PlanStep]:
        """Get all completed steps."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]

    def get_failed_steps(self) -> list[PlanStep]:
        """Get all failed steps."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def is_complete(self) -> bool:
        """Check if the plan is fully complete."""
        return all(s.status == StepStatus.COMPLETED for s in self.steps)

    def is_failed(self) -> bool:
        """Check if any step failed (and plan cannot continue)."""
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def total_estimated_tool_calls(self) -> int:
        """Calculate total estimated tool calls."""
        return sum(s.estimated_tool_calls for s in self.steps)

    def progress_percentage(self) -> float:
        """Calculate progress as percentage."""
        if not self.steps:
            return 100.0
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return (completed / len(self.steps)) * 100

    def to_markdown(self, presentation: Optional["PresentationProtocol"] = None) -> str:
        """Convert plan to markdown format for display/approval.

        Args:
            presentation: Optional presentation adapter for icons (creates default if None)

        Returns:
            Markdown string representation of the plan
        """
        # Get presentation adapter for arrow icon
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            presentation = create_presentation_adapter()

        arrow = presentation.icon("arrow_right", with_color=False)

        lines = [
            f"# Execution Plan: {self.goal}",
            "",
            f"**Plan ID:** {self.id}",
            f"**Total Steps:** {len(self.steps)}",
            f"**Estimated Tool Calls:** {self.total_estimated_tool_calls()}",
            "",
            "## Steps",
            "",
        ]

        for i, step in enumerate(self.steps, 1):
            status_icon = _get_step_status_icon(step.status, presentation)

            deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
            approval = " *[requires approval]*" if step.requires_approval else ""
            agent = f" {arrow} _{step.sub_agent_role}_" if step.sub_agent_role else ""

            lines.append(f"### {i}. {status_icon} {step.description}{approval}")
            lines.append(f"- **Type:** {step.step_type.value}")
            lines.append(f"- **Estimated calls:** ~{step.estimated_tool_calls}")
            if deps:
                lines.append(f"- **Dependencies:** {', '.join(step.depends_on)}")
            if agent:
                lines.append(f"- **Sub-agent:** {step.sub_agent_role}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "approved": self.approved,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionPlan":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            goal=data["goal"],
            steps=[PlanStep.from_dict(s) for s in data.get("steps", [])],
            created_at=data.get("created_at", time.time()),
            approved=data.get("approved", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PlanResult:
    """Overall result of executing a plan.

    Aggregates results from all steps and provides summary metrics.

    Attributes:
        plan_id: ID of the plan that was executed
        success: Whether all steps completed successfully
        steps_completed: Number of steps completed
        steps_failed: Number of steps failed
        total_tool_calls: Total tool calls across all steps
        total_duration: Total execution time
        final_output: Final summary/output of the plan
        step_results: Mapping of step ID to result
    """

    plan_id: str
    success: bool
    steps_completed: int = 0
    steps_failed: int = 0
    total_tool_calls: int = 0
    total_duration: float = 0.0
    final_output: str = ""
    step_results: dict[str, StepResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "success": self.success,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "total_tool_calls": self.total_tool_calls,
            "total_duration": self.total_duration,
            "final_output": self.final_output,
            "step_results": {k: v.to_dict() for k, v in self.step_results.items()},
        }


__all__ = [
    "StepStatus",
    "StepType",
    "PlanStep",
    "StepResult",
    "ExecutionPlan",
    "PlanResult",
]
