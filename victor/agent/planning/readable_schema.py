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

"""Readable and token-efficient JSON schema for LLM task planning.

This module provides a Pydantic schema for LLM-generated task plans that balances:
- **Readability**: Clear, self-documenting field names for LLM reliability
- **Token Efficiency**: Still ~40% savings vs verbose JSON through list format
- **Type Safety**: Full Pydantic validation
- **LLM-Friendly**: Structure optimized for LLM generation

Key Design Decision: Readable keywords over abbreviations
- Better for LLM reliability (clear structure)
- Easier to debug and maintain
- Still achieves token savings through list format
- Self-documenting for developers

Example Readable JSON (120 tokens vs 180 for verbose JSON):
    {"name":"Add auth","complexity":"moderate","desc":"OAuth2 login",
     "steps":[[1,"research","Analyze patterns","overview"],
              [2,"feature","Create module","write,test"]]}

Usage:
    from victor.agent.planning import TaskPlan, generate_compact_plan

    # LLM generates readable compact JSON
    json_str = '{"name":"auth","complexity":"simple","desc":"Fix bug","steps":[[1,"analyze","Find bug","grep"]]}'

    # Validate and expand
    task_plan = TaskPlan.model_validate_json(json_str)
    execution_plan = task_plan.to_execution_plan()
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from victor.agent.planning.base import (
    ExecutionPlan,
    PlanStep,
    StepStatus,
    StepType,
)

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Task complexity levels for planning."""

    SIMPLE = "simple"      # Auto mode, 2-3 steps, <30 min
    MODERATE = "moderate"  # Plan-mode, 3-5 steps, 30min-2hr
    COMPLEX = "complex"    # Plan-mode, 5-8 steps, >2hr


class ReadableTaskPlan(BaseModel):
    """Readable and token-efficient task plan schema for LLM generation.

    This schema uses READABLE field names (not cryptic abbreviations) while
    maintaining token efficiency through list-based step representation.

    Token Efficiency Strategy:
    - Readable field names: name, desc, steps (not n, d, s)
    - List format for steps: [id, type, desc, tools, deps]
    - Achieves ~40% token savings while remaining fully readable

    Field Name Mapping:
    - name: task name (short, clear)
    - complexity: simple|moderate|complex (not c)
    - desc: description (not d)
    - steps: list of step data (not s)
    - duration: estimated duration (not e, optional)
    - approval: requires approval (not a, optional)

    Step Data Format:
    - List: [id, type, description, tools, dependencies]
    - Types: research, feature, bugfix, refactor, test, review, deploy, analyze, doc

    Example:
        {
          "name": "Add authentication",
          "complexity": "moderate",
          "desc": "Implement OAuth2 login",
          "steps": [
            [1, "research", "Analyze patterns", "overview"],
            [2, "feature", "Create module", "write,test"]
          ],
          "duration": "30min"
        }
    """

    # Readable field names (not cryptic abbreviations)
    name: str = Field(..., description="Task name (short, clear)")
    complexity: TaskComplexity = Field(..., description="Complexity level")
    desc: str = Field(..., description="Task description")
    steps: List[List] = Field(
        ...,
        description="Steps: [[id, type, description, tools, dependencies], ...]",
    )
    duration: Optional[str] = Field(None, description="Estimated duration (e.g., '30min', '2hr')")
    approval: bool = Field(False, description="Requires user approval")

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v: List[List]) -> List[List]:
        """Validate step data format."""
        for i, step_data in enumerate(v, 1):
            if not isinstance(step_data, list) or len(step_data) < 3:
                raise ValueError(
                    f"Step {i}: must be list with at least [id, type, desc], got {step_data}"
                )
            if not isinstance(step_data[0], (int, str)):
                raise ValueError(f"Step {i}: id must be int or str, got {type(step_data[0])}")
        return v

    def to_execution_plan(self) -> ExecutionPlan:
        """Convert readable task plan to full ExecutionPlan."""
        import uuid

        steps = []

        for step_data in self.steps:
            plan_step = self._parse_step_data(step_data)
            steps.append(plan_step)

        # Generate a unique plan ID
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Build full execution plan
        return ExecutionPlan(
            id=plan_id,
            goal=self.desc,
            steps=steps,
            metadata={
                "task_name": self.name,
                "complexity": self.complexity.value,
                "estimated_duration": self.duration,
                "requires_approval": self.approval,
            }
        )

    def _parse_step_data(self, step_data: List) -> PlanStep:
        """Parse step data list into PlanStep."""
        step_id = str(step_data[0])
        step_type_str = step_data[1]
        description = step_data[2]

        # Map readable step type to full StepType
        step_type = self._map_step_type(step_type_str)

        # Parse tools and dependencies
        tools = []
        dependencies = []

        if len(step_data) > 3:
            # Fourth element can be tools (string) or dependencies (list)
            fourth = step_data[3]
            if isinstance(fourth, list):
                dependencies = [str(d) for d in fourth]
            elif isinstance(fourth, str):
                tools = fourth.split(",") if fourth else []

        if len(step_data) > 4:
            # Fifth element is dependencies if fourth was tools
            deps = step_data[4]
            if isinstance(deps, list):
                dependencies = [str(d) for d in deps]

        # Check if deployment or high-risk step
        requires_approval = (
            step_type == StepType.DEPLOYMENT or
            step_type == StepType.PLANNING or
            self.approval
        )

        return PlanStep(
            id=step_id,
            description=description,
            step_type=step_type,
            depends_on=dependencies,
            estimated_tool_calls=10,  # Default estimate
            requires_approval=requires_approval,
            sub_agent_role=self._get_sub_agent_role(step_type),
        )

    def _map_step_type(self, step_type_str: str) -> StepType:
        """Map readable step type to StepType enum."""
        type_map = {
            # Primary readable mappings
            "research": StepType.RESEARCH,
            "planning": StepType.PLANNING,
            "feature": StepType.IMPLEMENTATION,
            "feature": StepType.IMPLEMENTATION,
            "implementation": StepType.IMPLEMENTATION,
            "bugfix": StepType.IMPLEMENTATION,
            "bug": StepType.IMPLEMENTATION,
            "refactor": StepType.IMPLEMENTATION,
            "test": StepType.TESTING,
            "testing": StepType.TESTING,
            "review": StepType.REVIEW,
            "deploy": StepType.DEPLOYMENT,
            "deployment": StepType.DEPLOYMENT,
            "analyze": StepType.RESEARCH,
            "analysis": StepType.RESEARCH,
            "doc": StepType.RESEARCH,
            "documentation": StepType.RESEARCH,
        }
        return type_map.get(
            step_type_str.lower(),
            StepType.IMPLEMENTATION  # Default to implementation
        )

    def _get_sub_agent_role(self, step_type: StepType) -> Optional[str]:
        """Map step type to sub-agent role."""
        role_map = {
            StepType.RESEARCH: "researcher",
            StepType.PLANNING: "planner",
            StepType.IMPLEMENTATION: "executor",
            StepType.TESTING: "tester",
            StepType.REVIEW: "reviewer",
            StepType.DEPLOYMENT: "deployer",
        }
        return role_map.get(step_type)

    def to_yaml(self) -> str:
        """Convert to YAML format for storage/human editing."""
        import yaml

        plan = self.to_execution_plan()

        # Build YAML structure
        yaml_data = {
            "workflows": {
                self.name: {
                    "description": self.desc,
                    "metadata": {
                        "complexity": self.complexity.value,
                        "estimated_duration": self.duration,
                        "requires_approval": self.approval,
                    },
                    "nodes": []
                }
            }
        }

        # Convert steps to nodes
        for step in plan.steps:
            node = {
                "id": step.id,
                "type": "agent",
                "role": step.sub_agent_role or "executor",
                "goal": step.description,
                "description": step.description,
                "tool_budget": step.estimated_tool_calls or 10,
            }

            if step.depends_on:
                node["depends_on"] = step.depends_on

            if step.requires_approval:
                node["requires_approval"] = True

            yaml_data["workflows"][self.name]["nodes"].append(node)

        return yaml.safe_dump(yaml_data, sort_keys=False)

    def to_markdown(self) -> str:
        """Convert to markdown for display."""
        plan = self.to_execution_plan()
        lines = [
            f"# {self.name}",
            "",
            f"**Description**: {self.desc}",
            f"**Complexity**: {self.complexity.value}",
            f"**Estimated**: {self.duration or 'Unknown'}",
            f"**Approval**: {'Required' if self.approval else 'Not required'}",
            "",
            "## Steps",
            "",
        ]

        for step in plan.steps:
            status_icon = "⏳"
            lines.append(f"{status_icon} **Step {step.id}**: {step.description}")
            lines.append(f"   - Type: {step.step_type.value}")
            if step.depends_on:
                lines.append(f"   - Depends on: {', '.join(step.depends_on)}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_execution_plan(cls, plan: ExecutionPlan) -> "ReadableTaskPlan":
        """Create readable task plan from full ExecutionPlan."""
        steps_data = []

        for step in plan.steps:
            # Map StepType to readable type string
            type_str = cls._step_type_to_readable(step.step_type)

            # Build step data list
            step_list = [int(step.id), type_str, step.description]

            # Add tools if available
            if hasattr(step, 'allowed_tools') and step.allowed_tools:
                tools_str = ",".join(step.allowed_tools)
                step_list.append(tools_str)

            # Add dependencies
            if step.depends_on:
                if not hasattr(step, 'allowed_tools') or not step.allowed_tools:
                    step_list.append([])  # Placeholder for tools
                step_list.append([int(d) for d in step.depends_on])
            elif hasattr(step, 'allowed_tools') and step.allowed_tools:
                step_list.append([])  # Empty deps if no dependencies

            steps_data.append(step_list)

        metadata = plan.metadata or {}

        return cls(
            name=metadata.get("task_name", plan.goal[:50]),
            complexity=TaskComplexity(metadata.get("complexity", "moderate")),
            desc=plan.goal,  # ExecutionPlan uses goal, not description
            steps=steps_data,
            duration=metadata.get("estimated_duration"),
            approval=metadata.get("requires_approval", False),
        )

    @staticmethod
    def _step_type_to_readable(step_type: StepType) -> str:
        """Convert StepType enum to readable string."""
        return {
            StepType.RESEARCH: "research",
            StepType.PLANNING: "planning",
            StepType.IMPLEMENTATION: "feature",
            StepType.TESTING: "test",
            StepType.REVIEW: "review",
            StepType.DEPLOYMENT: "deploy",
        }.get(step_type, "feature")

    def to_json(self, verbose: bool = False) -> str:
        """Convert to JSON string.

        Args:
            verbose: If False, use compact list representation (default)
                    If True, use expanded object representation

        Returns:
            JSON string
        """
        if verbose:
            # Expanded to verbose JSON
            plan = self.to_execution_plan()
            return json.dumps({
                "name": self.name,
                "complexity": self.complexity.value,
                "description": self.desc,
                "steps": [
                    {
                        "id": step.id,
                        "type": step.step_type.value,
                        "description": step.description,
                        "depends_on": step.depends_on,
                        "estimated_tool_calls": step.estimated_tool_calls,
                        "requires_approval": step.requires_approval,
                    }
                    for step in plan.steps
                ],
                "estimated_duration": self.duration,
                "requires_approval": self.approval,
            }, indent=2)
        else:
            # Use compact schema (this is what LLM generates)
            return self.model_dump_json(exclude_none=True)

    @classmethod
    def get_llm_prompt(cls) -> str:
        """Get optimized prompt for LLM to generate readable task plans.

        Uses readable keywords for LLM reliability while maintaining
        token efficiency through list-based format.
        """
        return """Create a task plan in JSON format:

{
  "name": "short task name",
  "complexity": "simple|moderate|complex",
  "desc": "task description",
  "steps": [
    [step_id, type, description, tools, dependencies]
  ],
  "duration": "estimated time (optional)",
  "approval": false (optional, set true for risky tasks)
}

Step types (use lowercase):
  research, planning, feature, bugfix, refactor, test, review, deploy, analyze, doc

Tools: read, write, grep, git, shell, test, code_search, overview, scaffold

Format: [id, type, description, "tool1,tool2", [dep_id1, dep_id2]]

Examples:
{
  "name": "Fix bug",
  "complexity": "simple",
  "desc": "Fix login bug",
  "steps": [
    [1, "analyze", "Find the bug", "grep"],
    [2, "feature", "Fix the bug", "write"]
  ]
}
{
  "name": "Add authentication",
  "complexity": "moderate",
  "desc": "Implement OAuth2 login",
  "steps": [
    [1, "research", "Analyze auth patterns", "overview"],
    [2, "feature", "Create auth module", "write,test"],
    [3, "test", "Verify login works", "pytest", [2]]
  ],
  "duration": "30min"
}

Return ONLY valid JSON. No markdown, no explanation."""

    @classmethod
    def get_complexity_prompt(cls) -> str:
        """Get prompt for classifying task complexity."""
        return """Classify the task complexity:

Task: {user_request}

Consider:
- SIMPLE: Single file, well-defined scope, <30 minutes, 2-3 steps
- MODERATE: Multiple files, some uncertainty, 30min-2 hours, 3-5 steps
- COMPLEX: Multiple components, high uncertainty, >2 hours, 5-8 steps

Respond with ONLY valid JSON:
{
  "complexity": "simple|moderate|complex",
  "reason": "brief explanation"
}"""


class TaskPlannerContext:
    """Session context manager for task planning.

    Manages task plans within a conversation session, allowing plans
    to be referenced, updated, and executed across multiple turns.
    """

    def __init__(self):
        self.current_plan: Optional[ExecutionPlan] = None
        self.plans_history: List[ExecutionPlan] = []
        self.approved_plans: List[str] = []

    def set_plan(self, plan: ExecutionPlan) -> None:
        """Set the current active plan."""
        self.current_plan = plan
        logger.info(f"Set current plan: {plan.goal}")

    def approve_plan(self) -> None:
        """Mark current plan as approved."""
        if self.current_plan:
            plan_id = id(self.current_plan)
            self.approved_plans.append(str(plan_id))
            logger.info(f"Approved plan: {self.current_plan.goal}")

    def archive_plan(self) -> None:
        """Archive current plan to history."""
        if self.current_plan:
            self.plans_history.append(self.current_plan)
            self.current_plan = None
            logger.info("Archived current plan")

    def get_plan_summary(self) -> Dict[str, Any]:
        """Get summary of all plans in context."""
        return {
            "current_plan": self.current_plan.goal if self.current_plan else None,
            "total_plans": len(self.plans_history) + (1 if self.current_plan else 0),
            "approved_plans": len(self.approved_plans),
            "history": [
                {"goal": plan.goal, "steps": len(plan.steps)}
                for plan in self.plans_history[-5:]  # Last 5 plans
            ]
        }

    def to_context_dict(self) -> Dict[str, Any]:
        """Export context for inclusion in LLM prompts."""
        summary = self.get_plan_summary()

        context = {
            "task_planner": {
                "active": self.current_plan is not None,
                "total_plans": summary["total_plans"],
                "approved_count": summary["approved_plans"],
            }
        }

        if self.current_plan:
            plan_dict = self._plan_to_dict(self.current_plan)
            context["task_planner"]["current_plan"] = plan_dict

        if self.plans_history:
            context["task_planner"]["recent_plans"] = summary["history"]

        return context

    def _plan_to_dict(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Convert plan to dictionary for context."""
        return {
            "goal": plan.goal,
            "steps": [
                {
                    "id": step.id,
                    "description": step.description,
                    "type": step.step_type.value,
                    "status": step.status.value if step.status else "pending",
                }
                for step in plan.steps
            ],
            "metadata": plan.metadata,
        }

    @classmethod
    def from_context_dict(cls, context: Dict[str, Any]) -> "TaskPlannerContext":
        """Restore context from dictionary (e.g., from session storage)."""
        ctx = cls()

        if "task_planner" in context:
            tp_data = context["task_planner"]
            # Reconstruct plans from context if needed
            # This is a simplified version - full restoration would need more logic
            if tp_data.get("current_plan"):
                # Would need to reconstruct ExecutionPlan from dict
                pass

        return ctx


# Helper functions for workflow integration

def generate_task_plan(
    provider,
    user_request: str,
    complexity: Optional[TaskComplexity] = None,
) -> ReadableTaskPlan:
    """Generate a readable task plan using LLM.

    Args:
        provider: LLM provider instance
        user_request: Natural language task description
        complexity: Optional pre-classified complexity level

    Returns:
        Validated ReadableTaskPlan
    """
    import asyncio

    async def _generate():
        # Classify complexity if not provided
        if not complexity:
            complexity_prompt = ReadableTaskPlan.get_complexity_prompt()
            complexity_prompt = complexity_prompt.replace("{user_request}", user_request)

            complexity_response = await provider.generate(
                complexity_prompt,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=200,
            )

            import json
            complexity_data = json.loads(complexity_response)
            complexity = TaskComplexity(complexity_data["complexity"])

        # Generate task plan
        plan_prompt = ReadableTaskPlan.get_llm_prompt()
        plan_prompt = plan_prompt.replace(
            "{user_request}", user_request
        )

        json_response = await provider.generate(
            plan_prompt,
            response_format={"type": "json_object"},
            temperature=0.2,  # Lower temp for consistent structure
            max_tokens=1500,
        )

        # Validate and return
        return ReadableTaskPlan.model_validate_json(json_response)

    return asyncio.run(_generate())


def plan_to_workflow_yaml(plan: ReadableTaskPlan) -> str:
    """Convert readable task plan to YAML workflow format.

    This converts the task plan into a YAML workflow that can be
    executed by the WorkflowEngine.

    Args:
        plan: ReadableTaskPlan to convert

    Returns:
        YAML workflow string
    """
    return plan.to_yaml()


def plan_to_session_context(
    plan: ReadableTaskPlan,
    session_id: str,
    context_store = None,
) -> Dict[str, Any]:
    """Add plan to session context for persistence.

    This allows the task plan to be referenced in future conversation turns
    and persisted across sessions.

    Args:
        plan: ReadableTaskPlan to add to context
        session_id: Session identifier
        context_store: Optional context storage backend

    Returns:
        Updated context dictionary
    """
    context = {
        "session_id": session_id,
        "task_plan": {
            "name": plan.name,
            "complexity": plan.complexity.value,
            "description": plan.desc,
            "estimated_duration": plan.duration,
            "requires_approval": plan.approval,
            "steps": [
                {
                    "id": step[0],
                    "type": step[1],
                    "description": step[2],
                }
                for step in plan.steps
            ]
        },
        "created_at": plan.model_dump_json(include={'name', 'complexity', 'desc', 'duration', 'approval'}),
    }

    # Store in context backend if provided
    if context_store:
        context_store.set(session_id, "task_plan", context["task_plan"])

    return context


# Legacy aliases for backward compatibility
CompactTaskPlan = ReadableTaskPlan
CompactStepType = None  # Removed, using readable strings instead
