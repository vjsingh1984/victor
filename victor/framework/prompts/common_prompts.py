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

"""Common prompt patterns and templates for Victor Framework.

This module extracts and consolidates common prompt patterns used across
all verticals (Coding, Research, DevOps, DataAnalysis, RAG, Benchmark).

Common Pattern Categories:
1. Grounding Rules - Ensure responses are based on actual tool output
2. Task Hints - Task-type-specific guidance
3. System Prompt Sections - Vertical-specific instructions
4. Tool Selection Guidance - How to choose and use tools
5. Safety Rules - Constraints to ensure safe operation
6. Checklists - Verification steps

Usage:
    from victor.framework.prompts.common_prompts import (
        GroundingRulesBuilder,
        TaskHintBuilder,
        SystemPromptBuilder,
    )

    # Build grounding rules
    grounding = GroundingRulesBuilder().extended().build()

    # Build task hints
    hint = TaskHintBuilder() \\
        .for_task_type("edit") \\
        .with_priority_tools(["read", "edit"]) \\
        .with_guidance("Read target file first, then modify") \\
        .build()

    # Build system prompt section
    section = SystemPromptBuilder() \\
        .with_identity("You are an expert coding assistant") \\
        .with_guidelines(["Understand before modifying", "Make incremental changes"]) \\
        .build()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# ENUMS
# =============================================================================


class GroundingMode(Enum):
    """Grounding rule modes."""

    MINIMAL = "minimal"
    EXTENDED = "extended"
    CUSTOM = "custom"


class TaskCategory(Enum):
    """High-level task categories."""

    CREATION = "creation"
    MODIFICATION = "modification"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    GENERAL = "general"


# =============================================================================
# GROUNDING RULES BUILDER
# =============================================================================


class GroundingRulesBuilder:
    """Builder for constructing grounding rules.

    Grounding rules ensure the model bases responses on actual tool output
    rather than fabricating or hallucinating content.

    Attributes:
        _mode: Grounding mode (minimal, extended, or custom)
        _custom_rules: Custom grounding rules when mode is CUSTOM

    Example:
        # Minimal grounding
        grounding = GroundingRulesBuilder().minimal().build()

        # Extended grounding (for local models)
        grounding = GroundingRulesBuilder().extended().build()

        # Custom grounding
        grounding = GroundingRulesBuilder() \\
            .custom("Always verify with actual data") \\
            .build()
    """

    # Predefined grounding templates
    MINIMAL_TEMPLATE = """
GROUNDING: Base ALL responses on tool output only. Never invent file paths or content.
Quote code exactly from tool output. If more info needed, call another tool.
""".strip()

    EXTENDED_TEMPLATE = """
CRITICAL - TOOL OUTPUT GROUNDING:
When you receive tool output in <TOOL_OUTPUT> tags:
1. The content between markers is ACTUAL file/command output - NEVER ignore it
2. You MUST base your analysis ONLY on this actual content
3. NEVER fabricate, invent, or imagine file contents that differ from tool output
4. If you need more information, call another tool - do NOT guess
5. When citing code, quote EXACTLY from the tool output
6. If tool output is empty or truncated, acknowledge this limitation

VIOLATION OF THESE RULES WILL RESULT IN INCORRECT ANALYSIS.
""".strip()

    def __init__(self) -> None:
        """Initialize a new GroundingRulesBuilder."""
        self._mode: GroundingMode = GroundingMode.MINIMAL
        self._custom_rules: str = ""

    def minimal(self) -> "GroundingRulesBuilder":
        """Use minimal grounding rules.

        Returns:
            Self for method chaining
        """
        self._mode = GroundingMode.MINIMAL
        return self

    def extended(self) -> "GroundingRulesBuilder":
        """Use extended grounding rules (recommended for local models).

        Returns:
            Self for method chaining
        """
        self._mode = GroundingMode.EXTENDED
        return self

    def custom(self, rules: str) -> "GroundingRulesBuilder":
        """Use custom grounding rules.

        Args:
            rules: Custom grounding rules text

        Returns:
            Self for method chaining
        """
        self._mode = GroundingMode.CUSTOM
        self._custom_rules = rules
        return self

    def build(self) -> str:
        """Build the grounding rules string.

        Returns:
            Grounding rules text
        """
        if self._mode == GroundingMode.MINIMAL:
            return self.MINIMAL_TEMPLATE
        elif self._mode == GroundingMode.EXTENDED:
            return self.EXTENDED_TEMPLATE
        else:
            return self._custom_rules


# =============================================================================
# TASK HINT BUILDER
# =============================================================================


@dataclass
class TaskHint:
    """A task-specific hint for guiding model behavior.

    Attributes:
        task_type: The type of task (e.g., "edit", "create", "analyze")
        category: High-level task category
        hint: The hint text to display to the model
        tool_budget: Suggested maximum number of tool calls
        priority_tools: Tools that should be prioritized for this task
        workflow_steps: Optional step-by-step workflow
        rules: Optional rules to follow
        anti_patterns: Optional anti-patterns to avoid

    Example:
        hint = TaskHint(
            task_type="edit",
            category=TaskCategory.MODIFICATION,
            hint="[EDIT] Read target file first, then modify.",
            tool_budget=5,
            priority_tools=["read", "edit"]
        )
    """

    task_type: str
    category: TaskCategory
    hint: str
    tool_budget: int = 10
    priority_tools: List[str] = field(default_factory=list)
    workflow_steps: Optional[List[str]] = None
    rules: Optional[List[str]] = None
    anti_patterns: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "task_type": self.task_type,
            "category": self.category.value,
            "hint": self.hint,
            "tool_budget": self.tool_budget,
            "priority_tools": self.priority_tools,
            "workflow_steps": self.workflow_steps,
            "rules": self.rules,
            "anti_patterns": self.anti_patterns,
        }


class TaskHintBuilder:
    """Builder for constructing task hints.

    Task hints provide task-specific guidance to the model, including
    what tools to use, what budget to follow, and what patterns to avoid.

    Attributes:
        _task_type: The task type
        _category: Task category
        _hint: Hint text
        _tool_budget: Tool budget
        _priority_tools: Prioritized tools
        _workflow_steps: Workflow steps
        _rules: Rules to follow
        _anti_patterns: Anti-patterns to avoid

    Example:
        hint = TaskHintBuilder() \\
            .for_task_type("edit") \\
            .with_category(TaskCategory.MODIFICATION) \\
            .with_guidance("[EDIT] Read target file first") \\
            .with_tool_budget(5) \\
            .with_priority_tools(["read", "edit"]) \\
            .build()
    """

    def __init__(self) -> None:
        """Initialize a new TaskHintBuilder."""
        self._task_type: str = ""
        self._category: TaskCategory = TaskCategory.GENERAL
        self._hint: str = ""
        self._tool_budget: int = 10
        self._priority_tools: List[str] = []
        self._workflow_steps: Optional[List[str]] = None
        self._rules: Optional[List[str]] = None
        self._anti_patterns: Optional[List[str]] = None

    def for_task_type(self, task_type: str) -> "TaskHintBuilder":
        """Set the task type.

        Args:
            task_type: The task type identifier

        Returns:
            Self for method chaining
        """
        self._task_type = task_type
        return self

    def with_category(self, category: TaskCategory) -> "TaskHintBuilder":
        """Set the task category.

        Args:
            category: The task category

        Returns:
            Self for method chaining
        """
        self._category = category
        return self

    def with_guidance(self, hint: str) -> "TaskHintBuilder":
        """Set the hint text.

        Args:
            hint: The hint text

        Returns:
            Self for method chaining
        """
        self._hint = hint
        return self

    def with_tool_budget(self, budget: int) -> "TaskHintBuilder":
        """Set the tool budget.

        Args:
            budget: Maximum number of tool calls

        Returns:
            Self for method chaining
        """
        self._tool_budget = budget
        return self

    def with_priority_tools(self, tools: List[str]) -> "TaskHintBuilder":
        """Set priority tools.

        Args:
            tools: List of tool names to prioritize

        Returns:
            Self for method chaining
        """
        self._priority_tools = tools
        return self

    def with_workflow(self, steps: List[str]) -> "TaskHintBuilder":
        """Set workflow steps.

        Args:
            steps: Step-by-step workflow

        Returns:
            Self for method chaining
        """
        self._workflow_steps = steps
        return self

    def with_rules(self, rules: List[str]) -> "TaskHintBuilder":
        """Set rules to follow.

        Args:
            rules: List of rules

        Returns:
            Self for method chaining
        """
        self._rules = rules
        return self

    def with_anti_patterns(self, patterns: List[str]) -> "TaskHintBuilder":
        """Set anti-patterns to avoid.

        Args:
            patterns: List of anti-patterns

        Returns:
            Self for method chaining
        """
        self._anti_patterns = patterns
        return self

    def build(self) -> TaskHint:
        """Build the TaskHint.

        Returns:
            TaskHint object

        Raises:
            ValueError: If task_type or hint is not set
        """
        if not self._task_type:
            raise ValueError("task_type must be set using for_task_type()")
        if not self._hint:
            raise ValueError("hint must be set using with_guidance()")

        return TaskHint(
            task_type=self._task_type,
            category=self._category,
            hint=self._hint,
            tool_budget=self._tool_budget,
            priority_tools=self._priority_tools,
            workflow_steps=self._workflow_steps,
            rules=self._rules,
            anti_patterns=self._anti_patterns,
        )


# =============================================================================
# SYSTEM PROMPT BUILDER
# =============================================================================


class SystemPromptBuilder:
    """Builder for constructing system prompt sections.

    System prompt sections provide vertical-specific instructions to the model.
    They typically include identity, capabilities, guidelines, and best practices.

    Attributes:
        _identity: Who the assistant is
        _capabilities: What it can do
        _guidelines: How to behave
        _best_practices: Recommended practices
        _tool_usage: How to use tools effectively

    Example:
        section = SystemPromptBuilder() \\
            .with_identity("You are an expert coding assistant") \\
            .with_capabilities(["Code analysis", "Test generation"]) \\
            .with_guidelines(["Understand before modifying", "Make incremental changes"]) \\
            .build()
    """

    def __init__(self) -> None:
        """Initialize a new SystemPromptBuilder."""
        self._identity: str = ""
        self._capabilities: List[str] = []
        self._guidelines: List[str] = []
        self._best_practices: List[str] = []
        self._tool_usage: str = ""

    def with_identity(self, identity: str) -> "SystemPromptBuilder":
        """Set the identity section.

        Args:
            identity: Who the assistant is

        Returns:
            Self for method chaining
        """
        self._identity = identity
        return self

    def with_capabilities(self, capabilities: List[str]) -> "SystemPromptBuilder":
        """Set capabilities.

        Args:
            capabilities: List of capabilities

        Returns:
            Self for method chaining
        """
        self._capabilities = capabilities
        return self

    def with_guidelines(self, guidelines: List[str]) -> "SystemPromptBuilder":
        """Set guidelines.

        Args:
            guidelines: List of guidelines

        Returns:
            Self for method chaining
        """
        self._guidelines = guidelines
        return self

    def with_best_practices(self, practices: List[str]) -> "SystemPromptBuilder":
        """Set best practices.

        Args:
            practices: List of best practices

        Returns:
            Self for method chaining
        """
        self._best_practices = practices
        return self

    def with_tool_usage(self, tool_usage: str) -> "SystemPromptBuilder":
        """Set tool usage guidance.

        Args:
            tool_usage: Tool usage text

        Returns:
            Self for method chaining
        """
        self._tool_usage = tool_usage
        return self

    def build(self) -> str:
        """Build the system prompt section.

        Returns:
            Formatted system prompt section
        """
        parts: List[str] = []

        # Identity
        if self._identity:
            parts.append(self._identity)

        # Capabilities
        if self._capabilities:
            caps = "\n".join(f"- {cap}" for cap in self._capabilities)
            parts.append(f"\nYour capabilities:\n{caps}")

        # Guidelines
        if self._guidelines:
            guidelines = "\n".join(f"{i}. **{guide}**" for i, guide in enumerate(self._guidelines, 1))
            parts.append(f"\nGuidelines:\n{guidelines}")

        # Best practices
        if self._best_practices:
            practices = "\n".join(f"- {practice}" for practice in self._best_practices)
            parts.append(f"\nBest practices:\n{practices}")

        # Tool usage
        if self._tool_usage:
            parts.append(f"\n{self._tool_usage}")

        return "\n".join(parts).strip()


# =============================================================================
# CHECKLIST BUILDER
# =============================================================================


class ChecklistBuilder:
    """Builder for constructing verification checklists.

    Checklists provide step-by-step verification procedures to ensure
    quality and completeness.

    Attributes:
        _items: Checklist items
        _title: Checklist title

    Example:
        checklist = ChecklistBuilder() \\
            .with_title("Security Checklist") \\
            .add_item("No hardcoded secrets") \\
            .add_item("Least-privilege IAM policies") \\
            .build()
    """

    def __init__(self) -> None:
        """Initialize a new ChecklistBuilder."""
        self._items: List[str] = []
        self._title: str = "Checklist"

    def with_title(self, title: str) -> "ChecklistBuilder":
        """Set the checklist title.

        Args:
            title: Checklist title

        Returns:
            Self for method chaining
        """
        self._title = title
        return self

    def add_item(self, item: str) -> "ChecklistBuilder":
        """Add a checklist item.

        Args:
            item: Checklist item text

        Returns:
            Self for method chaining
        """
        self._items.append(item)
        return self

    def add_items(self, items: List[str]) -> "ChecklistBuilder":
        """Add multiple checklist items.

        Args:
            items: List of checklist item texts

        Returns:
            Self for method chaining
        """
        self._items.extend(items)
        return self

    def build(self) -> str:
        """Build the checklist.

        Returns:
            Formatted checklist as markdown checkboxes
        """
        if not self._items:
            return ""

        items = "\n".join(f"- [ ] {item}" for item in self._items)
        return f"## {self._title}\n\n{items}"


# =============================================================================
# SAFETY RULES BUILDER
# =============================================================================


class SafetyRulesBuilder:
    """Builder for constructing safety rules.

    Safety rules are constraints the model should follow to ensure
    safe operation (e.g., "Never expose credentials").

    Attributes:
        _rules: List of safety rules

    Example:
        rules = SafetyRulesBuilder() \\
            .add_rule("Never expose credentials") \\
            .add_rule("Validate all user input") \\
            .build()
    """

    def __init__(self) -> None:
        """Initialize a new SafetyRulesBuilder."""
        self._rules: List[str] = []

    def add_rule(self, rule: str) -> "SafetyRulesBuilder":
        """Add a safety rule.

        Args:
            rule: Safety rule text

        Returns:
            Self for method chaining
        """
        self._rules.append(rule)
        return self

    def add_rules(self, rules: List[str]) -> "SafetyRulesBuilder":
        """Add multiple safety rules.

        Args:
            rules: List of safety rules

        Returns:
            Self for method chaining
        """
        self._rules.extend(rules)
        return self

    def build(self) -> str:
        """Build the safety rules section.

        Returns:
            Formatted safety rules
        """
        if not self._rules:
            return ""

        rules = "\n".join(f"- {rule}" for rule in self._rules)
        return f"## Safety Rules\n\n{rules}"


# =============================================================================
# COMMON WORKFLOW TEMPLATES
# =============================================================================


class WorkflowTemplate(ABC):
    """Base class for workflow templates.

    Workflow templates provide reusable step-by-step procedures
    for common task patterns.
    """

    @abstractmethod
    def get_steps(self) -> List[str]:
        """Get workflow steps.

        Returns:
            List of workflow steps
        """
        ...

    def render(self) -> str:
        """Render workflow as formatted text.

        Returns:
            Formatted workflow steps
        """
        steps = "\n".join(f"{i}. {step}" for i, step in enumerate(self.get_steps(), 1))
        return f"Workflow:\n{steps}"


class BugFixWorkflow(WorkflowTemplate):
    """Bug fix workflow template.

    Provides structured workflow for fixing bugs efficiently.
    """

    def get_steps(self) -> List[str]:
        """Get bug fix workflow steps.

        Returns:
            List of workflow steps
        """
        return [
            "UNDERSTAND: Read error traceback and issue description carefully",
            "LOCATE: Use code_search to find the exact location of the bug",
            "ANALYZE: Read the buggy code and understand the root cause",
            "FIX: Make minimal, surgical changes - only modify what's necessary",
            "VERIFY: Run tests to confirm the fix works",
        ]


class CodeCreationWorkflow(WorkflowTemplate):
    """Code creation workflow template.

    Provides structured workflow for creating new code.
    """

    def get_steps(self) -> List[str]:
        """Get code creation workflow steps.

        Returns:
            List of workflow steps
        """
        return [
            "UNDERSTAND: Read the problem statement or requirements carefully",
            "IMPLEMENT: Write clean, correct code that solves the problem",
            "VERIFY: Test with provided examples or edge cases",
        ]


class AnalysisWorkflow(WorkflowTemplate):
    """Analysis workflow template.

    Provides structured workflow for analyzing code or data.
    """

    def get_steps(self) -> List[str]:
        """Get analysis workflow steps.

        Returns:
            List of workflow steps
        """
        return [
            "EXPLORE: Read relevant files and gather context",
            "ANALYZE: Examine patterns, relationships, and issues",
            "SYNTHESIZE: Structure findings and insights",
            "REPORT: Provide clear, actionable recommendations",
        ]


__all__ = [
    # Enums
    "GroundingMode",
    "TaskCategory",
    # Builders
    "GroundingRulesBuilder",
    "TaskHintBuilder",
    "SystemPromptBuilder",
    "ChecklistBuilder",
    "SafetyRulesBuilder",
    # Data classes
    "TaskHint",
    # Workflow templates
    "WorkflowTemplate",
    "BugFixWorkflow",
    "CodeCreationWorkflow",
    "AnalysisWorkflow",
]
