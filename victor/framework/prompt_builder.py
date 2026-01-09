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

"""Consolidated Prompt Builder for Victor Framework.

This module provides a unified, composable prompt building system that consolidates
duplicate prompt construction logic from across the codebase. It supports:

- Section-based composition with priority ordering
- Tool hints integration
- Safety rules injection
- Contextual information merging
- Grounding rules (minimal vs extended)
- Task-type-specific hints

The PromptBuilder uses a fluent API pattern for ergonomic construction:

    prompt = (
        PromptBuilder()
        .add_section("identity", "You are an expert code analyst.", priority=10)
        .add_section("guidelines", "Follow best practices.", priority=20)
        .add_tool_hints({"read": "Use to read file contents"})
        .add_safety_rules(["Never expose credentials"])
        .add_context("Working in Python codebase")
        .build()
    )

Integration with verticals:

    from victor.coding.prompts import CodingPromptContributor

    builder = PromptBuilder()
    builder.add_from_contributor(CodingPromptContributor())
    prompt = builder.build()
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# Python 3.10+ supports typing.Self
from typing import Self

if TYPE_CHECKING:
    from victor.core.verticals.protocols import PromptContributorProtocol

logger = logging.getLogger(__name__)


@dataclass
class PromptSection:
    """A section of the system prompt.

    Sections are composed together to form the final prompt, ordered by priority.
    Lower priority values appear earlier in the final prompt.

    Attributes:
        name: Unique identifier for the section (e.g., "identity", "grounding")
        content: The actual prompt text for this section
        priority: Ordering priority (lower = earlier). Default is 50.
        enabled: Whether this section is included in the final prompt
        header: Optional header format. If None, uses "## {name}". Set to "" to disable.
    """

    name: str
    content: str
    priority: int = 50
    enabled: bool = True
    header: Optional[str] = None

    def get_formatted_content(self) -> str:
        """Get the section content with optional header.

        Returns:
            Formatted section content with header if applicable
        """
        if not self.enabled:
            return ""

        # Determine header
        if self.header is None:
            # Default: use section name as header
            header_text = f"## {self.name.replace('_', ' ').title()}\n"
        elif self.header == "":
            # Empty string: no header
            header_text = ""
        else:
            # Custom header
            header_text = f"{self.header}\n"

        return f"{header_text}{self.content}"


@dataclass
class ToolHint:
    """A hint for using a specific tool.

    Attributes:
        tool_name: The canonical name of the tool
        hint: Usage guidance for the tool
        priority_boost: Boost to apply when selecting this tool (0.0-1.0)
    """

    tool_name: str
    hint: str
    priority_boost: float = 0.0


class PromptBuilder:
    """Builds system prompts from composable sections.

    This class consolidates prompt building logic from across the codebase,
    providing a unified API for constructing system prompts with:

    - Named sections with priority ordering
    - Tool usage hints
    - Safety rules
    - Contextual information
    - Grounding rules

    The builder uses a fluent API pattern, allowing method chaining:

        prompt = (
            PromptBuilder()
            .add_section("identity", "You are an AI assistant.")
            .add_tool_hints({"read": "Read file contents"})
            .build()
        )

    Attributes:
        _sections: Dictionary of named prompt sections
        _tool_hints: Dictionary of tool name to hint text
        _safety_rules: List of safety rules to include
        _context: List of contextual information strings
        _grounding_mode: "minimal" or "extended" grounding rules
    """

    # Standard priority levels for common sections
    PRIORITY_IDENTITY = 10  # Who the assistant is
    PRIORITY_CAPABILITIES = 20  # What it can do
    PRIORITY_GUIDELINES = 30  # How to behave
    PRIORITY_TASK_HINTS = 40  # Task-specific guidance
    PRIORITY_TOOL_GUIDANCE = 50  # Tool usage hints
    PRIORITY_SAFETY = 60  # Safety rules
    PRIORITY_CONTEXT = 70  # Contextual information
    PRIORITY_GROUNDING = 80  # Grounding rules (last)

    def __init__(self) -> None:
        """Initialize a new PromptBuilder."""
        self._sections: Dict[str, PromptSection] = {}
        self._tool_hints: Dict[str, ToolHint] = {}
        self._safety_rules: List[str] = []
        self._context: List[str] = []
        self._grounding_mode: str = "minimal"
        self._custom_grounding: Optional[str] = None

    def add_section(
        self,
        name: str,
        content: str,
        priority: int = 50,
        header: Optional[str] = None,
    ) -> Self:
        """Add a named section to the prompt.

        Sections are ordered by priority when building the final prompt.
        Lower priority values appear earlier.

        Args:
            name: Unique identifier for the section
            content: The prompt text for this section
            priority: Ordering priority (default 50, lower = earlier)
            header: Optional custom header. None = auto-generate, "" = no header

        Returns:
            Self for method chaining
        """
        self._sections[name] = PromptSection(
            name=name,
            content=content,
            priority=priority,
            enabled=True,
            header=header,
        )
        return self

    def add_tool_hints(self, hints: Dict[str, str]) -> Self:
        """Add tool usage hints.

        Tool hints provide guidance on how to use specific tools effectively.
        These are merged into the prompt in the tool guidance section.

        Args:
            hints: Dictionary mapping tool names to hint text

        Returns:
            Self for method chaining
        """
        for tool_name, hint in hints.items():
            self._tool_hints[tool_name] = ToolHint(tool_name=tool_name, hint=hint)
        return self

    def add_tool_hint(
        self,
        tool_name: str,
        hint: str,
        priority_boost: float = 0.0,
    ) -> Self:
        """Add a single tool hint with optional priority boost.

        Args:
            tool_name: The canonical name of the tool
            hint: Usage guidance for the tool
            priority_boost: Boost to apply when selecting this tool (0.0-1.0)

        Returns:
            Self for method chaining
        """
        self._tool_hints[tool_name] = ToolHint(
            tool_name=tool_name,
            hint=hint,
            priority_boost=priority_boost,
        )
        return self

    def add_safety_rules(self, rules: List[str]) -> Self:
        """Add safety rules.

        Safety rules are constraints the model should follow to ensure
        safe operation (e.g., "Never expose credentials").

        Args:
            rules: List of safety rule strings

        Returns:
            Self for method chaining
        """
        self._safety_rules.extend(rules)
        return self

    def add_safety_rule(self, rule: str) -> Self:
        """Add a single safety rule.

        Args:
            rule: Safety rule string

        Returns:
            Self for method chaining
        """
        self._safety_rules.append(rule)
        return self

    def add_context(self, context: str) -> Self:
        """Add contextual information.

        Context provides situational information like project details,
        file types being worked with, or current task state.

        Args:
            context: Contextual information string

        Returns:
            Self for method chaining
        """
        self._context.append(context)
        return self

    def set_grounding_mode(self, mode: str) -> Self:
        """Set the grounding rules mode.

        Args:
            mode: Either "minimal" or "extended"

        Returns:
            Self for method chaining

        Raises:
            ValueError: If mode is not "minimal" or "extended"
        """
        if mode not in ("minimal", "extended"):
            raise ValueError(f"Invalid grounding mode: {mode}. Use 'minimal' or 'extended'.")
        self._grounding_mode = mode
        return self

    def set_custom_grounding(self, grounding: str) -> Self:
        """Set custom grounding rules (overrides mode).

        Args:
            grounding: Custom grounding rules text

        Returns:
            Self for method chaining
        """
        self._custom_grounding = grounding
        return self

    def remove_section(self, name: str) -> Self:
        """Remove a section by name.

        Args:
            name: The name of the section to remove

        Returns:
            Self for method chaining
        """
        self._sections.pop(name, None)
        return self

    def disable_section(self, name: str) -> Self:
        """Disable a section without removing it.

        Args:
            name: The name of the section to disable

        Returns:
            Self for method chaining
        """
        if name in self._sections:
            self._sections[name].enabled = False
        return self

    def enable_section(self, name: str) -> Self:
        """Enable a previously disabled section.

        Args:
            name: The name of the section to enable

        Returns:
            Self for method chaining
        """
        if name in self._sections:
            self._sections[name].enabled = True
        return self

    def has_section(self, name: str) -> bool:
        """Check if a section exists.

        Args:
            name: The name of the section to check

        Returns:
            True if the section exists, False otherwise
        """
        return name in self._sections

    def get_section(self, name: str) -> Optional[PromptSection]:
        """Get a section by name.

        Args:
            name: The name of the section to get

        Returns:
            The PromptSection if found, None otherwise
        """
        return self._sections.get(name)

    def clear(self) -> Self:
        """Clear all sections, hints, rules, and context.

        Returns:
            Self for method chaining
        """
        self._sections.clear()
        self._tool_hints.clear()
        self._safety_rules.clear()
        self._context.clear()
        self._grounding_mode = "minimal"
        self._custom_grounding = None
        return self

    def add_from_contributor(
        self,
        contributor: "PromptContributorProtocol",
        task_type: Optional[str] = None,
    ) -> Self:
        """Add sections from a PromptContributorProtocol implementation.

        This integrates with vertical-specific prompt contributors like
        CodingPromptContributor, DevOpsPromptContributor, etc.

        Args:
            contributor: A PromptContributorProtocol implementation
            task_type: Optional task type to get specific hints for

        Returns:
            Self for method chaining
        """
        # Get priority from contributor
        priority_offset = contributor.get_priority() * 10

        # Add system prompt section
        system_section = contributor.get_system_prompt_section()
        if system_section:
            self.add_section(
                name=f"vertical_{type(contributor).__name__}",
                content=system_section,
                priority=self.PRIORITY_GUIDELINES + priority_offset,
            )

        # Add grounding rules
        grounding = contributor.get_grounding_rules()
        if grounding:
            self.set_custom_grounding(grounding)

        # Add task-type-specific hint if task_type provided
        if task_type:
            hints = contributor.get_task_type_hints()
            if task_type.lower() in hints:
                task_hint = hints[task_type.lower()]
                hint_text = task_hint.hint if hasattr(task_hint, "hint") else str(task_hint)
                self.add_section(
                    name="task_hint",
                    content=hint_text,
                    priority=self.PRIORITY_TASK_HINTS,
                    header="",  # Task hints usually have their own markers like [EDIT]
                )

                # Add tool hints from task type if available
                if hasattr(task_hint, "priority_tools") and task_hint.priority_tools:
                    for tool in task_hint.priority_tools:
                        self.add_tool_hint(tool, f"Prioritized for {task_type}", 0.2)

        return self

    def merge(self, other: "PromptBuilder") -> Self:
        """Merge another PromptBuilder into this one.

        Sections from the other builder override sections with the same name.
        Tool hints, safety rules, and context are merged.

        Args:
            other: Another PromptBuilder to merge from

        Returns:
            Self for method chaining
        """
        # Merge sections (other overrides same-named sections)
        for name, section in other._sections.items():
            self._sections[name] = section

        # Merge tool hints
        self._tool_hints.update(other._tool_hints)

        # Merge safety rules (deduplicate)
        for rule in other._safety_rules:
            if rule not in self._safety_rules:
                self._safety_rules.append(rule)

        # Merge context
        for ctx in other._context:
            if ctx not in self._context:
                self._context.append(ctx)

        # Take grounding from other if it has custom grounding
        if other._custom_grounding:
            self._custom_grounding = other._custom_grounding
        else:
            # Otherwise prefer "extended" mode
            if other._grounding_mode == "extended":
                self._grounding_mode = "extended"

        return self

    def _get_grounding_rules(self) -> str:
        """Get the appropriate grounding rules.

        Returns:
            Grounding rules text based on mode or custom setting
        """
        # Import here to avoid circular imports
        from victor.framework.prompt_sections import (
            GROUNDING_RULES_MINIMAL,
            GROUNDING_RULES_EXTENDED,
        )

        if self._custom_grounding:
            return self._custom_grounding

        if self._grounding_mode == "extended":
            return GROUNDING_RULES_EXTENDED
        return GROUNDING_RULES_MINIMAL

    def build(self) -> str:
        """Build the final prompt string.

        Assembles all sections, tool hints, safety rules, context, and
        grounding rules into a single prompt string, ordered by priority.

        Returns:
            The complete system prompt string
        """
        parts: List[str] = []

        # Add sections sorted by priority
        sorted_sections = sorted(
            [s for s in self._sections.values() if s.enabled],
            key=lambda s: s.priority,
        )
        for section in sorted_sections:
            formatted = section.get_formatted_content()
            if formatted:
                parts.append(formatted)

        # Add tool hints if any
        if self._tool_hints:
            hints_lines = [f"- {hint.tool_name}: {hint.hint}" for hint in self._tool_hints.values()]
            hints_text = "\n".join(hints_lines)
            parts.append(f"## Tool Hints\n{hints_text}")

        # Add safety rules if any
        if self._safety_rules:
            rules_lines = [f"- {rule}" for rule in self._safety_rules]
            rules_text = "\n".join(rules_lines)
            parts.append(f"## Safety Rules\n{rules_text}")

        # Add context if any
        if self._context:
            context_text = "\n\n".join(self._context)
            parts.append(f"## Context\n{context_text}")

        # Add grounding rules (always last)
        grounding = self._get_grounding_rules()
        if grounding:
            parts.append(grounding)

        return "\n\n".join(parts)

    def __repr__(self) -> str:
        """Return a string representation of the builder state."""
        return (
            f"PromptBuilder("
            f"sections={len(self._sections)}, "
            f"tool_hints={len(self._tool_hints)}, "
            f"safety_rules={len(self._safety_rules)}, "
            f"context={len(self._context)}, "
            f"grounding='{self._grounding_mode}'"
            f")"
        )


# Convenience factory functions


def create_coding_prompt_builder() -> PromptBuilder:
    """Create a PromptBuilder pre-configured for coding tasks.

    Returns:
        A PromptBuilder with coding-specific defaults
    """
    from victor.framework.prompt_sections import (
        CODING_IDENTITY,
        CODING_GUIDELINES,
        CODING_TOOL_USAGE,
    )

    return (
        PromptBuilder()
        .add_section("identity", CODING_IDENTITY, priority=PromptBuilder.PRIORITY_IDENTITY)
        .add_section("guidelines", CODING_GUIDELINES, priority=PromptBuilder.PRIORITY_GUIDELINES)
        .add_section("tool_usage", CODING_TOOL_USAGE, priority=PromptBuilder.PRIORITY_TOOL_GUIDANCE)
    )


def create_devops_prompt_builder() -> PromptBuilder:
    """Create a PromptBuilder pre-configured for DevOps tasks.

    Returns:
        A PromptBuilder with DevOps-specific defaults
    """
    from victor.framework.prompt_sections import (
        DEVOPS_IDENTITY,
        DEVOPS_SECURITY_CHECKLIST,
        DEVOPS_COMMON_PITFALLS,
    )

    return (
        PromptBuilder()
        .add_section("identity", DEVOPS_IDENTITY, priority=PromptBuilder.PRIORITY_IDENTITY)
        .add_section("security", DEVOPS_SECURITY_CHECKLIST, priority=PromptBuilder.PRIORITY_SAFETY)
        .add_section("pitfalls", DEVOPS_COMMON_PITFALLS, priority=PromptBuilder.PRIORITY_GUIDELINES)
    )


def create_research_prompt_builder() -> PromptBuilder:
    """Create a PromptBuilder pre-configured for research tasks.

    Returns:
        A PromptBuilder with research-specific defaults
    """
    from victor.framework.prompt_sections import (
        RESEARCH_IDENTITY,
        RESEARCH_QUALITY_CHECKLIST,
        RESEARCH_SOURCE_HIERARCHY,
    )

    return (
        PromptBuilder()
        .add_section("identity", RESEARCH_IDENTITY, priority=PromptBuilder.PRIORITY_IDENTITY)
        .add_section(
            "quality", RESEARCH_QUALITY_CHECKLIST, priority=PromptBuilder.PRIORITY_GUIDELINES
        )
        .add_section(
            "sources", RESEARCH_SOURCE_HIERARCHY, priority=PromptBuilder.PRIORITY_GUIDELINES + 5
        )
    )


def create_data_analysis_prompt_builder() -> PromptBuilder:
    """Create a PromptBuilder pre-configured for data analysis tasks.

    Returns:
        A PromptBuilder with data analysis-specific defaults
    """
    from victor.framework.prompt_sections import (
        DATA_ANALYSIS_IDENTITY,
        DATA_ANALYSIS_LIBRARIES,
        DATA_ANALYSIS_OPERATIONS,
    )

    return (
        PromptBuilder()
        .add_section("identity", DATA_ANALYSIS_IDENTITY, priority=PromptBuilder.PRIORITY_IDENTITY)
        .add_section(
            "libraries", DATA_ANALYSIS_LIBRARIES, priority=PromptBuilder.PRIORITY_CAPABILITIES
        )
        .add_section(
            "operations", DATA_ANALYSIS_OPERATIONS, priority=PromptBuilder.PRIORITY_GUIDELINES
        )
    )


__all__ = [
    "PromptSection",
    "ToolHint",
    "PromptBuilder",
    "create_coding_prompt_builder",
    "create_devops_prompt_builder",
    "create_research_prompt_builder",
    "create_data_analysis_prompt_builder",
]
