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

"""Prompt Coordinator - Coordinates system prompt assembly.

This module extracts prompt assembly logic from AgentOrchestrator,
providing a focused interface for:
- System prompt construction using PromptBuilder
- Vertical prompt contributor integration
- Task-type-specific hint injection
- Context and grounding rules assembly

Design Philosophy:
- Single Responsibility: Coordinates all prompt-related operations
- Composable: Works with PromptBuilder and PromptContributorProtocol
- Extensible: Supports vertical-specific prompt sections
- Backward Compatible: Maintains API compatibility with orchestrator

Usage:
    coordinator = PromptCoordinator(
        prompt_builder=PromptBuilder(),
        vertical_context=vertical_ctx,
    )

    # Build system prompt for task
    context = TaskContext(message="fix the bug", task_type="bugfix")
    prompt = coordinator.build_system_prompt(context)

    # Add task-specific hints
    coordinator.add_task_hint("bugfix", "Check error handling first")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

from victor.framework.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from victor.agent.vertical_context import VerticalContext
    from victor.core.verticals.protocols import PromptContributorProtocol
    from victor.core.vertical_types import TaskTypeHint

logger = logging.getLogger(__name__)


@dataclass
class TaskContext:
    """Context for prompt building.

    Attributes:
        message: The user's current message/query
        task_type: Detected task type (e.g., "edit", "analyze", "debug")
        complexity: Task complexity level
        stage: Current conversation stage
        model: Current model name
        provider: Current provider name
        additional_context: Extra context to include
    """

    message: str
    task_type: str = "unknown"
    complexity: str = "medium"
    stage: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptCoordinatorConfig:
    """Configuration for PromptCoordinator.

    Attributes:
        default_grounding_mode: Default grounding rules mode
        enable_task_hints: Whether to include task-type hints
        enable_vertical_sections: Whether to include vertical sections
        enable_safety_rules: Whether to include safety rules
        max_context_tokens: Maximum tokens for context section
    """

    default_grounding_mode: str = "minimal"
    enable_task_hints: bool = True
    enable_vertical_sections: bool = True
    enable_safety_rules: bool = True
    max_context_tokens: int = 2000


@runtime_checkable
class IPromptCoordinator(Protocol):
    """Protocol for prompt coordination operations."""

    def build_system_prompt(self, context: TaskContext) -> str: ...
    def add_task_hint(self, task_type: str, hint: str) -> None: ...
    def get_task_hint(self, task_type: str) -> Optional[str]: ...


class PromptCoordinator:
    """Coordinates system prompt assembly.

    This class consolidates prompt-related operations that were spread
    across the orchestrator, providing a unified interface for:

    1. Prompt Building: Uses PromptBuilder for section-based composition
    2. Vertical Integration: Adds vertical-specific prompt sections
    3. Task Hints: Injects task-type-specific guidance
    4. Context Injection: Adds situational context to prompts

    Example:
        coordinator = PromptCoordinator(
            prompt_builder=PromptBuilder(),
        )

        # Build prompt for current task
        context = TaskContext(
            message="Fix the authentication bug",
            task_type="bugfix",
            model="claude-opus-4",
        )
        prompt = coordinator.build_system_prompt(context)

        # Later, update for different task
        context.task_type = "refactor"
        prompt = coordinator.build_system_prompt(context)
    """

    def __init__(
        self,
        prompt_builder: Optional[PromptBuilder] = None,
        vertical_context: Optional["VerticalContext"] = None,
        config: Optional[PromptCoordinatorConfig] = None,
        base_identity: Optional[str] = None,
        on_prompt_built: Optional[Callable[[str, TaskContext], None]] = None,
    ) -> None:
        """Initialize the PromptCoordinator.

        Args:
            prompt_builder: Builder for prompt composition
            vertical_context: Optional vertical context for sections
            config: Configuration options
            base_identity: Base identity section for the prompt
            on_prompt_built: Callback when prompt is built
        """
        self._builder = prompt_builder or PromptBuilder()
        self._vertical_context = vertical_context
        self._config = config or PromptCoordinatorConfig()
        self._base_identity = base_identity
        self._on_prompt_built = on_prompt_built

        # Custom task hints (override vertical hints)
        self._task_hints: Dict[str, str] = {}

        # Additional sections added at runtime
        self._additional_sections: Dict[str, str] = {}

        # Safety rules
        self._safety_rules: List[str] = []

        logger.debug(
            f"PromptCoordinator initialized with grounding_mode={self._config.default_grounding_mode}"
        )

    @property
    def vertical_context(self) -> Optional["VerticalContext"]:
        """Get the current vertical context."""
        return self._vertical_context

    @vertical_context.setter
    def vertical_context(self, context: Optional["VerticalContext"]) -> None:
        """Set the vertical context."""
        self._vertical_context = context

    def build_system_prompt(
        self,
        context: TaskContext,
        include_hints: bool = True,
    ) -> str:
        """Build the complete system prompt.

        Assembles the prompt from:
        1. Base identity section
        2. Vertical-specific sections
        3. Task-type hints
        4. Safety rules
        5. Context information
        6. Grounding rules

        Args:
            context: Task context for prompt building
            include_hints: Whether to include task hints

        Returns:
            Complete system prompt string
        """
        # Start fresh
        builder = PromptBuilder()

        # Add base identity
        if self._base_identity:
            builder.add_section(
                "identity",
                self._base_identity,
                priority=PromptBuilder.PRIORITY_IDENTITY,
                header="",  # Identity usually doesn't need a header
            )

        # Add vertical sections
        if self._config.enable_vertical_sections and self._vertical_context:
            self._add_vertical_sections(builder, context)

        # Add task-specific hint
        if include_hints and self._config.enable_task_hints:
            self._add_task_hint(builder, context)

        # Add additional runtime sections
        for name, content in self._additional_sections.items():
            builder.add_section(name, content)

        # Add safety rules
        if self._config.enable_safety_rules and self._safety_rules:
            builder.add_safety_rules(self._safety_rules)

        # Add context
        if context.additional_context:
            for key, value in context.additional_context.items():
                if isinstance(value, str):
                    builder.add_context(f"{key}: {value}")

        # Set grounding mode
        builder.set_grounding_mode(self._config.default_grounding_mode)

        # Build the prompt
        prompt = builder.build()

        # Callback
        if self._on_prompt_built:
            self._on_prompt_built(prompt, context)

        logger.debug(
            f"Built system prompt for task_type={context.task_type}, " f"length={len(prompt)} chars"
        )

        return prompt

    def _add_vertical_sections(
        self,
        builder: PromptBuilder,
        context: TaskContext,
    ) -> None:
        """Add vertical-specific sections to the builder.

        Args:
            builder: PromptBuilder to add sections to
            context: Task context
        """
        if not self._vertical_context:
            return

        # Get prompt extensions from vertical
        prompt_ext = self._vertical_context.get_prompt_extensions()
        if not prompt_ext:
            return

        # Add system prompt sections from contributors
        sections = prompt_ext.get_combined_system_prompt_sections()
        if sections:
            builder.add_section(
                "vertical_guidelines",
                sections,
                priority=PromptBuilder.PRIORITY_GUIDELINES + 5,
            )

        # Add grounding rules if available
        grounding = prompt_ext.get_grounding_rules()
        if grounding:
            builder.set_custom_grounding(grounding)

    def _add_task_hint(
        self,
        builder: PromptBuilder,
        context: TaskContext,
    ) -> None:
        """Add task-type-specific hint to the builder.

        Args:
            builder: PromptBuilder to add hint to
            context: Task context
        """
        task_type = context.task_type.lower()

        # Check custom hints first
        if task_type in self._task_hints:
            builder.add_section(
                "task_hint",
                self._task_hints[task_type],
                priority=PromptBuilder.PRIORITY_TASK_HINTS,
                header="",  # Task hints often have their own markers
            )
            return

        # Fall back to vertical hints
        if self._vertical_context:
            prompt_ext = self._vertical_context.get_prompt_extensions()
            if prompt_ext:
                hint = prompt_ext.get_hint_for_task(task_type)
                if hint:
                    hint_text = hint.hint if hasattr(hint, "hint") else str(hint)
                    builder.add_section(
                        "task_hint",
                        hint_text,
                        priority=PromptBuilder.PRIORITY_TASK_HINTS,
                        header="",
                    )

                    # Add priority tool hints
                    if hasattr(hint, "priority_tools") and hint.priority_tools:
                        for tool in hint.priority_tools:
                            builder.add_tool_hint(
                                tool,
                                f"Prioritized for {task_type} tasks",
                                priority_boost=0.2,
                            )

    def add_task_hint(self, task_type: str, hint: str) -> None:
        """Add or update a task-type hint.

        Args:
            task_type: Task type (e.g., "edit", "debug")
            hint: Hint text for this task type
        """
        self._task_hints[task_type.lower()] = hint
        logger.debug(f"Added task hint for {task_type}")

    def remove_task_hint(self, task_type: str) -> None:
        """Remove a task-type hint.

        Args:
            task_type: Task type to remove hint for
        """
        self._task_hints.pop(task_type.lower(), None)

    def get_task_hint(self, task_type: str) -> Optional[str]:
        """Get the hint for a task type.

        Args:
            task_type: Task type to get hint for

        Returns:
            Hint string or None
        """
        return self._task_hints.get(task_type.lower())

    def add_section(
        self,
        name: str,
        content: str,
        priority: Optional[int] = None,
    ) -> None:
        """Add a runtime section to be included in prompts.

        Args:
            name: Section name (unique identifier)
            content: Section content
            priority: Optional priority (not stored, used in next build)
        """
        self._additional_sections[name] = content
        logger.debug(f"Added section '{name}'")

    def remove_section(self, name: str) -> None:
        """Remove a runtime section.

        Args:
            name: Section name to remove
        """
        self._additional_sections.pop(name, None)

    def add_safety_rule(self, rule: str) -> None:
        """Add a safety rule.

        Args:
            rule: Safety rule text
        """
        if rule not in self._safety_rules:
            self._safety_rules.append(rule)

    def clear_safety_rules(self) -> None:
        """Clear all safety rules."""
        self._safety_rules.clear()

    def set_grounding_mode(self, mode: str) -> None:
        """Set the grounding rules mode.

        Args:
            mode: "minimal" or "extended"
        """
        if mode in ("minimal", "extended"):
            self._config.default_grounding_mode = mode
        else:
            logger.warning(f"Invalid grounding mode: {mode}")

    def set_base_identity(self, identity: str) -> None:
        """Set the base identity section.

        Args:
            identity: Identity text
        """
        self._base_identity = identity

    def get_all_task_hints(self) -> Dict[str, str]:
        """Get all configured task hints.

        Returns:
            Dict mapping task types to hints
        """
        return dict(self._task_hints)

    def clear(self) -> None:
        """Clear all custom sections and hints."""
        self._task_hints.clear()
        self._additional_sections.clear()
        self._safety_rules.clear()
        logger.debug("PromptCoordinator cleared")


def create_prompt_coordinator(
    prompt_builder: Optional[PromptBuilder] = None,
    vertical_context: Optional["VerticalContext"] = None,
    config: Optional[PromptCoordinatorConfig] = None,
    base_identity: Optional[str] = None,
) -> PromptCoordinator:
    """Factory function to create a PromptCoordinator.

    Args:
        prompt_builder: Builder for prompt composition
        vertical_context: Optional vertical context for sections
        config: Configuration options
        base_identity: Base identity section for the prompt

    Returns:
        Configured PromptCoordinator instance
    """
    return PromptCoordinator(
        prompt_builder=prompt_builder,
        vertical_context=vertical_context,
        config=config,
        base_identity=base_identity,
    )


__all__ = [
    "PromptCoordinator",
    "PromptCoordinatorConfig",
    "TaskContext",
    "IPromptCoordinator",
    "create_prompt_coordinator",
]
