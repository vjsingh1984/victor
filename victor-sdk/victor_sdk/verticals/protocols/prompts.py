"""Prompt-related protocol definitions.

These protocols define how verticals provide and customize prompts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, List, Dict, Any, Optional

from victor_sdk.core.types import Tier


@runtime_checkable
class PromptProvider(Protocol):
    """Protocol for providing prompt configurations.

    Prompt providers define how prompts are structured and customized
    for specific use cases.
    """

    def get_base_prompt(self) -> str:
        """Return the base system prompt.

        Returns:
            Base system prompt text
        """
        ...

    def get_prompt_template(self, task_type: str) -> str:
        """Return prompt template for a specific task type.

        Args:
            task_type: Type of task (e.g., "code_generation", "analysis")

        Returns:
            Prompt template string
        """
        ...

    def format_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """Format a prompt template with context.

        Args:
            template: Prompt template string
            context: Variables to substitute in template

        Returns:
            Formatted prompt string
        """
        ...


@runtime_checkable
class PromptContributor(Protocol):
    """Protocol for contributing to prompt generation.

    Prompt contributors can add dynamic content to prompts based on
    context, task type, or other factors.
    """

    def contribute(self, context: Dict[str, Any]) -> str:
        """Generate contribution to the prompt.

        Args:
            context: Current execution context

        Returns:
            String to add to the prompt
        """
        ...

    def get_task_type_hints(self) -> List[TaskTypeHint]:
        """Return task type hints for this contributor.

        Returns:
            List of task type hints this contributor provides
        """
        ...


@runtime_checkable
class TaskTypeHint(Protocol):
    """Protocol for task type hints.

    Task type hints help the framework understand what types of tasks
    a vertical is optimized for.
    """

    def get_task_type(self) -> str:
        """Return the task type identifier."""
        ...

    def get_description(self) -> str:
        """Return description of this task type."""
        ...

    def matches_context(self, context: Dict[str, Any]) -> bool:
        """Check if this hint matches the given context.

        Args:
            context: Execution context to check

        Returns:
            True if this hint applies to the context
        """
        ...
