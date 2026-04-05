"""Safety-related protocol definitions.

These protocols define how verticals provide safety checks and constraints.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, List, Dict, Any, Optional, Callable


@runtime_checkable
class SafetyProvider(Protocol):
    """Protocol for providing safety rules and constraints.

    Safety providers can validate tool calls, prompts, and outputs
    to prevent harmful or unauthorized operations.
    """

    def get_safety_rules(self) -> Dict[str, Any]:
        """Return safety rules for this vertical.

        Returns:
            Dictionary of safety rule configurations
        """
        ...

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Validate a tool call before execution.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments being passed to the tool

        Returns:
            True if the tool call is safe, False otherwise
        """
        ...

    def validate_prompt(self, prompt: str) -> bool:
        """Validate a user prompt before processing.

        Args:
            prompt: User's input prompt

        Returns:
            True if the prompt is safe, False otherwise
        """
        ...


@runtime_checkable
class SafetyExtension(Protocol):
    """Protocol for extending safety capabilities.

    Safety extensions provide additional safety checks beyond the
    basic validation provided by SafetyProvider.
    """

    def pre_check(self, context: Dict[str, Any]) -> Optional[str]:
        """Perform pre-execution safety check.

        Args:
            context: Execution context with tool, arguments, etc.

        Returns:
            Error message if unsafe, None if safe
        """
        ...

    def post_check(self, result: Any, context: Dict[str, Any]) -> Optional[str]:
        """Perform post-execution safety check.

        Args:
            result: Result from tool execution
            context: Original execution context

        Returns:
            Error message if result is unsafe, None if safe
        """
        ...

    def get_guardrails(self) -> List[Callable[[Dict[str, Any]], Optional[str]]]:
        """Return list of guardrail functions.

        Returns:
            List of functions that take context and return error or None
        """
        ...


@runtime_checkable
class SafetyPattern(Protocol):
    """Protocol for safety patterns.

    Safety patterns are reusable safety configurations that can be
    applied across multiple verticals.
    """

    def get_pattern_name(self) -> str:
        """Return the name of this safety pattern."""
        ...

    def get_pattern_description(self) -> str:
        """Return description of this safety pattern."""
        ...

    def apply_to_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this safety pattern to an execution context.

        Args:
            context: Original execution context

        Returns:
            Modified context with safety patterns applied
        """
        ...
