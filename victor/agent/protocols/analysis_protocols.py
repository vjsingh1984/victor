"""Protocol definitions for analysis protocols."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Protocol,
    TYPE_CHECKING,
    runtime_checkable,
)


__all__ = [
    "TaskAnalyzerProtocol",
    "ComplexityClassifierProtocol",
    "ActionAuthorizerProtocol",
    "SearchRouterProtocol",
    "IntentClassifierProtocol",
    "TaskTypeHinterProtocol",
    "IToolCallClassifier",
]

@runtime_checkable
class TaskAnalyzerProtocol(Protocol):
    """Protocol for task analysis.

    Analyzes user prompts for complexity, intent, and routing.
    """

    def analyze(self, prompt: str) -> Dict[str, Any]:
        """Analyze a user prompt.

        Returns:
            Analysis results (complexity, intent, etc.)
        """
        ...

    def classify_complexity(self, prompt: str) -> Any:
        """Classify task complexity."""
        ...

    def detect_intent(self, prompt: str) -> Any:
        """Detect user intent."""
        ...


@runtime_checkable
class ComplexityClassifierProtocol(Protocol):
    """Protocol for complexity classification."""

    def classify(self, prompt: str) -> Any:
        """Classify prompt complexity.

        Returns:
            TaskComplexity enum value
        """
        ...


@runtime_checkable
class ActionAuthorizerProtocol(Protocol):
    """Protocol for action authorization."""

    def authorize(self, action: str, context: Dict[str, Any]) -> bool:
        """Check if an action is authorized.

        Returns:
            True if authorized
        """
        ...

    def detect_intent(self, prompt: str) -> Any:
        """Detect action intent from prompt."""
        ...


@runtime_checkable
class SearchRouterProtocol(Protocol):
    """Protocol for search routing."""

    def route(self, query: str) -> Any:
        """Route a search query to appropriate handler.

        Returns:
            SearchRoute with type and parameters
        """
        ...


@runtime_checkable
class IntentClassifierProtocol(Protocol):
    """Protocol for intent classification service.

    Classifies user intents using ML models.
    """

    def classify(self, text: str) -> Any:
        """Classify user intent.

        Args:
            text: User input text

        Returns:
            Classified intent (IntentType or similar)
        """
        ...

    def get_confidence(self, text: str, intent: Any) -> float:
        """Get confidence score for a specific intent.

        Args:
            text: User input text
            intent: Intent to check

        Returns:
            Confidence score (0-1)
        """
        ...


@runtime_checkable
class TaskTypeHinterProtocol(Protocol):
    """Protocol for task type hint retrieval.

    Provides task-specific guidance for the LLM.
    """

    def get_hint(self, task_type: str) -> str:
        """Get prompt hint for a specific task type.

        Args:
            task_type: Type of task (edit, search, explain, etc.)

        Returns:
            Formatted hint string for system prompt
        """
        ...


class IToolCallClassifier(Protocol):
    """Protocol for classifying tool calls.

    Defines interface for classifying tools by operation type.
    Supports Open/Closed Principle via strategy pattern.
    """

    def is_write_operation(self, tool_name: str) -> bool:
        """Check if tool is a write operation.

        Args:
            tool_name: Name of the tool

        Returns:
            True if write operation, False otherwise
        """
        ...

    def classify_operation(self, tool_name: str) -> str:
        """Classify tool operation type.

        Args:
            tool_name: Name of the tool

        Returns:
            Operation type category
        """
        ...

    def add_write_tool(self, tool_name: str) -> None:
        """Add a tool to the write operation classification.

        Args:
            tool_name: Name of the tool to add
        """
        ...

