"""Base protocol and data structures for tool output formatters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class FormattedOutput:
    """Result of formatting tool output for display.

    Attributes:
        content: Formatted content string (may contain Rich markup)
        format_type: Type of format (rich, plain, json, etc.)
        summary: Optional summary for headers/LLM context
        metadata: Tool-specific metadata (file counts, test stats, etc.)
        line_count: Total number of lines for preview sizing (auto-calculated if not provided)
        contains_markup: True if content has Rich markup tags ([green], etc.)
    """

    content: str = field(default_factory=str)
    format_type: str = "rich"
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    line_count: int = field(default=0)
    contains_markup: bool = False

    def __post_init__(self):
        """Auto-calculate line_count from content if not provided."""
        if self.line_count == 0 and self.content:
            self.line_count = len(self.content.splitlines())


class ToolFormatter(ABC):
    """Base protocol for tool output formatters.

    All formatters must inherit from this protocol and implement the format() method.
    Formatters should be stateless and thread-safe.

    Example:
        class TestResultsFormatter(ToolFormatter):
            def format(self, data: Dict, **kwargs) -> FormattedOutput:
                summary = data.get("summary", {})
                lines = [f"[green]✓ {summary['passed']} passed[/]"]
                content = "\\n".join(lines)
                return FormattedOutput(
                    content=content,
                    format_type="rich",
                    summary=f"{summary['total']} tests",
                    contains_markup=True,
                )
    """

    @abstractmethod
    def format(self, data: Dict[str, Any], **kwargs) -> FormattedOutput:
        """Format tool data into Rich markup for display.

        Args:
            data: Tool output data (test results, search results, git output, etc.)
            **kwargs: Formatter-specific options (max_failures, max_files, etc.)

        Returns:
            FormattedOutput with Rich markup content
        """
        pass

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data has required fields.

        Args:
            data: Tool output data to validate

        Returns:
            True if data is valid, False otherwise
        """
        return True  # Default: accept all data

    def get_fallback(self) -> Optional['ToolFormatter']:
        """Return fallback formatter if this one fails.

        Returns:
            Another formatter instance, or None if no fallback
        """
        return None  # Default: no fallback
