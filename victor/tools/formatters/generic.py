"""Generic fallback formatter for tools without specific formatters."""

from typing import Dict, Any

from .base import ToolFormatter, FormattedOutput


class GenericFormatter(ToolFormatter):
    """Generic fallback formatter that converts data to plain text.

    This formatter is used when no specific formatter is registered
    for a tool, or when a specific formatter fails. It provides
    basic string conversion without Rich markup.
    """

    def format(self, data: Dict[str, Any], **kwargs) -> FormattedOutput:
        """Format tool data as plain text.

        Args:
            data: Tool output data (any format)
            **kwargs: Ignored (no options for generic formatter)

        Returns:
            FormattedOutput with plain text content
        """
        # Convert data to string representation
        if isinstance(data, dict):
            # For dictionaries, try to format nicely
            lines = []
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    # Skip complex nested structures
                    continue
                lines.append(f"{key}: {value}")
            content = "\n".join(lines) if lines else str(data)
        elif isinstance(data, (list, tuple)):
            # For lists/tuples, join items
            content = "\n".join(str(item) for item in data)
        else:
            # For everything else, just convert to string
            content = str(data)

        return FormattedOutput(
            content=content,
            format_type="plain",
            summary=None,
            line_count=len(content.splitlines()) if content else 0,
            contains_markup=False,
        )
