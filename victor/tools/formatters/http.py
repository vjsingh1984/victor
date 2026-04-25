"""HTTP response formatter with Rich markup."""

import json
from typing import Dict, Any

from .base import ToolFormatter, FormattedOutput


class HTTPFormatter(ToolFormatter):
    """Format HTTP response with Rich markup.

    Produces color-coded output for:
    - Status codes (green=2xx, yellow=3xx, red=4xx/5xx)
    - Response headers (cyan keys, dim values)
    - Response body (truncated if large)
    - Duration metrics
    """

    def validate_input(self, data: Dict) -> bool:
        """Validate HTTP response has required fields."""
        return isinstance(data, dict) and ("status_code" in data or "body" in data)

    def format(
        self,
        data: Dict[str, Any],
        max_headers: int = 10,
        max_body_length: int = 500,
        **kwargs
    ) -> FormattedOutput:
        """Format HTTP response with Rich markup.

        Args:
            data: HTTP response dict with status_code, headers, body, etc.
            max_headers: Maximum headers to display (default: 10)
            max_body_length: Maximum body length to display (default: 500 chars)

        Returns:
            FormattedOutput with Rich markup
        """
        status_code = data.get("status_code", 0)
        status = data.get("status", "")
        duration_ms = data.get("duration_ms", 0)

        lines = []

        # Color-code status code
        if 200 <= status_code < 300:
            color = "green"
        elif 300 <= status_code < 400:
            color = "yellow"
        elif 400 <= status_code < 600:
            color = "red"
        else:
            color = "white"

        # Status line
        if status_code:
            lines.append(f"[{color} bold]{status_code} {status}[/] [dim]• {duration_ms}ms[/]")
            lines.append("")  # Blank line

        # Headers
        headers = data.get("headers", {})
        if headers:
            lines.append("[dim]Headers:[/]")
            for i, (key, value) in enumerate(list(headers.items())[:max_headers]):
                lines.append(f"  [cyan]{key}:[/] [dim]{value}[/]")

            if len(headers) > max_headers:
                lines.append(f"  [dim]... and {len(headers) - max_headers} more headers[/]")

            lines.append("")  # Blank line

        # Body
        body = data.get("body")
        if body:
            lines.append("[dim]Body:[/]")

            if isinstance(body, dict):
                # Format JSON with syntax highlighting
                body_str = json.dumps(body, indent=2)
                if len(body_str) > max_body_length:
                    body_str = body_str[:max_body_length] + "..."
                lines.append(f"  [yellow]{body_str}[/]")
            elif isinstance(body, list):
                # Format list as JSON
                body_str = json.dumps(body, indent=2)
                if len(body_str) > max_body_length:
                    body_str = body_str[:max_body_length] + "..."
                lines.append(f"  [yellow]{body_str}[/]")
            else:
                # String body
                body_str = str(body)
                if len(body_str) > max_body_length:
                    body_str = body_str[:max_body_length] + "..."
                lines.append(f"  {body_str}")

        content = "\n".join(lines)

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=f"{status_code} {status}",
            line_count=len(lines),
            contains_markup=True,
        )
