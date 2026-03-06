"""Tool call parser extracted from ToolCoordinator.

Parses raw model output into structured tool calls and normalizes arguments.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedToolCall:
    """A parsed tool call from model output."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    call_id: Optional[str] = None


class ToolCallParser:
    """Parse raw model output into structured tool calls.

    Extracted from ToolCoordinator for independent testing.
    """

    # Common patterns for tool call extraction
    FUNCTION_CALL_PATTERN = re.compile(
        r'(?:```(?:json)?\s*\{[^}]*"name"\s*:\s*"([^"]+)"[^}]*\})',
        re.DOTALL,
    )

    def parse(self, raw_output: str) -> list[ParsedToolCall]:
        """Parse raw model output for tool calls.

        Args:
            raw_output: Raw text output from the model.

        Returns:
            List of ParsedToolCall instances.
        """
        calls: list[ParsedToolCall] = []

        # Try JSON-based extraction first
        try:
            data = json.loads(raw_output)
            if isinstance(data, dict) and "tool_calls" in data:
                for tc in data["tool_calls"]:
                    calls.append(
                        ParsedToolCall(
                            tool_name=tc.get("name", ""),
                            arguments=tc.get("arguments", {}),
                            call_id=tc.get("id"),
                            raw_text=json.dumps(tc),
                        )
                    )
                return calls
        except (json.JSONDecodeError, TypeError):
            pass

        return calls

    def normalize_args(self, tool_name: str, raw_args: dict) -> dict:
        """Normalize tool arguments.

        Handles common issues: string booleans, nested JSON strings,
        missing defaults, etc.

        Args:
            tool_name: The tool being called.
            raw_args: Raw argument dict from parsing.

        Returns:
            Normalized argument dict.
        """
        normalized = {}
        for key, value in raw_args.items():
            # Convert string booleans
            if isinstance(value, str):
                if value.lower() in ("true", "yes", "1"):
                    normalized[key] = True
                    continue
                elif value.lower() in ("false", "no", "0"):
                    normalized[key] = False
                    continue

                # Try to parse nested JSON
                if value.startswith(("{", "[")):
                    try:
                        normalized[key] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass

            normalized[key] = value

        return normalized
