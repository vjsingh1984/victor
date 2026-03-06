"""Tool call validator extracted from ToolCoordinator.

Validates tool calls before execution: argument types, required params,
and safety checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of tool call validation."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    normalized_args: Optional[dict] = None


@dataclass
class SafetyResult:
    """Result of safety check on a tool call."""

    safe: bool = True
    reason: str = ""
    severity: str = "info"


class ToolCallValidator:
    """Validates tool calls before execution.

    Extracted from ToolCoordinator for cleaner separation of concerns.
    """

    def __init__(self, tool_schemas: Optional[dict[str, dict]] = None):
        self._tool_schemas = tool_schemas or {}

    def validate(self, tool_name: str, args: dict[str, Any]) -> ValidationResult:
        """Validate a tool call against its schema.

        Args:
            tool_name: Name of the tool to call.
            args: Arguments to pass.

        Returns:
            ValidationResult with errors if invalid.
        """
        result = ValidationResult()

        schema = self._tool_schemas.get(tool_name)
        if not schema:
            # No schema = allow (tool might be dynamically registered)
            return result

        # Check required parameters
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for param in required:
            if param not in args:
                result.valid = False
                result.errors.append(f"Missing required parameter: {param}")

        # Type check known parameters
        for param, value in args.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type and not self._type_matches(value, expected_type):
                    result.warnings.append(
                        f"Parameter '{param}' expected type '{expected_type}', "
                        f"got {type(value).__name__}"
                    )

        result.normalized_args = args
        return result

    def check_safety(self, tool_name: str, args: dict[str, Any]) -> SafetyResult:
        """Check if a tool call is safe to execute.

        Args:
            tool_name: Name of the tool.
            args: Arguments to the tool.

        Returns:
            SafetyResult indicating whether the call is safe.
        """
        # Check for dangerous patterns in string arguments
        for key, value in args.items():
            if isinstance(value, str):
                if "rm -rf /" in value:
                    return SafetyResult(
                        safe=False,
                        reason=f"Dangerous command in '{key}'",
                        severity="critical",
                    )
                if "DROP TABLE" in value.upper():
                    return SafetyResult(
                        safe=False,
                        reason=f"SQL injection pattern in '{key}'",
                        severity="critical",
                    )

        return SafetyResult()

    def register_schema(self, tool_name: str, schema: dict) -> None:
        """Register a tool schema for validation."""
        self._tool_schemas[tool_name] = schema

    @staticmethod
    def _type_matches(value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected JSON Schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)
