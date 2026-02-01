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

"""ResponseProcessor - Extracted from AgentOrchestrator.

This module handles tool call parsing, validation, and response formatting,
extracted from the AgentOrchestrator to reduce class size.

The processor handles:
- Tool call parsing from native and fallback formats
- Tool call validation and normalization
- Tool name resolution (aliases, shell variants)
- Argument coercion
- Tool output formatting

Example:
    processor = ResponseProcessor(
        tool_adapter=tool_adapter,
        tool_registry=tool_registry,
        sanitizer=sanitizer,
    )

    tool_calls, content = processor.parse_and_validate(
        raw_tool_calls, response_content
    )

    output = processor.format_tool_output("read_file", {"path": "foo.py"}, content)
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.tool_calling import BaseToolCallingAdapter, ToolCallParseResult
    from victor.agent.tool_output_formatter import ToolOutputFormatter
    from victor.tools import ToolRegistry

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolNameValidator(Protocol):
    """Protocol for tool name validation."""

    def is_valid_tool_name(self, name: str) -> bool:
        """Check if a tool name is valid and not a hallucination."""
        ...

    def strip_markup(self, text: str) -> str:
        """Remove simple XML/HTML-like tags to salvage plain text."""
        ...

    def sanitize(self, text: str) -> str:
        """Sanitize model response by removing malformed patterns."""
        ...


@runtime_checkable
class ShellVariantResolver(Protocol):
    """Protocol for resolving shell tool variants."""

    def resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell aliases to the appropriate enabled shell variant."""
        ...


@dataclass
class ParsedToolCall:
    """A parsed and validated tool call."""

    name: str
    arguments: dict[str, Any]
    original_name: str = ""
    was_resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class ParseResult:
    """Result of parsing and validating tool calls."""

    tool_calls: list[dict[str, Any]]
    remaining_content: str
    warnings: list[str] = field(default_factory=list)
    filtered_count: int = 0
    parse_method: str = "native"


class ResponseProcessor:
    """Handles tool call parsing, validation, and response formatting.

    Extracted from AgentOrchestrator to reduce class size while
    maintaining the same functionality. The orchestrator delegates
    to this processor for all response processing.

    Attributes:
        tool_adapter: Adapter for tool call parsing
        tool_registry: Registry for checking enabled tools
        sanitizer: Validator for tool names and content
        shell_resolver: Resolver for shell tool variants
        output_formatter: Formatter for tool output
    """

    def __init__(
        self,
        tool_adapter: BaseToolCallingAdapter,
        tool_registry: ToolRegistry,
        sanitizer: ToolNameValidator,
        shell_resolver: Optional[ShellVariantResolver] = None,
        output_formatter: Optional[ToolOutputFormatter] = None,
    ):
        """Initialize the response processor.

        Args:
            tool_adapter: Adapter for tool call parsing
            tool_registry: Registry for checking enabled tools
            sanitizer: Validator for tool names and content
            shell_resolver: Optional resolver for shell variants
            output_formatter: Optional formatter for tool output
        """
        self._tool_adapter = tool_adapter
        self._tool_registry = tool_registry
        self._sanitizer = sanitizer
        self._shell_resolver = shell_resolver
        self._output_formatter = output_formatter

    def parse_tool_calls_with_adapter(
        self, content: str, raw_tool_calls: Optional[list[dict[str, Any]]] = None
    ) -> ToolCallParseResult:
        """Parse tool calls using the tool calling adapter.

        This is the unified method for parsing tool calls that handles:
        1. Native tool calls from provider
        2. JSON fallback parsing
        3. XML fallback parsing
        4. Tool name validation

        Args:
            content: Response content text
            raw_tool_calls: Native tool_calls from provider (if any)

        Returns:
            ToolCallParseResult with parsed tool calls and metadata
        """
        result = self._tool_adapter.parse_tool_calls(content, raw_tool_calls)

        # Log any warnings
        for warning in result.warnings:
            logger.warning(f"Tool call parse warning: {warning}")

        # Log parse method for debugging
        if result.tool_calls:
            logger.debug(
                f"Parsed {len(result.tool_calls)} tool calls via {result.parse_method} "
                f"(confidence={result.confidence})"
            )

        return result

    def parse_and_validate(
        self,
        tool_calls: Optional[list[dict[str, Any]]],
        full_content: str,
    ) -> tuple[Optional[list[dict[str, Any]]], str]:
        """Parse, validate, and normalize tool calls from provider response.

        Handles:
        1. Fallback parsing from content if no native tool calls
        2. Normalization to ensure tool_calls are dicts
        3. Filtering out disabled/invalid tool names
        4. Coercing arguments to dicts (some providers send JSON strings)

        Args:
            tool_calls: Native tool calls from provider (may be None)
            full_content: Full response content for fallback parsing

        Returns:
            Tuple of (validated_tool_calls, remaining_content)
            - validated_tool_calls: List of valid tool call dicts, or None
            - remaining_content: Content after extracting any embedded tool calls
        """
        # Use unified adapter-based tool call parsing with fallbacks
        if not tool_calls and full_content:
            logger.debug(
                f"No native tool_calls, attempting fallback parsing on content len={len(full_content)}"
            )
            parse_result = self.parse_tool_calls_with_adapter(full_content, tool_calls)
            if parse_result.tool_calls:
                # Convert ToolCall objects to dicts for compatibility
                tool_calls = [tc.to_dict() for tc in parse_result.tool_calls]
                logger.debug(
                    f"Fallback parser found {len(tool_calls)} tool calls: "
                    f"{[tc.get('name') for tc in tool_calls]}"
                )
                full_content = parse_result.remaining_content
            else:
                logger.debug("Fallback parser found no tool calls")

        # Normalize to list of dicts
        tool_calls = self._normalize_tool_calls(tool_calls)

        # Filter invalid/disabled tools
        tool_calls = self._filter_invalid_tools(tool_calls)

        # Coerce arguments to dicts
        tool_calls = self._coerce_arguments(tool_calls)

        return tool_calls, full_content

    def _normalize_tool_calls(
        self, tool_calls: Optional[list[Any]]
    ) -> Optional[list[dict[str, Any]]]:
        """Ensure tool_calls is a list of dicts.

        Args:
            tool_calls: Raw tool calls from provider

        Returns:
            Normalized list of dict tool calls, or None
        """
        if not tool_calls:
            return None

        normalized = [tc for tc in tool_calls if isinstance(tc, dict)]
        if len(normalized) != len(tool_calls):
            logger.warning(f"Dropped non-dict tool_calls: {tool_calls}")

        return normalized if normalized else None

    def _filter_invalid_tools(
        self, tool_calls: Optional[list[dict[str, Any]]]
    ) -> Optional[list[dict[str, Any]]]:
        """Filter out invalid/disabled tool names.

        Args:
            tool_calls: Normalized tool calls

        Returns:
            Filtered list with only valid enabled tools
        """
        if not tool_calls:
            return None

        valid_calls = []
        for tc in tool_calls:
            name = tc.get("name", "")

            # Resolve shell aliases if resolver is available
            if self._shell_resolver:
                resolved_name = self._shell_resolver.resolve_shell_variant(name)
                if resolved_name != name:
                    tc["name"] = resolved_name
                    name = resolved_name

            # Check if tool is enabled
            is_enabled = self._tool_registry.is_tool_enabled(name)
            logger.debug(f"Tool '{name}' enabled={is_enabled}")

            if is_enabled:
                valid_calls.append(tc)
            else:
                logger.debug(f"Filtered out disabled tool: {name}")

        if len(valid_calls) != len(tool_calls):
            logger.warning(f"Filtered {len(tool_calls) - len(valid_calls)} invalid tool calls")

        return valid_calls if valid_calls else None

    def _coerce_arguments(
        self, tool_calls: Optional[list[dict[str, Any]]]
    ) -> Optional[list[dict[str, Any]]]:
        """Coerce arguments to dicts (providers may send JSON strings).

        Args:
            tool_calls: Tool calls with potentially string arguments

        Returns:
            Tool calls with dict arguments
        """
        if not tool_calls:
            return None

        for tc in tool_calls:
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    tc["arguments"] = json.loads(args)
                except Exception:
                    try:
                        tc["arguments"] = ast.literal_eval(args)
                    except Exception:
                        tc["arguments"] = {"value": args}
            elif args is None:
                tc["arguments"] = {}

        return tool_calls

    def sanitize_response(self, text: str) -> str:
        """Sanitize model response by removing malformed patterns.

        Args:
            text: Raw response text

        Returns:
            Sanitized text
        """
        return self._sanitizer.sanitize(text)

    def strip_markup(self, text: str) -> str:
        """Remove simple XML/HTML-like tags to salvage plain text.

        Args:
            text: Text potentially containing markup

        Returns:
            Plain text with markup removed
        """
        return self._sanitizer.strip_markup(text)

    def is_valid_tool_name(self, name: str) -> bool:
        """Check if a tool name is valid and not a hallucination.

        Args:
            name: Tool name to validate

        Returns:
            True if valid, False if hallucinated
        """
        return self._sanitizer.is_valid_tool_name(name)

    def format_tool_output(
        self,
        tool_name: str,
        args: dict[str, Any],
        output: Any,
        context: Optional[Any] = None,
    ) -> str:
        """Format tool output with clear boundaries.

        Delegates to ToolOutputFormatter for:
        - Structured output serialization (lists, dicts -> compact formats)
        - Anti-hallucination markers (TOOL_OUTPUT tags)
        - Smart truncation for large outputs
        - File structure extraction for very large files

        Args:
            tool_name: Name of the tool that was executed
            args: Arguments passed to the tool
            output: Raw output from the tool
            context: Optional formatting context

        Returns:
            Formatted string with clear TOOL_OUTPUT boundaries
        """
        if self._output_formatter:
            return self._output_formatter.format_tool_output(
                tool_name=tool_name,
                args=args,
                output=output,
                context=context,
            )

        # Fallback to simple formatting if no formatter
        return f"[TOOL_OUTPUT: {tool_name}]\n{output}\n[/TOOL_OUTPUT]"

    def log_tool_call(self, name: str, kwargs: dict[str, Any]) -> None:
        """Log a tool call for debugging.

        Args:
            name: Tool name
            kwargs: Tool arguments
        """
        # Truncate large args for readability
        args_str = str(kwargs)
        if len(args_str) > 200:
            args_str = args_str[:200] + "...(truncated)"
        logger.debug(f"Tool call: {name}({args_str})")


__all__ = [
    "ResponseProcessor",
    "ParsedToolCall",
    "ParseResult",
    "ToolNameValidator",
    "ShellVariantResolver",
]
