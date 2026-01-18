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

"""Response Coordinator - Coordinates response processing and sanitization.

This module extracts response processing logic from AgentOrchestrator:
- Response content sanitization
- Tool call parsing and validation
- Streaming chunk aggregation
- Garbage content detection
- Response formatting

Design Principles:
- Single Responsibility: Coordinates response processing only
- Composable: Works with existing ResponseSanitizer
- Observable: Provides metrics on response processing
- Testable: Pure functions for content processing

Usage:
    coordinator = ResponseCoordinator(
        sanitizer=ResponseSanitizer(),
        tool_adapter=tool_calling_adapter,
    )

    # Sanitize response
    cleaned = coordinator.sanitize_response(raw_content)

    # Parse tool calls
    tool_calls = coordinator.parse_tool_calls(content, native_tool_calls)

    # Validate tool name
    is_valid = coordinator.is_valid_tool_name(tool_name)
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TYPE_CHECKING,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.tool_calling.base import BaseToolCallingAdapter
    from victor.agent.tool_calling.base import ToolCallParseResult
    from victor.agent.response_sanitizer import ResponseSanitizer
    from victor.tools.registry import ToolRegistry

from victor.agent.coordinators.base_config import BaseCoordinatorConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessedResponse:
    """Result of processing a provider response.

    Attributes:
        content: The sanitized response content
        tool_calls: Validated and normalized tool calls (if any)
        tokens_used: Estimated token count for the response
        garbage_detected: Whether garbage content was detected
        is_final: Whether this is the final response chunk
    """

    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tokens_used: float = 0.0
    garbage_detected: bool = False
    is_final: bool = False

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)

    def is_empty(self) -> bool:
        """Check if response is empty (no content or tool calls)."""
        return not self.content and not self.tool_calls


@dataclass
class ChunkProcessResult:
    """Result of processing a streaming chunk.

    Attributes:
        chunk: The processed chunk (or None if filtered)
        consecutive_garbage_count: Updated count of consecutive garbage chunks
        garbage_detected: Whether garbage was detected
        should_stop: Whether streaming should stop
    """

    chunk: Optional[Any]
    consecutive_garbage_count: int
    garbage_detected: bool
    should_stop: bool = False


@dataclass
class ToolCallValidationResult:
    """Result of validating tool calls.

    Attributes:
        tool_calls: Validated tool calls (filtered and normalized)
        remaining_content: Content with tool calls extracted
        filtered_count: Number of invalid tool calls filtered out
    """

    tool_calls: Optional[List[Dict[str, Any]]]
    remaining_content: str
    filtered_count: int = 0


@runtime_checkable
class IResponseCoordinator(Protocol):
    """Protocol for response processing coordination.

    Defines the interface for processing LLM responses including:
    - Content sanitization
    - Tool call parsing and validation
    - Garbage detection
    - Response formatting
    """

    def sanitize_response(self, content: str) -> str:
        """Sanitize response content.

        Args:
            content: Raw response content

        Returns:
            Sanitized content
        """
        ...

    def is_garbage_content(self, content: str) -> bool:
        """Check if content is garbage.

        Args:
            content: Content to check

        Returns:
            True if content appears to be garbage
        """
        ...

    def is_valid_tool_name(self, name: str) -> bool:
        """Validate a tool name.

        Args:
            name: Tool name to validate

        Returns:
            True if tool name is valid
        """
        ...

    def parse_and_validate_tool_calls(
        self,
        tool_calls: Optional[List[Dict[str, Any]]],
        content: str,
        enabled_tools: Optional[set[str]] = None,
    ) -> ToolCallValidationResult:
        """Parse and validate tool calls from response.

        Args:
            tool_calls: Native tool calls from provider
            content: Response content for fallback parsing
            enabled_tools: Set of enabled tool names for validation

        Returns:
            ToolCallValidationResult with validated calls and remaining content
        """
        ...

    def process_stream_chunk(
        self,
        chunk: Any,
        consecutive_garbage_count: int,
        max_garbage_chunks: int = 3,
        garbage_detected: bool = False,
    ) -> ChunkProcessResult:
        """Process a streaming chunk with garbage detection.

        Args:
            chunk: The stream chunk to process
            consecutive_garbage_count: Current count of consecutive garbage chunks
            max_garbage_chunks: Max consecutive garbage chunks before stopping
            garbage_detected: Whether garbage has been detected

        Returns:
            ChunkProcessResult with processed chunk and updated state
        """
        ...

    def try_extract_tool_calls_from_text(
        self, content: str, valid_tool_names: set[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Try to extract tool calls from text response.

        Args:
            content: Text content to parse
            valid_tool_names: Set of valid tool names

        Returns:
            List of tool call dicts if extraction succeeded, None otherwise
        """
        ...

    def normalize_tool_call_arguments(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize tool call arguments to dicts.

        Args:
            tool_calls: Tool calls with potentially string arguments

        Returns:
            Tool calls with dict arguments
        """
        ...

    def aggregate_chunks(self, chunks: List[str]) -> str:
        """Aggregate response chunks into complete content.

        Args:
            chunks: List of content chunks

        Returns:
            Aggregated content
        """
        ...


@dataclass
class ResponseCoordinatorConfig(BaseCoordinatorConfig):
    """Configuration for ResponseCoordinator.

    Inherits common configuration from BaseCoordinatorConfig:
        enabled: Whether the coordinator is enabled
        timeout: Default timeout in seconds for operations
        max_retries: Maximum number of retry attempts for failed operations
        retry_enabled: Whether retry logic is enabled
        log_level: Logging level for coordinator messages
        enable_metrics: Whether to collect metrics

    Attributes:
        max_garbage_chunks: Maximum consecutive garbage chunks before stopping
        enable_tool_call_extraction: Enable tool call extraction from text
        enable_content_sanitization: Enable content sanitization
        min_content_length: Minimum content length to consider valid
    """

    max_garbage_chunks: int = 3
    enable_tool_call_extraction: bool = True
    enable_content_sanitization: bool = True
    min_content_length: int = 20


class ResponseCoordinator:
    """Coordinates response processing and sanitization.

    This coordinator consolidates response processing logic that was
    spread across the orchestrator, providing a unified interface for:

    1. Content Sanitization: Clean malformed content from local models
    2. Tool Call Parsing: Extract and validate tool calls from responses
    3. Garbage Detection: Detect and filter garbage content
    4. Chunk Aggregation: Combine streaming chunks
    5. Response Formatting: Format responses for display

    Example:
        coordinator = ResponseCoordinator(
            sanitizer=ResponseSanitizer(),
            tool_adapter=tool_calling_adapter,
            tool_registry=registry,
        )

        # In orchestrator streaming loop:
        result = coordinator.process_stream_chunk(
            chunk, consecutive_garbage, max_garbage
        )
        if result.should_stop:
            break

        # After streaming:
        validation = coordinator.parse_and_validate_tool_calls(
            tool_calls, full_content, enabled_tools
        )
    """

    def __init__(
        self,
        sanitizer: "ResponseSanitizer",
        tool_adapter: Optional["BaseToolCallingAdapter"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        config: Optional[ResponseCoordinatorConfig] = None,
        shell_variant_resolver: Optional["ShellVariantResolverProtocol"] = None,
        tool_enabled_checker: Optional["ToolEnabledCheckerProtocol"] = None,
    ):
        """Initialize the ResponseCoordinator.

        Args:
            sanitizer: Response sanitizer for content cleaning
            tool_adapter: Optional tool calling adapter for parsing
            tool_registry: Optional tool registry for validation
            config: Configuration options
            shell_variant_resolver: Optional resolver for shell tool variants
            tool_enabled_checker: Optional checker for tool enabled status
        """
        self._sanitizer = sanitizer
        self._tool_adapter = tool_adapter
        self._tool_registry = tool_registry
        self._config = config or ResponseCoordinatorConfig()
        self._shell_variant_resolver = shell_variant_resolver
        self._tool_enabled_checker = tool_enabled_checker

        logger.debug(
            f"ResponseCoordinator initialized with "
            f"sanitization={'enabled' if self._config.enable_content_sanitization else 'disabled'}, "
            f"max_garbage_chunks={self._config.max_garbage_chunks}"
        )

    # =====================================================================
    # Content Sanitization
    # =====================================================================

    def sanitize_response(self, content: str) -> str:
        """Sanitize response content.

        Delegates to the configured sanitizer if content sanitization is enabled.

        Args:
            content: Raw response content

        Returns:
            Sanitized content suitable for display
        """
        if not content:
            return content

        if not self._config.enable_content_sanitization:
            return content

        return self._sanitizer.sanitize(content)

    def is_garbage_content(self, content: str) -> bool:
        """Check if content is garbage.

        Delegates to the configured sanitizer.

        Args:
            content: Content to check

        Returns:
            True if content appears to be garbage
        """
        if not content:
            return False

        return self._sanitizer.is_garbage_content(content)

    # =====================================================================
    # Tool Call Validation
    # =====================================================================

    def is_valid_tool_name(self, name: str) -> bool:
        """Validate a tool name.

        Delegates to the configured sanitizer.

        Args:
            name: Tool name to validate

        Returns:
            True if tool name is valid
        """
        return self._sanitizer.is_valid_tool_name(name)

    def parse_and_validate_tool_calls(
        self,
        tool_calls: Optional[List[Dict[str, Any]]],
        content: str,
        enabled_tools: Optional[set[str]] = None,
    ) -> ToolCallValidationResult:
        """Parse and validate tool calls from response.

        This method:
        1. Attempts fallback parsing from content if no native tool calls
        2. Normalizes tool calls to dicts
        3. Filters out disabled/invalid tool names
        4. Coerces arguments to dicts

        Args:
            tool_calls: Native tool calls from provider (may be None)
            content: Full response content for fallback parsing
            enabled_tools: Set of enabled tool names for validation

        Returns:
            ToolCallValidationResult with validated calls and remaining content
        """
        filtered_count = 0
        remaining_content = content

        # Step 1: Try fallback parsing if no native tool calls
        if not tool_calls and content:
            logger.debug(
                f"No native tool_calls, attempting fallback parsing on content len={len(content)}"
            )
            parse_result = self._parse_tool_calls_with_adapter(content, tool_calls)
            if parse_result.tool_calls:
                tool_calls = [tc.to_dict() for tc in parse_result.tool_calls]
                logger.debug(
                    f"Fallback parser found {len(tool_calls)} tool calls: "
                    f"{[tc.get('name') for tc in tool_calls]}"
                )
                remaining_content = parse_result.remaining_content
            else:
                logger.debug("Fallback parser found no tool calls")

        # Step 2: Normalize to list of dicts
        if tool_calls:
            normalized_tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
            if len(normalized_tool_calls) != len(tool_calls):
                logger.warning(f"Dropped non-dict tool_calls: {tool_calls}")
                filtered_count += len(tool_calls) - len(normalized_tool_calls)
            tool_calls = normalized_tool_calls or None
            logger.debug(f"After normalization: {len(tool_calls) if tool_calls else 0} tool_calls")

        # Step 3: Filter by enabled tools
        if tool_calls and enabled_tools is not None:
            valid_tool_calls = []
            for tc in tool_calls:
                name = tc.get("name", "")
                if name in enabled_tools:
                    valid_tool_calls.append(tc)
                else:
                    logger.debug(f"Filtered out disabled tool: {name}")
                    filtered_count += 1

            if len(valid_tool_calls) != len(tool_calls):
                logger.warning(
                    f"Filtered {len(tool_calls) - len(valid_tool_calls)} invalid tool calls"
                )
            tool_calls = valid_tool_calls or None
            logger.debug(
                f"After filtering: {len(tool_calls) if tool_calls else 0} valid tool calls"
            )

        # Step 4: Normalize arguments to dicts
        if tool_calls:
            tool_calls = self.normalize_tool_call_arguments(tool_calls)

        return ToolCallValidationResult(
            tool_calls=tool_calls,
            remaining_content=remaining_content,
            filtered_count=filtered_count,
        )

    def _parse_tool_calls_with_adapter(
        self, content: str, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> "ToolCallParseResult":
        """Parse tool calls using the configured adapter.

        Args:
            content: Content to parse
            tool_calls: Existing tool calls

        Returns:
            ToolCallParseResult with extracted tool calls
        """
        if self._tool_adapter:
            return self._tool_adapter.parse_tool_calls(content, tool_calls)

        # Fallback: return empty result
        from victor.agent.tool_calling.base import ToolCallParseResult

        return ToolCallParseResult(tool_calls=[], remaining_content=content, parse_method="none")

    def normalize_tool_call_arguments(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize tool call arguments to dicts.

        Some providers send arguments as JSON strings. This method coerces
        them to dicts for consistent handling.

        Args:
            tool_calls: Tool calls with potentially string arguments

        Returns:
            Tool calls with dict arguments
        """
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

    # =====================================================================
    # Streaming Chunk Processing
    # =====================================================================

    def process_stream_chunk(
        self,
        chunk: Any,
        consecutive_garbage_count: int,
        max_garbage_chunks: int = 3,
        garbage_detected: bool = False,
    ) -> ChunkProcessResult:
        """Process a streaming chunk with garbage detection.

        Checks if the chunk contains garbage content and tracks consecutive
        garbage chunks. Returns a result indicating whether to stop streaming.

        Args:
            chunk: The stream chunk to process
            consecutive_garbage_count: Current count of consecutive garbage chunks
            max_garbage_chunks: Max consecutive garbage chunks before stopping
            garbage_detected: Whether garbage has been detected

        Returns:
            ChunkProcessResult with processed chunk and updated state
        """
        if hasattr(chunk, "content") and chunk.content:
            if self.is_garbage_content(chunk.content):
                consecutive_garbage_count += 1
                if consecutive_garbage_count >= max_garbage_chunks:
                    if not garbage_detected:
                        garbage_detected = True
                        logger.warning(
                            f"Garbage content detected after {len(chunk.content)} chars - "
                            "stopping stream early"
                        )
                    return ChunkProcessResult(
                        chunk=None,
                        consecutive_garbage_count=consecutive_garbage_count,
                        garbage_detected=garbage_detected,
                        should_stop=True,
                    )
                return ChunkProcessResult(
                    chunk=chunk,
                    consecutive_garbage_count=consecutive_garbage_count,
                    garbage_detected=garbage_detected,
                    should_stop=False,
                )
            else:
                # Reset counter on valid content
                consecutive_garbage_count = 0

        return ChunkProcessResult(
            chunk=chunk,
            consecutive_garbage_count=consecutive_garbage_count,
            garbage_detected=garbage_detected,
            should_stop=False,
        )

    # =====================================================================
    # Tool Call Extraction from Text
    # =====================================================================

    def try_extract_tool_calls_from_text(
        self, content: str, valid_tool_names: set[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Try to extract tool calls from text response.

        Uses the text extractor to find tool calls embedded in text responses.

        Args:
            content: Text content to parse
            valid_tool_names: Set of valid tool names

        Returns:
            List of tool call dicts if extraction succeeded, None otherwise
        """
        if not self._config.enable_tool_call_extraction:
            return None

        try:
            from victor.agent.tool_calling.text_extractor import (
                extract_tool_calls_from_text,
            )

            extraction_result = extract_tool_calls_from_text(
                content, valid_tool_names=valid_tool_names
            )

            if extraction_result.success and extraction_result.tool_calls:
                logger.info(
                    f"Extracted {len(extraction_result.tool_calls)} " f"tool calls from text output"
                )
                tool_calls = [
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "id": f"extracted_{idx}",
                    }
                    for idx, tc in enumerate(extraction_result.tool_calls)
                ]
                return tool_calls
        except Exception as e:
            logger.debug(f"Text extraction failed: {e}")

        return None

    # =====================================================================
    # Chunk Aggregation
    # =====================================================================

    def aggregate_chunks(self, chunks: List[str]) -> str:
        """Aggregate response chunks into complete content.

        Args:
            chunks: List of content chunks

        Returns:
            Aggregated content
        """
        if not chunks:
            return ""

        # Filter out None and empty chunks
        filtered = [c for c in chunks if c]
        return "".join(filtered)

    def aggregate_stream_chunks(self, chunks: List[Any], content_attr: str = "content") -> str:
        """Aggregate stream chunk objects into complete content.

        Args:
            chunks: List of chunk objects
            content_attr: Attribute name for content (default: "content")

        Returns:
            Aggregated content string
        """
        if not chunks:
            return ""

        contents = []
        for chunk in chunks:
            if hasattr(chunk, content_attr):
                content = getattr(chunk, content_attr)
                if content:
                    contents.append(str(content))

        return "".join(contents)

    # =====================================================================
    # Response Processing Utilities
    # =====================================================================

    def is_content_meaningful(self, content: str, min_length: Optional[int] = None) -> bool:
        """Check if content is meaningful (not too short, not just whitespace).

        Args:
            content: Content to check
            min_length: Minimum length threshold (uses config default if None)

        Returns:
            True if content appears meaningful
        """
        if not content:
            return False

        min_len = min_length or self._config.min_content_length
        stripped = content.strip()

        return len(stripped) >= min_len

    def extract_remaining_content(
        self, full_content: str, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Extract remaining content after removing tool call artifacts.

        Args:
            full_content: Full response content
            tool_calls: Tool calls that were extracted

        Returns:
            Content with tool call artifacts removed
        """
        if not full_content:
            return ""

        content = full_content

        # Remove JSON-like tool call attempts
        if tool_calls:
            content = re.sub(
                r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\}',
                "",
                content,
            )

        return content.strip()

    def format_response_for_display(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        show_tool_calls: bool = False,
    ) -> str:
        """Format response for display to user.

        Args:
            content: Response content
            tool_calls: Optional tool calls to include
            show_tool_calls: Whether to show tool call information

        Returns:
            Formatted response string
        """
        result = content

        if show_tool_calls and tool_calls:
            tool_names = [tc.get("name", "unknown") for tc in tool_calls]
            result = f"{content}\n\n[Tool calls: {', '.join(tool_names)}]"

        return result

    # =====================================================================
    # Extended Tool Call Processing (extracted from AgentOrchestrator)
    # =====================================================================

    def parse_and_validate_tool_calls_with_resolution(
        self,
        tool_calls: Optional[List[Dict[str, Any]]],
        full_content: str,
        enabled_tools: Optional[set[str]] = None,
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        """Parse, validate, and normalize tool calls with shell variant resolution.

        This is the enhanced version that handles:
        1. Fallback parsing from content if no native tool calls
        2. Normalization to ensure tool_calls are dicts
        3. Shell variant resolution (run/bash -> shell_readonly in INITIAL)
        4. Filtering out disabled/invalid tool names
        5. Coercing arguments to dicts (some providers send JSON strings)

        This method extracts the full logic from AgentOrchestrator._parse_and_validate_tool_calls
        to reduce orchestrator line count and improve testability.

        Args:
            tool_calls: Native tool calls from provider (may be None)
            full_content: Full response content for fallback parsing
            enabled_tools: Optional set of enabled tool names for validation

        Returns:
            Tuple of (validated_tool_calls, remaining_content)
            - validated_tool_calls: List of valid tool call dicts, or None
            - remaining_content: Content after extracting any embedded tool calls
        """
        # Step 1: Try fallback parsing if no native tool calls
        if not tool_calls and full_content:
            logger.debug(
                f"No native tool_calls, attempting fallback parsing on content len={len(full_content)}"
            )
            parse_result = self._parse_tool_calls_with_adapter(full_content, tool_calls)
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

        # Step 2: Normalize to list of dicts
        if tool_calls:
            normalized_tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
            if len(normalized_tool_calls) != len(tool_calls):
                logger.warning(f"Dropped non-dict tool_calls: {tool_calls}")
            tool_calls = normalized_tool_calls or None
            logger.debug(f"After normalization: {len(tool_calls) if tool_calls else 0} tool_calls")

        # Step 3: Filter out invalid/hallucinated tool names with shell variant resolution
        if tool_calls:
            valid_tool_calls = []
            for tc in tool_calls:
                name = tc.get("name", "")

                # Resolve shell aliases to appropriate enabled variant
                resolved_name = self._resolve_shell_variant_internal(name)
                if resolved_name != name:
                    tc["name"] = resolved_name
                    name = resolved_name

                # Check if tool is enabled
                is_enabled = self._is_tool_enabled_internal(name, enabled_tools)
                logger.debug(f"Tool '{name}' enabled={is_enabled}")
                if is_enabled:
                    valid_tool_calls.append(tc)
                else:
                    logger.debug(f"Filtered out disabled tool: {name}")

            if len(valid_tool_calls) != len(tool_calls):
                logger.warning(
                    f"Filtered {len(tool_calls) - len(valid_tool_calls)} invalid tool calls"
                )
            tool_calls = valid_tool_calls or None
            logger.debug(
                f"After filtering: {len(tool_calls) if tool_calls else 0} valid tool calls"
            )

        # Step 4: Coerce arguments to dicts
        if tool_calls:
            tool_calls = self.normalize_tool_call_arguments(tool_calls)

        return tool_calls, full_content

    def _resolve_shell_variant_internal(self, tool_name: str) -> str:
        """Resolve shell aliases to the appropriate enabled shell variant.

        Internal method that uses the injected resolver if available.

        Args:
            tool_name: Original tool name (may be alias like 'run')

        Returns:
            The appropriate enabled shell tool name, or original if not a shell alias
        """
        # Shell-related aliases that should resolve intelligently
        shell_aliases = {"run", "bash", "execute", "cmd", "execute_bash", "shell_readonly", "shell"}

        if tool_name not in shell_aliases:
            return tool_name

        if self._shell_variant_resolver:
            return self._shell_variant_resolver.resolve_shell_variant(tool_name)

        return tool_name

    def _is_tool_enabled_internal(
        self, name: str, enabled_tools: Optional[set[str]] = None
    ) -> bool:
        """Check if a tool is enabled using injected checker or fallback.

        Args:
            name: Tool name to check
            enabled_tools: Optional set of enabled tool names

        Returns:
            True if tool is enabled
        """
        if enabled_tools is not None:
            return name in enabled_tools

        if self._tool_enabled_checker:
            return self._tool_enabled_checker.is_tool_enabled(name)

        # Fallback: try to check via registry
        if self._tool_registry:
            try:
                tool = self._tool_registry.get_tool(name)
                return tool is not None
            except Exception:
                pass

        # Default to True if we can't determine
        return True


# Import for regex in extract_remaining_content
import re


@runtime_checkable
class ShellVariantResolverProtocol(Protocol):
    """Protocol for resolving shell tool variants based on mode.

    This allows the ResponseCoordinator to resolve shell aliases like 'run',
    'bash', 'execute' to the appropriate enabled variant ('shell' or 'shell_readonly')
    based on the current mode.
    """

    def resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell alias to appropriate enabled variant.

        Args:
            tool_name: The shell-related tool name being resolved

        Returns:
            The appropriate shell variant based on mode and tool availability
        """
        ...


@runtime_checkable
class ToolEnabledCheckerProtocol(Protocol):
    """Protocol for checking if a tool is enabled."""

    def is_tool_enabled(self, name: str) -> bool:
        """Check if a tool is enabled.

        Args:
            name: Tool name to check

        Returns:
            True if tool is enabled
        """
        ...
