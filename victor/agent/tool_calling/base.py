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

"""
Base classes for tool calling adapters.

Provides a unified interface for handling tool calling across different
LLM providers, abstracting away provider-specific formats and behaviors.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


from victor.providers.base import ToolDefinition


class ToolCallFormat(Enum):
    """Supported tool call formats across providers."""

    OPENAI = "openai"  # OpenAI function calling format
    ANTHROPIC = "anthropic"  # Anthropic tool use format
    GOOGLE = "google"  # Google Gemini function_declarations format
    OLLAMA_NATIVE = "ollama_native"  # Ollama's native tool_calls
    OLLAMA_JSON = "ollama_json"  # JSON in content (fallback)
    LMSTUDIO_NATIVE = "lmstudio_native"  # LMStudio native (hammer badge models)
    LMSTUDIO_DEFAULT = "lmstudio_default"  # LMStudio [TOOL_REQUEST] format
    VLLM = "vllm"  # vLLM with tool parser
    XML = "xml"  # XML-style tool calls in content
    UNKNOWN = "unknown"


@dataclass
class ToolCallingCapabilities:
    """Describes the tool calling capabilities of a provider/model combination.

    This metadata allows the orchestrator to adapt its behavior based on
    what the provider/model actually supports.
    """

    # Core capabilities
    native_tool_calls: bool = False  # Provider returns structured tool_calls
    streaming_tool_calls: bool = False  # Tool calls can be streamed
    parallel_tool_calls: bool = False  # Model can request multiple tools at once
    tool_choice_param: bool = False  # Supports tool_choice parameter

    # Fallback capabilities
    json_fallback_parsing: bool = False  # Can parse JSON tool calls from content
    xml_fallback_parsing: bool = False  # Can parse XML tool calls from content

    # Model-specific features
    thinking_mode: bool = False  # Model has a thinking/reasoning mode
    thinking_disable_prefix: Optional[str] = None  # Prefix to disable thinking (e.g., "/no_think")
    requires_strict_prompting: bool = False  # Needs strict system prompts

    # Format details
    tool_call_format: ToolCallFormat = ToolCallFormat.UNKNOWN
    argument_format: str = "json"  # "json" or "python_dict"

    # Recommended limits
    recommended_max_tools: int = 20  # Max tools to send
    recommended_tool_budget: int = 12  # Max tool calls per turn

    # Model-specific exploration behavior
    # Models that need more "thinking" turns get higher exploration_multiplier
    exploration_multiplier: float = 1.0  # Multiplies max_exploration_iterations
    continuation_patience: int = 3  # Empty turns to allow before forcing completion


@dataclass
class ToolCall:
    """Normalized tool call representation.

    All provider-specific formats are converted to this unified structure.
    """

    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None  # Tool call ID (for multi-turn)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {"name": self.name, "arguments": self.arguments}
        if self.id:
            result["id"] = self.id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments", {}),
            id=data.get("id"),
        )


@dataclass
class ToolCallParseResult:
    """Result of parsing tool calls from a response.

    Contains both the parsed tool calls and metadata about the parsing process.
    """

    tool_calls: List[ToolCall] = field(default_factory=list)
    remaining_content: str = ""  # Content after removing tool calls
    parse_method: str = "none"  # How the tool calls were parsed
    confidence: float = 1.0  # Confidence in the parsing (0-1)
    warnings: List[str] = field(default_factory=list)


class FallbackParsingMixin:
    """Mixin providing common fallback parsing methods for tool calls.

    This mixin extracts reusable parsing logic that was duplicated across
    multiple adapters:
    - JSON parsing from content
    - XML-style parsing from content
    - Native tool call parsing with format detection
    - Argument string-to-dict conversion

    Usage:
        class MyAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
            def parse_tool_calls(self, content, raw_tool_calls):
                # Try native first
                if raw_tool_calls:
                    result = self.parse_native_tool_calls(raw_tool_calls)
                    if result.tool_calls:
                        return result

                # Fall back to content parsing
                return self.parse_from_content(content)
    """

    def parse_json_arguments(self, args: Any) -> Tuple[Dict[str, Any], Optional[str]]:
        """Parse arguments that may be string or dict.

        Args:
            args: Arguments (string JSON or dict)

        Returns:
            Tuple of (parsed_dict, warning_message or None)
        """
        if isinstance(args, dict):
            return args, None

        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    return parsed, None
                return {}, f"Parsed JSON is not a dict: {type(parsed).__name__}"
            except json.JSONDecodeError:
                return {}, "Failed to parse JSON arguments"

        return {}, f"Unexpected argument type: {type(args).__name__}"

    def parse_native_tool_calls(
        self,
        raw_tool_calls: List[Dict[str, Any]],
        validate_name_fn: Optional[Callable[[str], bool]] = None,
    ) -> ToolCallParseResult:
        """Parse native tool calls from provider response.

        Handles multiple formats:
        - OpenAI: {function: {name, arguments}}
        - Anthropic: {name, arguments, id}
        - Ollama: {name, arguments} or {function: {...}}

        Args:
            raw_tool_calls: List of raw tool call dicts from provider
            validate_name_fn: Optional function to validate tool names

        Returns:
            ToolCallParseResult with parsed tool calls
        """
        if not raw_tool_calls:
            return ToolCallParseResult()

        tool_calls: List[ToolCall] = []
        warnings: List[str] = []

        for tc in raw_tool_calls:
            # Handle OpenAI/Ollama format: {function: {name, arguments}}
            if "function" in tc:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                tc_id = tc.get("id")
            else:
                # Direct format: {name, arguments}
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                tc_id = tc.get("id")

            # Validate tool name if validator provided
            if validate_name_fn and not validate_name_fn(name):
                warnings.append(f"Skipped invalid tool name: {name}")
                continue
            elif not name:
                warnings.append("Skipped tool call with empty name")
                continue

            # Parse string arguments
            parsed_args, warning = self.parse_json_arguments(args)
            if warning:
                warnings.append(f"{name}: {warning}")

            tool_calls.append(ToolCall(name=name, arguments=parsed_args, id=tc_id))

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content="",
            parse_method="native",
            confidence=1.0,
            warnings=warnings,
        )

    def parse_json_from_content(
        self,
        content: str,
        validate_name_fn: Optional[Callable[[str], bool]] = None,
    ) -> ToolCallParseResult:
        """Parse JSON tool calls from content (fallback).

        Looks for JSON objects with 'name' and 'arguments' or 'parameters'.

        Args:
            content: Response content to parse
            validate_name_fn: Optional function to validate tool names

        Returns:
            ToolCallParseResult with parsed tool calls
        """
        content = content.strip()
        if not content:
            return ToolCallParseResult()

        try:
            data = json.loads(content)
            if isinstance(data, dict) and "name" in data:
                name = data.get("name", "")

                # Validate name if validator provided
                if validate_name_fn and not validate_name_fn(name):
                    return ToolCallParseResult(remaining_content=content)

                args = data.get("arguments") or data.get("parameters", {})
                parsed_args, _ = self.parse_json_arguments(args)

                return ToolCallParseResult(
                    tool_calls=[ToolCall(name=name, arguments=parsed_args)],
                    remaining_content="",
                    parse_method="json_fallback",
                    confidence=0.9,
                )
        except json.JSONDecodeError:
            pass

        return ToolCallParseResult(remaining_content=content)

    def parse_xml_from_content(
        self,
        content: str,
        validate_name_fn: Optional[Callable[[str], bool]] = None,
    ) -> ToolCallParseResult:
        """Parse XML-style tool calls from content (fallback).

        Supports multiple formats:
        1. <function_call><name>X</name><arguments>{...}</arguments></function_call>
        2. <function=X><parameter=Y>value</parameter>...</function>
        3. <tool_call><name>X</name><arguments>{...}</arguments></tool_call>

        Args:
            content: Response content to parse
            validate_name_fn: Optional function to validate tool names

        Returns:
            ToolCallParseResult with parsed tool calls
        """
        if not content:
            return ToolCallParseResult()

        # Define all patterns
        patterns = [
            # Pattern 1: Standard function_call format
            (
                r"<function_call>\s*<name>([^<]+)</name>\s*<arguments>(.*?)</arguments>\s*</function_call>",
                "standard",
            ),
            # Pattern 2: <tool_call> format
            (
                r"<tool_call>\s*<name>([^<]+)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
                "tool_call",
            ),
        ]

        matches: List[Tuple[str, str]] = []
        matched_patterns: List[str] = []

        for pattern, _pattern_name in patterns:
            found = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if found:
                matches.extend(found)
                matched_patterns.append(pattern)
                break  # Use first matching pattern

        # Pattern 3: <function=name> format (Qwen3, some local models)
        if not matches:
            func_pattern = r"<function=([^>]+)>(.*?)</function>"
            func_matches = re.findall(func_pattern, content, re.DOTALL | re.IGNORECASE)
            if func_matches:
                matched_patterns.append(func_pattern)
                for name, params_content in func_matches:
                    name = name.strip()
                    # Parse <parameter=X>value</parameter> tags
                    param_pattern = r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>"
                    param_matches = re.findall(param_pattern, params_content, re.DOTALL)
                    args = {p.strip(): v.strip() for p, v in param_matches}
                    matches.append((name, json.dumps(args)))

        if not matches:
            return ToolCallParseResult(remaining_content=content)

        tool_calls: List[ToolCall] = []
        warnings: List[str] = []

        for name, args_str in matches:
            name = name.strip()

            # Validate name if validator provided
            if validate_name_fn and not validate_name_fn(name):
                warnings.append(f"Skipped invalid tool name: {name}")
                continue

            parsed_args, warning = self.parse_json_arguments(args_str.strip())
            if warning:
                warnings.append(f"{name}: {warning}")

            tool_calls.append(ToolCall(name=name, arguments=parsed_args))

        # Remove matched content
        remaining = content
        for pattern in matched_patterns:
            remaining = re.sub(pattern, "", remaining, flags=re.DOTALL | re.IGNORECASE)
        remaining = remaining.strip()

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content=remaining,
            parse_method="xml_fallback",
            confidence=0.7,
            warnings=warnings,
        )

    def parse_from_content(
        self,
        content: str,
        validate_name_fn: Optional[Callable[[str], bool]] = None,
    ) -> ToolCallParseResult:
        """Parse tool calls from content using all fallback methods.

        Tries JSON first, then XML parsing.

        Args:
            content: Response content to parse
            validate_name_fn: Optional function to validate tool names

        Returns:
            ToolCallParseResult with parsed tool calls
        """
        # Try JSON first (higher confidence)
        result = self.parse_json_from_content(content, validate_name_fn)
        if result.tool_calls:
            return result

        # Try XML parsing
        result = self.parse_xml_from_content(content, validate_name_fn)
        if result.tool_calls:
            return result

        return ToolCallParseResult(remaining_content=content)


class BaseToolCallingAdapter(ABC):
    """Abstract base class for provider-specific tool calling adapters.

    Each adapter handles:
    1. Converting tools to provider format
    2. Parsing tool calls from responses
    3. Providing capability metadata
    4. Generating system prompt hints
    5. Normalizing arguments

    Subclasses implement provider-specific logic while the orchestrator
    uses the unified interface.
    """

    def __init__(self, model: str = "", config: Optional[Dict[str, Any]] = None):
        """Initialize adapter.

        Args:
            model: Model name/identifier
            config: Optional configuration overrides
        """
        self.model = model
        self.model_lower = model.lower() if model else ""
        self.config = config or {}

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider name this adapter handles."""
        pass

    @abstractmethod
    def get_capabilities(self) -> ToolCallingCapabilities:
        """Get tool calling capabilities for the current model.

        Returns:
            ToolCallingCapabilities describing what this provider/model supports
        """
        pass

    @abstractmethod
    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert standard tool definitions to provider format.

        Args:
            tools: List of standard ToolDefinition objects

        Returns:
            Provider-formatted tool definitions
        """
        pass

    @abstractmethod
    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse tool calls from response.

        Handles both native tool_calls and fallback parsing from content.

        Args:
            content: Response content text
            raw_tool_calls: Native tool_calls from provider (if any)

        Returns:
            ToolCallParseResult with parsed tool calls and metadata
        """
        pass

    def get_system_prompt_hints(self) -> str:
        """Get system prompt additions for this provider/model.

        Returns hints that should be added to the system prompt to improve
        tool calling behavior for this specific provider/model combination.

        Returns:
            String to append to system prompt (empty if none needed)
        """
        capabilities = self.get_capabilities()
        hints = []

        if capabilities.requires_strict_prompting:
            hints.append("Call tools ONE AT A TIME. Wait for results.")
            hints.append("After 2-3 tool calls, provide your answer.")
            hints.append("Do NOT output JSON, XML, or tool syntax in responses.")
        elif capabilities.native_tool_calls and capabilities.parallel_tool_calls:
            # Encourage parallel tool calls for capable models
            hints.append("Call MULTIPLE tools in parallel when operations are independent.")

        if capabilities.thinking_mode:
            hints.append("Use /no_think for simple questions.")

        return "\n".join(hints)

    # Default values for required parameters that providers may omit.
    # These are sensible defaults that preserve expected tool behavior.
    # Key: tool_name, Value: dict of parameter defaults
    TOOL_ARGUMENT_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "list_directory": {"path": "."},
        "read_file": {"path": ""},  # Empty string will fail gracefully
        "execute_bash": {"command": ""},
        "code_search": {"query": "", "path": "."},
        "semantic_code_search": {"query": "", "path": "."},
    }

    def normalize_arguments(self, arguments: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Normalize tool arguments with sensible defaults.

        Providers may omit required parameters (e.g., Gemini returning empty args
        for list_directory when it means "current directory"). This method fills
        in sensible defaults to prevent tool execution failures.

        Design Principle:
            Tools define their contracts (required params). Adapters bridge the gap
            between what providers return and what tools expect. This keeps tools
            provider-agnostic while handling provider quirks in one place.

        Args:
            arguments: Raw arguments from model
            tool_name: Name of the tool

        Returns:
            Normalized arguments with defaults applied
        """
        # Get defaults for this tool
        defaults = self.TOOL_ARGUMENT_DEFAULTS.get(tool_name, {})
        if not defaults:
            return arguments

        # Apply defaults for missing required parameters
        normalized = dict(arguments)  # Copy to avoid mutation
        for param, default_value in defaults.items():
            if param not in normalized or normalized[param] is None:
                normalized[param] = default_value

        return normalized

    def is_valid_tool_name(self, name: str) -> bool:
        """Check if a tool name is valid.

        Rejects hallucinated/malformed tool names.

        Args:
            name: Tool name to validate

        Returns:
            True if valid, False if should be rejected
        """
        import re

        if not name or not isinstance(name, str):
            return False

        # Reject common hallucinated patterns
        invalid_patterns = [
            r"^example_",
            r"^func_",
            r"^function_",
            r"^tool_name",
            r"^my_",
            r"^test_tool",
            r"^sample_",
            r"/$",  # Ends with slash
            r"^<",  # XML tag start
            r">$",  # XML tag end
            r"\s",  # Contains whitespace
            r"^\d",  # Starts with number
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return False

        # Must be alphanumeric with underscores
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            return False

        return True

    def sanitize_content(self, content: str) -> str:
        """Sanitize response content by removing garbage patterns.

        Args:
            content: Raw response content

        Returns:
            Cleaned content
        """
        import re

        if not content:
            return content

        # Remove repeated closing tags
        content = re.sub(r"(</\w+>\s*){3,}", "", content)

        # Remove orphaned XML tags
        content = re.sub(r"</?function[^>]*>", "", content)
        content = re.sub(r"</?parameter[^>]*>", "", content)
        content = re.sub(r"</?tool[^>]*>", "", content)
        content = re.sub(r"</?IMPORTANT[^>]*>", "", content, flags=re.IGNORECASE)
        content = re.sub(r"</?important[^>]*>", "", content, flags=re.IGNORECASE)

        # Remove instruction leakage
        leakage_patterns = [
            r"^\s*Do NOT.*$",
            r"^\s*NEVER\s+.*$",
            r"^\s*ALWAYS include.*$",
            r"^\s*- Do NOT.*$",
            r"^\s*- Use lowercase.*$",
            r"^\s*- The parameters.*$",
        ]
        for pattern in leakage_patterns:
            content = re.sub(pattern, "", content, flags=re.MULTILINE | re.IGNORECASE)

        # Clean up whitespace
        content = re.sub(r"\n{4,}", "\n\n\n", content)

        return content.strip()
