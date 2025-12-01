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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


from victor.providers.base import ToolDefinition


class ToolCallFormat(Enum):
    """Supported tool call formats across providers."""

    OPENAI = "openai"  # OpenAI function calling format
    ANTHROPIC = "anthropic"  # Anthropic tool use format
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
    thinking_mode: bool = False  # Supports /think /no_think (Qwen3)
    requires_strict_prompting: bool = False  # Needs strict system prompts

    # Format details
    tool_call_format: ToolCallFormat = ToolCallFormat.UNKNOWN
    argument_format: str = "json"  # "json" or "python_dict"

    # Recommended limits
    recommended_max_tools: int = 20  # Max tools to send
    recommended_tool_budget: int = 12  # Max tool calls per turn


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

        if capabilities.native_tool_calls and not capabilities.requires_strict_prompting:
            return ""  # Native support, no hints needed

        hints = []

        if capabilities.requires_strict_prompting:
            hints.append("Call tools ONE AT A TIME. Wait for results.")
            hints.append("After 2-3 tool calls, provide your answer.")
            hints.append("Do NOT output JSON, XML, or tool syntax in responses.")

        if capabilities.thinking_mode:
            hints.append("Use /no_think for simple questions.")

        return "\n".join(hints)

    def normalize_arguments(self, arguments: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Normalize tool arguments.

        Default implementation returns arguments unchanged.
        Subclasses can override for provider-specific normalization.

        Args:
            arguments: Raw arguments from model
            tool_name: Name of the tool

        Returns:
            Normalized arguments
        """
        return arguments

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
