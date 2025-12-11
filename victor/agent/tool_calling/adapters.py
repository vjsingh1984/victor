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
Provider-specific tool calling adapters.

Each adapter implements the BaseToolCallingAdapter interface for its provider,
handling format conversion, parsing, and capability detection.

Capabilities are resolved from model_capabilities.yaml when available,
falling back to hardcoded defaults for robustness.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from victor.agent.tool_calling.base import (
    BaseToolCallingAdapter,
    FallbackParsingMixin,
    ToolCall,
    ToolCallingCapabilities,
    ToolCallFormat,
    ToolCallParseResult,
)
from victor.agent.tool_calling.capabilities import ModelCapabilityLoader
from victor.providers.base import ToolDefinition

logger = logging.getLogger(__name__)

# Capability loader instance (lazy-initialized)
class _CapabilityLoaderHolder:
    """Holder for capability loader singleton."""
    _instance: Optional[ModelCapabilityLoader] = None

    @classmethod
    def get(cls) -> ModelCapabilityLoader:
        """Get or create the capability loader singleton."""
        if cls._instance is None:
            cls._instance = ModelCapabilityLoader()
        return cls._instance


def _get_capability_loader() -> ModelCapabilityLoader:
    """Get or create the capability loader singleton."""
    return _CapabilityLoaderHolder.get()


class AnthropicToolCallingAdapter(BaseToolCallingAdapter):
    """Adapter for Anthropic Claude models.

    Anthropic uses a distinct tool format with input_schema instead of parameters.
    Tool calls are returned as content blocks with type="tool_use".
    """

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def get_capabilities(self) -> ToolCallingCapabilities:
        # Load from YAML config, use format hint for Anthropic
        loader = _get_capability_loader()
        caps = loader.get_capabilities("anthropic", self.model, ToolCallFormat.ANTHROPIC)

        # Anthropic always has native tool calls, override if YAML is missing
        if not caps.native_tool_calls:
            return ToolCallingCapabilities(
                native_tool_calls=True,
                streaming_tool_calls=True,
                parallel_tool_calls=True,
                tool_choice_param=True,
                json_fallback_parsing=False,
                xml_fallback_parsing=False,
                thinking_mode=False,
                requires_strict_prompting=False,
                tool_call_format=ToolCallFormat.ANTHROPIC,
                argument_format="json",
                recommended_max_tools=50,
                recommended_tool_budget=20,
            )
        return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to Anthropic format (name, description, input_schema)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse Anthropic tool calls.

        Anthropic returns tool calls in the raw_tool_calls list with format:
        {"id": "...", "name": "...", "arguments": {...}}
        """
        if not raw_tool_calls:
            return ToolCallParseResult(remaining_content=content)

        tool_calls = []
        for tc in raw_tool_calls:
            if not self.is_valid_tool_name(tc.get("name", "")):
                continue
            tool_calls.append(
                ToolCall(
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", {}),
                    id=tc.get("id"),
                )
            )

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content=content,
            parse_method="native",
            confidence=1.0,
        )


class OpenAIToolCallingAdapter(BaseToolCallingAdapter):
    """Adapter for OpenAI GPT models.

    OpenAI uses function calling format with tool_calls containing
    function.name and function.arguments (as JSON string).
    """

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_capabilities(self) -> ToolCallingCapabilities:
        # Load from YAML config
        loader = _get_capability_loader()
        caps = loader.get_capabilities("openai", self.model, ToolCallFormat.OPENAI)

        # OpenAI always has native tool calls, override if YAML is missing
        if not caps.native_tool_calls:
            return ToolCallingCapabilities(
                native_tool_calls=True,
                streaming_tool_calls=True,
                parallel_tool_calls=True,
                tool_choice_param=True,
                json_fallback_parsing=False,
                xml_fallback_parsing=False,
                thinking_mode=False,
                requires_strict_prompting=False,
                tool_call_format=ToolCallFormat.OPENAI,
                argument_format="json",
                recommended_max_tools=50,
                recommended_tool_budget=20,
            )
        return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse OpenAI tool calls.

        OpenAI returns: {"id": "...", "name": "...", "arguments": "..."}
        where arguments is a JSON string that needs parsing.
        """
        if not raw_tool_calls:
            return ToolCallParseResult(remaining_content=content)

        tool_calls = []
        warnings = []

        for tc in raw_tool_calls:
            name = tc.get("name", "")
            if not self.is_valid_tool_name(name):
                warnings.append(f"Skipped invalid tool name: {name}")
                continue

            # Parse arguments (may be string or dict)
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    warnings.append(f"Failed to parse arguments for {name}")
                    args = {}

            tool_calls.append(ToolCall(name=name, arguments=args, id=tc.get("id")))

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content=content,
            parse_method="native",
            confidence=1.0,
            warnings=warnings,
        )


class OllamaToolCallingAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
    """Adapter for Ollama local models.

    Ollama supports native tool calling for Llama 3.1+, Qwen 2.5+, Mistral, etc.
    Falls back to JSON/XML parsing from content for unsupported models.

    Uses FallbackParsingMixin for common parsing logic shared across adapters.

    References:
    - https://docs.ollama.com/capabilities/tool-calling
    - https://ollama.com/search?c=tools
    """

    # Models with known good native tool calling support
    # Full list from https://ollama.com/search?c=tools (December 2025)
    NATIVE_TOOL_MODELS = frozenset(
        [
            # Llama family
            "llama3.1",
            "llama-3.1",
            "llama3.2",
            "llama-3.2",
            "llama3.3",
            "llama-3.3",
            "llama4",
            "llama-4",
            "llama3-groq-tool-use",
            # Qwen family
            "qwen2",
            "qwen-2",
            "qwen2.5",
            "qwen-2.5",
            "qwen3",
            "qwen-3",
            "qwq",
            # Mistral family
            "mistral",
            "mixtral",
            "mistral-small",
            "mistral-nemo",
            "mistral-large",
            "devstral",
            "magistral",
            # Command-R family
            "command-r",
            "command-r-plus",
            "command-r7b",
            "command-a",
            # Tool-specialized models
            "firefunction",
            "hermes",
            "hermes3",
            "functionary",
            "athene-v2",
            # DeepSeek (when using -tools variant)
            "deepseek",
            # Granite family (IBM)
            "granite3",
            "granite3.1",
            "granite3.2",
            "granite3.3",
            "granite4",
            # Other tool-capable models
            "gpt-oss",
            "nemotron",
            "nemotron-mini",
            "smollm2",
            "aya-expanse",
            "cogito",
            "phi4",
        ]
    )

    # Parameter aliases: maps model-specific parameter names to standard names
    # Format: {tool_name: {model_param: standard_param}}
    PARAMETER_ALIASES = {
        "read": {
            "line_start": "offset",  # gpt-oss uses line_start/line_end
            "line_end": "_line_end",  # Special handling needed
            "loc": "offset",  # browser.open style
            "num_lines": "limit",
        },
        "list_directory": {
            "dir": "path",
            "directory": "path",
        },
        "write": {
            "file": "path",
            "file_path": "path",
            "text": "content",
        },
        "edit": {
            "file": "path",
            "file_path": "path",
        },
        "shell": {
            "cmd": "command",
        },
        "search": {
            "q": "query",
            "term": "query",
        },
    }

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _has_native_support(self) -> bool:
        """Check if current model has native tool calling."""
        return any(pattern in self.model_lower for pattern in self.NATIVE_TOOL_MODELS)

    def _has_thinking_mode(self) -> bool:
        """Check if model supports Qwen3 thinking mode."""
        return "qwen3" in self.model_lower or "qwen-3" in self.model_lower

    def get_capabilities(self) -> ToolCallingCapabilities:
        # Determine format based on model detection
        has_native = self._has_native_support()
        format_hint = ToolCallFormat.OLLAMA_NATIVE if has_native else ToolCallFormat.OLLAMA_JSON

        # Load from YAML config with model pattern matching
        loader = _get_capability_loader()
        caps = loader.get_capabilities("ollama", self.model, format_hint)

        # Apply model-specific overrides that YAML can't detect
        if self._has_thinking_mode() and not caps.thinking_mode:
            # Qwen3 thinking mode detected but not in YAML
            caps = ToolCallingCapabilities(
                native_tool_calls=caps.native_tool_calls,
                streaming_tool_calls=caps.streaming_tool_calls,
                parallel_tool_calls=caps.parallel_tool_calls,
                tool_choice_param=caps.tool_choice_param,
                json_fallback_parsing=caps.json_fallback_parsing,
                xml_fallback_parsing=caps.xml_fallback_parsing,
                thinking_mode=True,
                requires_strict_prompting=caps.requires_strict_prompting,
                tool_call_format=format_hint,
                argument_format=caps.argument_format,
                recommended_max_tools=caps.recommended_max_tools,
                recommended_tool_budget=caps.recommended_tool_budget,
            )

        return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to Ollama format (OpenAI-compatible)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _parse_hybrid_xml_format(
        self,
        content: str,
        validate_name_fn: Optional[Callable[[str], bool]] = None,
    ) -> ToolCallParseResult:
        """Parse Ollama-specific hybrid XML format.

        Handles malformed patterns like <function=X>...</tool_call> that some
        Ollama models (especially Qwen3-coder) produce. This is specific to Ollama
        and not added to the base class to avoid regressions for other adapters.

        Args:
            content: Response content to parse
            validate_name_fn: Optional function to validate tool names

        Returns:
            ToolCallParseResult with parsed tool calls
        """
        import re
        import json

        # Hybrid pattern: <function=name> with </tool_call> closing
        # This is a common malformed pattern from Qwen3 models on Ollama
        hybrid_pattern = r"<function=([^>]+)>(.*?)</tool_call>"
        hybrid_matches = re.findall(hybrid_pattern, content, re.DOTALL | re.IGNORECASE)

        if not hybrid_matches:
            return ToolCallParseResult(remaining_content=content)

        tool_calls: List[ToolCall] = []
        warnings: List[str] = ["Used hybrid XML pattern recovery (Ollama-specific)"]

        for name, params_content in hybrid_matches:
            name = name.strip()

            # Validate name if validator provided
            if validate_name_fn and not validate_name_fn(name):
                warnings.append(f"Skipped invalid tool name: {name}")
                continue

            # Parse <parameter=X>value</parameter> tags
            param_pattern = r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>"
            param_matches = re.findall(param_pattern, params_content, re.DOTALL)
            args = {p.strip(): v.strip() for p, v in param_matches}

            tool_calls.append(ToolCall(name=name, arguments=args))

        if not tool_calls:
            return ToolCallParseResult(remaining_content=content)

        # Remove matched content
        remaining = re.sub(hybrid_pattern, "", content, flags=re.DOTALL | re.IGNORECASE)
        remaining = remaining.strip()

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content=remaining,
            parse_method="ollama_hybrid_xml",
            confidence=0.65,  # Slightly lower confidence for recovered format
            warnings=warnings,
        )

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse Ollama tool calls with fallback support.

        Uses FallbackParsingMixin methods for common parsing logic.
        Includes Ollama-specific hybrid XML pattern recovery.
        """
        # Try native tool calls first (using mixin method)
        if raw_tool_calls:
            result = self.parse_native_tool_calls(raw_tool_calls, self.is_valid_tool_name)
            if result.tool_calls:
                # Preserve original content for reference
                return ToolCallParseResult(
                    tool_calls=result.tool_calls,
                    remaining_content=content,
                    parse_method=result.parse_method,
                    confidence=result.confidence,
                    warnings=result.warnings,
                )

        # Fallback: Try content parsing (using mixin method)
        if content:
            result = self.parse_from_content(content, self.is_valid_tool_name)
            if result.tool_calls:
                return result

            # Ollama-specific: Try hybrid XML pattern recovery
            # This handles <function=X>...</tool_call> malformed patterns
            hybrid_result = self._parse_hybrid_xml_format(content, self.is_valid_tool_name)
            if hybrid_result.tool_calls:
                return hybrid_result

        return ToolCallParseResult(remaining_content=content)

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints for Ollama models."""
        capabilities = self.get_capabilities()

        if capabilities.native_tool_calls:
            hints = [
                "TOOL USAGE:",
                "- Use list_directory and read_file to inspect code.",
                "- Call tools one at a time, waiting for results.",
                "- After 2-3 successful tool calls, provide your answer.",
                "- Do NOT make identical repeated tool calls.",
                "",
                "RESPONSE FORMAT:",
                "- Write your answer in plain, readable text.",
                "- Do NOT output raw JSON in your response.",
                "- Do NOT output XML tags or function call syntax.",
                "- Be concise and answer the question directly.",
            ]

            if capabilities.thinking_mode:
                hints.extend(
                    [
                        "",
                        "QWEN3 MODE:",
                        "- Use /no_think for simple questions.",
                        "- Provide direct answers without excessive reasoning.",
                    ]
                )

            return "\n".join(hints)
        else:
            return "\n".join(
                [
                    "CRITICAL TOOL RULES:",
                    "1. Call tools ONE AT A TIME. Never batch calls.",
                    "2. After reading 2-3 files, STOP and answer.",
                    "3. Do NOT repeat the same tool call.",
                    "4. Do NOT invent file contents.",
                    "",
                    "CRITICAL OUTPUT RULES:",
                    "1. Write your answer in plain English.",
                    '2. Do NOT output JSON objects like {"name": ...}.',
                    "3. Do NOT output XML tags like </function> or </parameter>.",
                    "4. Do NOT output function call syntax.",
                    "5. Keep your answer focused and concise.",
                ]
            )


class GoogleToolCallingAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
    """Adapter for Google Gemini models.

    Google Gemini uses a different function calling format than OpenAI.
    Tool definitions use 'function_declarations' format.

    Gemini models may also output tool calls in text format like:
    - <execute_bash>command</execute_bash>
    - <tool_code>python_code</tool_code>
    - ```tool_code ... ```

    This adapter handles both native function calls and fallback text parsing.

    References:
    - https://ai.google.dev/gemini-api/docs/function-calling
    """

    # Patterns for text-based tool calls from Gemini
    EXECUTE_BASH_PATTERN = re.compile(r"<execute_bash>\s*(.*?)\s*</execute_bash>", re.DOTALL)
    TOOL_CODE_PATTERN = re.compile(r"<tool_code>\s*(.*?)\s*</tool_code>", re.DOTALL)
    CODE_BLOCK_PATTERN = re.compile(r"```tool_code\s*(.*?)\s*```", re.DOTALL)
    # Pattern for <ctrl42>call: `tool_name` followed by ```json {...}```
    CTRL42_CALL_PATTERN = re.compile(
        r"<ctrl42>call:\s*`?(\w+)`?\s*```json\s*(.*?)\s*```", re.DOTALL
    )
    # Pattern for simpler call: `tool_name` followed by json block
    SIMPLE_CALL_PATTERN = re.compile(
        r"(?:call|Call):\s*`?(\w+)`?\s*```json\s*(.*?)\s*```", re.DOTALL
    )
    # Pattern for Python-style function call: tool_name("value") or tool_name(path="value")
    # Handles both positional and keyword arguments
    PYTHON_CALL_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:print\s*\(\s*)?(\w+)\s*\(\s*"
        r"(?:(?:path|command|file_path|query)\s*=\s*)?"  # Optional keyword
        r"['\"]([^'\"]+)['\"]\s*"
        r"\)?",
        re.MULTILINE,
    )
    # Pattern to detect hallucinated tool output (Gemini likes to fake results)
    TOOL_OUTPUT_HALLUCINATION = re.compile(
        r"<TOOL_OUTPUT>.*?</TOOL_OUTPUT>|```text\n.*?```", re.DOTALL
    )

    @property
    def provider_name(self) -> str:
        return "google"

    def get_capabilities(self) -> ToolCallingCapabilities:
        """Get capabilities for Google Gemini models."""
        loader = _get_capability_loader()
        caps = loader.get_capabilities("google", self.model, ToolCallFormat.GOOGLE)

        # Gemini always has native tool calls when tools are provided
        if not caps.native_tool_calls:
            return ToolCallingCapabilities(
                native_tool_calls=True,
                streaming_tool_calls=True,
                parallel_tool_calls=True,
                tool_choice_param=False,  # Gemini has different tool_config
                json_fallback_parsing=True,
                xml_fallback_parsing=True,  # For <execute_bash> style
                thinking_mode=False,
                requires_strict_prompting=False,
                tool_call_format=ToolCallFormat.GOOGLE,
                argument_format="json",
                recommended_max_tools=30,
                recommended_tool_budget=15,
            )
        return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to Google Gemini function_declarations format.

        Google format requires:
        {
            "function_declarations": [
                {
                    "name": "tool_name",
                    "description": "description",
                    "parameters": {...}
                }
            ]
        }
        """
        function_declarations = []
        for tool in tools:
            func_decl = {
                "name": tool.name,
                "description": tool.description,
            }
            # Only add parameters if they exist and are not empty
            if tool.parameters and tool.parameters.get("properties"):
                func_decl["parameters"] = tool.parameters
            function_declarations.append(func_decl)

        return [{"function_declarations": function_declarations}]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse Google Gemini tool calls.

        Handles:
        1. Native function calls from API response
        2. Text-based <execute_bash>...</execute_bash> format
        3. Text-based <tool_code>...</tool_code> format
        """
        warnings = []

        # Try native tool calls first (from function_call response)
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                name = tc.get("name", "")
                if not self.is_valid_tool_name(name):
                    warnings.append(f"Skipped invalid tool name: {name}")
                    continue

                # Google returns args as dict directly, not as JSON string
                args = tc.get("args", {}) or tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        warnings.append(f"Failed to parse arguments for {name}")
                        args = {}

                tool_calls.append(ToolCall(name=name, arguments=args, id=tc.get("id")))

            if tool_calls:
                return ToolCallParseResult(
                    tool_calls=tool_calls,
                    remaining_content=content,
                    parse_method="native",
                    confidence=1.0,
                    warnings=warnings,
                )

        # Fallback: Parse text-based tool calls from Gemini output
        if content:
            result = self._parse_gemini_text_tools(content)
            if result.tool_calls:
                return result

            # Try standard JSON fallback (using mixin)
            result = self.parse_from_content(content, self.is_valid_tool_name)
            if result.tool_calls:
                return result

        return ToolCallParseResult(remaining_content=content)

    def _parse_gemini_text_tools(self, content: str) -> ToolCallParseResult:
        """Parse Gemini's text-based tool call format.

        Handles patterns like:
        - <execute_bash>ls -la</execute_bash>
        - <tool_code>print(read_file("path"))</tool_code>
        - <ctrl42>call: `tool_name` ```json {...}```
        - call: `tool_name` ```json {...}```
        - print(list_directory(path="..."))
        """
        tool_calls = []
        warnings = []
        remaining = content

        # Parse <ctrl42>call: patterns first (most specific)
        for match in self.CTRL42_CALL_PATTERN.finditer(content):
            tool_name = match.group(1).strip()
            json_str = match.group(2).strip()
            if tool_name and self.is_valid_tool_name(tool_name):
                try:
                    args = json.loads(json_str) if json_str else {}
                    tool_calls.append(ToolCall(name=tool_name, arguments=args))
                    remaining = remaining.replace(match.group(0), "")
                    logger.debug(f"Parsed ctrl42 call: {tool_name}({args})")
                except json.JSONDecodeError:
                    warnings.append(f"Failed to parse JSON for {tool_name}: {json_str[:50]}")

        # Parse simpler call: patterns
        for match in self.SIMPLE_CALL_PATTERN.finditer(content):
            tool_name = match.group(1).strip()
            json_str = match.group(2).strip()
            if tool_name and self.is_valid_tool_name(tool_name):
                try:
                    args = json.loads(json_str) if json_str else {}
                    tool_calls.append(ToolCall(name=tool_name, arguments=args))
                    remaining = remaining.replace(match.group(0), "")
                    logger.debug(f"Parsed simple call: {tool_name}({args})")
                except json.JSONDecodeError:
                    warnings.append(f"Failed to parse JSON for {tool_name}")

        # Strip hallucinated tool output (Gemini likes to make up results)
        content_cleaned = self.TOOL_OUTPUT_HALLUCINATION.sub("", content)
        if content_cleaned != content:
            logger.warning("Stripped hallucinated tool output from Gemini response")
            remaining = content_cleaned

        # Parse Python-style function calls: tool_name("value") or tool_name(path="value")
        for match in self.PYTHON_CALL_PATTERN.finditer(content_cleaned):
            tool_name = match.group(1).strip()
            arg_value = match.group(2).strip()
            if tool_name and self.is_valid_tool_name(tool_name):
                # Map common tool argument names
                if tool_name in ("list_directory", "read_file"):
                    args = {"path": arg_value}
                elif tool_name == "execute_bash":
                    args = {"command": arg_value}
                elif tool_name in ("code_search", "semantic_code_search"):
                    args = {"query": arg_value}
                else:
                    args = {"path": arg_value}  # Default
                tool_calls.append(ToolCall(name=tool_name, arguments=args))
                remaining = remaining.replace(match.group(0), "")
                logger.debug(f"Parsed Python call: {tool_name}({args})")

        # Parse <execute_bash> patterns
        for match in self.EXECUTE_BASH_PATTERN.finditer(content):
            command = match.group(1).strip()
            if command:
                tool_calls.append(
                    ToolCall(
                        name="execute_bash",
                        arguments={"command": command},
                    )
                )
                remaining = remaining.replace(match.group(0), "")

        # Parse <tool_code> patterns (Python code that reads files, etc.)
        for match in self.TOOL_CODE_PATTERN.finditer(content):
            code = match.group(1).strip()
            if code:
                # Try to extract read_file calls
                read_file_match = re.search(r'read_file\s*\(\s*["\']([^"\']+)["\']\s*\)', code)
                if read_file_match:
                    path = read_file_match.group(1)
                    tool_calls.append(
                        ToolCall(
                            name="read_file",
                            arguments={"path": path},
                        )
                    )
                else:
                    # Generic tool code - might be execute_python
                    warnings.append(f"Unrecognized tool_code: {code[:50]}...")
                remaining = remaining.replace(match.group(0), "")

        # Parse ```tool_code blocks
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            code = match.group(1).strip()
            if code:
                read_file_match = re.search(r'read_file\s*\(\s*["\']([^"\']+)["\']\s*\)', code)
                if read_file_match:
                    path = read_file_match.group(1)
                    tool_calls.append(
                        ToolCall(
                            name="read_file",
                            arguments={"path": path},
                        )
                    )
                remaining = remaining.replace(match.group(0), "")

        if tool_calls:
            return ToolCallParseResult(
                tool_calls=tool_calls,
                remaining_content=remaining.strip(),
                parse_method="gemini_text_fallback",
                confidence=0.8,
                warnings=warnings,
            )

        return ToolCallParseResult(remaining_content=content)

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints for Google Gemini."""
        return "\n".join(
            [
                "TOOL USAGE:",
                "- Use the available tools to gather information when needed.",
                "- Call MULTIPLE tools in parallel when operations are independent.",
                "  Example: Read multiple files simultaneously, search while listing.",
                "- After gathering sufficient information, provide your answer.",
                "- Do NOT repeat identical tool calls.",
                "",
                "RESPONSE FORMAT:",
                "- Provide your final answer in clear, readable text.",
                "- Be concise and directly answer the question.",
            ]
        )


class LMStudioToolCallingAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
    """Dedicated adapter for LMStudio local models.

    LMStudio provides an OpenAI-compatible API but requires specialized handling:
    - Native tool support for "hammer badge" models (Qwen2.5, Llama3.1, Ministral)
    - [TOOL_REQUEST]...[END_TOOL_REQUEST] format for default mode
    - JSON fallback parsing for unsupported models

    Uses FallbackParsingMixin for common parsing logic (shared with OllamaToolCallingAdapter).

    References:
    - https://lmstudio.ai/docs/advanced/tool-use
    - https://lmstudio.ai/docs/api/openai-api
    """

    # Models with native LMStudio tool support (hammer badge)
    NATIVE_TOOL_MODELS = frozenset(
        [
            "qwen2.5",
            "qwen-2.5",
            "qwen3",
            "qwen-3",
            "llama-3.1",
            "llama3.1",
            "llama-3.2",
            "llama3.2",
            "llama-3.3",
            "llama3.3",
            "ministral",
            "mistral",
            "mixtral",
            "command-r",
            "hermes",
            "functionary",
        ]
    )

    @property
    def provider_name(self) -> str:
        return "lmstudio"

    def _has_native_support(self) -> bool:
        """Check if current model has native tool calling support."""
        return any(pattern in self.model_lower for pattern in self.NATIVE_TOOL_MODELS)

    def _has_thinking_mode(self) -> bool:
        """Check if model supports Qwen3 thinking mode."""
        return "qwen3" in self.model_lower or "qwen-3" in self.model_lower

    def get_capabilities(self) -> ToolCallingCapabilities:
        """Get capabilities for LMStudio models."""
        has_native = self._has_native_support()
        format_hint = (
            ToolCallFormat.LMSTUDIO_NATIVE if has_native else ToolCallFormat.LMSTUDIO_DEFAULT
        )

        # Load from YAML config with model pattern matching
        loader = _get_capability_loader()
        caps = loader.get_capabilities("lmstudio", self.model, format_hint)

        # Apply runtime model detection for thinking mode
        if self._has_thinking_mode() and not caps.thinking_mode:
            caps = ToolCallingCapabilities(
                native_tool_calls=caps.native_tool_calls,
                streaming_tool_calls=caps.streaming_tool_calls,
                parallel_tool_calls=caps.parallel_tool_calls,
                tool_choice_param=caps.tool_choice_param,
                json_fallback_parsing=True,
                xml_fallback_parsing=False,
                thinking_mode=True,
                requires_strict_prompting=caps.requires_strict_prompting,
                tool_call_format=format_hint,
                argument_format=caps.argument_format,
                recommended_max_tools=caps.recommended_max_tools,
                recommended_tool_budget=caps.recommended_tool_budget,
            )

        return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to OpenAI format (LMStudio is OpenAI-compatible)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse LMStudio tool calls with multi-format support.

        Handles:
        1. Native tool calls from API response (hammer badge models)
        2. [TOOL_REQUEST]...[END_TOOL_REQUEST] format (default mode)
        3. JSON fallback from content (using FallbackParsingMixin)
        """

        # 1. Try native tool calls first (using mixin method)
        if raw_tool_calls:
            result = self.parse_native_tool_calls(raw_tool_calls, self.is_valid_tool_name)
            if result.tool_calls:
                return ToolCallParseResult(
                    tool_calls=result.tool_calls,
                    remaining_content=content,
                    parse_method=result.parse_method,
                    confidence=result.confidence,
                    warnings=result.warnings,
                )

        # 2. Try [TOOL_REQUEST] format (LMStudio default mode)
        if content:
            result = self._parse_tool_request_format(content)
            if result.tool_calls:
                return result

        # 3. JSON fallback from content (using mixin method)
        if content:
            result = self.parse_from_content(content, self.is_valid_tool_name)
            if result.tool_calls:
                return result

        return ToolCallParseResult(remaining_content=content)

    def _parse_tool_request_format(self, content: str) -> ToolCallParseResult:
        """Parse LMStudio [TOOL_REQUEST]...[END_TOOL_REQUEST] format.

        Example:
        [TOOL_REQUEST]{"name": "read_file", "arguments": {"path": "/etc/hosts"}}[END_TOOL_REQUEST]
        """
        pattern = r"\[TOOL_REQUEST\](.*?)\[END_TOOL_REQUEST\]"
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            return ToolCallParseResult(remaining_content=content)

        tool_calls = []
        warnings = []

        for match in matches:
            try:
                data = json.loads(match.strip())
                name = data.get("name", "")
                if self.is_valid_tool_name(name):
                    args = data.get("arguments") or data.get("parameters", {})
                    tool_calls.append(ToolCall(name=name, arguments=args))
                else:
                    warnings.append(f"Skipped invalid tool name: {name}")
            except json.JSONDecodeError:
                warnings.append(f"Failed to parse TOOL_REQUEST JSON: {match[:50]}...")

        remaining = re.sub(pattern, "", content, flags=re.DOTALL).strip()

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content=remaining,
            parse_method="tool_request_format",
            confidence=0.85,
            warnings=warnings,
        )

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints for LMStudio models."""
        capabilities = self.get_capabilities()

        if capabilities.native_tool_calls:
            hints = [
                "TOOL USAGE:",
                "- Use the available tools to gather information.",
                "- Call tools one at a time and wait for results.",
                "- After 2-3 successful tool calls, provide your answer.",
                "- Do NOT repeat identical tool calls.",
                "",
                "RESPONSE FORMAT:",
                "- Provide your answer in plain, readable text.",
                "- Do NOT include JSON, XML, or tool syntax in your response.",
                "- Be direct and answer the user's question.",
            ]

            if capabilities.thinking_mode:
                hints.extend(
                    [
                        "",
                        "QWEN3 MODE:",
                        "- Use /no_think for simple questions.",
                        "- Provide direct answers without excessive reasoning.",
                    ]
                )

            return "\n".join(hints)
        else:
            return "\n".join(
                [
                    "CRITICAL RULES:",
                    "1. Call tools ONE AT A TIME. Wait for each result.",
                    "2. After reading 2-3 files, STOP and provide your answer.",
                    "3. Do NOT repeat the same tool call.",
                    "4. Do NOT invent or guess file contents.",
                    "",
                    "OUTPUT FORMAT:",
                    "1. Your answer must be in plain English text.",
                    "2. Do NOT output JSON objects in your response.",
                    "3. Do NOT output XML tags.",
                    "4. Do NOT output [TOOL_REQUEST] markers in your final answer.",
                ]
            )


class OpenAICompatToolCallingAdapter(BaseToolCallingAdapter):
    """Adapter for OpenAI-compatible local providers (vLLM).

    Note: LMStudio now uses the dedicated LMStudioToolCallingAdapter.
    This adapter is retained for vLLM compatibility.

    vLLM:
    - Requires --enable-auto-tool-choice and --tool-call-parser flags
    - Supports many model-specific parsers (hermes, mistral, llama3_json, etc.)

    References:
    - https://docs.vllm.ai/en/stable/features/tool_calling/
    """

    # Models with native LMStudio tool support
    LMSTUDIO_NATIVE_MODELS = frozenset(
        [
            "qwen2.5",
            "qwen-2.5",
            "qwen3",
            "qwen-3",
            "llama-3.1",
            "llama3.1",
            "llama-3.2",
            "llama3.2",
            "ministral",
            "mistral",
        ]
    )

    def __init__(
        self,
        model: str = "",
        config: Optional[Dict[str, Any]] = None,
        provider_variant: str = "lmstudio",
    ):
        """Initialize adapter.

        Args:
            model: Model name
            config: Configuration
            provider_variant: "lmstudio" or "vllm"
        """
        super().__init__(model, config)
        self.provider_variant = provider_variant

    @property
    def provider_name(self) -> str:
        return self.provider_variant

    def _has_native_support(self) -> bool:
        """Check if model has native tool support."""
        return any(pattern in self.model_lower for pattern in self.LMSTUDIO_NATIVE_MODELS)

    def get_capabilities(self) -> ToolCallingCapabilities:
        has_native = self._has_native_support()

        if self.provider_variant == "vllm":
            # vLLM: Load from YAML with model pattern matching
            loader = _get_capability_loader()
            caps = loader.get_capabilities("vllm", self.model, ToolCallFormat.VLLM)

            # vLLM always has native when properly configured
            if not caps.native_tool_calls:
                return ToolCallingCapabilities(
                    native_tool_calls=True,
                    streaming_tool_calls=True,
                    parallel_tool_calls=True,
                    tool_choice_param=True,
                    json_fallback_parsing=True,
                    xml_fallback_parsing=False,
                    thinking_mode=False,
                    requires_strict_prompting=False,
                    tool_call_format=ToolCallFormat.VLLM,
                    argument_format="json",
                    recommended_max_tools=30,
                    recommended_tool_budget=15,
                )
            return caps
        else:
            # LMStudio: Load from YAML with model pattern matching
            format_hint = (
                ToolCallFormat.LMSTUDIO_NATIVE if has_native else ToolCallFormat.LMSTUDIO_DEFAULT
            )
            loader = _get_capability_loader()
            caps = loader.get_capabilities("lmstudio", self.model, format_hint)

            # Apply runtime model detection for thinking mode
            if "qwen3" in self.model_lower and not caps.thinking_mode:
                caps = ToolCallingCapabilities(
                    native_tool_calls=caps.native_tool_calls,
                    streaming_tool_calls=caps.streaming_tool_calls,
                    parallel_tool_calls=caps.parallel_tool_calls,
                    tool_choice_param=caps.tool_choice_param,
                    json_fallback_parsing=caps.json_fallback_parsing,
                    xml_fallback_parsing=caps.xml_fallback_parsing,
                    thinking_mode=True,
                    requires_strict_prompting=caps.requires_strict_prompting,
                    tool_call_format=format_hint,
                    argument_format=caps.argument_format,
                    recommended_max_tools=caps.recommended_max_tools,
                    recommended_tool_budget=caps.recommended_tool_budget,
                )
            return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse tool calls from OpenAI-compatible response."""
        warnings = []

        # Try native tool calls
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                name = tc.get("name", "")
                if not self.is_valid_tool_name(name):
                    warnings.append(f"Skipped invalid tool name: {name}")
                    continue

                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        warnings.append(f"Failed to parse arguments for {name}")
                        args = {}

                tool_calls.append(ToolCall(name=name, arguments=args, id=tc.get("id")))

            if tool_calls:
                return ToolCallParseResult(
                    tool_calls=tool_calls,
                    remaining_content=content,
                    parse_method="native",
                    confidence=1.0,
                    warnings=warnings,
                )

        # Fallback: Parse JSON from content
        if content:
            result = self._parse_json_from_content(content)
            if result.tool_calls:
                return result

            # LMStudio default format: [TOOL_REQUEST]...[END_TOOL_REQUEST]
            result = self._parse_tool_request_format(content)
            if result.tool_calls:
                return result

        return ToolCallParseResult(remaining_content=content)

    def _parse_json_from_content(self, content: str) -> ToolCallParseResult:
        """Parse JSON tool calls from content."""
        content = content.strip()
        if not content:
            return ToolCallParseResult()

        try:
            data = json.loads(content)
            if isinstance(data, dict) and "name" in data:
                name = data.get("name", "")
                if self.is_valid_tool_name(name):
                    args = data.get("arguments") or data.get("parameters", {})
                    return ToolCallParseResult(
                        tool_calls=[ToolCall(name=name, arguments=args)],
                        remaining_content="",
                        parse_method="json_fallback",
                        confidence=0.9,
                    )
        except json.JSONDecodeError:
            pass

        return ToolCallParseResult(remaining_content=content)

    def _parse_tool_request_format(self, content: str) -> ToolCallParseResult:
        """Parse LMStudio [TOOL_REQUEST] format."""
        pattern = r"\[TOOL_REQUEST\](.*?)\[END_TOOL_REQUEST\]"
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            return ToolCallParseResult(remaining_content=content)

        tool_calls = []
        warnings = []

        for match in matches:
            try:
                data = json.loads(match.strip())
                name = data.get("name", "")
                if self.is_valid_tool_name(name):
                    args = data.get("arguments") or data.get("parameters", {})
                    tool_calls.append(ToolCall(name=name, arguments=args))
            except json.JSONDecodeError:
                warnings.append("Failed to parse TOOL_REQUEST content")

        remaining = re.sub(pattern, "", content, flags=re.DOTALL).strip()

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content=remaining,
            parse_method="tool_request_format",
            confidence=0.8,
            warnings=warnings,
        )

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints."""
        capabilities = self.get_capabilities()

        if self.provider_variant == "vllm":
            return "\n".join(
                [
                    "TOOL CALLING:",
                    "- Tools are called using the standard OpenAI function calling format.",
                    "- You may call tools when needed to gather information.",
                    "- After gathering sufficient information (2-4 tool calls), provide your answer.",
                    "- Do NOT repeat the same tool call with identical arguments.",
                    "",
                    "IMPORTANT:",
                    "- Do not output raw JSON tool calls in your text response.",
                    "- When you're done using tools, provide a human-readable answer.",
                ]
            )

        if capabilities.native_tool_calls:
            return "\n".join(
                [
                    "TOOL USAGE:",
                    "- Use tools via the OpenAI function calling format.",
                    "- Call tools one at a time and wait for results.",
                    "- After 2-3 tool calls, provide your answer.",
                    "- Do NOT repeat identical tool calls.",
                    "",
                    "RESPONSE FORMAT:",
                    "- Provide answers in plain, readable text.",
                    "- Do NOT include JSON, XML, or tool syntax in your response.",
                    "- Be direct and answer the user's question.",
                ]
            )
        else:
            return "\n".join(
                [
                    "CRITICAL RULES:",
                    "1. Call tools ONE AT A TIME. Wait for each result.",
                    "2. After reading 2-3 files, STOP and provide your answer.",
                    "3. Do NOT repeat the same tool call.",
                    "4. Do NOT invent or guess file contents.",
                    "",
                    "OUTPUT FORMAT:",
                    "1. Your answer must be in plain English text.",
                    "2. Do NOT output JSON objects in your response.",
                    "3. Do NOT output XML tags like </function> or <parameter>.",
                    "4. Do NOT output [TOOL_REQUEST] or similar markers.",
                ]
            )
