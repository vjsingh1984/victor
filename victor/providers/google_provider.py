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

"""Google Gemini provider implementation.

Requires optional dependency: pip install victor[google]

Note: This module uses the new google-genai SDK (google.genai) instead of
the deprecated google-generativeai package.
"""

import logging
import warnings
from typing import Any, Optional
from collections.abc import AsyncIterator

# Suppress Google SDK warning about non-text parts (we handle it correctly)
warnings.filterwarnings(
    "ignore",
    message=".*non-text parts in the response.*",
    category=UserWarning,
    module="google_genai.types",
)

# New Google GenAI SDK (replaces deprecated google.generativeai)
try:
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types  # type: ignore[import-untyped]

    # Verify that the Client class is available
    if not hasattr(genai, "Client"):
        HAS_GOOGLE_GENAI = False
        genai = None
        types = None
    else:
        HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False
    genai = None
    types = None

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.error_handler import HTTPErrorHandlerMixin

logger = logging.getLogger(__name__)

# Safety threshold levels mapping to new SDK string values
# The new SDK uses string-based category and threshold identifiers
SAFETY_LEVELS: dict[str, str] = {
    "block_none": "BLOCK_NONE",
    "off": "OFF",
    "block_few": "BLOCK_ONLY_HIGH",
    "block_some": "BLOCK_MEDIUM_AND_ABOVE",
    "block_most": "BLOCK_LOW_AND_ABOVE",
}

# Harm categories supported by the new SDK
HARM_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
]


class GoogleProvider(BaseProvider, HTTPErrorHandlerMixin):
    """Provider for Google Gemini models.

    Safety Settings:
        The `safety_level` parameter controls content filtering:
        - "block_none": No blocking (least restrictive, for development)
        - "block_few": Block only high probability harmful content
        - "block_some": Block medium and above (default)
        - "block_most": Block low and above (most restrictive)

    Example:
        # For code generation without safety blocks
        provider = GoogleProvider(api_key=key, safety_level="block_none")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 60,
        safety_level: str = "block_none",
        **kwargs: Any,
    ):
        """Initialize Google provider.

        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var, or use keyring)
            timeout: Request timeout in seconds
            safety_level: Safety filter level - "block_none", "block_few",
                         "block_some", or "block_most" (default: "block_none")
            **kwargs: Additional configuration

        Raises:
            ImportError: If google-genai package is not installed
        """
        # Resolve API key using centralized helper
        resolved_key = self._resolve_api_key(api_key, "google")

        if not HAS_GOOGLE_GENAI:
            raise ImportError(
                "google-genai package not installed. " "Install with: pip install google-genai"
            )
        super().__init__(api_key=resolved_key, timeout=timeout, **kwargs)

        # Initialize client with API key (new SDK pattern)
        self.client = genai.Client(api_key=resolved_key)

        # Configure safety settings using string-based thresholds
        threshold = SAFETY_LEVELS.get(safety_level, "BLOCK_NONE")
        self.safety_settings = [
            types.SafetySetting(category=cat, threshold=threshold) for cat in HARM_CATEGORIES
        ]
        logger.info(
            f"GoogleProvider initialized with safety_level={safety_level}, categories={len(self.safety_settings)}"
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "google"

    def supports_tools(self) -> bool:
        """Google Gemini supports function calling."""
        return True

    def supports_streaming(self) -> bool:
        """Google supports streaming."""
        return True

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request to Google.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "gemini-1.5-pro", "gemini-1.5-flash")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional Google parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If request fails
        """
        try:
            # Convert tools to Gemini format
            gemini_tools = self._convert_tools(tools) if tools else None

            # Build generation config
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                safety_settings=self.safety_settings,
            )

            # Add tools if present
            if gemini_tools:
                generation_config.tools = gemini_tools
                generation_config.tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                )

            # Convert messages to Gemini format
            contents = self._convert_messages(messages)

            logger.debug(
                f"Gemini chat: model={model}, tools={len(tools) if tools else 0}, messages={len(contents)}"
            )

            # Generate response using the new client pattern
            response = await self._execute_with_circuit_breaker(
                self.client.aio.models.generate_content,
                model=model,
                contents=contents,
                config=generation_config,
            )

            # Debug logging for raw response
            logger.debug(f"Gemini raw response type: {type(response)}")
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                logger.debug(
                    f"Gemini candidate: finish_reason={getattr(candidate, 'finish_reason', 'N/A')}"
                )
                if hasattr(candidate, "content") and candidate.content:
                    content = candidate.content
                    logger.debug(
                        f"Gemini content: role={getattr(content, 'role', 'N/A')}, parts_count={len(content.parts) if hasattr(content, 'parts') else 0}"
                    )
                    if hasattr(content, "parts"):
                        for i, part in enumerate(content.parts):
                            part_type = (
                                "text"
                                if hasattr(part, "text") and part.text
                                else (
                                    "function_call"
                                    if hasattr(part, "function_call") and part.function_call
                                    else "other"
                                )
                            )
                            if part_type == "function_call":
                                fc = part.function_call
                                logger.info(f"Gemini part[{i}]: NATIVE FUNCTION CALL: {fc.name}")
                            else:
                                logger.debug(f"Gemini part[{i}]: type={part_type}")
            else:
                logger.warning("Gemini response has no candidates!")

            return self._parse_response(response, model)

        except ProviderError:
            raise
        except Exception as e:
            raise self._handle_error(e, self.name)

    async def stream(  # type: ignore[override,misc]
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from Google.

        When tools are provided, this method uses non-streaming chat() internally
        because Gemini's native function calling returns structured function_call
        parts that can't be easily represented in streaming text chunks.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional parameters

        Yields:
            StreamChunk with incremental content

        Raises:
            ProviderError: If request fails
        """
        # When tools are provided, use non-streaming to properly handle native function calls
        # Gemini's native function calling returns structured parts that need to be processed
        # differently from streaming text chunks
        if tools:
            logger.debug(
                "Gemini stream with tools: falling back to non-streaming for native function calling"
            )
            response = await self.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **kwargs,
            )
            # Yield the full response as a single chunk with tool_calls attached
            # The orchestrator will handle extracting tool calls from the response
            chunk = StreamChunk(
                content=response.content or "",
                is_final=True,
                tool_calls=response.tool_calls,
            )
            yield chunk
            return

        try:
            # No tools - use true streaming for text-only responses
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                safety_settings=self.safety_settings,
            )

            # Convert messages
            contents = self._convert_messages(messages)

            logger.debug(f"Gemini stream (no tools): model={model}, messages={len(contents)}")

            # Stream response using the new client pattern
            response_stream = self.client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generation_config,
            )

            async for chunk in response_stream:
                parsed_chunk = self._parse_stream_chunk(chunk)
                if parsed_chunk:
                    yield parsed_chunk

        except ProviderError:
            raise
        except Exception as e:
            raise self._handle_error(e, self.name)

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[Any]:
        """Convert tools to Gemini format using proper SDK types.

        Uses google.genai.types for proper native function calling support.

        Args:
            tools: List of ToolDefinition objects

        Returns:
            List containing Tool objects with function declarations
        """
        if not tools:
            return []

        function_declarations = []
        for tool in tools:
            # Build parameters dict for FunctionDeclaration
            params = None
            if tool.parameters and tool.parameters.get("properties"):
                # Clean parameters - Gemini doesn't support some JSON Schema fields
                params = self._clean_schema_for_gemini(tool.parameters)

            # Create FunctionDeclaration using proper SDK types
            func_decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=params,
            )
            function_declarations.append(func_decl)

        logger.debug(
            f"Converted {len(function_declarations)} tools to Gemini FunctionDeclaration format"
        )
        # Wrap in Tool object - this is the proper way to pass tools
        return [types.Tool(function_declarations=function_declarations)]

    def _clean_schema_for_gemini(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Remove unsupported JSON Schema fields for Gemini API compatibility.

        Gemini's Schema proto doesn't support certain standard JSON Schema fields
        like 'default', 'examples', etc. This method recursively strips them.

        Args:
            schema: JSON Schema dictionary

        Returns:
            Cleaned schema compatible with Gemini API
        """
        # Fields not supported by Gemini's Schema proto
        unsupported_fields = {"default", "examples", "$schema", "$id", "definitions"}

        def clean_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                cleaned = {}
                for key, value in obj.items():
                    if key not in unsupported_fields:
                        cleaned[key] = clean_recursive(value)
                return cleaned
            elif isinstance(obj, list):
                return [clean_recursive(item) for item in obj]
            else:
                return obj

        return clean_recursive(schema)

    def _convert_messages(self, messages: list[Message]) -> list[Any]:
        """Convert messages to Gemini format.

        Args:
            messages: Standard messages

        Returns:
            List of Content objects for the new SDK
        """
        contents = []

        for msg in messages:
            # Gemini uses "user" and "model" roles
            role = "model" if msg.role == "assistant" else "user"

            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg.content)],
                )
            )

        return contents

    def _parse_response(self, response: Any, model: str) -> CompletionResponse:
        """Parse Google API response.

        Args:
            response: Raw Google response
            model: Model name

        Returns:
            Normalized CompletionResponse

        Raises:
            ProviderError: If response was blocked by safety filters
        """
        # Check for blocked responses (safety filters, recitation, etc.)
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)

            # Handle finish reasons (new SDK may use string or enum)
            finish_reason_str = str(finish_reason) if finish_reason else ""
            logger.debug(f"Gemini response finish_reason: {finish_reason_str}")

            if "SAFETY" in finish_reason_str:
                safety_ratings = getattr(candidate, "safety_ratings", [])
                logger.warning(f"Gemini safety block - raw ratings: {safety_ratings}")
                # Get explicitly blocked categories
                blocked_categories = [
                    f"{r.category}: {r.probability}"
                    for r in safety_ratings
                    if hasattr(r, "blocked") and r.blocked
                ]
                # If no blocked categories, show all ratings for debugging
                if not blocked_categories:
                    all_ratings = [
                        f"{r.category}: {r.probability}"
                        for r in safety_ratings
                        if hasattr(r, "category") and hasattr(r, "probability")
                    ]
                    details = all_ratings or ["no details available"]
                else:
                    details = blocked_categories
                logger.error(f"Gemini safety filter triggered: {details}")
                raise ProviderError(
                    message=(
                        f"Response blocked by Gemini safety filters. "
                        f"Categories: {details}. "
                        f"This is a Google API restriction - try rephrasing your request."
                    ),
                    provider=self.name,
                )
            elif "RECITATION" in finish_reason_str:
                raise ProviderError(
                    message="Response blocked due to potential recitation of copyrighted content",
                    provider=self.name,
                )

        # Extract text content and function calls from parts
        content = ""
        tool_calls: list[Any] = []

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # Extract text content
                if hasattr(part, "text") and part.text:
                    content += part.text
                # Extract function calls
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    # Convert args to dict
                    args = {}
                    if hasattr(fc, "args") and fc.args:
                        try:
                            # In new SDK, args may already be a dict or need conversion
                            if isinstance(fc.args, dict):
                                args = fc.args
                            else:
                                # Fallback: try to convert from protobuf Struct
                                from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]

                                args = MessageToDict(fc.args)
                        except Exception:
                            # Fallback: try direct dict access
                            args = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        {
                            "name": fc.name,
                            "arguments": args,
                            "id": f"gemini-{fc.name}-{len(tool_calls)}",
                        }
                    )
                    logger.debug(f"Extracted Gemini function call: {fc.name}({args})")

        # If no content from parts, try response.text as fallback
        if not content:
            try:
                if hasattr(response, "text") and response.text:
                    content = response.text
            except (ValueError, AttributeError):
                # response.text raises ValueError if no valid parts
                pass

        if tool_calls:
            logger.info(f"Gemini returned {len(tool_calls)} native function call(s)")

        # Parse usage (if available)
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
            }

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=model,
            stop_reason=None,
            raw_response=response,
            metadata=None,
        )

    def _parse_stream_chunk(self, chunk: Any) -> Optional[StreamChunk]:
        """Parse streaming chunk from Google.

        Args:
            chunk: Stream chunk

        Returns:
            StreamChunk or None

        Raises:
            ProviderError: If chunk was blocked by safety filters
        """
        # Check for safety blocks in streaming
        if hasattr(chunk, "candidates") and chunk.candidates:
            candidate = chunk.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            finish_reason_str = str(finish_reason) if finish_reason else ""
            if "SAFETY" in finish_reason_str:
                raise ProviderError(
                    message="Streaming response blocked by safety filters",
                    provider=self.name,
                )

        # Extract content safely
        content = ""
        try:
            if hasattr(chunk, "text") and chunk.text:
                content = chunk.text
        except (ValueError, AttributeError):
            # Ignore ValueError when text is not available
            pass

        # Check if this is the final chunk
        is_final = (
            hasattr(chunk, "candidates")
            and chunk.candidates
            and hasattr(chunk.candidates[0], "finish_reason")
            and chunk.candidates[0].finish_reason is not None
        )

        return StreamChunk(
            content=content,
            is_final=is_final,
        )

    async def list_models(self) -> list[dict[str, Any]]:
        """List available Google Gemini models.

        Queries the Google API to list available generative models.

        Returns:
            List of available models with metadata

        Raises:
            ProviderError: If request fails
        """
        try:
            # Use client to list models (new SDK pattern)
            models = []
            for model in self.client.models.list():
                # Filter to generative models that support generateContent
                supported_methods = getattr(model, "supported_generation_methods", [])
                if "generateContent" in supported_methods:
                    model_name = getattr(model, "name", "")
                    models.append(
                        {
                            "id": model_name.replace("models/", ""),
                            "name": getattr(model, "display_name", model_name),
                            "description": getattr(model, "description", ""),
                            "input_token_limit": getattr(model, "input_token_limit", 0),
                            "output_token_limit": getattr(model, "output_token_limit", 0),
                            "supported_methods": supported_methods,
                        }
                    )
            # Sort by name for consistent output
            models.sort(key=lambda x: x["id"])
            return models
        except Exception as e:
            raise ProviderError(
                message=f"Failed to list models: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    async def close(self) -> None:
        """Close connections (Gemini client doesn't need explicit closing)."""
        pass
