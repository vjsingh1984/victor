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

"""Google Gemini provider implementation."""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import (
    FunctionDeclaration,
    GenerateContentResponse,
    HarmBlockThreshold,
    HarmCategory,
    Tool,
)

# Try to import ToolConfig for explicit function calling mode
try:
    from google.generativeai.types import ToolConfig

    HAS_TOOL_CONFIG = True
except ImportError:
    HAS_TOOL_CONFIG = False
    ToolConfig = None

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    StreamChunk,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

# Safety threshold levels (from most to least restrictive)
# Note: BLOCK_NONE is known to be unreliable - Google's internal filters may override it
# See: https://discuss.ai.google.dev/t/safety-settings-2025-update-broken-again/59360
SAFETY_LEVELS = {
    "block_none": HarmBlockThreshold.BLOCK_NONE,
    "off": (
        HarmBlockThreshold.OFF
        if hasattr(HarmBlockThreshold, "OFF")
        else HarmBlockThreshold.BLOCK_NONE
    ),
    "block_few": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    "block_some": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    "block_most": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}


class GoogleProvider(BaseProvider):
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
        api_key: str,
        timeout: int = 60,
        safety_level: str = "block_none",
        **kwargs: Any,
    ):
        """Initialize Google provider.

        Args:
            api_key: Google API key
            timeout: Request timeout in seconds
            safety_level: Safety filter level - "block_none", "block_few",
                         "block_some", or "block_most" (default: "block_none")
            **kwargs: Additional configuration
        """
        super().__init__(api_key=api_key, timeout=timeout, **kwargs)
        genai.configure(api_key=api_key)

        # Configure safety settings for all known categories
        # Some categories may not exist in all SDK versions
        threshold = SAFETY_LEVELS.get(safety_level, HarmBlockThreshold.BLOCK_NONE)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: threshold,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: threshold,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: threshold,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold,
        }
        # Add CIVIC_INTEGRITY if available (newer SDK versions)
        if hasattr(HarmCategory, "HARM_CATEGORY_CIVIC_INTEGRITY"):
            self.safety_settings[HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY] = threshold
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
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
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
            # Convert tools to Gemini function_declarations format
            gemini_tools = self._convert_tools(tools) if tools else None

            # Build tool_config to explicitly enable function calling mode (AUTO)
            tool_config = None
            if gemini_tools and HAS_TOOL_CONFIG and ToolConfig:
                try:
                    # AUTO mode: model decides when to use functions
                    # ANY mode: model must use a function
                    # NONE mode: model cannot use functions
                    tool_config = ToolConfig(function_calling_config={"mode": "AUTO"})
                    logger.debug("Enabled Gemini function calling with mode=AUTO")
                except Exception as e:
                    logger.warning(f"Failed to create ToolConfig: {e}")

            # Initialize model with safety settings and tools
            model_kwargs = {
                "model_name": model,
                "generation_config": {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
                "safety_settings": self.safety_settings,
                "tools": gemini_tools,
            }
            if tool_config:
                model_kwargs["tool_config"] = tool_config

            model_instance = genai.GenerativeModel(**model_kwargs)

            # Convert messages to Gemini format
            history, latest_message = self._convert_messages(messages)

            logger.debug(
                f"Gemini chat: model={model}, tools={len(tools) if tools else 0}, history={len(history)}"
            )

            # Start chat
            chat = model_instance.start_chat(history=history)

            # Generate response with circuit breaker protection
            response = await self._execute_with_circuit_breaker(
                chat.send_message_async, latest_message
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
            raise ProviderError(
                message=f"Google API error: {str(e)}",
                provider=self.name,
                raw_error=e,
            )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
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
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
                safety_settings=self.safety_settings,
            )

            # Convert messages
            history, latest_message = self._convert_messages(messages)

            logger.debug(f"Gemini stream (no tools): model={model}, history={len(history)}")

            # Start chat
            chat = model_instance.start_chat(history=history)

            # Stream response
            response_stream = await chat.send_message_async(
                latest_message,
                stream=True,
            )

            async for chunk in response_stream:
                parsed_chunk = self._parse_stream_chunk(chunk)
                if parsed_chunk:
                    yield parsed_chunk

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                message=f"Google streaming error: {str(e)}",
                provider=self.name,
                raw_error=e,
            )

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Tool]:
        """Convert tools to Gemini format using proper SDK types.

        Uses google.generativeai.types.FunctionDeclaration and Tool
        for proper native function calling support.

        Args:
            tools: List of ToolDefinition objects

        Returns:
            List containing a Tool object with function declarations
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
            func_decl = FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=params,
            )
            function_declarations.append(func_decl)

        logger.debug(
            f"Converted {len(function_declarations)} tools to Gemini FunctionDeclaration format"
        )
        # Wrap in Tool object - this is the proper way to pass tools to GenerativeModel
        return [Tool(function_declarations=function_declarations)]

    def _clean_schema_for_gemini(self, schema: Dict[str, Any]) -> Dict[str, Any]:
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

    def _convert_messages(self, messages: List[Message]) -> tuple[List[Dict[str, str]], str]:
        """Convert messages to Gemini format.

        Args:
            messages: Standard messages

        Returns:
            Tuple of (history, latest_message)
        """
        history = []
        latest_message = ""

        for i, msg in enumerate(messages):
            # Gemini uses "user" and "model" roles
            role = "model" if msg.role == "assistant" else "user"

            if i == len(messages) - 1:
                # Last message is sent separately
                latest_message = msg.content
            else:
                history.append(
                    {
                        "role": role,
                        "parts": [msg.content],
                    }
                )

        return history, latest_message

    def _parse_response(self, response: GenerateContentResponse, model: str) -> CompletionResponse:
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
            # finish_reason values: 1=STOP, 2=SAFETY, 3=MAX_TOKENS, 4=RECITATION, 5=OTHER, 12=?
            finish_reason_map = {
                1: "STOP",
                2: "SAFETY",
                3: "MAX_TOKENS",
                4: "RECITATION",
                5: "OTHER",
            }
            finish_name = finish_reason_map.get(finish_reason, f"UNKNOWN({finish_reason})")
            logger.debug(f"Gemini response finish_reason: {finish_name}")
            if finish_reason == 2:  # SAFETY
                safety_ratings = getattr(candidate, "safety_ratings", [])
                logger.warning(f"Gemini safety block - raw ratings: {safety_ratings}")
                # Get explicitly blocked categories
                blocked_categories = [
                    f"{r.category.name}: {r.probability.name}"
                    for r in safety_ratings
                    if hasattr(r, "blocked") and r.blocked
                ]
                # If no blocked categories, show all ratings for debugging
                if not blocked_categories:
                    all_ratings = [
                        f"{r.category.name}: {r.probability.name}"
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
            elif finish_reason == 4:  # RECITATION
                raise ProviderError(
                    message="Response blocked due to potential recitation of copyrighted content",
                    provider=self.name,
                )

        # Extract text content and function calls from parts
        content = ""
        tool_calls = []

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # Extract text content
                if hasattr(part, "text") and part.text:
                    content += part.text
                # Extract function calls
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    # Convert Struct args to dict
                    args = {}
                    if hasattr(fc, "args") and fc.args:
                        try:
                            # fc.args is a protobuf Struct, convert to dict
                            from google.protobuf.json_format import MessageToDict

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
                if response.text:
                    content = response.text
            except ValueError:
                # response.text raises ValueError if no valid parts
                pass

        if tool_calls:
            logger.info(f"Gemini returned {len(tool_calls)} native function call(s)")

        # Parse usage (if available)
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=model,
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
            if finish_reason == 2:  # SAFETY
                raise ProviderError(
                    message="Streaming response blocked by safety filters",
                    provider=self.name,
                )

        # Extract content safely
        content = ""
        try:
            if hasattr(chunk, "text") and chunk.text:
                content = chunk.text
        except ValueError:
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

    async def close(self) -> None:
        """Close connections (Gemini client doesn't need explicit closing)."""
        pass
