"""xAI Grok provider implementation."""

from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    StreamChunk,
    ToolDefinition,
)


class XAIProvider(BaseProvider):
    """Provider for xAI Grok models.

    xAI uses an OpenAI-compatible API, so this provider follows similar patterns.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize xAI provider.

        Args:
            api_key: xAI API key
            base_url: xAI API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries, **kwargs)
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "xai"

    def supports_tools(self) -> bool:
        """xAI Grok supports function calling."""
        return True

    def supports_streaming(self) -> bool:
        """xAI supports streaming."""
        return True

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str = "grok-beta",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request to xAI.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "grok-beta", "grok-vision-beta")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional xAI parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If request fails
        """
        try:
            # Convert messages to API format
            api_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            # Build request payload
            payload = {
                "model": model,
                "messages": api_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
                **kwargs,
            }

            if tools:
                payload["tools"] = self._convert_tools(tools)
                payload["tool_choice"] = "auto"

            # Make API call
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()

            result = response.json()
            return self._parse_response(result, model)

        except httpx.HTTPStatusError as e:
            return self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"xAI API error: {str(e)}",
                provider=self.name,
                raw_error=e,
            )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str = "grok-beta",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from xAI.

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
        try:
            # Convert messages
            api_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            # Build request payload
            payload = {
                "model": model,
                "messages": api_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                **kwargs,
            }

            if tools:
                payload["tools"] = self._convert_tools(tools)
                payload["tool_choice"] = "auto"

            # Stream response
            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip() or line.strip() == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        import json
                        try:
                            chunk_data = json.loads(line[6:])
                            chunk = self._parse_stream_chunk(chunk_data)
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"xAI streaming error: {str(e)}",
                provider=self.name,
                raw_error=e,
            )

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert standard tools to xAI format (OpenAI-compatible).

        Args:
            tools: Standard tool definitions

        Returns:
            xAI-formatted tools
        """
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

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse xAI API response.

        Args:
            result: Raw API response
            model: Model name

        Returns:
            Normalized CompletionResponse
        """
        choice = result["choices"][0]
        message = choice["message"]

        # Extract tool calls
        tool_calls = None
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = [
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
                for tc in message["tool_calls"]
            ]

        # Parse usage
        usage = None
        if "usage" in result:
            usage = {
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "total_tokens": result["usage"]["total_tokens"],
            }

        return CompletionResponse(
            content=message.get("content") or "",
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
        )

    def _parse_stream_chunk(self, chunk_data: Dict[str, Any]) -> Optional[StreamChunk]:
        """Parse streaming chunk from xAI.

        Args:
            chunk_data: Stream chunk data

        Returns:
            StreamChunk or None
        """
        if not chunk_data.get("choices"):
            return None

        choice = chunk_data["choices"][0]
        delta = choice.get("delta", {})

        content = delta.get("content", "")
        is_final = choice.get("finish_reason") is not None

        return StreamChunk(
            content=content,
            stop_reason=choice.get("finish_reason"),
            is_final=is_final,
        )

    def _handle_http_error(self, error: httpx.HTTPStatusError) -> ProviderError:
        """Handle HTTP errors from xAI API.

        Args:
            error: HTTP error

        Raises:
            ProviderError: Converted error
        """
        status_code = error.response.status_code
        error_msg = error.response.text

        if status_code == 401:
            raise ProviderAuthenticationError(
                message=f"Authentication failed: {error_msg}",
                provider=self.name,
                status_code=status_code,
                raw_error=error,
            )
        elif status_code == 429:
            raise ProviderRateLimitError(
                message=f"Rate limit exceeded: {error_msg}",
                provider=self.name,
                status_code=status_code,
                raw_error=error,
            )
        else:
            raise ProviderError(
                message=f"xAI API error ({status_code}): {error_msg}",
                provider=self.name,
                status_code=status_code,
                raw_error=error,
            )

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
