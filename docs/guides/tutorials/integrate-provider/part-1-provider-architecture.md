# Tutorial: Integrating a New LLM Provider - Part 1

**Part 1 of 5:** Provider Architecture Overview and Implementation Steps 1-2

---

## Navigation

- **[Part 1: Architecture & Steps 1-2](#)** (Current)
- [Part 2: Streaming & Error Handling](part-2-streaming-error-handling.md)
- [Part 3: Tool Calling Adapter](part-3-tool-calling-adapter.md)
- [Part 4: Registration & Testing](part-4-registration-testing.md)
- [Part 5: Best Practices & Examples](part-5-best-practices-examples.md)
- [**Complete Guide**](integrate-provider.md)

---

## What You Will Build

A complete LLM provider integration that:
- Connects to any LLM API (cloud or local)
- Supports both synchronous and streaming chat completions
- Handles tool/function calling
- Integrates with Victor's circuit breaker for resilience
- Is properly registered and testable

## Prerequisites

- Python 3.11+
- Victor development environment set up (`pip install -e ".[dev]"`)
- API access to the LLM provider you want to integrate
- Basic understanding of async/await in Python

**Time estimate:** 45-60 minutes

---

## 1. Provider Architecture Overview

### The BaseProvider Class

All Victor providers inherit from `BaseProvider` located at `victor/providers/base.py`. This abstract base class defines the contract that every provider must fulfill.

```python
from victor.providers.base import BaseProvider

class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        use_circuit_breaker: bool = True,
        **kwargs: Any,
    ):
        # Initialization with built-in circuit breaker support
        ...
```

### Required Methods

Every provider must implement these abstract methods:

| Method | Description |
|--------|-------------|
| `name` (property) | Returns the provider identifier (e.g., "anthropic", "openai") |
| `chat()` | Sends a chat completion request and returns a response |
| `stream()` | Streams a chat completion response incrementally |
| `close()` | Closes any open connections or resources |

### Optional Capabilities

Providers can declare additional capabilities by overriding these methods:

| Method | Default | Description |
|--------|---------|-------------|
| `supports_tools()` | `False` | Whether the provider supports tool/function calling |
| `supports_streaming()` | `False` | Whether the provider supports streaming responses |
| `discover_capabilities()` | Config-based | Runtime capability discovery |
| `count_tokens()` | Character estimate | Token counting for the provider |

### Core Data Types

Victor uses standardized data types across all providers:

```python
from victor.providers.base import (
    Message,           # Input message format
    CompletionResponse,# Non-streaming response
    StreamChunk,       # Streaming response chunk
    ToolDefinition,    # Tool schema for function calling
)
```

---

## 2. Step-by-Step Implementation

### Step 1: Create the Provider File

Create a new file at `victor/providers/custom_provider.py`:

```python
# Copyright 2025 Your Name
#
# Licensed under the Apache License, Version 2.0

"""Custom LLM provider implementation.

This provider integrates with CustomLLM's API to provide chat completions,
streaming, and tool calling support.

Features:
- Native tool calling support
- Streaming responses
- Circuit breaker protection

References:
- https://customllm.example.com/docs
"""

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StreamChunk,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

# Default API endpoint
DEFAULT_BASE_URL = "https://api.customllm.example.com/v1"


class CustomLLMProvider(BaseProvider):
    """Provider for CustomLLM API.

    Features:
    - Native tool calling support
    - Streaming responses
    - Circuit breaker protection
    """

    # Default timeout in seconds
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize CustomLLM provider.

        Args:
            api_key: API key (or set CUSTOM_LLM_API_KEY env var)
            base_url: API endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        # Resolve API key: parameter > env var > keyring
        resolved_key = api_key or os.environ.get("CUSTOM_LLM_API_KEY", "")
        if not resolved_key:
            try:
                from victor.config.api_keys import get_api_key
                resolved_key = get_api_key("custom_llm") or ""
            except ImportError:
                pass

        if not resolved_key:
            logger.warning(
                "CustomLLM API key not provided. Set CUSTOM_LLM_API_KEY "
                "environment variable or use 'victor keys --set custom_llm --keyring'"
            )

        # Call parent constructor (sets up circuit breaker)
        super().__init__(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "custom_llm"

    def supports_tools(self) -> bool:
        """Whether this provider supports tool/function calling."""
        return True

    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses."""
        return True
```

### Step 2: Implement the chat() Method

The `chat()` method sends a non-streaming request and returns a complete response:

```python
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
        """Send chat completion request.

        Args:
            messages: Conversation messages
            model: Model identifier (e.g., "custom-llm-large")
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Available tools for function calling
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If the request fails
        """
        try:
            # Build request payload
            payload = self._build_request_payload(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=False,
                **kwargs,
            )

            # Execute with circuit breaker protection
            response = await self._execute_with_circuit_breaker(
                self.client.post, "/chat/completions", json=payload
            )
            response.raise_for_status()

            # Parse and return response
            result = response.json()
            return self._parse_response(result, model)

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Request timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            return self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"Unexpected error: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    def _build_request_payload(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[ToolDefinition]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request payload for the API.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Available tools
            stream: Whether to stream response
            **kwargs: Additional options

        Returns:
            Request payload dictionary
        """
        # Format messages for the API
        formatted_messages = []
        for msg in messages:
            formatted_msg: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            # Handle tool results if present
            if msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id
            formatted_messages.append(formatted_msg)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        # Add tools in OpenAI-compatible format
        if tools:
            payload["tools"] = [
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
            payload["tool_choice"] = "auto"

        # Merge additional options
        for key, value in kwargs.items():
            if key not in {"api_key"} and value is not None:
                payload[key] = value

        return payload

    def _parse_response(
        self, result: Dict[str, Any], model: str
    ) -> CompletionResponse:
        """Parse API response into CompletionResponse.

        Args:
            result: Raw API response
            model: Model name

        Returns:
            Normalized CompletionResponse
        """
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="",
                role="assistant",
                model=model,
                raw_response=result,
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        # Parse tool calls if present
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))

        # Parse usage statistics
        usage = None
        usage_data = result.get("usage")
        if usage_data:
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
        )

    def _normalize_tool_calls(
        self, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Normalize tool calls from API format.

        Args:
            tool_calls: Raw tool calls from API

        Returns:
            Normalized tool calls list
        """
        if not tool_calls:
            return None

        normalized = []
        for call in tool_calls:
            if isinstance(call, dict) and "function" in call:
                function = call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments", "{}")

                # Parse JSON arguments if they're a string
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                if name:
                    normalized.append({
                        "id": call.get("id", ""),
                        "name": name,
                        "arguments": arguments,
                    })

        return normalized if normalized else None
```

---

**Continue to [Part 2: Streaming & Error Handling](part-2-streaming-error-handling.md)**

---

**Last Updated:** February 01, 2026
**Part 1 of 5**
