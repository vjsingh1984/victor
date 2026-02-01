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

"""Comprehensive tests for ZAI provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from victor.providers.zai_provider import ZAIProvider
from victor.core.errors import (
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
)
from victor.providers.base import (
    Message,
    ToolDefinition,
)


@pytest.fixture
def zai_provider():
    """Create ZAIProvider instance for testing."""
    return ZAIProvider(
        api_key="test-api-key",
        base_url="https://api.z.ai/api/paas/v4/",
        timeout=30,
        max_retries=2,
    )


@pytest.mark.asyncio
async def test_initialization():
    """Test provider initialization."""
    provider = ZAIProvider(
        api_key="test-key",
        base_url="https://custom.url",
        timeout=45,
        max_retries=5,
    )

    assert provider.api_key == "test-key"
    assert provider.base_url == "https://custom.url"
    assert provider.timeout == 45
    assert provider.max_retries == 5
    assert provider.client is not None


@pytest.mark.asyncio
async def test_initialization_defaults():
    """Test provider initialization with defaults."""
    provider = ZAIProvider(api_key="test-key")

    assert provider.api_key == "test-key"
    assert provider.base_url == "https://api.z.ai/api/paas/v4/"
    assert provider.client is not None


@pytest.mark.asyncio
async def test_provider_name(zai_provider):
    """Test provider name property."""
    assert zai_provider.name == "zai"


@pytest.mark.asyncio
async def test_supports_tools(zai_provider):
    """Test tools support."""
    assert zai_provider.supports_tools() is True


@pytest.mark.asyncio
async def test_supports_streaming(zai_provider):
    """Test streaming support."""
    assert zai_provider.supports_streaming() is True


@pytest.mark.asyncio
async def test_get_context_window(zai_provider):
    """Test context window retrieval."""
    assert zai_provider.get_context_window("glm-4.7") == 128000
    assert zai_provider.get_context_window("glm-4.6") == 128000
    assert zai_provider.get_context_window("glm-4.5") == 128000
    assert zai_provider.get_context_window("glm-4.5-air") == 128000
    # Default for unknown models
    assert zai_provider.get_context_window("glm-unknown") == 128000


@pytest.mark.asyncio
async def test_chat_server_error(zai_provider):
    """Test server error handling."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"
    mock_response.json.return_value = {"error": {"message": "Internal server error"}}

    # Mock raise_for_status to raise HTTPStatusError
    def raise_status_error():
        raise httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response)

    mock_response.raise_for_status = raise_status_error

    with patch.object(
        zai_provider.client, "post", new_callable=AsyncMock, return_value=mock_response
    ):
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError):
            await zai_provider.chat(messages=messages, model="glm-4.7")


@pytest.mark.asyncio
async def test_chat_network_error(zai_provider):
    """Test network error handling."""
    with patch.object(
        zai_provider.client,
        "post",
        side_effect=httpx.NetworkError("Connection failed"),
    ):
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError):
            await zai_provider.chat(messages=messages, model="glm-4.7")


@pytest.mark.asyncio
async def test_chat_with_system_message(zai_provider):
    """Test chat with system message."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    with patch.object(zai_provider.client, "post", return_value=mock_response):
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]
        response = await zai_provider.chat(messages=messages, model="glm-4.7")

        assert response.content == "Response"


@pytest.mark.asyncio
async def test_chat_with_max_tokens(zai_provider):
    """Test chat with max_tokens parameter."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Short"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }

    with patch.object(zai_provider.client, "post", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]
        response = await zai_provider.chat(messages=messages, model="glm-4.7", max_tokens=10)

        assert response.content == "Short"
        assert response.stop_reason == "length"


@pytest.mark.asyncio
async def test_chat_with_tools(zai_provider):
    """Test chat with tool definitions."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Beijing"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }

    with patch.object(zai_provider.client, "post", return_value=mock_response):
        messages = [Message(role="user", content="What's the weather?")]
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather for a location",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            )
        ]
        response = await zai_provider.chat(messages=messages, model="glm-4.7", tools=tools)

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tool_calls[0]["id"] == "call_123"


@pytest.mark.asyncio
async def test_chat_authentication_error(zai_provider):
    """Test authentication error handling."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Invalid API key"

    error = httpx.HTTPStatusError("Auth failed", request=MagicMock(), response=mock_response)

    with patch.object(zai_provider.client, "post", side_effect=error):
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderAuthError):
            await zai_provider.chat(messages=messages, model="glm-4.7")


@pytest.mark.asyncio
async def test_chat_rate_limit_error(zai_provider):
    """Test rate limit error handling."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Rate limit exceeded"

    error = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=mock_response)

    with patch.object(zai_provider.client, "post", side_effect=error):
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderRateLimitError):
            await zai_provider.chat(messages=messages, model="glm-4.7")


@pytest.mark.asyncio
async def test_stream_basic(zai_provider):
    """Test basic streaming functionality."""

    # Mock streaming response
    async def mock_aiter_lines():
        yield "data: " + '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}'
        yield "data: " + '{"choices":[{"delta":{"content":" world"},"finish_reason":null}]}'
        yield "data: " + '{"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}'
        yield "data: [DONE]"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()

    with patch.object(zai_provider.client, "stream", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]
        chunks = []

        async for chunk in zai_provider.stream(messages=messages, model="glm-4.7"):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].content == "!"
        assert chunks[2].is_final is True


@pytest.mark.asyncio
async def test_stream_with_tools(zai_provider):
    """Test streaming with tools."""

    async def mock_aiter_lines():
        yield "data: " + '{"choices":[{"delta":{"content":"Using tool"},"finish_reason":null}]}'
        yield "data: [DONE]"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()

    with patch.object(zai_provider.client, "stream", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]
        tools = [
            ToolDefinition(
                name="test_tool",
                description="Test tool",
                parameters={"type": "object"},
            )
        ]

        chunks = []
        async for chunk in zai_provider.stream(messages=messages, model="glm-4.7", tools=tools):
            chunks.append(chunk)

        # Should get 2 chunks: content chunk + final [DONE] chunk
        assert len(chunks) == 2
        assert chunks[0].content == "Using tool"
        assert chunks[0].is_final is False
        assert chunks[1].content == ""
        assert chunks[1].is_final is True


@pytest.mark.asyncio
async def test_stream_error(zai_provider):
    """Test streaming error handling."""

    # Create a mock that raises error when __aenter__ is called (entering context)
    async def mock_aenter():
        raise httpx.HTTPStatusError(
            "Server error",
            request=MagicMock(),
            response=MagicMock(status_code=500, text="Server error"),
        )

    mock_response = MagicMock()
    mock_response.__aenter__ = mock_aenter
    mock_response.__aexit__ = AsyncMock()

    with patch.object(zai_provider.client, "stream", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError):
            async for _ in zai_provider.stream(messages=messages, model="glm-4.7"):
                pass


@pytest.mark.asyncio
async def test_thinking_mode_disabled_by_default(zai_provider):
    """Test that thinking mode is disabled by default."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    # Capture the actual request payload
    actual_payload = None

    async def mock_post(*args, **kwargs):
        nonlocal actual_payload
        actual_payload = kwargs.get("json", {})
        return mock_response

    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    with patch.object(zai_provider.client, "post", side_effect=mock_post):
        messages = [Message(role="user", content="Hello")]
        await zai_provider.chat(messages=messages, model="glm-4.7")

        # Verify thinking parameter is NOT in payload by default
        assert "thinking" not in actual_payload


@pytest.mark.asyncio
async def test_thinking_mode_enabled(zai_provider):
    """Test that thinking mode can be enabled."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    # Capture the actual request payload
    actual_payload = None

    async def mock_post(*args, **kwargs):
        nonlocal actual_payload
        actual_payload = kwargs.get("json", {})
        return mock_response

    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    with patch.object(zai_provider.client, "post", side_effect=mock_post):
        messages = [Message(role="user", content="Hello")]
        await zai_provider.chat(messages=messages, model="glm-4.6", thinking=True)

        # Verify thinking parameter is in payload
        assert "thinking" in actual_payload
        assert actual_payload["thinking"]["type"] == "enabled"


@pytest.mark.asyncio
async def test_close(zai_provider):
    """Test closing the provider."""
    with patch.object(zai_provider.client, "aclose", new_callable=AsyncMock) as mock_aclose:
        await zai_provider.close()
        mock_aclose.assert_called_once()
