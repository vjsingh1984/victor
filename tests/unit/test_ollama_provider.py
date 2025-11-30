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

"""Tests for Ollama provider."""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.base import (
    Message,
    ProviderError,
    ProviderTimeoutError,
    ToolDefinition,
)
from victor.providers.ollama import OllamaProvider


@pytest.fixture
def ollama_provider():
    """Create OllamaProvider instance for testing."""
    return OllamaProvider(base_url="http://localhost:11434")


@pytest.mark.asyncio
async def test_provider_name(ollama_provider):
    """Test provider name property."""
    assert ollama_provider.name == "ollama"


@pytest.mark.asyncio
async def test_supports_tools(ollama_provider):
    """Test tools support."""
    assert ollama_provider.supports_tools() is True


@pytest.mark.asyncio
async def test_supports_streaming(ollama_provider):
    """Test streaming support."""
    assert ollama_provider.supports_streaming() is True


@pytest.mark.asyncio
async def test_chat_success(ollama_provider):
    """Test successful chat completion."""
    # Mock the HTTP response
    mock_response = {
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you?",
        },
        "done": True,
        "done_reason": "stop",
        "eval_count": 10,
        "prompt_eval_count": 5,
    }

    with patch.object(
        ollama_provider.client,
        "post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Create a mock response object
        # Note: json() is synchronous in httpx, not async
        mock_response_obj = AsyncMock()
        mock_response_obj.json = lambda: mock_response  # Synchronous method
        mock_response_obj.raise_for_status = lambda: None  # Synchronous method
        mock_post.return_value = mock_response_obj

        messages = [Message(role="user", content="Hello")]
        response = await ollama_provider.chat(
            messages=messages,
            model="qwen2.5-coder:7b",
        )

        assert response.content == "Hello! How can I help you?"
        assert response.role == "assistant"
        assert response.model == "qwen2.5-coder:7b"
        assert response.usage is not None
        assert response.usage["completion_tokens"] == 10


@pytest.mark.asyncio
async def test_build_request_payload(ollama_provider):
    """Test request payload building."""
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]

    payload = ollama_provider._build_request_payload(
        messages=messages,
        model="llama3:8b",
        temperature=0.8,
        max_tokens=2048,
        tools=None,
        stream=False,
    )

    assert payload["model"] == "llama3:8b"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.8
    assert payload["options"]["num_predict"] == 2048
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_parse_response(ollama_provider):
    """Test response parsing."""
    raw_response = {
        "message": {
            "role": "assistant",
            "content": "Test response",
        },
        "done": True,
        "done_reason": "stop",
        "eval_count": 20,
        "prompt_eval_count": 10,
    }

    response = ollama_provider._parse_response(raw_response, "test-model")

    assert response.content == "Test response"
    assert response.role == "assistant"
    assert response.model == "test-model"
    assert response.stop_reason == "stop"
    assert response.usage["completion_tokens"] == 20
    assert response.usage["prompt_tokens"] == 10
    assert response.usage["total_tokens"] == 30


@pytest.mark.asyncio
async def test_chat_timeout_error(ollama_provider):
    """Test chat timeout error handling."""
    with patch.object(
        ollama_provider.client,
        "post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderTimeoutError):
            await ollama_provider.chat(messages=messages, model="llama3:8b")


@pytest.mark.asyncio
async def test_chat_http_error(ollama_provider):
    """Test chat HTTP error handling."""
    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch.object(
        ollama_provider.client,
        "post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=MagicMock(),
            response=mock_response,
        )

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError):
            await ollama_provider.chat(messages=messages, model="llama3:8b")


@pytest.mark.asyncio
async def test_chat_generic_error(ollama_provider):
    """Test chat generic error handling."""
    with patch.object(
        ollama_provider.client,
        "post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.side_effect = RuntimeError("Unexpected error")

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError):
            await ollama_provider.chat(messages=messages, model="llama3:8b")


@pytest.mark.asyncio
async def test_chat_with_tools(ollama_provider):
    """Test chat with tool definitions."""
    mock_response = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "San Francisco"},
                    },
                }
            ],
        },
        "done": True,
        "done_reason": "tool_calls",
    }

    with patch.object(
        ollama_provider.client,
        "post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_response_obj = AsyncMock()
        mock_response_obj.json = lambda: mock_response
        mock_response_obj.raise_for_status = lambda: None
        mock_post.return_value = mock_response_obj

        messages = [Message(role="user", content="What's the weather?")]
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather for a location",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            )
        ]

        response = await ollama_provider.chat(
            messages=messages,
            model="llama3:8b",
            tools=tools,
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_stream_basic(ollama_provider):
    """Test basic streaming functionality."""

    async def mock_aiter_lines():
        yield '{"message":{"content":"Hello"},"done":false}'
        yield '{"message":{"content":" world"},"done":false}'
        yield '{"message":{"content":"!"},"done":true,"done_reason":"stop"}'

    mock_response = MagicMock()
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()

    with patch.object(ollama_provider.client, "stream", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]
        chunks = []

        async for chunk in ollama_provider.stream(messages=messages, model="llama3:8b"):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].content == "!"
        assert chunks[2].is_final is True


@pytest.mark.asyncio
async def test_stream_timeout_error(ollama_provider):
    """Test stream timeout error handling."""
    mock_response = MagicMock()
    mock_response.__aenter__ = AsyncMock(side_effect=httpx.TimeoutException("Stream timed out"))
    mock_response.__aexit__ = AsyncMock()

    with patch.object(ollama_provider.client, "stream", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderTimeoutError):
            async for _ in ollama_provider.stream(messages=messages, model="llama3:8b"):
                pass


@pytest.mark.asyncio
async def test_stream_http_error(ollama_provider):
    """Test stream HTTP error handling."""

    async def mock_aenter():
        raise httpx.HTTPStatusError(
            "Server error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

    mock_response = MagicMock()
    mock_response.__aenter__ = mock_aenter
    mock_response.__aexit__ = AsyncMock()

    with patch.object(ollama_provider.client, "stream", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError):
            async for _ in ollama_provider.stream(messages=messages, model="llama3:8b"):
                pass


@pytest.mark.asyncio
async def test_build_payload_with_tools(ollama_provider):
    """Test payload building with tools."""
    tools = [
        ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object"},
        )
    ]

    payload = ollama_provider._build_request_payload(
        messages=[Message(role="user", content="test")],
        model="llama3:8b",
        temperature=0.7,
        max_tokens=1024,
        tools=tools,
        stream=False,
    )

    assert "tools" in payload
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["function"]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_build_payload_with_options(ollama_provider):
    """Test payload building with custom options."""
    payload = ollama_provider._build_request_payload(
        messages=[Message(role="user", content="test")],
        model="llama3:8b",
        temperature=0.7,
        max_tokens=1024,
        tools=None,
        stream=False,
        options={"top_p": 0.9, "top_k": 40},
    )

    assert payload["options"]["top_p"] == 0.9
    assert payload["options"]["top_k"] == 40
    assert payload["options"]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_normalize_tool_calls_openai_format(ollama_provider):
    """Test tool call normalization from OpenAI format."""
    tool_calls = [
        {
            "id": "call_1",
            "function": {
                "name": "get_weather",
                "arguments": {"location": "NYC"},
            },
        }
    ]

    normalized = ollama_provider._normalize_tool_calls(tool_calls)

    assert normalized is not None
    assert len(normalized) == 1
    assert normalized[0]["name"] == "get_weather"
    assert normalized[0]["arguments"] == {"location": "NYC"}


@pytest.mark.asyncio
async def test_normalize_tool_calls_already_normalized(ollama_provider):
    """Test tool call normalization when already normalized."""
    tool_calls = [
        {
            "name": "get_weather",
            "arguments": {"location": "NYC"},
        }
    ]

    normalized = ollama_provider._normalize_tool_calls(tool_calls)

    assert normalized is not None
    assert len(normalized) == 1
    assert normalized[0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_normalize_tool_calls_none(ollama_provider):
    """Test tool call normalization with None."""
    normalized = ollama_provider._normalize_tool_calls(None)
    assert normalized is None


@pytest.mark.asyncio
async def test_normalize_tool_calls_empty(ollama_provider):
    """Test tool call normalization with empty list."""
    normalized = ollama_provider._normalize_tool_calls([])
    assert normalized is None
