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

"""Tests for LMStudio provider.

Tests the dedicated LMStudioProvider which uses:
- httpx.AsyncClient (not AsyncOpenAI SDK)
- Tiered URL selection with /v1/models health check
- 300s timeout (matching Ollama)
- OpenAI-compatible API format
"""

import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.base import (
    Message,
    ProviderError,
    ProviderTimeoutError,
    ToolDefinition,
)
from victor.providers.lmstudio_provider import LMStudioProvider


@pytest.fixture
def lmstudio_provider():
    """Create LMStudioProvider instance for testing."""
    return LMStudioProvider(base_url="http://127.0.0.1:1234", _skip_discovery=True)


@pytest.mark.asyncio
async def test_provider_name(lmstudio_provider):
    """Test provider name property."""
    assert lmstudio_provider.name == "lmstudio"


@pytest.mark.asyncio
async def test_supports_tools(lmstudio_provider):
    """Test tools support."""
    assert lmstudio_provider.supports_tools() is True


@pytest.mark.asyncio
async def test_supports_streaming(lmstudio_provider):
    """Test streaming support."""
    assert lmstudio_provider.supports_streaming() is True


@pytest.mark.asyncio
async def test_default_timeout():
    """Test that default timeout is 300s (matching Ollama)."""
    assert LMStudioProvider.DEFAULT_TIMEOUT == 300


@pytest.mark.asyncio
async def test_default_port():
    """Test that default port is 1234."""
    assert LMStudioProvider.DEFAULT_PORT == 1234


@pytest.mark.asyncio
async def test_chat_success(lmstudio_provider):
    """Test successful chat completion."""
    # Mock the HTTP response (OpenAI-compatible format)
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen3-coder-30b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }

    with patch.object(
        lmstudio_provider.client,
        "post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_response_obj = AsyncMock()
        mock_response_obj.json = lambda: mock_response
        mock_response_obj.raise_for_status = lambda: None
        mock_post.return_value = mock_response_obj

        messages = [Message(role="user", content="Hello")]
        response = await lmstudio_provider.chat(
            messages=messages,
            model="qwen3-coder-30b",
        )

        assert response.content == "Hello! How can I help you?"
        assert response.role == "assistant"
        assert response.model == "qwen3-coder-30b"
        assert response.usage is not None
        assert response.usage["completion_tokens"] == 8
        assert response.usage["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_build_request_payload(lmstudio_provider):
    """Test request payload building (OpenAI format)."""
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]

    payload = lmstudio_provider._build_request_payload(
        messages=messages,
        model="llama-3.1-8b",
        temperature=0.8,
        max_tokens=2048,
        tools=None,
        stream=False,
    )

    assert payload["model"] == "llama-3.1-8b"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.8
    assert payload["max_tokens"] == 2048
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_build_request_payload_with_tools(lmstudio_provider):
    """Test request payload building with tools."""
    messages = [Message(role="user", content="List files")]
    tools = [
        ToolDefinition(
            name="list_directory",
            description="List directory contents",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
    ]

    payload = lmstudio_provider._build_request_payload(
        messages=messages,
        model="qwen3-coder-30b",
        temperature=0.7,
        max_tokens=4096,
        tools=tools,
        stream=False,
    )

    assert "tools" in payload
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["function"]["name"] == "list_directory"
    assert payload["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_parse_response(lmstudio_provider):
    """Test response parsing (OpenAI format)."""
    raw_response = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    response = lmstudio_provider._parse_response(raw_response, "test-model")

    assert response.content == "Test response"
    assert response.role == "assistant"
    assert response.model == "test-model"
    assert response.stop_reason == "stop"
    assert response.usage["completion_tokens"] == 5
    assert response.usage["prompt_tokens"] == 10
    assert response.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_parse_response_with_tool_calls(lmstudio_provider):
    """Test response parsing with native tool calls."""
    raw_response = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "/etc/hosts"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    response = lmstudio_provider._parse_response(raw_response, "qwen3-coder-30b")

    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "read_file"
    assert response.tool_calls[0]["arguments"] == {"path": "/etc/hosts"}


@pytest.mark.asyncio
async def test_parse_json_tool_call_from_content(lmstudio_provider):
    """Test fallback JSON tool call parsing from content."""
    content = '{"name": "list_directory", "arguments": {"path": "/tmp"}}'
    result = lmstudio_provider._parse_json_tool_call_from_content(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "list_directory"
    assert result[0]["arguments"] == {"path": "/tmp"}


@pytest.mark.asyncio
async def test_parse_tool_request_format(lmstudio_provider):
    """Test [TOOL_REQUEST] format parsing."""
    content = '[TOOL_REQUEST]{"name": "read_file", "arguments": {"path": "/etc/hosts"}}[END_TOOL_REQUEST]'
    result = lmstudio_provider._parse_json_tool_call_from_content(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "read_file"
    assert result[0]["arguments"] == {"path": "/etc/hosts"}


@pytest.mark.asyncio
async def test_normalize_tool_calls(lmstudio_provider):
    """Test tool call normalization from OpenAI format."""
    raw_tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "/tmp/test.txt"}',
            },
        }
    ]

    normalized = lmstudio_provider._normalize_tool_calls(raw_tool_calls)

    assert normalized is not None
    assert len(normalized) == 1
    assert normalized[0]["name"] == "read_file"
    assert normalized[0]["arguments"] == {"path": "/tmp/test.txt"}


@pytest.mark.asyncio
async def test_chat_timeout_error(lmstudio_provider):
    """Test chat timeout error handling."""
    with patch.object(
        lmstudio_provider.client,
        "post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderTimeoutError) as exc_info:
            await lmstudio_provider.chat(
                messages=messages,
                model="qwen3-coder-30b",
            )

        assert "timed out" in str(exc_info.value).lower()
        assert exc_info.value.provider == "lmstudio"


@pytest.mark.asyncio
async def test_chat_http_error(lmstudio_provider):
    """Test chat HTTP error handling."""
    with patch.object(
        lmstudio_provider.client,
        "post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Create a proper HTTPStatusError
        request = httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")
        response = httpx.Response(500, text="Internal Server Error", request=request)
        mock_post.side_effect = httpx.HTTPStatusError(
            "Server error", request=request, response=response
        )

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError) as exc_info:
            await lmstudio_provider.chat(
                messages=messages,
                model="qwen3-coder-30b",
            )

        assert "500" in str(exc_info.value)
        assert exc_info.value.provider == "lmstudio"


@pytest.mark.asyncio
async def test_list_models(lmstudio_provider):
    """Test model listing."""
    mock_response = {
        "object": "list",
        "data": [
            {"id": "qwen3-coder-30b", "object": "model"},
            {"id": "llama-3.1-8b", "object": "model"},
        ],
    }

    with patch.object(
        lmstudio_provider.client,
        "get",
        new_callable=AsyncMock,
    ) as mock_get:
        mock_response_obj = AsyncMock()
        mock_response_obj.json = lambda: mock_response
        mock_response_obj.raise_for_status = lambda: None
        mock_get.return_value = mock_response_obj

        models = await lmstudio_provider.list_models()

        assert len(models) == 2
        assert models[0]["id"] == "qwen3-coder-30b"


def test_tiered_url_selection_single():
    """Test URL selection with single URL."""
    with patch("httpx.Client") as mock_client:
        # Mock successful health check
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"id": "model"}]}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance

        provider = LMStudioProvider(base_url="http://192.168.1.20:1234")
        assert "192.168.1.20" in provider.base_url


def test_tiered_url_selection_list():
    """Test URL selection with list of URLs (tiered fallback)."""
    with patch("httpx.Client") as mock_client:
        call_count = 0

        def mock_get(url):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First URL fails
                raise httpx.ConnectError("Connection refused")
            # Second URL succeeds
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [{"id": "model"}]}
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.side_effect = mock_get
        mock_client.return_value = mock_client_instance

        provider = LMStudioProvider(
            base_url=["http://192.168.1.20:1234", "http://127.0.0.1:1234"]
        )
        # Should fall back to second URL
        assert "127.0.0.1" in provider.base_url


def test_env_var_url_override():
    """Test LMSTUDIO_ENDPOINTS environment variable override."""
    import os

    with patch.dict(os.environ, {"LMSTUDIO_ENDPOINTS": "http://custom:1234"}):
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [{"id": "model"}]}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance

            provider = LMStudioProvider(base_url="http://localhost:1234")
            # Should use env var URL, not the provided base_url
            assert "custom" in provider.base_url


@pytest.mark.asyncio
async def test_close(lmstudio_provider):
    """Test client cleanup."""
    with patch.object(
        lmstudio_provider.client,
        "aclose",
        new_callable=AsyncMock,
    ) as mock_close:
        await lmstudio_provider.close()
        mock_close.assert_called_once()


# Registry integration tests
def test_registry_returns_lmstudio_provider():
    """Test that ProviderRegistry returns LMStudioProvider for 'lmstudio'."""
    from victor.providers.registry import ProviderRegistry

    provider_class = ProviderRegistry.get("lmstudio")
    assert provider_class == LMStudioProvider


def test_registry_create_lmstudio():
    """Test creating LMStudio provider via registry."""
    from victor.providers.registry import ProviderRegistry

    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance

        provider = ProviderRegistry.create(
            "lmstudio",
            base_url="http://127.0.0.1:1234",
        )
        assert isinstance(provider, LMStudioProvider)
        assert provider.name == "lmstudio"
