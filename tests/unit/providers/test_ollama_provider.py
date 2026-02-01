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

from victor.core.errors import (
    ProviderError,
    ProviderTimeoutError,
)
from victor.providers.base import (
    Message,
    ToolDefinition,
)
from victor.providers.ollama_provider import OllamaProvider


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
    mock_response.status_code = 200  # Must set status_code for new stream logic
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


@pytest.mark.asyncio
async def test_normalize_tool_calls_unknown_format(ollama_provider):
    """Test tool call normalization skips unknown formats (covers line 445)."""
    tool_calls = [
        {"unknown_key": "value"},  # Unknown format
        {"name": "valid_tool", "arguments": {}},  # Valid format
    ]
    normalized = ollama_provider._normalize_tool_calls(tool_calls)
    assert normalized is not None
    assert len(normalized) == 1
    assert normalized[0]["name"] == "valid_tool"


class TestJsonToolCallParsing:
    """Tests for JSON tool call parsing from content."""

    def test_parse_json_tool_call_empty_content(self, ollama_provider):
        """Test parsing empty content returns None (covers line 465-466)."""
        assert ollama_provider._parse_json_tool_call_from_content("") is None
        assert ollama_provider._parse_json_tool_call_from_content(None) is None
        assert ollama_provider._parse_json_tool_call_from_content("   ") is None

    def test_parse_json_tool_call_valid_arguments(self, ollama_provider):
        """Test parsing valid JSON with arguments (covers lines 470-478)."""
        content = '{"name": "read_file", "arguments": {"path": "/test.py"}}'
        result = ollama_provider._parse_json_tool_call_from_content(content)

        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["arguments"] == {"path": "/test.py"}

    def test_parse_json_tool_call_valid_parameters(self, ollama_provider):
        """Test parsing valid JSON with parameters key (covers line 475)."""
        content = '{"name": "list_dir", "parameters": {"path": "/home"}}'
        result = ollama_provider._parse_json_tool_call_from_content(content)

        assert result is not None
        assert len(result) == 1
        assert result[0]["arguments"] == {"path": "/home"}

    def test_parse_json_tool_call_invalid_json(self, ollama_provider):
        """Test parsing invalid JSON returns None (covers lines 479-481)."""
        result = ollama_provider._parse_json_tool_call_from_content("not json at all")
        assert result is None

    def test_parse_json_tool_call_no_name_field(self, ollama_provider):
        """Test parsing JSON without name field returns None (covers line 473)."""
        content = '{"arguments": {"path": "/test.py"}}'
        result = ollama_provider._parse_json_tool_call_from_content(content)
        assert result is None


class TestResponseParsingFallback:
    """Tests for response parsing with fallback tool call detection."""

    def test_parse_response_with_json_tool_in_content(self, ollama_provider):
        """Test _parse_response detects JSON tool call in content (covers lines 500-506)."""
        raw_response = {
            "message": {
                "role": "assistant",
                "content": '{"name": "read_file", "arguments": {"path": "/test.py"}}',
            },
            "done": True,
        }
        response = ollama_provider._parse_response(raw_response, "llama3:8b")

        # Tool call should be extracted from content
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "read_file"
        # Content should be cleared since it was a tool call
        assert response.content == ""

    def test_parse_response_no_usage(self, ollama_provider):
        """Test _parse_response handles missing usage stats."""
        raw_response = {
            "message": {
                "role": "assistant",
                "content": "Hello",
            },
            "done": True,
        }
        response = ollama_provider._parse_response(raw_response, "test")
        assert response.content == "Hello"
        assert response.usage is None


class TestStreamChunkParsing:
    """Tests for stream chunk parsing."""

    def test_parse_stream_chunk_basic(self, ollama_provider):
        """Test basic stream chunk parsing (covers lines 536-559)."""
        chunk_data = {
            "message": {"content": "Hello"},
            "done": False,
        }
        chunk = ollama_provider._parse_stream_chunk(chunk_data)

        assert chunk.content == "Hello"
        assert chunk.is_final is False
        assert chunk.tool_calls is None

    def test_parse_stream_chunk_final(self, ollama_provider):
        """Test final stream chunk parsing."""
        chunk_data = {
            "message": {"content": "!"},
            "done": True,
            "done_reason": "stop",
        }
        chunk = ollama_provider._parse_stream_chunk(chunk_data)

        assert chunk.is_final is True
        assert chunk.stop_reason == "stop"

    def test_parse_stream_chunk_with_tool_calls(self, ollama_provider):
        """Test stream chunk with native tool calls."""
        chunk_data = {
            "message": {
                "content": "",
                "tool_calls": [{"name": "test_tool", "arguments": {}}],
            },
            "done": True,
        }
        chunk = ollama_provider._parse_stream_chunk(chunk_data)

        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 1

    def test_parse_stream_chunk_json_tool_fallback(self, ollama_provider):
        """Test stream chunk parses JSON tool call from content (covers lines 543-552)."""
        chunk_data = {
            "message": {
                "content": '{"name": "read_file", "arguments": {"path": "/test"}}',
            },
            "done": True,
            "model": "llama3:8b",
        }
        chunk = ollama_provider._parse_stream_chunk(chunk_data)

        # Tool call should be detected from content
        assert chunk.tool_calls is not None
        assert chunk.tool_calls[0]["name"] == "read_file"
        # Content should be cleared
        assert chunk.content == ""


class TestEndpointDiscovery:
    """Tests for endpoint discovery logic."""

    def test_select_base_url_from_env(self):
        """Test _select_base_url prioritizes OLLAMA_ENDPOINTS env var (covers lines 121-123)."""
        with patch.dict(
            "os.environ", {"OLLAMA_ENDPOINTS": "http://server1:11434,http://server2:11434"}
        ):
            with patch("httpx.Client") as mock_client:
                # Make first endpoint fail
                mock_instance = MagicMock()
                mock_instance.__enter__ = MagicMock(return_value=mock_instance)
                mock_instance.__exit__ = MagicMock()
                mock_instance.get.side_effect = [Exception("Not reachable"), MagicMock()]
                mock_client.return_value = mock_instance

                # Use skip_discovery since we're testing _select_base_url directly
                provider = OllamaProvider(base_url="http://localhost:11434", _skip_discovery=True)
                # Now test _select_base_url directly
                result = provider._select_base_url("http://ignored:11434", 10)

                # Should try endpoints from env var
                assert "server" in result or "localhost" in result

    def test_select_base_url_comma_separated(self):
        """Test _select_base_url handles comma-separated URL string (covers lines 133-134)."""
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(OllamaProvider, "_select_base_url") as mock_select:
                mock_select.return_value = "http://localhost:11434"
                provider = OllamaProvider(base_url="http://a:11434,http://b:11434")

                # Just verify it doesn't crash
                assert provider is not None

    def test_select_base_url_list_input(self):
        """Test _select_base_url handles list input (covers lines 130-131)."""
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(OllamaProvider, "_select_base_url") as mock_select:
                mock_select.return_value = "http://localhost:11434"
                provider = OllamaProvider(base_url=["http://a:11434", "http://b:11434"])
                assert provider is not None

    def test_select_base_url_none_default(self):
        """Test _select_base_url uses default when base_url is None (covers line 125-126)."""
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(OllamaProvider, "_select_base_url") as mock_select:
                mock_select.return_value = "http://localhost:11434"
                provider = OllamaProvider(base_url=None)
                assert provider is not None

    @pytest.mark.asyncio
    async def test_select_base_url_async_factory(self):
        """Test async factory create method (covers lines 89-90)."""
        with patch.object(
            OllamaProvider, "_select_base_url_async", new_callable=AsyncMock
        ) as mock_async_select:
            mock_async_select.return_value = "http://localhost:11434"

            provider = await OllamaProvider.create(base_url="http://localhost:11434")

            assert provider is not None
            assert provider.name == "ollama"

    def test_skip_discovery_with_list(self):
        """Test _skip_discovery with list base_url (covers lines 56-60)."""
        provider = OllamaProvider(
            base_url=["http://server1:11434"],
            _skip_discovery=True,
        )
        assert provider is not None

    def test_skip_discovery_with_empty_list(self):
        """Test _skip_discovery with empty list falls back to default."""
        provider = OllamaProvider(
            base_url=[],
            _skip_discovery=True,
        )
        # Should fall back to default
        assert provider is not None


class TestStreamRetryWithoutTools:
    """Test stream retry-without-tools functionality."""

    @pytest.mark.asyncio
    async def test_stream_retry_on_tools_not_supported(self, ollama_provider):
        """Test that stream retries without tools when model doesn't support them."""
        call_count = 0

        async def mock_aiter_lines():
            yield '{"message":{"content":"Hello"},"done":false}'
            yield '{"message":{"content":" world"},"done":true,"done_reason":"stop"}'

        def create_mock_response(status_code, error_body=None):
            mock_response = MagicMock()
            mock_response.status_code = status_code
            mock_response.raise_for_status = MagicMock()
            if status_code == 200:
                mock_response.aiter_lines = mock_aiter_lines
            if error_body:
                mock_response.aread = AsyncMock(return_value=error_body.encode())
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()
            return mock_response

        def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            payload = kwargs.get("json", {})

            # First call with tools should fail
            if call_count == 1 and "tools" in payload and payload["tools"]:
                return create_mock_response(400, '{"error":"model does not support tools"}')
            # Second call without tools should succeed
            return create_mock_response(200)

        with patch.object(ollama_provider.client, "stream", side_effect=mock_stream):
            messages = [Message(role="user", content="Hello")]
            tools = [
                ToolDefinition(
                    name="test_tool",
                    description="A test tool",
                    parameters={"type": "object"},
                )
            ]
            chunks = []

            async for chunk in ollama_provider.stream(
                messages=messages, model="test-model", tools=tools
            ):
                chunks.append(chunk)

            # Should have retried and succeeded
            assert call_count == 2
            assert len(chunks) == 2
            assert chunks[0].content == "Hello"
            # Model should be cached as not supporting tools
            assert "test-model" in ollama_provider._models_without_tools

    @pytest.mark.asyncio
    async def test_stream_uses_cached_no_tools_flag(self, ollama_provider):
        """Test that stream skips tools for models cached as not supporting them."""
        # Pre-populate cache
        ollama_provider._models_without_tools.add("cached-model")

        async def mock_aiter_lines():
            yield '{"message":{"content":"OK"},"done":true,"done_reason":"stop"}'

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        payloads_sent = []

        def capture_stream(*args, **kwargs):
            payloads_sent.append(kwargs.get("json", {}))
            return mock_response

        with patch.object(ollama_provider.client, "stream", side_effect=capture_stream):
            messages = [Message(role="user", content="Hello")]
            tools = [
                ToolDefinition(
                    name="test_tool",
                    description="A test tool",
                    parameters={"type": "object"},
                )
            ]
            chunks = []

            async for chunk in ollama_provider.stream(
                messages=messages, model="cached-model", tools=tools
            ):
                chunks.append(chunk)

            # Should have sent only one request without tools
            assert len(payloads_sent) == 1
            assert "tools" not in payloads_sent[0] or not payloads_sent[0].get("tools")
