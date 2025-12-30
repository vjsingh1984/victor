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

"""Comprehensive tests for Anthropic provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.base import (
    Message,
    ToolDefinition,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
)


@pytest.fixture
def anthropic_provider():
    """Create AnthropicProvider instance for testing."""
    return AnthropicProvider(
        api_key="test-api-key",
        base_url="https://api.anthropic.com",
        timeout=30,
        max_retries=2,
    )


@pytest.mark.asyncio
async def test_initialization():
    """Test provider initialization."""
    provider = AnthropicProvider(
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
async def test_provider_name(anthropic_provider):
    """Test provider name property."""
    assert anthropic_provider.name == "anthropic"


@pytest.mark.asyncio
async def test_supports_tools(anthropic_provider):
    """Test tools support."""
    assert anthropic_provider.supports_tools() is True


@pytest.mark.asyncio
async def test_supports_streaming(anthropic_provider):
    """Test streaming support."""
    assert anthropic_provider.supports_streaming() is True


@pytest.mark.asyncio
async def test_chat_success_basic(anthropic_provider):
    """Test successful chat completion with basic message."""
    # Create mock response
    mock_message = MagicMock()
    mock_message.content = [MagicMock(type="text", text="Hello! How can I help you?")]
    mock_message.stop_reason = "end_turn"
    mock_message.usage = MagicMock(
        input_tokens=10,
        output_tokens=20,
    )
    mock_message.model_dump = lambda: {"test": "response"}

    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_message

        messages = [Message(role="user", content="Hello")]
        response = await anthropic_provider.chat(
            messages=messages,
            model="claude-sonnet-4-5",
            temperature=0.7,
            max_tokens=1024,
        )

        assert response.content == "Hello! How can I help you?"
        assert response.role == "assistant"
        assert response.model == "claude-sonnet-4-5"
        assert response.stop_reason == "end_turn"
        assert response.usage is not None
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30


@pytest.mark.asyncio
async def test_chat_with_system_message(anthropic_provider):
    """Test chat with system message."""
    mock_message = MagicMock()
    mock_message.content = [MagicMock(type="text", text="Response")]
    mock_message.stop_reason = "end_turn"
    mock_message.usage = None
    mock_message.model_dump = lambda: {}

    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_message

        messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Hello"),
        ]
        await anthropic_provider.chat(
            messages=messages,
            model="claude-3-opus",
        )

        # Verify system message was separated
        call_args = mock_create.call_args
        assert "system" in call_args.kwargs
        assert call_args.kwargs["system"] == "You are a helpful assistant"
        assert len(call_args.kwargs["messages"]) == 1
        assert call_args.kwargs["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_chat_with_tools(anthropic_provider):
    """Test chat with tool definitions."""
    # Create mock tool_use block
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "tool_123"
    mock_tool_block.name = "get_weather"
    mock_tool_block.input = {"location": "London"}

    mock_message = MagicMock()
    mock_message.content = [mock_tool_block]
    mock_message.stop_reason = "tool_use"
    mock_message.usage = MagicMock(input_tokens=15, output_tokens=25)
    mock_message.model_dump = lambda: {}

    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_message

        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            )
        ]

        messages = [Message(role="user", content="What's the weather in London?")]
        response = await anthropic_provider.chat(
            messages=messages,
            model="claude-sonnet-4-5",
            tools=tools,
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "tool_123"
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tool_calls[0]["arguments"] == {"location": "London"}
        assert response.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_chat_with_mixed_content(anthropic_provider):
    """Test chat with mixed text and tool call content."""
    # Create mock blocks
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Let me check that for you. "

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "tool_456"
    mock_tool_block.name = "search"
    mock_tool_block.input = {"query": "test"}

    mock_message = MagicMock()
    mock_message.content = [mock_text_block, mock_tool_block]
    mock_message.stop_reason = "tool_use"
    mock_message.usage = None
    mock_message.model_dump = lambda: {}

    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_message

        messages = [Message(role="user", content="Search for test")]
        response = await anthropic_provider.chat(
            messages=messages,
            model="claude-3-opus",
        )

        assert response.content == "Let me check that for you. "
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1


@pytest.mark.asyncio
async def test_chat_authentication_error(anthropic_provider):
    """Test authentication error handling."""
    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = Exception("authentication failed")

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderAuthError) as exc_info:
            await anthropic_provider.chat(
                messages=messages,
                model="claude-sonnet-4-5",
            )

        assert "Authentication failed" in str(exc_info.value)
        assert exc_info.value.provider == "anthropic"


@pytest.mark.asyncio
async def test_chat_rate_limit_error(anthropic_provider):
    """Test rate limit error handling."""
    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = Exception("rate_limit exceeded 429")

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderRateLimitError) as exc_info:
            await anthropic_provider.chat(
                messages=messages,
                model="claude-sonnet-4-5",
            )

        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_chat_generic_error(anthropic_provider):
    """Test generic error handling."""
    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = Exception("Something went wrong")

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError) as exc_info:
            await anthropic_provider.chat(
                messages=messages,
                model="claude-sonnet-4-5",
            )

        assert "Anthropic API error" in str(exc_info.value)
        assert exc_info.value.provider == "anthropic"


@pytest.mark.asyncio
async def test_stream_success(anthropic_provider):
    """Test successful streaming."""
    # Create mock stream events - delta needs type="text_delta" for the provider to recognize it
    mock_delta1 = MagicMock()
    mock_delta1.type = "text_delta"
    mock_delta1.text = "Hello "

    mock_event1 = MagicMock()
    mock_event1.type = "content_block_delta"
    mock_event1.delta = mock_delta1

    mock_delta2 = MagicMock()
    mock_delta2.type = "text_delta"
    mock_delta2.text = "world!"

    mock_event2 = MagicMock()
    mock_event2.type = "content_block_delta"
    mock_event2.delta = mock_delta2

    mock_event3 = MagicMock()
    mock_event3.type = "message_stop"

    # Create async context manager mock
    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock()

    async def async_iter():
        for event in [mock_event1, mock_event2, mock_event3]:
            yield event

    mock_stream.__aiter__ = lambda self: async_iter()

    with patch.object(
        anthropic_provider.client.messages,
        "stream",
    ) as mock_stream_method:
        mock_stream_method.return_value = mock_stream

        messages = [Message(role="user", content="Hello")]
        chunks = []

        async for chunk in anthropic_provider.stream(
            messages=messages,
            model="claude-sonnet-4-5",
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello "
        assert chunks[0].is_final is False
        assert chunks[1].content == "world!"
        assert chunks[1].is_final is False
        assert chunks[2].content == ""
        assert chunks[2].is_final is True


@pytest.mark.asyncio
async def test_stream_with_system_message(anthropic_provider):
    """Test streaming with system message."""
    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock()

    async def async_iter():
        yield MagicMock(type="message_stop")

    mock_stream.__aiter__ = lambda self: async_iter()

    with patch.object(
        anthropic_provider.client.messages,
        "stream",
    ) as mock_stream_method:
        mock_stream_method.return_value = mock_stream

        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello"),
        ]

        chunks = []
        async for chunk in anthropic_provider.stream(
            messages=messages,
            model="claude-3-opus",
        ):
            chunks.append(chunk)

        # Verify system message was separated
        call_args = mock_stream_method.call_args
        assert "system" in call_args.kwargs
        assert call_args.kwargs["system"] == "System prompt"


@pytest.mark.asyncio
async def test_stream_with_tools(anthropic_provider):
    """Test streaming with tools."""
    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock()

    async def async_iter():
        yield MagicMock(type="message_stop")

    mock_stream.__aiter__ = lambda self: async_iter()

    with patch.object(
        anthropic_provider.client.messages,
        "stream",
    ) as mock_stream_method:
        mock_stream_method.return_value = mock_stream

        tools = [
            ToolDefinition(
                name="calculate",
                description="Perform calculation",
                parameters={"type": "object", "properties": {}},
            )
        ]

        messages = [Message(role="user", content="Calculate 2+2")]

        chunks = []
        async for chunk in anthropic_provider.stream(
            messages=messages,
            model="claude-sonnet-4-5",
            tools=tools,
        ):
            chunks.append(chunk)

        # Verify tools were converted
        call_args = mock_stream_method.call_args
        assert "tools" in call_args.kwargs


@pytest.mark.asyncio
async def test_stream_error(anthropic_provider):
    """Test streaming error handling."""
    with patch.object(
        anthropic_provider.client.messages,
        "stream",
    ) as mock_stream_method:
        mock_stream_method.side_effect = Exception("Stream error")

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError):
            async for _chunk in anthropic_provider.stream(
                messages=messages,
                model="claude-sonnet-4-5",
            ):
                pass


@pytest.mark.asyncio
async def test_convert_tools(anthropic_provider):
    """Test tool conversion to Anthropic format."""
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather information",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        ),
        ToolDefinition(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        ),
    ]

    converted = anthropic_provider._convert_tools(tools)

    assert len(converted) == 2
    assert converted[0]["name"] == "get_weather"
    assert converted[0]["description"] == "Get weather information"
    assert converted[0]["input_schema"]["type"] == "object"
    assert "location" in converted[0]["input_schema"]["properties"]
    assert converted[1]["name"] == "search"


@pytest.mark.asyncio
async def test_parse_response_text_only(anthropic_provider):
    """Test parsing response with text content only."""
    mock_message = MagicMock()
    mock_message.content = [
        MagicMock(type="text", text="First part. "),
        MagicMock(type="text", text="Second part."),
    ]
    mock_message.stop_reason = "end_turn"
    mock_message.usage = MagicMock(input_tokens=5, output_tokens=10)
    mock_message.model_dump = lambda: {"raw": "data"}

    response = anthropic_provider._parse_response(mock_message, "test-model")

    assert response.content == "First part. Second part."
    assert response.role == "assistant"
    assert response.model == "test-model"
    assert response.stop_reason == "end_turn"
    assert response.tool_calls is None
    assert response.usage["prompt_tokens"] == 5
    assert response.usage["completion_tokens"] == 10


@pytest.mark.asyncio
async def test_parse_response_no_usage(anthropic_provider):
    """Test parsing response without usage information."""
    mock_message = MagicMock()
    mock_message.content = [MagicMock(type="text", text="Test")]
    mock_message.stop_reason = "end_turn"
    mock_message.usage = None
    mock_message.model_dump = lambda: {}

    response = anthropic_provider._parse_response(mock_message, "test-model")

    assert response.content == "Test"
    assert response.usage is None


@pytest.mark.asyncio
async def test_close(anthropic_provider):
    """Test closing the provider client."""
    with patch.object(
        anthropic_provider.client,
        "close",
        new_callable=AsyncMock,
    ) as mock_close:
        await anthropic_provider.close()
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_chat_with_custom_kwargs(anthropic_provider):
    """Test chat with additional custom parameters."""
    mock_message = MagicMock()
    mock_message.content = [MagicMock(type="text", text="Response")]
    mock_message.stop_reason = "end_turn"
    mock_message.usage = None
    mock_message.model_dump = lambda: {}

    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_message

        messages = [Message(role="user", content="Hello")]
        await anthropic_provider.chat(
            messages=messages,
            model="claude-sonnet-4-5",
            top_p=0.9,
            top_k=40,
        )

        # Verify custom kwargs were passed
        call_args = mock_create.call_args
        assert "top_p" in call_args.kwargs
        assert call_args.kwargs["top_p"] == 0.9
        assert "top_k" in call_args.kwargs
        assert call_args.kwargs["top_k"] == 40


@pytest.mark.asyncio
async def test_multiple_system_messages(anthropic_provider):
    """Test handling multiple system messages (last one wins)."""
    mock_message = MagicMock()
    mock_message.content = [MagicMock(type="text", text="Response")]
    mock_message.stop_reason = "end_turn"
    mock_message.usage = None
    mock_message.model_dump = lambda: {}

    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_message

        messages = [
            Message(role="system", content="First system message"),
            Message(role="system", content="Second system message"),
            Message(role="user", content="Hello"),
        ]
        await anthropic_provider.chat(
            messages=messages,
            model="claude-3-opus",
        )

        # Verify the last system message was used
        call_args = mock_create.call_args
        assert call_args.kwargs["system"] == "Second system message"
