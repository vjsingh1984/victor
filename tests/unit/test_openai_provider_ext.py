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

"""Comprehensive tests for OpenAI provider."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.openai_provider import OpenAIProvider
from victor.providers.base import (
    Message,
    ToolDefinition,
    ProviderError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
)


@pytest.fixture
def openai_provider():
    """Create OpenAIProvider instance for testing."""
    return OpenAIProvider(
        api_key="test-api-key",
        organization="test-org",
        base_url="https://api.openai.com/v1",
        timeout=30,
        max_retries=2,
    )


@pytest.mark.asyncio
async def test_initialization():
    """Test provider initialization."""
    provider = OpenAIProvider(
        api_key="test-key",
        organization="test-org",
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
async def test_initialization_without_organization():
    """Test provider initialization without organization."""
    provider = OpenAIProvider(api_key="test-key")

    assert provider.api_key == "test-key"
    assert provider.client is not None


@pytest.mark.asyncio
async def test_provider_name(openai_provider):
    """Test provider name property."""
    assert openai_provider.name == "openai"


@pytest.mark.asyncio
async def test_supports_tools(openai_provider):
    """Test tools support."""
    assert openai_provider.supports_tools() is True


@pytest.mark.asyncio
async def test_supports_streaming(openai_provider):
    """Test streaming support."""
    assert openai_provider.supports_streaming() is True


@pytest.mark.asyncio
async def test_chat_success_basic(openai_provider):
    """Test successful chat completion with basic message."""
    # Create mock response
    mock_message = MagicMock()
    mock_message.content = "Hello! How can I help you?"
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model_dump = lambda: {"test": "response"}

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_response

        messages = [Message(role="user", content="Hello")]
        response = await openai_provider.chat(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=1024,
        )

        assert response.content == "Hello! How can I help you?"
        assert response.role == "assistant"
        assert response.model == "gpt-4"
        assert response.stop_reason == "stop"
        assert response.usage is not None
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30


@pytest.mark.asyncio
async def test_chat_with_system_message(openai_provider):
    """Test chat with system message."""
    mock_message = MagicMock()
    mock_message.content = "Response"
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_response.model_dump = lambda: {}

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_response

        messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Hello"),
        ]
        response = await openai_provider.chat(
            messages=messages,
            model="gpt-4-turbo",
        )

        # Verify system message was included
        call_args = mock_create.call_args
        openai_messages = call_args.kwargs["messages"]
        assert len(openai_messages) == 2
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[0]["content"] == "You are a helpful assistant"
        assert openai_messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_chat_with_tools(openai_provider):
    """Test chat with tool definitions."""
    # Create mock tool call
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function = MagicMock()
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = json.dumps({"location": "London"})

    mock_message = MagicMock()
    mock_message.content = None
    mock_message.tool_calls = [mock_tool_call]

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "tool_calls"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 15
    mock_usage.completion_tokens = 25
    mock_usage.total_tokens = 40

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model_dump = lambda: {}

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_response

        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                },
            )
        ]

        messages = [Message(role="user", content="What's the weather in London?")]
        response = await openai_provider.chat(
            messages=messages,
            model="gpt-4",
            tools=tools,
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "call_123"
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tool_calls[0]["arguments"] == json.dumps({"location": "London"})
        assert response.stop_reason == "tool_calls"

        # Verify tools were passed
        call_args = mock_create.call_args
        assert "tools" in call_args.kwargs
        assert "tool_choice" in call_args.kwargs
        assert call_args.kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_chat_with_multiple_tool_calls(openai_provider):
    """Test chat with multiple tool calls."""
    # Create mock tool calls
    mock_tool_call1 = MagicMock()
    mock_tool_call1.id = "call_1"
    mock_tool_call1.function = MagicMock()
    mock_tool_call1.function.name = "tool1"
    mock_tool_call1.function.arguments = "{}"

    mock_tool_call2 = MagicMock()
    mock_tool_call2.id = "call_2"
    mock_tool_call2.function = MagicMock()
    mock_tool_call2.function.name = "tool2"
    mock_tool_call2.function.arguments = "{}"

    mock_message = MagicMock()
    mock_message.content = None
    mock_message.tool_calls = [mock_tool_call1, mock_tool_call2]

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "tool_calls"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_response.model_dump = lambda: {}

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_response

        messages = [Message(role="user", content="Test")]
        response = await openai_provider.chat(
            messages=messages,
            model="gpt-4",
        )

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0]["id"] == "call_1"
        assert response.tool_calls[1]["id"] == "call_2"


@pytest.mark.asyncio
async def test_chat_authentication_error(openai_provider):
    """Test authentication error handling."""
    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = Exception("Invalid API_KEY provided")

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderAuthenticationError) as exc_info:
            await openai_provider.chat(
                messages=messages,
                model="gpt-4",
            )

        assert "Authentication failed" in str(exc_info.value)
        assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
async def test_chat_rate_limit_error(openai_provider):
    """Test rate limit error handling."""
    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = Exception("Rate limit exceeded: 429")

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderRateLimitError) as exc_info:
            await openai_provider.chat(
                messages=messages,
                model="gpt-4",
            )

        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_chat_generic_error(openai_provider):
    """Test generic error handling."""
    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = Exception("Something went wrong")

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError) as exc_info:
            await openai_provider.chat(
                messages=messages,
                model="gpt-4",
            )

        assert "OpenAI API error" in str(exc_info.value)
        assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
async def test_stream_success(openai_provider):
    """Test successful streaming."""
    # Create mock stream chunks
    mock_delta1 = MagicMock()
    mock_delta1.content = "Hello "
    mock_choice1 = MagicMock()
    mock_choice1.delta = mock_delta1
    mock_choice1.finish_reason = None
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [mock_choice1]

    mock_delta2 = MagicMock()
    mock_delta2.content = "world!"
    mock_choice2 = MagicMock()
    mock_choice2.delta = mock_delta2
    mock_choice2.finish_reason = None
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [mock_choice2]

    mock_delta3 = MagicMock()
    mock_delta3.content = None
    mock_choice3 = MagicMock()
    mock_choice3.delta = mock_delta3
    mock_choice3.finish_reason = "stop"
    mock_chunk3 = MagicMock()
    mock_chunk3.choices = [mock_choice3]

    async def async_iter():
        for chunk in [mock_chunk1, mock_chunk2, mock_chunk3]:
            yield chunk

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = async_iter()

        messages = [Message(role="user", content="Hello")]
        chunks = []

        async for chunk in openai_provider.stream(
            messages=messages,
            model="gpt-4",
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello "
        assert chunks[0].is_final is False
        assert chunks[1].content == "world!"
        assert chunks[1].is_final is False
        assert chunks[2].content == ""
        assert chunks[2].is_final is True
        assert chunks[2].stop_reason == "stop"


@pytest.mark.asyncio
async def test_stream_with_tools(openai_provider):
    """Test streaming with tools."""
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content=""), finish_reason="stop")]

    async def async_iter():
        yield mock_chunk

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = async_iter()

        tools = [
            ToolDefinition(
                name="calculate",
                description="Perform calculation",
                parameters={"type": "object", "properties": {}},
            )
        ]

        messages = [Message(role="user", content="Calculate 2+2")]

        chunks = []
        async for chunk in openai_provider.stream(
            messages=messages,
            model="gpt-4",
            tools=tools,
        ):
            chunks.append(chunk)

        # Verify tools and stream=True were passed
        call_args = mock_create.call_args
        assert "tools" in call_args.kwargs
        assert "tool_choice" in call_args.kwargs
        assert call_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_stream_empty_chunks(openai_provider):
    """Test streaming with empty choices."""
    mock_chunk = MagicMock()
    mock_chunk.choices = []

    async def async_iter():
        yield mock_chunk

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = async_iter()

        messages = [Message(role="user", content="Hello")]

        chunks = []
        async for chunk in openai_provider.stream(
            messages=messages,
            model="gpt-4",
        ):
            chunks.append(chunk)

        # Should handle empty choices gracefully
        assert len(chunks) == 0


@pytest.mark.asyncio
async def test_stream_error(openai_provider):
    """Test streaming error handling."""
    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = Exception("Stream error")

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError):
            async for chunk in openai_provider.stream(
                messages=messages,
                model="gpt-4",
            ):
                pass


@pytest.mark.asyncio
async def test_convert_tools(openai_provider):
    """Test tool conversion to OpenAI format."""
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

    converted = openai_provider._convert_tools(tools)

    assert len(converted) == 2
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "get_weather"
    assert converted[0]["function"]["description"] == "Get weather information"
    assert converted[0]["function"]["parameters"]["type"] == "object"
    assert "location" in converted[0]["function"]["parameters"]["properties"]
    assert converted[1]["type"] == "function"
    assert converted[1]["function"]["name"] == "search"


@pytest.mark.asyncio
async def test_parse_response_with_content(openai_provider):
    """Test parsing response with content."""
    mock_message = MagicMock()
    mock_message.content = "Test response"
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 5
    mock_usage.completion_tokens = 10
    mock_usage.total_tokens = 15

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model_dump = lambda: {"raw": "data"}

    response = openai_provider._parse_response(mock_response, "test-model")

    assert response.content == "Test response"
    assert response.role == "assistant"
    assert response.model == "test-model"
    assert response.stop_reason == "stop"
    assert response.tool_calls is None
    assert response.usage["prompt_tokens"] == 5


@pytest.mark.asyncio
async def test_parse_response_no_content(openai_provider):
    """Test parsing response without content."""
    mock_message = MagicMock()
    mock_message.content = None
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_response.model_dump = lambda: {}

    response = openai_provider._parse_response(mock_response, "test-model")

    assert response.content == ""
    assert response.usage is None


@pytest.mark.asyncio
async def test_parse_stream_chunk_with_content(openai_provider):
    """Test parsing stream chunk with content."""
    mock_delta = MagicMock()
    mock_delta.content = "Test"

    mock_choice = MagicMock()
    mock_choice.delta = mock_delta
    mock_choice.finish_reason = None

    mock_chunk = MagicMock()
    mock_chunk.choices = [mock_choice]

    chunk = openai_provider._parse_stream_chunk(mock_chunk)

    assert chunk is not None
    assert chunk.content == "Test"
    assert chunk.is_final is False
    assert chunk.stop_reason is None


@pytest.mark.asyncio
async def test_parse_stream_chunk_final(openai_provider):
    """Test parsing final stream chunk."""
    mock_delta = MagicMock()
    mock_delta.content = None

    mock_choice = MagicMock()
    mock_choice.delta = mock_delta
    mock_choice.finish_reason = "stop"

    mock_chunk = MagicMock()
    mock_chunk.choices = [mock_choice]

    chunk = openai_provider._parse_stream_chunk(mock_chunk)

    assert chunk is not None
    assert chunk.content == ""
    assert chunk.is_final is True
    assert chunk.stop_reason == "stop"


@pytest.mark.asyncio
async def test_parse_stream_chunk_no_choices(openai_provider):
    """Test parsing stream chunk with no choices."""
    mock_chunk = MagicMock()
    mock_chunk.choices = []

    chunk = openai_provider._parse_stream_chunk(mock_chunk)

    assert chunk is None


@pytest.mark.asyncio
async def test_close(openai_provider):
    """Test closing the provider client."""
    with patch.object(
        openai_provider.client,
        "close",
        new_callable=AsyncMock,
    ) as mock_close:
        await openai_provider.close()
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_chat_with_custom_kwargs(openai_provider):
    """Test chat with additional custom parameters."""
    mock_message = MagicMock()
    mock_message.content = "Response"
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_response.model_dump = lambda: {}

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_response

        messages = [Message(role="user", content="Hello")]
        response = await openai_provider.chat(
            messages=messages,
            model="gpt-4",
            top_p=0.9,
            presence_penalty=0.5,
            frequency_penalty=0.3,
        )

        # Verify custom kwargs were passed
        call_args = mock_create.call_args
        assert "top_p" in call_args.kwargs
        assert call_args.kwargs["top_p"] == 0.9
        assert "presence_penalty" in call_args.kwargs
        assert "frequency_penalty" in call_args.kwargs


@pytest.mark.asyncio
async def test_message_conversion(openai_provider):
    """Test message conversion to OpenAI format."""
    mock_message = MagicMock()
    mock_message.content = "Response"
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_response.model_dump = lambda: {}

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_response

        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="User"),
            Message(role="assistant", content="Assistant"),
        ]
        response = await openai_provider.chat(
            messages=messages,
            model="gpt-4",
        )

        # Verify messages were converted correctly
        call_args = mock_create.call_args
        openai_messages = call_args.kwargs["messages"]
        assert len(openai_messages) == 3
        assert openai_messages[0] == {"role": "system", "content": "System"}
        assert openai_messages[1] == {"role": "user", "content": "User"}
        assert openai_messages[2] == {"role": "assistant", "content": "Assistant"}
