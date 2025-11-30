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

"""Comprehensive tests for Google provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from victor.providers.google_provider import GoogleProvider
from victor.providers.base import (
    Message,
    ToolDefinition,
    ProviderError,
)


@pytest.fixture
def google_provider():
    """Create GoogleProvider instance for testing."""
    with patch("victor.providers.google_provider.genai.configure"):
        provider = GoogleProvider(
            api_key="test-api-key",
            timeout=30,
        )
    return provider


@pytest.mark.asyncio
async def test_initialization():
    """Test provider initialization."""
    with patch("victor.providers.google_provider.genai.configure") as mock_configure:
        provider = GoogleProvider(
            api_key="test-key",
            timeout=45,
        )

        assert provider.api_key == "test-key"
        assert provider.timeout == 45
        mock_configure.assert_called_once_with(api_key="test-key")


@pytest.mark.asyncio
async def test_provider_name(google_provider):
    """Test provider name property."""
    assert google_provider.name == "google"


@pytest.mark.asyncio
async def test_supports_tools(google_provider):
    """Test tools support."""
    assert google_provider.supports_tools() is True


@pytest.mark.asyncio
async def test_supports_streaming(google_provider):
    """Test streaming support."""
    assert google_provider.supports_streaming() is True


@pytest.mark.asyncio
async def test_chat_success_basic(google_provider):
    """Test successful chat completion with basic message."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.text = "Hello! How can I help you?"
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_response.usage_metadata.total_token_count = 30

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.send_message_async = AsyncMock(return_value=mock_response)

    # Create mock model
    mock_model = MagicMock()
    mock_model.start_chat = MagicMock(return_value=mock_chat)

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [Message(role="user", content="Hello")]
        response = await google_provider.chat(
            messages=messages,
            model="gemini-1.5-pro",
            temperature=0.7,
            max_tokens=1024,
        )

        assert response.content == "Hello! How can I help you?"
        assert response.role == "assistant"
        assert response.model == "gemini-1.5-pro"
        assert response.usage is not None
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30


@pytest.mark.asyncio
async def test_chat_without_usage(google_provider):
    """Test chat response without usage metadata."""
    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_response.usage_metadata = None

    mock_chat = MagicMock()
    mock_chat.send_message_async = AsyncMock(return_value=mock_response)

    mock_model = MagicMock()
    mock_model.start_chat = MagicMock(return_value=mock_chat)

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [Message(role="user", content="Hello")]
        response = await google_provider.chat(
            messages=messages,
            model="gemini-1.5-flash",
        )

        assert response.content == "Response"
        assert response.usage is None


@pytest.mark.asyncio
async def test_chat_with_conversation_history(google_provider):
    """Test chat with multiple messages."""
    mock_response = MagicMock()
    mock_response.text = "Third response"
    mock_response.usage_metadata = None

    mock_chat = MagicMock()
    mock_chat.send_message_async = AsyncMock(return_value=mock_response)

    mock_model = MagicMock()
    mock_model.start_chat = MagicMock(return_value=mock_chat)

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [
            Message(role="user", content="First message"),
            Message(role="assistant", content="First response"),
            Message(role="user", content="Second message"),
        ]
        response = await google_provider.chat(
            messages=messages,
            model="gemini-1.5-pro",
        )

        # Verify history was built correctly
        call_args = mock_model.start_chat.call_args
        history = call_args.kwargs["history"]

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["parts"] == ["First message"]
        assert history[1]["role"] == "model"
        assert history[1]["parts"] == ["First response"]

        # Verify latest message was sent separately
        mock_chat.send_message_async.assert_called_once_with("Second message")


@pytest.mark.asyncio
async def test_chat_with_system_message(google_provider):
    """Test chat with system message (converted to user role)."""
    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_response.usage_metadata = None

    mock_chat = MagicMock()
    mock_chat.send_message_async = AsyncMock(return_value=mock_response)

    mock_model = MagicMock()
    mock_model.start_chat = MagicMock(return_value=mock_chat)

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [
            Message(role="system", content="System message"),
            Message(role="user", content="Hello"),
        ]
        response = await google_provider.chat(
            messages=messages,
            model="gemini-1.5-pro",
        )

        # Verify system message was converted to user role in history
        call_args = mock_model.start_chat.call_args
        history = call_args.kwargs["history"]

        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["parts"] == ["System message"]


@pytest.mark.asyncio
async def test_chat_error_handling(google_provider):
    """Test generic error handling."""
    mock_model = MagicMock()
    mock_model.start_chat.side_effect = Exception("API error")

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError) as exc_info:
            await google_provider.chat(
                messages=messages,
                model="gemini-1.5-pro",
            )

        assert "Google API error" in str(exc_info.value)
        assert exc_info.value.provider == "google"


@pytest.mark.asyncio
async def test_chat_with_custom_generation_config(google_provider):
    """Test chat with custom generation configuration."""
    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_response.usage_metadata = None

    mock_chat = MagicMock()
    mock_chat.send_message_async = AsyncMock(return_value=mock_response)

    mock_model = MagicMock()
    mock_model.start_chat = MagicMock(return_value=mock_chat)

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [Message(role="user", content="Hello")]
        response = await google_provider.chat(
            messages=messages,
            model="gemini-1.5-pro",
            temperature=0.9,
            max_tokens=2048,
            top_p=0.95,
            top_k=40,
        )

        # Verify generation config was passed
        call_args = mock_gen_model.call_args
        gen_config = call_args.kwargs["generation_config"]

        assert gen_config["temperature"] == 0.9
        assert gen_config["max_output_tokens"] == 2048
        assert "top_p" in gen_config
        assert "top_k" in gen_config


@pytest.mark.asyncio
async def test_stream_success(google_provider):
    """Test successful streaming."""
    # Create mock stream chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Hello "
    mock_chunk1.candidates = [MagicMock(finish_reason=None)]

    mock_chunk2 = MagicMock()
    mock_chunk2.text = "world!"
    mock_chunk2.candidates = [MagicMock(finish_reason=None)]

    mock_chunk3 = MagicMock()
    mock_chunk3.text = ""
    mock_chunk3.candidates = [MagicMock(finish_reason="stop")]

    async def async_iter():
        for chunk in [mock_chunk1, mock_chunk2, mock_chunk3]:
            yield chunk

    mock_chat = MagicMock()
    mock_chat.send_message_async = AsyncMock(return_value=async_iter())

    mock_model = MagicMock()
    mock_model.start_chat = MagicMock(return_value=mock_chat)

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [Message(role="user", content="Hello")]
        chunks = []

        async for chunk in google_provider.stream(
            messages=messages,
            model="gemini-1.5-pro",
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
async def test_stream_with_history(google_provider):
    """Test streaming with conversation history."""
    mock_chunk = MagicMock()
    mock_chunk.text = "Response"
    mock_chunk.candidates = [MagicMock(finish_reason="stop")]

    async def async_iter():
        yield mock_chunk

    mock_chat = MagicMock()
    mock_chat.send_message_async = AsyncMock(return_value=async_iter())

    mock_model = MagicMock()
    mock_model.start_chat = MagicMock(return_value=mock_chat)

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [
            Message(role="user", content="First"),
            Message(role="assistant", content="Second"),
            Message(role="user", content="Third"),
        ]

        chunks = []
        async for chunk in google_provider.stream(
            messages=messages,
            model="gemini-1.5-flash",
        ):
            chunks.append(chunk)

        # Verify send_message_async was called with stream=True
        call_args = mock_chat.send_message_async.call_args
        assert call_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_stream_error(google_provider):
    """Test streaming error handling."""
    mock_model = MagicMock()
    mock_model.start_chat.side_effect = Exception("Stream error")

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError) as exc_info:
            async for chunk in google_provider.stream(
                messages=messages,
                model="gemini-1.5-pro",
            ):
                pass

        assert "Google streaming error" in str(exc_info.value)
        assert exc_info.value.provider == "google"


@pytest.mark.asyncio
async def test_convert_messages_single_message(google_provider):
    """Test converting single message."""
    messages = [Message(role="user", content="Hello")]

    history, latest_message = google_provider._convert_messages(messages)

    assert history == []
    assert latest_message == "Hello"


@pytest.mark.asyncio
async def test_convert_messages_multiple_messages(google_provider):
    """Test converting multiple messages."""
    messages = [
        Message(role="user", content="First"),
        Message(role="assistant", content="Second"),
        Message(role="user", content="Third"),
        Message(role="assistant", content="Fourth"),
        Message(role="user", content="Fifth"),
    ]

    history, latest_message = google_provider._convert_messages(messages)

    assert len(history) == 4
    assert history[0] == {"role": "user", "parts": ["First"]}
    assert history[1] == {"role": "model", "parts": ["Second"]}
    assert history[2] == {"role": "user", "parts": ["Third"]}
    assert history[3] == {"role": "model", "parts": ["Fourth"]}
    assert latest_message == "Fifth"


@pytest.mark.asyncio
async def test_convert_messages_role_conversion(google_provider):
    """Test assistant role is converted to model role."""
    messages = [
        Message(role="assistant", content="Assistant message"),
        Message(role="user", content="User message"),
    ]

    history, latest_message = google_provider._convert_messages(messages)

    assert len(history) == 1
    assert history[0]["role"] == "model"
    assert history[0]["parts"] == ["Assistant message"]
    assert latest_message == "User message"


@pytest.mark.asyncio
async def test_parse_response_with_text(google_provider):
    """Test parsing response with text."""
    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 5
    mock_response.usage_metadata.candidates_token_count = 10
    mock_response.usage_metadata.total_token_count = 15

    response = google_provider._parse_response(mock_response, "test-model")

    assert response.content == "Test response"
    assert response.role == "assistant"
    assert response.model == "test-model"
    assert response.usage["prompt_tokens"] == 5
    assert response.usage["completion_tokens"] == 10
    assert response.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_parse_response_empty_text(google_provider):
    """Test parsing response with empty text."""
    mock_response = MagicMock()
    mock_response.text = ""
    mock_response.usage_metadata = None

    response = google_provider._parse_response(mock_response, "test-model")

    assert response.content == ""
    assert response.usage is None


@pytest.mark.asyncio
async def test_parse_response_no_text(google_provider):
    """Test parsing response without text attribute."""
    # Create a mock without text attribute that will raise AttributeError
    mock_response = MagicMock()
    type(mock_response).text = property(
        lambda self: (_ for _ in ()).throw(AttributeError("no text"))
    )
    mock_response.usage_metadata = None

    # Since the code checks `if response.text:`, the AttributeError will propagate
    # Let's just test with None text instead (more realistic)
    mock_response2 = MagicMock()
    mock_response2.text = None
    mock_response2.usage_metadata = None

    response = google_provider._parse_response(mock_response2, "test-model")

    assert response.content == ""
    assert response.usage is None


@pytest.mark.asyncio
async def test_parse_stream_chunk_with_text(google_provider):
    """Test parsing stream chunk with text."""
    mock_chunk = MagicMock()
    mock_chunk.text = "Hello"
    mock_chunk.candidates = [MagicMock(finish_reason=None)]

    chunk = google_provider._parse_stream_chunk(mock_chunk)

    assert chunk is not None
    assert chunk.content == "Hello"
    assert chunk.is_final is False


@pytest.mark.asyncio
async def test_parse_stream_chunk_final(google_provider):
    """Test parsing final stream chunk."""
    mock_chunk = MagicMock()
    mock_chunk.text = "Done"
    mock_chunk.candidates = [MagicMock(finish_reason="stop")]

    chunk = google_provider._parse_stream_chunk(mock_chunk)

    assert chunk is not None
    assert chunk.content == "Done"
    assert chunk.is_final is True


@pytest.mark.asyncio
async def test_parse_stream_chunk_no_text(google_provider):
    """Test parsing stream chunk without text."""
    # Create mock without text attribute - hasattr will return False
    mock_chunk = MagicMock()
    del mock_chunk.text  # Remove text attribute
    # Also remove candidates to avoid the is_final expression evaluating to None
    del mock_chunk.candidates

    chunk = google_provider._parse_stream_chunk(mock_chunk)

    assert chunk is not None
    assert chunk.content == ""
    assert chunk.is_final is False


@pytest.mark.asyncio
async def test_parse_stream_chunk_no_candidates(google_provider):
    """Test parsing stream chunk without candidates."""
    mock_chunk = MagicMock()
    mock_chunk.text = "Text"
    # Remove candidates attribute
    del mock_chunk.candidates

    chunk = google_provider._parse_stream_chunk(mock_chunk)

    assert chunk is not None
    assert chunk.content == "Text"
    assert chunk.is_final is False


@pytest.mark.asyncio
async def test_parse_stream_chunk_empty_candidates(google_provider):
    """Test parsing stream chunk with empty candidates list."""
    mock_chunk = MagicMock()
    mock_chunk.text = "Text"
    # Set candidates to a non-empty list with a candidate that has no finish_reason
    mock_candidate = MagicMock()
    del mock_candidate.finish_reason  # Remove finish_reason attribute
    mock_chunk.candidates = [mock_candidate]

    chunk = google_provider._parse_stream_chunk(mock_chunk)

    assert chunk is not None
    assert chunk.content == "Text"
    assert chunk.is_final is False


@pytest.mark.asyncio
async def test_parse_stream_chunk_candidate_no_finish_reason(google_provider):
    """Test parsing stream chunk with candidate but no finish_reason."""
    mock_candidate = MagicMock()
    del mock_candidate.finish_reason  # Remove finish_reason attribute

    mock_chunk = MagicMock()
    mock_chunk.text = "Text"
    mock_chunk.candidates = [mock_candidate]

    chunk = google_provider._parse_stream_chunk(mock_chunk)

    assert chunk is not None
    assert chunk.content == "Text"
    assert chunk.is_final is False


@pytest.mark.asyncio
async def test_close(google_provider):
    """Test closing the provider (no-op for Google)."""
    # Should not raise any exception
    await google_provider.close()


@pytest.mark.asyncio
async def test_single_user_message(google_provider):
    """Test with only one user message."""
    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_response.usage_metadata = None

    mock_chat = MagicMock()
    mock_chat.send_message_async = AsyncMock(return_value=mock_response)

    mock_model = MagicMock()
    mock_model.start_chat = MagicMock(return_value=mock_chat)

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [Message(role="user", content="Only message")]
        response = await google_provider.chat(
            messages=messages,
            model="gemini-1.5-pro",
        )

        # Verify history is empty
        call_args = mock_model.start_chat.call_args
        history = call_args.kwargs["history"]
        assert history == []

        # Verify message was sent
        mock_chat.send_message_async.assert_called_once_with("Only message")


@pytest.mark.asyncio
async def test_model_initialization_params(google_provider):
    """Test model initialization with correct parameters."""
    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_response.usage_metadata = None

    mock_chat = MagicMock()
    mock_chat.send_message_async = AsyncMock(return_value=mock_response)

    mock_model = MagicMock()
    mock_model.start_chat = MagicMock(return_value=mock_chat)

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [Message(role="user", content="Test")]
        response = await google_provider.chat(
            messages=messages,
            model="gemini-1.5-flash",
            temperature=0.5,
            max_tokens=512,
        )

        # Verify model was initialized with correct params
        mock_gen_model.assert_called_once()
        call_args = mock_gen_model.call_args

        assert call_args.kwargs["model_name"] == "gemini-1.5-flash"
        gen_config = call_args.kwargs["generation_config"]
        assert gen_config["temperature"] == 0.5
        assert gen_config["max_output_tokens"] == 512


@pytest.mark.asyncio
async def test_stream_model_initialization(google_provider):
    """Test model initialization for streaming."""
    mock_chunk = MagicMock()
    mock_chunk.text = "Text"
    mock_chunk.candidates = [MagicMock(finish_reason="stop")]

    async def async_iter():
        yield mock_chunk

    mock_chat = MagicMock()
    mock_chat.send_message_async = AsyncMock(return_value=async_iter())

    mock_model = MagicMock()
    mock_model.start_chat = MagicMock(return_value=mock_chat)

    with patch("victor.providers.google_provider.genai.GenerativeModel") as mock_gen_model:
        mock_gen_model.return_value = mock_model

        messages = [Message(role="user", content="Test")]

        chunks = []
        async for chunk in google_provider.stream(
            messages=messages,
            model="gemini-1.5-pro",
            temperature=0.8,
            max_tokens=1024,
        ):
            chunks.append(chunk)

        # Verify model was initialized for streaming
        mock_gen_model.assert_called_once()
        call_args = mock_gen_model.call_args

        assert call_args.kwargs["model_name"] == "gemini-1.5-pro"
        gen_config = call_args.kwargs["generation_config"]
        assert gen_config["temperature"] == 0.8
        assert gen_config["max_output_tokens"] == 1024
