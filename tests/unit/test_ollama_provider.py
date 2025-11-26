"""Tests for Ollama provider."""

import pytest
from unittest.mock import AsyncMock, patch

from victor.providers.base import Message, ProviderError
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
