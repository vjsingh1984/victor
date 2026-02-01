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

"""Tests for GroqProvider."""

import pytest
from unittest.mock import MagicMock, patch
import httpx

from victor.providers.groq_provider import (
    GroqProvider,
    GROQ_MODELS,
    DEFAULT_BASE_URL,
)
from victor.core.errors import (
    ProviderError,
    ProviderTimeoutError,
)
from victor.providers.base import (
    CompletionResponse,
    Message,
    ToolDefinition,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test-groq-api-key"


@pytest.fixture
def groq_provider(mock_api_key):
    """Create a GroqProvider instance for testing."""
    with patch.dict("os.environ", {"GROQ_API_KEY": mock_api_key}):
        provider = GroqProvider(api_key=mock_api_key)
        yield provider


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello, world!"),
    ]


@pytest.fixture
def sample_tool():
    """Sample tool definition."""
    return ToolDefinition(
        name="get_weather",
        description="Get the weather for a location",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string", "description": "The city name"}},
            "required": ["location"],
        },
    )


@pytest.fixture
def mock_chat_response():
    """Mock chat completion response from Groq."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
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


@pytest.fixture
def mock_tool_call_response():
    """Mock response with tool calls from Groq."""
    return {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama-3.3-70b-versatile",
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
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "total_tokens": 35,
        },
    }


# =============================================================================
# GROQ MODELS TESTS
# =============================================================================


class TestGroqModels:
    """Tests for GROQ_MODELS configuration."""

    def test_models_have_required_fields(self):
        """Test all models have required configuration fields."""
        required_fields = ["description", "context_window", "max_output", "supports_tools"]
        for model_name, config in GROQ_MODELS.items():
            for field in required_fields:
                assert field in config, f"Model {model_name} missing {field}"

    def test_llama_model_exists(self):
        """Test that main Llama model exists."""
        assert "llama-3.3-70b-versatile" in GROQ_MODELS
        assert GROQ_MODELS["llama-3.3-70b-versatile"]["supports_tools"] is True

    def test_instant_model_exists(self):
        """Test that instant model exists."""
        assert "llama-3.1-8b-instant" in GROQ_MODELS

    def test_preview_models_marked(self):
        """Test that preview models are marked appropriately."""
        for model_name, config in GROQ_MODELS.items():
            if "qwen" in model_name.lower() or "kimi" in model_name.lower():
                assert config.get("preview") is True


# =============================================================================
# PROVIDER INITIALIZATION TESTS
# =============================================================================


class TestGroqProviderInit:
    """Tests for GroqProvider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = GroqProvider(api_key="test-key")
        assert provider._api_key == "test-key"
        assert provider.name == "groq"

    def test_init_from_env_var(self):
        """Test initialization from GROQ_API_KEY env var."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "env-test-key"}, clear=False):
            provider = GroqProvider()
            assert provider._api_key == "env-test-key"

    def test_default_base_url(self):
        """Test default base URL is set correctly."""
        provider = GroqProvider(api_key="test-key")
        assert DEFAULT_BASE_URL == "https://api.groq.com/openai/v1"

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        provider = GroqProvider(api_key="test-key", timeout=120)
        assert provider.timeout == 120

    def test_supports_tools(self):
        """Test that provider indicates tool support."""
        provider = GroqProvider(api_key="test-key")
        assert provider.supports_tools() is True

    def test_supports_streaming(self):
        """Test that provider indicates streaming support."""
        provider = GroqProvider(api_key="test-key")
        assert provider.supports_streaming() is True

    def test_provider_name(self):
        """Test provider name property."""
        provider = GroqProvider(api_key="test-key")
        assert provider.name == "groq"


# =============================================================================
# CHAT TESTS
# =============================================================================


class TestGroqProviderChat:
    """Tests for GroqProvider chat method."""

    @pytest.mark.asyncio
    async def test_chat_timeout_error(self, groq_provider, sample_messages):
        """Test handling of timeout error."""
        with patch.object(
            groq_provider,
            "_execute_with_circuit_breaker",
            side_effect=httpx.TimeoutException("Timeout"),
        ):
            with pytest.raises(ProviderTimeoutError) as exc_info:
                await groq_provider.chat(
                    messages=sample_messages,
                    model="llama-3.3-70b-versatile",
                )
            assert "timed out" in str(exc_info.value.message).lower()

    @pytest.mark.asyncio
    async def test_chat_http_error(self, groq_provider, sample_messages):
        """Test handling of HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        error = httpx.HTTPStatusError("Rate limit", request=MagicMock(), response=mock_response)

        with patch.object(groq_provider, "_execute_with_circuit_breaker", side_effect=error):
            with pytest.raises(ProviderError) as exc_info:
                await groq_provider.chat(
                    messages=sample_messages,
                    model="llama-3.3-70b-versatile",
                )
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_chat_generic_error(self, groq_provider, sample_messages):
        """Test handling of generic error."""
        with patch.object(
            groq_provider,
            "_execute_with_circuit_breaker",
            side_effect=Exception("Unknown error"),
        ):
            with pytest.raises(ProviderError) as exc_info:
                await groq_provider.chat(
                    messages=sample_messages,
                    model="llama-3.3-70b-versatile",
                )
            assert "unexpected error" in str(exc_info.value.message).lower()


# =============================================================================
# STREAMING TESTS
# =============================================================================


class TestGroqProviderStreaming:
    """Tests for GroqProvider streaming method."""

    def test_streaming_supported(self, groq_provider):
        """Test that streaming is supported."""
        assert groq_provider.supports_streaming() is True


# =============================================================================
# PAYLOAD LIMITER TESTS
# =============================================================================


class TestGroqPayloadLimiter:
    """Tests for payload limiting in Groq provider."""

    def test_payload_limiter_initialized(self, groq_provider):
        """Test that payload limiter is initialized."""
        assert groq_provider._payload_limiter is not None
        assert groq_provider._payload_limiter.provider_name == "groq"

    def test_payload_limit_configured(self, groq_provider):
        """Test that payload limit is configured correctly (4MB for Groq)."""
        assert groq_provider._payload_limiter.max_payload_bytes == 4 * 1024 * 1024


# =============================================================================
# REQUEST BUILDING TESTS
# =============================================================================


class TestGroqRequestBuilding:
    """Tests for request payload building."""

    def test_payload_limiter_check(self, groq_provider, sample_messages):
        """Test payload limiter check_limit method."""
        ok, warning = groq_provider._payload_limiter.check_limit(sample_messages, None)
        # Small messages should be ok
        assert ok is True


# =============================================================================
# RESPONSE PARSING TESTS
# =============================================================================


class TestGroqResponseParsing:
    """Tests for response parsing."""

    def test_parse_response_basic(self, groq_provider, mock_chat_response):
        """Test parsing basic response."""
        result = groq_provider._parse_response(mock_chat_response, "llama-3.3-70b-versatile")

        assert isinstance(result, CompletionResponse)
        assert result.content == "Hello! How can I help you today?"
        assert result.stop_reason == "stop"

    def test_parse_response_with_tool_calls(self, groq_provider, mock_tool_call_response):
        """Test parsing response with tool calls."""
        result = groq_provider._parse_response(mock_tool_call_response, "llama-3.3-70b-versatile")

        assert isinstance(result, CompletionResponse)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_parse_response_empty_choices(self, groq_provider):
        """Test parsing response with empty choices."""
        response = {"choices": []}
        result = groq_provider._parse_response(response, "test-model")
        assert result.content == ""

    def test_parse_response_no_content(self, groq_provider):
        """Test parsing response with no content."""
        response = {"choices": [{"message": {"role": "assistant"}, "finish_reason": "stop"}]}
        result = groq_provider._parse_response(response, "test-model")
        assert result.content == ""


# =============================================================================
# CLIENT CLEANUP TESTS
# =============================================================================


class TestGroqProviderCleanup:
    """Tests for provider cleanup."""

    @pytest.mark.asyncio
    async def test_client_exists(self, groq_provider):
        """Test that HTTP client is initialized."""
        assert groq_provider.client is not None
        assert isinstance(groq_provider.client, httpx.AsyncClient)
