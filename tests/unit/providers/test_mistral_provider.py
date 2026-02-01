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

"""Comprehensive tests for Mistral AI provider.

Tests cover:
- Provider initialization and configuration
- Chat completion with and without tools
- Tool calling format (matches Mistral API spec)
- Streaming responses
- Error handling
"""

import pytest
from unittest.mock import MagicMock, patch
import httpx

from victor.providers.mistral_provider import MistralProvider, MISTRAL_MODELS
from victor.core.errors import (
    ProviderError,
    ProviderTimeoutError,
)
from victor.providers.base import (
    Message,
    ToolDefinition,
)


@pytest.fixture
def mistral_provider():
    """Create MistralProvider instance for testing."""
    return MistralProvider(
        api_key="test-api-key",
        base_url="https://api.mistral.ai/v1",
        timeout=60,
    )


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message(role="user", content="What is 2+2?"),
    ]


@pytest.fixture
def sample_tools():
    """Sample tools for testing."""
    return [
        ToolDefinition(
            name="calculator",
            description="Perform arithmetic calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        ),
    ]


class TestMistralProviderInitialization:
    """Test provider initialization."""

    def test_initialization_with_api_key(self):
        """Test provider initialization with API key."""
        provider = MistralProvider(api_key="test-key")
        assert provider._api_key == "test-key"
        assert provider.name == "mistral"

    def test_initialization_with_custom_base_url(self):
        """Test provider with custom base URL."""
        provider = MistralProvider(
            api_key="test-key",
            base_url="https://custom.mistral.ai/v1",
        )
        assert provider.base_url == "https://custom.mistral.ai/v1"

    def test_initialization_with_timeout(self):
        """Test provider with custom timeout."""
        provider = MistralProvider(api_key="test-key", timeout=180)
        assert provider.timeout == 180

    def test_default_base_url(self):
        """Test default base URL."""
        provider = MistralProvider(api_key="test-key")
        assert provider.base_url == "https://api.mistral.ai/v1"


class TestMistralProviderCapabilities:
    """Test provider capability methods."""

    def test_provider_name(self, mistral_provider):
        """Test provider name property."""
        assert mistral_provider.name == "mistral"

    def test_supports_tools(self, mistral_provider):
        """Test tools support."""
        assert mistral_provider.supports_tools() is True

    def test_supports_streaming(self, mistral_provider):
        """Test streaming support."""
        assert mistral_provider.supports_streaming() is True


class TestMistralModels:
    """Test model definitions."""

    def test_models_have_required_fields(self):
        """Test that all models have required fields."""
        required_fields = ["description", "context_window", "max_output", "supports_tools"]
        for model_id, model_info in MISTRAL_MODELS.items():
            for field in required_fields:
                assert field in model_info, f"Model {model_id} missing field {field}"

    def test_mistral_large_latest(self):
        """Test mistral-large-latest model config."""
        model = MISTRAL_MODELS["mistral-large-latest"]
        assert model["context_window"] == 131072
        assert model["supports_tools"] is True
        assert model["supports_parallel_tools"] is True

    def test_codestral_latest(self):
        """Test codestral-latest model config."""
        model = MISTRAL_MODELS["codestral-latest"]
        assert model["supports_tools"] is True
        assert "code" in model["description"].lower()


class TestMistralRequestPayload:
    """Test request payload building."""

    def test_build_basic_payload(self, mistral_provider, sample_messages):
        """Test basic payload building."""
        payload = mistral_provider._build_request_payload(
            messages=sample_messages,
            model="mistral-small-latest",
            temperature=0.7,
            max_tokens=1000,
            tools=None,
            stream=False,
        )

        assert payload["model"] == "mistral-small-latest"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 1000
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    def test_build_payload_with_tools(self, mistral_provider, sample_messages, sample_tools):
        """Test payload building with tools."""
        payload = mistral_provider._build_request_payload(
            messages=sample_messages,
            model="mistral-large-latest",
            temperature=0.7,
            max_tokens=1000,
            tools=sample_tools,
            stream=False,
        )

        assert "tools" in payload
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["function"]["name"] == "calculator"
        assert payload["tool_choice"] == "auto"

    def test_tool_format_matches_mistral_spec(
        self, mistral_provider, sample_messages, sample_tools
    ):
        """Test that tool format matches Mistral API specification."""
        payload = mistral_provider._build_request_payload(
            messages=sample_messages,
            model="mistral-large-latest",
            temperature=0.7,
            max_tokens=1000,
            tools=sample_tools,
            stream=False,
        )

        tool = payload["tools"][0]
        # Verify Mistral tool format
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]
        assert tool["function"]["parameters"]["type"] == "object"


class TestMistralResponseParsing:
    """Test response parsing."""

    def test_parse_response_basic(self, mistral_provider):
        """Test parsing basic response."""
        raw_response = {
            "id": "chat-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mistral-small-latest",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "4",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
        }

        response = mistral_provider._parse_response(raw_response, "mistral-small-latest")
        assert response.content == "4"
        assert response.role == "assistant"
        assert response.stop_reason == "stop"
        assert response.usage["total_tokens"] == 11

    def test_parse_response_with_tool_calls(self, mistral_provider):
        """Test parsing response with tool calls."""
        raw_response = {
            "id": "chat-123",
            "object": "chat.completion",
            "model": "mistral-large-latest",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expression": "2+2"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        response = mistral_provider._parse_response(raw_response, "mistral-large-latest")
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "calculator"
        assert response.tool_calls[0]["arguments"] == {"expression": "2+2"}

    def test_normalize_tool_calls(self, mistral_provider):
        """Test tool call normalization."""
        raw_tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "test_func",
                    "arguments": '{"param": "value"}',
                },
            }
        ]

        normalized = mistral_provider._normalize_tool_calls(raw_tool_calls)
        assert normalized is not None
        assert len(normalized) == 1
        assert normalized[0]["id"] == "call_123"
        assert normalized[0]["name"] == "test_func"
        assert normalized[0]["arguments"] == {"param": "value"}


class TestMistralStreamParsing:
    """Test stream chunk parsing."""

    def test_parse_stream_chunk_content(self, mistral_provider):
        """Test parsing stream chunk with content."""
        chunk_data = {
            "id": "chat-123",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        accumulated = []
        chunk = mistral_provider._parse_stream_chunk(chunk_data, accumulated)
        assert chunk.content == "Hello"
        assert chunk.is_final is False

    def test_parse_stream_chunk_final(self, mistral_provider):
        """Test parsing final stream chunk."""
        chunk_data = {
            "id": "chat-123",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": "stop",
                }
            ],
        }

        accumulated = []
        chunk = mistral_provider._parse_stream_chunk(chunk_data, accumulated)
        assert chunk.is_final is True
        assert chunk.stop_reason == "stop"

    def test_parse_stream_chunk_tool_calls(self, mistral_provider):
        """Test parsing stream chunk with tool calls."""
        # First chunk with tool call start
        chunk1 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "function": {"name": "calculator", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }

        # Second chunk with arguments
        chunk2 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"expr": "2+2"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }

        # Final chunk
        chunk3 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ],
        }

        accumulated = []
        mistral_provider._parse_stream_chunk(chunk1, accumulated)
        mistral_provider._parse_stream_chunk(chunk2, accumulated)
        result = mistral_provider._parse_stream_chunk(chunk3, accumulated)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculator"


class TestMistralErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_timeout_error(self, mistral_provider, sample_messages):
        """Test timeout error handling."""
        with patch.object(mistral_provider.client, "post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timed out")

            with pytest.raises(ProviderTimeoutError) as exc_info:
                await mistral_provider.chat(
                    messages=sample_messages,
                    model="mistral-small-latest",
                )

            assert "timed out" in str(exc_info.value.message).lower()

    @pytest.mark.asyncio
    async def test_http_error(self, mistral_provider, sample_messages):
        """Test HTTP error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(mistral_provider.client, "post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Unauthorized",
                request=MagicMock(),
                response=mock_response,
            )

            with pytest.raises(ProviderError) as exc_info:
                await mistral_provider.chat(
                    messages=sample_messages,
                    model="mistral-small-latest",
                )

            assert exc_info.value.status_code == 401


class TestMistralListModels:
    """Test list_models functionality."""

    @pytest.mark.asyncio
    async def test_list_models_fallback(self, mistral_provider):
        """Test list_models returns fallback when API fails."""
        with patch.object(mistral_provider.client, "get") as mock_get:
            mock_get.side_effect = Exception("API error")

            models = await mistral_provider.list_models()

            assert len(models) == len(MISTRAL_MODELS)
            model_ids = [m["id"] for m in models]
            assert "mistral-large-latest" in model_ids
            assert "codestral-latest" in model_ids


class TestMistralProviderCleanup:
    """Test provider cleanup."""

    @pytest.mark.asyncio
    async def test_close(self, mistral_provider):
        """Test provider close method."""
        with patch.object(mistral_provider.client, "aclose") as mock_close:
            mock_close.return_value = None
            await mistral_provider.close()
            mock_close.assert_called_once()
