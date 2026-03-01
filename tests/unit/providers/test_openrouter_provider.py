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

"""Comprehensive tests for OpenRouter API provider.

Tests cover:
- Provider initialization and configuration
- API key loading from keyring
- Chat completion with and without tools
- Tool calling format (OpenAI-compatible)
- Streaming responses
- Error handling
- Special headers (HTTP-Referer, X-Title)
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from victor.providers.openrouter_provider import OpenRouterProvider, OPENROUTER_MODELS
from victor.providers.base import (
    Message,
    ToolDefinition,
    ProviderError,
    ProviderTimeoutError,
)


@pytest.fixture
def openrouter_provider():
    """Create OpenRouterProvider instance for testing."""
    return OpenRouterProvider(
        api_key="test-api-key",
        base_url="https://openrouter.ai/api/v1",
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


class TestOpenRouterProviderInitialization:
    """Test provider initialization."""

    def test_initialization_with_api_key(self):
        """Test provider initialization with API key."""
        provider = OpenRouterProvider(api_key="test-key")
        assert provider._api_key == "test-key"
        assert provider.name == "openrouter"

    def test_initialization_with_custom_base_url(self):
        """Test provider with custom base URL."""
        provider = OpenRouterProvider(
            api_key="test-key",
            base_url="https://custom.openrouter.ai/v1",
        )
        assert provider.base_url == "https://custom.openrouter.ai/v1"

    def test_initialization_with_timeout(self):
        """Test provider with custom timeout."""
        provider = OpenRouterProvider(api_key="test-key", timeout=180)
        assert provider.timeout == 180

    def test_default_base_url(self):
        """Test default base URL."""
        provider = OpenRouterProvider(api_key="test-key")
        assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_initialization_with_site_headers(self):
        """Test provider with site URL and name headers."""
        provider = OpenRouterProvider(
            api_key="test-key",
            site_url="https://myapp.com",
            site_name="My App",
        )
        # Headers should be set on the client
        assert "HTTP-Referer" in provider.client.headers
        assert "X-Title" in provider.client.headers
        assert provider.client.headers["HTTP-Referer"] == "https://myapp.com"
        assert provider.client.headers["X-Title"] == "My App"

    def test_initialization_from_env_var(self):
        """Test API key loading from environment variable."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-test-key"}):
            provider = OpenRouterProvider()
            assert provider._api_key == "env-test-key"

    def test_initialization_from_keyring(self):
        """Test API key loading from keyring when env var not set."""
        from victor.providers.resolution import KeySource

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False):
            # Mock the UnifiedApiKeyResolver to return a key
            mock_result = MagicMock()
            mock_result.key = "keyring-test-key"
            mock_result.source = "keyring"
            mock_result.source_detail = "System keyring"
            mock_result.sources_attempted = [
                KeySource(source="explicit", description="Explicit parameter", found=False),
                KeySource(source="environment", description="OPENROUTER_API_KEY env var", found=False),
                KeySource(source="keyring", description="System keyring", found=True),
            ]
            mock_result.non_interactive = False

            with patch(
                "victor.providers.resolution.UnifiedApiKeyResolver.get_api_key",
                return_value=mock_result,
            ):
                provider = OpenRouterProvider()
                assert provider._api_key == "keyring-test-key"


class TestOpenRouterProviderCapabilities:
    """Test provider capability methods."""

    def test_provider_name(self, openrouter_provider):
        """Test provider name property."""
        assert openrouter_provider.name == "openrouter"

    def test_supports_tools(self, openrouter_provider):
        """Test tools support."""
        assert openrouter_provider.supports_tools() is True

    def test_supports_streaming(self, openrouter_provider):
        """Test streaming support."""
        assert openrouter_provider.supports_streaming() is True


class TestOpenRouterModels:
    """Test model definitions."""

    def test_models_have_required_fields(self):
        """Test that all models have required fields."""
        required_fields = ["description", "context_window", "supports_tools"]
        for model_id, model_info in OPENROUTER_MODELS.items():
            for field in required_fields:
                assert field in model_info, f"Model {model_id} missing field {field}"

    def test_free_models_marked(self):
        """Test that free models are properly marked."""
        free_models = [m for m, info in OPENROUTER_MODELS.items() if info.get("free")]
        assert len(free_models) >= 3  # Should have multiple free models

    def test_gemini_flash_free(self):
        """Test Gemini 2.5 Flash is free with tools."""
        model = OPENROUTER_MODELS.get("google/gemini-2.5-flash:free")
        assert model is not None
        assert model.get("free") is True
        assert model.get("supports_tools") is True


class TestOpenRouterRequestPayload:
    """Test request payload building."""

    def test_build_basic_payload(self, openrouter_provider, sample_messages):
        """Test basic payload building."""
        payload = openrouter_provider._build_request_payload(
            messages=sample_messages,
            model="meta-llama/llama-3.2-3b-instruct:free",
            temperature=0.7,
            max_tokens=1000,
            tools=None,
            stream=False,
        )

        assert payload["model"] == "meta-llama/llama-3.2-3b-instruct:free"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 1000
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    def test_build_payload_with_tools(self, openrouter_provider, sample_messages, sample_tools):
        """Test payload building with tools."""
        payload = openrouter_provider._build_request_payload(
            messages=sample_messages,
            model="openai/gpt-4o",
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

    def test_tool_format_matches_openai_spec(
        self, openrouter_provider, sample_messages, sample_tools
    ):
        """Test that tool format matches OpenAI API specification."""
        payload = openrouter_provider._build_request_payload(
            messages=sample_messages,
            model="openai/gpt-4o",
            temperature=0.7,
            max_tokens=1000,
            tools=sample_tools,
            stream=False,
        )

        tool = payload["tools"][0]
        # Verify OpenAI tool format
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]
        assert tool["function"]["parameters"]["type"] == "object"


class TestOpenRouterResponseParsing:
    """Test response parsing."""

    def test_parse_response_basic(self, openrouter_provider):
        """Test parsing basic response."""
        raw_response = {
            "id": "gen-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "meta-llama/llama-3.2-3b-instruct:free",
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

        response = openrouter_provider._parse_response(
            raw_response, "meta-llama/llama-3.2-3b-instruct:free"
        )
        assert response.content == "4"
        assert response.role == "assistant"
        assert response.stop_reason == "stop"
        assert response.usage["total_tokens"] == 11

    def test_parse_response_with_tool_calls(self, openrouter_provider):
        """Test parsing response with tool calls."""
        raw_response = {
            "id": "gen-123",
            "object": "chat.completion",
            "model": "openai/gpt-4o",
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

        response = openrouter_provider._parse_response(raw_response, "openai/gpt-4o")
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "calculator"
        assert response.tool_calls[0]["arguments"] == {"expression": "2+2"}

    def test_normalize_tool_calls(self, openrouter_provider):
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

        normalized = openrouter_provider._normalize_tool_calls(raw_tool_calls)
        assert normalized is not None
        assert len(normalized) == 1
        assert normalized[0]["id"] == "call_123"
        assert normalized[0]["name"] == "test_func"
        assert normalized[0]["arguments"] == {"param": "value"}


class TestOpenRouterStreamParsing:
    """Test stream chunk parsing."""

    def test_parse_stream_chunk_content(self, openrouter_provider):
        """Test parsing stream chunk with content."""
        chunk_data = {
            "id": "gen-123",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        accumulated = []
        chunk = openrouter_provider._parse_stream_chunk(chunk_data, accumulated)
        assert chunk.content == "Hello"
        assert chunk.is_final is False

    def test_parse_stream_chunk_final(self, openrouter_provider):
        """Test parsing final stream chunk."""
        chunk_data = {
            "id": "gen-123",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": "stop",
                }
            ],
        }

        accumulated = []
        chunk = openrouter_provider._parse_stream_chunk(chunk_data, accumulated)
        assert chunk.is_final is True
        assert chunk.stop_reason == "stop"

    def test_parse_stream_chunk_tool_calls(self, openrouter_provider):
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
        openrouter_provider._parse_stream_chunk(chunk1, accumulated)
        openrouter_provider._parse_stream_chunk(chunk2, accumulated)
        result = openrouter_provider._parse_stream_chunk(chunk3, accumulated)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculator"


class TestOpenRouterErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_timeout_error(self, openrouter_provider, sample_messages):
        """Test timeout error handling."""
        with patch.object(openrouter_provider.client, "post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timed out")

            with pytest.raises(ProviderTimeoutError) as exc_info:
                await openrouter_provider.chat(
                    messages=sample_messages,
                    model="meta-llama/llama-3.2-3b-instruct:free",
                )

            assert "timed out" in str(exc_info.value.message).lower()

    @pytest.mark.asyncio
    async def test_http_error(self, openrouter_provider, sample_messages):
        """Test HTTP error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(openrouter_provider.client, "post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Unauthorized",
                request=MagicMock(),
                response=mock_response,
            )

            with pytest.raises(ProviderError) as exc_info:
                await openrouter_provider.chat(
                    messages=sample_messages,
                    model="meta-llama/llama-3.2-3b-instruct:free",
                )

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, openrouter_provider, sample_messages):
        """Test rate limit (429) error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        with patch.object(openrouter_provider.client, "post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Rate limit exceeded",
                request=MagicMock(),
                response=mock_response,
            )

            with pytest.raises(ProviderError) as exc_info:
                await openrouter_provider.chat(
                    messages=sample_messages,
                    model="meta-llama/llama-3.2-3b-instruct:free",
                )

            assert exc_info.value.status_code == 429


class TestOpenRouterListModels:
    """Test list_models functionality."""

    @pytest.mark.asyncio
    async def test_list_models_fallback(self, openrouter_provider):
        """Test list_models returns fallback when API fails."""
        with patch.object(openrouter_provider.client, "get") as mock_get:
            mock_get.side_effect = Exception("API error")

            models = await openrouter_provider.list_models()

            assert len(models) == len(OPENROUTER_MODELS)
            model_ids = [m["id"] for m in models]
            assert "google/gemini-2.5-flash:free" in model_ids


class TestOpenRouterProviderCleanup:
    """Test provider cleanup."""

    @pytest.mark.asyncio
    async def test_close(self, openrouter_provider):
        """Test provider close method."""
        with patch.object(openrouter_provider.client, "aclose") as mock_close:
            mock_close.return_value = None
            await openrouter_provider.close()
            mock_close.assert_called_once()
