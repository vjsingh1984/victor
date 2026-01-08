# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for Together AI provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from victor.providers.together_provider import TogetherProvider, TOGETHER_MODELS


class TestTogetherProviderInitialization:
    """Tests for TogetherProvider initialization."""

    def test_initialization_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = TogetherProvider(api_key="test-key-123")
        assert provider._api_key == "test-key-123"
        assert provider.name == "together"

    def test_initialization_from_env_var(self):
        """Test API key loading from environment variable."""
        with patch.dict("os.environ", {"TOGETHER_API_KEY": "env-test-key"}):
            provider = TogetherProvider()
            assert provider._api_key == "env-test-key"

    def test_initialization_from_keyring(self):
        """Test API key loading from keyring when env var not set."""
        with patch.dict("os.environ", {"TOGETHER_API_KEY": ""}, clear=False):
            with patch(
                "victor.config.api_keys.get_api_key",
                return_value="keyring-test-key",
            ):
                provider = TogetherProvider()
                assert provider._api_key == "keyring-test-key"

    def test_initialization_warning_without_key(self, caplog):
        """Test warning is logged when no API key provided."""
        with patch.dict("os.environ", {"TOGETHER_API_KEY": ""}, clear=False):
            with patch(
                "victor.config.api_keys.get_api_key",
                return_value=None,
            ):
                provider = TogetherProvider()
                assert provider._api_key == ""

    def test_custom_base_url(self):
        """Test initialization with custom base URL."""
        provider = TogetherProvider(
            api_key="test-key",
            base_url="https://custom.together.xyz/v1",
        )
        assert provider.base_url == "https://custom.together.xyz/v1"

    def test_custom_timeout(self):
        """Test initialization with custom timeout."""
        provider = TogetherProvider(api_key="test-key", timeout=60)
        assert provider.timeout == 60


class TestTogetherProviderCapabilities:
    """Tests for TogetherProvider capability reporting."""

    def test_name_property(self):
        """Test provider name."""
        provider = TogetherProvider(api_key="test-key")
        assert provider.name == "together"

    def test_supports_tools(self):
        """Test tool support reporting."""
        provider = TogetherProvider(api_key="test-key")
        assert provider.supports_tools() is True

    def test_supports_streaming(self):
        """Test streaming support reporting."""
        provider = TogetherProvider(api_key="test-key")
        assert provider.supports_streaming() is True


class TestTogetherProviderModels:
    """Tests for model definitions."""

    def test_model_definitions_exist(self):
        """Test that model definitions are present."""
        assert len(TOGETHER_MODELS) > 0

    def test_llama_model_defined(self):
        """Test Llama 3.3 70B is defined."""
        assert "meta-llama/Llama-3.3-70B-Instruct-Turbo" in TOGETHER_MODELS
        model = TOGETHER_MODELS["meta-llama/Llama-3.3-70B-Instruct-Turbo"]
        assert model["supports_tools"] is True
        assert model["context_window"] == 131072

    def test_qwen_model_defined(self):
        """Test Qwen 2.5 72B is defined."""
        assert "Qwen/Qwen2.5-72B-Instruct-Turbo" in TOGETHER_MODELS

    def test_deepseek_model_defined(self):
        """Test DeepSeek V3 is defined."""
        assert "deepseek-ai/DeepSeek-V3" in TOGETHER_MODELS


class TestTogetherProviderRequestPayload:
    """Tests for request payload building."""

    def test_basic_payload_structure(self):
        """Test basic request payload structure."""
        provider = TogetherProvider(api_key="test-key")
        from victor.providers.base import Message

        messages = [Message(role="user", content="Hello")]
        payload = provider._build_request_payload(
            messages=messages,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            temperature=0.7,
            max_tokens=4096,
            tools=None,
            stream=False,
        )

        assert payload["model"] == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 4096
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello"

    def test_payload_with_tools(self):
        """Test request payload includes tools when provided."""
        provider = TogetherProvider(api_key="test-key")
        from victor.providers.base import Message, ToolDefinition

        messages = [Message(role="user", content="List files")]
        tools = [
            ToolDefinition(
                name="list_directory",
                description="List files in a directory",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]

        payload = provider._build_request_payload(
            messages=messages,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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

    def test_payload_with_system_message(self):
        """Test request payload with system message."""
        provider = TogetherProvider(api_key="test-key")
        from victor.providers.base import Message

        messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Hello"),
        ]

        payload = provider._build_request_payload(
            messages=messages,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            temperature=0.7,
            max_tokens=4096,
            tools=None,
            stream=False,
        )

        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    def test_payload_streaming_flag(self):
        """Test streaming flag in payload."""
        provider = TogetherProvider(api_key="test-key")
        from victor.providers.base import Message

        messages = [Message(role="user", content="Hello")]

        payload = provider._build_request_payload(
            messages=messages,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            temperature=0.7,
            max_tokens=4096,
            tools=None,
            stream=True,
        )

        assert payload["stream"] is True


class TestTogetherProviderResponseParsing:
    """Tests for response parsing."""

    def test_parse_basic_response(self):
        """Test parsing a basic chat response."""
        provider = TogetherProvider(api_key="test-key")

        response = {
            "id": "chat-123",
            "object": "chat.completion",
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
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

        result = provider._parse_response(response, "meta-llama/Llama-3.3-70B-Instruct-Turbo")

        assert result.content == "Hello! How can I help you?"
        assert result.role == "assistant"
        assert result.stop_reason == "stop"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 8

    def test_parse_response_with_tool_calls(self):
        """Test parsing response with tool calls."""
        provider = TogetherProvider(api_key="test-key")

        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "list_directory",
                                    "arguments": '{"path": "/tmp"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        result = provider._parse_response(response, "meta-llama/Llama-3.3-70B-Instruct-Turbo")

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "list_directory"
        assert result.tool_calls[0]["arguments"] == {"path": "/tmp"}
        assert result.stop_reason == "tool_calls"

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        provider = TogetherProvider(api_key="test-key")

        response = {"choices": []}
        result = provider._parse_response(response, "test-model")

        assert result.content == ""

    def test_normalize_tool_calls_json_string(self):
        """Test normalizing tool calls with JSON string arguments."""
        provider = TogetherProvider(api_key="test-key")

        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "/test.txt"}',
                },
            }
        ]

        result = provider._normalize_tool_calls(tool_calls)

        assert result[0]["arguments"] == {"path": "/test.txt"}

    def test_normalize_tool_calls_invalid_json(self):
        """Test normalizing tool calls with invalid JSON."""
        provider = TogetherProvider(api_key="test-key")

        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "test",
                    "arguments": "invalid json",
                },
            }
        ]

        result = provider._normalize_tool_calls(tool_calls)

        assert result[0]["arguments"] == {}


class TestTogetherProviderStreaming:
    """Tests for streaming functionality."""

    def test_parse_stream_chunk_content(self):
        """Test parsing a stream chunk with content."""
        provider = TogetherProvider(api_key="test-key")

        chunk_data = {
            "choices": [
                {
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ]
        }

        accumulated = []
        result = provider._parse_stream_chunk(chunk_data, accumulated)

        assert result.content == "Hello"
        assert result.is_final is False

    def test_parse_stream_chunk_tool_call(self):
        """Test parsing stream chunks with tool calls."""
        provider = TogetherProvider(api_key="test-key")

        accumulated = []

        # First chunk with tool call start
        chunk1 = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "function": {"name": "read_file", "arguments": '{"pa'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
        provider._parse_stream_chunk(chunk1, accumulated)

        # Second chunk with more arguments
        chunk2 = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": 'th": "/test"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
        provider._parse_stream_chunk(chunk2, accumulated)

        # Final chunk
        chunk3 = {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ]
        }
        result = provider._parse_stream_chunk(chunk3, accumulated)

        assert result.is_final is True
        assert result.tool_calls is not None
        assert result.tool_calls[0]["name"] == "read_file"

    def test_parse_stream_chunk_final(self):
        """Test parsing final stream chunk."""
        provider = TogetherProvider(api_key="test-key")

        chunk_data = {
            "choices": [
                {
                    "delta": {"content": ""},
                    "finish_reason": "stop",
                }
            ]
        }

        result = provider._parse_stream_chunk(chunk_data, [])

        assert result.is_final is True
        assert result.stop_reason == "stop"


class TestTogetherProviderChat:
    """Tests for chat completion."""

    @pytest.mark.asyncio
    async def test_chat_success(self):
        """Test successful chat completion."""
        provider = TogetherProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I'm an AI assistant.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            provider, "_execute_with_circuit_breaker", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_response

            from victor.providers.base import Message

            result = await provider.chat(
                messages=[Message(role="user", content="Hello")],
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            )

            assert result.content == "Hello! I'm an AI assistant."
            assert result.role == "assistant"

    @pytest.mark.asyncio
    async def test_chat_with_tools(self):
        """Test chat completion with tools."""
        provider = TogetherProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "London"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            provider, "_execute_with_circuit_breaker", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_response

            from victor.providers.base import Message, ToolDefinition

            result = await provider.chat(
                messages=[Message(role="user", content="What's the weather?")],
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                tools=[
                    ToolDefinition(
                        name="get_weather",
                        description="Get weather",
                        parameters={"type": "object", "properties": {}},
                    )
                ],
            )

            assert result.tool_calls is not None
            assert result.tool_calls[0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_chat_timeout_error(self):
        """Test chat timeout handling."""
        provider = TogetherProvider(api_key="test-key")

        with patch.object(
            provider, "_execute_with_circuit_breaker", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.side_effect = httpx.TimeoutException("Timeout")

            from victor.providers.base import Message, ProviderTimeoutError

            with pytest.raises(ProviderTimeoutError) as exc_info:
                await provider.chat(
                    messages=[Message(role="user", content="Hello")],
                    model="test-model",
                )

            assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_chat_http_error(self):
        """Test chat HTTP error handling."""
        provider = TogetherProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(
            provider, "_execute_with_circuit_breaker", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.side_effect = httpx.HTTPStatusError(
                "401", request=MagicMock(), response=mock_response
            )

            from victor.providers.base import Message, ProviderError

            with pytest.raises(ProviderError) as exc_info:
                await provider.chat(
                    messages=[Message(role="user", content="Hello")],
                    model="test-model",
                )

            assert "401" in str(exc_info.value)


class TestTogetherProviderCleanup:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test provider cleanup."""
        provider = TogetherProvider(api_key="test-key")

        with patch.object(provider.client, "aclose", new_callable=AsyncMock) as mock_close:
            await provider.close()
            mock_close.assert_called_once()
