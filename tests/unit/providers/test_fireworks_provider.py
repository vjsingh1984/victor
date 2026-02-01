# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for Fireworks AI provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from victor.providers.fireworks_provider import FireworksProvider, FIREWORKS_MODELS


class TestFireworksProviderInitialization:
    """Tests for FireworksProvider initialization."""

    def test_initialization_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = FireworksProvider(api_key="test-key-123")
        assert provider._api_key == "test-key-123"
        assert provider.name == "fireworks"

    def test_initialization_from_env_var(self):
        """Test API key loading from environment variable."""
        with patch.dict("os.environ", {"FIREWORKS_API_KEY": "env-test-key"}):
            provider = FireworksProvider()
            assert provider._api_key == "env-test-key"

    def test_initialization_from_keyring(self):
        """Test API key loading from keyring when env var not set."""
        with patch.dict("os.environ", {"FIREWORKS_API_KEY": ""}, clear=False):
            with patch(
                "victor.config.api_keys.get_api_key",
                return_value="keyring-test-key",
            ):
                provider = FireworksProvider()
                assert provider._api_key == "keyring-test-key"

    def test_initialization_warning_without_key(self, caplog):
        """Test warning is logged when no API key provided."""
        with patch.dict("os.environ", {"FIREWORKS_API_KEY": ""}, clear=False):
            with patch(
                "victor.config.api_keys.get_api_key",
                return_value=None,
            ):
                provider = FireworksProvider()
                assert provider._api_key == ""

    def test_custom_base_url(self):
        """Test initialization with custom base URL."""
        provider = FireworksProvider(
            api_key="test-key",
            base_url="https://custom.fireworks.ai/v1",
        )
        assert provider.base_url == "https://custom.fireworks.ai/v1"

    def test_default_base_url(self):
        """Test default base URL."""
        provider = FireworksProvider(api_key="test-key")
        assert "api.fireworks.ai" in provider.base_url


class TestFireworksProviderCapabilities:
    """Tests for FireworksProvider capability reporting."""

    def test_name_property(self):
        """Test provider name."""
        provider = FireworksProvider(api_key="test-key")
        assert provider.name == "fireworks"

    def test_supports_tools(self):
        """Test tool support reporting."""
        provider = FireworksProvider(api_key="test-key")
        assert provider.supports_tools() is True

    def test_supports_streaming(self):
        """Test streaming support reporting."""
        provider = FireworksProvider(api_key="test-key")
        assert provider.supports_streaming() is True


class TestFireworksProviderModels:
    """Tests for model definitions."""

    def test_model_definitions_exist(self):
        """Test that model definitions are present."""
        assert len(FIREWORKS_MODELS) > 0

    def test_llama_model_defined(self):
        """Test Llama 3.3 70B is defined."""
        assert "accounts/fireworks/models/llama-v3p3-70b-instruct" in FIREWORKS_MODELS

    def test_qwen_coder_model_defined(self):
        """Test Qwen3 Coder is defined."""
        assert "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct" in FIREWORKS_MODELS

    def test_deepseek_model_defined(self):
        """Test DeepSeek V3.2 is defined."""
        assert "accounts/fireworks/models/deepseek-v3p2" in FIREWORKS_MODELS

    def test_all_models_support_tools(self):
        """Test all defined models support tools."""
        for model_id, model_info in FIREWORKS_MODELS.items():
            assert model_info.get("supports_tools") is True, f"{model_id} should support tools"


class TestFireworksProviderRequestPayload:
    """Tests for request payload building."""

    def test_basic_payload_structure(self):
        """Test basic request payload structure."""
        provider = FireworksProvider(api_key="test-key")
        from victor.providers.base import Message

        messages = [Message(role="user", content="Hello")]
        payload = provider._build_request_payload(
            messages=messages,
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            temperature=0.7,
            max_tokens=4096,
            tools=None,
            stream=False,
        )

        assert payload["model"] == "accounts/fireworks/models/llama-v3p3-70b-instruct"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 4096
        assert payload["stream"] is False

    def test_payload_with_tools(self):
        """Test request payload includes tools when provided."""
        provider = FireworksProvider(api_key="test-key")
        from victor.providers.base import Message, ToolDefinition

        messages = [Message(role="user", content="Search code")]
        tools = [
            ToolDefinition(
                name="code_search",
                description="Search for code patterns",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]

        payload = provider._build_request_payload(
            messages=messages,
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            temperature=0.7,
            max_tokens=4096,
            tools=tools,
            stream=False,
        )

        assert "tools" in payload
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["function"]["name"] == "code_search"
        assert payload["tool_choice"] == "auto"

    def test_payload_with_tool_message(self):
        """Test request payload with tool result message."""
        provider = FireworksProvider(api_key="test-key")
        from victor.providers.base import Message

        messages = [
            Message(role="user", content="List files"),
            Message(
                role="assistant",
                content="",  # Empty string for assistant with tool calls
                tool_calls=[{"id": "call_1", "name": "list_dir", "arguments": {}}],
            ),
            Message(role="tool", content="file1.py\nfile2.py", tool_call_id="call_1"),
        ]

        payload = provider._build_request_payload(
            messages=messages,
            model="test-model",
            temperature=0.7,
            max_tokens=4096,
            tools=None,
            stream=False,
        )

        assert len(payload["messages"]) == 3
        # Check tool call in assistant message
        assert "tool_calls" in payload["messages"][1]


class TestFireworksProviderResponseParsing:
    """Tests for response parsing."""

    def test_parse_basic_response(self):
        """Test parsing a basic chat response."""
        provider = FireworksProvider(api_key="test-key")

        response = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I can help with that!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 6,
                "total_tokens": 21,
            },
        }

        result = provider._parse_response(response, "test-model")

        assert result.content == "I can help with that!"
        assert result.role == "assistant"
        assert result.stop_reason == "stop"
        assert result.usage["total_tokens"] == 21

    def test_parse_response_with_tool_calls(self):
        """Test parsing response with tool calls."""
        provider = FireworksProvider(api_key="test-key")

        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_fw_123",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "main.py"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        result = provider._parse_response(response, "test-model")

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[0]["arguments"] == {"path": "main.py"}

    def test_parse_multiple_tool_calls(self):
        """Test parsing response with multiple tool calls."""
        provider = FireworksProvider(api_key="test-key")

        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "b.py"}'},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        result = provider._parse_response(response, "test-model")

        assert len(result.tool_calls) == 2


class TestFireworksProviderStreaming:
    """Tests for streaming functionality."""

    def test_parse_stream_chunk_content(self):
        """Test parsing a stream chunk with content."""
        provider = FireworksProvider(api_key="test-key")

        chunk_data = {
            "choices": [
                {
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ]
        }

        result = provider._parse_stream_chunk(chunk_data, [])

        assert result.content == "Hello"
        assert result.is_final is False

    def test_parse_stream_chunk_final(self):
        """Test parsing final stream chunk."""
        provider = FireworksProvider(api_key="test-key")

        chunk_data = {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                }
            ]
        }

        result = provider._parse_stream_chunk(chunk_data, [])

        assert result.is_final is True
        assert result.stop_reason == "stop"

    def test_parse_stream_empty_choices(self):
        """Test parsing stream chunk with empty choices."""
        provider = FireworksProvider(api_key="test-key")

        chunk_data = {"choices": []}
        result = provider._parse_stream_chunk(chunk_data, [])

        assert result.content == ""
        assert result.is_final is False


class TestFireworksProviderChat:
    """Tests for chat completion."""

    @pytest.mark.asyncio
    async def test_chat_success(self):
        """Test successful chat completion."""
        provider = FireworksProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Here's my response.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            provider, "_execute_with_circuit_breaker", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_response

            from victor.providers.base import Message

            result = await provider.chat(
                messages=[Message(role="user", content="Hello")],
                model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            )

            assert result.content == "Here's my response."

    @pytest.mark.asyncio
    async def test_chat_timeout_error(self):
        """Test chat timeout handling."""
        provider = FireworksProvider(api_key="test-key")

        with patch.object(
            provider, "_execute_with_circuit_breaker", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.side_effect = httpx.TimeoutException("Timeout")

from victor.core.errors import ProviderTimeoutError
from victor.providers.base import Message

            with pytest.raises(ProviderTimeoutError) as exc_info:
                await provider.chat(
                    messages=[Message(role="user", content="Hello")],
                    model="test-model",
                )

            assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_chat_rate_limit_error(self):
        """Test chat rate limit handling (429)."""
        provider = FireworksProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        with patch.object(
            provider, "_execute_with_circuit_breaker", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.side_effect = httpx.HTTPStatusError(
                "429", request=MagicMock(), response=mock_response
            )

from victor.core.errors import ProviderError
from victor.providers.base import Message

            with pytest.raises(ProviderError) as exc_info:
                await provider.chat(
                    messages=[Message(role="user", content="Hello")],
                    model="test-model",
                )

            assert "429" in str(exc_info.value)


class TestFireworksProviderCleanup:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test provider cleanup."""
        provider = FireworksProvider(api_key="test-key")

        with patch.object(provider.client, "aclose", new_callable=AsyncMock) as mock_close:
            await provider.close()
            mock_close.assert_called_once()
