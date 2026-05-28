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

"""TDD tests for Qwen provider with OAuth and API-key support."""

import os

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.qwen_provider import (
    QwenProvider,
    QWEN_MODELS,
    QWEN_BASE_URLS,
    QWEN_OAUTH_CONFIG,
)
from victor.providers.base import Message
from victor.workflows.services.credentials import SSOConfig, SSOProvider, SSOTokens
from victor.providers.oauth_manager import OAuthTokenManager, OAUTH_PROVIDERS

# ---------------------------------------------------------------------------
# Qwen constants
# ---------------------------------------------------------------------------


class TestQwenConstants:
    """Test Qwen provider constants."""

    def test_base_urls(self):
        assert (
            QWEN_BASE_URLS["standard"]
            == "https://dashscope.aliyuncs.com/compatible-mode/v1/"
        )
        assert QWEN_BASE_URLS["portal"] == "https://portal.qwen.ai/v1/"

    def test_oauth_config(self):
        assert QWEN_OAUTH_CONFIG["oauth_base_url"] == "https://chat.qwen.ai"
        assert QWEN_OAUTH_CONFIG["api_base_url"] == "https://portal.qwen.ai/v1/"

    def test_models_defined(self):
        assert (
            "qwen3-coder-plus" in QWEN_MODELS
            or "qwen3.5" in QWEN_MODELS
            or len(QWEN_MODELS) > 0
        )


# ---------------------------------------------------------------------------
# Qwen Provider — API key mode
# ---------------------------------------------------------------------------


class TestQwenProviderAPIKey:
    """Test QwenProvider with standard API key auth."""

    @pytest.fixture
    def provider(self):
        return QwenProvider(api_key="test-qwen-key")

    def test_init_with_api_key(self, provider):
        assert provider.client is not None
        assert provider._api_key == "test-qwen-key"

    def test_default_base_url(self, provider):
        assert "dashscope" in str(provider.client.base_url)

    def test_custom_base_url(self):
        p = QwenProvider(
            api_key="test-key",
            base_url="https://custom.qwen.endpoint/v1/",
        )
        assert "custom.qwen.endpoint" in str(p.client.base_url)

    def test_provider_name(self, provider):
        assert provider.name == "qwen"

    def test_supports_tools(self, provider):
        assert provider.supports_tools() is True

    def test_supports_streaming(self, provider):
        assert provider.supports_streaming() is True


# ---------------------------------------------------------------------------
# Qwen Provider — OAuth mode
# ---------------------------------------------------------------------------


class TestQwenProviderOAuth:
    """Test QwenProvider with OAuth subscription auth."""

    @pytest.mark.asyncio
    async def test_oauth_mode_uses_token(self):
        tokens = SSOTokens(
            access_token="qwen_oauth_token",
            refresh_token="qwen_refresh",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.qwen_provider.OAuthTokenManager") as MockMgr:
            mock_instance = MagicMock()
            mock_instance.get_valid_token = AsyncMock(return_value="qwen_oauth_token")
            mock_instance._load_cached = MagicMock(return_value=tokens)
            MockMgr.return_value = mock_instance

            provider = QwenProvider(auth_mode="oauth")

        assert provider._api_key == "qwen_oauth_token"

    def test_oauth_uses_portal_base_url(self):
        """OAuth mode should use portal.qwen.ai, not dashscope."""
        tokens = SSOTokens(
            access_token="qwen_tok",
            refresh_token="ref",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.qwen_provider.OAuthTokenManager") as MockMgr:
            mock_instance = MagicMock()
            mock_instance.get_valid_token = AsyncMock(return_value="qwen_tok")
            mock_instance._load_cached = MagicMock(return_value=tokens)
            MockMgr.return_value = mock_instance

            provider = QwenProvider(auth_mode="oauth")

        assert "portal.qwen.ai" in str(provider.client.base_url)

    @pytest.mark.asyncio
    async def test_ensure_valid_token_refreshes(self):
        tokens = SSOTokens(
            access_token="initial",
            refresh_token="ref",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.qwen_provider.OAuthTokenManager") as MockMgr:
            mock_instance = MagicMock()
            mock_instance.get_valid_token = AsyncMock(return_value="initial")
            mock_instance._load_cached = MagicMock(return_value=tokens)
            MockMgr.return_value = mock_instance

            provider = QwenProvider(auth_mode="oauth")
            mock_instance.get_valid_token = AsyncMock(return_value="refreshed")
            await provider._ensure_valid_token()

        assert provider.client.api_key == "refreshed"


# ---------------------------------------------------------------------------
# Qwen OAuth in OAuthTokenManager registry
# ---------------------------------------------------------------------------


class TestQwenOAuthRegistry:
    """Test Qwen in the OAUTH_PROVIDERS registry."""

    def test_qwen_registered(self):
        assert "qwen" in OAUTH_PROVIDERS

    def test_qwen_config_values(self):
        cfg = OAUTH_PROVIDERS["qwen"]
        assert cfg.provider_name == "qwen"
        assert cfg.sso_provider == SSOProvider.QWEN
        assert "chat.qwen.ai" in cfg.issuer_url

    def test_qwen_to_sso_config(self):
        cfg = OAUTH_PROVIDERS["qwen"]
        with patch.dict(os.environ, {"VICTOR_QWEN_OAUTH_CLIENT_ID": "test-qwen-id"}):
            sso = cfg.to_sso_config()
        assert isinstance(sso, SSOConfig)
        assert sso.provider == SSOProvider.QWEN
        assert sso.use_pkce is True
        assert sso.client_id == "test-qwen-id"


# ---------------------------------------------------------------------------
# Qwen Config Strategy
# ---------------------------------------------------------------------------


class TestQwenConfigStrategy:
    """Test QwenConfig registration in provider config registry."""

    def test_qwen_config_registered(self):
        from victor.config.provider_config_registry import get_provider_config_registry

        registry = get_provider_config_registry()
        providers = registry.list_providers()
        assert "qwen" in providers

    def test_qwen_aliases(self):
        from victor.config.provider_config_registry import get_provider_config_registry

        registry = get_provider_config_registry()
        assert registry._aliases.get("alibaba") == "qwen"
        assert registry._aliases.get("dashscope") == "qwen"

    def test_qwen_default_base_url(self):
        from victor.config.provider_config_registry import QwenConfig

        config = QwenConfig()
        settings = MagicMock()
        result = config.get_settings(settings, {})
        assert "dashscope" in result["base_url"] or "portal" in result["base_url"]


# ---------------------------------------------------------------------------
# Qwen Provider — chat() method
# ---------------------------------------------------------------------------


class TestQwenChat:
    """Test QwenProvider.chat() with mocked OpenAI client."""

    @pytest.fixture
    def provider(self):
        return QwenProvider(api_key="test-qwen-key")

    @pytest.mark.asyncio
    async def test_chat_basic(self, provider):
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from Qwen"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model_dump.return_value = {}

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)
        messages = [Message(role="user", content="Hi")]
        result = await provider.chat(messages=messages, model="qwen3.5")

        assert result.content == "Hello from Qwen"
        assert result.stop_reason == "stop"
        assert result.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, provider):
        from victor.providers.base import ToolDefinition

        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"city": "Beijing"}'

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30
        mock_response.model_dump.return_value = {}

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)
        messages = [Message(role="user", content="Weather?")]
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            )
        ]
        result = await provider.chat(messages=messages, model="qwen3.5", tools=tools)

        assert result.tool_calls is not None
        assert result.tool_calls[0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_chat_auth_error(self, provider):
        from victor.providers.base import ProviderAuthError

        provider.client.chat.completions.create = AsyncMock(
            side_effect=Exception("401 Unauthorized")
        )
        messages = [Message(role="user", content="Hi")]
        with pytest.raises(ProviderAuthError):
            await provider.chat(messages=messages)

    @pytest.mark.asyncio
    async def test_chat_rate_limit_error(self, provider):
        from victor.providers.base import ProviderRateLimitError

        provider.client.chat.completions.create = AsyncMock(
            side_effect=Exception("429 rate limit exceeded")
        )
        messages = [Message(role="user", content="Hi")]
        with pytest.raises(ProviderRateLimitError):
            await provider.chat(messages=messages)

    @pytest.mark.asyncio
    async def test_chat_generic_error(self, provider):
        from victor.providers.base import ProviderError

        provider.client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection timeout")
        )
        messages = [Message(role="user", content="Hi")]
        with pytest.raises(ProviderError):
            await provider.chat(messages=messages)

    @pytest.mark.asyncio
    async def test_chat_empty_choices(self, provider):
        mock_response = MagicMock()
        mock_response.choices = []
        mock_response.usage = None
        mock_response.model_dump.return_value = {}

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)
        messages = [Message(role="user", content="Hi")]
        result = await provider.chat(messages=messages)
        assert result.content == ""


# ---------------------------------------------------------------------------
# Qwen Provider — stream() method
# ---------------------------------------------------------------------------


class TestQwenStream:
    """Test QwenProvider.stream() with mocked OpenAI client."""

    @pytest.fixture
    def provider(self):
        return QwenProvider(api_key="test-qwen-key")

    @pytest.mark.asyncio
    async def test_stream_basic(self, provider):
        async def mock_stream():
            for content, finish in [("Hello", None), (" world", None), ("!", "stop")]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = content
                chunk.choices[0].finish_reason = finish
                yield chunk

        provider.client.chat.completions.create = AsyncMock(return_value=mock_stream())
        messages = [Message(role="user", content="Hi")]
        chunks = []
        async for c in provider.stream(messages=messages, model="qwen3.5"):
            chunks.append(c)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[2].is_final is True

    @pytest.mark.asyncio
    async def test_stream_skips_empty_choices(self, provider):
        async def mock_stream():
            empty = MagicMock()
            empty.choices = []
            yield empty
            real = MagicMock()
            real.choices = [MagicMock()]
            real.choices[0].delta.content = "data"
            real.choices[0].finish_reason = "stop"
            yield real

        provider.client.chat.completions.create = AsyncMock(return_value=mock_stream())
        messages = [Message(role="user", content="Hi")]
        chunks = []
        async for c in provider.stream(messages=messages):
            chunks.append(c)

        assert len(chunks) == 1
        assert chunks[0].content == "data"

    @pytest.mark.asyncio
    async def test_stream_error(self, provider):
        from victor.providers.base import ProviderError

        provider.client.chat.completions.create = AsyncMock(
            side_effect=Exception("stream failed")
        )
        messages = [Message(role="user", content="Hi")]
        with pytest.raises(ProviderError):
            async for _ in provider.stream(messages=messages):
                pass

    @pytest.mark.asyncio
    async def test_stream_with_tools(self, provider):
        from victor.providers.base import ToolDefinition

        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Using tool"
            chunk.choices[0].finish_reason = "stop"
            yield chunk

        provider.client.chat.completions.create = AsyncMock(return_value=mock_stream())
        messages = [Message(role="user", content="Do it")]
        tools = [
            ToolDefinition(
                name="test_tool", description="Test", parameters={"type": "object"}
            )
        ]
        chunks = []
        async for c in provider.stream(messages=messages, tools=tools):
            chunks.append(c)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Qwen Provider — close / ensure_valid_token
# ---------------------------------------------------------------------------


class TestQwenLifecycle:
    """Test lifecycle methods."""

    @pytest.mark.asyncio
    async def test_close(self):
        provider = QwenProvider(api_key="test-key")
        provider.client.close = AsyncMock()
        await provider.close()
        provider.client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_valid_token_noop_for_api_key(self):
        provider = QwenProvider(api_key="test-key")
        # Should not raise — no-op for api_key mode
        await provider._ensure_valid_token()
        assert provider._oauth_manager is None
