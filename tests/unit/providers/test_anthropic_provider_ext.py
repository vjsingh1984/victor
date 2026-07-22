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
from datetime import datetime, timedelta, timezone

from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.base import (
    ToolDefinition,
)
from victor.workflows.services.credentials import SSOTokens


@pytest.fixture
def anthropic_provider():
    """Create AnthropicProvider instance for testing."""
    return AnthropicProvider(
        api_key="test-api-key",
        base_url="https://api.anthropic.com",
        timeout=30,
        max_retries=0,
        use_circuit_breaker=False,
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
async def test_oauth_mode_uses_claude_code_token_source():
    """Anthropic OAuth mode should consume Claude Code tokens via OAuthTokenManager."""
    tokens = SSOTokens(
        access_token="claude_oauth_token",
        refresh_token="ref",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    with patch("victor.providers.anthropic_provider.OAuthTokenManager") as MockMgr:
        mock_instance = MagicMock()
        mock_instance.get_valid_token = AsyncMock(return_value="claude_oauth_token")
        mock_instance._load_cached = MagicMock(return_value=tokens)
        MockMgr.return_value = mock_instance

        provider = AnthropicProvider(auth_mode="oauth", oauth_source="claude-code")

    MockMgr.assert_called_once_with("anthropic", token_source="claude-code")
    assert provider.client.auth_token == "claude_oauth_token"


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
async def test_close(anthropic_provider):
    """Test closing the provider client."""
    with patch.object(
        anthropic_provider.client,
        "close",
        new_callable=AsyncMock,
    ) as mock_close:
        await anthropic_provider.close()
        mock_close.assert_called_once()
