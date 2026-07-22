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

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


from victor.providers.openai_provider import OpenAIProvider


@pytest.fixture
def openai_provider():
    """Create OpenAIProvider instance for testing."""
    return OpenAIProvider(
        api_key="test-api-key",
        organization="test-org",
        base_url="https://api.openai.com/v1",
        timeout=30,
        max_retries=0,
        use_circuit_breaker=False,
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
async def test_close(openai_provider):
    """Test closing the provider client."""
    with patch.object(
        openai_provider.client,
        "close",
        new_callable=AsyncMock,
    ) as mock_close:
        await openai_provider.close()
        mock_close.assert_called_once()


def _mock_chat_response():
    msg = MagicMock()
    msg.content = "ok"
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    resp.model_dump = lambda: {}
    return resp


@pytest.mark.asyncio
async def test_supports_reasoning_effort(openai_provider):
    assert openai_provider.supports_reasoning_effort("o3-mini") is True
    assert openai_provider.supports_reasoning_effort("gpt-5.1") is True
    assert openai_provider.supports_reasoning_effort("gpt-4o") is False
    assert openai_provider.supports_reasoning_effort(None) is False
