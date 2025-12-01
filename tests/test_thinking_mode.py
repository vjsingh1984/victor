#!/usr/bin/env python3
"""Test extended thinking mode implementation.

Tests that the thinking mode parameter is properly passed from CLI through
orchestrator to provider, enabling extended reasoning for supported models.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from victor.config.settings import Settings
from victor.agent.orchestrator import AgentOrchestrator
from victor.providers.base import BaseProvider


def create_mock_settings():
    """Create a mock settings object with all required attributes."""
    mock_settings = MagicMock(spec=Settings)
    mock_settings.tool_call_budget = 300
    mock_settings.airgapped_mode = False
    mock_settings.use_semantic_tool_selection = False
    mock_settings.use_mcp_tools = False
    mock_settings.analytics_log_file = "/tmp/test_analytics.jsonl"
    mock_settings.analytics_enabled = False
    mock_settings.load_tool_config.return_value = {}
    return mock_settings


def test_orchestrator_accepts_thinking_parameter():
    """Test that AgentOrchestrator accepts thinking parameter in __init__."""
    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True

    # Create orchestrator with thinking enabled
    orchestrator = AgentOrchestrator(
        settings=mock_settings,
        provider=mock_provider,
        model="claude-3-5-sonnet-20241022",
        thinking=True,
    )

    assert orchestrator.thinking is True


def test_orchestrator_thinking_defaults_to_false():
    """Test that thinking mode defaults to False when not specified."""
    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True

    orchestrator = AgentOrchestrator(
        settings=mock_settings, provider=mock_provider, model="claude-3-5-sonnet-20241022"
    )

    assert orchestrator.thinking is False


@pytest.mark.asyncio
async def test_from_settings_passes_thinking_parameter():
    """Test that from_settings correctly passes thinking parameter to constructor."""
    from victor.config.settings import ProfileConfig

    mock_settings = create_mock_settings()

    # Mock profile
    mock_profile = ProfileConfig(
        provider="anthropic", model="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=4096
    )

    mock_settings.load_profiles.return_value = {"default": mock_profile}
    mock_settings.get_provider_settings.return_value = {"api_key": "test"}

    # Mock provider creation
    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True
    mock_provider.supports_streaming.return_value = True

    with patch("victor.agent.orchestrator.ProviderRegistry.create", return_value=mock_provider):
        orchestrator = await AgentOrchestrator.from_settings(
            settings=mock_settings, profile_name="default", thinking=True
        )

        assert orchestrator.thinking is True


@pytest.mark.asyncio
async def test_chat_passes_thinking_to_provider():
    """Test that chat method passes thinking parameter to provider when enabled."""
    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True
    mock_provider.supports_streaming.return_value = False

    # Mock the chat method
    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_response.tool_calls = None
    mock_provider.chat = AsyncMock(return_value=mock_response)

    orchestrator = AgentOrchestrator(
        settings=mock_settings,
        provider=mock_provider,
        model="claude-3-5-sonnet-20241022",
        thinking=True,
    )

    await orchestrator.chat("Test message")

    # Verify provider.chat was called with thinking parameter
    mock_provider.chat.assert_called_once()
    call_kwargs = mock_provider.chat.call_args[1]

    assert "thinking" in call_kwargs
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}


@pytest.mark.asyncio
async def test_chat_without_thinking_omits_parameter():
    """Test that chat method doesn't pass thinking when disabled."""
    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True
    mock_provider.supports_streaming.return_value = False

    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_response.tool_calls = None
    mock_provider.chat = AsyncMock(return_value=mock_response)

    orchestrator = AgentOrchestrator(
        settings=mock_settings,
        provider=mock_provider,
        model="claude-3-5-sonnet-20241022",
        thinking=False,  # Explicitly disabled
    )

    await orchestrator.chat("Test message")

    # Verify provider.chat was called without thinking parameter
    mock_provider.chat.assert_called_once()
    call_kwargs = mock_provider.chat.call_args[1]

    assert "thinking" not in call_kwargs


@pytest.mark.asyncio
async def test_stream_chat_passes_thinking_to_provider():
    """Test that stream_chat method passes thinking parameter to provider when enabled."""
    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True
    mock_provider.supports_streaming.return_value = True

    # Mock the stream method to return an async generator
    async def mock_stream(*args, **kwargs):
        from victor.providers.base import StreamChunk

        yield StreamChunk(content="Test ", is_final=False, tool_calls=None)
        yield StreamChunk(content="response", is_final=False, tool_calls=None)
        yield StreamChunk(content="", is_final=True, tool_calls=None)

    mock_provider.stream = mock_stream

    orchestrator = AgentOrchestrator(
        settings=mock_settings,
        provider=mock_provider,
        model="claude-3-5-sonnet-20241022",
        thinking=True,
    )

    # Capture stream call arguments using a wrapper
    stream_called = False
    stream_kwargs = {}

    original_stream = mock_provider.stream

    async def stream_wrapper(*args, **kwargs):
        nonlocal stream_called, stream_kwargs
        stream_called = True
        stream_kwargs = kwargs
        async for chunk in original_stream(*args, **kwargs):
            yield chunk

    mock_provider.stream = stream_wrapper

    # Consume the stream
    chunks = []
    async for chunk in orchestrator.stream_chat("Test message"):
        chunks.append(chunk)

    # Verify stream was called with thinking parameter
    assert stream_called
    assert "thinking" in stream_kwargs
    assert stream_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}


@pytest.mark.asyncio
async def test_thinking_mode_anthropic_format():
    """Test that thinking parameter uses correct Anthropic format."""
    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True
    mock_provider.supports_streaming.return_value = False

    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_response.tool_calls = None
    mock_provider.chat = AsyncMock(return_value=mock_response)

    orchestrator = AgentOrchestrator(
        settings=mock_settings,
        provider=mock_provider,
        model="claude-3-5-sonnet-20241022",
        thinking=True,
    )

    await orchestrator.chat("Complex reasoning task")

    call_kwargs = mock_provider.chat.call_args[1]
    thinking_param = call_kwargs.get("thinking")

    # Verify correct Anthropic format
    assert isinstance(thinking_param, dict)
    assert thinking_param["type"] == "enabled"
    assert "budget_tokens" in thinking_param
    assert isinstance(thinking_param["budget_tokens"], int)
    assert thinking_param["budget_tokens"] > 0


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
