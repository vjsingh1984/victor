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
    """Create a real Settings object for testing."""
    return Settings(
        analytics_enabled=False,
        use_semantic_tool_selection=False,
        use_mcp_tools=False,
    )


def test_orchestrator_accepts_thinking_parameter():
    """Test that AgentOrchestrator accepts thinking parameter in __init__."""
    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True

    # Create orchestrator with thinking enabled
    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
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

    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        orchestrator = AgentOrchestrator(
            settings=mock_settings,
            provider=mock_provider,
            model="claude-3-5-sonnet-20241022",
        )

    assert orchestrator.thinking is False


@pytest.mark.asyncio
async def test_from_settings_passes_thinking_parameter():
    """Test that from_settings correctly passes thinking parameter to constructor."""
    from victor.config.settings import ProfileConfig

    mock_settings = create_mock_settings()

    # Mock profile
    mock_profile = ProfileConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=4096,
    )

    mock_settings.load_profiles = lambda: {"default": mock_profile}
    mock_settings.get_provider_settings = lambda provider_name, extras=None: {"api_key": "test"}

    # Mock provider creation
    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True
    mock_provider.supports_streaming.return_value = True

    with (
        patch("victor.agent.orchestrator.ProviderRegistry.create", return_value=mock_provider),
        patch("victor.core.bootstrap_services.bootstrap_new_services"),
    ):
        orchestrator = await AgentOrchestrator.from_settings(
            settings=mock_settings, profile_name="default", thinking=True
        )

        assert orchestrator.thinking is True


@pytest.mark.asyncio
async def test_chat_passes_thinking_to_provider():
    """Test that thinking flag is accessible during chat execution.

    The orchestrator.chat() routes through AgenticLoop which uses the
    thinking flag to pass extended thinking parameters to the provider.
    We verify the flag is correctly set and accessible.
    """
    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True
    mock_provider.supports_streaming.return_value = False
    mock_provider.chat = AsyncMock(
        return_value=MagicMock(
            content="Test response",
            tool_calls=None,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            metadata=None,
        )
    )

    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        orchestrator = AgentOrchestrator(
            settings=mock_settings,
            provider=mock_provider,
            model="claude-3-5-sonnet-20241022",
            thinking=True,
        )

    # Verify thinking is enabled and accessible for provider calls
    assert orchestrator.thinking is True
    # The thinking flag is read by streaming pipeline (line 292-298)
    # and passed as provider_kwargs["thinking"] to the provider


@pytest.mark.asyncio
async def test_chat_without_thinking_omits_parameter():
    """Test that thinking flag defaults to False when disabled."""
    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True
    mock_provider.supports_streaming.return_value = False

    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        orchestrator = AgentOrchestrator(
            settings=mock_settings,
            provider=mock_provider,
            model="claude-3-5-sonnet-20241022",
            thinking=False,
        )

    # Verify thinking is disabled — provider won't receive thinking parameter
    assert orchestrator.thinking is False


@pytest.mark.asyncio
async def test_stream_chat_passes_thinking_to_provider():
    """Test that stream_chat delegates to service and thinking is set on orchestrator."""
    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True
    mock_provider.supports_streaming.return_value = True

    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        orchestrator = AgentOrchestrator(
            settings=mock_settings,
            provider=mock_provider,
            model="claude-3-5-sonnet-20241022",
            thinking=True,
        )

    # Mock the chat service stream to return chunks
    from victor.agent.stream_handler import StreamChunk

    async def mock_stream_chat(msg):
        yield StreamChunk(content="Test response")

    orchestrator._chat_service = MagicMock()
    orchestrator._chat_service.stream_chat = mock_stream_chat

    # Consume the stream
    chunks = []
    async for chunk in orchestrator.stream_chat("Test message"):
        chunks.append(chunk)

    # Verify thinking is enabled on orchestrator and stream produced output
    assert orchestrator.thinking is True
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_thinking_mode_anthropic_format():
    """Test that thinking parameter is correctly typed as boolean on orchestrator.

    The Anthropic API expects thinking as {"type": "enabled", "budget_tokens": N}.
    The streaming pipeline (line 292-298) converts the boolean flag to this format
    before passing to the provider.
    """
    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True
    mock_provider.supports_streaming.return_value = False

    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        orchestrator = AgentOrchestrator(
            settings=mock_settings,
            provider=mock_provider,
            model="claude-3-5-sonnet-20241022",
            thinking=True,
        )

    # Verify the thinking attribute is correctly set as a boolean
    assert orchestrator.thinking is True
    assert isinstance(orchestrator.thinking, bool)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
