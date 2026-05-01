from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.client import VictorClient
from victor.framework.events import AgentExecutionEvent, EventType
from victor.framework.session_config import SessionConfig


@pytest.mark.asyncio
async def test_victor_client_ensure_initialized_uses_provider_defaults() -> None:
    config = SessionConfig.from_cli_flags(tool_budget=4)
    client = VictorClient(config, container=object())
    settings = SimpleNamespace(
        provider=SimpleNamespace(
            default_provider="ollama",
            default_model="mistral-tools:7b-instruct",
        )
    )
    mock_agent = object()

    with (
        patch("victor.config.settings.load_settings", return_value=settings),
        patch(
            "victor.framework.agent.Agent.create", new=AsyncMock(return_value=mock_agent)
        ) as create,
    ):
        agent = await client._ensure_initialized()

    assert agent is mock_agent
    create.assert_awaited_once_with(
        profile=None,  # agent_profile from SessionConfig (defaults to None)
        provider="ollama",
        model="mistral-tools:7b-instruct",
        session_config=config,
    )


@pytest.mark.asyncio
async def test_victor_client_does_not_clobber_profile_with_settings_defaults() -> None:
    config = SessionConfig.from_cli_flags(agent_profile="zai-coding")
    client = VictorClient(config, container=object())
    settings = SimpleNamespace(
        provider=SimpleNamespace(
            default_provider="ollama",
            default_model="qwen3.5:27b-q4_K_M",
        )
    )
    mock_agent = object()

    with (
        patch("victor.config.settings.load_settings", return_value=settings),
        patch(
            "victor.framework.agent.Agent.create", new=AsyncMock(return_value=mock_agent)
        ) as create,
    ):
        agent = await client._ensure_initialized()

    assert agent is mock_agent
    create.assert_awaited_once_with(
        profile="zai-coding",
        provider=None,
        model=None,
        session_config=config,
    )


def test_session_config_normalizes_provider_override_and_applies_endpoint() -> None:
    config = SessionConfig.from_cli_flags(
        provider="OLLAMA",
        endpoint="http://localhost:11434",
        provider_timeout=180,
        observability_logging=True,
        auto_skill_enabled=False,
        one_shot_mode=True,
    )
    settings = SimpleNamespace(
        provider=SimpleNamespace(
            default_provider="anthropic",
            default_model="claude-sonnet",
            ollama_base_url="http://old-host:11434",
            timeout=60,
        ),
        tool_settings=SimpleNamespace(
            tool_output_preview_enabled=True,
            tool_output_pruning_enabled=False,
            tool_output_pruning_safe_only=True,
        ),
        observability=SimpleNamespace(enable_observability_logging=False),
        automation=SimpleNamespace(one_shot_mode=False),
        enable_observability_logging=False,
        skill_auto_select_enabled=True,
        one_shot_mode=False,
    )

    config.apply_to_settings(settings)

    assert config.provider_override.provider == "ollama"
    assert config.provider_override.model == "qwen2.5-coder:7b"
    assert config.provider_override.timeout == 180
    assert settings.provider.default_provider == "ollama"
    assert settings.provider.default_model == "qwen2.5-coder:7b"
    assert settings.provider.ollama_base_url == "http://localhost:11434"
    assert settings.provider.timeout == 180
    assert settings.observability.enable_observability_logging is True
    assert settings.enable_observability_logging is True
    assert settings.skill_auto_select_enabled is False
    assert settings.automation.one_shot_mode is True
    assert settings.one_shot_mode is True


def test_session_config_rejects_endpoint_for_cloud_provider() -> None:
    with pytest.raises(ValueError, match="--endpoint is only supported for local providers"):
        SessionConfig.from_cli_flags(provider="openai", endpoint="https://custom.example")


def test_victor_client_bootstrap_container_uses_bootstrap_factory_result() -> None:
    client = VictorClient(SessionConfig())
    sentinel_container = object()

    with patch(
        "victor.core.bootstrap.bootstrap_container",
        return_value=sentinel_container,
    ) as bootstrap_container:
        result = client._bootstrap_container()

    assert result is sentinel_container
    bootstrap_container.assert_called_once_with()


@pytest.mark.asyncio
async def test_agent_public_api_for_runtime_configuration() -> None:
    """Test Agent exposes runtime configuration methods for CLI integration."""
    from victor.framework.agent import Agent

    # Mock orchestrator with unified_tracker and provider
    mock_tracker = MagicMock()
    mock_provider = MagicMock()
    mock_provider.supports_streaming.return_value = True

    mock_orchestrator = MagicMock()
    mock_orchestrator.unified_tracker = mock_tracker
    mock_orchestrator.provider = mock_provider
    # Make the mock pass Agent's type validation
    mock_orchestrator.__class__.__name__ = "AgentOrchestrator"

    # Create agent via from_orchestrator (escape hatch)
    agent = Agent.from_orchestrator(mock_orchestrator)

    # Test set_tool_budget
    agent.set_tool_budget(50, user_override=True)
    mock_tracker.set_tool_budget.assert_called_once_with(50, user_override=True)

    # Test set_max_iterations
    agent.set_max_iterations(20, user_override=True)
    mock_tracker.set_max_iterations.assert_called_once_with(20, user_override=True)

    # Test supports_streaming
    assert agent.supports_streaming() is True
    mock_provider.supports_streaming.assert_called_once()


@pytest.mark.asyncio
async def test_agent_public_api_delegates_to_orchestrator_internals() -> None:
    """Test Agent methods correctly delegate to orchestrator internals."""
    from victor.framework.agent import Agent

    # Mock orchestrator with unified_tracker and provider
    mock_tracker = MagicMock()
    mock_provider = MagicMock()
    mock_provider.supports_streaming.return_value = False

    mock_orchestrator = MagicMock()
    mock_orchestrator.unified_tracker = mock_tracker
    mock_orchestrator.provider = mock_provider
    mock_orchestrator.__class__.__name__ = "AgentOrchestrator"

    agent = Agent.from_orchestrator(mock_orchestrator)

    # Test that methods delegate correctly
    agent.set_tool_budget(100)
    mock_tracker.set_tool_budget.assert_called_once_with(100, user_override=False)

    agent.set_max_iterations(30)
    mock_tracker.set_max_iterations.assert_called_once_with(30, user_override=False)

    assert agent.supports_streaming() is False
    mock_provider.supports_streaming.assert_called_once()


@pytest.mark.asyncio
async def test_victor_client_stream_preserves_framework_event_contract() -> None:
    config = SessionConfig()
    client = VictorClient(config, container=object())

    class _FakeAgent:
        async def stream(self, _message: str):
            yield AgentExecutionEvent(type=EventType.CONTENT, content="READY")
            yield AgentExecutionEvent(
                type=EventType.TOOL_RESULT,
                tool_name="database",
                arguments={"sql": "select 1"},
                result="row",
                metadata={"elapsed": 0.02},
            )

    client._agent = _FakeAgent()

    events = [event async for event in client.stream("ping")]

    assert events[0].type == EventType.CONTENT
    assert events[0].event_type == "content"
    assert events[0].content == "READY"

    assert events[1].type == EventType.TOOL_RESULT
    assert events[1].event_type == "tool_result"
    assert events[1].tool_name == "database"
    assert events[1].arguments == {"sql": "select 1"}
    assert events[1].result["result"] == "row"
    assert events[1].result["arguments"] == {"sql": "select 1"}
