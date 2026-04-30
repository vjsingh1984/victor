from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from victor.framework.client import VictorClient
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
