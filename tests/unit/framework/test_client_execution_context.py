from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from victor.framework.client import VictorClient
from victor.framework.session_config import SessionConfig
from victor.providers.base import CompletionResponse


@pytest.mark.asyncio
async def test_victor_client_ensure_initialized_captures_execution_context() -> None:
    config = SessionConfig()
    client = VictorClient(config, container=object())
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=None))
    orchestrator = SimpleNamespace(_execution_context=execution_context)

    class _FakeAgent:
        def get_orchestrator(self):
            return orchestrator

    settings = SimpleNamespace(
        provider=SimpleNamespace(default_provider="ollama", default_model="test-model")
    )

    with (
        patch("victor.config.settings.load_settings", return_value=settings),
        patch("victor.framework.agent.Agent.create", new=AsyncMock(return_value=_FakeAgent())),
    ):
        await client._ensure_initialized()

    assert client._context is execution_context


@pytest.mark.asyncio
async def test_victor_client_ensure_initialized_prefers_agent_execution_context_surface() -> None:
    config = SessionConfig()
    client = VictorClient(config, container=object())
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=None))

    class _FakeAgent:
        def __init__(self, ctx):
            self.execution_context = ctx

        def get_orchestrator(self):
            raise AssertionError("execution_context surface should be preferred over orchestrator")

    settings = SimpleNamespace(
        provider=SimpleNamespace(default_provider="ollama", default_model="test-model")
    )

    with (
        patch("victor.config.settings.load_settings", return_value=settings),
        patch(
            "victor.framework.agent.Agent.create",
            new=AsyncMock(return_value=_FakeAgent(execution_context)),
        ),
    ):
        await client._ensure_initialized()

    assert client._context is execution_context


@pytest.mark.asyncio
async def test_victor_client_chat_prefers_execution_context_chat_service() -> None:
    config = SessionConfig.from_cli_flags(tool_budget=4)
    client = VictorClient(config, container=object())
    chat_service = SimpleNamespace(
        chat=AsyncMock(
            return_value=CompletionResponse(
                content="Service response",
                role="assistant",
                tool_calls=[{"name": "read"}],
                model="test-model",
            )
        )
    )
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))
    orchestrator = SimpleNamespace(_execution_context=execution_context)

    class _FakeAgent:
        async def run(self, _message: str):
            raise AssertionError("VictorClient.chat() should prefer execution-context chat service")

        def get_orchestrator(self):
            return orchestrator

    client._agent = _FakeAgent()
    client._context = execution_context

    result = await client.chat("ping")

    chat_service.chat.assert_awaited_once_with("ping", stream=False)
    assert result.content == "Service response"
    assert result.tool_calls == 1
