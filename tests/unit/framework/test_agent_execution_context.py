from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.core.shared_types import ConversationStage
from victor.framework.agent import Agent
from victor.providers.base import CompletionResponse


class AgentOrchestrator:
    """Lightweight orchestrator double that passes Agent type validation."""


def _make_orchestrator(
    *,
    execution_context=None,
    chat_service=None,
):
    orchestrator = AgentOrchestrator()
    orchestrator.provider = SimpleNamespace(name="test_provider")
    orchestrator.model = "test-model"
    orchestrator._execution_context = execution_context
    orchestrator._chat_service = chat_service
    orchestrator._container = None
    orchestrator.chat = AsyncMock(
        return_value=CompletionResponse(
            content="orchestrator response",
            role="assistant",
            model="test-model",
        )
    )
    orchestrator.get_stage = MagicMock(return_value=ConversationStage.INITIAL)
    return orchestrator


def test_agent_exposes_cached_execution_context():
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=object()))
    orchestrator = _make_orchestrator(execution_context=execution_context)

    agent = Agent.from_orchestrator(orchestrator)

    assert agent.execution_context is execution_context


def test_agent_execution_context_refreshes_from_orchestrator():
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=object()))
    orchestrator = _make_orchestrator(execution_context=execution_context)
    agent = Agent.from_orchestrator(orchestrator)
    agent._context = None

    assert agent.execution_context is execution_context


@pytest.mark.asyncio
async def test_agent_run_prefers_execution_context_chat_service():
    runtime_chat_service = SimpleNamespace(
        chat=AsyncMock(
            return_value=CompletionResponse(
                content="runtime service response",
                role="assistant",
                model="runtime-model",
            )
        )
    )
    legacy_chat_service = SimpleNamespace(
        chat=AsyncMock(
            side_effect=AssertionError("legacy orchestrator-bound service should not be used")
        )
    )
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=runtime_chat_service))
    orchestrator = _make_orchestrator(
        execution_context=execution_context,
        chat_service=legacy_chat_service,
    )

    agent = Agent.from_orchestrator(orchestrator)
    result = await agent.run("test prompt")

    runtime_chat_service.chat.assert_awaited_once_with("test prompt")
    assert result.content == "runtime service response"
