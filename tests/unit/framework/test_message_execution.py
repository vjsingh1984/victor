from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.agent import Agent
from victor.framework.client import VictorClient
from victor.framework.events import AgentExecutionEvent, EventType
from victor.framework.session_config import SessionConfig
from victor.framework.task import TaskResult
from victor.providers.base import CompletionResponse


class AgentOrchestrator(MagicMock):
    """MagicMock subclass whose type name satisfies Agent.__init__ validation."""


def _make_orchestrator() -> AgentOrchestrator:
    orchestrator = AgentOrchestrator()
    orchestrator.model = "test-model"
    orchestrator.provider = MagicMock()
    orchestrator.provider.name = "test-provider"
    orchestrator.get_stage = MagicMock(return_value=SimpleNamespace(value="INITIAL"))
    orchestrator._container = MagicMock()
    return orchestrator


def test_prepare_message_prepends_context_but_preserves_response_message() -> None:
    from victor.framework.message_execution import prepare_message

    prepared = prepare_message("Fix the bug", {"file": "auth.py"})

    assert "File: auth.py" in prepared.runtime_message
    assert prepared.runtime_message.endswith("Fix the bug")
    assert prepared.response_message == "Fix the bug"


def test_resolve_chat_runtime_prefers_execution_context_chat_service() -> None:
    from victor.framework.message_execution import resolve_chat_runtime

    orchestrator = _make_orchestrator()
    chat_service = MagicMock()
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))

    runtime = resolve_chat_runtime(orchestrator, execution_context)

    assert runtime is chat_service


@pytest.mark.asyncio
async def test_execute_message_normalizes_direct_response_output() -> None:
    from victor.framework.message_execution import execute_message

    orchestrator = _make_orchestrator()
    chat_service = SimpleNamespace(chat=AsyncMock(return_value="READY"))
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))

    result = await execute_message(
        orchestrator=orchestrator,
        execution_context=execution_context,
        user_message="Reply with exactly READY",
    )

    chat_service.chat.assert_awaited_once_with("Reply with exactly READY")
    assert result.success is True
    assert result.content == "READY"
    assert result.metadata["stage"] == "INITIAL"


@pytest.mark.asyncio
async def test_execute_message_can_forward_stream_flag_when_requested() -> None:
    from victor.framework.message_execution import execute_message

    orchestrator = _make_orchestrator()
    response = CompletionResponse(
        content="Streamed response",
        role="assistant",
        tool_calls=[],
        stop_reason="stop",
        usage=None,
        model="test-model",
    )
    chat_service = SimpleNamespace(chat=AsyncMock(return_value=response))
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))

    await execute_message(
        orchestrator=orchestrator,
        execution_context=execution_context,
        user_message="hello",
        stream=True,
        forward_stream_option=True,
    )

    chat_service.chat.assert_awaited_once_with("hello", stream=True)


@pytest.mark.asyncio
async def test_iter_runtime_stream_events_uses_stream_with_events_for_chat_runtime() -> None:
    from victor.framework.message_execution import iter_runtime_stream_events

    runtime = SimpleNamespace(stream_chat=MagicMock())

    async def fake_stream_with_events(runtime_obj, prompt, **kwargs):
        assert runtime_obj is runtime
        assert prompt == "ping"
        assert kwargs["response_prompt"] == "ping"
        yield AgentExecutionEvent(type=EventType.CONTENT, content="pong")

    with patch("victor.framework._internal.stream_with_events", fake_stream_with_events):
        events = [event async for event in iter_runtime_stream_events(runtime, "ping")]

    assert len(events) == 1
    assert events[0].content == "pong"


@pytest.mark.asyncio
async def test_agent_run_delegates_to_shared_message_executor() -> None:
    orchestrator = _make_orchestrator()
    agent = Agent(orchestrator)
    expected = TaskResult(content="READY", success=True)

    with patch(
        "victor.framework.agent.execute_message",
        new=AsyncMock(return_value=expected),
    ) as mock_execute:
        result = await agent.run("Reply with exactly READY", context={"file": "auth.py"})

    assert result is expected
    mock_execute.assert_awaited_once_with(
        orchestrator=orchestrator,
        execution_context=None,
        user_message="Reply with exactly READY",
        context={"file": "auth.py"},
        compatibility_warning_origin="Agent.run()",
    )


@pytest.mark.asyncio
async def test_victor_client_chat_delegates_to_shared_message_executor() -> None:
    config = SessionConfig.from_cli_flags(tool_budget=4, enable_smart_routing=True)
    client = VictorClient(config, container=object())
    orchestrator = _make_orchestrator()
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=MagicMock()))

    class _FakeAgent:
        def __init__(self):
            self.execution_context = execution_context

        def get_orchestrator(self):
            return orchestrator

    client._agent = _FakeAgent()
    client._context = execution_context

    expected = TaskResult(
        content="Service response",
        tool_calls=[{"name": "read"}],
        success=True,
        metadata={"stage": "INITIAL"},
    )

    with patch(
        "victor.framework.client.execute_message",
        new=AsyncMock(return_value=expected),
    ) as mock_execute:
        result = await client.chat("ping", stream=True)

    assert isinstance(result, TaskResult)
    assert result.content == "Service response"
    assert result.tool_count == 1
    assert result.metadata["tool_budget"] == 4
    assert result.metadata["smart_routing"] is True
    mock_execute.assert_awaited_once_with(
        orchestrator=orchestrator,
        execution_context=execution_context,
        user_message="ping",
        stream=True,
        forward_stream_option=True,
    )
