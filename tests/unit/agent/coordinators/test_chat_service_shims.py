from __future__ import annotations

import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.protocols.chat_runtime import ExecutionMode
from victor.agent.coordinators.streaming_chat_coordinator import (
    StreamingChatCoordinator,
)
from victor.agent.coordinators.chat_coordinator import ChatCoordinator
from victor.agent.coordinators.sync_chat_coordinator import SyncChatCoordinator
from victor.agent.coordinators.unified_chat_coordinator import UnifiedChatCoordinator
from victor.providers.base import CompletionResponse, StreamChunk


def _make_deprecated_chat_coordinator(orchestrator) -> ChatCoordinator:
    with pytest.warns(DeprecationWarning, match="ChatCoordinator is deprecated"):
        return ChatCoordinator(orchestrator=orchestrator)


def test_specialized_chat_coordinator_modules_reexport_service_shims():
    """Legacy specialized coordinator modules should re-export service-hosted shims."""
    from victor.agent.services.sync_chat_compat import SyncChatCoordinator as service_sync
    from victor.agent.services.streaming_chat_compat import (
        StreamingChatCoordinator as service_streaming,
    )
    from victor.agent.services.unified_chat_compat import UnifiedChatCoordinator as service_unified

    assert SyncChatCoordinator is service_sync
    assert StreamingChatCoordinator is service_streaming
    assert UnifiedChatCoordinator is service_unified


def test_chat_coordinator_module_reexports_service_shim():
    """Legacy chat coordinator module should re-export the service-hosted shim."""
    from victor.agent.services.chat_compat import ChatCoordinator as service_chat

    assert ChatCoordinator is service_chat


def test_legacy_execution_mode_reexports_service_runtime_enum():
    """Legacy coordinator protocol path should re-export service-hosted ExecutionMode."""
    from victor.agent.coordinators.protocols import ExecutionMode as legacy_execution_mode
    from victor.agent.services.protocols.chat_runtime import ExecutionMode as service_execution_mode

    assert legacy_execution_mode is service_execution_mode


@pytest.mark.asyncio
async def test_sync_chat_coordinator_delegates_to_bound_chat_service():
    chat_service = AsyncMock()
    response = CompletionResponse(content="service", role="assistant")
    chat_service.chat.return_value = response

    coordinator = SyncChatCoordinator(
        chat_context=MagicMock(),
        tool_context=MagicMock(),
        provider_context=MagicMock(),
        turn_executor=MagicMock(),
        chat_service=chat_service,
    )

    with pytest.warns(
        DeprecationWarning,
        match="SyncChatCoordinator.chat\\(\\) is deprecated compatibility surface",
    ):
        result = await coordinator.chat("hello", use_planning=True)

    assert result is response
    chat_service.chat.assert_awaited_once_with("hello", use_planning=True)


@pytest.mark.asyncio
async def test_sync_chat_coordinator_planning_prefers_bound_chat_service():
    chat_service = AsyncMock()
    response = CompletionResponse(content="planned-service", role="assistant")
    chat_service.chat.return_value = response

    coordinator = SyncChatCoordinator(
        chat_context=MagicMock(),
        tool_context=MagicMock(),
        provider_context=MagicMock(),
        turn_executor=MagicMock(),
        chat_service=chat_service,
    )

    result = await coordinator._chat_with_planning("hello")

    assert result is response
    chat_service.chat.assert_awaited_once_with("hello", use_planning=True)


@pytest.mark.asyncio
async def test_sync_chat_coordinator_planning_requires_canonical_runtime():
    with pytest.warns(
        DeprecationWarning,
        match="SyncChatCoordinator without a bound ChatService is deprecated",
    ):
        coordinator = SyncChatCoordinator(
            chat_context=MagicMock(),
            tool_context=MagicMock(),
            provider_context=MagicMock(),
            turn_executor=MagicMock(),
        )

    with pytest.raises(RuntimeError, match="planning requires a bound ChatService or orchestrator"):
        await coordinator._chat_with_planning("hello")


def test_sync_chat_coordinator_constructor_warns_without_chat_service():
    with pytest.warns(
        DeprecationWarning,
        match="SyncChatCoordinator without a bound ChatService is deprecated",
    ):
        SyncChatCoordinator(
            chat_context=MagicMock(),
            tool_context=MagicMock(),
            provider_context=MagicMock(),
            turn_executor=MagicMock(),
        )


@pytest.mark.asyncio
async def test_streaming_chat_coordinator_delegates_to_bound_chat_service():
    chunk = StreamChunk(content="partial", role="assistant")
    chat_service = MagicMock()

    async def _stream_chat(user_message: str):
        assert user_message == "hello"
        yield chunk

    chat_service.stream_chat = _stream_chat

    coordinator = StreamingChatCoordinator(
        chat_context=MagicMock(),
        tool_context=MagicMock(),
        provider_context=MagicMock(),
        chat_service=chat_service,
    )

    with pytest.warns(
        DeprecationWarning,
        match="StreamingChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
    ):
        chunks = [item async for item in coordinator.stream_chat("hello")]

    assert chunks == [chunk]


def test_streaming_chat_coordinator_constructor_warns_without_chat_service():
    with pytest.warns(
        DeprecationWarning,
        match="StreamingChatCoordinator without a bound ChatService is deprecated",
    ):
        StreamingChatCoordinator(
            chat_context=MagicMock(),
            tool_context=MagicMock(),
            provider_context=MagicMock(),
        )


@pytest.mark.asyncio
async def test_streaming_chat_coordinator_tool_calls_use_canonical_tool_context_surface():
    chat_context = MagicMock()
    tool_context = MagicMock()
    tool_context.execute_tool_calls = AsyncMock(
        return_value=[
            {
                "name": "read",
                "content": "file contents",
                "tool_call_id": "call_123",
            }
        ]
    )
    tool_context._handle_tool_calls = AsyncMock(
        side_effect=AssertionError("legacy _handle_tool_calls bridge should not be used")
    )
    provider_context = MagicMock()

    with pytest.warns(
        DeprecationWarning,
        match="StreamingChatCoordinator without a bound ChatService is deprecated",
    ):
        coordinator = StreamingChatCoordinator(
            chat_context=chat_context,
            tool_context=tool_context,
            provider_context=provider_context,
        )

    await coordinator._execute_tool_calls_during_stream([{"name": "read", "arguments": {}}])

    tool_context.execute_tool_calls.assert_awaited_once_with([{"name": "read", "arguments": {}}])
    chat_context.add_message.assert_called_once_with(
        "tool",
        "file contents",
        name="read",
        tool_call_id="call_123",
    )


@pytest.mark.asyncio
async def test_streaming_chat_coordinator_tool_calls_require_canonical_tool_context_surface():
    from types import SimpleNamespace

    chat_context = MagicMock()
    provider_context = MagicMock()
    legacy_handle_tool_calls = AsyncMock(return_value=[{"name": "read", "content": "ignored"}])
    tool_context = SimpleNamespace(
        tool_calls_used=0,
        tool_budget=10,
        tool_selector=MagicMock(),
        use_semantic_selection=False,
        observed_files=set(),
        _handle_tool_calls=legacy_handle_tool_calls,
    )

    with pytest.warns(
        DeprecationWarning,
        match="StreamingChatCoordinator without a bound ChatService is deprecated",
    ):
        coordinator = StreamingChatCoordinator(
            chat_context=chat_context,
            tool_context=tool_context,
            provider_context=provider_context,
        )

    with pytest.raises(AttributeError, match="execute_tool_calls"):
        await coordinator._execute_tool_calls_during_stream([{"name": "read", "arguments": {}}])

    legacy_handle_tool_calls.assert_not_awaited()


@pytest.mark.asyncio
async def test_unified_chat_coordinator_delegates_to_bound_chat_service():
    sync = MagicMock()
    streaming = MagicMock()
    chat_service = AsyncMock()
    response = CompletionResponse(content="unified", role="assistant")
    chat_service.chat.return_value = response

    coordinator = UnifiedChatCoordinator(
        sync_coordinator=sync,
        streaming_coordinator=streaming,
        default_mode=ExecutionMode.SYNC,
        chat_service=chat_service,
    )

    with pytest.warns(
        DeprecationWarning,
        match="UnifiedChatCoordinator.chat\\(\\) is deprecated compatibility surface",
    ):
        result = await coordinator.chat("hello", mode=ExecutionMode.STREAMING, use_planning=False)

    assert result is response
    chat_service.chat.assert_awaited_once_with("hello", stream=True, use_planning=False)


def test_unified_chat_coordinator_constructor_warns_without_chat_service():
    with pytest.warns(
        DeprecationWarning,
        match="UnifiedChatCoordinator without a bound ChatService is deprecated",
    ):
        UnifiedChatCoordinator(
            sync_coordinator=MagicMock(),
            streaming_coordinator=MagicMock(),
        )


@pytest.mark.asyncio
async def test_unified_chat_coordinator_requires_bound_chat_service_for_chat():
    with pytest.warns(
        DeprecationWarning,
        match="UnifiedChatCoordinator without a bound ChatService is deprecated",
    ):
        coordinator = UnifiedChatCoordinator(
            sync_coordinator=MagicMock(),
            streaming_coordinator=MagicMock(),
        )

    with pytest.warns(
        DeprecationWarning,
        match="UnifiedChatCoordinator.chat\\(\\) is deprecated compatibility surface",
    ):
        with pytest.raises(RuntimeError, match="requires a bound ChatService"):
            await coordinator.chat("hello", mode=ExecutionMode.SYNC, use_planning=False)


@pytest.mark.asyncio
async def test_unified_chat_coordinator_requires_bound_chat_service_for_streaming():
    with pytest.warns(
        DeprecationWarning,
        match="UnifiedChatCoordinator without a bound ChatService is deprecated",
    ):
        coordinator = UnifiedChatCoordinator(
            sync_coordinator=MagicMock(),
            streaming_coordinator=MagicMock(),
        )

    with pytest.warns(
        DeprecationWarning,
        match="UnifiedChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
    ):
        with pytest.raises(RuntimeError, match="requires a bound ChatService"):
            await anext(coordinator.stream_chat("hello"))


@pytest.mark.asyncio
async def test_chat_coordinator_planning_prefers_orchestrator_runtime_helper():
    response = CompletionResponse(content="planned", role="assistant")
    orchestrator = MagicMock()
    orchestrator._run_planning_chat_runtime = AsyncMock(return_value=response)

    coordinator = _make_deprecated_chat_coordinator(orchestrator)

    result = await coordinator._chat_with_planning("hello")

    assert result is response
    orchestrator._run_planning_chat_runtime.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_chat_coordinator_planning_prefers_bound_chat_service():
    response = CompletionResponse(content="planned-service", role="assistant")
    orchestrator = MagicMock()
    chat_service = MagicMock()
    chat_service.chat = AsyncMock(return_value=response)

    coordinator = _make_deprecated_chat_coordinator(orchestrator)
    coordinator.bind_chat_service(chat_service)

    result = await coordinator._chat_with_planning("hello")

    assert result is response
    chat_service.chat.assert_awaited_once_with("hello", use_planning=True)


@pytest.mark.asyncio
async def test_chat_coordinator_planning_requires_canonical_runtime():
    coordinator = _make_deprecated_chat_coordinator(MagicMock())

    with pytest.raises(RuntimeError, match="planning requires a bound ChatService or orchestrator runtime"):
        await coordinator._chat_with_planning("hello")


@pytest.mark.asyncio
async def test_chat_coordinator_context_limits_prefer_orchestrator_runtime_helper():
    chunk = StreamChunk(content="done", is_final=True)
    orchestrator = MagicMock()
    orchestrator._handle_context_and_iteration_limits_runtime = AsyncMock(
        return_value=(True, chunk)
    )

    coordinator = _make_deprecated_chat_coordinator(orchestrator)

    result = await coordinator._handle_context_and_iteration_limits("hello", 4, 1000, 5, 0.8)

    assert result == (True, chunk)
    orchestrator._handle_context_and_iteration_limits_runtime.assert_awaited_once_with(
        "hello",
        4,
        1000,
        5,
        0.8,
    )


@pytest.mark.asyncio
async def test_chat_coordinator_streaming_prefers_bound_chat_service():
    chunk = StreamChunk(content="service-stream", is_final=True)
    orchestrator = MagicMock()
    chat_service = MagicMock()

    async def _service_stream_chat(user_message: str, **kwargs):
        assert user_message == "hello"
        assert kwargs == {"mode": "test"}
        yield chunk

    coordinator = _make_deprecated_chat_coordinator(orchestrator)
    chat_service.stream_chat = _service_stream_chat
    coordinator.bind_chat_service(chat_service)

    with pytest.warns(
        DeprecationWarning,
        match="ChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
    ):
        chunks = [item async for item in coordinator.stream_chat("hello", mode="test")]

    assert chunks == [chunk]


@pytest.mark.asyncio
async def test_chat_coordinator_streaming_prefers_orchestrator_public_runtime():
    chunk = StreamChunk(content="orchestrator-runtime", is_final=True)
    orchestrator = MagicMock()

    async def _runtime_stream_chat(user_message: str, **kwargs):
        assert user_message == "hello"
        assert kwargs == {"mode": "test"}
        yield chunk

    orchestrator.stream_chat = _runtime_stream_chat

    coordinator = _make_deprecated_chat_coordinator(orchestrator)

    with pytest.warns(
        DeprecationWarning,
        match="ChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
    ):
        chunks = [item async for item in coordinator.stream_chat("hello", mode="test")]

    assert chunks == [chunk]


@pytest.mark.asyncio
async def test_chat_coordinator_streaming_public_runtime_beats_internal_hooks():
    chunk = StreamChunk(content="orchestrator-runtime", is_final=True)
    orchestrator = MagicMock()

    async def _runtime_stream_chat(user_message: str, **kwargs):
        assert user_message == "hello"
        assert kwargs == {"mode": "test"}
        yield chunk

    async def _legacy_stream_chat(user_message: str, **kwargs):
        raise AssertionError("legacy compatibility hook should not run")

    orchestrator.stream_chat = _runtime_stream_chat
    orchestrator._get_service_streaming_runtime = MagicMock(
        side_effect=AssertionError("internal streaming runtime getter should not run")
    )
    orchestrator._stream_chat_runtime = _legacy_stream_chat

    coordinator = _make_deprecated_chat_coordinator(orchestrator)

    with pytest.warns(
        DeprecationWarning,
        match="ChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
    ):
        chunks = [item async for item in coordinator.stream_chat("hello", mode="test")]

    assert chunks == [chunk]
    orchestrator._get_service_streaming_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_chat_coordinator_chat_with_planning_warns_and_delegates():
    response = CompletionResponse(content="planned", role="assistant")
    orchestrator = MagicMock()
    chat_service = MagicMock()
    chat_service.chat = AsyncMock(return_value=response)

    coordinator = _make_deprecated_chat_coordinator(orchestrator)
    coordinator.bind_chat_service(chat_service)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await coordinator.chat_with_planning("hello", use_planning=True)

    assert result is response
    assert any(
        "ChatCoordinator.chat_with_planning() is deprecated compatibility surface"
        in str(item.message)
        for item in caught
    )
    chat_service.chat.assert_awaited_once_with("hello", use_planning=True)


def test_chat_coordinator_turn_executor_prefers_bound_chat_service():
    orchestrator = MagicMock()
    coordinator = _make_deprecated_chat_coordinator(orchestrator)
    service_executor = MagicMock(name="service_executor")

    class ChatServiceStub:
        @property
        def turn_executor(self):
            return service_executor

    chat_service = ChatServiceStub()
    coordinator.bind_chat_service(chat_service)

    with pytest.warns(DeprecationWarning, match="ChatCoordinator.turn_executor is deprecated"):
        result = coordinator.turn_executor

    assert result is service_executor
    assert coordinator._turn_executor is None


@pytest.mark.asyncio
async def test_chat_coordinator_uses_bound_chat_service_getter():
    orchestrator = MagicMock()
    response = CompletionResponse(content="getter", role="assistant")
    chat_service = MagicMock()
    chat_service.chat = AsyncMock(return_value=response)

    coordinator = _make_deprecated_chat_coordinator(orchestrator)
    coordinator.bind_chat_service_getter(lambda: chat_service)

    with pytest.warns(
        DeprecationWarning,
        match="ChatCoordinator.chat\\(\\) is deprecated compatibility surface",
    ):
        result = await coordinator.chat("hello", use_planning=False)

    assert result is response
    chat_service.chat.assert_awaited_once_with("hello", use_planning=False)


def test_chat_coordinator_turn_executor_requires_bound_chat_service_runtime():
    coordinator = _make_deprecated_chat_coordinator(MagicMock())

    with pytest.warns(DeprecationWarning, match="ChatCoordinator.turn_executor is deprecated"):
        with pytest.raises(RuntimeError, match="requires a bound ChatService runtime"):
            _ = coordinator.turn_executor
