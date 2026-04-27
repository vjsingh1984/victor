import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.chat_stream_runtime import ServiceStreamingRuntime
from victor.providers.base import StreamChunk


def _make_orchestrator_stub():
    orch = MagicMock()
    orch.has_capability.return_value = False
    orch.get_capability_value.return_value = None
    orch._cumulative_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    orch._conversation_controller = MagicMock()
    orch.messages = []
    return orch


def test_service_streaming_runtime_caches_pipeline(monkeypatch):
    orch = _make_orchestrator_stub()
    runtime = ServiceStreamingRuntime(orch)
    created = []

    class DummyPipeline:
        pass

    def fake_factory(owner, **kwargs):
        created.append((owner, kwargs))
        return DummyPipeline()

    streaming_module = importlib.import_module("victor.agent.streaming")

    monkeypatch.setattr(streaming_module, "create_streaming_chat_pipeline", fake_factory)

    first = runtime.get_pipeline()
    second = runtime.get_pipeline()

    assert first is second
    assert len(created) == 1
    owner, kwargs = created[0]
    assert owner is runtime
    assert kwargs["perception"] is None
    assert kwargs["fulfillment"] is None
    assert kwargs["runtime_intelligence"] is orch._runtime_intelligence


@pytest.mark.asyncio
async def test_service_streaming_runtime_stream_chat_uses_pipeline(monkeypatch):
    orch = _make_orchestrator_stub()
    runtime = ServiceStreamingRuntime(orch)
    chunk = StreamChunk(content="service", is_final=True)

    class DummyPipeline:
        def __init__(self):
            self.calls = []

        async def run(self, user_message: str, **kwargs):
            self.calls.append((user_message, kwargs))
            yield chunk

    pipeline = DummyPipeline()

    def fake_factory(owner, **kwargs):
        return pipeline

    streaming_module = importlib.import_module("victor.agent.streaming")

    monkeypatch.setattr(streaming_module, "create_streaming_chat_pipeline", fake_factory)

    chunks = [item async for item in runtime.stream_chat("hello", mode="test")]

    assert chunks == [chunk]
    assert pipeline.calls == [("hello", {"mode": "test"})]


@pytest.mark.asyncio
async def test_service_streaming_runtime_create_stream_context_uses_blocked_threshold_setting():
    orch = _make_orchestrator_stub()
    orch.settings = SimpleNamespace(recovery_blocked_consecutive_threshold=7)
    orch._classify_task_keywords.return_value = {}
    orch._tool_planner = SimpleNamespace(infer_goals_from_message=lambda _: [])
    orch.tool_budget = 200
    orch.tool_calls_used = 0
    orch._task_completion_detector = None

    runtime = ServiceStreamingRuntime(orch)
    runtime._prepare_stream = AsyncMock(
        return_value=(
            SimpleNamespace(),
            0.0,
            0.0,
            {},
            30,
            10,
            0,
            False,
            SimpleNamespace(value="default"),
            None,
            None,
        )
    )

    ctx = await runtime._create_stream_context("hello")

    assert ctx.max_blocked_before_force == 7
