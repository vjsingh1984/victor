import importlib
from unittest.mock import MagicMock

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
    assert created == [(runtime, {"perception": None, "fulfillment": None})]


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
