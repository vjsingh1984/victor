import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.stream_handler import StreamChunk
from victor.config.settings import Settings, ProfileConfig


@dataclass
class FakeStreamChunk:
    content: str = ""
    tool_calls: Any = None
    prompt_cache: Dict[str, int] | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class FakeProvider:
    def __init__(self, *, stream_chunks: List[FakeStreamChunk], supports_tools: bool = True):
        self._stream_chunks = stream_chunks
        self._supports_tools = supports_tools
        self.called_chat = False
        self.stream_called = False
        self.chat_response_content = "final"
        self.raise_on_stream = None
        self.force_overflow = False

    def supports_tools(self) -> bool:
        return self._supports_tools

    def supports_streaming(self) -> bool:
        return True

    async def chat(self, **kwargs: Any):
        self.called_chat = True

        class Resp:
            content = self.chat_response_content
            tool_calls = None
            usage = None
            model = "fake-model"

        return Resp()

    async def stream(self, **kwargs: Any) -> AsyncIterator[FakeStreamChunk]:
        self.stream_called = True
        if self.raise_on_stream:
            raise self.raise_on_stream
        if self.force_overflow:
            # Simulate many messages to trigger overflow quickly
            for _ in range(3):
                yield FakeStreamChunk(content="x" * 5000)
            return
        for chunk in self._stream_chunks:
            yield chunk

    async def close(self) -> None:
        return None


def _make_settings() -> Settings:
    s = Settings()
    s.analytics_enabled = False
    return s


def _make_orchestrator(provider: FakeProvider) -> AgentOrchestrator:
    settings = _make_settings()
    settings.load_profiles = lambda: {
        "default": ProfileConfig(
            provider="fake",
            model="fake",
            temperature=0.7,
            max_tokens=2048,
        )
    }
    # Minimal required settings for orchestrator
    settings.use_semantic_tool_selection = False
    settings.use_mcp_tools = False
    settings.analytics_log_file = None
    settings.analytics_enabled = False
    settings.tool_call_budget = 300
    settings.airgapped_mode = False
    settings.load_tool_config = lambda: {}

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model="fake",
    )
    return orchestrator


@pytest.mark.asyncio
async def test_streaming_basic_passes_through_chunks():
    provider = FakeProvider(stream_chunks=[FakeStreamChunk(content="hello"), FakeStreamChunk(content=" world")])
    orchestrator = _make_orchestrator(provider)

    out = []
    async for chunk in orchestrator.stream_chat("hi"):
        out.append(chunk.content)

    assert "".join(out) != ""  # streamed something
    assert provider.stream_called is True


@pytest.mark.asyncio
async def test_sticky_budget_not_overridden():
    provider = FakeProvider(stream_chunks=[FakeStreamChunk(content="done")])
    orchestrator = _make_orchestrator(provider)

    # Set sticky budget and iteration overrides
    orchestrator.unified_tracker.set_tool_budget(5, user_override=True)
    orchestrator.unified_tracker.set_max_iterations(10, user_override=True)

    before_budget = orchestrator.unified_tracker.progress.tool_budget
    before_iter = orchestrator.unified_tracker.config.max_total_iterations

    out = []
    async for chunk in orchestrator.stream_chat("hi"):
        out.append(chunk.content)

    after_budget = orchestrator.unified_tracker.progress.tool_budget
    after_iter = orchestrator.unified_tracker.config.max_total_iterations

    assert after_budget == before_budget
    assert after_iter == before_iter


@pytest.mark.asyncio
async def test_non_streaming_path_uses_chat_when_streaming_not_supported():
    provider = FakeProvider(stream_chunks=[], supports_tools=False)
    provider.supports_streaming = lambda: False
    orchestrator = _make_orchestrator(provider)

    out = []
    async for chunk in orchestrator.stream_chat("hi"):
        out.append(chunk.content)

    assert provider.called_chat is True


@pytest.mark.asyncio
async def test_garbage_content_sets_force_completion():
    # Chunk with garbage patterns should trigger garbage detection and stop accumulation
    garbage = FakeStreamChunk(content="<<<<garbage>>>>")
    provider = FakeProvider(stream_chunks=[garbage, FakeStreamChunk(content="ok")])
    orchestrator = _make_orchestrator(provider)

    out = []
    async for chunk in orchestrator.stream_chat("hi"):
        out.append(chunk.content)

    # We still streamed something, but the garbage should not crash
    assert provider.stream_called is True


@pytest.mark.asyncio
async def test_iteration_limit_forces_completion():
    provider = FakeProvider(stream_chunks=[FakeStreamChunk(content="data")])
    orchestrator = _make_orchestrator(provider)
    # Force very small iteration budget\n    orchestrator.unified_tracker.set_max_iterations(1, user_override=True)\n\n    out = []\n    async for chunk in orchestrator.stream_chat(\"hi\"):\n        out.append(chunk.content)\n\n    assert provider.stream_called is True\n+\n+\n+@pytest.mark.asyncio\n+async def test_context_overflow_triggers_completion():\n+    provider = FakeProvider(stream_chunks=[], supports_tools=False)\n+    provider.force_overflow = True\n+    orchestrator = _make_orchestrator(provider)\n+    orchestrator.max_tokens = 256\n+\n+    out = []\n+    async for chunk in orchestrator.stream_chat(\"hi\"):\n+        out.append(chunk.content)\n+\n+    assert provider.stream_called is True\n*** End Patch
