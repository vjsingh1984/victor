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
        self.chat_response_content = "final"

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

