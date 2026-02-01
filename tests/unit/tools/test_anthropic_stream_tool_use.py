from types import SimpleNamespace
from typing import Any
from collections.abc import AsyncIterator

import pytest

from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.base import Message, StreamChunk, ToolDefinition


class FakeStream:
    """Async iterator context manager for anthropic stream events."""

    def __init__(self, events: list[Any]):
        self._events = list(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


@pytest.mark.asyncio
async def test_anthropic_stream_emits_tool_calls(monkeypatch):
    provider = AnthropicProvider(api_key="dummy-key")

    events = [
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="hello"),
        ),
        SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(
                type="tool_use", id="tool_1", name="dummy_tool", input={}, index=0
            ),
            index=0,
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"echo": "hi"}'),
            index=0,
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(type="message_stop"),
    ]

    # Patch the SDK stream to our fake stream
    monkeypatch.setattr(
        provider.client.messages,
        "stream",
        lambda **kwargs: FakeStream(events),
    )

    chunks: list[StreamChunk] = []
    async for chunk in provider.stream(
        messages=[Message(role="user", content="hi")],
        model="claude-3-5-sonnet-20241022",
        tools=[ToolDefinition(name="dummy_tool", description="d", parameters={"type": "object"})],
    ):
        chunks.append(chunk)

    # First chunk should carry text, final chunk should carry tool_calls
    assert any(c.content for c in chunks)
    final = chunks[-1]
    assert final.is_final
    assert final.tool_calls
    assert final.tool_calls[0]["name"] == "dummy_tool"
    assert final.tool_calls[0]["arguments"] == {"echo": "hi"}
