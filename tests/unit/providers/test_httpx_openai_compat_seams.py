"""Wire-seam tests for HttpxOpenAICompatProvider (FEP-0020 Phase 4b, T1).

The pilot swaps ONLY the wire: `_complete_raw` (POST → parsed body dict) and
`_open_stream_lines` (POST → (closer, async-iterator-of-SSE-lines)). These tests pin that
`chat()`/`stream()` consume those seams exclusively, so a transport override cannot change
any parsing/translation behavior.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

import httpx
import pytest
import respx

from victor.providers.base import Message
from victor.providers.deepseek_provider import DeepSeekProvider


def make_provider() -> DeepSeekProvider:
    return DeepSeekProvider(api_key="k", base_url="https://api.deepseek.com/v1")


OK_BODY = {
    "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class TestCompleteRawSeam:
    @respx.mock
    async def test_complete_raw_returns_parsed_json(self):
        respx.post("https://api.deepseek.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OK_BODY)
        )
        provider = make_provider()
        result = await provider._complete_raw({"model": "deepseek-chat", "messages": []})
        assert result == OK_BODY

    async def test_chat_routes_through_complete_raw(self, monkeypatch):
        provider = make_provider()
        seen: List[Dict[str, Any]] = []

        async def fake_complete_raw(payload: Dict[str, Any]) -> Dict[str, Any]:
            seen.append(payload)
            return OK_BODY

        monkeypatch.setattr(provider, "_complete_raw", fake_complete_raw)
        response = await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")
        assert response.content == "hello"
        assert response.usage is not None and response.usage["prompt_tokens"] == 10
        assert len(seen) == 1 and seen[0]["model"] == "deepseek-chat"


SSE_LINES = [
    'data: {"choices":[{"delta":{"content":"he"}}]}',
    'data: {"choices":[{"delta":{"content":"llo"}}]}',
    "data: [DONE]",
]


class TestOpenStreamLinesSeam:
    async def test_stream_routes_through_open_stream_lines(self, monkeypatch):
        provider = make_provider()
        closed = {"n": 0}

        async def lines() -> AsyncIterator[str]:
            for line in SSE_LINES:
                yield line

        async def closer() -> None:
            closed["n"] += 1

        async def fake_open(payload: Dict[str, Any]):
            assert payload["stream"] is True
            return closer, lines()

        monkeypatch.setattr(provider, "_open_stream_lines", fake_open)
        chunks = [
            c
            async for c in provider.stream(
                [Message(role="user", content="hi")], model="deepseek-chat"
            )
        ]
        assert "".join(c.content for c in chunks) == "hello"
        assert chunks[-1].is_final
        assert closed["n"] == 1, "stream() must close the seam's closer exactly once"

    @respx.mock
    async def test_open_stream_lines_yields_upstream_lines(self):
        sse_body = "\n".join(SSE_LINES) + "\n"
        respx.post("https://api.deepseek.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=sse_body, headers={"content-type": "text/event-stream"}
            )
        )
        provider = make_provider()
        closer, lines = await provider._open_stream_lines(
            {"model": "deepseek-chat", "messages": [], "stream": True}
        )
        try:
            collected = [line async for line in lines if line.strip()]
        finally:
            await closer()
        assert collected == SSE_LINES
