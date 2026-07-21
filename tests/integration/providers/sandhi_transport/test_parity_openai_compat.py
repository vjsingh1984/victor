"""Golden parity battery: native httpx transport vs the real sandhi binding (T5).

Both providers hit the SAME localhost fixture server with identical inputs; the pilot's
contract is byte-level request parity and semantic response parity. Fixture bodies mirror
the sandhi recorded corpus (commit 3102dd8) plus tool-call shapes the corpus lacks.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("sandhi_gateway", reason="requires the victor[sandhi] extra")

from victor.providers.base import (
    Message,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

pytestmark = pytest.mark.integration

COMPLETE_BODY = {
    "id": "chatcmpl-corpus",
    "object": "chat.completion",
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello, world"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 250,
        "total_tokens": 1250,
        "prompt_tokens_details": {"cached_tokens": 800},
    },
}

TOOL_CALL_BODY = {
    "id": "chatcmpl-tools",
    "object": "chat.completion",
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
}

STREAM_SSE = (
    'data: {"id":"c","object":"chat.completion.chunk","model":"deepseek-chat",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}],"usage":null}\n\n'
    'data: {"id":"c","object":"chat.completion.chunk","model":"deepseek-chat",'
    '"choices":[{"index":0,"delta":{"content":", world"},"finish_reason":null}],"usage":null}\n\n'
    'data: {"id":"c","object":"chat.completion.chunk","model":"deepseek-chat",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":1000,"completion_tokens":250,"total_tokens":1250,'
    '"prompt_tokens_details":{"cached_tokens":800}}}\n\n'
    "data: [DONE]\n\n"
).encode()

MESSAGES = [Message(role="user", content="hi there")]


async def run_chat(provider):
    return await provider.chat(MESSAGES, model="deepseek-chat", temperature=0.2, max_tokens=64)


async def run_stream(provider):
    return [
        chunk
        async for chunk in provider.stream(
            MESSAGES, model="deepseek-chat", temperature=0.2, max_tokens=64
        )
    ]


class TestCompletionParity:
    async def test_completion_response_and_request_parity(self, fixture_server, make_pair):
        native_srv = fixture_server(body=json.dumps(COMPLETE_BODY).encode())
        sandhi_srv = fixture_server(body=json.dumps(COMPLETE_BODY).encode())
        native, _ = make_pair(native_srv.url)
        _, sandhi = make_pair(sandhi_srv.url)

        native_resp = await run_chat(native)
        sandhi_resp = await run_chat(sandhi)

        # (a) semantic response parity — full model comparison
        assert native_resp.model_dump() == sandhi_resp.model_dump()
        assert sandhi_resp.content == "Hello, world"
        assert sandhi_resp.usage["prompt_tokens"] == 1000
        assert sandhi_resp.usage["cache_read_input_tokens"] == 800

        # (c) request-side parity: identical JSON body and auth header
        assert len(native_srv.requests) == 1 and len(sandhi_srv.requests) == 1
        native_req, sandhi_req = native_srv.requests[0], sandhi_srv.requests[0]
        assert json.loads(native_req.body) == json.loads(sandhi_req.body)
        assert native_req.headers.get("authorization") == sandhi_req.headers.get("authorization")
        assert native_req.path == sandhi_req.path == "/v1/chat/completions"

    async def test_tool_call_parity(self, fixture_server, make_pair):
        native_srv = fixture_server(body=json.dumps(TOOL_CALL_BODY).encode())
        sandhi_srv = fixture_server(body=json.dumps(TOOL_CALL_BODY).encode())
        native, _ = make_pair(native_srv.url)
        _, sandhi = make_pair(sandhi_srv.url)

        native_resp = await run_chat(native)
        sandhi_resp = await run_chat(sandhi)
        assert native_resp.model_dump() == sandhi_resp.model_dump()
        assert sandhi_resp.tool_calls == [
            {"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}
        ]

    async def test_no_double_request(self, fixture_server, make_pair):
        # (d) the layering contract: sandhi is called with max_retries=0, victor's
        # resilience is not engaged at this layer — exactly ONE upstream POST per chat.
        srv = fixture_server(body=json.dumps(COMPLETE_BODY).encode())
        _, sandhi = make_pair(srv.url)
        await run_chat(sandhi)
        assert len(srv.requests) == 1


class TestStreamParity:
    async def test_stream_chunk_sequence_parity(self, fixture_server, make_pair):
        native_srv = fixture_server(body=STREAM_SSE, content_type="text/event-stream")
        sandhi_srv = fixture_server(body=STREAM_SSE, content_type="text/event-stream")
        native, _ = make_pair(native_srv.url)
        _, sandhi = make_pair(sandhi_srv.url)

        native_chunks = await run_stream(native)
        sandhi_chunks = await run_stream(sandhi)

        assert [c.model_dump() for c in native_chunks] == [c.model_dump() for c in sandhi_chunks]
        assert "".join(c.content for c in sandhi_chunks) == "Hello, world"
        final = sandhi_chunks[-1]
        assert final.is_final
        # (b) usage flows through victor's own parsing (sandhi terminal usage ignored)
        usage_chunks = [c for c in sandhi_chunks if c.usage]
        assert usage_chunks and usage_chunks[-1].usage["cache_read_input_tokens"] == 800

        # Request-side parity for the streaming call: sandhi's adapter injects
        # stream_options.include_usage=true (its at-the-source usage guarantee — documented
        # sandhi behavior, outside the cacheable prompt prefix). Everything else, including
        # the full prompt assembly, must be identical.
        native_body = json.loads(native_srv.requests[0].body)
        sandhi_body = json.loads(sandhi_srv.requests[0].body)
        assert sandhi_body.pop("stream_options", None) == {"include_usage": True}
        assert native_body == sandhi_body


class TestErrorParity:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (401, ProviderAuthError),
            (429, ProviderRateLimitError),
            (500, ProviderError),
        ],
    )
    async def test_error_class_parity(self, fixture_server, make_pair, status, expected):
        body = json.dumps({"error": {"message": "boom"}}).encode()
        native_srv = fixture_server(status=status, body=body)
        sandhi_srv = fixture_server(status=status, body=body)
        native, _ = make_pair(native_srv.url)
        _, sandhi = make_pair(sandhi_srv.url)

        with pytest.raises(expected):
            await run_chat(native)
        with pytest.raises(expected):
            await run_chat(sandhi)
        assert sandhi._sandhi_demoted is False

    async def test_timeout_surfaces_within_bound(self, fixture_server, make_pair):
        srv = fixture_server(body=json.dumps(COMPLETE_BODY).encode(), delay_secs=8.0)
        _, sandhi = make_pair(srv.url, timeout=1)
        import time

        start = time.monotonic()
        with pytest.raises(ProviderTimeoutError):
            await run_chat(sandhi)
        assert time.monotonic() - start < 7.0, "timeout must fire well before the 8s delay"
        assert sandhi._sandhi_demoted is False
