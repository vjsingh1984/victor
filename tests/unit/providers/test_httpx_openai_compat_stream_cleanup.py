# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provider-level guards for the streaming context-close refactor.

``stream()`` now closes the response context in its own ``finally`` (lexically paired with
the ``__aenter__`` in ``_open_chat_completion_stream``), instead of stashing the context on
``response._victor_stream_context`` and reading it back. These tests guard that the attribute
hack is gone and that the shared client is not closed by an individual stream.
"""

from unittest.mock import AsyncMock

import httpx

from victor.providers.base import Message
from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider


class _StreamProvider(HttpxOpenAICompatProvider):
    @property
    def name(self) -> str:
        return "test-stream"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True


class _FakeStreamingResponse:
    def __init__(self, lines):
        self.status_code = 200
        self._lines = lines
        self.request = httpx.Request("POST", "https://example.com/chat/completions")
        self.closed = False

    async def aread(self) -> bytes:
        return b""

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self) -> None:
        self.closed = True


def _provider() -> _StreamProvider:
    return _StreamProvider(api_key="k", base_url="https://example.com", provider_name="test-stream")


async def test_stream_closes_response_without_attribute_hack():
    provider = _provider()
    response = _FakeStreamingResponse(
        [
            'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}',
            "data: [DONE]",
        ]
    )
    provider.client.send = AsyncMock(return_value=response)

    chunks = [
        chunk
        async for chunk in provider.stream(
            messages=[Message(role="user", content="hello")],
            model="test-model",
        )
    ]

    assert any(c.content == "hi" for c in chunks)
    # The response context was closed in stream()'s finally...
    assert response.closed is True
    # ...without the removed _victor_stream_context cross-task readback hack.
    assert not hasattr(response, "_victor_stream_context")
    # The shared/pooled client must NOT be closed by an individual stream.
    assert provider.client.is_closed is False
    await provider.close()
