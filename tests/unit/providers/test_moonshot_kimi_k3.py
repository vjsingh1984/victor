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

"""Tests for Kimi K3 support in the Moonshot provider.

K3 facts under test: model id ``kimi-k3`` on the international
``https://api.moonshot.ai/v1`` endpoint (K2 stays on ``.cn``), 1M context,
top-level ``reasoning_effort`` ∈ {low, high, max} with server default "max"
(there is no "medium").
"""

import pytest
import respx
from httpx import Response

from victor.providers.base import Message
from victor.providers.moonshot_provider import (
    DEFAULT_BASE_URL,
    KIMI_K3_BASE_URL,
    MoonshotProvider,
)


def _provider(**kwargs) -> MoonshotProvider:
    return MoonshotProvider(api_key="test-key", **kwargs)


def _ok_completion() -> dict:
    return {
        "id": "cmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class TestEndpointRouting:
    def test_k3_resolves_to_ai_endpoint(self):
        assert MoonshotProvider.resolve_base_url_for_model("kimi-k3") == KIMI_K3_BASE_URL
        assert MoonshotProvider.resolve_base_url_for_model("kimi-k3-preview") == KIMI_K3_BASE_URL

    def test_k2_resolves_to_cn_endpoint(self):
        assert MoonshotProvider.resolve_base_url_for_model("kimi-k2-thinking") == DEFAULT_BASE_URL

    @respx.mock
    async def test_k3_chat_hits_ai_host(self):
        route = respx.post(f"{KIMI_K3_BASE_URL}/chat/completions").mock(
            return_value=Response(200, json=_ok_completion())
        )
        provider = _provider()
        try:
            await provider.chat([Message(role="user", content="hi")], model="kimi-k3")
        finally:
            await provider.close()
        assert route.called

    @respx.mock
    async def test_k2_chat_stays_on_cn_host(self):
        route = respx.post(f"{DEFAULT_BASE_URL}/chat/completions").mock(
            return_value=Response(200, json=_ok_completion())
        )
        provider = _provider()
        try:
            await provider.chat([Message(role="user", content="hi")], model="kimi-k2-thinking")
        finally:
            await provider.close()
        assert route.called

    @respx.mock
    async def test_explicit_base_url_pins_all_models(self):
        custom = "https://proxy.example.com/v1"
        route = respx.post(f"{custom}/chat/completions").mock(
            return_value=Response(200, json=_ok_completion())
        )
        provider = _provider(base_url=custom)
        try:
            await provider.chat([Message(role="user", content="hi")], model="kimi-k3")
        finally:
            await provider.close()
        assert route.called


class TestReasoningEffort:
    @respx.mock
    async def test_reasoning_effort_injected(self):
        route = respx.post(f"{KIMI_K3_BASE_URL}/chat/completions").mock(
            return_value=Response(200, json=_ok_completion())
        )
        provider = _provider()
        try:
            await provider.chat(
                [Message(role="user", content="hi")],
                model="kimi-k3",
                reasoning_effort="low",
            )
        finally:
            await provider.close()
        import json

        sent = json.loads(route.calls[0].request.content)
        assert sent["reasoning_effort"] == "low"

    @respx.mock
    async def test_reasoning_effort_omitted_by_default(self):
        route = respx.post(f"{KIMI_K3_BASE_URL}/chat/completions").mock(
            return_value=Response(200, json=_ok_completion())
        )
        provider = _provider()
        try:
            await provider.chat([Message(role="user", content="hi")], model="kimi-k3")
        finally:
            await provider.close()
        import json

        sent = json.loads(route.calls[0].request.content)
        assert "reasoning_effort" not in sent

    def test_invalid_reasoning_effort_rejected(self):
        provider = _provider()
        with pytest.raises(ValueError, match="medium"):
            provider._build_request_payload(
                messages=[Message(role="user", content="hi")],
                model="kimi-k3",
                temperature=0.6,
                max_tokens=64,
                tools=None,
                stream=False,
                reasoning_effort="medium",  # not a valid K3 value
            )


class TestCatalog:
    async def test_list_models_includes_k3_and_k2(self):
        provider = _provider()
        try:
            models = {m["id"]: m for m in await provider.list_models()}
        finally:
            await provider.close()
        assert "kimi-k3" in models
        assert models["kimi-k3"]["context_window"] == 1_048_576
        assert models["kimi-k3"]["supports_thinking"] is True
        assert "kimi-k2-thinking" in models
