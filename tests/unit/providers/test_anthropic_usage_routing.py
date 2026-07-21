# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Wave-1 adapter adoption (ADR-0047 D10a step 2): AnthropicProvider routes usage
parsing through ``victor.providers.usage_parsing.parse_usage_dict``.

The routed path fixes a latent bug — the native non-streaming parse dropped the
prompt-cache split entirely. When the sandhi binding is absent (``_sg is None``)
the historical native dict is preserved verbatim.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import victor.providers.usage_parsing as up
from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.base import Message


def fake_sg(returns: Dict[str, int], calls: list | None = None) -> SimpleNamespace:
    """A fake sandhi_gateway whose parse_usage returns canned neutral usage."""

    def parse_usage(slug: str, payload_json: str) -> Dict[str, int]:
        if calls is not None:
            calls.append((slug, payload_json))
        return dict(returns)

    return SimpleNamespace(parse_usage=parse_usage)


def _make_provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="sk-test-fake")


def _fake_usage_obj(
    raw_block: Dict[str, Any], *, input_tokens: int, output_tokens: int
) -> SimpleNamespace:
    """SDK-usage-like object: pydantic-ish model_dump() plus attribute access."""
    return SimpleNamespace(
        model_dump=lambda: dict(raw_block),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=raw_block.get("cache_creation_input_tokens"),
        cache_read_input_tokens=raw_block.get("cache_read_input_tokens"),
    )


def _fake_response(usage_obj: Any) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")],
        stop_reason="end_turn",
        usage=usage_obj,
    )


RAW_BLOCK = {
    "input_tokens": 10,
    "output_tokens": 5,
    "cache_creation_input_tokens": 3,
    "cache_read_input_tokens": 2,
}

NEUTRAL = {
    "tokens_in": 10,
    "tokens_out": 5,
    "cache_creation_tokens": 3,
    "cache_read_tokens": 2,
}


class TestNonStreaming:
    def test_nonstream_usage_routed_gains_cache_split(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL))
        provider = _make_provider()
        usage_obj = _fake_usage_obj(RAW_BLOCK, input_tokens=10, output_tokens=5)
        parsed = provider._parse_response(_fake_response(usage_obj), model="claude-test")
        assert parsed.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
        }

    def test_nonstream_fallback_matches_legacy_dict(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", None)
        provider = _make_provider()
        usage_obj = _fake_usage_obj(RAW_BLOCK, input_tokens=10, output_tokens=5)
        parsed = provider._parse_response(_fake_response(usage_obj), model="claude-test")
        # Exactly the historical 3-key native dict (cache split still dropped).
        assert parsed.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_nonstream_no_usage_stays_none(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL))
        provider = _make_provider()
        parsed = provider._parse_response(_fake_response(None), model="claude-test")
        assert parsed.usage is None


class _FakeSDKStream:
    """Duck-typed stand-in for the anthropic SDK's streaming context manager."""

    def __init__(self, events: List[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> "_FakeSDKStream":
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for event in self._events:
            yield event


def _stream_events(usage_obj: Any, output_tokens: int = 5) -> List[SimpleNamespace]:
    return [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage_obj)),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="hi"),
            index=0,
        ),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=output_tokens)),
        SimpleNamespace(type="message_stop"),
    ]


async def _collect_final_usage(provider: AnthropicProvider, events: List[Any]):
    provider.client.messages.stream = lambda **kwargs: _FakeSDKStream(events)
    final = None
    async for chunk in provider.stream([Message(role="user", content="hi")], model="claude-test"):
        if chunk.is_final:
            final = chunk
    assert final is not None
    return final.usage


class TestStreaming:
    async def test_stream_final_usage_routed_gains_cache_split(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL))
        provider = _make_provider()
        # message_start carries input + cache tokens; output arrives in message_delta.
        start_block = {
            "input_tokens": 10,
            "output_tokens": 0,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
        }
        usage_obj = _fake_usage_obj(start_block, input_tokens=10, output_tokens=0)
        usage = await _collect_final_usage(provider, _stream_events(usage_obj))
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
        }

    async def test_stream_final_usage_fallback_matches_legacy_shape(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", None)
        provider = _make_provider()
        start_block = {
            "input_tokens": 10,
            "output_tokens": 0,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
        }
        usage_obj = _fake_usage_obj(start_block, input_tokens=10, output_tokens=0)
        usage = await _collect_final_usage(provider, _stream_events(usage_obj))
        # Legacy incremental assembly (attrs-based) — preserved as the fallback.
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
        }

    async def test_stream_usage_obj_without_model_dump_uses_fallback(self, monkeypatch):
        # Sandhi binding present but message_start usage lacks model_dump()
        # → _raw_start_usage stays None → legacy assembly result kept.
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL))
        provider = _make_provider()
        usage_obj = SimpleNamespace(
            input_tokens=10,
            output_tokens=0,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        )
        usage = await _collect_final_usage(provider, _stream_events(usage_obj))
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


class TestBinding:
    def test_nonstream_routed_with_real_binding(self):
        pytest.importorskip("sandhi_gateway")
        provider = _make_provider()
        usage_obj = _fake_usage_obj(RAW_BLOCK, input_tokens=10, output_tokens=5)
        parsed = provider._parse_response(_fake_response(usage_obj), model="claude-test")
        assert parsed.usage is not None
        assert parsed.usage["prompt_tokens"] == 10
        assert parsed.usage["cache_creation_input_tokens"] == 3
        assert parsed.usage["cache_read_input_tokens"] == 2
