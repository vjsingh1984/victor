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

"""Wave-1 adapter adoption (ADR-0047 D10a step 2): OpenAIProvider routes usage
parsing through ``victor.providers.usage_parsing.parse_usage_dict``.

The routed path fixes a latent bug — the native parse dropped
``prompt_tokens_details.cached_tokens``. ``prompt_tokens`` stays the FULL count
(fresh + cached), identical to the SDK value, so happy-path values match today's
plus the new cache keys. Without the binding the historical 3-key dict is kept.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

import victor.providers.usage_parsing as up
from victor.providers.openai_provider import OpenAIProvider


def fake_sg(returns: Dict[str, int]) -> SimpleNamespace:
    def parse_usage(slug: str, payload_json: str) -> Dict[str, int]:
        return dict(returns)

    return SimpleNamespace(parse_usage=parse_usage)


def _make_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="sk-test-fake")


RAW_BLOCK = {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120,
    "prompt_tokens_details": {"cached_tokens": 60},
}

NEUTRAL = {
    "tokens_in": 40,  # fresh-only; 40 + 60 cached = 100 full
    "tokens_out": 20,
    "cache_creation_tokens": 0,
    "cache_read_tokens": 60,
}


def _fake_usage_obj(raw_block: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        model_dump=lambda: dict(raw_block),
        prompt_tokens=raw_block.get("prompt_tokens", 0),
        completion_tokens=raw_block.get("completion_tokens", 0),
        total_tokens=raw_block.get("total_tokens", 0),
    )


def _fake_response(usage_obj: Any) -> SimpleNamespace:
    message = SimpleNamespace(content="hello", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=usage_obj)


class TestNonStreaming:
    def test_nonstream_gains_cached_split(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL))
        provider = _make_provider()
        parsed = provider._parse_response(_fake_response(_fake_usage_obj(RAW_BLOCK)), "gpt-test")
        assert parsed.usage == {
            "prompt_tokens": 100,  # FULL count preserved (40 fresh + 60 cached)
            "completion_tokens": 20,
            "total_tokens": 120,  # raw total preserved
            "cache_read_input_tokens": 60,
        }
        assert "cache_creation_input_tokens" not in parsed.usage

    def test_fallback_without_binding(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", None)
        provider = _make_provider()
        parsed = provider._parse_response(_fake_response(_fake_usage_obj(RAW_BLOCK)), "gpt-test")
        # Exactly the historical 3-key native dict.
        assert parsed.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }

    def test_no_usage_stays_none(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL))
        provider = _make_provider()
        parsed = provider._parse_response(_fake_response(None), "gpt-test")
        assert parsed.usage is None


class TestStreamingFinalChunk:
    def _usage_only_chunk(self, usage_obj: Any) -> SimpleNamespace:
        # stream_options.include_usage sends a final usage-only chunk (no choices).
        return SimpleNamespace(choices=[], usage=usage_obj)

    def test_stream_final_chunk_gains_cached_split(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL))
        provider = _make_provider()
        chunk = provider._parse_stream_chunk(self._usage_only_chunk(_fake_usage_obj(RAW_BLOCK)))
        assert chunk is not None and chunk.is_final
        assert chunk.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cache_read_input_tokens": 60,
        }

    def test_stream_final_chunk_fallback_without_binding(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", None)
        provider = _make_provider()
        chunk = provider._parse_stream_chunk(self._usage_only_chunk(_fake_usage_obj(RAW_BLOCK)))
        assert chunk is not None and chunk.is_final
        assert chunk.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }


class TestBinding:
    def test_nonstream_routed_with_real_binding(self):
        pytest.importorskip("sandhi_gateway")
        provider = _make_provider()
        parsed = provider._parse_response(_fake_response(_fake_usage_obj(RAW_BLOCK)), "gpt-test")
        assert parsed.usage is not None
        assert parsed.usage["prompt_tokens"] == 100
        assert parsed.usage["cache_read_input_tokens"] == 60
