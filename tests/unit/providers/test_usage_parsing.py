"""Unit tests for victor.providers.usage_parsing (sandhi-routed usage extraction).

All tests run without the ``sandhi-gateway`` binding installed: the module-level ``_sg``
seam is monkeypatched with a fake whose canned returns replay the vendored
``expected_usage.json`` corpus ground truth (a lookup, never a reimplemented parser).
Tests marked ``[binding]`` additionally exercise the real wheel and skip when absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

import victor.providers.usage_parsing as up

FIXTURES = Path(__file__).parent / "fixtures" / "sandhi_usage"


def fake_sg(returns: Dict[str, int], calls: list | None = None) -> SimpleNamespace:
    """A fake sandhi_gateway whose parse_usage returns canned neutral usage."""

    def parse_usage(slug: str, payload_json: str) -> Dict[str, int]:
        if calls is not None:
            calls.append((slug, payload_json))
        return dict(returns)

    return SimpleNamespace(parse_usage=parse_usage)


class TestOpenAiMapping:
    def test_openai_mapping_preserves_full_prompt_tokens(self, monkeypatch):
        # sandhi normalizes tokens_in to fresh-only; victor's prompt_tokens stays FULL
        monkeypatch.setattr(
            up,
            "_sg",
            fake_sg(
                {
                    "tokens_in": 40,
                    "tokens_out": 20,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 60,
                }
            ),
        )
        raw = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        result = up.parse_usage_dict("openai", raw)
        assert result == {
            "prompt_tokens": 100,  # 40 fresh + 60 cached
            "completion_tokens": 20,
            "total_tokens": 120,  # raw total preserved
            "cache_read_input_tokens": 60,
        }
        assert "cache_creation_input_tokens" not in result

    def test_openai_total_falls_back_to_sum_when_absent(self, monkeypatch):
        monkeypatch.setattr(
            up,
            "_sg",
            fake_sg(
                {
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 0,
                }
            ),
        )
        result = up.parse_usage_dict("openai", {"prompt_tokens": 10, "completion_tokens": 5})
        assert result["total_tokens"] == 15
        assert "cache_read_input_tokens" not in result


class TestAnthropicMapping:
    def test_anthropic_mapping_is_fresh_only_with_cache_split(self, monkeypatch):
        monkeypatch.setattr(
            up,
            "_sg",
            fake_sg(
                {
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "cache_creation_tokens": 3,
                    "cache_read_tokens": 2,
                }
            ),
        )
        raw = {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
        }
        result = up.parse_usage_dict("anthropic", raw)
        assert result == {
            "prompt_tokens": 10,  # fresh-only == SDK input_tokens (current behavior)
            "completion_tokens": 5,
            "total_tokens": 15,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
        }

    def test_anthropic_cache_keys_only_when_raw_carries_them(self, monkeypatch):
        monkeypatch.setattr(
            up,
            "_sg",
            fake_sg(
                {
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 0,
                }
            ),
        )
        result = up.parse_usage_dict("anthropic", {"input_tokens": 10, "output_tokens": 5})
        assert "cache_creation_input_tokens" not in result
        assert "cache_read_input_tokens" not in result


class TestOllamaMapping:
    def test_ollama_maps_eval_counts(self, monkeypatch):
        monkeypatch.setattr(
            up,
            "_sg",
            fake_sg(
                {
                    "tokens_in": 512,
                    "tokens_out": 128,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 0,
                }
            ),
        )
        raw = {"prompt_eval_count": 512, "eval_count": 128, "done": True}
        result = up.parse_usage_dict("ollama", raw)
        assert result == {"prompt_tokens": 512, "completion_tokens": 128, "total_tokens": 640}


class TestFallbackSemantics:
    def test_returns_none_without_binding(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", None)
        assert up.parse_usage_dict("openai", {"prompt_tokens": 1}) is None
        assert up.sandhi_parse_usage("openai", {"prompt_tokens": 1}) is None

    def test_returns_none_when_parse_raises(self, monkeypatch):
        def boom(slug: str, payload: str) -> Dict[str, int]:
            raise RuntimeError("parser exploded")

        monkeypatch.setattr(up, "_sg", SimpleNamespace(parse_usage=boom))
        assert up.parse_usage_dict("openai", {"prompt_tokens": 1}) is None

    def test_returns_none_for_none_usage(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg({"tokens_in": 1, "tokens_out": 1}))
        assert up.parse_usage_dict("openai", None) is None

    def test_returns_none_when_coercion_fails(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg({"tokens_in": 1, "tokens_out": 1}))
        assert up.parse_usage_dict("openai", object()) is None


class TestCoercionAndEnvelope:
    def test_sdk_object_coerced_via_model_dump(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(
            up,
            "_sg",
            fake_sg(
                {
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 0,
                },
                calls,
            ),
        )

        class FakeSdkUsage:
            def model_dump(self) -> Dict[str, Any]:
                return {"input_tokens": 10, "output_tokens": 5}

        result = up.parse_usage_dict("anthropic", FakeSdkUsage())
        assert result is not None
        slug, payload = calls[0]
        assert slug == "anthropic"
        assert json.loads(payload) == {"usage": {"input_tokens": 10, "output_tokens": 5}}

    def test_envelope_per_slug(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(
            up,
            "_sg",
            fake_sg(
                {
                    "tokens_in": 1,
                    "tokens_out": 1,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 0,
                },
                calls,
            ),
        )
        up.parse_usage_dict("openai", {"prompt_tokens": 1})
        up.parse_usage_dict("ollama", {"prompt_eval_count": 1, "eval_count": 1})
        openai_payload = json.loads(calls[0][1])
        ollama_payload = json.loads(calls[1][1])
        assert "usage" in openai_payload  # wrapped
        assert "usage" not in ollama_payload  # top-level as-is
        assert ollama_payload["prompt_eval_count"] == 1

    def test_slug_map_excludes_google(self):
        assert "google" not in up.SANDHI_SLUGS
        assert "vertex" not in up.SANDHI_SLUGS


class TestRealBinding:
    """[binding] tests — run only when the sandhi-gateway wheel is installed."""

    def test_real_binding_single_sources_the_parse(self, monkeypatch):
        sg = pytest.importorskip("sandhi_gateway", reason="requires the victor[sandhi] extra")
        monkeypatch.setattr(up, "_sg", sg)
        raw = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 60},
        }
        result = up.parse_usage_dict("openai", raw)
        assert result["prompt_tokens"] == 100  # full, reconstructed from fresh + cached
        assert result["cache_read_input_tokens"] == 60

        anthropic_raw = {
            "input_tokens": 1024,
            "output_tokens": 256,
            "cache_creation_input_tokens": 2048,
            "cache_read_input_tokens": 4096,
        }
        result = up.parse_usage_dict("anthropic", anthropic_raw)
        assert result == {
            "prompt_tokens": 1024,
            "completion_tokens": 256,
            "total_tokens": 1280,
            "cache_creation_input_tokens": 2048,
            "cache_read_input_tokens": 4096,
        }
