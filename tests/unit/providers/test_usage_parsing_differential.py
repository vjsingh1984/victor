"""Differential fixture oracle: victor's mapping vs the vendored sandhi corpus ground truth.

The corpus (``tests/unit/providers/fixtures/sandhi_usage/``) is copied from sandhi's
recorded-fixture QA suite (TD-0001 W1); ``expected_usage.json`` is the neutral ground truth
sandhi's own parsers are verified against. These tests prove:

1. victor's mapping arithmetic over that ground truth (always run, fake ``_sg``),
2. the real wheel reproduces the ground truth on the raw fixture bodies ([binding]),
3. native adapter parsing vs the sandhi-routed path are semantically equivalent,
   with the two intentional divergences asserted explicitly:
   - anthropic non-streaming native LACKS the cache-split keys (latent bug, fixed by routing),
   - openai-SDK native LACKS ``cache_read_input_tokens`` (latent bug, fixed by routing).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

import victor.providers.usage_parsing as up

FIXTURES = Path(__file__).parent / "fixtures" / "sandhi_usage"

CASES = {
    "anthropic": ("complete_cache_split.json", "anthropic"),
    "openai": ("complete.json", "openai"),
    "ollama": ("complete.json", "ollama"),
}


def load(provider: str) -> tuple[Dict[str, Any], Dict[str, int]]:
    body_file, _ = CASES[provider]
    body = json.loads((FIXTURES / provider / body_file).read_text())
    expected = json.loads((FIXTURES / provider / "expected_usage.json").read_text())
    return body, expected


def usage_block(provider: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """The raw usage block an adapter would hold at its parse point."""
    if provider == "ollama":
        return body  # top-level prompt_eval_count/eval_count
    return body["usage"]


def canned_sg(expected: Dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(parse_usage=lambda slug, payload: dict(expected))


@pytest.mark.parametrize("provider", list(CASES))
def test_mapping_arithmetic_matches_expected_usage(provider: str, monkeypatch) -> None:
    body, expected = load(provider)
    monkeypatch.setattr(up, "_sg", canned_sg(expected))
    block = usage_block(provider, body)
    result = up.parse_usage_dict(CASES[provider][1], block)
    assert result is not None

    if provider == "openai":
        # full prompt = fresh (200) + cached (800) = raw prompt_tokens 1000
        assert result["prompt_tokens"] == expected["tokens_in"] + expected["cache_read_tokens"]
        assert result["prompt_tokens"] == block["prompt_tokens"]
        assert result["total_tokens"] == block["total_tokens"]
        assert result["cache_read_input_tokens"] == expected["cache_read_tokens"] == 800
        assert "cache_creation_input_tokens" not in result  # creation==0 → omitted
    elif provider == "anthropic":
        assert result["prompt_tokens"] == expected["tokens_in"] == 1024
        assert result["cache_creation_input_tokens"] == 2048
        assert result["cache_read_input_tokens"] == 4096
        assert result["total_tokens"] == 1024 + 256
    else:  # ollama
        assert result == {"prompt_tokens": 512, "completion_tokens": 128, "total_tokens": 640}
    assert result["completion_tokens"] == expected["tokens_out"]


@pytest.mark.parametrize("provider", list(CASES))
def test_sandhi_routed_matches_expected_usage(provider: str, monkeypatch) -> None:
    """[binding] the real wheel reproduces the corpus ground truth on the fixture bodies."""
    sg = pytest.importorskip("sandhi_gateway", reason="requires the victor[sandhi] extra")
    body, expected = load(provider)
    parsed = sg.parse_usage(CASES[provider][1], json.dumps(body))
    assert {k: parsed[k] for k in expected} == expected


@pytest.mark.parametrize("provider", list(CASES))
def test_native_vs_sandhi_equivalence(provider: str, monkeypatch) -> None:
    """Native adapter parse vs sandhi-routed parse: equivalent, divergences explicit."""
    body, expected = load(provider)
    monkeypatch.setattr(up, "_sg", canned_sg(expected))
    block = usage_block(provider, body)
    routed = up.parse_usage_dict(CASES[provider][1], block)
    assert routed is not None

    if provider == "anthropic":
        # Native non-streaming parse (anthropic_provider._parse_response today):
        native = {
            "prompt_tokens": block["input_tokens"],
            "completion_tokens": block["output_tokens"],
            "total_tokens": block["input_tokens"] + block["output_tokens"],
        }
        for key, value in native.items():
            assert routed[key] == value
        # Intentional divergence: native drops the cache split (latent bug); routed has it.
        assert "cache_creation_input_tokens" not in native
        assert routed["cache_creation_input_tokens"] == 2048
        assert routed["cache_read_input_tokens"] == 4096
    elif provider == "openai":
        # Native SDK parse (openai_provider._parse_response today):
        native = {
            "prompt_tokens": block["prompt_tokens"],
            "completion_tokens": block["completion_tokens"],
            "total_tokens": block["total_tokens"],
        }
        for key, value in native.items():
            assert routed[key] == value
        # Intentional divergence: native drops cached_tokens (latent bug); routed has it.
        assert "cache_read_input_tokens" not in native
        assert routed["cache_read_input_tokens"] == 800
    else:  # ollama — native parse maps eval counts identically; no divergence
        native = {
            "prompt_tokens": block["prompt_eval_count"],
            "completion_tokens": block["eval_count"],
            "total_tokens": block["prompt_eval_count"] + block["eval_count"],
        }
        assert routed == native
