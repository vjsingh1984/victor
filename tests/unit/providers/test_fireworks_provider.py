"""Fireworks policy tests; HTTP/SSE behavior is owned and tested by Sandhi."""

from __future__ import annotations

import pytest

from victor.providers.fireworks_provider import FIREWORKS_MODELS, FireworksProvider
from victor.providers.resolution import APIKeyNotFoundError
from victor.providers.sandhi_transport import SandhiTypedProviderMixin


def test_fireworks_is_a_typed_policy_shell() -> None:
    provider = FireworksProvider(api_key="test-key")
    assert isinstance(provider, SandhiTypedProviderMixin)
    assert provider.name == "fireworks"
    assert provider.base_url == "https://api.fireworks.ai/inference/v1"
    assert provider.supports_tools()
    assert provider.supports_streaming()
    assert provider.cache_cost_model().read_discount == 0.5


def test_fireworks_models_and_context_policy_are_victor_owned() -> None:
    provider = FireworksProvider(api_key="test-key")
    assert "accounts/fireworks/models/llama-v3p3-70b-instruct" in FIREWORKS_MODELS
    assert provider.get_default_model() in FIREWORKS_MODELS
    assert provider.context_window("accounts/fireworks/models/deepseek-v3p2") == 64_000


def test_fireworks_requires_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(APIKeyNotFoundError):
        FireworksProvider(non_interactive=True)
