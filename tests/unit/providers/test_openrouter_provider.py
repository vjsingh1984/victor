"""OpenRouter policy tests; HTTP/SSE behavior is owned and tested by Sandhi."""

from __future__ import annotations

import pytest

from victor.providers.openrouter_provider import OPENROUTER_MODELS, OpenRouterProvider
from victor.providers.resolution import APIKeyNotFoundError
from victor.providers.sandhi_transport import SandhiTypedProviderMixin


def test_openrouter_is_a_typed_policy_shell_with_sandhi_owned_headers() -> None:
    provider = OpenRouterProvider(
        api_key="test-key", site_url="https://example.test", site_name="Victor"
    )
    assert isinstance(provider, SandhiTypedProviderMixin)
    assert provider.name == "openrouter"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    assert provider._wire_headers == {
        "HTTP-Referer": "https://example.test",
        "X-Title": "Victor",
    }
    assert provider.cache_cost_model().read_discount == 0.575


def test_openrouter_models_and_context_policy_are_victor_owned() -> None:
    provider = OpenRouterProvider(api_key="test-key")
    assert "meta-llama/llama-3.3-70b-instruct:free" in OPENROUTER_MODELS
    assert provider.get_default_model() in OPENROUTER_MODELS
    assert provider.context_window("google/gemini-2.5-flash:free") == 128_000


def test_openrouter_requires_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(APIKeyNotFoundError):
        OpenRouterProvider(non_interactive=True)
