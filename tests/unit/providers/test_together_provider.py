"""Together policy tests; transport parity lives at the shared typed Sandhi boundary."""

from __future__ import annotations

import pytest

from victor.providers.resolution import APIKeyNotFoundError
from victor.providers.sandhi_transport import SandhiTypedProviderMixin
from victor.providers.together_provider import TOGETHER_MODELS, TogetherProvider


def test_together_is_a_typed_policy_shell() -> None:
    provider = TogetherProvider(api_key="test-key")
    assert isinstance(provider, SandhiTypedProviderMixin)
    assert provider.name == "together"
    assert provider.base_url == "https://api.together.xyz/v1"
    assert provider.supports_tools()
    assert provider.supports_streaming()


def test_together_models_and_context_policy_are_victor_owned() -> None:
    provider = TogetherProvider(api_key="test-key")
    assert "meta-llama/Llama-3.3-70B-Instruct-Turbo" in TOGETHER_MODELS
    assert provider.get_default_model() in TOGETHER_MODELS
    assert provider.context_window("meta-llama/Llama-3.3-70B-Instruct-Turbo") == 128_000
    assert provider.context_window("openai/gpt-oss-120b") == 32_768


def test_together_requires_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(APIKeyNotFoundError):
        TogetherProvider(non_interactive=True)
