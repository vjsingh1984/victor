"""Contract tests for BaseProvider capability defaults.

Provider capabilities are a single surface: the ``supports_*()`` methods on
BaseProvider. Any provider can be asked about any capability without an
AttributeError (Liskov); each defaults to ``False`` and providers opt in by
overriding (Interface Segregation). These tests pin that contract — notably
that ``supports_vision`` is answerable on the base like its siblings — so the
surface cannot silently drift or sprout a parallel helper/Protocol layer again.
"""

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    StreamChunk,
)

# Every capability query the base contract must answer, with its default.
CAPABILITY_DEFAULTS = {
    "supports_tools": False,
    "supports_streaming": False,
    "supports_vision": False,
    "supports_prompt_caching": False,
    "supports_kv_prefix_caching": False,
}


class _MinimalProvider(BaseProvider):
    """A provider that opts into nothing — pure base defaults."""

    @property
    def name(self) -> str:
        return "minimal"

    async def chat(self, messages, *, model: str, **kwargs) -> CompletionResponse:
        return CompletionResponse(content="ok")

    async def stream(self, messages, *, model: str, **kwargs):
        yield StreamChunk(content="ok")

    async def close(self) -> None:
        return None


def test_base_answers_every_capability_query() -> None:
    """A bare provider answers all capability queries (no AttributeError)."""
    provider = _MinimalProvider()
    for method, expected in CAPABILITY_DEFAULTS.items():
        assert getattr(provider, method)() is expected, f"{method} default"


def test_supports_reasoning_effort_defaults_false() -> None:
    provider = _MinimalProvider()
    assert provider.supports_reasoning_effort() is False
    assert provider.supports_reasoning_effort("any-model") is False


def test_opt_in_overrides_are_reflected() -> None:
    """Overriding a single capability does not disturb the others."""

    class _VisionOnly(_MinimalProvider):
        def supports_vision(self) -> bool:
            return True

    provider = _VisionOnly()
    assert provider.supports_vision() is True
    # Siblings remain at their defaults.
    assert provider.supports_tools() is False
    assert provider.supports_streaming() is False
