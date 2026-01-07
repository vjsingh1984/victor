import pytest

from victor.agent.provider_manager import ProviderManager
from victor.providers.base import BaseProvider, CompletionResponse, StreamChunk
from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities


class _DummyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "dummy"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    async def chat(
        self,
        messages,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools=None,
        **kwargs,
    ) -> CompletionResponse:
        return CompletionResponse(content="ok")

    async def stream(
        self,
        messages,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools=None,
        **kwargs,
    ):
        yield StreamChunk(content="ok")

    async def discover_capabilities(self, model: str) -> ProviderRuntimeCapabilities:
        return ProviderRuntimeCapabilities(
            provider=self.name,
            model=model,
            context_window=999,
            supports_tools=True,
            supports_streaming=True,
            source="discovered",
        )

    async def close(self) -> None:
        return None


class _FailingDiscoverProvider(_DummyProvider):
    @property
    def name(self) -> str:
        return "dummy-fallback"

    async def discover_capabilities(self, model: str) -> ProviderRuntimeCapabilities:
        raise RuntimeError("discovery failed")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_provider_manager_caches_discovery_capabilities() -> None:
    provider = _DummyProvider()
    manager = ProviderManager(settings=None, initial_provider=provider, initial_model="model-x")

    caps = await manager._discover_and_cache_capabilities()  # private but intentional for test
    assert caps.context_window == 999
    assert manager.get_context_window() == 999
    info = manager.get_info()
    assert info["context_window"] == 999


@pytest.mark.asyncio
async def test_provider_manager_falls_back_to_config_on_discovery_error() -> None:
    provider = _FailingDiscoverProvider()
    manager = ProviderManager(settings=None, initial_provider=provider, initial_model="model-y")

    caps = await manager._discover_and_cache_capabilities()  # falls back to config limits
    assert caps.context_window > 0
    assert manager.get_context_window() == caps.context_window
