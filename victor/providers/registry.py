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

"""Provider registry for managing and discovering LLM providers."""

from typing import Any, Dict, Type

from victor.providers.base import BaseProvider, ProviderNotFoundError

__all__ = ["ProviderRegistry", "ProviderNotFoundError"]


class ProviderRegistry:
    """Registry for LLM provider management."""

    _providers: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Register a provider.

        Args:
            name: Provider name (e.g., "ollama", "anthropic")
            provider_class: Provider class
        """
        cls._providers[name] = provider_class

    @classmethod
    def get(cls, name: str) -> Type[BaseProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ProviderNotFoundError: If provider not found
        """
        provider = cls._providers.get(name)
        if provider is None:
            raise ProviderNotFoundError(
                message=f"Provider '{name}' not found. Available: {', '.join(cls._providers.keys())}",
                provider=name,
            )
        return provider

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseProvider:
        """Create a provider instance.

        Args:
            name: Provider name
            **kwargs: Provider initialization arguments

        Returns:
            Provider instance

        Raises:
            ProviderNotFoundError: If provider not found
        """
        provider_class = cls.get(name)
        return provider_class(**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name

        Returns:
            True if registered, False otherwise
        """
        return name in cls._providers


# Auto-register all providers
def _register_default_providers() -> None:
    """Register all default providers."""
    from victor.providers.ollama_provider import OllamaProvider
    from victor.providers.anthropic_provider import AnthropicProvider
    from victor.providers.openai_provider import OpenAIProvider
    from victor.providers.google_provider import GoogleProvider
    from victor.providers.xai_provider import XAIProvider
    from victor.providers.lmstudio_provider import LMStudioProvider

    ProviderRegistry.register("ollama", OllamaProvider)
    ProviderRegistry.register("anthropic", AnthropicProvider)
    ProviderRegistry.register("openai", OpenAIProvider)
    # LMStudio uses dedicated provider (similar to Ollama)
    # with httpx, tiered URL selection, and 300s timeout
    ProviderRegistry.register("lmstudio", LMStudioProvider)
    # vLLM uses OpenAI-compatible endpoints
    ProviderRegistry.register("vllm", OpenAIProvider)
    ProviderRegistry.register("google", GoogleProvider)
    ProviderRegistry.register("xai", XAIProvider)
    ProviderRegistry.register("grok", XAIProvider)  # Alias for xai


# Register providers on module import
_register_default_providers()
