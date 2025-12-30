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

from typing import Any, Dict, List, Optional, Type

from victor.core.registry import BaseRegistry
from victor.providers.base import BaseProvider, ProviderNotFoundError

__all__ = ["ProviderRegistry", "ProviderNotFoundError"]


class _ProviderRegistryImpl(BaseRegistry[str, Type[BaseProvider]]):
    """Internal registry implementation for provider management.

    Extends BaseRegistry to provide provider-specific functionality including:
    - Factory method for instantiating providers
    - Raises ProviderNotFoundError on missing providers
    """

    def get_or_raise(self, name: str) -> Type[BaseProvider]:
        """Get a provider class by name, raising if not found.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ProviderNotFoundError: If provider not found
        """
        provider = self.get(name)
        if provider is None:
            raise ProviderNotFoundError(
                message=f"Provider '{name}' not found. Available: {', '.join(self.list_all())}",
                provider=name,
            )
        return provider

    def create(self, name: str, **kwargs: Any) -> BaseProvider:
        """Create a provider instance.

        Args:
            name: Provider name
            **kwargs: Provider initialization arguments

        Returns:
            Provider instance

        Raises:
            ProviderNotFoundError: If provider not found
        """
        provider_class = self.get_or_raise(name)
        return provider_class(**kwargs)


# Singleton instance for backward compatibility
_registry_instance = _ProviderRegistryImpl()


class ProviderRegistry:
    """Registry for LLM provider management.

    This is a static class facade that maintains backward compatibility
    with existing code while delegating to a BaseRegistry-based implementation.

    For new code, consider using the instance-based approach via
    `get_provider_registry()` for better testability.
    """

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Register a provider.

        Args:
            name: Provider name (e.g., "ollama", "anthropic")
            provider_class: Provider class
        """
        _registry_instance.register(name, provider_class)

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
        return _registry_instance.get_or_raise(name)

    @classmethod
    def get_optional(cls, name: str) -> Optional[Type[BaseProvider]]:
        """Get a provider class by name, returning None if not found.

        Args:
            name: Provider name

        Returns:
            Provider class or None if not found
        """
        return _registry_instance.get(name)

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
        return _registry_instance.create(name, **kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """
        return _registry_instance.list_all()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name

        Returns:
            True if registered, False otherwise
        """
        return name in _registry_instance

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a provider.

        Args:
            name: Provider name

        Returns:
            True if the provider was found and removed, False otherwise
        """
        return _registry_instance.unregister(name)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers."""
        _registry_instance.clear()


def get_provider_registry() -> _ProviderRegistryImpl:
    """Get the provider registry instance.

    This function provides access to the underlying BaseRegistry-based
    implementation for testing or advanced use cases.

    Returns:
        The singleton provider registry instance
    """
    return _registry_instance


# Auto-register all providers
def _register_default_providers() -> None:
    """Register all default providers."""
    from victor.providers.ollama_provider import OllamaProvider
    from victor.providers.anthropic_provider import AnthropicProvider
    from victor.providers.openai_provider import OpenAIProvider
    from victor.providers.google_provider import GoogleProvider
    from victor.providers.xai_provider import XAIProvider
    from victor.providers.lmstudio_provider import LMStudioProvider
    from victor.providers.moonshot_provider import MoonshotProvider
    from victor.providers.deepseek_provider import DeepSeekProvider
    from victor.providers.groq_provider import GroqProvider
    from victor.providers.mistral_provider import MistralProvider
    from victor.providers.together_provider import TogetherProvider
    from victor.providers.openrouter_provider import OpenRouterProvider
    from victor.providers.fireworks_provider import FireworksProvider
    from victor.providers.cerebras_provider import CerebrasProvider

    ProviderRegistry.register("ollama", OllamaProvider)
    ProviderRegistry.register("anthropic", AnthropicProvider)
    ProviderRegistry.register("openai", OpenAIProvider)
    # LMStudio uses dedicated provider (similar to Ollama)
    # with httpx, tiered URL selection, and 300s timeout
    ProviderRegistry.register("lmstudio", LMStudioProvider)
    # vLLM high-throughput inference server
    from victor.providers.vllm_provider import VLLMProvider

    ProviderRegistry.register("vllm", VLLMProvider)
    # llama.cpp server (CPU-friendly local inference)
    from victor.providers.llamacpp_provider import LlamaCppProvider

    ProviderRegistry.register("llamacpp", LlamaCppProvider)
    ProviderRegistry.register("llama-cpp", LlamaCppProvider)  # Alias
    ProviderRegistry.register("llama.cpp", LlamaCppProvider)  # Alias
    ProviderRegistry.register("google", GoogleProvider)
    ProviderRegistry.register("xai", XAIProvider)
    ProviderRegistry.register("grok", XAIProvider)  # Alias for xai
    # Moonshot AI for Kimi K2 models (OpenAI-compatible with reasoning traces)
    ProviderRegistry.register("moonshot", MoonshotProvider)
    ProviderRegistry.register("kimi", MoonshotProvider)  # Alias for moonshot
    # DeepSeek for DeepSeek-V3 models (chat and reasoner)
    ProviderRegistry.register("deepseek", DeepSeekProvider)
    # Groq Cloud for ultra-fast LLM inference (free tier available)
    ProviderRegistry.register("groqcloud", GroqProvider)

    # New free-tier providers (2025)
    # Mistral AI - 500K tokens/min free tier
    ProviderRegistry.register("mistral", MistralProvider)
    # Together AI - $25 free credits
    ProviderRegistry.register("together", TogetherProvider)
    # OpenRouter - unified gateway, free tier with daily limits
    ProviderRegistry.register("openrouter", OpenRouterProvider)
    # Fireworks AI - $1 free credits, fast inference
    ProviderRegistry.register("fireworks", FireworksProvider)
    # Cerebras - ultra-fast inference, free tier
    ProviderRegistry.register("cerebras", CerebrasProvider)

    # Enterprise cloud providers
    from victor.providers.vertex_provider import VertexAIProvider
    from victor.providers.azure_openai_provider import AzureOpenAIProvider
    from victor.providers.bedrock_provider import BedrockProvider
    from victor.providers.huggingface_provider import HuggingFaceProvider
    from victor.providers.replicate_provider import ReplicateProvider

    # Google Cloud Vertex AI - Enterprise Gemini access
    ProviderRegistry.register("vertex", VertexAIProvider)
    ProviderRegistry.register("vertexai", VertexAIProvider)  # Alias
    # Azure OpenAI - Enterprise OpenAI + Phi models
    ProviderRegistry.register("azure", AzureOpenAIProvider)
    ProviderRegistry.register("azure-openai", AzureOpenAIProvider)  # Alias
    # AWS Bedrock - Claude, Llama, Mistral, Titan
    ProviderRegistry.register("bedrock", BedrockProvider)
    ProviderRegistry.register("aws", BedrockProvider)  # Alias
    # Hugging Face Inference API - 1000s of open models
    ProviderRegistry.register("huggingface", HuggingFaceProvider)
    ProviderRegistry.register("hf", HuggingFaceProvider)  # Alias
    # Replicate - Neocloud for open models
    ProviderRegistry.register("replicate", ReplicateProvider)


# Register providers on module import
_register_default_providers()
