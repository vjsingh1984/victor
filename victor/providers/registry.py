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

from typing import Any, Optional

from victor.core.registry import BaseRegistry
from victor.providers.base import BaseProvider, ProviderNotFoundError

__all__ = ["ProviderRegistry", "ProviderNotFoundError"]


class _ProviderRegistryImpl(BaseRegistry[str, type[BaseProvider]]):
    """Internal registry implementation for provider management.

    Extends BaseRegistry to provide provider-specific functionality including:
    - Lazy import of provider classes (startup optimization)
    - Factory method for instantiating providers
    - Raises ProviderNotFoundError on missing providers
    """

    def get_or_raise(self, name: str) -> type[BaseProvider]:
        """Get a provider class by name, raising if not found.

        This method implements lazy loading - provider classes are only imported
        when first requested. This significantly improves startup time.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ProviderNotFoundError: If provider not found
        """
        # First check if already in registry (custom registered provider)
        provider = self.get(name)
        if provider is not None:
            return provider

        # Try lazy import from _PROVIDER_IMPORTS
        if name in _PROVIDER_IMPORTS:
            module_path, class_name = _PROVIDER_IMPORTS[name]
            import importlib

            try:
                module = importlib.import_module(module_path)
                provider_class = getattr(module, class_name)

                # Cache it in the registry for faster subsequent access
                self.register(name, provider_class)

                # Also register any aliases
                canonical_name = _PROVIDER_ALIASES.get(name, name)
                if canonical_name != name and canonical_name not in self:
                    self.register(canonical_name, provider_class)

                return provider_class
            except (ImportError, AttributeError) as e:
                raise ProviderNotFoundError(
                    message=f"Failed to import provider '{name}': {e}",
                    provider=name,
                ) from e

        # Provider not found
        raise ProviderNotFoundError(
            message=f"Provider '{name}' not found. Available: {', '.join(sorted(_PROVIDER_IMPORTS.keys()))}",
            provider=name,
        )

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

    def list_all(self) -> list[str]:
        """List all available provider names (including lazy-loaded providers).

        Returns:
            List of all available provider names
        """
        # Combine registered providers with lazy import providers
        registered = set(super().list_all())
        lazy_providers = set(_PROVIDER_IMPORTS.keys())
        return sorted(registered | lazy_providers)


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
    def register(cls, name: str, provider_class: type[BaseProvider]) -> None:
        """Register a provider.

        Args:
            name: Provider name (e.g., "ollama", "anthropic")
            provider_class: Provider class
        """
        _registry_instance.register(name, provider_class)

    @classmethod
    def get(cls, name: str) -> type[BaseProvider]:
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
    def get_optional(cls, name: str) -> Optional[type[BaseProvider]]:
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
    def list_providers(cls) -> list[str]:
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


# Lazy provider import map for startup performance optimization
# Maps provider names to their (module_path, class_name) tuples
# Providers are only imported when actually requested via ProviderRegistry.get()
_PROVIDER_IMPORTS: dict[str, tuple[str, str]] = {
    # Local providers
    "ollama": ("victor.providers.ollama_provider", "OllamaProvider"),
    "lmstudio": ("victor.providers.lmstudio_provider", "LMStudioProvider"),
    "vllm": ("victor.providers.vllm_provider", "VLLMProvider"),
    "llamacpp": ("victor.providers.llamacpp_provider", "LlamaCppProvider"),
    "llama-cpp": ("victor.providers.llamacpp_provider", "LlamaCppProvider"),  # Alias
    "llama.cpp": ("victor.providers.llamacpp_provider", "LlamaCppProvider"),  # Alias
    # Major cloud providers
    "anthropic": ("victor.providers.anthropic_provider", "AnthropicProvider"),
    "claude": ("victor.providers.anthropic_provider", "AnthropicProvider"),  # Alias
    "openai": ("victor.providers.openai_provider", "OpenAIProvider"),
    "google": ("victor.providers.google_provider", "GoogleProvider"),
    "gemini": ("victor.providers.google_provider", "GoogleProvider"),  # Alias
    # AI research companies
    "xai": ("victor.providers.xai_provider", "XAIProvider"),
    "grok": ("victor.providers.xai_provider", "XAIProvider"),  # Alias for xai
    "zai": ("victor.providers.zai_provider", "ZAIProvider"),
    "zhipuai": ("victor.providers.zai_provider", "ZAIProvider"),  # Alias
    "zhipu": ("victor.providers.zai_provider", "ZAIProvider"),  # Alias
    "moonshot": ("victor.providers.moonshot_provider", "MoonshotProvider"),
    "kimi": ("victor.providers.moonshot_provider", "MoonshotProvider"),  # Alias
    "deepseek": ("victor.providers.deepseek_provider", "DeepSeekProvider"),
    # Free-tier providers (2025)
    "groqcloud": ("victor.providers.groq_provider", "GroqProvider"),
    "mistral": ("victor.providers.mistral_provider", "MistralProvider"),
    "together": ("victor.providers.together_provider", "TogetherProvider"),
    "openrouter": ("victor.providers.openrouter_provider", "OpenRouterProvider"),
    "fireworks": ("victor.providers.fireworks_provider", "FireworksProvider"),
    "cerebras": ("victor.providers.cerebras_provider", "CerebrasProvider"),
    # Enterprise cloud providers
    "vertex": ("victor.providers.vertex_provider", "VertexAIProvider"),
    "vertexai": ("victor.providers.vertex_provider", "VertexAIProvider"),  # Alias
    "azure": ("victor.providers.azure_openai_provider", "AzureOpenAIProvider"),
    "azure-openai": ("victor.providers.azure_openai_provider", "AzureOpenAIProvider"),  # Alias
    "bedrock": ("victor.providers.bedrock_provider", "BedrockProvider"),
    "aws": ("victor.providers.bedrock_provider", "BedrockProvider"),  # Alias
    "huggingface": ("victor.providers.huggingface_provider", "HuggingFaceProvider"),
    "hf": ("victor.providers.huggingface_provider", "HuggingFaceProvider"),  # Alias
    "replicate": ("victor.providers.replicate_provider", "ReplicateProvider"),
}

# Reverse alias map to canonical name (for listing providers)
_PROVIDER_ALIASES: dict[str, str] = {
    "claude": "anthropic",
    "gemini": "google",
    "grok": "xai",
    "zhipuai": "zai",
    "zhipu": "zai",
    "kimi": "moonshot",
    "llama-cpp": "llamacpp",
    "llama.cpp": "llamacpp",
    "vertexai": "vertex",
    "azure-openai": "azure",
    "aws": "bedrock",
    "hf": "huggingface",
}


# Auto-register all providers (lazy loading - only registers metadata)
def _register_default_providers() -> None:
    """Register all default providers using lazy loading.

    This function only registers the provider metadata, not the actual classes.
    Provider classes are imported lazily when first requested via ProviderRegistry.get().
    This significantly improves startup time by avoiding importing all 21 providers.
    """
    # No actual imports happen here - just metadata registration
    # The real imports happen in _ProviderRegistryImpl.get_or_raise()
    pass


# Register providers on module import
_register_default_providers()
