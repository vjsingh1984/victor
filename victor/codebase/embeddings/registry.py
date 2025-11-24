"""Registry for embedding providers."""

from typing import Dict, List, Type

from victor.codebase.embeddings.base import BaseEmbeddingProvider, EmbeddingConfig


class EmbeddingRegistry:
    """Central registry for embedding providers.

    Similar to the LLM provider registry, this allows plugins to register
    themselves and be discovered/created by name.

    Usage:
        # Register a provider
        EmbeddingRegistry.register("chromadb", ChromaDBProvider)

        # Create a provider
        config = EmbeddingConfig(provider="chromadb")
        provider = EmbeddingRegistry.create(config)
    """

    _providers: Dict[str, Type[BaseEmbeddingProvider]] = {}

    @classmethod
    def register(
        cls, name: str, provider_class: Type[BaseEmbeddingProvider]
    ) -> None:
        """Register an embedding provider.

        Args:
            name: Provider name (e.g., "chromadb", "proximadb")
            provider_class: Provider class (must inherit from BaseEmbeddingProvider)
        """
        if not issubclass(provider_class, BaseEmbeddingProvider):
            raise TypeError(
                f"{provider_class} must inherit from BaseEmbeddingProvider"
            )

        cls._providers[name] = provider_class
        print(f"Registered embedding provider: {name}")

    @classmethod
    def get(cls, name: str) -> Type[BaseEmbeddingProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            KeyError: If provider not found
        """
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise KeyError(
                f"Unknown embedding provider: {name}. "
                f"Available: {available if available else 'none'}"
            )

        return cls._providers[name]

    @classmethod
    def create(cls, config: EmbeddingConfig) -> BaseEmbeddingProvider:
        """Create a provider instance from configuration.

        Args:
            config: Embedding configuration

        Returns:
            Initialized provider instance
        """
        provider_class = cls.get(config.provider)
        return provider_class(config)

    @classmethod
    def list_providers(cls) -> List[str]:
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


# Auto-discovery: Import and register all providers
def _auto_register_providers() -> None:
    """Automatically discover and register embedding providers."""
    try:
        from victor.codebase.embeddings.chromadb_provider import ChromaDBProvider

        EmbeddingRegistry.register("chromadb", ChromaDBProvider)
    except ImportError:
        pass  # ChromaDB not installed

    try:
        from victor.codebase.embeddings.proximadb_provider import ProximaDBProvider

        EmbeddingRegistry.register("proximadb", ProximaDBProvider)
    except ImportError:
        pass  # ProximaDB not available


# Auto-register on module import
_auto_register_providers()
