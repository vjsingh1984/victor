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

"""Registry for embedding providers."""

import logging
from typing import Dict, List, Type

from victor.coding.codebase.embeddings.base import BaseEmbeddingProvider, EmbeddingConfig

logger = logging.getLogger(__name__)

# Flag to track if ProximaDB is available
PROXIMADB_AVAILABLE = False


class EmbeddingRegistry:
    """Central registry for embedding providers.

    Similar to the LLM provider registry, this allows plugins to register
    themselves and be discovered/created by name.

    Usage:
        # Register a provider
        EmbeddingRegistry.register("chromadb", ChromaDBProvider)

        # Create a provider
        config = EmbeddingConfig(vector_store="chromadb")
        provider = EmbeddingRegistry.create(config)
    """

    _providers: Dict[str, Type[BaseEmbeddingProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseEmbeddingProvider]) -> None:
        """Register an embedding provider.

        Args:
            name: Provider name (e.g., "chromadb", "lancedb")
            provider_class: Provider class (must inherit from BaseEmbeddingProvider)
        """
        if not issubclass(provider_class, BaseEmbeddingProvider):
            raise TypeError(f"{provider_class} must inherit from BaseEmbeddingProvider")

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

        Raises:
            KeyError: If provider is not available
        """
        # Warn if trying to use ProximaDB when not available
        if config.vector_store == "proximadb" and not PROXIMADB_AVAILABLE:
            logger.warning(
                "ProximaDB is configured but not installed. "
                "Install with: pip install victor-ai[vector-experimental]. "
                "Using LanceDB as fallback."
            )
            # Fallback to LanceDB - preserve extra_config (includes rebuild_on_corruption flag)
            from victor.coding.codebase.embeddings.base import EmbeddingConfig as Config

            config = Config(
                vector_store="lancedb",
                persist_directory=config.persist_directory,
                embedding_model_type=config.embedding_model_type,
                embedding_model=config.embedding_model,
                distance_metric=config.distance_metric,
                extra_config=config.extra_config,  # Preserve extra_config
            )

        provider_class = cls.get(config.vector_store)
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
    global PROXIMADB_AVAILABLE

    # ChromaDB (optional)
    try:
        from victor.coding.codebase.embeddings.chromadb_provider import ChromaDBProvider

        EmbeddingRegistry.register("chromadb", ChromaDBProvider)
        logger.debug("Registered ChromaDB provider")
    except ImportError:
        logger.debug("ChromaDB not installed - skipping registration")

    # LanceDB (default, always installed)
    try:
        from victor.coding.codebase.embeddings.lancedb_provider import LanceDBProvider

        EmbeddingRegistry.register("lancedb", LanceDBProvider)
        logger.info("Registered LanceDB provider (default)")
    except ImportError:
        logger.error(
            "LanceDB is required but not installed. "
            "Please reinstall victor-ai to ensure all dependencies are available."
        )

    # ProximaDB (experimental, optional)
    # Note: ProximaDB is an experimental vector store under active development.
    # It is not required for core functionality. LanceDB is the default.
    try:
        # First check if the actual proximaDB package is importable
        import proximadb  # type: ignore[import-untyped]

        from victor.coding.codebase.embeddings.proximadb_provider import ProximaDBProvider

        EmbeddingRegistry.register("proximadb", ProximaDBProvider)
        PROXIMADB_AVAILABLE = True
        logger.info("Registered ProximaDB provider (experimental)")
    except ImportError:
        logger.debug(
            "ProximaDB not installed (optional experimental feature). "
            "Install with: pip install victor-ai[vector-experimental]. "
            "Using LanceDB as default vector store."
        )
        PROXIMADB_AVAILABLE = False


# Auto-register on module import
_auto_register_providers()
