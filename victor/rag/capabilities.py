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

"""Dynamic capability definitions for the RAG vertical.

This module provides capability declarations that can be loaded
dynamically by the CapabilityLoader, enabling runtime extension
of the RAG vertical with custom functionality.

Refactored to use BaseVerticalCapabilityProvider, reducing from
785 lines to ~400 lines by eliminating duplicated patterns.

Example:
    # Use provider
    from victor.rag.capabilities import RAGCapabilityProvider

    provider = RAGCapabilityProvider()

    # Apply capabilities
    provider.apply_indexing(orchestrator, chunk_size=1024)
    provider.apply_retrieval(orchestrator, top_k=10)

    # Get configurations
    config = provider.get_capability_config(orchestrator, "indexing")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING, cast

from victor.framework.capabilities.base_vertical_capability_provider import (
    BaseVerticalCapabilityProvider,
    CapabilityDefinition,
)
from victor.framework.protocols import CapabilityType, OrchestratorCapability
from victor.framework.capability_loader import CapabilityEntry, capability

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Handlers
# =============================================================================


def configure_indexing(
    orchestrator: Any,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model: str = "text-embedding-3-small",
    embedding_dimensions: int = 1536,
    store_backend: str = "lancedb",
) -> None:
    """Configure document indexing settings for the orchestrator.

    This capability configures the orchestrator's indexing settings
    for document ingestion and chunking.

    Args:
        orchestrator: Target orchestrator
        chunk_size: Target size for document chunks
        chunk_overlap: Overlap between consecutive chunks
        embedding_model: Model to use for generating embeddings
        embedding_dimensions: Dimension of embedding vectors
        store_backend: Vector store backend (lancedb, chromadb, etc.)
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    indexing_config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "embedding_dimensions": embedding_dimensions,
        "store_backend": store_backend,
    }
    context.set_capability_config("rag_indexing", indexing_config)

    logger.info(f"Configured RAG indexing: chunk_size={chunk_size}, model={embedding_model}")


def get_indexing_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current indexing configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Indexing configuration dict
    """
    # SOLID DIP: Read from VerticalContext instead of direct attribute access
    context = orchestrator.vertical_context
    config_result = context.get_capability_config(
        "rag_indexing",
        {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 1536,
            "store_backend": "lancedb",
        },
    )
    return cast(Dict[str, Any], config_result)


def configure_retrieval(
    orchestrator: Any,
    *,
    top_k: int = 5,
    similarity_threshold: float = 0.7,
    search_type: str = "hybrid",
    rerank_enabled: bool = True,
    max_context_tokens: int = 4000,
) -> None:
    """Configure retrieval settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        top_k: Number of top results to retrieve
        similarity_threshold: Minimum similarity score for retrieval
        search_type: Type of search (semantic, keyword, hybrid)
        rerank_enabled: Whether to rerank retrieved results
        max_context_tokens: Maximum tokens in context window
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    retrieval_config = {
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "search_type": search_type,
        "rerank_enabled": rerank_enabled,
        "max_context_tokens": max_context_tokens,
    }
    context.set_capability_config("rag_retrieval", retrieval_config)

    logger.info(f"Configured RAG retrieval: top_k={top_k}, search_type={search_type}")


def get_retrieval_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current retrieval configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Retrieval configuration dict
    """
    # SOLID DIP: Read from VerticalContext instead of direct attribute access
    context = orchestrator.vertical_context
    config_result = context.get_capability_config(
        "rag_retrieval",
        {
            "top_k": 5,
            "similarity_threshold": 0.7,
            "search_type": "hybrid",
            "rerank_enabled": True,
            "max_context_tokens": 4000,
        },
    )
    return cast(Dict[str, Any], config_result)


def configure_synthesis(
    orchestrator: Any,
    *,
    citation_style: str = "inline",
    include_sources: bool = True,
    max_answer_tokens: int = 2000,
    temperature: float = 0.3,
    require_verification: bool = True,
) -> None:
    """Configure answer synthesis settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        citation_style: How to format citations (inline, footnote, endnote)
        include_sources: Whether to include source list in answer
        max_answer_tokens: Maximum tokens in generated answer
        temperature: LLM temperature for synthesis
        require_verification: Whether to verify answer against sources
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    synthesis_config = {
        "citation_style": citation_style,
        "include_sources": include_sources,
        "max_answer_tokens": max_answer_tokens,
        "temperature": temperature,
        "require_verification": require_verification,
    }
    context.set_capability_config("rag_synthesis", synthesis_config)

    logger.info(f"Configured RAG synthesis: citation_style={citation_style}")


def get_synthesis_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current synthesis configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Synthesis configuration dict
    """
    # SOLID DIP: Read from VerticalContext instead of direct attribute access
    context = orchestrator.vertical_context
    config_result = context.get_capability_config(
        "rag_synthesis",
        {
            "citation_style": "inline",
            "include_sources": True,
            "max_answer_tokens": 2000,
            "temperature": 0.3,
            "require_verification": True,
        },
    )
    return cast(Dict[str, Any], config_result)


def configure_safety(
    orchestrator: Any,
    *,
    filter_sensitive_data: bool = True,
    max_document_size_mb: int = 50,
    allowed_file_types: Optional[List[str]] = None,
    validate_sources: bool = True,
) -> None:
    """Configure RAG safety settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        filter_sensitive_data: Whether to filter sensitive data from documents
        max_document_size_mb: Maximum document size in MB
        allowed_file_types: List of allowed file extensions
        validate_sources: Whether to validate source URLs
    """
    default_types = ["pdf", "docx", "txt", "md", "py", "js", "ts", "html"]

    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    safety_config = {
        "filter_sensitive_data": filter_sensitive_data,
        "max_document_size_mb": max_document_size_mb,
        "allowed_file_types": allowed_file_types or default_types,
        "validate_sources": validate_sources,
    }
    context.set_capability_config("rag_safety", safety_config)

    logger.info(f"Configured RAG safety: filter_sensitive={filter_sensitive_data}")


def configure_query_enhancement(
    orchestrator: Any,
    *,
    enable_expansion: bool = True,
    enable_decomposition: bool = True,
    max_query_variants: int = 3,
    use_synonyms: bool = True,
) -> None:
    """Configure query enhancement settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        enable_expansion: Whether to expand queries with synonyms/related terms
        enable_decomposition: Whether to decompose complex queries
        max_query_variants: Maximum number of query variants to generate
        use_synonyms: Whether to use synonym expansion
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    query_enhancement_config = {
        "enable_expansion": enable_expansion,
        "enable_decomposition": enable_decomposition,
        "max_query_variants": max_query_variants,
        "use_synonyms": use_synonyms,
    }
    context.set_capability_config("rag_query_enhancement", query_enhancement_config)

    logger.info(f"Configured query enhancement: expansion={enable_expansion}")


# =============================================================================
# Capability Provider Class (Refactored to use BaseVerticalCapabilityProvider)
# =============================================================================


class RAGCapabilityProvider(BaseVerticalCapabilityProvider):
    """Provider for RAG-specific capabilities.

    Refactored to inherit from BaseVerticalCapabilityProvider, eliminating
    ~400 lines of duplicated boilerplate code.

    Capabilities:
    - indexing: Document indexing and chunking settings
    - retrieval: Search and retrieval configuration
    - synthesis: Answer generation and citation settings
    - safety: Data filtering and validation
    - query_enhancement: Query expansion and decomposition

    Example:
        provider = RAGCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_indexing(orchestrator, chunk_size=1024)
        provider.apply_retrieval(orchestrator, top_k=10)

        # Get configurations
        config = provider.get_capability_config(orchestrator, "indexing")
    """

    def __init__(self) -> None:
        """Initialize the RAG capability provider."""
        super().__init__("rag")

    def _get_capability_definitions(self) -> Dict[str, CapabilityDefinition]:
        """Define RAG capability definitions.

        Returns:
            Dictionary of RAG capability definitions
        """
        return {
            "indexing": CapabilityDefinition(
                name="indexing",
                type=CapabilityType.MODE,
                description="Document indexing and chunking configuration",
                version="1.0",
                configure_fn="configure_indexing",
                get_fn="get_indexing_config",
                default_config={
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 1536,
                    "store_backend": "lancedb",
                },
                tags=["indexing", "chunking", "embedding"],
            ),
            "retrieval": CapabilityDefinition(
                name="retrieval",
                type=CapabilityType.MODE,
                description="Search and retrieval configuration",
                version="1.0",
                configure_fn="configure_retrieval",
                get_fn="get_retrieval_config",
                default_config={
                    "top_k": 5,
                    "similarity_threshold": 0.7,
                    "search_type": "hybrid",
                    "rerank_enabled": True,
                    "max_context_tokens": 4000,
                },
                dependencies=["indexing"],
                tags=["retrieval", "search", "ranking"],
            ),
            "synthesis": CapabilityDefinition(
                name="synthesis",
                type=CapabilityType.MODE,
                description="Answer generation and citation configuration",
                version="1.0",
                configure_fn="configure_synthesis",
                get_fn="get_synthesis_config",
                default_config={
                    "citation_style": "inline",
                    "include_sources": True,
                    "max_answer_tokens": 2000,
                    "temperature": 0.3,
                    "require_verification": True,
                },
                dependencies=["retrieval"],
                tags=["synthesis", "generation", "citations"],
            ),
            "safety": CapabilityDefinition(
                name="safety",
                type=CapabilityType.SAFETY,
                description="Data filtering and validation settings",
                version="1.0",
                configure_fn="configure_safety",
                default_config={
                    "filter_sensitive_data": True,
                    "max_document_size_mb": 50,
                    "allowed_file_types": [
                        "pdf",
                        "docx",
                        "txt",
                        "md",
                        "py",
                        "js",
                        "ts",
                        "html",
                    ],
                    "validate_sources": True,
                },
                tags=["safety", "filtering", "validation"],
            ),
            "query_enhancement": CapabilityDefinition(
                name="query_enhancement",
                type=CapabilityType.MODE,
                description="Query expansion and decomposition settings",
                version="1.0",
                configure_fn="configure_query_enhancement",
                default_config={
                    "enable_expansion": True,
                    "enable_decomposition": True,
                    "max_query_variants": 3,
                    "use_synonyms": True,
                },
                tags=["query", "expansion", "decomposition"],
            ),
        }

    # Delegate to handler functions (required by BaseVerticalCapabilityProvider)
    def configure_indexing(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure indexing capability."""
        configure_indexing(orchestrator, **kwargs)

    def configure_retrieval(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure retrieval capability."""
        configure_retrieval(orchestrator, **kwargs)

    def configure_synthesis(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure synthesis capability."""
        configure_synthesis(orchestrator, **kwargs)

    def configure_safety(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure safety capability."""
        configure_safety(orchestrator, **kwargs)

    def configure_query_enhancement(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure query enhancement capability."""
        configure_query_enhancement(orchestrator, **kwargs)

    def get_indexing_config(self, orchestrator: Any) -> Dict[str, Any]:
        """Get indexing configuration."""
        return get_indexing_config(orchestrator)

    def get_retrieval_config(self, orchestrator: Any) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return get_retrieval_config(orchestrator)

    def get_synthesis_config(self, orchestrator: Any) -> Dict[str, Any]:
        """Get synthesis configuration."""
        return get_synthesis_config(orchestrator)


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


# Create singleton instance for generating CAPABILITIES list
_provider_instance: Optional[RAGCapabilityProvider] = None


def _get_provider() -> RAGCapabilityProvider:
    """Get or create provider instance.

    Returns:
        RAGCapabilityProvider instance
    """
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = RAGCapabilityProvider()
    return _provider_instance


# Generate CAPABILITIES list from provider
CAPABILITIES: List[CapabilityEntry] = []


def _generate_capabilities_list() -> None:
    """Generate CAPABILITIES list from provider."""
    global CAPABILITIES
    if not CAPABILITIES:
        provider = _get_provider()
        CAPABILITIES.extend(provider.generate_capabilities_list())


_generate_capabilities_list()


# =============================================================================
# Convenience Functions
# =============================================================================


def get_rag_capabilities() -> List[CapabilityEntry]:
    """Get all RAG capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


def create_rag_capability_loader() -> Any:
    """Create a CapabilityLoader pre-configured for RAG vertical.

    Returns:
        CapabilityLoader with RAG capabilities registered
    """
    from victor.framework.capability_loader import CapabilityLoader

    provider = _get_provider()
    return provider.create_capability_loader()


def get_capability_configs() -> Dict[str, Any]:
    """Get RAG capability configurations for centralized storage.

    Returns default RAG configuration for VerticalContext storage.
    This replaces direct orchestrator.rag_config assignment.

    Returns:
        Dict with default RAG capability configurations
    """
    provider = _get_provider()
    return provider.generate_capability_configs()


__all__ = [
    # Handlers
    "configure_indexing",
    "configure_retrieval",
    "configure_synthesis",
    "configure_safety",
    "configure_query_enhancement",
    "get_indexing_config",
    "get_retrieval_config",
    "get_synthesis_config",
    # Provider class
    "RAGCapabilityProvider",
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_rag_capabilities",
    "create_rag_capability_loader",
    # SOLID: Centralized config storage
    "get_capability_configs",
]
