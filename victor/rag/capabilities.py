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

The module follows the CapabilityLoader's discovery patterns:
1. CAPABILITIES list for batch registration
2. @capability decorator for function-based capabilities
3. Capability classes for complex implementations

Example:
    # Register capabilities with loader
    from victor.framework import CapabilityLoader
    loader = CapabilityLoader()
    loader.load_from_module("victor.rag.capabilities")

    # Or use directly
    from victor.rag.capabilities import (
        get_rag_capabilities,
        RAGCapabilityProvider,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.framework.protocols import CapabilityType, OrchestratorCapability
from victor.framework.capability_loader import CapabilityEntry, capability
from victor.framework.capabilities import BaseCapabilityProvider, CapabilityMetadata

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
    if hasattr(orchestrator, "rag_config"):
        orchestrator.rag_config["indexing"] = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
            "store_backend": store_backend,
        }
    else:
        orchestrator.rag_config = {
            "indexing": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": embedding_model,
                "embedding_dimensions": embedding_dimensions,
                "store_backend": store_backend,
            }
        }

    logger.info(f"Configured RAG indexing: chunk_size={chunk_size}, model={embedding_model}")


def get_indexing_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current indexing configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Indexing configuration dict
    """
    if hasattr(orchestrator, "rag_config"):
        return orchestrator.rag_config.get("indexing", {})
    return {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "store_backend": "lancedb",
    }


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
    if hasattr(orchestrator, "rag_config"):
        orchestrator.rag_config["retrieval"] = {
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "search_type": search_type,
            "rerank_enabled": rerank_enabled,
            "max_context_tokens": max_context_tokens,
        }
    else:
        orchestrator.rag_config = {
            "retrieval": {
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "search_type": search_type,
                "rerank_enabled": rerank_enabled,
                "max_context_tokens": max_context_tokens,
            }
        }

    logger.info(f"Configured RAG retrieval: top_k={top_k}, search_type={search_type}")


def get_retrieval_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current retrieval configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Retrieval configuration dict
    """
    if hasattr(orchestrator, "rag_config"):
        return orchestrator.rag_config.get("retrieval", {})
    return {
        "top_k": 5,
        "similarity_threshold": 0.7,
        "search_type": "hybrid",
        "rerank_enabled": True,
        "max_context_tokens": 4000,
    }


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
    if hasattr(orchestrator, "rag_config"):
        orchestrator.rag_config["synthesis"] = {
            "citation_style": citation_style,
            "include_sources": include_sources,
            "max_answer_tokens": max_answer_tokens,
            "temperature": temperature,
            "require_verification": require_verification,
        }
    else:
        orchestrator.rag_config = {
            "synthesis": {
                "citation_style": citation_style,
                "include_sources": include_sources,
                "max_answer_tokens": max_answer_tokens,
                "temperature": temperature,
                "require_verification": require_verification,
            }
        }

    logger.info(f"Configured RAG synthesis: citation_style={citation_style}")


def get_synthesis_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current synthesis configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Synthesis configuration dict
    """
    if hasattr(orchestrator, "rag_config"):
        return orchestrator.rag_config.get("synthesis", {})
    return {
        "citation_style": "inline",
        "include_sources": True,
        "max_answer_tokens": 2000,
        "temperature": 0.3,
        "require_verification": True,
    }


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

    if hasattr(orchestrator, "rag_config"):
        orchestrator.rag_config["safety"] = {
            "filter_sensitive_data": filter_sensitive_data,
            "max_document_size_mb": max_document_size_mb,
            "allowed_file_types": allowed_file_types or default_types,
            "validate_sources": validate_sources,
        }
    else:
        orchestrator.rag_config = {
            "safety": {
                "filter_sensitive_data": filter_sensitive_data,
                "max_document_size_mb": max_document_size_mb,
                "allowed_file_types": allowed_file_types or default_types,
                "validate_sources": validate_sources,
            }
        }

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
    if hasattr(orchestrator, "rag_config"):
        orchestrator.rag_config["query_enhancement"] = {
            "enable_expansion": enable_expansion,
            "enable_decomposition": enable_decomposition,
            "max_query_variants": max_query_variants,
            "use_synonyms": use_synonyms,
        }
    else:
        orchestrator.rag_config = {
            "query_enhancement": {
                "enable_expansion": enable_expansion,
                "enable_decomposition": enable_decomposition,
                "max_query_variants": max_query_variants,
                "use_synonyms": use_synonyms,
            }
        }

    logger.info(f"Configured query enhancement: expansion={enable_expansion}")


# =============================================================================
# Decorated Capability Functions
# =============================================================================


@capability(
    name="rag_indexing",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="Document indexing and chunking configuration",
)
def rag_indexing(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs: Any,
) -> Callable:
    """Indexing capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_indexing(
            orchestrator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )

    return handler


@capability(
    name="rag_retrieval",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="Retrieval and search configuration",
    getter="get_retrieval_config",
)
def rag_retrieval(
    top_k: int = 5,
    search_type: str = "hybrid",
    **kwargs: Any,
) -> Callable:
    """Retrieval capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_retrieval(
            orchestrator,
            top_k=top_k,
            search_type=search_type,
            **kwargs,
        )

    return handler


@capability(
    name="rag_synthesis",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="Answer synthesis and citation configuration",
)
def rag_synthesis(
    citation_style: str = "inline",
    include_sources: bool = True,
    **kwargs: Any,
) -> Callable:
    """Synthesis capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_synthesis(
            orchestrator,
            citation_style=citation_style,
            include_sources=include_sources,
            **kwargs,
        )

    return handler


@capability(
    name="rag_safety",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="RAG safety and filtering configuration",
)
def rag_safety(
    filter_sensitive_data: bool = True,
    **kwargs: Any,
) -> Callable:
    """Safety capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_safety(
            orchestrator,
            filter_sensitive_data=filter_sensitive_data,
            **kwargs,
        )

    return handler


# =============================================================================
# Capability Provider Class
# =============================================================================


class RAGCapabilityProvider(BaseCapabilityProvider[Callable[..., None]]):
    """Provider for RAG-specific capabilities.

    This class provides a structured way to access and apply
    RAG capabilities to an orchestrator. It inherits from
    BaseCapabilityProvider for consistent capability registration
    and discovery across all verticals.

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

        # Use BaseCapabilityProvider interface
        cap = provider.get_capability("indexing")
        if cap:
            cap(orchestrator, chunk_size=512)
    """

    def __init__(self):
        """Initialize the capability provider."""
        self._applied: Set[str] = set()
        # Map capability names to their handler functions
        self._capabilities: Dict[str, Callable[..., None]] = {
            "indexing": configure_indexing,
            "retrieval": configure_retrieval,
            "synthesis": configure_synthesis,
            "safety": configure_safety,
            "query_enhancement": configure_query_enhancement,
        }
        # Capability metadata for discovery
        self._metadata: Dict[str, CapabilityMetadata] = {
            "indexing": CapabilityMetadata(
                name="indexing",
                description="Document indexing and chunking configuration",
                version="1.0",
                tags=["indexing", "chunking", "embedding"],
            ),
            "retrieval": CapabilityMetadata(
                name="retrieval",
                description="Search and retrieval configuration",
                version="1.0",
                dependencies=["indexing"],
                tags=["retrieval", "search", "ranking"],
            ),
            "synthesis": CapabilityMetadata(
                name="synthesis",
                description="Answer generation and citation configuration",
                version="1.0",
                dependencies=["retrieval"],
                tags=["synthesis", "generation", "citations"],
            ),
            "safety": CapabilityMetadata(
                name="safety",
                description="Data filtering and validation settings",
                version="1.0",
                tags=["safety", "filtering", "validation"],
            ),
            "query_enhancement": CapabilityMetadata(
                name="query_enhancement",
                description="Query expansion and decomposition settings",
                version="1.0",
                tags=["query", "expansion", "decomposition"],
            ),
        }

    def get_capabilities(self) -> Dict[str, Callable[..., None]]:
        """Return all registered capabilities.

        Returns:
            Dictionary mapping capability names to handler functions.
        """
        return self._capabilities.copy()

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        """Return metadata for all registered capabilities.

        Returns:
            Dictionary mapping capability names to their metadata.
        """
        return self._metadata.copy()

    def apply_indexing(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply indexing capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Indexing options
        """
        configure_indexing(orchestrator, **kwargs)
        self._applied.add("indexing")

    def apply_retrieval(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply retrieval capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Retrieval options
        """
        configure_retrieval(orchestrator, **kwargs)
        self._applied.add("retrieval")

    def apply_synthesis(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply synthesis capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Synthesis options
        """
        configure_synthesis(orchestrator, **kwargs)
        self._applied.add("synthesis")

    def apply_safety(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply safety capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Safety options
        """
        configure_safety(orchestrator, **kwargs)
        self._applied.add("safety")

    def apply_query_enhancement(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply query enhancement capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Query enhancement options
        """
        configure_query_enhancement(orchestrator, **kwargs)
        self._applied.add("query_enhancement")

    def apply_all(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply all RAG capabilities with defaults.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Shared options
        """
        self.apply_indexing(orchestrator)
        self.apply_retrieval(orchestrator)
        self.apply_synthesis(orchestrator)
        self.apply_safety(orchestrator)
        self.apply_query_enhancement(orchestrator)

    def get_applied(self) -> Set[str]:
        """Get set of applied capability names.

        Returns:
            Set of applied capability names
        """
        return self._applied.copy()


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


CAPABILITIES: List[CapabilityEntry] = [
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="rag_indexing",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_indexing",
            getter="get_indexing_config",
            description="Document indexing and chunking configuration",
        ),
        handler=configure_indexing,
        getter_handler=get_indexing_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="rag_retrieval",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_retrieval",
            getter="get_retrieval_config",
            description="Search and retrieval configuration",
        ),
        handler=configure_retrieval,
        getter_handler=get_retrieval_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="rag_synthesis",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_synthesis",
            getter="get_synthesis_config",
            description="Answer synthesis and citation configuration",
        ),
        handler=configure_synthesis,
        getter_handler=get_synthesis_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="rag_safety",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_safety",
            description="RAG safety and filtering configuration",
        ),
        handler=configure_safety,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="rag_query_enhancement",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_query_enhancement",
            description="Query expansion and decomposition configuration",
        ),
        handler=configure_query_enhancement,
    ),
]


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
    from victor.framework import CapabilityLoader

    loader = CapabilityLoader()

    # Register all RAG capabilities
    for entry in CAPABILITIES:
        loader._register_capability_internal(
            capability=entry.capability,
            handler=entry.handler,
            getter_handler=entry.getter_handler,
            source_module="victor.rag.capabilities",
        )

    return loader


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
    # Provider class and base types
    "RAGCapabilityProvider",
    "CapabilityMetadata",  # Re-exported from framework for convenience
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_rag_capabilities",
    "create_rag_capability_loader",
]
