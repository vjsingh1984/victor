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

"""Search router protocol for dependency inversion.

This module defines the ISearchRouter protocol that enables
dependency injection for search operations, following the
Dependency Inversion Principle (DIP).

Design Principles:
    - DIP: High-level modules depend on this protocol, not concrete routers
    - OCP: New search backends can be added without modifying existing code
    - Strategy Pattern: Different routing strategies for different use cases

Usage:
    class SemanticSearchRouter(ISearchRouter):
        async def route_search(self, query: str, search_type: SearchType, context: SearchContext) -> SearchResult:
            # Route to semantic search backend
            ...

        def register_backend(self, backend: ISearchBackend, priority: int) -> None:
            # Register search backend with priority
            ...
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class SearchType(Enum):
    """Type of search operation."""

    SEMANTIC = "semantic"  # Semantic similarity search
    KEYWORD = "keyword"  # Keyword/exact match search
    HYBRID = "hybrid"  # Combination of semantic and keyword
    CODE = "code"  # Code-specific search
    SYMBOL = "symbol"  # Symbol/definition search


@dataclass
class SearchContext:
    """Context for search operations.

    Attributes:
        session_id: Session identifier
        vertical: Current vertical (coding, research, etc.)
        file_path: Current file path (if applicable)
        project_root: Project root directory
        filters: Additional search filters
        metadata: Additional context metadata
    """

    session_id: str
    vertical: str
    file_path: Optional[str] = None
    project_root: Optional[str] = None
    filters: Dict[str, Any] | None = None
    metadata: Dict[str, Any] | None = None


@dataclass
class SearchResult:
    """Result from search operation.

    Attributes:
        results: List of search results
        search_type: Type of search performed
        backend_used: Which search backend handled the request
        total_results: Total number of results
        query_time_ms: Time taken for query in milliseconds
        metadata: Additional search metadata
    """

    results: List[Dict[str, Any]]
    search_type: SearchType
    backend_used: str
    total_results: int
    query_time_ms: float
    metadata: Dict[str, Any] | None = None


@runtime_checkable
class ISearchRouter(Protocol):
    """Protocol for search operation routing.

    Implementations route search queries to appropriate backends
    based on search type, query characteristics, and availability.

    Backends can be registered with priorities, and the router
    selects the best backend for each query.
    """

    async def route_search(
        self,
        query: str,
        search_type: SearchType,
        context: SearchContext,
    ) -> SearchResult:
        """Route search query to appropriate backend.

        Args:
            query: Search query string
            search_type: Type of search to perform
            context: Search context with session info and filters

        Returns:
            SearchResult with results and metadata

        Example:
            result = await router.route_search(
                query="authentication logic",
                search_type=SearchType.SEMANTIC,
                context=SearchContext(session_id="abc", vertical="coding")
            )
        """
        ...

    def register_backend(
        self,
        backend: "ISearchBackend",
        priority: int,
    ) -> None:
        """Register a search backend with priority.

        Backends with higher priority are preferred when multiple
        backends can handle a search type.

        Args:
            backend: Search backend to register
            priority: Priority for this backend (higher = preferred)

        Example:
            router.register_backend(semantic_backend, priority=100)
            router.register_backend(keyword_backend, priority=50)
        """
        ...


@runtime_checkable
class ISearchBackend(Protocol):
    """Protocol for search backends.

    Implementations provide search functionality for different
    search types and data sources.
    """

    async def search(
        self,
        query: str,
        search_type: SearchType,
        context: SearchContext,
    ) -> List[Dict[str, Any]]:
        """Perform search operation.

        Args:
            query: Search query
            search_type: Type of search
            context: Search context

        Returns:
            List of search results

        Example:
            results = await backend.search(
                query="authentication",
                search_type=SearchType.SEMANTIC,
                context=context
            )
        """
        ...

    def supported_search_types(self) -> List[SearchType]:
        """Get list of supported search types.

        Returns:
            List of SearchType values this backend supports

        Example:
            def supported_search_types(self) -> List[SearchType]:
                return [SearchType.SEMANTIC, SearchType.CODE]
        """
        ...

    def is_available(self) -> bool:
        """Check if backend is available.

        Returns:
            True if backend can accept requests, False otherwise

        Example:
            def is_available(self) -> bool:
                return self._connected and self._healthy
        """
        ...


__all__ = [
    "SearchType",
    "SearchContext",
    "SearchResult",
    "ISearchRouter",
    "ISearchBackend",
]
