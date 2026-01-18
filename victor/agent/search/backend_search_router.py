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

"""Backend search router with protocol-based design.

This module provides BackendSearchRouter, an implementation of the
ISearchRouter protocol that coordinates multiple search backends
with priority-based routing and fallback support.

Design Principles:
    - DIP: Depends on ISearchBackend protocol, not concrete implementations
    - OCP: New backends can be added without modifying router
    - Strategy Pattern: Priority-based routing with fallback
    - LSP: Complies with ISearchRouter protocol

Usage:
    router = BackendSearchRouter()
    router.register_backend(semantic_backend, priority=100)
    router.register_backend(keyword_backend, priority=50)

    result = await router.route_search(
        query="authentication logic",
        search_type=SearchType.SEMANTIC,
        context=SearchContext(session_id="abc", vertical="coding")
    )
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Tuple

from victor.core.errors import SearchError
from victor.protocols.search_router import (
    ISearchRouter,
    ISearchBackend,
    SearchType,
    SearchContext,
    SearchResult,
)


logger = logging.getLogger(__name__)


class BackendSearchRouter(ISearchRouter):
    """Search router coordinating multiple backends with priority routing.

    This router manages multiple search backends, routing queries to the
    highest priority available backend for each search type. If the primary
    backend is unavailable, it automatically falls back to lower priority
    backends.

    Attributes:
        _backends: Dict mapping search types to prioritized backend lists

    Example:
        router = BackendSearchRouter()

        # Register backends with priorities (higher = preferred)
        semantic = SemanticSearchBackend()
        keyword = KeywordSearchBackend()
        router.register_backend(semantic, priority=100)
        router.register_backend(keyword, priority=50)

        # Route search to best available backend
        context = SearchContext(session_id="session1", vertical="coding")
        result = await router.route_search(
            query="authentication logic",
            search_type=SearchType.SEMANTIC,
            context=context
        )
    """

    def __init__(self) -> None:
        """Initialize the backend search router."""
        self._backends: Dict[SearchType, List[Tuple[int, ISearchBackend]]] = {}
        logger.debug("BackendSearchRouter initialized")

    def register_backend(
        self,
        backend: ISearchBackend,
        priority: int,
    ) -> None:
        """Register a search backend with priority.

        Backends can support multiple search types. Higher priority backends
        are preferred when available. If a backend is registered multiple
        times, the highest priority is used.

        Args:
            backend: Search backend implementing ISearchBackend protocol
            priority: Priority level (higher values preferred)

        Example:
            router.register_backend(semantic_backend, priority=100)
            router.register_backend(fallback_backend, priority=50)
        """
        supported_types = backend.supported_search_types()

        for search_type in supported_types:
            if search_type not in self._backends:
                self._backends[search_type] = []

            # Check if backend already registered for this type
            existing_indices = [
                i for i, (p, b) in enumerate(self._backends[search_type]) if b is backend
            ]

            if existing_indices:
                # Update existing backend priority
                idx = existing_indices[0]
                self._backends[search_type][idx] = (priority, backend)
                logger.debug(
                    f"Updated backend {backend.__class__.__name__} "
                    f"priority to {priority} for {search_type.value}"
                )
            else:
                # Add new backend
                self._backends[search_type].append((priority, backend))
                logger.debug(
                    f"Registered backend {backend.__class__.__name__} "
                    f"with priority {priority} for {search_type.value}"
                )

            # Sort by priority (descending)
            self._backends[search_type].sort(key=lambda x: -x[0])

    async def route_search(
        self,
        query: str,
        search_type: SearchType,
        context: SearchContext,
    ) -> SearchResult:
        """Route search query to appropriate backend.

        Routes the query to the highest priority available backend for the
        specified search type. Automatically falls back to lower priority
        backends if the primary is unavailable.

        Args:
            query: Search query string
            search_type: Type of search to perform
            context: Search context with session info and filters

        Returns:
            SearchResult with results and metadata

        Raises:
            ValueError: If no backends registered for search type
            SearchError: If all backends are unavailable

        Example:
            result = await router.route_search(
                query="authentication error handling",
                search_type=SearchType.SEMANTIC,
                context=SearchContext(session_id="abc", vertical="coding")
            )
        """
        start_time = time.time()

        # Get backends for this search type
        backends = self._backends.get(search_type, [])

        if not backends:
            available_types = [st.value for st in self._backends.keys()]
            raise ValueError(
                f"No backends for search type: {search_type.value}. "
                f"Available: {available_types}"
            )

        # Track failed backends for detailed error reporting
        failed_backends: Dict[str, Exception] = {}

        # Try backends in priority order
        for priority, backend in backends:
            backend_name = getattr(
                backend,
                "__name__",
                getattr(backend.__class__, "__name__", type(backend).__name__),
            )

            if not backend.is_available():
                logger.debug(f"Backend {backend_name} unavailable, trying next backend")
                failed_backends[backend_name] = RuntimeError("Backend not available")
                continue

            try:
                # Execute search
                results = await backend.search(query, search_type, context)
                query_time_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"Search routed to {backend_name}: "
                    f"{len(results)} results in {query_time_ms:.2f}ms"
                )

                return SearchResult(
                    results=results,
                    search_type=search_type,
                    backend_used=backend_name,
                    total_results=len(results),
                    query_time_ms=query_time_ms,
                    metadata={"priority": priority},
                )

            except Exception as e:
                logger.warning(f"Backend {backend_name} failed: {e}, trying next backend")
                failed_backends[backend_name] = e
                continue

        # All backends failed - provide detailed error information
        correlation_id = str(uuid.uuid4())[:8]
        errors_str = "; ".join([f"{name}: {str(err)}" for name, err in failed_backends.items()])

        logger.error(
            f"[{correlation_id}] All {len(failed_backends)} search backends failed for "
            f"'{search_type.value}'. Failures: {errors_str}"
        )

        raise SearchError(
            f"All {len(failed_backends)} search backends failed for '{search_type.value}'. "
            f"Failures: {errors_str}",
            search_type=search_type.value,
            failed_backends=list(failed_backends.keys()),
            failure_details={name: str(err) for name, err in failed_backends.items()},
            query=query,
            correlation_id=correlation_id,
        )

    def get_available_backends(self) -> List[str]:
        """Get list of all registered backend names.

        Returns:
            List of backend class names

        Example:
            backends = router.get_available_backends()
            print(backends)  # ['SemanticSearchBackend', 'KeywordSearchBackend']
        """
        backend_names = set()
        for backends in self._backends.values():
            for priority, backend in backends:
                backend_names.add(backend.__class__.__name__)
        return list(backend_names)

    def get_backends_for_type(self, search_type: SearchType) -> List[str]:
        """Get backends registered for a specific search type.

        Args:
            search_type: Type of search

        Returns:
            List of backend names (ordered by priority)

        Example:
            semantic_backends = router.get_backends_for_type(SearchType.SEMANTIC)
        """
        if search_type not in self._backends:
            return []

        return [backend.__class__.__name__ for priority, backend in self._backends[search_type]]


__all__ = [
    "BackendSearchRouter",
]
