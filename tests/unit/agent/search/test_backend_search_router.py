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

"""Tests for BackendSearchRouter implementation.

Tests protocol compliance, backend registration, search routing,
fallback behavior, and result aggregation.
"""

import pytest
from typing import List, Dict, Any

from victor.core.errors import SearchError
from victor.protocols.search_router import (
    ISearchRouter,
    ISearchBackend,
    SearchType,
    SearchContext,
    SearchResult,
)


class MockSearchBackend(ISearchBackend):
    """Mock search backend for testing."""

    def __init__(
        self,
        name: str,
        supported_types: List[SearchType],
        available: bool = True,
        results: List[Dict[str, Any]] | None = None,
        latency_ms: float = 50.0,
    ):
        self._backend_name = name
        self._supported_types = supported_types
        self._available = available
        self._results = results or []
        self._latency_ms = latency_ms
        self.search_count = 0

    def __class_getitem__(cls, item):
        return cls

    @property
    def __name__(self):
        return self._backend_name

    async def search(
        self,
        query: str,
        search_type: SearchType,
        context: SearchContext,
    ) -> List[Dict[str, Any]]:
        """Perform mock search."""
        self.search_count += 1
        return self._results

    def supported_search_types(self) -> List[SearchType]:
        """Get supported search types."""
        return self._supported_types

    def is_available(self) -> bool:
        """Check availability."""
        return self._available

    def set_available(self, available: bool) -> None:
        """Set availability for testing."""
        self._available = available


class TestBackendSearchRouter:
    """Test suite for BackendSearchRouter."""

    @pytest.mark.asyncio
    async def test_protocol_compliance(self):
        """Test that BackendSearchRouter implements ISearchRouter protocol."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()

        # Verify protocol compliance
        assert isinstance(router, ISearchRouter)

        # Verify required methods exist
        assert hasattr(router, "route_search")
        assert hasattr(router, "register_backend")

    @pytest.mark.asyncio
    async def test_register_single_backend(self):
        """Test registering a single search backend."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()
        backend = MockSearchBackend(
            name="semantic_backend",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "result1"}],
        )

        router.register_backend(backend, priority=100)

        # Verify backend is registered
        assert SearchType.SEMANTIC in router._backends
        assert len(router._backends[SearchType.SEMANTIC]) == 1
        assert router._backends[SearchType.SEMANTIC][0] == (100, backend)

    @pytest.mark.asyncio
    async def test_register_multiple_backends_same_type(self):
        """Test registering multiple backends for same search type."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()
        backend1 = MockSearchBackend(
            name="backend1",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "result1"}],
        )
        backend2 = MockSearchBackend(
            name="backend2",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "result2"}],
        )

        router.register_backend(backend1, priority=50)
        router.register_backend(backend2, priority=100)

        # Verify both backends registered
        assert len(router._backends[SearchType.SEMANTIC]) == 2

        # Verify priority ordering (highest first)
        assert router._backends[SearchType.SEMANTIC][0] == (100, backend2)
        assert router._backends[SearchType.SEMANTIC][1] == (50, backend1)

    @pytest.mark.asyncio
    async def test_register_backend_multiple_types(self):
        """Test registering backend supporting multiple search types."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()
        backend = MockSearchBackend(
            name="hybrid_backend",
            supported_types=[SearchType.SEMANTIC, SearchType.KEYWORD],
            results=[{"content": "result"}],
        )

        router.register_backend(backend, priority=100)

        # Verify registered for both types
        assert SearchType.SEMANTIC in router._backends
        assert SearchType.KEYWORD in router._backends
        assert len(router._backends[SearchType.SEMANTIC]) == 1
        assert len(router._backends[SearchType.KEYWORD]) == 1

    @pytest.mark.asyncio
    async def test_route_search_to_highest_priority_backend(self):
        """Test that search routes to highest priority available backend."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()
        backend1 = MockSearchBackend(
            name="low_priority",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "low"}],
        )
        backend2 = MockSearchBackend(
            name="high_priority",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "high"}],
        )

        router.register_backend(backend1, priority=50)
        router.register_backend(backend2, priority=100)

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
        )
        result = await router.route_search(
            query="test query",
            search_type=SearchType.SEMANTIC,
            context=context,
        )

        # Verify high priority backend was used
        assert result.backend_used == "high_priority"
        assert result.results == [{"content": "high"}]
        assert backend2.search_count == 1
        assert backend1.search_count == 0

    @pytest.mark.asyncio
    async def test_route_search_fallback_on_unavailable(self):
        """Test that search falls back to next available backend."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()
        backend1 = MockSearchBackend(
            name="primary",
            supported_types=[SearchType.SEMANTIC],
            available=False,
            results=[{"content": "primary"}],
        )
        backend2 = MockSearchBackend(
            name="fallback",
            supported_types=[SearchType.SEMANTIC],
            available=True,
            results=[{"content": "fallback"}],
        )

        router.register_backend(backend1, priority=100)
        router.register_backend(backend2, priority=50)

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
        )
        result = await router.route_search(
            query="test query",
            search_type=SearchType.SEMANTIC,
            context=context,
        )

        # Verify fallback backend was used
        assert result.backend_used == "fallback"
        assert result.results == [{"content": "fallback"}]
        assert backend1.search_count == 0
        assert backend2.search_count == 1

    @pytest.mark.asyncio
    async def test_route_search_no_backend_error(self):
        """Test error when no backend available for search type."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
        )

        with pytest.raises(ValueError, match="No backends for search type"):
            await router.route_search(
                query="test query",
                search_type=SearchType.SEMANTIC,
                context=context,
            )

    @pytest.mark.asyncio
    async def test_route_search_all_backends_unavailable(self):
        """Test error when all backends are unavailable."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()
        backend = MockSearchBackend(
            name="unavailable",
            supported_types=[SearchType.SEMANTIC],
            available=False,
            results=[],
        )

        router.register_backend(backend, priority=100)

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
        )

        with pytest.raises(SearchError, match="All.*search backends failed") as exc_info:
            await router.route_search(
                query="test query",
                search_type=SearchType.SEMANTIC,
                context=context,
            )

        # Verify SearchError has proper error details
        error = exc_info.value
        assert error.search_type == "semantic"
        assert "unavailable" in error.failed_backends
        assert error.query == "test query"
        assert error.correlation_id is not None
        assert "Backend not available" in error.failure_details.get("unavailable", "")

    @pytest.mark.asyncio
    async def test_search_result_metadata(self):
        """Test that search result includes proper metadata."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()
        backend = MockSearchBackend(
            name="test_backend",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "result1"}, {"content": "result2"}],
        )

        router.register_backend(backend, priority=100)

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
        )
        result = await router.route_search(
            query="test query",
            search_type=SearchType.SEMANTIC,
            context=context,
        )

        # Verify result structure
        assert isinstance(result, SearchResult)
        assert result.search_type == SearchType.SEMANTIC
        assert result.backend_used == "test_backend"
        assert result.total_results == 2
        assert result.query_time_ms > 0
        assert result.metadata is not None
        assert "priority" in result.metadata

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test search with context filters."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()
        backend = MockSearchBackend(
            name="filtered_backend",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "filtered_result"}],
        )

        router.register_backend(backend, priority=100)

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
            filters={"file_type": "py", "path": "/src"},
        )
        result = await router.route_search(
            query="test query",
            search_type=SearchType.SEMANTIC,
            context=context,
        )

        # Verify search executed with context
        assert result.backend_used == "filtered_backend"
        assert backend.search_count == 1

    @pytest.mark.asyncio
    async def test_different_search_types(self):
        """Test routing for different search types."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()
        semantic_backend = MockSearchBackend(
            name="semantic",
            supported_types=[SearchType.SEMANTIC],
            results=[{"type": "semantic"}],
        )
        keyword_backend = MockSearchBackend(
            name="keyword",
            supported_types=[SearchType.KEYWORD],
            results=[{"type": "keyword"}],
        )

        router.register_backend(semantic_backend, priority=100)
        router.register_backend(keyword_backend, priority=100)

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
        )

        # Test semantic search
        result1 = await router.route_search(
            query="conceptual query",
            search_type=SearchType.SEMANTIC,
            context=context,
        )
        assert result1.backend_used == "semantic"
        assert result1.search_type == SearchType.SEMANTIC

        # Test keyword search
        result2 = await router.route_search(
            query="exact query",
            search_type=SearchType.KEYWORD,
            context=context,
        )
        assert result2.backend_used == "keyword"
        assert result2.search_type == SearchType.KEYWORD

    @pytest.mark.asyncio
    async def test_backend_priority_updates(self):
        """Test updating backend priority."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()
        backend1 = MockSearchBackend(
            name="backend1",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "result1"}],
        )
        backend2 = MockSearchBackend(
            name="backend2",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "result2"}],
        )

        # Register with initial priorities
        router.register_backend(backend1, priority=100)
        router.register_backend(backend2, priority=50)

        # Update backend1 priority
        router.register_backend(backend1, priority=25)

        # Verify new priority ordering
        assert len(router._backends[SearchType.SEMANTIC]) == 2
        # backend2 should now be first (50 > 25)
        assert router._backends[SearchType.SEMANTIC][0] == (50, backend2)

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Test handling of empty search results."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()
        backend = MockSearchBackend(
            name="empty_backend",
            supported_types=[SearchType.SEMANTIC],
            results=[],
        )

        router.register_backend(backend, priority=100)

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
        )
        result = await router.route_search(
            query="test query",
            search_type=SearchType.SEMANTIC,
            context=context,
        )

        # Verify empty result handling
        assert result.total_results == 0
        assert result.results == []

    @pytest.mark.asyncio
    async def test_query_timing_measurement(self):
        """Test that query time is measured accurately."""
        from victor.agent.search.backend_search_router import BackendSearchRouter
        import time

        router = BackendSearchRouter()

        # Create backend with simulated delay
        class SlowMockBackend(MockSearchBackend):
            async def search(
                self,
                query: str,
                search_type: SearchType,
                context: SearchContext,
            ) -> List[Dict[str, Any]]:
                await asyncio.sleep(0.1)  # 100ms delay
                return await super().search(query, search_type, context)

        import asyncio

        backend = SlowMockBackend(
            name="slow_backend",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "result"}],
        )

        router.register_backend(backend, priority=100)

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
        )

        start = time.time()
        result = await router.route_search(
            query="test query",
            search_type=SearchType.SEMANTIC,
            context=context,
        )
        elapsed = time.time() - start

        # Verify timing captured
        assert result.query_time_ms >= 100  # At least 100ms
        assert elapsed >= 0.1  # At least 100ms actual time

    @pytest.mark.asyncio
    async def test_context_metadata_propagation(self):
        """Test that context metadata is available to backends."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()

        # Create backend that captures context
        class ContextCapturingBackend(MockSearchBackend):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.last_context = None

            async def search(
                self,
                query: str,
                search_type: SearchType,
                context: SearchContext,
            ) -> List[Dict[str, Any]]:
                self.last_context = context
                return await super().search(query, search_type, context)

        backend = ContextCapturingBackend(
            name="context_backend",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "result"}],
        )

        router.register_backend(backend, priority=100)

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
            file_path="/src/main.py",
            project_root="/project",
            filters={"lang": "python"},
            metadata={"user_id": "123"},
        )

        await router.route_search(
            query="test query",
            search_type=SearchType.SEMANTIC,
            context=context,
        )

        # Verify context was passed to backend
        assert backend.last_context is not None
        assert backend.last_context.session_id == "test_session"
        assert backend.last_context.vertical == "coding"
        assert backend.last_context.file_path == "/src/main.py"
        assert backend.last_context.filters == {"lang": "python"}

    @pytest.mark.asyncio
    async def test_multiple_concurrent_searches(self):
        """Test handling multiple concurrent searches."""
        from victor.agent.search.backend_search_router import BackendSearchRouter
        import asyncio

        router = BackendSearchRouter()
        backend = MockSearchBackend(
            name="concurrent_backend",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "result"}],
        )

        router.register_backend(backend, priority=100)

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
        )

        # Execute concurrent searches
        tasks = [
            router.route_search(
                query=f"query{i}",
                search_type=SearchType.SEMANTIC,
                context=context,
            )
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all searches completed
        assert len(results) == 10
        assert all(r.backend_used == "concurrent_backend" for r in results)
        assert backend.search_count == 10

    @pytest.mark.asyncio
    async def test_backend_exception_fallback(self):
        """Test fallback when backend raises exception."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()

        # Create backend that raises exception
        class FailingMockBackend(ISearchBackend):
            def __init__(self):
                self.search_count = 0

            async def search(
                self,
                query: str,
                search_type: SearchType,
                context: SearchContext,
            ) -> List[Dict[str, Any]]:
                self.search_count += 1
                raise RuntimeError("Backend failure")

            def supported_search_types(self) -> List[SearchType]:
                return [SearchType.SEMANTIC]

            def is_available(self) -> bool:
                return True

            @property
            def __name__(self):
                return "FailingMockBackend"

        failing_backend = FailingMockBackend()
        fallback_backend = MockSearchBackend(
            name="fallback",
            supported_types=[SearchType.SEMANTIC],
            results=[{"content": "fallback_result"}],
        )

        router.register_backend(failing_backend, priority=100)
        router.register_backend(fallback_backend, priority=50)

        context = SearchContext(
            session_id="test_session",
            vertical="coding",
        )

        result = await router.route_search(
            query="test query",
            search_type=SearchType.SEMANTIC,
            context=context,
        )

        # Verify fallback was used
        assert result.backend_used == "fallback"
        assert failing_backend.search_count == 1
        assert fallback_backend.search_count == 1

    def test_get_available_backends(self):
        """Test getting list of available backends."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()

        # Create different backend classes for distinct names
        class MockBackend1(ISearchBackend):
            @property
            def __name__(self):
                return "MockBackend1"

            async def search(self, query, search_type, context):
                return []

            def supported_search_types(self):
                return [SearchType.SEMANTIC]

            def is_available(self):
                return True

        class MockBackend2(ISearchBackend):
            @property
            def __name__(self):
                return "MockBackend2"

            async def search(self, query, search_type, context):
                return []

            def supported_search_types(self):
                return [SearchType.KEYWORD]

            def is_available(self):
                return True

        backend1 = MockBackend1()
        backend2 = MockBackend2()

        router.register_backend(backend1, priority=100)
        router.register_backend(backend2, priority=100)

        backends = router.get_available_backends()

        # Verify both backends listed
        assert len(backends) == 2
        assert "MockBackend1" in backends
        assert "MockBackend2" in backends

    def test_get_backends_for_type(self):
        """Test getting backends for specific search type."""
        from victor.agent.search.backend_search_router import BackendSearchRouter

        router = BackendSearchRouter()

        # Create different backend classes
        class MockBackend1(ISearchBackend):
            @property
            def __name__(self):
                return "MockBackend1"

            async def search(self, query, search_type, context):
                return []

            def supported_search_types(self):
                return [SearchType.SEMANTIC]

            def is_available(self):
                return True

        class MockBackend2(ISearchBackend):
            @property
            def __name__(self):
                return "MockBackend2"

            async def search(self, query, search_type, context):
                return []

            def supported_search_types(self):
                return [SearchType.SEMANTIC, SearchType.KEYWORD]

            def is_available(self):
                return True

        backend1 = MockBackend1()
        backend2 = MockBackend2()

        router.register_backend(backend1, priority=50)
        router.register_backend(backend2, priority=100)

        semantic_backends = router.get_backends_for_type(SearchType.SEMANTIC)

        # Verify both backends for semantic (priority ordered)
        assert len(semantic_backends) == 2
        assert semantic_backends[0] == "MockBackend2"  # Higher priority
        assert semantic_backends[1] == "MockBackend1"

        keyword_backends = router.get_backends_for_type(SearchType.KEYWORD)

        # Verify only backend2 for keyword
        assert len(keyword_backends) == 1
        assert keyword_backends[0] == "MockBackend2"

        # Verify empty list for unregistered type
        code_backends = router.get_backends_for_type(SearchType.CODE)
        assert code_backends == []
