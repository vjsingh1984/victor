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

"""Tests for database optimization module."""

import pytest

from victor.optimization.runtime.database import (
    DatabaseOptimizer,
    QueryCache,
    QueryMetrics,
    cached_query,
)


class TestQueryCache:
    """Test QueryCache functionality."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit returns cached value."""
        cache = QueryCache(max_size=10, ttl_seconds=60)

        # Set value
        await cache.set("SELECT 1", (), "result")

        # Get value
        result = await cache.get("SELECT 1", ())

        assert result == "result"

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = QueryCache(max_size=10, ttl_seconds=60)

        result = await cache.get("SELECT 1", ())

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_ttl(self):
        """Test cache TTL expiration."""
        cache = QueryCache(max_size=10, ttl_seconds=0)  # Immediate expiration

        await cache.set("SELECT 1", (), "result")
        result = await cache.get("SELECT 1", ())

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_max_size(self):
        """Test cache respects max size."""
        cache = QueryCache(max_size=2, ttl_seconds=60)

        # Fill cache
        await cache.set("SELECT 1", (), "result1")
        await cache.set("SELECT 2", (), "result2")
        await cache.set("SELECT 3", (), "result3")  # Evicts SELECT 1

        # First entry should be evicted
        result1 = await cache.get("SELECT 1", ())
        result2 = await cache.get("SELECT 2", ())
        result3 = await cache.get("SELECT 3", ())

        assert result1 is None
        assert result2 == "result2"
        assert result3 == "result3"

    @pytest.mark.asyncio
    async def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        cache = QueryCache(max_size=10, ttl_seconds=60)

        await cache.set("SELECT 1", (), "result")

        await cache.get("SELECT 1", ())  # Hit
        await cache.get("SELECT 2", ())  # Miss

        assert cache.hit_rate == 0.5

    @pytest.mark.asyncio
    async def test_cache_invalidate(self):
        """Test cache invalidation."""
        cache = QueryCache(max_size=10, ttl_seconds=60)

        await cache.set("SELECT 1", (), "result")
        cache.invalidate("SELECT 1")

        result = await cache.get("SELECT 1", ())

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_invalidate_all(self):
        """Test cache invalidates all entries."""
        cache = QueryCache(max_size=10, ttl_seconds=60)

        await cache.set("SELECT 1", (), "result1")
        await cache.set("SELECT 2", (), "result2")
        cache.invalidate()

        assert await cache.get("SELECT 1", ()) is None
        assert await cache.get("SELECT 2", ()) is None


class TestQueryMetrics:
    """Test QueryMetrics functionality."""

    def test_metrics_update(self):
        """Test metrics update with new execution."""
        metrics = QueryMetrics(query_hash="test")

        metrics.update(100.0)
        metrics.update(200.0)
        metrics.update(300.0)

        assert metrics.execution_count == 3
        assert metrics.total_time_ms == 600.0
        assert metrics.avg_time_ms == 200.0
        assert metrics.min_time_ms == 100.0
        assert metrics.max_time_ms == 300.0


class TestDatabaseOptimizer:
    """Test DatabaseOptimizer functionality."""

    @pytest.mark.asyncio
    async def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = DatabaseOptimizer(
            cache_size=100,
            cache_ttl=60,
        )

        assert optimizer is not None
        assert optimizer._query_cache is not None

    @pytest.mark.asyncio
    async def test_execute_query_with_cache(self):
        """Test query execution with caching."""
        optimizer = DatabaseOptimizer()
        await optimizer.initialize()

        # First call - cache miss
        result1 = await optimizer.execute_query(
            "SELECT 1",
            (),
            use_cache=True,
        )

        # Second call - cache hit
        result2 = await optimizer.execute_query(
            "SELECT 1",
            (),
            use_cache=True,
        )

        assert result1 == result2

    @pytest.mark.asyncio
    async def test_execute_query_no_cache(self):
        """Test query execution without caching."""
        optimizer = DatabaseOptimizer()
        await optimizer.initialize()

        result = await optimizer.execute_query(
            "SELECT 1",
            (),
            use_cache=False,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_batch(self):
        """Test batch query execution."""
        optimizer = DatabaseOptimizer()
        await optimizer.initialize()

        results = await optimizer.execute_batch(
            "SELECT ?",
            [(1,), (2,), (3,)],
        )

        assert len(results) == 3

    def test_get_query_metrics(self):
        """Test getting query metrics."""
        optimizer = DatabaseOptimizer()

        metrics = optimizer.get_query_metrics()

        assert isinstance(metrics, dict)

    def test_get_slow_queries(self):
        """Test getting slow queries."""
        optimizer = DatabaseOptimizer()

        slow_queries = optimizer.get_slow_queries(threshold_ms=100.0)

        assert isinstance(slow_queries, list)

    def test_invalidate_cache(self):
        """Test cache invalidation."""
        optimizer = DatabaseOptimizer()

        # Should not raise
        optimizer.invalidate_cache()
        optimizer.invalidate_cache("SELECT 1")

    def test_reset_metrics(self):
        """Test resetting metrics."""
        optimizer = DatabaseOptimizer()

        # Should not raise
        optimizer.reset_metrics()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test optimizer cleanup."""
        optimizer = DatabaseOptimizer()

        # Should not raise
        await optimizer.close()


class TestCachedQueryDecorator:
    """Test cached_query decorator."""

    @pytest.mark.asyncio
    async def test_decorator_caches_results(self):
        """Test decorator caches function results."""
        call_count = 0

        @cached_query(cache_ttl=60)
        async def get_user(user_id: int) -> dict:
            nonlocal call_count
            call_count += 1
            return {"id": user_id, "name": f"User{user_id}"}

        # First call
        result1 = await get_user(1)
        assert call_count == 1

        # Second call (cached)
        result2 = await get_user(1)
        assert call_count == 1

        # Different user
        result3 = await get_user(2)
        assert call_count == 2

        assert result1 == result2

    @pytest.mark.asyncio
    async def test_decorator_ttl_expiration(self):
        """Test decorator respects TTL."""

        @cached_query(cache_ttl=0)  # Immediate expiration
        async def get_value() -> int:
            return 42

        result1 = await get_value()
        result2 = await get_value()

        # Both should execute due to TTL=0
        assert result1 == 42
        assert result2 == 42
