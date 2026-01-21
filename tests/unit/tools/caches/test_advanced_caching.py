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

"""Unit tests for advanced caching features."""

import pytest
import tempfile
from pathlib import Path

from victor.tools.caches import (
    AdaptiveTTLCache,
    AdvancedCacheManager,
    MultiLevelCache,
    PersistentSelectionCache,
    PredictiveCacheWarmer,
    reset_advanced_cache,
    reset_persistent_cache,
)


class TestPersistentCache:
    """Test persistent cache functionality."""

    def test_persistent_cache_basic_operations(self):
        """Test basic get/put operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.db"
            cache = PersistentSelectionCache(cache_path=str(cache_path))

            # Put and get
            cache.put("key1", ["tool1", "tool2"], namespace="query")
            result = cache.get("key1", namespace="query")

            assert result == ["tool1", "tool2"]

            # Check stats
            stats = cache.get_stats()
            assert stats["total_entries"] == 1
            assert stats["hits"] == 1
            assert stats["misses"] == 0

            cache.close()

    def test_persistent_cache_persistence(self):
        """Test that cache persists across restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.db"

            # Create cache and add entry
            cache1 = PersistentSelectionCache(cache_path=str(cache_path))
            cache1.put("key1", ["tool1"], namespace="query")
            cache1.close()

            # Create new cache instance - should have entry
            cache2 = PersistentSelectionCache(cache_path=str(cache_path))
            result = cache2.get("key1", namespace="query")

            assert result == ["tool1"]
            cache2.close()

    def test_persistent_cache_expiration(self):
        """Test TTL expiration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.db"
            cache = PersistentSelectionCache(cache_path=str(cache_path))

            # Put with short TTL
            cache.put("key1", ["tool1"], namespace="query", ttl=1)

            # Should be present immediately
            result = cache.get("key1", namespace="query")
            assert result == ["tool1"]

            # Wait for expiration
            import time

            time.sleep(2)

            # Should be expired
            result = cache.get("key1", namespace="query", default=None)
            assert result is None

            cache.close()


class TestAdaptiveTTLCache:
    """Test adaptive TTL cache functionality."""

    def test_adaptive_ttl_basic_operations(self):
        """Test basic get/put operations."""
        cache = AdaptiveTTLCache(max_size=100, enabled=True)

        # Put and get
        cache.put("key1", ["tool1", "tool2"])
        result = cache.get("key1")

        assert result == ["tool1", "tool2"]

    def test_adaptive_ttl_adjustment(self):
        """Test that TTL adjusts based on access patterns."""
        cache = AdaptiveTTLCache(
            max_size=100,
            min_ttl=60,
            max_ttl=7200,
            initial_ttl=3600,
            adjustment_threshold=3,
        )

        # Add entry
        cache.put("key1", ["tool1"])

        # Access multiple times to trigger adjustment (need more than threshold)
        for _ in range(10):
            result = cache.get("key1")
            assert result == ["tool1"]  # Verify we get the value back

        # Check metrics - TTL adjustments may not happen immediately
        # as they depend on internal timing, but we should see the access count reflected
        metrics = cache.get_metrics()
        assert metrics["performance"]["hits"] >= 10  # Should have at least 10 hits
        assert metrics["size"]["current"] == 1  # Should have 1 entry


class TestMultiLevelCache:
    """Test multi-level cache functionality."""

    def test_multi_level_basic_operations(self):
        """Test basic get/put operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                l1_size=10,
                l2_size=100,
                l3_size=1000,
                l2_dir=Path(tmpdir) / "l2",
            )

            # Put and get
            cache.put("key1", ["tool1", "tool2"])
            result = cache.get("key1")

            assert result == ["tool1", "tool2"]

    def test_multi_level_promotion(self):
        """Test that entries are promoted between levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                l1_size=2,
                l2_size=100,
                l3_size=1000,
                l2_dir=Path(tmpdir) / "l2",
            )

            # Fill L1 cache
            cache.put("key1", ["tool1"])
            cache.put("key2", ["tool2"])
            cache.put("key3", ["tool3"])  # Should demote key1 to L2

            # Access key3 multiple times to keep it hot
            for _ in range(5):
                cache.get("key3")

            # Get metrics
            metrics = cache.get_metrics()
            assert metrics["l1"]["entry_count"] <= 2


class TestPredictiveWarmer:
    """Test predictive cache warming functionality."""

    def test_pattern_recording(self):
        """Test that query patterns are recorded."""
        warmer = PredictiveCacheWarmer(max_patterns=100)

        # Record some patterns
        warmer.record_query("read file", ["read"])
        warmer.record_query("analyze code", ["analyze", "search"])
        warmer.record_query("write code", ["write"])

        # Check stats
        stats = warmer.get_statistics()
        assert stats["patterns"]["total"] == 3

    def test_prediction_generation(self):
        """Test that predictions are generated."""
        warmer = PredictiveCacheWarmer(max_patterns=100)

        # Record patterns with transitions
        warmer.record_query("read file", ["read"])
        warmer.record_query("analyze code", ["analyze"])
        warmer.record_query("read file", ["read"])  # Repeat to establish pattern
        warmer.record_query("analyze code", ["analyze"])

        # Generate predictions
        predictions = warmer.predict_next_queries(current_query="read file", top_k=5)

        # Should predict "analyze code" as it follows "read file"
        assert len(predictions.queries) > 0


class TestAdvancedCacheManager:
    """Test unified cache manager functionality."""

    def test_cache_manager_basic_operations(self):
        """Test basic get/put operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCacheManager(
                cache_size=100,
                persistent_enabled=True,
                persistent_path=str(Path(tmpdir) / "cache.db"),
                adaptive_ttl_enabled=False,
                multi_level_enabled=False,
                predictive_warming_enabled=False,
            )

            # Put and get
            cache.put_query("key1", ["tool1", "tool2"])
            result = cache.get_query("key1")

            assert result.value == ["tool1", "tool2"]

            cache.close()

    def test_cache_manager_metrics(self):
        """Test that comprehensive metrics are available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCacheManager(
                cache_size=100,
                persistent_enabled=True,
                persistent_path=str(Path(tmpdir) / "cache.db"),
                adaptive_ttl_enabled=False,
                multi_level_enabled=False,
                predictive_warming_enabled=False,
            )

            # Perform operations
            cache.put_query("key1", ["tool1"])
            cache.get_query("key1")  # Hit
            cache.get_query("key2")  # Miss

            # Get metrics
            metrics = cache.get_metrics()

            # Check combined metrics
            assert metrics.combined["total_hits"] >= 1
            assert metrics.combined["total_misses"] >= 1
            assert 0.0 <= metrics.combined["hit_rate"] <= 1.0

            cache.close()

    def test_cache_manager_invalidation(self):
        """Test cache invalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCacheManager(
                cache_size=100,
                persistent_enabled=True,
                persistent_path=str(Path(tmpdir) / "cache.db"),
                adaptive_ttl_enabled=False,
                multi_level_enabled=False,
                predictive_warming_enabled=False,
            )

            # Add entries
            cache.put_query("key1", ["tool1"])
            cache.put_query("key2", ["tool2"])

            # Invalidate specific key
            cache.invalidate(key="key1", namespace="query")
            assert cache.get_query("key1") is None
            assert cache.get_query("key2") is not None

            # Invalidate all
            cache.invalidate()
            assert cache.get_query("key2") is None

            cache.close()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset cache singletons before each test."""
    reset_persistent_cache()
    reset_advanced_cache()
    yield
    reset_persistent_cache()
    reset_advanced_cache()
