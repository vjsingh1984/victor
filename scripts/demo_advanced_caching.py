#!/usr/bin/env python
"""Demonstration script for advanced caching features.

This script demonstrates all advanced caching strategies:
1. Persistent cache (SQLite)
2. Adaptive TTL
3. Multi-level cache
4. Predictive warming
5. Unified cache manager

Usage:
    python scripts/demo_advanced_caching.py
"""

import tempfile
from pathlib import Path

from victor.tools.caches import (
    AdaptiveTTLCache,
    AdvancedCacheManager,
    MultiLevelCache,
    PersistentSelectionCache,
    PredictiveCacheWarmer,
)


def demo_persistent_cache():
    """Demonstrate persistent cache functionality."""
    print("\n" + "=" * 80)
    print("1. PERSISTENT CACHE (SQLite)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "demo_cache.db"
        cache = PersistentSelectionCache(cache_path=str(cache_path))

        print(f"✓ Created persistent cache at: {cache_path}")

        # Add entries
        cache.put("query1", ["read", "search"], namespace="query", ttl=3600)
        cache.put("query2", ["write", "edit"], namespace="query", ttl=3600)
        print("✓ Added 2 entries to cache")

        # Retrieve
        result = cache.get("query1", namespace="query")
        print(f"✓ Retrieved: {result}")

        # Get stats
        stats = cache.get_stats()
        print(f"✓ Total entries: {stats['total_entries']}")
        print(f"✓ Hit rate: {stats['hit_rate']:.1%}")
        print(f"✓ Database size: {stats['database_size_bytes'] / 1024:.1f} KB")

        cache.close()
        print("✓ Cache closed and saved to disk")


def demo_adaptive_ttl():
    """Demonstrate adaptive TTL functionality."""
    print("\n" + "=" * 80)
    print("2. ADAPTIVE TTL CACHE")
    print("=" * 80)

    cache = AdaptiveTTLCache(
        max_size=100,
        min_ttl=60,
        max_ttl=7200,
        initial_ttl=3600,
        adjustment_threshold=3,
    )

    print("✓ Created adaptive TTL cache")
    print(f"  - Min TTL: {cache._min_ttl}s")
    print(f"  - Max TTL: {cache._max_ttl}s")
    print(f"  - Initial TTL: {cache._initial_ttl}s")

    # Add entries
    cache.put("hot_key", ["tool1", "tool2"])
    cache.put("cold_key", ["tool3"])

    print("✓ Added 2 entries")

    # Simulate hot access pattern
    for i in range(10):
        cache.get("hot_key")
    print("✓ Accessed 'hot_key' 10 times (should have longer TTL)")

    # Simulate cold access pattern
    cache.get("cold_key")
    print("✓ Accessed 'cold_key' 1 time (should have shorter TTL)")

    # Get metrics
    metrics = cache.get_metrics()
    print(f"✓ Total hits: {metrics['performance']['hits']}")
    print(f"✓ Hit rate: {metrics['performance']['hit_rate']:.1%}")
    print(f"✓ TTL adjustments: {metrics['performance']['ttl_adjustments']}")

    # Show TTL distribution
    ttl_dist = metrics["ttl"]["distribution"]
    print(f"✓ TTL distribution:")
    print(f"  - Min TTL: {ttl_dist['min_ttl']} entries")
    print(f"  - Low TTL: {ttl_dist['low_ttl']} entries")
    print(f"  - Medium TTL: {ttl_dist['medium_ttl']} entries")
    print(f"  - High TTL: {ttl_dist['high_ttl']} entries")
    print(f"  - Max TTL: {ttl_dist['max_ttl']} entries")


def demo_multi_level_cache():
    """Demonstrate multi-level cache functionality."""
    print("\n" + "=" * 80)
    print("3. MULTI-LEVEL CACHE (L1/L2/L3)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MultiLevelCache(
            l1_size=2,
            l2_size=10,
            l3_size=100,
            l2_dir=Path(tmpdir) / "l2",
        )

        print("✓ Created multi-level cache")
        print(f"  - L1 (in-memory): {cache._l1_size} entries")
        print(f"  - L2 (disk): {cache._l2_size} entries")
        print(f"  - L3 (large): {cache._l3_size} entries")

        # Add entries (will fill L1, then overflow to L2)
        cache.put("key1", ["tool1"])
        cache.put("key2", ["tool2"])
        cache.put("key3", ["tool3"])  # Will demote key1 to L2
        print("✓ Added 3 entries (L1 has 2, key1 demoted to L2)")

        # Access to demonstrate promotion
        for i in range(5):
            cache.get("key3")  # Hot entry
        print("✓ Accessed 'key3' 5 times (hot entry)")

        # Get metrics
        metrics = cache.get_metrics()
        print(f"✓ L1 hit rate: {metrics['l1']['hit_rate']:.1%}")
        print(f"✓ L2 hit rate: {metrics['l2']['hit_rate']:.1%}")
        print(f"✓ L3 hit rate: {metrics['l3']['hit_rate']:.1%}")
        print(f"✓ Combined hit rate: {metrics['combined']['hit_rate']:.1%}")


def demo_predictive_warming():
    """Demonstrate predictive cache warming."""
    print("\n" + "=" * 80)
    print("4. PREDICTIVE CACHE WARMING")
    print("=" * 80)

    warmer = PredictiveCacheWarmer(max_patterns=100, ngram_size=3)

    print("✓ Created predictive warmer")
    print(f"  - Max patterns: {warmer._max_patterns}")
    print(f"  - N-gram size: {warmer._ngram_size}")

    # Record query patterns
    warmer.record_query("read file", ["read"])
    warmer.record_query("analyze code", ["analyze", "search"])
    warmer.record_query("write code", ["write"])
    warmer.record_query("read file", ["read"])  # Repeat to establish pattern
    warmer.record_query("analyze code", ["analyze", "search"])
    print("✓ Recorded 5 query patterns")

    # Get predictions
    predictions = warmer.predict_next_queries(current_query="read file", top_k=5)
    print(f"✓ Generated {len(predictions.queries)} predictions for 'read file':")
    for query, confidence in zip(predictions.queries, predictions.confidences):
        print(f"  - '{query[:40]}...' (confidence: {confidence:.2f})")

    # Get statistics
    stats = warmer.get_statistics()
    print(f"✓ Total patterns: {stats['patterns']['total']}")
    print(f"✓ Average frequency: {stats['patterns']['avg_frequency']:.1f}")


def demo_advanced_cache_manager():
    """Demonstrate unified cache manager."""
    print("\n" + "=" * 80)
    print("5. UNIFIED CACHE MANAGER")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = AdvancedCacheManager(
            cache_size=100,
            persistent_enabled=True,
            persistent_path=str(Path(tmpdir) / "unified_cache.db"),
            adaptive_ttl_enabled=True,
            multi_level_enabled=False,
            predictive_warming_enabled=False,
        )

        print("✓ Created advanced cache manager")
        print("  - Basic cache: enabled")
        print("  - Persistent cache: enabled")
        print("  - Adaptive TTL: enabled")

        # Add entries
        cache.put_query("query1", ["read", "search"])
        cache.put_query("query2", ["write", "edit"])
        cache.put_query("query3", ["analyze"])
        print("✓ Added 3 entries")

        # Retrieve
        result1 = cache.get_query("query1")
        result2 = cache.get_query("query2")
        result3 = cache.get_query("query3")
        print(f"✓ Retrieved 3 entries")

        # Miss
        miss = cache.get_query("query4")
        print(f"✓ Cache miss for 'query4': {miss}")

        # Get comprehensive metrics
        metrics = cache.get_metrics()
        print("\n✓ Comprehensive Metrics:")
        print(f"  - Combined hit rate: {metrics.combined['hit_rate']:.1%}")
        print(f"  - Total hits: {metrics.combined['total_hits']}")
        print(f"  - Total misses: {metrics.combined['total_misses']}")
        print(f"  - Total entries: {metrics.combined['total_entries']}")

        print("\n✓ Strategies Enabled:")
        for strategy, enabled in metrics.combined["strategies_enabled"].items():
            status = "✓" if enabled else "✗"
            print(f"  {status} {strategy}")

        cache.close()
        print("\n✓ Cache manager closed (persistent cache saved)")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("ADVANCED CACHING STRATEGIES DEMONSTRATION")
    print("Track 5.2: Advanced Caching Features")
    print("=" * 80)

    try:
        demo_persistent_cache()
        demo_adaptive_ttl()
        demo_multi_level_cache()
        demo_predictive_warming()
        demo_advanced_cache_manager()

        print("\n" + "=" * 80)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. Persistent cache: Survives process restarts")
        print("  2. Adaptive TTL: Automatically optimizes based on access patterns")
        print("  3. Multi-level cache: Hierarchical storage for optimal performance")
        print("  4. Predictive warming: Proactive cache preparation")
        print("  5. Unified manager: Single interface for all strategies")
        print("\nFor more information, see: docs/performance/advanced_caching.md")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
