#!/usr/bin/env python3
"""Example demonstrating advanced caching features in Victor AI.

This example shows:
1. Multi-level cache setup
2. Cache warming
3. Semantic caching
4. Cache invalidation
5. Analytics and monitoring
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def expensive_computation(key: str) -> str:
    """Simulate an expensive computation."""
    logger.info(f"Computing result for {key}...")
    await asyncio.sleep(1)  # Simulate work
    return f"Result for {key} at {asyncio.get_event_loop().time()}"


async def example_multi_level_cache():
    """Demonstrate multi-level cache usage."""
    from victor.core.cache import MultiLevelCache, CacheLevelConfig, WritePolicy

    logger.info("=== Multi-Level Cache Example ===")

    # Create cache
    cache = MultiLevelCache(
        l1_config=CacheLevelConfig(max_size=10, ttl=60),
        l2_config=CacheLevelConfig(max_size=100, ttl=300),
        write_policy=WritePolicy.WRITE_THROUGH,
    )

    # Store values
    await cache.set("key1", await expensive_computation("key1"), namespace="example")
    await cache.set("key2", await expensive_computation("key2"), namespace="example")

    # Retrieve from cache
    result1 = await cache.get("key1", namespace="example")
    logger.info(f"Retrieved: {result1}")

    # Get statistics
    stats = cache.get_stats()
    logger.info(f"L1 Hit Rate: {stats['l1']['hit_rate']:.1%}")
    logger.info(f"L2 Hit Rate: {stats['l2']['hit_rate']:.1%}")
    logger.info(f"Combined: {stats['combined_hit_rate']:.1%}")


async def example_cache_warming():
    """Demonstrate cache warming."""
    from victor.core.cache import MultiLevelCache, CacheWarmer, WarmingStrategy

    logger.info("=== Cache Warming Example ===")

    cache = MultiLevelCache()

    # Value loader for warming
    async def load_value(key: str, namespace: str) -> Any:
        return await expensive_computation(key)

    # Create warmer
    warmer = CacheWarmer(
        cache=cache,
        strategy=WarmingStrategy.HYBRID,
        value_loader=load_value,
    )

    # Record some access patterns
    for i in range(10):
        await warmer.record_access(f"key{i}", "example", hit=True)

    # Warm top items
    count = await warmer.warm_top_items(n=5)
    logger.info(f"Warmed {count} items")

    # Get statistics
    stats = warmer.get_stats()
    logger.info(f"Total patterns: {stats['total_patterns']}")
    logger.info(f"Unique keys: {stats['unique_keys']}")


async def example_semantic_cache():
    """Demonstrate semantic caching."""
    from victor.core.cache import SemanticCache
    from victor.providers.base import Message, CompletionResponse

    logger.info("=== Semantic Cache Example ===")

    # Note: This requires embedding service to be available
    # For demonstration, we'll show the API

    cache = SemanticCache(
        similarity_threshold=0.85,
        max_size=100,
    )

    # Store a response
    messages1 = [Message(role="user", content="How do I parse JSON in Python?")]
    response1 = CompletionResponse(
        content="Use json.loads() to parse JSON in Python.",
        role="assistant",
    )

    await cache.put(messages1, response1)
    logger.info("Stored response for 'How do I parse JSON in Python?'")

    # Similar query (different wording)
    messages2 = [Message(role="user", content="Python JSON parsing example")]

    # Will find semantically similar cached response
    result = await cache.get_similar(messages2)
    if result:
        logger.info(f"Found similar response: {result.content[:50]}...")
    else:
        logger.info("No similar response found (embedding service may not be available)")

    # Get statistics
    stats = cache.get_stats()
    logger.info(f"Hit Rate: {stats['hit_rate']:.1%}")
    logger.info(f"Semantic Hit Rate: {stats['semantic_hit_rate']:.1%}")


async def example_cache_invalidation():
    """Demonstrate cache invalidation."""
    from victor.core.cache import MultiLevelCache, CacheInvalidator

    logger.info("=== Cache Invalidation Example ===")

    cache = MultiLevelCache()
    invalidator = CacheInvalidator(
        cache=cache,
        enable_tagging=True,
        enable_dependencies=True,
    )

    # Store and tag values
    await cache.set("file1", "content1", namespace="example")
    await cache.set("file2", "content2", namespace="example")

    invalidator.tag("file1", "example", ["python_files", "src"])
    invalidator.tag("file2", "example", ["python_files", "tests"])

    # Add dependencies
    invalidator.add_dependency("analysis1", "example", "/src/file1.py")
    invalidator.add_dependency("analysis2", "example", "/tests/test_file1.py")

    # Invalidate by tag
    count = await invalidator.invalidate_tag("python_files")
    logger.info(f"Invalidated {count} entries with tag 'python_files'")

    # Add dependency and invalidate
    await cache.set("analysis1", "result1", namespace="example")
    invalidator.add_dependency("analysis1", "example", "/src/file1.py")

    count = await invalidator.invalidate_dependents("/src/file1.py")
    logger.info(f"Invalidated {count} dependent entries")


async def example_cache_analytics():
    """Demonstrate cache analytics."""
    from victor.core.cache import MultiLevelCache, CacheAnalytics

    logger.info("=== Cache Analytics Example ===")

    cache = MultiLevelCache()
    analytics = CacheAnalytics(cache=cache, track_hot_keys=True)

    # Simulate cache accesses
    for i in range(100):
        key = f"key{i % 10}"  # Some keys accessed more frequently
        hit = i % 2 == 0  # 50% hit rate
        latency_ms = 0.1 if hit else 5.0

        analytics.record_access(key, "example", hit=hit, latency_ms=latency_ms)

    # Get statistics
    stats = analytics.get_comprehensive_stats()
    logger.info(f"Hit Rate: {stats['hit_rate']:.1%}")
    logger.info(f"Miss Rate: {stats['miss_rate']:.1%}")
    logger.info(f"Avg Latency: {stats['latency']['avg_ms']:.2f}ms")
    logger.info(f"P95 Latency: {stats['latency']['p95_ms']:.2f}ms")

    # Get hot keys
    hot_keys = analytics.get_hot_keys(top_n=5)
    logger.info("Top 5 Hot Keys:")
    for hot_key in hot_keys:
        logger.info(
            f"  {hot_key.key}: {hot_key.access_count} accesses, "
            f"hit rate {hot_key.hit_count/hot_key.access_count:.1%}"
        )

    # Get recommendations
    recommendations = analytics.get_recommendations()
    if recommendations:
        logger.info("Recommendations:")
        for rec in recommendations[:3]:
            logger.info(f"  [{rec.priority.upper()}] {rec.title}")


async def main():
    """Run all examples."""
    logger.info("Starting Advanced Caching Examples\n")

    await example_multi_level_cache()
    print()

    await example_cache_warming()
    print()

    await example_semantic_cache()
    print()

    await example_cache_invalidation()
    print()

    await example_cache_analytics()
    print()

    logger.info("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
