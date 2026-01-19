#!/usr/bin/env python3
"""
Integration Example: Advanced Caching with SemanticToolSelector

This example demonstrates how to integrate advanced caching features
with the existing SemanticToolSelector for maximum performance.
"""

import asyncio
import logging
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.caches import (
    get_tool_selection_cache,
    PredictiveCacheWarmer,
    AdaptiveLRUCache,
    MultiLevelCache,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_caching():
    """Example 1: Basic caching with SemanticToolSelector."""
    print("\n=== Example 1: Basic Caching ===\n")

    # Initialize selector
    selector = SemanticToolSelector()
    await selector.initialize_tool_embeddings(tools_registry)  # Your tools registry

    # Get cache
    cache = get_tool_selection_cache()

    # First call - cache miss
    tools1 = await selector.select_relevant_tools(
        "read the file",
        tools_registry,
    )

    # Second call - cache hit (36% faster)
    tools2 = await selector.select_relevant_tools(
        "read the file",
        tools_registry,
    )

    # Check cache stats
    stats = cache.get_stats()
    print(f"Cache hit rate: {stats['combined']['hit_rate']:.1%}")


async def example_predictive_warming():
    """Example 2: Predictive cache warming."""
    print("\n=== Example 2: Predictive Warming ===\n")

    selector = SemanticToolSelector()
    cache = get_tool_selection_cache()
    warmer = PredictiveCacheWarmer(cache=cache, max_patterns=500)

    # Simulate query sequence
    queries = [
        ("read the file", ["read", "search"]),
        ("analyze code", ["analyze", "search"]),
        ("run tests", ["test", "shell"]),
        ("show diff", ["git", "shell"]),
    ]

    # Record patterns
    for query, tools in queries:
        warmer.record_query(query, tools)
        print(f"Recorded: {query}")

    # Predict next queries
    current_query = "analyze code"
    predictions = warmer.predict_next_queries(current_query, top_k=3)

    print(f"\nPredictions after '{current_query}':")
    for i, (query, conf, tools) in enumerate(
        zip(predictions.queries, predictions.confidences, predictions.tools), 1
    ):
        print(f"  {i}. {query} (confidence: {conf:.2f})")

    # Prewarm cache for predicted queries
    async def prewarm_selection(query):
        """Prewarm function for a predicted query."""
        return await selector.select_relevant_tools(query, tools_registry)

    prewarmed = await warmer.prewarm_predictions(predictions, selection_fn=prewarm_selection)
    print(f"\nPrewarmed {prewarmed} cache entries")


async def example_multi_level_cache():
    """Example 3: Multi-level caching."""
    print("\n=== Example 3: Multi-Level Cache ===\n")

    import tempfile
    from pathlib import Path

    # Create multi-level cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MultiLevelCache(
            l1_size=10,  # Fast, in-memory
            l2_size=100,  # Medium, local disk
            l3_size=1000,  # Slow, backup
            l2_dir=Path(tmpdir),
        )

        # Add entries
        for i in range(15):
            cache.put(f"key{i}", f"value{i}")

        # Access some entries
        for i in range(5):
            value = cache.get(f"key{i}")
            print(f"Retrieved key{i}: {value}")

        # Get metrics
        metrics = cache.get_metrics()
        print(f"\nL1 hit rate: {metrics['l1']['hit_rate']:.1%}")
        print(f"L2 hit rate: {metrics['l2']['hit_rate']:.1%}")
        print(f"L3 hit rate: {metrics['l3']['hit_rate']:.1%}")
        print(f"Overall: {metrics['combined']['hit_rate']:.1%}")


async def example_adaptive_cache():
    """Example 4: Adaptive cache sizing."""
    print("\n=== Example 4: Adaptive Cache Sizing ===\n")

    # Create adaptive cache
    cache = AdaptiveLRUCache(
        initial_size=100,
        max_size=500,
        target_hit_rate=0.6,
    )

    # Simulate workload
    for i in range(200):
        cache.put(f"key{i}", f"value{i}")
        cache.get(f"key{i}")  # Hit

    # Add some misses
    for i in range(200, 300):
        cache.get(f"key{i}")  # Miss

    # Check if adjustment needed
    if cache.should_adjust():
        result = cache.adjust_size()
        print(f"Cache adjusted: {result['old_size']} -> {result['new_size']}")
        print(f"Reason: {result['reason']}")

    # Get metrics
    metrics = cache.get_metrics()
    print(f"\nHit rate: {metrics['performance']['hit_rate']:.1%}")
    print(f"Cache size: {metrics['size']['current']}")
    print(f"Adjustments: {metrics['adaptive']['adjustments']}")


async def example_full_integration():
    """Example 5: Full integration with all optimizations."""
    print("\n=== Example 5: Full Integration ===\n")

    import tempfile
    from pathlib import Path

    # Create multi-level cache
    with tempfile.TemporaryDirectory() as tmpdir:
        ml_cache = MultiLevelCache(
            l1_size=50,
            l2_size=500,
            l3_size=5000,
            l2_dir=Path(tmpdir),
        )

        # Create adaptive L1 cache
        l1_cache = AdaptiveLRUCache(initial_size=50, max_size=200)

        # Create predictive warmer
        warmer = PredictiveCacheWarmer(cache=ml_cache, max_patterns=500)

        # Initialize selector
        selector = SemanticToolSelector()
        await selector.initialize_tool_embeddings(tools_registry)

        # Process queries with all optimizations
        queries = [
            "read the file",
            "analyze code",
            "run tests",
            "show diff",
            "edit files",
        ]

        for query in queries:
            print(f"\nProcessing: {query}")

            # Check cache
            cached = ml_cache.get(query)
            if cached:
                print(f"  Cache hit!")
                tools = cached
            else:
                # Cache miss - perform selection
                tools = await selector.select_relevant_tools(query, tools_registry)
                ml_cache.put(query, tools)
                print(f"  Selected {len(tools)} tools")

            # Record for predictions
            warmer.record_query(query, [t.name for t in tools])

            # Get predictions and prewarm
            predictions = warmer.predict_next_queries(query, top_k=3)
            if predictions.total_confidence > 0.3:
                print(f"  Predictions: {predictions.queries}")
                # Prewarm would happen here...

        # Auto-adjust cache size
        if l1_cache.should_adjust():
            result = l1_cache.adjust_size()
            print(f"\nAdaptive adjustment: {result['old_size']} -> {result['new_size']}")

        # Show final metrics
        ml_metrics = ml_cache.get_metrics()
        print(f"\n=== Final Metrics ===")
        print(f"Overall hit rate: {ml_metrics['combined']['hit_rate']:.1%}")
        print(f"Total promotions: {ml_metrics['combined']['total_promotions']}")
        print(f"Total demotions: {ml_metrics['combined']['total_demotions']}")

        l1_metrics = l1_cache.get_metrics()
        print(f"\nAdaptive cache:")
        print(f"  Hit rate: {l1_metrics['performance']['hit_rate']:.1%}")
        print(f"  Size: {l1_metrics['size']['current']}")

        warmer_stats = warmer.get_statistics()
        print(f"\nPredictive warmer:")
        print(f"  Patterns learned: {warmer_stats['patterns']['total']}")
        print(f"  Prediction accuracy: {warmer_stats['predictions']['accuracy']:.1%}")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Advanced Caching Integration Examples")
    print("=" * 60)

    # Note: You need to provide a real tools_registry for these examples
    # tools_registry = ...  # Your ToolRegistry instance

    # Run examples (commented out - need tools_registry)
    # await example_basic_caching()
    # await example_predictive_warming()
    # await example_multi_level_cache()
    # await example_adaptive_cache()
    # await example_full_integration()

    print("\n=== Examples Complete ===")
    print("\nNote: Uncomment the examples above and provide a tools_registry")
    print("to run the full integration examples.")


if __name__ == "__main__":
    asyncio.run(main())
