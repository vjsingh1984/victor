#!/usr/bin/env python3
"""Tool Selector Accelerator Demo.

Demonstrates the performance and usage of the ToolSelectorAccelerator.
"""

import random
import time
from typing import List

# Import the accelerator
from victor.native.rust import get_tool_selector_accelerator
from victor.native.observability import NativeMetrics


def generate_embedding() -> List[float]:
    """Generate a random 384-dimensional embedding."""
    return [random.random() for _ in range(384)]


def main():
    """Run the demo."""
    print("=" * 60)
    print("Tool Selection Accelerator Demo")
    print("=" * 60)

    # Get accelerator instance
    accelerator = get_tool_selector_accelerator()

    print(f"\nBackend: {accelerator.backend}")
    print(f"Rust Available: {accelerator.rust_available}")

    if accelerator.rust_available:
        print(f"Version: {accelerator.get_version()}")

    # Generate test data
    print("\n" + "-" * 60)
    print("Generating test data...")
    print("-" * 60)

    num_tools = 100
    query = generate_embedding()
    tools = [generate_embedding() for _ in range(num_tools)]
    tool_names = [f"tool_{i}" for i in range(num_tools)]

    # Category mapping
    categories = ["file_ops", "search", "execution", "git", "analysis"]
    tool_category_map = {
        name: categories[i % len(categories)]
        for i, name in enumerate(tool_names)
    }

    print(f"Query embedding: {len(query)} dimensions")
    print(f"Tools: {num_tools}")
    print(f"Categories: {', '.join(categories)}")

    # Test cosine similarity
    print("\n" + "-" * 60)
    print("1. Cosine Similarity Batch")
    print("-" * 60)

    start = time.perf_counter()
    similarities = accelerator.cosine_similarity_batch(query, tools)
    duration = (time.perf_counter() - start) * 1000

    print(f"Computed {len(similarities)} similarities in {duration:.3f}ms")
    print(f"Sample similarities: {similarities[:5]}")

    # Test top-k selection
    print("\n" + "-" * 60)
    print("2. Top-K Selection")
    print("-" * 60)

    k = 10
    start = time.perf_counter()
    top_k_indices = accelerator.topk_indices(similarities, k)
    duration = (time.perf_counter() - start) * 1000

    print(f"Selected top {k} indices in {duration:.3f}ms")
    print(f"Top indices: {top_k_indices}")
    print(f"Top scores: {[similarities[i] for i in top_k_indices]}")

    # Test top-k with scores
    start = time.perf_counter()
    top_k_with_scores = accelerator.topk_with_scores(similarities, k)
    duration = (time.perf_counter() - start) * 1000

    print(f"\nTop-K with scores in {duration:.3f}ms:")
    for idx, score in top_k_with_scores[:3]:
        print(f"  {tool_names[idx]}: {score:.4f}")

    # Test category filtering
    print("\n" + "-" * 60)
    print("3. Category Filtering")
    print("-" * 60)

    allowed_categories = {"file_ops", "git"}
    start = time.perf_counter()
    filtered = accelerator.filter_by_category(
        tool_names, allowed_categories, tool_category_map
    )
    duration = (time.perf_counter() - start) * 1000

    print(f"Filtered to {len(filtered)} tools in {duration:.3f}ms")
    print(f"Allowed categories: {', '.join(allowed_categories)}")
    print(f"Filtered tools: {filtered[:10]}")

    # Test combined filter and rank
    print("\n" + "-" * 60)
    print("4. Combined Filter and Rank")
    print("-" * 60)

    start = time.perf_counter()
    results = accelerator.filter_and_rank(
        query=query,
        tools=tools,
        tool_names=tool_names,
        available_categories=allowed_categories,
        tool_category_map=tool_category_map,
        k=5,
    )
    duration = (time.perf_counter() - start) * 1000

    print(f"Filter and rank completed in {duration:.3f}ms")
    print(f"Top 5 tools in '{', '.join(allowed_categories)}':")
    for tool_name, score in results:
        print(f"  {tool_name}: {score:.4f}")

    # Performance benchmark
    print("\n" + "-" * 60)
    print("5. Performance Benchmark")
    print("-" * 60)

    iterations = 100
    print(f"Running {iterations} iterations...")

    # Cosine similarity benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = accelerator.cosine_similarity_batch(query, tools)
    duration = time.perf_counter() - start

    print(f"\nCosine Similarity:")
    print(f"  Total: {duration:.3f}s")
    print(f"  Average: {duration/iterations*1000:.3f}ms")
    print(f"  Throughput: {iterations*num_tools/duration:.0f} tools/sec")

    # Top-k benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = accelerator.topk_indices(similarities, k)
    duration = time.perf_counter() - start

    print(f"\nTop-K Selection:")
    print(f"  Total: {duration:.3f}s")
    print(f"  Average: {duration/iterations*1000:.3f}ms")

    # Metrics summary
    print("\n" + "-" * 60)
    print("6. Metrics Summary")
    print("-" * 60)

    metrics = NativeMetrics.get_instance()
    stats = metrics.get_stats()

    print("\nOperation Statistics:")
    for op, op_stats in sorted(stats.items()):
        if op["calls_total"] > 0:
            print(f"\n{op}:")
            print(f"  Calls: {op_stats['calls_total']:.0f}")
            print(f"  Avg Duration: {op_stats['duration_ms_avg']:.3f}ms")
            print(f"  Rust Ratio: {op_stats['rust_ratio']:.1%}")

    summary = metrics.get_summary()
    print(f"\nOverall Summary:")
    print(f"  Total Calls: {summary['total_calls']:.0f}")
    print(f"  Total Duration: {summary['total_duration_ms']:.3f}ms")
    print(f"  Avg Duration: {summary['avg_duration_ms']:.3f}ms")
    print(f"  Rust Ratio: {summary['rust_ratio']:.1%}")
    print(f"  Error Rate: {summary['error_rate']:.1%}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
