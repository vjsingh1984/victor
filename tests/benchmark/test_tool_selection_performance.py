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

"""Performance tests for tool selection optimization (Phase 2.1).

These tests verify that tool selection meets the following performance targets:
- Cold cache: <50ms per selection
- Warm cache: <10ms per selection (10x speedup)
- Cache hit rate: >80% for common queries

Run with:
    pytest tests/benchmark/test_tool_selection_performance.py -v
    pytest tests/benchmark/test_tool_selection_performance.py::test_tool_selection_cold_cache -v
    pytest tests/benchmark/test_tool_selection_performance.py -k "cache_hit_rate" -v
"""

# CRITICAL: Set environment variable to skip .env loading and prevent framework import issues
import os

os.environ["VICTOR_SKIP_ENV_FILE"] = "1"

import asyncio
import pytest
import time
from typing import List
import sys

# Import directly from modules to avoid framework import issues
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.registry import ToolRegistry
from victor.providers.base import ToolDefinition


# Common queries for cache hit rate testing (from production logs)
COMMON_QUERIES = [
    "read the file",
    "write to file",
    "search code",
    "find classes",
    "analyze codebase",
    "run tests",
    "git commit",
    "edit files",
    "show diff",
    "create endpoint",
    "list directory",
    "find functions",
    "run command",
    "check status",
]


@pytest.fixture
async def tool_selector():
    """Create and initialize a tool selector for testing."""
    selector = SemanticToolSelector(
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
        cache_embeddings=True,
    )

    # Create tool registry and populate with test tools
    tool_registry = ToolRegistry()

    # Register a subset of tools for testing (not full 55 tools)
    test_tools = [
        {"name": "read", "description": "Read file contents"},
        {"name": "write", "description": "Write content to file"},
        {"name": "edit", "description": "Edit file with replacement"},
        {"name": "bash", "description": "Execute bash command"},
        {"name": "grep", "description": "Search for text in files"},
        {"name": "ls", "description": "List directory contents"},
        {"name": "semantic_search", "description": "Semantic code search"},
        {"name": "code_search", "description": "Search code patterns"},
        {"name": "git", "description": "Git operations"},
        {"name": "test", "description": "Run tests"},
    ]

    for tool_spec in test_tools:
        from victor.tools.base import Tool, tool

        # Create a simple tool wrapper
        class SimpleTool(Tool):
            def __init__(self, name, description):
                self.name = name
                self.description = description
                self.parameters = {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return f"Executed {self.name}"

        simple_tool = SimpleTool(tool_spec["name"], tool_spec["description"])
        tool_registry.register(simple_tool)

    # Initialize embeddings
    await selector.initialize_tool_embeddings(tool_registry)

    yield selector

    # Cleanup
    await selector.close()


@pytest.mark.benchmark
@pytest.mark.slow
async def test_tool_selection_cold_cache(tool_selector: SemanticToolSelector):
    """Test cold cache performance target: <50ms per selection.

    This test simulates first-time queries with no cache hits.
    Target: Cold cache latency should be <50ms for 90th percentile.
    """
    # Clear query cache to ensure cold start
    tool_selector._query_embedding_cache.clear()

    cold_queries = [
        "implement a REST API endpoint for user authentication",
        "find all subclasses of BaseController in the codebase",
        "analyze the performance bottleneck in the database query",
        "create a unit test for the payment processing module",
        "refactor the legacy authentication system",
    ]

    latencies = []

    for query in cold_queries:
        start = time.perf_counter()
        tools = await tool_selector.select_relevant_tools(
            user_message=query,
            tools=tool_selector._tools_registry,
            max_tools=5,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    p90_latency = sorted(latencies)[int(len(latencies) * 0.9)]
    max_latency = max(latencies)

    print(f"\nCold cache latencies (ms): {latencies}")
    print(f"Average: {avg_latency:.2f}ms")
    print(f"P90: {p90_latency:.2f}ms")
    print(f"Max: {max_latency:.2f}ms")

    # Assert performance targets
    assert p90_latency < 50, f"P90 cold cache latency too high: {p90_latency:.2f}ms (target: <50ms)"
    assert (
        avg_latency < 40
    ), f"Average cold cache latency too high: {avg_latency:.2f}ms (target: <40ms)"


@pytest.mark.benchmark
@pytest.mark.slow
async def test_tool_selection_warm_cache(tool_selector: SemanticToolSelector):
    """Test warm cache performance target: <10ms per selection (10x speedup).

    This test measures performance with cached query embeddings.
    Target: Warm cache latency should be <10ms for 95th percentile.
    """
    # Warm up the cache with common queries
    for query in COMMON_QUERIES:
        await tool_selector.select_relevant_tools(
            user_message=query,
            tools=tool_selector._tools_registry,
            max_tools=5,
        )

    # Now test warm cache performance with repeated queries
    warm_latencies = []

    for query in COMMON_QUERIES:
        start = time.perf_counter()
        tools = await tool_selector.select_relevant_tools(
            user_message=query,
            tools=tool_selector._tools_registry,
            max_tools=5,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        warm_latencies.append(latency_ms)

    # Calculate statistics
    avg_latency = sum(warm_latencies) / len(warm_latencies)
    p95_latency = sorted(warm_latencies)[int(len(warm_latencies) * 0.95)]
    max_latency = max(warm_latencies)

    print(f"\nWarm cache latencies (ms): {[f'{l:.2f}' for l in warm_latencies]}")
    print(f"Average: {avg_latency:.2f}ms")
    print(f"P95: {p95_latency:.2f}ms")
    print(f"Max: {max_latency:.2f}ms")

    # Assert performance targets
    assert p95_latency < 10, f"P95 warm cache latency too high: {p95_latency:.2f}ms (target: <10ms)"
    assert (
        avg_latency < 8
    ), f"Average warm cache latency too high: {avg_latency:.2f}ms (target: <8ms)"


@pytest.mark.benchmark
@pytest.mark.slow
async def test_cache_hit_rate(tool_selector: SemanticToolSelector):
    """Test cache hit rate target: >80% for common queries.

    This test measures the effectiveness of the query embedding cache
    and category pre-filtering. Target: >80% cache hit rate.
    """
    # Clear stats
    tool_selector._cache_hit_count = 0
    tool_selector._cache_miss_count = 0
    tool_selector._total_selections = 0

    # Run mixed workload: 70% common queries, 30% unique queries
    num_iterations = 100

    for i in range(num_iterations):
        if i % 10 < 7:  # 70% common queries
            query = COMMON_QUERIES[i % len(COMMON_QUERIES)]
        else:  # 30% unique queries
            query = f"unique query {i} with specific requirements"

        await tool_selector.select_relevant_tools(
            user_message=query,
            tools=tool_selector._tools_registry,
            max_tools=5,
        )

    # Calculate hit rate
    hit_rate = tool_selector._cache_hit_count / tool_selector._total_selections
    total_selections = tool_selector._total_selections

    print(f"\nCache hit rate: {hit_rate:.2%}")
    print(f"Total selections: {total_selections}")
    print(f"Cache hits: {tool_selector._cache_hit_count}")
    print(f"Cache misses: {tool_selector._cache_miss_count}")

    # Assert target
    assert hit_rate > 0.8, f"Cache hit rate too low: {hit_rate:.2%} (target: >80%)"


@pytest.mark.benchmark
@pytest.mark.slow
async def test_category_pre_filtering(tool_selector: SemanticToolSelector):
    """Test category pre-filtering reduces candidate tools.

    This test verifies PERF-002: Category pre-filtering should reduce
    the candidate set from 55 tools to 10-15 tools for typical queries.
    """
    from unittest.mock import patch

    query = "find all Python files with authentication logic"

    # Track which tools are evaluated
    evaluated_tools = []

    original_get_embedding = tool_selector._get_embedding

    async def mock_get_embedding(text):
        # Track which tool embeddings are requested
        if text.startswith("Tool: "):
            tool_name = text.replace("Tool: ", "")
            evaluated_tools.append(tool_name)
        return await original_get_embedding(text)

    with patch.object(tool_selector, "_get_embedding", side_effect=mock_get_embedding):
        tools = await tool_selector.select_relevant_tools(
            user_message=query,
            tools=tool_selector._tools_registry,
            max_tools=5,
        )

    # Verify pre-filtering reduced the candidate set
    total_tools = len(tool_selector._tools_registry.list_tools())
    evaluated_count = len(evaluated_tools)

    print(f"\nTotal tools in registry: {total_tools}")
    print(f"Tools evaluated after pre-filtering: {evaluated_count}")
    print(f"Reduction: {((total_tools - evaluated_count) / total_tools) * 100:.1f}%")

    # Assert pre-filtering is effective (should evaluate <50% of tools)
    assert evaluated_count < total_tools * 0.5, (
        f"Category pre-filtering ineffective: "
        f"evaluated {evaluated_count}/{total_tools} tools "
        f"(target: <50%)"
    )

    # Verify we still got relevant tools
    assert len(tools) > 0, "No tools selected despite pre-filtering"
    print(f"Selected {len(tools)} tools: {[t.name for t in tools]}")


@pytest.mark.benchmark
@pytest.mark.slow
async def test_query_cache_size(tool_selector: SemanticToolSelector):
    """Test query cache expansion to 500 entries (Phase 2.1).

    Verifies that the expanded cache provides higher hit rates
    for diverse workloads.
    """
    # Verify cache size configuration
    assert tool_selector._query_cache_max_size == 500, (
        f"Query cache size not expanded: {tool_selector._query_cache_max_size} " f"(target: 500)"
    )

    # Test cache can handle 500 unique queries
    unique_queries = [f"test query {i} with specific context" for i in range(500)]

    # Populate cache
    for query in unique_queries:
        await tool_selector.select_relevant_tools(
            user_message=query,
            tools=tool_selector._tools_registry,
            max_tools=5,
        )

    # Verify cache size
    cache_size = len(tool_selector._query_embedding_cache)
    print(f"\nQuery cache size after 500 unique queries: {cache_size}")
    print(f"Cache utilization: {cache_size / 500:.2%}")

    # Cache should be close to full (allowing 5% margin for LRU eviction during warmup)
    assert cache_size >= 475, f"Cache size too low: {cache_size} (target: >=475)"


@pytest.mark.benchmark
@pytest.mark.slow
async def test_performance_stats(tool_selector: SemanticToolSelector):
    """Test performance statistics collection (PERF-005).

    Verifies that get_performance_stats() returns accurate metrics.
    """
    # Clear stats
    tool_selector._cache_hit_count = 0
    tool_selector._cache_miss_count = 0
    tool_selector._total_selections = 0

    # Run some selections
    for query in COMMON_QUERIES[:5]:
        await tool_selector.select_relevant_tools(
            user_message=query,
            tools=tool_selector._tools_registry,
            max_tools=5,
        )

    # Get stats
    stats = tool_selector.get_performance_stats()

    print("\nPerformance stats:")
    print(f"  Total selections: {stats['total_selections']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Cache hit count: {stats['cache_hit_count']}")
    print(f"  Cache miss count: {stats['cache_miss_count']}")
    print(f"  Last latency: {stats['last_latency_ms']:.2f}ms")
    print(f"  Query cache size: {stats['query_cache_size']}")
    print(f"  Query cache utilization: {stats['query_cache_utilization']:.2%}")

    # Verify stats are accurate
    assert stats["total_selections"] == 5, "Total selections mismatch"
    assert stats["query_cache_size"] > 0, "Query cache should not be empty"
    assert 0 <= stats["cache_hit_rate"] <= 1, "Cache hit rate should be between 0 and 1"
    assert stats["query_cache_max_size"] == 500, "Cache max size should be 500"


@pytest.mark.benchmark
async def test_warmup_performance(tool_selector: SemanticToolSelector):
    """Test cache warmup performance (PERF-004).

    Verifies that cache warmup completes in <200ms for 18 common patterns.
    """
    start = time.perf_counter()

    # Warmup is already done in fixture, but we can measure it again
    await tool_selector._warmup_query_cache()

    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\nWarmup completed in {elapsed_ms:.2f}ms")
    print(f"Query cache size after warmup: {len(tool_selector._query_embedding_cache)}")

    # Warmup should complete quickly
    assert elapsed_ms < 200, f"Warmup too slow: {elapsed_ms:.2f}ms (target: <200ms)"

    # Cache should have entries
    assert len(tool_selector._query_embedding_cache) > 0, "Warmup failed to populate cache"


@pytest.mark.benchmark
async def test_selection_latency_distribution(tool_selector: SemanticToolSelector):
    """Test selection latency distribution across query types.

    Measures performance for different query complexity levels:
    - Simple: single operation (e.g., "read file")
    - Medium: multi-step (e.g., "find and edit")
    - Complex: analysis (e.g., "analyze architecture")
    """
    simple_queries = ["read the file", "write code", "list files"]
    medium_queries = [
        "find all classes and edit them",
        "search code and create test",
        "analyze functions and refactor",
    ]
    complex_queries = [
        "analyze the codebase architecture and identify bottlenecks",
        "find all authentication flows and review security",
        "search for error handling patterns and suggest improvements",
    ]

    async def measure_latencies(queries, label):
        latencies = []
        for query in queries:
            start = time.perf_counter()
            await tool_selector.select_relevant_tools(
                user_message=query,
                tools=tool_selector._tools_registry,
                max_tools=5,
            )
            latencies.append((time.perf_counter() - start) * 1000)

        avg = sum(latencies) / len(latencies)
        p90 = sorted(latencies)[int(len(latencies) * 0.9)]
        print(f"{label}: avg={avg:.2f}ms, p90={p90:.2f}ms")
        return latencies

    print("\nSelection latency by complexity:")

    # Clear cache for cold start measurements
    tool_selector._query_embedding_cache.clear()

    simple_latencies = await measure_latencies(simple_queries, "Simple queries")
    medium_latencies = await measure_latencies(medium_queries, "Medium queries")
    complex_latencies = await measure_latencies(complex_queries, "Complex queries")

    # All query types should meet cold cache target
    all_latencies = simple_latencies + medium_latencies + complex_latencies
    p90_all = sorted(all_latencies)[int(len(all_latencies) * 0.9)]

    assert p90_all < 50, f"P90 latency too high: {p90_all:.2f}ms (target: <50ms)"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
