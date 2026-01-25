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

"""Benchmark suite for performance optimizations.

Measures the effectiveness of various optimizations:
- Response caching
- Request batching
- Hot path optimizations
- Overall system performance
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import pytest

from victor.core.cache import ResponseCache
from victor.core.batching import RequestBatcher, BatchPriority
from victor.core.optimizations import json_dumps, json_loads
from victor.providers.base import Message, CompletionResponse


# =============================================================================
# Benchmark Results
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    throughput: float  # Operations per second
    metadata: Dict[str, Any]

    def speedup(self, other: "BenchmarkResult") -> float:
        """Calculate speedup compared to another result.

        Args:
            other: Other benchmark result

        Returns:
            Speedup factor (> 1 means this is faster)
        """
        return other.avg_time / self.avg_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time": round(self.total_time, 4),
            "avg_time": round(self.avg_time, 6),
            "min_time": round(self.min_time, 6),
            "max_time": round(self.max_time, 6),
            "throughput": round(self.throughput, 2),
            "metadata": self.metadata,
        }


# =============================================================================
# Benchmark Utilities
# =============================================================================


def run_benchmark(func, iterations: int, warmup: int = 0) -> BenchmarkResult:
    """Run a synchronous benchmark.

    Args:
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    start_total = time.perf_counter()

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    total_time = time.perf_counter() - start_total

    return BenchmarkResult(
        name=func.__name__,
        iterations=iterations,
        total_time=total_time,
        avg_time=sum(times) / len(times),
        min_time=min(times),
        max_time=max(times),
        throughput=iterations / total_time,
        metadata={},
    )


async def run_async_benchmark(func, iterations: int, warmup: int = 0) -> BenchmarkResult:
    """Run an asynchronous benchmark.

    Args:
        func: Async function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        await func()

    # Benchmark
    times = []
    start_total = time.perf_counter()

    for _ in range(iterations):
        start = time.perf_counter()
        await func()
        end = time.perf_counter()
        times.append(end - start)

    total_time = time.perf_counter() - start_total

    return BenchmarkResult(
        name=func.__name__,
        iterations=iterations,
        total_time=total_time,
        avg_time=sum(times) / len(times),
        min_time=min(times),
        max_time=max(times),
        throughput=iterations / total_time,
        metadata={},
    )


def print_benchmark_results(results: List[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table.

    Args:
        results: List of benchmark results
    """
    print("\n" + "=" * 100)
    print(f"{'Benchmark':<40} {'Avg Time (ms)':<15} {'Throughput':<15} {'Speedup':<10}")
    print("=" * 100)

    baseline = results[0] if results else None

    for result in results:
        avg_ms = result.avg_time * 1000
        throughput = result.throughput

        if baseline and result != baseline:
            speedup = result.speedup(baseline)
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "1.00x (baseline)"

        print(f"{result.name:<40} {avg_ms:<15.4f} {throughput:<15.2f} {speedup_str:<10}")

    print("=" * 100 + "\n")


def save_benchmark_results(results: List[BenchmarkResult], path: Path) -> None:
    """Save benchmark results to JSON file.

    Args:
        results: List of benchmark results
        path: Path to save results
    """
    data = {
        "timestamp": time.time(),
        "results": [r.to_dict() for r in results],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Benchmark results saved to {path}")


# =============================================================================
# JSON Serialization Benchmarks
# =============================================================================


class TestJSONSerializationBenchmarks:
    """Benchmarks for JSON serialization optimizations."""

    @staticmethod
    def create_test_data(size: str = "medium") -> Dict[str, Any]:
        """Create test data for serialization.

        Args:
            size: Size of test data (small, medium, large)

        Returns:
            Test data dictionary
        """
        if size == "small":
            return {"key": "value", "number": 42, "list": [1, 2, 3]}

        elif size == "medium":
            return {
                "users": [
                    {
                        "id": i,
                        "name": f"User {i}",
                        "email": f"user{i}@example.com",
                        "metadata": {"created": "2024-01-01", "active": True},
                    }
                    for i in range(100)
                ]
            }

        else:  # large
            return {
                "data": [
                    {
                        "id": i,
                        "values": list(range(100)),
                        "nested": {
                            "level1": {"level2": {"level3": [f"data_{i}_{j}" for j in range(100)]}}
                        },
                    }
                    for i in range(1000)
                ]
            }

    @pytest.mark.benchmark
    def test_json_serialize_standard(self):
        """Benchmark standard json serialization."""
        data = self.create_test_data("medium")

        def serialize():
            return json.dumps(data)

        result = run_benchmark(serialize, iterations=1000, warmup=10)
        print(f"Standard JSON serialization: {result.avg_time * 1000:.4f}ms")
        return result

    @pytest.mark.benchmark
    def test_json_serialize_optimized(self):
        """Benchmark optimized json serialization (orjson)."""
        data = self.create_test_data("medium")

        def serialize():
            return json_dumps(data)

        result = run_benchmark(serialize, iterations=1000, warmup=10)
        print(f"Optimized JSON serialization: {result.avg_time * 1000:.4f}ms")
        return result

    @pytest.mark.benchmark
    def test_json_deserialize_standard(self):
        """Benchmark standard json deserialization."""
        data = self.create_test_data("medium")
        serialized = json.dumps(data)

        def deserialize():
            return json.loads(serialized)

        result = run_benchmark(deserialize, iterations=1000, warmup=10)
        print(f"Standard JSON deserialization: {result.avg_time * 1000:.4f}ms")
        return result

    @pytest.mark.benchmark
    def test_json_deserialize_optimized(self):
        """Benchmark optimized json deserialization (orjson)."""
        data = self.create_test_data("medium")
        serialized = json_dumps(data)

        def deserialize():
            return json_loads(serialized)

        result = run_benchmark(deserialize, iterations=1000, warmup=10)
        print(f"Optimized JSON deserialization: {result.avg_time * 1000:.4f}ms")
        return result


# =============================================================================
# Response Cache Benchmarks
# =============================================================================


class TestResponseCacheBenchmarks:
    """Benchmarks for response caching."""

    @pytest.fixture
    def cache(self):
        """Create test cache."""
        return ResponseCache(
            max_size=1000,
            default_ttl=3600,
            enable_semantic=False,
        )

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, cache):
        """Benchmark cache hit performance."""
        messages = [Message(role="user", content="Test message")]
        response = CompletionResponse(content="Test response")

        # Populate cache
        await cache.put(messages, response)

        async def get_cached():
            return await cache.get(messages)

        result = await run_async_benchmark(get_cached, iterations=10000, warmup=100)
        print(f"Cache hit: {result.avg_time * 1000:.4f}ms")
        return result

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_miss_performance(self, cache):
        """Benchmark cache miss performance."""
        messages = [Message(role="user", content="Test message")]

        async def get_miss():
            return await cache.get(messages)

        result = await run_async_benchmark(get_miss, iterations=10000, warmup=100)
        print(f"Cache miss: {result.avg_time * 1000:.4f}ms")
        return result

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_write_performance(self, cache):
        """Benchmark cache write performance."""
        messages = [Message(role="user", content="Test message")]
        response = CompletionResponse(content="Test response")

        async def write_cache():
            await cache.put(messages, response)

        result = await run_async_benchmark(write_cache, iterations=1000, warmup=10)
        print(f"Cache write: {result.avg_time * 1000:.4f}ms")
        return result


# =============================================================================
# Request Batching Benchmarks
# =============================================================================


class TestRequestBatchingBenchmarks:
    """Benchmarks for request batching."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_batched_vs_sequential(self):
        """Compare batched vs sequential execution."""

        # Simulated API delay
        async def mock_api_call(request):
            await asyncio.sleep(0.01)  # 10ms delay
            return f"result_{request}"

        # Sequential execution
        async def sequential_execution():
            results = []
            for i in range(50):
                result = await mock_api_call(i)
                results.append(result)
            return results

        # Batched execution
        batcher = RequestBatcher(
            key_func=lambda req: "default",
            batch_func=lambda entries: asyncio.sleep(0.01, [f"result_{e}" for e in entries]),
            max_batch_size=10,
            batch_timeout=0.05,
        )
        await batcher.start()

        try:

            async def batched_execution():
                tasks = [batcher.submit(req=i) for i in range(50)]
                return await asyncio.gather(*tasks)

            # Benchmark sequential
            seq_result = await run_async_benchmark(sequential_execution, iterations=10, warmup=1)

            # Benchmark batched
            batch_result = await run_async_benchmark(batched_execution, iterations=10, warmup=1)

            print(f"\nSequential execution: {seq_result.avg_time * 1000:.2f}ms")
            print(f"Batched execution: {batch_result.avg_time * 1000:.2f}ms")
            print(f"Speedup: {seq_result.speedup(batch_result):.2f}x")

            return [seq_result, batch_result]

        finally:
            await batcher.stop()


# =============================================================================
# Overall Performance Benchmark
# =============================================================================


class TestOverallPerformance:
    """Overall performance benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_end_to_end_scenario(self):
        """Benchmark end-to-end scenario with all optimizations."""

        # Simulate realistic workload
        cache = ResponseCache(max_size=100, enable_semantic=False)

        # Create batcher
        batcher = RequestBatcher(
            key_func=lambda **kwargs: kwargs.get("operation"),
            batch_func=lambda entries: asyncio.sleep(
                0.01, [f"result_{i}" for i in range(len(entries))]
            ),
            max_batch_size=10,
            batch_timeout=0.05,
        )
        await batcher.start()

        try:
            # Simulate workload with cache and batching
            # Use a counter to track iterations
            counter = 0

            async def simulated_workload():
                nonlocal counter
                # Try cache first (70% hit rate)
                messages = [Message(role="user", content=f"Query {counter % 10}")]
                cached = await cache.get(messages)

                if cached is None:
                    # Cache miss - use batcher
                    result = await batcher.submit(operation="query", id=counter)
                    response = CompletionResponse(content=result)
                    await cache.put(messages, response)
                    counter += 1
                    return result
                else:
                    counter += 1
                    return cached.content

            # Benchmark
            result = await run_async_benchmark(simulated_workload, iterations=1000, warmup=10)

            print(f"End-to-end workload: {result.avg_time * 1000:.4f}ms")
            print(f"Throughput: {result.throughput:.2f} ops/sec")

            # Get cache stats
            cache_stats = cache.get_stats()
            print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")

            # Get batcher stats
            batcher_stats = batcher.get_stats()
            print(f"Average batch size: {batcher_stats.get('avg_batch_size', 0):.2f}")

            return result

        finally:
            await batcher.stop()


# =============================================================================
# Main Benchmark Runner
# =============================================================================


if __name__ == "__main__":
    """Run all benchmarks and generate report."""
    import sys

    print("Running performance optimization benchmarks...\n")

    all_results = []

    # JSON benchmarks
    print("Running JSON serialization benchmarks...")
    json_bench = TestJSONSerializationBenchmarks()
    all_results.append(json_bench.test_json_serialize_standard())
    all_results.append(json_bench.test_json_serialize_optimized())

    # Run async benchmarks
    async def run_async_benchmarks():
        # Cache benchmarks
        print("\nRunning cache benchmarks...")
        cache_bench = TestResponseCacheBenchmarks()
        cache = ResponseCache(max_size=1000, enable_semantic=False)
        all_results.append(await cache_bench.test_cache_hit_performance(cache))
        all_results.append(await cache_bench.test_cache_miss_performance(cache))

        # Batching benchmarks
        print("\nRunning batching benchmarks...")
        batch_bench = TestRequestBatchingBenchmarks()
        batch_results = await batch_bench.test_batched_vs_sequential()
        all_results.extend(batch_results)

        # Overall benchmark
        print("\nRunning overall performance benchmark...")
        overall = TestOverallPerformance()
        all_results.append(await overall.test_end_to_end_scenario())

    asyncio.run(run_async_benchmarks())

    # Print results
    print_benchmark_results(all_results)

    # Save results
    output_path = Path(__file__).parent / "benchmark_results.json"
    save_benchmark_results(all_results, output_path)
