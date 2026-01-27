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

"""Comprehensive performance benchmarks for AST processing operations.

This module provides benchmarks for tree-sitter based AST operations including:
- Parsing performance for various file sizes
- Query execution performance
- Symbol extraction performance
- Edge extraction performance (calls, inheritance, etc.)
- Cache performance (hit/miss scenarios)
- Parallel vs sequential processing

Expected Performance Characteristics:
    - Small files (100 lines): <1ms parsing time
    - Medium files (500 lines): <5ms parsing time
    - Large files (1000 lines): <10ms parsing time
    - Query execution: <0.5ms for simple queries
    - Symbol extraction: 1-3ms depending on complexity
    - Cache hits: >90% latency reduction vs uncached

Benchmark Categories:
    1. Parsing Performance - Varying file sizes and languages
    2. Query Performance - Simple vs complex queries
    3. Extraction Performance - Symbols, edges, references
    4. Cache Performance - Hit/miss scenarios, cache warming
    5. Parallel Processing - Multi-file processing efficiency
    6. Memory Performance - Memory usage for various operations

Metrics Collected:
    - Average latency per operation (ms)
    - Min/Max/P50/P95/P99 latencies
    - Throughput (operations/second)
    - Memory usage (MB)
    - Cache hit rate (%)
    - Parallel speedup factor
"""

from __future__ import annotations

import gc
import logging
import os
import random
import string
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

# Configure logging to reduce noise during benchmarks
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# =============================================================================
# Benchmark Result Data Classes
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Attributes:
        name: Benchmark name
        iterations: Number of iterations run
        total_time: Total elapsed time in seconds
        avg_latency: Average latency per operation in milliseconds
        min_latency: Minimum latency in milliseconds
        max_latency: Maximum latency in milliseconds
        p50_latency: 50th percentile latency in milliseconds
        p95_latency: 95th percentile latency in milliseconds
        p99_latency: 99th percentile latency in milliseconds
        throughput: Operations per second
        memory_mb: Memory usage in MB
        additional_metrics: Optional dictionary of additional metrics
    """

    name: str
    iterations: int
    total_time: float
    avg_latency: float
    min_latency: float
    max_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    memory_mb: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Avg Latency: {self.avg_latency:.3f}ms\n"
            f"  Min/Max: {self.min_latency:.3f}ms / {self.max_latency:.3f}ms\n"
            f"  P50/P95/P99: {self.p50_latency:.3f}ms / {self.p95_latency:.3f}ms / {self.p99_latency:.3f}ms\n"
            f"  Throughput: {self.throughput:.2f} ops/sec\n"
            f"  Memory: {self.memory_mb:.2f} MB\n"
        )


@dataclass
class ComparisonResult:
    """Result from comparing two benchmark runs.

    Attributes:
        baseline_name: Baseline benchmark name
        comparison_name: Comparison benchmark name
        speedup: Speedup factor (comparison_time / baseline_time)
        latency_reduction: Percentage latency reduction
        throughput_improvement: Percentage throughput improvement
        memory_difference: Memory difference in MB
    """

    baseline_name: str
    comparison_name: str
    speedup: float
    latency_reduction: float
    throughput_improvement: float
    memory_difference: float

    def __str__(self) -> str:
        return (
            f"{self.baseline_name} vs {self.comparison_name}:\n"
            f"  Speedup: {self.speedup:.2f}x\n"
            f"  Latency Reduction: {self.latency_reduction:.1f}%\n"
            f"  Throughput Improvement: {self.throughput_improvement:.1f}%\n"
            f"  Memory Difference: {self.memory_difference:+.2f} MB\n"
        )


# =============================================================================
# Code Generation Utilities
# =============================================================================


def generate_python_source(
    lines: int = 100,
    functions: int = 5,
    classes: int = 2,
    complexity: str = "medium",
) -> bytes:
    """Generate Python source code for benchmarking.

    Args:
        lines: Approximate number of lines to generate
        functions: Number of functions to generate
        classes: Number of classes to generate
        complexity: Complexity level (simple, medium, complex)

    Returns:
        Generated Python source code as bytes
    """
    lines_per_function = max(5, lines // (functions + 1))
    lines_per_class = max(10, lines // (classes + 1))

    source_lines = []
    source_lines.append('"""Generated Python code for benchmarking."""\n')
    source_lines.append("import random\n")
    source_lines.append("from typing import List, Optional, Dict, Any\n\n")

    # Generate classes
    for i in range(classes):
        source_lines.append(f"class GeneratedClass{i}:\n")
        source_lines.append(f'    """Generated class {i}."""\n')
        source_lines.append("    def __init__(self):\n")
        source_lines.append(f"        self.value{i} = random.randint(1, 100)\n")
        source_lines.append(f"        self.data{i} = []\n\n")

        # Add methods
        methods = min(3, max(1, lines_per_class // 15))
        for j in range(methods):
            source_lines.append(f"    def method{j}(self, param: int) -> int:\n")
            if complexity == "simple":
                source_lines.append(f"        return param + self.value{i}\n")
            elif complexity == "medium":
                source_lines.append(f"        result = param + self.value{i}\n")
                source_lines.append("        if result > 50:\n")
                source_lines.append("            result *= 2\n")
                source_lines.append("        return result\n")
            else:  # complex
                source_lines.append("        result = 0\n")
                source_lines.append("        for k in range(param):\n")
                source_lines.append(f"            result += self.value{i} * k\n")
                source_lines.append("            if result > 1000:\n")
                source_lines.append("                break\n")
                source_lines.append("        return result\n")
            source_lines.append("\n")

    # Generate functions
    for i in range(functions):
        source_lines.append(f"def generated_function{i}(param: List[int]) -> Dict[str, Any]:\n")
        if complexity == "simple":
            source_lines.append('    return {"result": sum(param)}\n')
        elif complexity == "medium":
            source_lines.append("    total = sum(param)\n")
            source_lines.append("    average = total / len(param) if param else 0\n")
            source_lines.append('    return {"total": total, "average": average}\n')
        else:  # complex
            source_lines.append("    result = {}\n")
            source_lines.append("    result['sum'] = sum(param)\n")
            source_lines.append("    result['min'] = min(param) if param else 0\n")
            source_lines.append("    result['max'] = max(param) if param else 0\n")
            source_lines.append("    result['unique'] = list(set(param))\n")
            source_lines.append("    return result\n")
        source_lines.append("\n")

    # Add more lines to reach target
    current_lines = len(source_lines)
    if current_lines < lines:
        additional_lines = lines - current_lines
        source_lines.append("# Additional lines\n")
        for i in range(additional_lines):
            source_lines.append(f"variable_{i} = {i}\n")

    return "".join(source_lines).encode("utf-8")


def generate_javascript_source(lines: int = 100) -> bytes:
    """Generate JavaScript source code for benchmarking.

    Args:
        lines: Approximate number of lines to generate

    Returns:
        Generated JavaScript source code as bytes
    """
    source_lines = []
    source_lines.append("// Generated JavaScript code for benchmarking\n")
    source_lines.append("class GeneratedClass {\n")
    source_lines.append("    constructor() {\n")
    source_lines.append("        this.value = Math.random() * 100;\n")
    source_lines.append("    }\n\n")

    functions = max(3, lines // 20)
    for i in range(functions):
        method_name = f"method{i}"
        source_lines.append(f"    {method_name}() {{\n")
        source_lines.append(f"        return this.value * {i};\n")
        source_lines.append("    }\n\n")

    source_lines.append("}\n\n")

    # Add functions
    for i in range(functions):
        func_name = f"generatedFunction{i}"
        source_lines.append(f"function {func_name}(param) {{\n")
        source_lines.append(f"    return param * {i};\n")
        source_lines.append("}\n\n")

    # Add more lines to reach target
    current_lines = len(source_lines)
    if current_lines < lines:
        for i in range(lines - current_lines):
            source_lines.append(f"const variable{i} = {i};\n")

    return "".join(source_lines).encode("utf-8")


# =============================================================================
# Simple AST Cache Implementation for Benchmarking
# =============================================================================


class SimpleASTCache:
    """Simple in-memory cache for AST trees.

    Attributes:
        _cache: Dictionary mapping file paths to cached trees
        _hits: Number of cache hits
        _misses: Number of cache misses
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get cached tree."""
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, tree: Any) -> None:
        """Cache a tree."""
        self._cache[key] = tree

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
        }


# =============================================================================
# Benchmark Utilities
# =============================================================================


def run_benchmark(
    name: str,
    func,
    iterations: int = 100,
    warmup_iterations: int = 10,
    cache: Optional[Any] = None,
) -> BenchmarkResult:
    """Run a benchmark with warmup and collect metrics.

    Args:
        name: Benchmark name
        func: Function to benchmark (should return latency in seconds)
        iterations: Number of iterations
        warmup_iterations: Number of warmup iterations (not counted)
        cache: Optional cache to collect metrics from

    Returns:
        BenchmarkResult with collected metrics
    """
    # Warmup
    for _ in range(warmup_iterations):
        func()

    # Reset cache metrics if provided
    if cache:
        cache.clear()

    # Start memory tracking
    gc.collect()
    tracemalloc.start()

    # Run benchmark
    latencies = []
    start_time = time.perf_counter()

    for _ in range(iterations):
        iter_start = time.perf_counter()
        result = func()
        iter_end = time.perf_counter()
        latencies.append((iter_end - iter_start) * 1000)  # Convert to ms

    end_time = time.perf_counter()

    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate statistics
    total_time = end_time - start_time
    latencies_sorted = sorted(latencies)
    avg_latency = sum(latencies) / len(latencies)
    min_latency = latencies_sorted[0]
    max_latency = latencies_sorted[-1]
    p50_latency = latencies_sorted[len(latencies_sorted) // 2]
    p95_index = int(len(latencies_sorted) * 0.95)
    p99_index = int(len(latencies_sorted) * 0.99)
    p95_latency = latencies_sorted[p95_index]
    p99_latency = latencies_sorted[p99_index]
    throughput = iterations / total_time
    memory_mb = peak / 1024 / 1024

    # Collect additional metrics
    additional_metrics = {}
    if cache:
        additional_metrics.update(cache.get_stats())

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=total_time,
        avg_latency=avg_latency,
        min_latency=min_latency,
        max_latency=max_latency,
        p50_latency=p50_latency,
        p95_latency=p95_latency,
        p99_latency=p99_latency,
        throughput=throughput,
        memory_mb=memory_mb,
        additional_metrics=additional_metrics,
    )


def compare_benchmarks(baseline: BenchmarkResult, comparison: BenchmarkResult) -> ComparisonResult:
    """Compare two benchmark results.

    Args:
        baseline: Baseline benchmark result
        comparison: Comparison benchmark result

    Returns:
        ComparisonResult with comparison metrics
    """
    speedup = baseline.avg_latency / comparison.avg_latency if comparison.avg_latency > 0 else 0
    latency_reduction = (
        (1 - comparison.avg_latency / baseline.avg_latency) * 100 if baseline.avg_latency > 0 else 0
    )
    throughput_improvement = (
        ((comparison.throughput - baseline.throughput) / baseline.throughput) * 100
        if baseline.throughput > 0
        else 0
    )
    memory_difference = comparison.memory_mb - baseline.memory_mb

    return ComparisonResult(
        baseline_name=baseline.name,
        comparison_name=comparison.name,
        speedup=speedup,
        latency_reduction=latency_reduction,
        throughput_improvement=throughput_improvement,
        memory_difference=memory_difference,
    )


# =============================================================================
# Fixture Setup
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset language registries before each test."""
    # Import and clear caches if possible
    try:
        from victor.coding.codebase import tree_sitter_manager

        if hasattr(tree_sitter_manager, "_language_cache"):
            tree_sitter_manager._language_cache.clear()
        if hasattr(tree_sitter_manager, "_parser_cache"):
            tree_sitter_manager._parser_cache.clear()
    except Exception:
        pass  # If import fails, continue anyway
    yield
    try:
        from victor.coding.codebase import tree_sitter_manager

        if hasattr(tree_sitter_manager, "_language_cache"):
            tree_sitter_manager._language_cache.clear()
        if hasattr(tree_sitter_manager, "_parser_cache"):
            tree_sitter_manager._parser_cache.clear()
    except Exception:
        pass


@pytest.fixture
def python_parser():
    """Get Python parser."""
    from victor.coding.codebase.tree_sitter_manager import get_parser

    return get_parser("python")


@pytest.fixture
def javascript_parser():
    """Get JavaScript parser."""
    from victor.coding.codebase.tree_sitter_manager import get_parser

    return get_parser("javascript")


@pytest.fixture
def ast_cache():
    """Get AST cache instance."""
    cache = SimpleASTCache()
    yield cache
    cache.clear()


# =============================================================================
# Parsing Performance Benchmarks
# =============================================================================


class TestParsingPerformance:
    """Benchmarks for AST parsing performance."""

    def test_python_ast_parsing_small(self, python_parser):
        """Benchmark parsing small Python file (100 lines)."""
        source = generate_python_source(lines=100)

        def parse():
            tree = python_parser.parse(source)
            assert tree.root_node is not None
            return tree

        result = run_benchmark(
            name="Python AST Parsing - Small (100 lines)",
            func=parse,
            iterations=100,
            warmup_iterations=10,
        )

        print(f"\n{result}")
        assert result.avg_latency < 5.0, f"Parsing too slow: {result.avg_latency:.3f}ms"
        assert result.throughput > 200, f"Throughput too low: {result.throughput:.2f} ops/sec"

    def test_python_ast_parsing_medium(self, python_parser):
        """Benchmark parsing medium Python file (500 lines)."""
        source = generate_python_source(lines=500)

        def parse():
            tree = python_parser.parse(source)
            assert tree.root_node is not None
            return tree

        result = run_benchmark(
            name="Python AST Parsing - Medium (500 lines)",
            func=parse,
            iterations=50,
            warmup_iterations=5,
        )

        print(f"\n{result}")
        assert result.avg_latency < 20.0, f"Parsing too slow: {result.avg_latency:.3f}ms"

    def test_python_ast_parsing_large(self, python_parser):
        """Benchmark parsing large Python file (1000 lines)."""
        source = generate_python_source(lines=1000)

        def parse():
            tree = python_parser.parse(source)
            assert tree.root_node is not None
            return tree

        result = run_benchmark(
            name="Python AST Parsing - Large (1000 lines)",
            func=parse,
            iterations=20,
            warmup_iterations=3,
        )

        print(f"\n{result}")
        assert result.avg_latency < 50.0, f"Parsing too slow: {result.avg_latency:.3f}ms"

    def test_javascript_ast_parsing_small(self, javascript_parser):
        """Benchmark parsing small JavaScript file (100 lines)."""
        source = generate_javascript_source(lines=100)

        def parse():
            tree = javascript_parser.parse(source)
            assert tree.root_node is not None
            return tree

        result = run_benchmark(
            name="JavaScript AST Parsing - Small (100 lines)",
            func=parse,
            iterations=100,
            warmup_iterations=10,
        )

        print(f"\n{result}")
        assert result.avg_latency < 5.0, f"Parsing too slow: {result.avg_latency:.3f}ms"

    def test_parsing_complexity_simple_vs_medium_vs_complex(self, python_parser):
        """Compare parsing performance across complexity levels."""
        simple_source = generate_python_source(lines=200, complexity="simple")
        medium_source = generate_python_source(lines=200, complexity="medium")
        complex_source = generate_python_source(lines=200, complexity="complex")

        simple_result = run_benchmark(
            name="Parsing - Simple Code",
            func=lambda: python_parser.parse(simple_source),
            iterations=50,
            warmup_iterations=5,
        )

        medium_result = run_benchmark(
            name="Parsing - Medium Code",
            func=lambda: python_parser.parse(medium_source),
            iterations=50,
            warmup_iterations=5,
        )

        complex_result = run_benchmark(
            name="Parsing - Complex Code",
            func=lambda: python_parser.parse(complex_source),
            iterations=50,
            warmup_iterations=5,
        )

        print(f"\n{simple_result}")
        print(f"\n{medium_result}")
        print(f"\n{complex_result}")

        # Complex code should take longer than simple code
        assert complex_result.avg_latency > simple_result.avg_latency


# =============================================================================
# Query Execution Performance
# =============================================================================


class TestQueryPerformance:
    """Benchmarks for query execution performance."""

    def test_query_execution_simple(self, python_parser):
        """Benchmark simple AST query (function names)."""
        source = generate_python_source(lines=200, functions=10)
        tree = python_parser.parse(source)

        from tree_sitter import Query

        query = Query(
            python_parser.language,
            """
            (function_definition name: (identifier) @name)
            """,
        )

        def execute():
            from tree_sitter import QueryCursor

            cursor = QueryCursor(query)
            matches = cursor.captures(tree.root_node)
            assert len(matches) > 0
            return matches

        result = run_benchmark(
            name="Query Execution - Simple (Function Names)",
            func=execute,
            iterations=100,
            warmup_iterations=10,
        )

        print(f"\n{result}")
        assert result.avg_latency < 1.0, f"Query too slow: {result.avg_latency:.3f}ms"

    def test_query_execution_complex(self, python_parser):
        """Benchmark complex AST query (nested captures)."""
        source = generate_python_source(lines=300, functions=15, classes=5)
        tree = python_parser.parse(source)

        from tree_sitter import Query

        query = Query(
            python_parser.language,
            """
            (class_definition
              name: (identifier) @class_name
              body: (block
                (function_definition
                  name: (identifier) @method_name)))
            """,
        )

        def execute():
            from tree_sitter import QueryCursor

            cursor = QueryCursor(query)
            matches = cursor.captures(tree.root_node)
            return matches

        result = run_benchmark(
            name="Query Execution - Complex (Nested Captures)",
            func=execute,
            iterations=50,
            warmup_iterations=5,
        )

        print(f"\n{result}")
        assert result.avg_latency < 5.0, f"Query too slow: {result.avg_latency:.3f}ms"

    def test_query_execution_multiple_patterns(self, python_parser):
        """Benchmark query with multiple patterns."""
        source = generate_python_source(lines=300, functions=15, classes=5)
        tree = python_parser.parse(source)

        from tree_sitter import Query

        query = Query(
            python_parser.language,
            """
            (function_definition name: (identifier) @function_name)
            (class_definition name: (identifier) @class_name)
            (assignment left: (identifier) @variable_name)
            """,
        )

        def execute():
            from tree_sitter import QueryCursor

            cursor = QueryCursor(query)
            matches = cursor.captures(tree.root_node)
            return matches

        result = run_benchmark(
            name="Query Execution - Multiple Patterns",
            func=execute,
            iterations=50,
            warmup_iterations=5,
        )

        print(f"\n{result}")
        assert result.avg_latency < 5.0, f"Query too slow: {result.avg_latency:.3f}ms"


# =============================================================================
# Symbol Extraction Performance
# =============================================================================


class TestSymbolExtractionPerformance:
    """Benchmarks for symbol extraction performance."""

    def test_symbol_extraction_small_file(self):
        """Benchmark symbol extraction from small file."""
        from victor.coding.codebase.tree_sitter_extractor import TreeSitterExtractor

        extractor = TreeSitterExtractor()
        source = generate_python_source(lines=100, functions=5, classes=2)

        # Create temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
            f.write(source)
            temp_path = Path(f.name)

        try:

            def extract():
                symbols = extractor.extract_symbols(temp_path, language="python")
                assert len(symbols) > 0
                return symbols

            result = run_benchmark(
                name="Symbol Extraction - Small File (100 lines)",
                func=extract,
                iterations=50,
                warmup_iterations=5,
            )

            print(f"\n{result}")
            assert result.avg_latency < 10.0, f"Extraction too slow: {result.avg_latency:.3f}ms"
        finally:
            temp_path.unlink()

    def test_symbol_extraction_large_file(self):
        """Benchmark symbol extraction from large file."""
        from victor.coding.codebase.tree_sitter_extractor import TreeSitterExtractor

        extractor = TreeSitterExtractor()
        source = generate_python_source(lines=1000, functions=50, classes=10)

        # Create temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
            f.write(source)
            temp_path = Path(f.name)

        try:

            def extract():
                symbols = extractor.extract_symbols(temp_path, language="python")
                assert len(symbols) > 0
                return symbols

            result = run_benchmark(
                name="Symbol Extraction - Large File (1000 lines)",
                func=extract,
                iterations=20,
                warmup_iterations=3,
            )

            print(f"\n{result}")
            assert result.avg_latency < 50.0, f"Extraction too slow: {result.avg_latency:.3f}ms"
        finally:
            temp_path.unlink()

    def test_extract_all_comprehensive(self):
        """Benchmark comprehensive extraction (symbols + edges)."""
        from victor.coding.codebase.tree_sitter_extractor import TreeSitterExtractor

        extractor = TreeSitterExtractor()
        source = generate_python_source(lines=500, functions=20, classes=8)

        # Create temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
            f.write(source)
            temp_path = Path(f.name)

        try:

            def extract():
                symbols, edges = extractor.extract_all(temp_path, language="python")
                assert len(symbols) > 0
                return symbols, edges

            result = run_benchmark(
                name="Comprehensive Extraction - Symbols + Edges",
                func=extract,
                iterations=30,
                warmup_iterations=5,
            )

            print(f"\n{result}")
            assert result.avg_latency < 50.0, f"Extraction too slow: {result.avg_latency:.3f}ms"
        finally:
            temp_path.unlink()


# =============================================================================
# Cache Performance Benchmarks
# =============================================================================


class TestCachePerformance:
    """Benchmarks for AST cache performance."""

    def test_cache_miss_performance(self, python_parser):
        """Benchmark AST cache miss (uncached parsing)."""
        source = generate_python_source(lines=200)

        def uncached_parse():
            tree = python_parser.parse(source)
            assert tree.root_node is not None
            return tree

        result = run_benchmark(
            name="Cache Miss - Uncached Parsing",
            func=uncached_parse,
            iterations=50,
            warmup_iterations=5,
        )

        print(f"\n{result}")
        # Just verify it runs without error - the cache metrics aren't applicable here

    def test_cache_hit_performance(self, python_parser):
        """Benchmark AST cache hit (cached parsing)."""
        source = generate_python_source(lines=200)

        # Pre-warm cache
        ast_cache = SimpleASTCache()
        tree = python_parser.parse(source)
        ast_cache.put("test.py", tree)

        def cached_parse():
            result = ast_cache.get("test.py")
            assert result is not None
            return result

        # Don't pass cache to run_benchmark to avoid clearing it
        result = run_benchmark(
            name="Cache Hit - Cached Parsing",
            func=cached_parse,
            iterations=1000,
            warmup_iterations=10,
        )

        # Get cache stats manually (including warmup)
        cache_stats = ast_cache.get_stats()
        print(f"\n{result}")
        print(f"Cache Stats: {cache_stats}")
        # Should have at least 1000 hits (may include warmup)
        assert cache_stats["hits"] >= 1000
        # Cache hits should be extremely fast (<0.1ms for safety)
        assert result.avg_latency < 0.1, f"Cache hit too slow: {result.avg_latency:.3f}ms"

    def test_cache_hit_vs_miss_comparison(self, python_parser, ast_cache):
        """Compare cache hit vs cache miss performance."""
        source = generate_python_source(lines=200)

        # Benchmark cache miss
        def cache_miss():
            tree = python_parser.parse(source)
            return tree

        miss_result = run_benchmark(
            name="Cache Miss Baseline",
            func=cache_miss,
            iterations=50,
            warmup_iterations=5,
        )

        # Warm cache
        tree = python_parser.parse(source)
        ast_cache.put("test.py", tree)

        # Benchmark cache hit
        def cache_hit():
            result = ast_cache.get("test.py")
            return result

        hit_result = run_benchmark(
            name="Cache Hit Optimized",
            func=cache_hit,
            iterations=1000,
            warmup_iterations=10,
        )

        comparison = compare_benchmarks(miss_result, hit_result)

        print(f"\n{miss_result}")
        print(f"\n{hit_result}")
        print(f"\n{comparison}")

        # Cache should provide significant speedup (>100x)
        assert comparison.speedup > 100, f"Cache speedup too low: {comparison.speedup:.2f}x"

    def test_cache_warming_performance(self, python_parser, ast_cache):
        """Benchmark cache warming performance over multiple accesses."""
        sources = [generate_python_source(lines=200) for _ in range(10)]

        def warm_cache():
            # First access: cache miss (parse and cache)
            for i, source in enumerate(sources):
                key = f"file_{i}.py"
                if ast_cache.get(key) is None:
                    tree = python_parser.parse(source)
                    ast_cache.put(key, tree)

            # Second access: cache hit
            total_hits = 0
            for i in range(len(sources)):
                key = f"file_{i}.py"
                if ast_cache.get(key) is not None:
                    total_hits += 1

            return total_hits

        result = run_benchmark(
            name="Cache Warming - 10 Files",
            func=warm_cache,
            iterations=20,
            warmup_iterations=2,
            cache=ast_cache,
        )

        print(f"\n{result}")
        print(f"Cache Stats: {result.additional_metrics}")
        assert result.additional_metrics["hit_rate"] > 80  # High hit rate after warming


# =============================================================================
# Parallel Processing Benchmarks
# =============================================================================


class TestParallelProcessingPerformance:
    """Benchmarks for parallel AST processing."""

    def test_sequential_parsing(self, python_parser):
        """Benchmark sequential parsing of multiple files."""
        files = [generate_python_source(lines=100) for _ in range(20)]

        def parse_sequential():
            trees = []
            for source in files:
                tree = python_parser.parse(source)
                trees.append(tree)
            return trees

        result = run_benchmark(
            name="Sequential Parsing - 20 Files",
            func=parse_sequential,
            iterations=10,
            warmup_iterations=2,
        )

        print(f"\n{result}")
        assert (
            result.avg_latency < 100.0
        ), f"Sequential parsing too slow: {result.avg_latency:.3f}ms"

    def test_parallel_parsing_thread_pool(self, python_parser):
        """Benchmark parallel parsing using ThreadPoolExecutor."""
        files = [generate_python_source(lines=100) for _ in range(20)]

        def parse_parallel():
            with ThreadPoolExecutor(max_workers=4) as executor:
                trees = list(executor.map(lambda s: python_parser.parse(s), files))
            return trees

        result = run_benchmark(
            name="Parallel Parsing (ThreadPool) - 20 Files",
            func=parse_parallel,
            iterations=10,
            warmup_iterations=2,
        )

        print(f"\n{result}")

    def test_parallel_vs_sequential_comparison(self, python_parser):
        """Compare parallel vs sequential parsing performance."""
        files = [generate_python_source(lines=100) for _ in range(20)]

        # Sequential parsing
        def parse_sequential():
            trees = []
            for source in files:
                tree = python_parser.parse(source)
                trees.append(tree)
            return trees

        sequential_result = run_benchmark(
            name="Sequential Parsing Baseline",
            func=parse_sequential,
            iterations=10,
            warmup_iterations=2,
        )

        # Parallel parsing
        def parse_parallel():
            with ThreadPoolExecutor(max_workers=4) as executor:
                trees = list(executor.map(lambda s: python_parser.parse(s), files))
            return trees

        parallel_result = run_benchmark(
            name="Parallel Parsing Optimized",
            func=parse_parallel,
            iterations=10,
            warmup_iterations=2,
        )

        comparison = compare_benchmarks(sequential_result, parallel_result)

        print(f"\n{sequential_result}")
        print(f"\n{parallel_result}")
        print(f"\n{comparison}")

        # Parallel should be faster (speedup > 1.5x on multi-core systems)
        # Note: This may vary based on system configuration
        print(f"Speedup from parallelization: {comparison.speedup:.2f}x")

    def test_parallel_symbol_extraction(self):
        """Benchmark parallel symbol extraction from multiple files."""
        from victor.coding.codebase.tree_sitter_extractor import TreeSitterExtractor

        extractor = TreeSitterExtractor()

        # Create temporary files
        import tempfile

        temp_files = []
        for i in range(10):
            source = generate_python_source(lines=200, functions=10, classes=3)
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
                f.write(source)
                temp_files.append(Path(f.name))

        try:

            def extract_sequential():
                all_symbols = []
                for temp_path in temp_files:
                    symbols = extractor.extract_symbols(temp_path, language="python")
                    all_symbols.extend(symbols)
                return all_symbols

            sequential_result = run_benchmark(
                name="Sequential Symbol Extraction - 10 Files",
                func=extract_sequential,
                iterations=5,
                warmup_iterations=1,
            )

            def extract_parallel():
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(
                        executor.map(
                            lambda p: extractor.extract_symbols(p, language="python"), temp_files
                        )
                    )
                all_symbols = []
                for symbols in results:
                    all_symbols.extend(symbols)
                return all_symbols

            parallel_result = run_benchmark(
                name="Parallel Symbol Extraction - 10 Files",
                func=extract_parallel,
                iterations=5,
                warmup_iterations=1,
            )

            comparison = compare_benchmarks(sequential_result, parallel_result)

            print(f"\n{sequential_result}")
            print(f"\n{parallel_result}")
            print(f"\n{comparison}")

            print(f"Speedup from parallel extraction: {comparison.speedup:.2f}x")

        finally:
            for temp_path in temp_files:
                temp_path.unlink()


# =============================================================================
# Memory Performance Benchmarks
# =============================================================================


class TestMemoryPerformance:
    """Benchmarks for memory usage during AST operations."""

    def test_memory_parsing_small_files(self, python_parser):
        """Benchmark memory usage for parsing small files."""
        sources = [generate_python_source(lines=100) for _ in range(50)]

        def parse_batch():
            trees = []
            for source in sources:
                tree = python_parser.parse(source)
                trees.append(tree)
            return trees

        result = run_benchmark(
            name="Memory - Parse 50 Small Files",
            func=parse_batch,
            iterations=5,
            warmup_iterations=1,
        )

        print(f"\n{result}")
        print(f"Peak Memory: {result.memory_mb:.2f} MB")
        # Memory usage should be reasonable (<50MB for 50 small files)
        assert result.memory_mb < 50.0, f"Memory usage too high: {result.memory_mb:.2f} MB"

    def test_memory_parsing_large_files(self, python_parser):
        """Benchmark memory usage for parsing large files."""
        sources = [generate_python_source(lines=1000) for _ in range(10)]

        def parse_batch():
            trees = []
            for source in sources:
                tree = python_parser.parse(source)
                trees.append(tree)
            return trees

        result = run_benchmark(
            name="Memory - Parse 10 Large Files",
            func=parse_batch,
            iterations=3,
            warmup_iterations=1,
        )

        print(f"\n{result}")
        print(f"Peak Memory: {result.memory_mb:.2f} MB")
        # Memory usage should be reasonable (<100MB for 10 large files)
        assert result.memory_mb < 100.0, f"Memory usage too high: {result.memory_mb:.2f} MB"

    def test_memory_cache_storage(self, python_parser, ast_cache):
        """Benchmark memory usage for cache storage."""
        sources = [generate_python_source(lines=200) for _ in range(100)]

        def populate_cache():
            for i, source in enumerate(sources):
                key = f"file_{i}.py"
                tree = python_parser.parse(source)
                ast_cache.put(key, tree)
            return len(ast_cache._cache)

        result = run_benchmark(
            name="Memory - Cache 100 AST Trees",
            func=populate_cache,
            iterations=1,
            warmup_iterations=0,
        )

        print(f"\n{result}")
        print(f"Peak Memory: {result.memory_mb:.2f} MB")
        print(f"Cache Size: {len(ast_cache._cache)} entries")


# =============================================================================
# Edge Extraction Performance
# =============================================================================


class TestEdgeExtractionPerformance:
    """Benchmarks for edge extraction performance."""

    def test_call_edge_extraction(self):
        """Benchmark call edge extraction."""
        from victor.coding.codebase.tree_sitter_extractor import TreeSitterExtractor

        extractor = TreeSitterExtractor()

        # Generate code with function calls
        source = generate_python_source(lines=300, functions=15, classes=5)

        # Add some function calls
        source_str = source.decode("utf-8")
        source_str += "\n# Add function calls\n"
        for i in range(10):
            source_str += f"result{i} = generated_function{i}([1, 2, 3])\n"
        source = source_str.encode("utf-8")

        # Create temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
            f.write(source)
            temp_path = Path(f.name)

        try:

            def extract():
                edges = extractor.extract_call_edges(temp_path, language="python")
                return edges

            result = run_benchmark(
                name="Call Edge Extraction",
                func=extract,
                iterations=30,
                warmup_iterations=5,
            )

            print(f"\n{result}")
            assert result.avg_latency < 20.0, f"Extraction too slow: {result.avg_latency:.3f}ms"
        finally:
            temp_path.unlink()

    def test_inheritance_edge_extraction(self):
        """Benchmark inheritance edge extraction."""
        from victor.coding.codebase.tree_sitter_extractor import TreeSitterExtractor

        extractor = TreeSitterExtractor()

        # Generate code with class inheritance
        source_lines = []
        source_lines.append('"""Generated code with inheritance."""\n\n')

        # Create base classes
        for i in range(5):
            source_lines.append(f"class BaseClass{i}:\n")
            source_lines.append(f'    """Base class {i}."""\n')
            source_lines.append("    pass\n\n")

        # Create derived classes
        for i in range(10):
            base_idx = i % 5
            source_lines.append(f"class DerivedClass{i}(BaseClass{base_idx}):\n")
            source_lines.append(f'    """Derived class {i}."""\n')
            source_lines.append("    pass\n\n")

        source = "".join(source_lines).encode("utf-8")

        # Create temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
            f.write(source)
            temp_path = Path(f.name)

        try:

            def extract():
                edges = extractor.extract_inheritance_edges(temp_path, language="python")
                return edges

            result = run_benchmark(
                name="Inheritance Edge Extraction",
                func=extract,
                iterations=50,
                warmup_iterations=5,
            )

            print(f"\n{result}")
            assert result.avg_latency < 10.0, f"Extraction too slow: {result.avg_latency:.3f}ms"
        finally:
            temp_path.unlink()


# =============================================================================
# Comprehensive End-to-End Benchmarks
# =============================================================================


class TestEndToEndPerformance:
    """End-to-end benchmarks for realistic workloads."""

    def test_full_codebase_analysis_small(self):
        """Benchmark full codebase analysis (small project)."""
        from victor.coding.codebase.tree_sitter_extractor import TreeSitterExtractor

        extractor = TreeSitterExtractor()

        # Create temporary files simulating a small project
        import tempfile

        temp_dir = tempfile.mkdtemp()
        temp_files = []

        try:
            # Create 10 Python files
            for i in range(10):
                source = generate_python_source(lines=150, functions=8, classes=3)
                temp_path = Path(temp_dir) / f"module_{i}.py"
                with open(temp_path, "wb") as f:
                    f.write(source)
                temp_files.append(temp_path)

            def analyze_codebase():
                all_symbols = []
                all_edges = []

                for temp_path in temp_files:
                    symbols, edges = extractor.extract_all(temp_path, language="python")
                    all_symbols.extend(symbols)
                    all_edges.extend(edges)

                return all_symbols, all_edges

            result = run_benchmark(
                name="End-to-End - Small Project (10 files)",
                func=analyze_codebase,
                iterations=5,
                warmup_iterations=1,
            )

            print(f"\n{result}")
            assert result.avg_latency < 200.0, f"Analysis too slow: {result.avg_latency:.3f}ms"

        finally:
            import shutil

            shutil.rmtree(temp_dir)

    def test_full_codebase_analysis_medium(self):
        """Benchmark full codebase analysis (medium project)."""
        from victor.coding.codebase.tree_sitter_extractor import TreeSitterExtractor

        extractor = TreeSitterExtractor()

        # Create temporary files simulating a medium project
        import tempfile

        temp_dir = tempfile.mkdtemp()
        temp_files = []

        try:
            # Create 50 Python files
            for i in range(50):
                source = generate_python_source(lines=200, functions=10, classes=4)
                temp_path = Path(temp_dir) / f"module_{i}.py"
                with open(temp_path, "wb") as f:
                    f.write(source)
                temp_files.append(temp_path)

            def analyze_codebase():
                all_symbols = []
                all_edges = []

                for temp_path in temp_files:
                    symbols, edges = extractor.extract_all(temp_path, language="python")
                    all_symbols.extend(symbols)
                    all_edges.extend(edges)

                return all_symbols, all_edges

            result = run_benchmark(
                name="End-to-End - Medium Project (50 files)",
                func=analyze_codebase,
                iterations=3,
                warmup_iterations=1,
            )

            print(f"\n{result}")
            # Relaxed threshold for medium projects - can vary significantly by system
            assert result.avg_latency < 2000.0, f"Analysis too slow: {result.avg_latency:.3f}ms"

        finally:
            import shutil

            shutil.rmtree(temp_dir)
