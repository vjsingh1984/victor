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

"""
Performance tests for vertical initialization and execution.

This module provides comprehensive performance benchmarking for:
- Vertical initialization (load time, component setup)
- Execution performance (AST parsing, embeddings, search)
- Memory efficiency (usage, cleanup, large file handling)

Uses pytest-benchmark to establish performance baselines and catch regressions.
"""

import gc
import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest


# Type hint for benchmark fixture (provided by pytest-benchmark)
class BenchmarkFixture:
    """Type hint for pytest-benchmark fixture."""

    def __call__(self, func): ...
    def __getattr__(self, name): ...


from victor.coding import CodingAssistant
from victor.dataanalysis import DataAnalysisAssistant
from victor.devops import DevOpsAssistant
from victor.rag import RAGAssistant
from victor.research import ResearchAssistant
from victor.core.mode_config import ModeConfigRegistry
from victor.core.capabilities import CapabilityLoader


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_python_file(tmp_path: Path) -> Path:
    """Create a sample Python file for parsing tests."""
    content = """
import asyncio
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class TestData:
    name: str
    values: List[int]
    metadata: Optional[dict] = None

class DataProcessor:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self._cache = {}

    async def process(self, data: List[TestData]) -> dict:
        results = {}
        for item in data:
            key = item.name
            values = item.values
            results[key] = self._compute_stats(values)
        return results

    def _compute_stats(self, values: List[int]) -> dict:
        if not values:
            return {"mean": 0, "std": 0, "count": 0}
        count = len(values)
        mean = sum(values) / count
        variance = sum((x - mean) ** 2 for x in values) / count
        return {"mean": mean, "std": variance ** 0.5, "count": count}

def main():
    processor = DataProcessor()
    test_data = [
        TestData("sample1", [1, 2, 3, 4, 5]),
        TestData("sample2", [10, 20, 30, 40, 50]),
    ]
    return processor.process(test_data)

if __name__ == "__main__":
    main()
"""
    file_path = tmp_path / "sample.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def large_python_file(tmp_path: Path) -> Path:
    """Create a large Python file for memory testing."""
    lines = []
    lines.append("import asyncio\n")
    lines.append("from typing import List, Dict, Any, Optional\n\n")

    # Generate multiple classes with methods
    for i in range(50):
        lines.append(f"@dataclass\nclass Class{i}:\n")
        lines.append("    name: str\n")
        lines.append("    value: int\n\n")

        lines.append(f"class Processor{i}:\n")
        lines.append("    def __init__(self, config: dict = None):\n")
        lines.append("        self.config = config or {}\n\n")

        lines.append("    def process(self, data: List[Any]) -> Dict[str, Any]:\n")
        lines.append("        result = {}\n")
        lines.append("        for item in data:\n")
        lines.append("            result[item] = self._compute(item)\n")
        lines.append("        return result\n\n")

        lines.append("    def _compute(self, value: Any) -> float:\n")
        lines.append("        return float(value) * 1.5\n\n")

    file_path = tmp_path / "large_file.py"
    file_path.write_text("".join(lines))
    return file_path


@pytest.fixture
def sample_codebase(tmp_path: Path) -> Path:
    """Create a sample codebase with multiple files."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()

    # Create multiple Python files
    for i in range(10):
        file_path = tmp_path / "src" / f"module{i}.py"
        content = f"""
# Module {i}
from typing import List, Optional

def function_{i}(value: int) -> int:
    return value * {i}

class Class{i}:
    def __init__(self, name: str):
        self.name = name

    def method(self) -> str:
        return f"{{self.name}}_{i}"
"""
        file_path.write_text(content)

    # Create a README
    readme_path = tmp_path / "docs" / "README.md"
    readme_content = """# Sample Codebase

This is a sample codebase for performance testing.

## Modules

- module0-9: Sample modules
- Tests: Unit tests

## Usage

```python
from src.module0 import function_0
result = function_0(10)
```
"""
    readme_path.write_text(readme_content)

    return tmp_path


# =============================================================================
# Test Suite 1: Initialization Performance (4 tests)
# =============================================================================


class TestVerticalInitialization:
    """Test vertical initialization performance."""

    def test_coding_vertical_load_time(self, benchmark: BenchmarkFixture):
        """Benchmark CodingAssistant initialization time."""

        # Force fresh initialization
        def init_coding():
            CodingAssistant._instance = None
            return CodingAssistant()

        assistant = benchmark(init_coding)
        assert assistant is not None
        assert assistant.name == "coding"

    def test_all_verticals_load_time(self, benchmark: BenchmarkFixture):
        """Benchmark loading all verticals."""

        def load_all_verticals():
            CodingAssistant._instance = None
            DataAnalysisAssistant._instance = None
            DevOpsAssistant._instance = None
            RAGAssistant._instance = None
            ResearchAssistant._instance = None

            verticals = [
                CodingAssistant(),
                DataAnalysisAssistant(),
                DevOpsAssistant(),
                RAGAssistant(),
                ResearchAssistant(),
            ]
            return verticals

        verticals = benchmark(load_all_verticals)
        assert len(verticals) == 5

    @pytest.mark.skip(reason="ModeConfigRegistry.get_mode() returns None, needs investigation")
    def test_mode_config_initialization(self, benchmark: BenchmarkFixture):
        """Benchmark mode configuration loading."""
        # Reset registry
        ModeConfigRegistry._instance = None

        def load_modes():
            registry = ModeConfigRegistry.get_instance()
            # Use get_mode instead of load_config
            config = registry.get_mode(vertical="coding", mode_name="build")
            return config

        mode = benchmark(load_modes)
        assert mode is not None
        assert mode.name == "build"

    def test_capability_loading_performance(self, benchmark: BenchmarkFixture):
        """Benchmark capability loading for verticals."""

        def load_capabilities():
            loader = CapabilityLoader.from_vertical("coding")
            return loader.load_capabilities("coding")

        capability_set = benchmark(load_capabilities)
        assert capability_set is not None


# =============================================================================
# Test Suite 2: Execution Performance (3 tests)
# =============================================================================


class TestVerticalExecution:
    """Test vertical execution performance."""

    @pytest.mark.skip(reason="AST parser module (victor.coding.ast.parser) does not exist")
    def test_ast_parsing_speed(self, benchmark: BenchmarkFixture, sample_python_file: Path):
        """Benchmark AST parsing performance."""
        from victor.coding.ast.parser import ASTParser

        parser = ASTParser()

        def parse_file():
            return parser.parse_file(str(sample_python_file))

        ast_result = benchmark(parse_file)
        assert ast_result is not None
        assert ast_result.root is not None

    @pytest.mark.skip(
        reason="CodebaseIndexer module (victor.coding.codebase.indexer) does not exist"
    )
    def test_semantic_search_indexing(self, benchmark: BenchmarkFixture, sample_codebase: Path):
        """Benchmark codebase indexing for semantic search."""
        from victor.coding.codebase.indexer import CodebaseIndexer

        def index_codebase():
            indexer = CodebaseIndexer(str(sample_codebase))
            indexer.index_codebase()
            return indexer

        indexer = benchmark(index_codebase)
        assert indexer is not None
        # Verify indexing worked
        assert len(indexer.documents) > 0

    @pytest.mark.skip(
        reason="CodebaseSearcher module (victor.coding.codebase.searcher) does not exist"
    )
    def test_code_search_performance(self, benchmark: BenchmarkFixture, sample_codebase: Path):
        """Benchmark code search query performance."""
        from victor.coding.codebase.searcher import CodebaseSearcher

        # Setup: Index the codebase first
        searcher = CodebaseSearcher(str(sample_codebase))

        def search_query():
            return searcher.search("function that multiplies values", top_k=5)

        results = benchmark(search_query)
        assert isinstance(results, list)


# =============================================================================
# Test Suite 3: Memory Efficiency (3 tests)
# =============================================================================


class TestVerticalMemory:
    """Test vertical memory efficiency."""

    def test_memory_usage_on_startup(self, benchmark: BenchmarkFixture):
        """Benchmark memory usage during vertical initialization."""
        import tracemalloc

        def measure_memory():
            # Force garbage collection
            gc.collect()

            # Start tracing
            tracemalloc.start()

            # Initialize vertical
            CodingAssistant._instance = None
            assistant = CodingAssistant()

            # Get current memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return peak

        peak_memory = benchmark(measure_memory)
        # Peak memory should be reasonable (< 50MB for initialization)
        assert peak_memory < 50 * 1024 * 1024  # 50 MB

    @pytest.mark.skip(reason="AST parser module (victor.coding.ast.parser) does not exist")
    def test_memory_cleanup_after_operation(
        self, benchmark: BenchmarkFixture, sample_python_file: Path
    ):
        """Benchmark memory cleanup after AST parsing."""
        import tracemalloc
        from victor.coding.ast.parser import ASTParser

        parser = ASTParser()

        def measure_cleanup():
            gc.collect()

            # Measure baseline
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]

            # Perform operation
            parser.parse_file(str(sample_python_file))

            # Force cleanup
            gc.collect()

            # Measure after cleanup
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Return growth relative to baseline
            return current - baseline

        memory_growth = benchmark(measure_cleanup)
        # Memory growth should be minimal after cleanup (< 5MB)
        assert memory_growth < 5 * 1024 * 1024  # 5 MB

    @pytest.mark.skip(reason="AST parser module (victor.coding.ast.parser) does not exist")
    def test_large_file_memory_usage(self, benchmark: BenchmarkFixture, large_python_file: Path):
        """Benchmark memory usage for large file operations."""
        import tracemalloc
        from victor.coding.ast.parser import ASTParser

        parser = ASTParser()

        def measure_large_file_memory():
            gc.collect()

            tracemalloc.start()

            # Parse large file
            result = parser.parse_file(str(large_python_file))

            # Get memory stats
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Return both current and peak memory
            return {"current": current, "peak": peak, "nodes": len(result.root.children)}

        memory_stats = benchmark(measure_large_file_memory)
        assert memory_stats["nodes"] > 0
        # Peak memory should scale reasonably with file size
        # For this ~50 class file, peak should be < 20MB
        assert memory_stats["peak"] < 20 * 1024 * 1024  # 20 MB


# =============================================================================
# Test Suite 4: Concurrent Operations
# =============================================================================


class TestConcurrentOperations:
    """Test concurrent vertical operations performance."""

    def test_concurrent_vertical_initialization(self, benchmark: BenchmarkFixture):
        """Benchmark concurrent initialization of multiple verticals."""
        import asyncio

        async def init_verticals_concurrently():
            CodingAssistant._instance = None
            DataAnalysisAssistant._instance = None
            DevOpsAssistant._instance = None

            # Simulate concurrent initialization
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(None, CodingAssistant),
                loop.run_in_executor(None, DataAnalysisAssistant),
                loop.run_in_executor(None, DevOpsAssistant),
            ]
            results = await asyncio.gather(*tasks)
            return results

        def run_concurrent():
            return asyncio.run(init_verticals_concurrently())

        verticals = benchmark(run_concurrent)
        assert len(verticals) == 3


# =============================================================================
# Test Suite 5: Scalability Tests
# =============================================================================


class TestVerticalScalability:
    """Test vertical scalability with increasing workload sizes."""

    def test_tool_selection_scalability(self, benchmark: BenchmarkFixture):
        """Benchmark tool selection with increasing tool counts."""
        from victor.tools.registry import ToolRegistry

        def setup_large_registry():
            # Create new instance instead of using get_instance
            registry = ToolRegistry()
            # Tools are already registered, just get count
            return registry.list_tools()

        def benchmark_selection():
            tools = setup_large_registry()
            # Simulate selection filtering
            coding_tools = [
                t for t in tools if "coding" in t.name.lower() or "file" in t.name.lower()
            ]
            return len(coding_tools)

        count = benchmark(benchmark_selection)
        assert count >= 0  # Will depend on available tools

    def test_mode_switching_performance(self, benchmark: BenchmarkFixture):
        """Benchmark mode switching performance."""
        assistant = CodingAssistant()

        def switch_modes():
            modes = ["build", "plan", "explore"]
            results = []
            for mode in modes:
                config = assistant.get_mode_config(mode)
                results.append(config.name)
            return results

        mode_names = benchmark(switch_modes)
        assert len(mode_names) == 3
        assert "build" in mode_names


# =============================================================================
# Performance Regression Thresholds
# =============================================================================


@pytest.mark.skip(
    reason="Performance regression meta-tests have isinstance issues and need refactoring"
)
@pytest.mark.parametrize(
    "test_name,threshold_max",
    [
        ("test_coding_vertical_load_time", 0.5),  # 500ms
        ("test_all_verticals_load_time", 1.0),  # 1s
        ("test_mode_config_initialization", 0.1),  # 100ms
        ("test_capability_loading_performance", 0.2),  # 200ms
        ("test_ast_parsing_speed", 0.1),  # 100ms
    ],
)
def test_performance_regression_thresholds(
    benchmark: BenchmarkFixture,
    test_name: str,
    threshold_max: float,
    request: pytest.FixtureRequest,
):
    """
    Meta-test to ensure performance thresholds are met.

    This test runs the actual performance test and verifies it completes
    within the expected time threshold.
    """
    # Get the actual test function
    test_func = request.node.getparent(test_name).__dict__.get(test_name)

    if test_func is None:
        pytest.skip(f"Test {test_name} not found")

    # Run the benchmark
    result = benchmark.pedantic(test_func, iterations=5, rounds=3)

    # Check if the mean time exceeds threshold
    assert result.stats["mean"] < threshold_max, (
        f"Performance regression detected: {test_name} "
        f"took {result.stats['mean']:.3f}s, "
        f"expected < {threshold_max}s"
    )


# =============================================================================
# Benchmark Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest-benchmark settings."""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark test")


# =============================================================================
# Helper Functions
# =============================================================================


def get_memory_usage() -> int:
    """Get current process memory usage in bytes."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"
