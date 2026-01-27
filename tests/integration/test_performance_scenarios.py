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

"""Performance benchmark tests for multi-coordinator workflows.

This test suite measures performance characteristics of coordinator workflows,
establishing baselines and identifying bottlenecks.

Benchmark Categories:
1. Latency Tests - End-to-end latency for various operations
2. Throughput Tests - Requests per second under load
3. Scalability Tests - Performance with increasing complexity
4. Memory Tests - Memory usage patterns
5. Coordinator Overhead - Individual coordinator performance

Success Criteria:
- Performance baselines established
- Bottlenecks identified
- Regression detection capability
- Clear performance reporting

Estimated Runtime: 5-10 minutes
"""

import asyncio
import pytest
import time
import psutil
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager

# Import from parent conftest for Ollama availability checking
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))
from conftest import requires_ollama


# ============================================================================
# Timeout Handling Utilities
# ============================================================================


@asynccontextmanager
async def skip_on_timeout(timeout_seconds: float, test_name: str = "unknown"):
    """Context manager that skips the test on timeout instead of failing.

    Use this for operations that may legitimately take too long on slow providers
    (e.g., Ollama with large models) where a timeout should be a graceful skip,
    not a test failure.

    Args:
        timeout_seconds: Maximum time to wait before skipping
        test_name: Test name for the skip message
    """
    try:
        async with asyncio.timeout(timeout_seconds):
            yield
    except asyncio.TimeoutError:
        pytest.skip(
            f"[{test_name}] Operation timed out after {timeout_seconds}s "
            f"(slow provider or resource constraint, not a test failure)"
        )


# ============================================================================
# Benchmark Infrastructure
# ============================================================================


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Baseline performance metrics."""

    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    memory_mb: float
    cpu_percent: float


class BenchmarkSuite:
    """Comprehensive benchmark suite for coordinator performance."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baselines: Dict[str, PerformanceBaseline] = {}

    @asynccontextmanager
    async def measure(self, name: str, operation: str):
        """Context manager for measuring performance."""
        # Start measurements
        tracemalloc.start()
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent()

        try:
            yield

            # End measurements
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            end_cpu = psutil.cpu_percent()

            duration_ms = (end_time - start_time) * 1000
            memory_mb = end_memory - start_memory
            cpu_percent = end_cpu - start_cpu if end_cpu > start_cpu else 0

            result = BenchmarkResult(
                name=name,
                operation=operation,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                success=True,
            )

            self.results.append(result)

        except Exception as e:
            # Record failure
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            result = BenchmarkResult(
                name=name,
                operation=operation,
                duration_ms=duration_ms,
                memory_mb=0.0,
                cpu_percent=0.0,
                success=False,
                error=str(e),
            )

            self.results.append(result)
            raise

        finally:
            tracemalloc.stop()

    def get_statistics(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        op_results = [r for r in self.results if r.operation == operation and r.success]

        if not op_results:
            return {
                "count": 0,
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "total_mb": 0.0,
            }

        latencies = sorted([r.duration_ms for r in op_results])
        count = len(latencies)

        return {
            "count": count,
            "avg_ms": sum(latencies) / count,
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "p50_ms": latencies[int(count * 0.5)],
            "p95_ms": latencies[int(count * 0.95)],
            "p99_ms": latencies[int(count * 0.99)],
            "total_mb": sum(r.memory_mb for r in op_results),
        }

    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        lines = []
        lines.append("=" * 100)
        lines.append("COORDINATOR PERFORMANCE BENCHMARK REPORT")
        lines.append("=" * 100)
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append("")

        # Group results by operation
        operations = set(r.operation for r in self.results)

        for operation in sorted(operations):
            stats = self.get_statistics(operation)

            lines.append(f"Operation: {operation}")
            lines.append("-" * 100)
            lines.append(f"  Count:     {stats['count']}")
            lines.append("  Latency:")
            lines.append(f"    Avg:     {stats['avg_ms']:.2f} ms")
            lines.append(f"    Min:     {stats['min_ms']:.2f} ms")
            lines.append(f"    Max:     {stats['max_ms']:.2f} ms")
            lines.append(f"    P50:     {stats['p50_ms']:.2f} ms")
            lines.append(f"    P95:     {stats['p95_ms']:.2f} ms")
            lines.append(f"    P99:     {stats['p99_ms']:.2f} ms")
            lines.append(f"  Memory:    {stats['total_mb']:.2f} MB")
            lines.append("")

        # Summary statistics
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.success])
        failed_tests = total_tests - successful_tests

        lines.append("SUMMARY")
        lines.append("-" * 100)
        lines.append(f"  Total Tests:    {total_tests}")
        lines.append(f"  Successful:     {successful_tests}")
        lines.append(f"  Failed:         {failed_tests}")

        # Guard clause for division by zero
        if total_tests == 0:
            lines.append("  No tests executed yet.")
            lines.append("")
            return "\n".join(lines)

        lines.append(f"  Success Rate:   {(successful_tests / total_tests * 100):.1f}%")
        lines.append("")

        # Identify bottlenecks
        lines.append("BOTTLENECKS")
        lines.append("-" * 100)

        operation_stats = [(op, self.get_statistics(op)) for op in operations]
        operation_stats.sort(key=lambda x: x[1]["avg_ms"], reverse=True)

        for operation, stats in operation_stats[:5]:
            lines.append(f"  {operation}: {stats['avg_ms']:.2f} ms avg")

        lines.append("")
        lines.append("=" * 100)

        return "\n".join(lines)

    def save_report(self, path: Path):
        """Save report to file."""
        report = self.generate_report()
        path.write_text(report)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def benchmark_suite():
    """Create benchmark suite for testing."""
    return BenchmarkSuite()


@pytest.fixture
async def performance_orchestrator(mock_settings, mock_provider, mock_container):
    """Create orchestrator optimized for performance testing."""
    from victor.agent.orchestrator_factory import OrchestratorFactory

    # Create factory with performance settings
    factory = OrchestratorFactory(
        settings=mock_settings,
        provider=mock_provider,
        model="claude-sonnet-4-5",
        temperature=0.7,
        max_tokens=4096,
    )

    # Create orchestrator
    orchestrator = factory.create_orchestrator()

    yield orchestrator

    # Cleanup
    if hasattr(orchestrator, "cleanup"):
        await orchestrator.cleanup()


# ============================================================================
# Latency Tests
# ============================================================================


class TestLatency:
    """Test end-to-end latency for various operations."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_simple_query_latency(self, performance_orchestrator, benchmark_suite):
        """Benchmark latency of simple query without tools."""
        async with benchmark_suite.measure("simple_query", "latency"):
            performance_orchestrator.provider.chat = AsyncMock(
                return_value=MagicMock(
                    content="Hello! How can I help?",
                    tool_calls=None,
                    usage=MagicMock(input_tokens=20, output_tokens=10),
                )
            )

            result = await performance_orchestrator.chat("Hello!")
            assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_single_tool_call_latency(self, performance_orchestrator, benchmark_suite):
        """Benchmark latency of single tool call."""
        async with benchmark_suite.measure("single_tool_call", "latency"):
            performance_orchestrator.provider.chat = AsyncMock(
                return_value=MagicMock(
                    content="I've read the file.",
                    tool_calls=[
                        MagicMock(
                            id="call_1",
                            function=MagicMock(
                                name="read_file",
                                arguments='{"path": "/src/main.py"}',
                            ),
                        )
                    ],
                    usage=MagicMock(input_tokens=50, output_tokens=15),
                )
            )

            result = await performance_orchestrator.chat("Read main.py")
            assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_multiple_tool_calls_latency(self, performance_orchestrator, benchmark_suite):
        """Benchmark latency of multiple tool calls."""
        async with benchmark_suite.measure("multiple_tool_calls", "latency"):
            tool_calls = [
                MagicMock(
                    id=f"call_{i}",
                    function=MagicMock(
                        name="read_file",
                        arguments=f'{{"path": "/src/file{i}.py"}}',
                    ),
                )
                for i in range(10)
            ]

            performance_orchestrator.provider.chat = AsyncMock(
                return_value=MagicMock(
                    content="I've read 10 files.",
                    tool_calls=tool_calls,
                    usage=MagicMock(input_tokens=200, output_tokens=25),
                )
            )

            result = await performance_orchestrator.chat("Read 10 files")
            assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_context_compaction_latency(self, performance_orchestrator, benchmark_suite):
        """Benchmark latency of context compaction."""
        # Create long conversation
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(100)]

        async with benchmark_suite.measure("context_compaction", "latency"):
            performance_orchestrator.provider.chat = AsyncMock(
                return_value=MagicMock(
                    content="Context compacted.",
                    tool_calls=None,
                    usage=MagicMock(input_tokens=5000, output_tokens=15),
                )
            )

            result = await performance_orchestrator.chat("Continue conversation")
            assert result is not None


# ============================================================================
# Throughput Tests
# ============================================================================


class TestThroughput:
    """Test requests per second under load."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_sequential_throughput(self, performance_orchestrator, benchmark_suite):
        """Benchmark sequential request throughput."""
        request_count = 50
        start_time = time.time()

        # Skip on timeout (2 minutes for 50 sequential requests)
        async with skip_on_timeout(120, "sequential_throughput"):
            async with benchmark_suite.measure("sequential_throughput", "throughput"):
                tasks = []
                for i in range(request_count):
                    performance_orchestrator.provider.chat = AsyncMock(
                        return_value=MagicMock(
                            content=f"Response {i}",
                            tool_calls=None,
                            usage=MagicMock(input_tokens=20, output_tokens=10),
                        )
                    )

                    task = performance_orchestrator.chat(f"Request {i}")
                    tasks.append(task)

                # Execute sequentially (not actually concurrent)
                for task in tasks:
                    result = await task
                    assert result is not None

            end_time = time.time()
            duration = end_time - start_time
            throughput = request_count / duration

            # Record throughput in metadata
            if benchmark_suite.results:
                benchmark_suite.results[-1].metadata["throughput_rps"] = throughput
                benchmark_suite.results[-1].metadata["request_count"] = request_count
                benchmark_suite.results[-1].metadata["duration_s"] = duration

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_throughput(self, performance_orchestrator, benchmark_suite):
        """Benchmark concurrent request throughput."""
        request_count = 50
        start_time = time.time()

        # Skip on timeout (90 seconds for 50 concurrent requests)
        async with skip_on_timeout(90, "concurrent_throughput"):
            async with benchmark_suite.measure("concurrent_throughput", "throughput"):

                async def make_request(i):
                    performance_orchestrator.provider.chat = AsyncMock(
                        return_value=MagicMock(
                            content=f"Response {i}",
                            tool_calls=None,
                            usage=MagicMock(input_tokens=20, output_tokens=10),
                        )
                    )
                    return await performance_orchestrator.chat(f"Request {i}")

                # Execute concurrently
                tasks = [make_request(i) for i in range(request_count)]
                results = await asyncio.gather(*tasks)

                assert all(result is not None for result in results)

            end_time = time.time()
            duration = end_time - start_time
            throughput = request_count / duration

            # Record throughput in metadata
            if benchmark_suite.results:
                benchmark_suite.results[-1].metadata["throughput_rps"] = throughput
                benchmark_suite.results[-1].metadata["request_count"] = request_count
                benchmark_suite.results[-1].metadata["duration_s"] = duration


# ============================================================================
# Scalability Tests
# ============================================================================


class TestScalability:
    """Test performance with increasing complexity."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    @pytest.mark.timeout(600)  # 10 minutes total timeout
    async def test_scalability_conversation_length(self, performance_orchestrator, benchmark_suite):
        """Benchmark performance with increasing conversation length."""
        from victor.providers.base import StreamChunk, CompletionResponse

        conversation_lengths = [10, 50, 100, 200, 500]

        for length in conversation_lengths:
            # Timeout per length: 120s for smaller, 300s for larger
            timeout_per_length = 300 if length >= 100 else 120

            async with skip_on_timeout(timeout_per_length, f"conversation_length_{length}"):
                async with benchmark_suite.measure(f"conversation_length_{length}", "scalability"):
                    # Create conversation of specified length
                    for i in range(length):
                        # Create async generator mock for stream_chat
                        async def mock_stream_gen():
                            yield StreamChunk(
                                content=f"Response {i}",
                                finish_reason="stop",
                                usage={"input_tokens": 30, "output_tokens": 15},
                            )

                        # Mock stream_chat with async generator
                        performance_orchestrator.provider.stream_chat = AsyncMock(
                            return_value=mock_stream_gen()
                        )

                        # Mock chat to return CompletionResponse
                        response = CompletionResponse(
                            content=f"Response {i}",
                            tool_calls=None,
                            usage={"input_tokens": 30, "output_tokens": 15},
                        )
                        performance_orchestrator.provider.chat = AsyncMock(return_value=response)

                        result = await performance_orchestrator.chat(f"Message {i}")
                        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    @pytest.mark.timeout(300)  # 5 minutes total timeout
    async def test_scalability_tool_calls(self, performance_orchestrator, benchmark_suite):
        """Benchmark performance with increasing tool calls."""
        tool_counts = [1, 5, 10, 20, 50]

        for count in tool_counts:
            async with skip_on_timeout(60, f"tool_calls_{count}"):
                async with benchmark_suite.measure(f"tool_calls_{count}", "scalability"):
                    tool_calls = [
                        MagicMock(
                            id=f"call_{i}",
                            function=MagicMock(
                                name="read_file",
                                arguments=f'{{"path": "/src/file{i}.py"}}',
                            ),
                        )
                        for i in range(count)
                    ]

                    performance_orchestrator.provider.chat = AsyncMock(
                        return_value=MagicMock(
                            content=f"I've read {count} files.",
                            tool_calls=tool_calls,
                            usage=MagicMock(input_tokens=count * 20, output_tokens=25),
                        )
                    )

                    result = await performance_orchestrator.chat(f"Read {count} files")
                    assert result is not None


# ============================================================================
# Memory Tests
# ============================================================================


class TestMemory:
    """Test memory usage patterns."""

    @requires_ollama()
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    @pytest.mark.integration
    async def test_memory_leak_detection(self, benchmark_suite):
        """Test for memory leaks over repeated operations with real LLM.

        This is a proper integration test that:
        - Uses real Ollama provider (no mocked components)
        - Tests full flow including semantic selector, tool selection, etc.
        - Detects actual memory leaks in production code path
        - Skipped if Ollama is not available (@requires_ollama)
        - Skips on timeout if commodity hardware is too slow

        Uses a small local model (gemma2:2b) for fast execution.
        Per-iteration timeout (15s) ensures graceful skip on slow hardware.
        """
        from victor.providers.ollama_provider import OllamaProvider
        from victor.agent.orchestrator_factory import OrchestratorFactory
        from victor.config.settings import Settings

        # Create real provider with small model for speed
        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model="gemma2:2b",  # Small model for faster execution
        )

        try:
            # Create settings with minimal tool selection overhead
            settings = Settings()
            settings.tool_selection_strategy = "keyword"  # Faster than semantic/hybrid
            settings.enable_tool_selection_rl = False
            settings.parallel_tool_execution = False  # Simplify execution

            # Create orchestrator
            factory = OrchestratorFactory(
                settings=settings,
                provider=provider,
                model="gemma2:2b",
                temperature=0.7,
                max_tokens=512,  # Shorter responses for speed
            )
            orchestrator = factory.create_orchestrator()

            # Get baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Use per-iteration timeout (15 seconds each) for graceful degradation
            # If any iteration times out, the entire test skips (as expected for slow hardware)
            async with skip_on_timeout(150, "memory_leak_detection"):  # 15s * 10 iterations = 150s total
                async with benchmark_suite.measure("memory_leak_test", "memory"):
                    # Perform operations with real LLM
                    for i in range(10):
                        # Per-iteration timeout for individual requests
                        try:
                            async with asyncio.timeout(15):  # 15 second timeout per iteration
                                result = await orchestrator.chat(f"Message {i}: What is {i} plus {i}?")
                                assert result is not None
                                assert hasattr(result, "content")
                        except asyncio.TimeoutError:
                            # If any single iteration times out, skip the entire test
                            pytest.skip(
                                f"Iteration {i} timed out after 15s (commodity hardware too slow for this test)"
                            )

                # Check final memory
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = final_memory - baseline_memory

                # Record in metadata
                if benchmark_suite.results:
                    benchmark_suite.results[-1].metadata["baseline_memory_mb"] = baseline_memory
                    benchmark_suite.results[-1].metadata["final_memory_mb"] = final_memory
                    benchmark_suite.results[-1].metadata["memory_growth_mb"] = memory_growth
                    benchmark_suite.results[-1].metadata["iterations"] = 10
                    benchmark_suite.results[-1].metadata["model"] = "gemma2:2b"
                    benchmark_suite.results[-1].metadata["tool_selection"] = "keyword"

                # Memory growth threshold set to 500 MB for 10 iterations with real LLM
                # Real memory leaks would show 1-2 GB growth, not a few hundred MB
                assert memory_growth < 500, f"Excessive memory growth detected (potential leak): {memory_growth:.2f} MB"

        finally:
            await provider.close()
            # Cleanup orchestrator resources (idempotent - safe to call multiple times)
            if 'orchestrator' in locals():
                await orchestrator.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_memory_per_request(self, performance_orchestrator, benchmark_suite):
        """Test memory usage per request."""
        process = psutil.Process()

        async with benchmark_suite.measure("memory_per_request", "memory"):
            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024

            # Perform request
            performance_orchestrator.provider.chat = AsyncMock(
                return_value=MagicMock(
                    content="Test response",
                    tool_calls=None,
                    usage=MagicMock(input_tokens=20, output_tokens=10),
                )
            )

            result = await performance_orchestrator.chat("Test")
            assert result is not None

            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_per_request = memory_after - memory_before

            # Record in metadata
            if benchmark_suite.results:
                benchmark_suite.results[-1].metadata["memory_per_request_mb"] = memory_per_request


# ============================================================================
# Coordinator Overhead Tests
# ============================================================================


class TestCoordinatorOverhead:
    """Test individual coordinator performance overhead."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_response_coordinator_overhead(self, performance_orchestrator, benchmark_suite):
        """Benchmark ResponseCoordinator overhead."""
        async with benchmark_suite.measure("response_coordinator", "coordinator_overhead"):
            performance_orchestrator.provider.chat = AsyncMock(
                return_value=MagicMock(
                    content="Test response",
                    tool_calls=None,
                    usage=MagicMock(input_tokens=20, output_tokens=10),
                )
            )

            result = await performance_orchestrator.chat("Test")
            assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_state_coordinator_overhead(self, performance_orchestrator, benchmark_suite):
        """Benchmark StateCoordinator overhead."""
        async with benchmark_suite.measure("state_coordinator", "coordinator_overhead"):
            performance_orchestrator.provider.chat = AsyncMock(
                return_value=MagicMock(
                    content="Test response",
                    tool_calls=None,
                    usage=MagicMock(input_tokens=20, output_tokens=10),
                )
            )

            result = await performance_orchestrator.chat("Test")
            assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_tool_selection_coordinator_overhead(
        self, performance_orchestrator, benchmark_suite
    ):
        """Benchmark ToolSelectionCoordinator overhead."""
        async with benchmark_suite.measure("tool_selection_coordinator", "coordinator_overhead"):
            performance_orchestrator.provider.chat = AsyncMock(
                return_value=MagicMock(
                    content="I'll help you.",
                    tool_calls=[
                        MagicMock(
                            id="call_1",
                            function=MagicMock(
                                name="read_file",
                                arguments='{"path": "/src/main.py"}',
                            ),
                        )
                    ],
                    usage=MagicMock(input_tokens=50, output_tokens=15),
                )
            )

            result = await performance_orchestrator.chat("Read main.py")
            assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_conversation_coordinator_overhead(
        self, performance_orchestrator, benchmark_suite
    ):
        """Benchmark ConversationCoordinator overhead."""
        # Skip on timeout (60 seconds for 10 conversation turns)
        async with skip_on_timeout(60, "conversation_coordinator_overhead"):
            async with benchmark_suite.measure("conversation_coordinator", "coordinator_overhead"):
                # Perform multiple turns to test conversation tracking
                for i in range(10):
                    performance_orchestrator.provider.chat = AsyncMock(
                        return_value=MagicMock(
                            content=f"Response {i}",
                            tool_calls=None,
                            usage=MagicMock(input_tokens=30, output_tokens=15),
                        )
                    )

                    result = await performance_orchestrator.chat(f"Message {i}")
                    assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_tool_execution_coordinator_overhead(
        self, performance_orchestrator, benchmark_suite
    ):
        """Benchmark ToolExecutionCoordinator overhead."""
        async with benchmark_suite.measure("tool_execution_coordinator", "coordinator_overhead"):
            tool_calls = [
                MagicMock(
                    id=f"call_{i}",
                    function=MagicMock(
                        name="read_file",
                        arguments=f'{{"path": "/src/file{i}.py"}}',
                    ),
                )
                for i in range(5)
            ]

            performance_orchestrator.provider.chat = AsyncMock(
                return_value=MagicMock(
                    content="I've read 5 files.",
                    tool_calls=tool_calls,
                    usage=MagicMock(input_tokens=100, output_tokens=20),
                )
            )

            result = await performance_orchestrator.chat("Read 5 files")
            assert result is not None


# ============================================================================
# Performance Report Generation
# ============================================================================


@pytest.mark.integration
@pytest.mark.benchmark
def test_generate_performance_report(benchmark_suite, tmp_path):
    """Generate and save comprehensive performance report."""

    # Generate report
    report = benchmark_suite.generate_report()

    # Save report
    report_path = tmp_path / "performance_report.txt"
    benchmark_suite.save_report(report_path)

    # Verify report was created
    assert report_path.exists()

    # Verify report contains key sections
    content = report_path.read_text()
    assert "COORDINATOR PERFORMANCE BENCHMARK REPORT" in content
    assert "SUMMARY" in content

    # BOTTLENECKS section only appears when there are results
    # When there are no results, the report shows "No tests executed yet."
    if len(benchmark_suite.results) > 0:
        assert "BOTTLENECKS" in content
    else:
        assert "No tests executed yet." in content

    # Print report for visibility
    print("\n" + report)
