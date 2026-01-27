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

"""Comprehensive scalability test suite for Victor AI.

This module provides detailed scalability tests covering:
1. Concurrent request handling
2. Large conversation context management
3. Memory leak detection
4. Resource utilization under load
5. Stress testing to find breaking points

Performance Targets:
- Single request latency: <100ms (p50), <500ms (p99)
- Concurrent throughput: >100 requests/second
- Memory usage: <1GB for 100 concurrent sessions
- Error rate: <1% under normal load
- Graceful degradation above capacity limits

Note:
These tests require the locust package and use gevent for async load testing.
They are excluded from normal test runs due to gevent/SSL monkey-patching conflicts.
Run explicitly with: pytest -m load_test tests/load_test/
"""

import asyncio
import gc
import os
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.core.container import ServiceContainer
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from .load_test_framework import AsyncLoadTestFramework


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
async def load_test_framework():
    """Provide load test framework instance."""
    return AsyncLoadTestFramework(base_url="http://localhost:8000")


@pytest.fixture
async def mock_orchestrator():
    """Create mock orchestrator for testing without API server."""
    from victor.agent.protocols import (
        ToolExecutorProtocol,
        ToolRegistryProtocol,
        ConversationControllerProtocol,
    )

    orchestrator = MagicMock(spec=AgentOrchestrator)

    # Mock async methods
    async def mock_chat(message: str, **kwargs):
        return f"Response to: {message}"

    orchestrator.chat = mock_chat
    orchestrator.stream_chat = AsyncMock()

    return orchestrator


# =============================================================================
# Concurrent Request Tests
# =============================================================================


@pytest.mark.usefixtures("api_server_available")
class TestConcurrentRequests:
    """Test system behavior under concurrent load.

    Requires API server running at localhost:8000.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_10_concurrent_requests(self, mock_orchestrator):
        """Test system with 10 concurrent requests."""
        num_requests = 10
        start_time = time.time()

        # Create concurrent tasks
        tasks = [mock_orchestrator.chat(f"Test message {i}") for i in range(num_requests)]

        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        # Verify all requests completed
        assert len(results) == num_requests
        assert all(r for r in results)

        # Performance check: should complete in reasonable time
        assert elapsed < 5.0, f"10 requests took {elapsed:.2f}s (expected <5s)"

        print(f"\n✓ 10 concurrent requests completed in {elapsed:.2f}s")
        print(f"  Throughput: {num_requests / elapsed:.2f} requests/second")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_100_concurrent_requests(self, mock_orchestrator):
        """Test system with 100 concurrent requests."""
        num_requests = 100
        start_time = time.time()

        tasks = [mock_orchestrator.chat(f"Test message {i}") for i in range(num_requests)]

        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        assert len(results) == num_requests
        assert elapsed < 30.0, f"100 requests took {elapsed:.2f}s (expected <30s)"

        print(f"\n✓ 100 concurrent requests completed in {elapsed:.2f}s")
        print(f"  Throughput: {num_requests / elapsed:.2f} requests/second")

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_concurrent_throughput_degradation(self):
        """Test throughput degradation as concurrency increases."""
        framework = AsyncLoadTestFramework()

        payload = {
            "message": "Hello",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
        }

        concurrency_levels = [1, 10, 50, 100]
        results = []

        for concurrency in concurrency_levels:
            result = await framework.execute_concurrent_requests(
                num_requests=concurrency * 5,  # 5 requests per concurrent user
                concurrency=concurrency,
                payload=payload,
            )
            results.append(result)

            print(f"\nConcurrency: {concurrency}")
            print(f"  Throughput: {result['requests_per_second']:.2f} req/s")
            print(f"  P95 latency: {result['response_times']['p95']:.2f}ms")
            print(f"  Success rate: {result['success_rate']:.2f}%")

        # Verify throughput doesn't degrade severely
        # Allow up to 50% degradation from lowest to highest concurrency
        lowest_throughput = results[0]["requests_per_second"]
        highest_throughput = results[-1]["requests_per_second"]

        if highest_throughput > 0:
            degradation = (lowest_throughput - highest_throughput) / lowest_throughput
            assert (
                degradation < 0.5
            ), f"Throughput degraded by {degradation * 100:.1f}% (limit: 50%)"


# =============================================================================
# Large Context Tests
# =============================================================================


@pytest.mark.usefixtures("api_server_available")
class TestLargeContext:
    """Test system behavior with large conversation contexts.

    Requires API server running at localhost:8000.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_conversation_history(self, mock_orchestrator):
        """Test handling of conversations with 100+ turns."""
        conversation_history = []
        num_turns = 100

        start_time = time.time()

        for i in range(num_turns):
            message = f"This is turn {i + 1} of our conversation"
            response = await mock_orchestrator.chat(message)

            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": response})

        elapsed = time.time() - start_time

        assert len(conversation_history) == num_turns * 2
        print(f"\n✓ {num_turns}-turn conversation completed in {elapsed:.2f}s")
        print(f"  Average per turn: {elapsed / num_turns * 1000:.2f}ms")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_usage_with_large_context(self):
        """Test memory usage with growing conversation context."""
        process = psutil.Process()
        gc.collect()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]

        # Simulate growing conversation
        context_size = 10
        for i in range(10):
            # Create large context
            large_context = [{"role": "user", "content": "x" * 1000} for _ in range(context_size)]
            context_size *= 2  # Double each time

            # Force garbage collection
            gc.collect()

            # Measure memory
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_samples.append(memory_mb)

        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory

        print("\n✓ Memory usage with large context:")
        print(f"  Initial: {initial_memory:.2f}MB")
        print(f"  Final: {final_memory:.2f}MB")
        print(f"  Growth: {memory_growth:.2f}MB")

        # Memory growth should be reasonable (<500MB)
        assert memory_growth < 500, f"Memory grew {memory_growth:.2f}MB (limit: 500MB)"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_context_compaction_performance(self):
        """Test performance of context compaction under load."""
        from victor.agent.context_compactor import ContextCompactor
        from victor.agent.coordinators.compaction_strategies import (
            TruncationCompactionStrategy,
            LLMCompactionStrategy,
        )

        # Create large conversation
        large_conversation = [{"role": "user", "content": f"Message {i}"} for i in range(1000)]

        compactor = ContextCompactor()
        strategies = [
            ("truncation", TruncationCompactionStrategy(max_chars=100)),
            ("llm", LLMCompactionStrategy()),
        ]

        results = {}

        for name, strategy in strategies:
            start = time.time()

            # Mock LLM compaction to avoid actual API calls
            if name == "llm":
                with patch.object(strategy, "_summarize_with_llm", return_value="Summary"):
                    compacted = compactor.compact(large_conversation, strategy)
            else:
                compacted = compactor.compact(large_conversation, strategy)

            elapsed = time.time() - start

            results[name] = {
                "time_seconds": elapsed,
                "original_length": len(large_conversation),
                "compacted_length": len(compacted),
            }

            print(f"\n✓ {name.capitalize()} compaction:")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Reduction: {len(large_conversation)} → {len(compacted)} messages")

        # Truncation should be fast
        assert results["truncation"]["time_seconds"] < 0.1, (
            "Truncation too slow: " f"{results['truncation']['time_seconds']:.3f}s"
        )


# =============================================================================
# Memory Leak Tests
# =============================================================================


@pytest.mark.usefixtures("api_server_available")
class TestMemoryLeaks:
    """Test for memory leaks in long-running sessions.

    Requires API server running at localhost:8000.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_leak_detection(self, mock_orchestrator):
        """Test for memory leaks over 1000 requests."""
        process = psutil.Process()
        gc.collect()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]

        # Execute many requests
        num_requests = 1000
        sample_interval = 100

        for i in range(num_requests):
            await mock_orchestrator.chat(f"Test message {i}")

            # Sample memory periodically
            if i % sample_interval == 0:
                gc.collect()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)

        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory

        print(f"\n✓ Memory leak test ({num_requests} requests):")
        print(f"  Initial: {initial_memory:.2f}MB")
        print(f"  Final: {final_memory:.2f}MB")
        print(f"  Growth: {memory_growth:.2f}MB")
        print(f"  Per request: {memory_growth / num_requests * 1024:.2f}KB")

        # Flag significant growth (>200MB for 1000 requests)
        if memory_growth > 200:
            pytest.fail(f"Potential memory leak detected: {memory_growth:.2f}MB growth")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_resource_cleanup(self):
        """Test that resources are properly cleaned up after sessions."""
        from victor.core.container import ServiceContainer

        # Track object creation
        initial_objects = gc.get_count()
        initial_containers = len(
            [obj for obj in gc.get_objects() if isinstance(obj, ServiceContainer)]
        )

        # Create and destroy multiple containers
        for _ in range(10):
            container = ServiceContainer()
            # Simulate some operations
            container.register(str, lambda c: "test")
            del container

        gc.collect()

        final_objects = gc.get_count()
        final_containers = len(
            [obj for obj in gc.get_objects() if isinstance(obj, ServiceContainer)]
        )

        # Check for container leaks
        container_leak = final_containers - initial_containers
        assert (
            container_leak <= 1
        ), f"Container leak detected: {container_leak} containers not cleaned up"

        print("\n✓ Resource cleanup test:")
        print(f"  Container leak: {container_leak} containers")


# =============================================================================
# Stress Tests
# =============================================================================


@pytest.mark.usefixtures("api_server_available")
class TestStressTesting:
    """Find system breaking points and test graceful degradation.

    Requires API server running at localhost:8000.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_find_breaking_point(self):
        """Find the concurrency level where system starts failing."""
        framework = AsyncLoadTestFramework()

        payload = {
            "message": "Stress test",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
        }

        # Test increasing concurrency levels
        concurrency_levels = [10, 50, 100, 200, 500]
        breaking_point = None

        for concurrency in concurrency_levels:
            result = await framework.execute_concurrent_requests(
                num_requests=concurrency,
                concurrency=concurrency,
                payload=payload,
            )

            error_rate = (result["failed_requests"] / result["total_requests"]) * 100

            print(f"\nConcurrency: {concurrency}")
            print(f"  Success rate: {result['success_rate']:.2f}%")
            print(f"  Error rate: {error_rate:.2f}%")
            print(f"  P95 latency: {result['response_times']['p95']:.2f}ms")

            # Mark breaking point if error rate exceeds 10%
            if error_rate > 10 and breaking_point is None:
                breaking_point = concurrency
                print(f"  ⚠ Breaking point detected at {concurrency} concurrent users")

        if breaking_point:
            print(f"\n✓ Breaking point identified: {breaking_point} concurrent users")
        else:
            print(f"\n✓ No breaking point found up to {concurrency_levels[-1]} users")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_graceful_degradation(self):
        """Test that system degrades gracefully under overload."""
        framework = AsyncLoadTestFramework()

        payload = {
            "message": "Degradation test",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
        }

        # Test at extreme concurrency
        result = await framework.execute_concurrent_requests(
            num_requests=1000,
            concurrency=500,
            payload=payload,
        )

        # Even under extreme load, system should:
        # 1. Not crash completely (>0% success rate)
        assert result["success_rate"] > 0, "System completely failed under load"

        # 2. Have reasonable error recovery (<95% failure)
        assert result["success_rate"] > 5, "Error rate too high (no graceful degradation)"

        print("\n✓ Graceful degradation test:")
        print(f"  Success rate under extreme load: {result['success_rate']:.2f}%")
        print("  System remained partially functional")


# =============================================================================
# Endurance Tests
# =============================================================================


@pytest.mark.usefixtures("api_server_available")
class TestEndurance:
    """Test long-running system stability.

    Long-running tests to detect memory leaks and performance degradation.

    Requires API server running at localhost:8000.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skip("Skip in CI - takes too long")
    async def test_extended_load_test(self, mock_orchestrator):
        """Run system under load for extended period (1+ hours)."""
        duration_seconds = 60 * 60  # 1 hour
        sample_interval = 60  # Sample every minute
        requests_per_second = 10

        process = psutil.Process()
        samples = []

        start_time = time.time()
        request_count = 0

        while (time.time() - start_time) < duration_seconds:
            # Make requests
            for _ in range(requests_per_second * sample_interval):
                await mock_orchestrator.chat(f"Endurance test {request_count}")
                request_count += 1

            # Sample metrics
            gc.collect()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()

            samples.append(
                {
                    "elapsed_seconds": time.time() - start_time,
                    "memory_mb": memory_mb,
                    "cpu_percent": cpu_percent,
                    "request_count": request_count,
                }
            )

            print("\nEndurance test progress:")
            print(f"  Elapsed: {samples[-1]['elapsed_seconds'] / 60:.1f} minutes")
            print(f"  Requests: {request_count}")
            print(f"  Memory: {memory_mb:.2f}MB")
            print(f"  CPU: {cpu_percent:.1f}%")

        # Analyze trends
        initial_memory = samples[0]["memory_mb"]
        final_memory = samples[-1]["memory_mb"]
        memory_growth = final_memory - initial_memory

        print("\n✓ Endurance test completed:")
        print(f"  Duration: {duration_seconds / 60:.1f} minutes")
        print(f"  Total requests: {request_count}")
        print(f"  Memory growth: {memory_growth:.2f}MB")

        # Check for significant memory leak
        if memory_growth > 1000:  # >1GB growth
            pytest.fail(f"Significant memory leak: {memory_growth:.2f}MB growth")


# =============================================================================
# Utility Functions
# =============================================================================


def generate_scalability_report(
    results: Dict[str, Any], output_dir: str = "/tmp/scalability_reports"
):
    """Generate detailed scalability report from test results."""
    from pathlib import Path
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"scalability_report_{timestamp}.json"

    report = {
        "timestamp": datetime.now().isoformat(),
        "test_results": results,
        "summary": {
            "total_tests": len(results),
            "passed": sum(1 for r in results.values() if r.get("success", False)),
            "failed": sum(1 for r in results.values() if not r.get("success", True)),
        },
        "recommendations": _generate_recommendations(results),
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Scalability report saved to: {report_file}")
    return report


def _generate_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on test results."""
    recommendations = []

    # Analyze concurrent request performance
    concurrent = results.get("concurrent_requests", {})
    if concurrent.get("p95_latency", 0) > 1000:
        recommendations.append(
            "High P95 latency under concurrent load. " "Consider implementing request queuing."
        )

    # Analyze memory usage
    memory = results.get("memory_usage", {})
    if memory.get("growth_mb", 0) > 500:
        recommendations.append(
            "Significant memory growth detected. "
            "Investigate context management and garbage collection."
        )

    # Analyze stress test results
    stress = results.get("stress_test", {})
    if stress.get("breaking_point", 1000) < 100:
        recommendations.append(
            "System fails at low concurrency. " "Review connection pooling and resource limits."
        )

    if not recommendations:
        recommendations.append("System performs well within tested parameters.")

    return recommendations
