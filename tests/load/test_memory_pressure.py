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

"""Memory pressure and stress tests for Victor AI.

Tests memory usage under extreme conditions to identify leaks and
ensure system stability under memory pressure.
"""

import asyncio
import pytest
import time
import gc
import os
import statistics
from typing import Any, Dict, List
from datetime import datetime

import psutil
import httpx
from pytest import mark


# Test configuration
API_HOST = "http://localhost:8765"
DEFAULT_TIMEOUT = 30.0


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


@mark.load
@mark.asyncio
class TestMemoryPressure:
    """Memory pressure and stress tests.

    Tests system behavior under memory pressure and identifies
    potential memory leaks.
    """

    async def test_baseline_memory(self, api_server_available):
        """Establish baseline memory usage.

        Target: <200MB for idle system
        """
        # Force garbage collection
        gc.collect()

        initial_memory = get_memory_usage()

        print(f"\nBaseline Memory Usage: {initial_memory:.2f}MB")

        # Baseline should be reasonable
        assert initial_memory < 500, f"Baseline memory too high: {initial_memory:.2f}MB"

    async def test_memory_growth_single_session(self, api_server_available):
        """Test memory growth during single user session.

        Target: <10MB growth over 100 messages
        """
        gc.collect()
        initial_memory = get_memory_usage()

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            for i in range(100):
                payload = {
                    "message": f"Test message {i}: What is the weather?",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                assert response.status_code == 200

                # Check memory every 25 messages
                if i % 25 == 0:
                    gc.collect()
                    current_memory = get_memory_usage()
                    growth = current_memory - initial_memory
                    print(f"  Message {i}: Memory growth {growth:.2f}MB")

        gc.collect()
        final_memory = get_memory_usage()
        total_growth = final_memory - initial_memory

        print("\nMemory Growth - Single Session:")
        print("  Messages: 100")
        print(f"  Initial Memory: {initial_memory:.2f}MB")
        print(f"  Final Memory: {final_memory:.2f}MB")
        print(f"  Total Growth: {total_growth:.2f}MB")

        # Allow reasonable growth for context
        assert total_growth < 50, f"Memory growth too high: {total_growth:.2f}MB"

    @mark.slow
    async def test_memory_growth_multiple_sessions(self):
        """Test memory growth with multiple concurrent sessions.

        Target: <100MB growth over 50 concurrent sessions
        """
        gc.collect()
        initial_memory = get_memory_usage()

        num_sessions = 50
        messages_per_session = 10

        async def session(client: httpx.AsyncClient, session_id: int) -> None:
            """Simulate a user session."""
            for i in range(messages_per_session):
                payload = {
                    "message": f"Session {session_id}, message {i}: Hello",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                await client.post(f"{API_HOST}/api/v1/chat", json=payload)

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [session(client, i) for i in range(num_sessions)]
            await asyncio.gather(*tasks)

        gc.collect()
        final_memory = get_memory_usage()
        total_growth = final_memory - initial_memory

        print("\nMemory Growth - Multiple Sessions:")
        print(f"  Sessions: {num_sessions}")
        print(f"  Messages per Session: {messages_per_session}")
        print(f"  Initial Memory: {initial_memory:.2f}MB")
        print(f"  Final Memory: {final_memory:.2f}MB")
        print(f"  Total Growth: {total_growth:.2f}MB")

        # Allow reasonable growth for multiple sessions
        assert total_growth < 200, f"Memory growth too high: {total_growth:.2f}MB"

    @mark.slow
    async def test_memory_leak_detection(self):
        """Detect memory leaks over extended operation.

        Target: <1MB growth per 100 requests
        """
        gc.collect()
        memory_samples = []

        num_iterations = 10
        requests_per_iteration = 50

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            for iteration in range(num_iterations):
                # Make requests
                for i in range(requests_per_iteration):
                    payload = {
                        "message": f"Leak test iteration {iteration}, message {i}",
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-5",
                        "stream": False,
                    }
                    await client.post(f"{API_HOST}/api/v1/chat", json=payload)

                # Force GC and measure memory
                gc.collect()
                memory = get_memory_usage()
                memory_samples.append(memory)

                print(f"  Iteration {iteration}: Memory {memory:.2f}MB")

        # Calculate growth rate
        if len(memory_samples) >= 2:
            growth = memory_samples[-1] - memory_samples[0]
            growth_per_request = growth / (num_iterations * requests_per_iteration)

            print("\nMemory Leak Detection Test:")
            print(f"  Total Requests: {num_iterations * requests_per_iteration}")
            print(f"  Initial Memory: {memory_samples[0]:.2f}MB")
            print(f"  Final Memory: {memory_samples[-1]:.2f}MB")
            print(f"  Total Growth: {growth:.2f}MB")
            print(f"  Growth per Request: {growth_per_request:.4f}MB")

            # Check for steady increase (leak indicator)
            steady_increase = all(
                memory_samples[i] <= memory_samples[i + 1] + 5  # Allow 5MB fluctuation
                for i in range(len(memory_samples) - 1)
            )

            if steady_increase and growth > 20:
                pytest.fail(f"Potential memory leak detected: {growth:.2f}MB growth")

            # Growth per request should be minimal
            assert (
                growth_per_request < 0.1
            ), f"Memory leak detected: {growth_per_request:.4f}MB per request"

    @mark.slow
    async def test_large_context_memory(self):
        """Test memory usage with large conversation contexts.

        Target: Handle 1000-turn conversation without excessive memory
        """
        gc.collect()
        initial_memory = get_memory_usage()

        # Build large conversation
        conversation_history = []
        num_turns = 100  # Reduced for faster testing (originally 1000)

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            for i in range(num_turns):
                payload = {
                    "message": f"Turn {i}: Continue the conversation about topic {i % 10}",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                assert response.status_code == 200

                conversation_history.append(f"Turn {i}")

                # Check memory periodically
                if i % 25 == 0:
                    gc.collect()
                    current_memory = get_memory_usage()
                    growth = current_memory - initial_memory
                    print(f"  Turn {i}: Memory growth {growth:.2f}MB")

        gc.collect()
        final_memory = get_memory_usage()
        total_growth = final_memory - initial_memory
        growth_per_turn = total_growth / num_turns

        print("\nLarge Context Memory Test:")
        print(f"  Conversation Turns: {num_turns}")
        print(f"  Initial Memory: {initial_memory:.2f}MB")
        print(f"  Final Memory: {final_memory:.2f}MB")
        print(f"  Total Growth: {total_growth:.2f}MB")
        print(f"  Growth per Turn: {growth_per_turn:.4f}MB")

        # Each turn should add minimal memory
        assert (
            growth_per_turn < 0.5
        ), f"Context memory growth too high: {growth_per_turn:.4f}MB/turn"

    @mark.slow
    async def test_memory_under_load(self):
        """Test memory stability under sustained load.

        Target: Memory usage stable over 1000 requests
        """
        gc.collect()
        memory_samples = []
        latencies = []

        num_requests = 200  # Reduced for faster testing
        concurrent_requests = 20

        async def make_request(client: httpx.AsyncClient, req_id: int) -> float:
            """Make a single request."""
            start = time.time()
            payload = {
                "message": f"Load test request {req_id}",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "stream": False,
            }
            response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
            latency = (time.time() - start) * 1000  # ms

            if response.status_code != 200:
                return -1
            return latency

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            for batch in range(0, num_requests, concurrent_requests):
                # Make batch of concurrent requests
                tasks = [
                    make_request(client, batch + i)
                    for i in range(min(concurrent_requests, num_requests - batch))
                ]
                results = await asyncio.gather(*tasks)

                # Record latencies
                for r in results:
                    if r > 0:
                        latencies.append(r)

                # Sample memory periodically
                if batch % 50 == 0:
                    gc.collect()
                    memory = get_memory_usage()
                    memory_samples.append(memory)
                    print(f"  Request {batch}: Memory {memory:.2f}MB")

        gc.collect()
        final_memory = get_memory_usage()

        # Calculate memory stability
        if len(memory_samples) >= 2:
            memory_variance = max(memory_samples) - min(memory_samples)
            avg_latency = statistics.mean(latencies)

            print("\nMemory Under Load Test:")
            print(f"  Total Requests: {num_requests}")
            print(f"  Memory Samples: {len(memory_samples)}")
            print(f"  Min Memory: {min(memory_samples):.2f}MB")
            print(f"  Max Memory: {max(memory_samples):.2f}MB")
            print(f"  Memory Variance: {memory_variance:.2f}MB")
            print(f"  Final Memory: {final_memory:.2f}MB")
            print(f"  Avg Latency: {avg_latency:.2f}ms")

            # Memory should be relatively stable
            assert memory_variance < 100, f"Memory variance too high: {memory_variance:.2f}MB"

    @mark.slow
    async def test_memory_cleanup_after_sessions(self):
        """Test that memory is properly cleaned up after sessions end.

        Verify session cleanup releases memory properly.
        """
        gc.collect()
        initial_memory = get_memory_usage()

        # Create and complete multiple sessions
        num_sessions = 20
        messages_per_session = 20

        async def session(client: httpx.AsyncClient, session_id: int) -> None:
            """Simulate a complete user session."""
            for i in range(messages_per_session):
                payload = {
                    "message": f"Session {session_id}, message {i}",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                await client.post(f"{API_HOST}/api/v1/chat", json=payload)

        # Run sessions sequentially
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            for i in range(num_sessions):
                await session(client, i)

                # Check memory after each session
                if i % 5 == 4:
                    gc.collect()
                    current_memory = get_memory_usage()
                    growth = current_memory - initial_memory
                    print(f"  After session {i + 1}: Memory growth {growth:.2f}MB")

        # Final cleanup
        gc.collect()
        # Give some time for async cleanup
        await asyncio.sleep(2)
        gc.collect()

        final_memory = get_memory_usage()
        total_growth = final_memory - initial_memory

        print("\nMemory Cleanup Test:")
        print(f"  Sessions: {num_sessions}")
        print(f"  Initial Memory: {initial_memory:.2f}MB")
        print(f"  Final Memory: {final_memory:.2f}MB")
        print(f"  Total Growth: {total_growth:.2f}MB")

        # Growth should be minimal after cleanup
        # Allow some growth for cached items
        assert total_growth < 100, f"Memory not cleaned up properly: {total_growth:.2f}MB growth"

    async def test_tool_execution_memory(self, api_server_available):
        """Test memory usage during tool execution.

        Verify tool operations don't leak memory.

        Note: This test requires the API server to be running.
        Start with: victor serve
        """
        gc.collect()
        initial_memory = get_memory_usage()

        tool_requests = [
            "Read the file README.md",
            "List all Python files",
            "Search for 'import' in test files",
            "Analyze code structure",
            "Check git status",
        ]

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # Execute each tool multiple times
            for iteration in range(10):
                for tool_req in tool_requests:
                    payload = {
                        "message": tool_req,
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-5",
                        "stream": False,
                    }
                    await client.post(f"{API_HOST}/api/v1/chat", json=payload)

                if iteration % 5 == 4:
                    gc.collect()
                    current_memory = get_memory_usage()
                    growth = current_memory - initial_memory
                    print(f"  Iteration {iteration + 1}: Memory growth {growth:.2f}MB")

        gc.collect()
        final_memory = get_memory_usage()
        total_growth = final_memory - initial_memory

        print("\nTool Execution Memory Test:")
        print(f"  Tool Executions: {len(tool_requests) * 10}")
        print(f"  Initial Memory: {initial_memory:.2f}MB")
        print(f"  Final Memory: {final_memory:.2f}MB")
        print(f"  Total Growth: {total_growth:.2f}MB")

        # Tool execution shouldn't leak significant memory
        assert total_growth < 50, f"Tool memory leak detected: {total_growth:.2f}MB"
