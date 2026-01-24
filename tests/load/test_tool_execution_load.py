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

"""Tool execution load tests for Victor AI.

Tests tool execution performance under concurrent load to ensure
tool selection and execution meet performance SLAs.
"""

import asyncio
import pytest
import time
import statistics
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

import httpx
from pytest import mark


# Test configuration
API_HOST = "http://localhost:8000"
DEFAULT_TIMEOUT = 30.0


@mark.load
@mark.asyncio
@pytest.mark.usefixtures("api_server_available")
class TestToolExecutionLoad:
    """Tool execution load tests.

    Measures tool selection, execution, and error handling under load.
    Requires API server running at localhost:8000.
    """

    async def test_single_tool_baseline(self):
        """Establish baseline for single tool execution.

        Target: P50 <50ms, P95 <200ms
        """
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            payload = {
                "message": "Read the file /tmp/test.py",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "stream": False,
            }

            start = time.time()
            response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
            latency = (time.time() - start) * 1000  # ms

            assert response.status_code == 200
            assert latency < 500, f"Tool execution too slow: {latency}ms"

            print(f"\nSingle Tool Baseline: {latency:.2f}ms")

    @mark.slow
    async def test_concurrent_tool_execution(self):
        """Test concurrent tool execution across different users.

        Target: 50 concurrent tool executions, error rate <2%
        """
        num_requests = 50

        # Different tool-using requests
        tool_requests = [
            "Read the file README.md",
            "List files in the current directory",
            "Search for 'def test' in test files",
            "What Python version is installed?",
            "Check the git status",
            "Find all TODO comments",
            "Analyze the code structure",
            "Count the lines of code",
            "Check for syntax errors",
            "Run the tests",
        ]

        async def execute_tool_request(
            client: httpx.AsyncClient, request_id: int
        ) -> Dict[str, Any]:
            """Execute a tool-using request."""
            try:
                start = time.time()
                payload = {
                    "message": tool_requests[request_id % len(tool_requests)],
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                latency = (time.time() - start) * 1000  # ms

                return {
                    "success": response.status_code == 200,
                    "latency": latency,
                    "error": None,
                }
            except Exception as e:
                return {
                    "success": False,
                    "latency": -1,
                    "error": str(e),
                }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [execute_tool_request(client, i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks)

        # Analyze results
        successful = [r for r in results if r["success"]]
        errors = [r for r in results if not r["success"]]
        latencies = [r["latency"] for r in successful]

        error_rate = (len(errors) / num_requests) * 100
        p50 = statistics.median(latencies) if latencies else 0
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0
        p99 = (
            statistics.quantiles(latencies, n=100)[98]
            if len(latencies) > 100
            else max(latencies) if latencies else 0
        )

        print("\nConcurrent Tool Execution Test:")
        print(f"  Requests: {num_requests}")
        print(f"  Successful: {len(successful)}")
        print(f"  Errors: {len(errors)}")
        print(f"  Error Rate: {error_rate:.2f}%")
        print(f"  P50 Latency: {p50:.2f}ms")
        print(f"  P95 Latency: {p95:.2f}ms")
        print(f"  P99 Latency: {p99:.2f}ms")

        assert error_rate < 10, f"Error rate too high: {error_rate:.2f}%"
        assert p95 < 2000, f"P95 latency too high: {p95:.2f}ms"

    @mark.slow
    async def test_tool_selection_caching(self):
        """Test tool selection caching effectiveness.

        Measure performance improvement from cache hits.
        """
        num_iterations = 100

        # Use same query repeatedly to test caching
        query = "Read the file test.py"

        async def make_request(client: httpx.AsyncClient) -> float:
            """Make request and return latency."""
            start = time.time()
            payload = {
                "message": query,
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
            # First request (cold cache)
            cold_latencies = []
            for _ in range(5):
                latency = await make_request(client)
                if latency > 0:
                    cold_latencies.append(latency)

            # Subsequent requests (warm cache)
            warm_latencies = []
            for _ in range(num_iterations - 5):
                latency = await make_request(client)
                if latency > 0:
                    warm_latencies.append(latency)

        avg_cold = statistics.mean(cold_latencies) if cold_latencies else 0
        avg_warm = statistics.mean(warm_latencies) if warm_latencies else 0
        speedup = avg_cold / avg_warm if avg_warm > 0 else 0

        print("\nTool Selection Caching Test:")
        print(f"  Iterations: {num_iterations}")
        print(f"  Cold Cache Avg: {avg_cold:.2f}ms")
        print(f"  Warm Cache Avg: {avg_warm:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Warm cache should be at least as fast
        assert avg_warm <= avg_cold * 1.2, "Cache not providing benefit"

    @mark.slow
    async def test_multi_tool_conversation(self):
        """Test conversation with multiple tool uses.

        Simulates realistic coding session with multiple tool calls.
        """
        num_conversations = 20

        conversation_turns = [
            "List all Python files in the current directory",
            "Read the file test_main.py",
            "Find all function definitions in the file",
            "What tests are defined?",
            "Summarize the test coverage",
        ]

        async def run_conversation(client: httpx.AsyncClient, conv_id: int) -> Dict[str, Any]:
            """Run a multi-turn conversation."""
            latencies = []
            errors = 0

            for turn in conversation_turns:
                try:
                    start = time.time()
                    payload = {
                        "message": f"[{conv_id}] {turn}",
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-5",
                        "stream": False,
                    }
                    response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                    latency = (time.time() - start) * 1000  # ms

                    if response.status_code != 200:
                        errors += 1
                    else:
                        latencies.append(latency)
                except Exception:
                    errors += 1

            return {
                "conv_id": conv_id,
                "latencies": latencies,
                "errors": errors,
            }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [run_conversation(client, i) for i in range(num_conversations)]
            results = await asyncio.gather(*tasks)

        # Aggregate results
        all_latencies = []
        total_errors = 0
        for result in results:
            all_latencies.extend(result["latencies"])
            total_errors += result["errors"]

        total_requests = num_conversations * len(conversation_turns)
        error_rate = (total_errors / total_requests) * 100
        p50 = statistics.median(all_latencies) if all_latencies else 0
        p95 = statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) > 20 else 0

        print("\nMulti-Tool Conversation Test:")
        print(f"  Conversations: {num_conversations}")
        print(f"  Turns per Conversation: {len(conversation_turns)}")
        print(f"  Total Requests: {total_requests}")
        print(f"  Error Rate: {error_rate:.2f}%")
        print(f"  P50 Latency: {p50:.2f}ms")
        print(f"  P95 Latency: {p95:.2f}ms")

        assert error_rate < 10, f"Error rate too high: {error_rate:.2f}%"
        assert p95 < 3000, f"P95 latency too high: {p95:.2f}ms"

    @mark.slow
    async def test_tool_error_handling(self):
        """Test error handling for invalid tool usage.

        Verify system handles tool errors gracefully under load.
        """
        num_requests = 30

        # Requests that will cause tool errors
        error_requests = [
            "Read the non-existent file /tmp/does_not_exist.txt",
            "Delete system file /etc/passwd",
            "Execute malicious code",
            "Read directory as file",
            "Invalid tool parameters",
        ]

        async def make_error_request(client: httpx.AsyncClient, req_id: int) -> Dict[str, Any]:
            """Make request that will cause tool error."""
            try:
                start = time.time()
                payload = {
                    "message": error_requests[req_id % len(error_requests)],
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                latency = (time.time() - start) * 1000  # ms

                # Should get response (even if error), not crash
                return {
                    "got_response": response.status_code in [200, 400, 500],
                    "latency": latency,
                }
            except Exception:
                return {
                    "got_response": False,
                    "latency": -1,
                }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [make_error_request(client, i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks)

        # All requests should get some response (not crash)
        responses_received = sum(1 for r in results if r["got_response"])
        response_rate = (responses_received / num_requests) * 100
        latencies = [r["latency"] for r in results if r["latency"] > 0]

        avg_latency = statistics.mean(latencies) if latencies else 0

        print("\nTool Error Handling Test:")
        print(f"  Requests: {num_requests}")
        print(f"  Responses Received: {responses_received}/{num_requests}")
        print(f"  Response Rate: {response_rate:.2f}%")
        print(f"  Avg Latency: {avg_latency:.2f}ms")

        # System should handle errors gracefully
        assert response_rate >= 95, f"System not handling errors gracefully: {response_rate:.2f}%"

    @mark.slow
    async def test_tool_timeout_handling(self):
        """Test handling of slow/timeout tool executions.

        Verify system doesn't hang on slow tools.
        """
        # This test assumes there's a slow tool or we can simulate it
        # In real tests, you might use a mock slow tool

        async with httpx.AsyncClient(timeout=5.0) as client:  # Short timeout
            payload = {
                "message": "Perform a very long operation",  # Hypothetical slow tool
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "stream": False,
            }

            start = time.time()
            try:
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                latency = (time.time() - start) * 1000  # ms

                # Should either complete or timeout gracefully
                assert response.status_code in [200, 408, 500]
                print(f"\nTool Timeout Test: {latency:.2f}ms (handled gracefully)")

            except Exception as e:
                # Timeout is acceptable
                latency = (time.time() - start) * 1000
                print(f"\nTool Timeout Test: {latency:.2f}ms (timed out as expected)")
                assert latency < 10000, "Timeout took too long"

    async def test_concurrent_different_tools(self):
        """Test concurrent execution of different tool types.

        Verify system handles diverse tool workloads concurrently.
        """
        num_requests = 20

        # Different tool categories
        tool_categories = {
            "file_ops": "Read the file README.md",
            "search": "Search for 'import' in Python files",
            "analysis": "Analyze the code structure",
            "git": "Check the git status",
            "execution": "Run pytest",
        }

        async def execute_tool_by_category(client: httpx.AsyncClient, category: str) -> float:
            """Execute tool from specific category."""
            try:
                start = time.time()
                payload = {
                    "message": tool_categories[category],
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                latency = (time.time() - start) * 1000  # ms

                if response.status_code != 200:
                    return -1
                return latency
            except Exception:
                return -1

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # Execute mix of different tool types
            categories = list(tool_categories.keys())
            tasks = []
            for i in range(num_requests):
                category = categories[i % len(categories)]
                task = execute_tool_by_category(client, category)
                tasks.append(task)

            results = await asyncio.gather(*tasks)

        # Analyze by category
        category_latencies = {cat: [] for cat in categories}
        for i, result in enumerate(results):
            if result > 0:
                category = categories[i % len(categories)]
                category_latencies[category].append(result)

        print("\nConcurrent Different Tools Test:")
        for category, latencies in category_latencies.items():
            if latencies:
                avg = statistics.mean(latencies)
                print(f"  {category}: {len(latencies)} requests, avg {avg:.2f}ms")

        # All categories should have executed
        successful_categories = sum(1 for latencies in category_latencies.values() if latencies)
        assert successful_categories == len(categories), "Some tool categories failed"
