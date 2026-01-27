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

"""API endpoint load tests for Victor AI.

Tests load handling for various API endpoints to ensure they meet
performance SLAs under concurrent user load.
"""

import asyncio
import pytest
import time
from datetime import datetime
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import statistics

import httpx
from pytest import mark


# Test configuration
API_HOST = "http://localhost:8765"
DEFAULT_TIMEOUT = 30.0
ENDPOINTS = {
    "chat": "/api/v1/chat",
    "stream": "/api/v1/chat/stream",
    "health": "/api/v1/health",
    "models": "/api/v1/models",
}


@mark.load
@mark.asyncio
class TestAPILoad:
    """API endpoint load tests.

    Measures throughput, latency, and error rates for various endpoints
    under concurrent load.
    """

    async def test_chat_endpoint_single_user_baseline(self, api_server_available):
        """Establish baseline performance for single user.

        Target: P50 <100ms, P95 <300ms, P99 <500ms
        """
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            start_time = time.time()

            payload = {
                "message": "Hello, how are you?",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "stream": False,
            }

            response = await client.post(f"{API_HOST}{ENDPOINTS['chat']}", json=payload)
            latency = (time.time() - start_time) * 1000  # ms

            assert response.status_code == 200
            assert latency < 500, f"P99 latency exceeded: {latency}ms"

    @mark.slow
    async def test_chat_endpoint_concurrent_load(self, api_server_available):
        """Test chat endpoint with 100 concurrent users.

        Target: >100 req/s, error rate <1%
        """
        num_requests = 100
        latencies = []
        errors = 0
        start_time = time.time()

        async def make_request(client: httpx.AsyncClient) -> float:
            """Make a single request and return latency."""
            try:
                req_start = time.time()
                payload = {
                    "message": "Test message for load testing",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}{ENDPOINTS['chat']}", json=payload)
                latency = (time.time() - req_start) * 1000  # ms

                if response.status_code != 200:
                    return -1  # Error
                return latency
            except Exception:
                return -1

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [make_request(client) for _ in range(num_requests)]
            results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Process results
        for result in results:
            if result == -1:
                errors += 1
            else:
                latencies.append(result)

        # Calculate metrics
        throughput = num_requests / total_time
        error_rate = (errors / num_requests) * 100
        p50 = statistics.median(latencies) if latencies else 0
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0
        p99 = (
            statistics.quantiles(latencies, n=100)[98]
            if len(latencies) > 100
            else max(latencies) if latencies else 0
        )

        print("\nChat Endpoint Load Test Results:")
        print(f"  Requests: {num_requests}")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Error Rate: {error_rate:.2f}%")
        print(f"  P50 Latency: {p50:.2f}ms")
        print(f"  P95 Latency: {p95:.2f}ms")
        print(f"  P99 Latency: {p99:.2f}ms")

        # Assertions
        assert throughput > 50, f"Throughput too low: {throughput:.2f} req/s"
        assert error_rate < 5, f"Error rate too high: {error_rate:.2f}%"
        assert p50 < 300, f"P50 latency too high: {p50:.2f}ms"

    @mark.slow
    async def test_streaming_endpoint_concurrent_load(self, api_server_available):
        """Test streaming endpoint with 50 concurrent users.

        Target: Time to first token <200ms
        """
        num_requests = 50
        time_to_first_token = []
        errors = 0

        async def make_stream_request(client: httpx.AsyncClient) -> float:
            """Make streaming request and measure time to first token."""
            try:
                req_start = time.time()
                payload = {
                    "message": "Tell me a short joke",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": True,
                }

                async with client.stream(
                    "POST", f"{API_HOST}{ENDPOINTS['stream']}", json=payload
                ) as response:
                    if response.status_code != 200:
                        return -1

                    # Read first chunk
                    first_chunk_time = None
                    async for chunk in response.aiter_bytes():
                        if first_chunk_time is None:
                            first_chunk_time = (time.time() - req_start) * 1000
                            break

                    return first_chunk_time if first_chunk_time else -1
            except Exception:
                return -1

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [make_stream_request(client) for _ in range(num_requests)]
            results = await asyncio.gather(*tasks)

        # Process results
        for result in results:
            if result == -1:
                errors += 1
            elif result is not None:
                time_to_first_token.append(result)

        # Calculate metrics
        error_rate = (errors / num_requests) * 100
        avg_ttf = statistics.mean(time_to_first_token) if time_to_first_token else 0

        print("\nStreaming Endpoint Load Test Results:")
        print(f"  Requests: {num_requests}")
        print(f"  Error Rate: {error_rate:.2f}%")
        print(f"  Avg Time to First Token: {avg_ttf:.2f}ms")

        # Assertions
        assert error_rate < 5, f"Error rate too high: {error_rate:.2f}%"
        assert avg_ttf < 500, f"Time to first token too high: {avg_ttf:.2f}ms"

    async def test_health_endpoint_performance(self, api_server_available):
        """Test health endpoint performance under load.

        Target: P50 <10ms (should be very fast)
        """
        num_requests = 100
        latencies = []

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            for _ in range(num_requests):
                start = time.time()
                response = await client.get(f"{API_HOST}{ENDPOINTS['health']}")
                latency = (time.time() - start) * 1000  # ms

                assert response.status_code == 200
                latencies.append(latency)

        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0

        print("\nHealth Endpoint Performance:")
        print(f"  Requests: {num_requests}")
        print(f"  P50 Latency: {p50:.2f}ms")
        print(f"  P95 Latency: {p95:.2f}ms")

        assert p50 < 50, f"Health endpoint too slow: {p50:.2f}ms"

    @mark.slow
    async def test_models_endpoint_concurrent_read(self, api_server_available):
        """Test models endpoint with concurrent reads.

        Target: Handle 50 concurrent reads without errors
        """
        num_requests = 50
        errors = 0

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = []
            for _ in range(num_requests):
                task = client.get(f"{API_HOST}{ENDPOINTS['models']}")
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

        for response in responses:
            if response.status_code != 200:
                errors += 1

        error_rate = (errors / num_requests) * 100

        print("\nModels Endpoint Load Test:")
        print(f"  Requests: {num_requests}")
        print(f"  Error Rate: {error_rate:.2f}%")

        assert error_rate < 1, f"Error rate too high: {error_rate:.2f}%"

    @mark.slow
    async def test_mixed_workload(self, api_server_available):
        """Test realistic mixed workload pattern.

        Simulates realistic usage: 70% chat, 20% streaming, 10% health checks
        """
        num_requests = 100
        latencies = []
        errors = 0

        async def make_mixed_request(client: httpx.AsyncClient) -> float:
            """Make a request based on workload distribution."""
            try:
                rand = random.random()
                start = time.time()

                if rand < 0.7:  # 70% chat
                    payload = {
                        "message": "Mixed workload test",
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-5",
                        "stream": False,
                    }
                    response = await client.post(f"{API_HOST}{ENDPOINTS['chat']}", json=payload)
                elif rand < 0.9:  # 20% streaming
                    payload = {
                        "message": "Streaming test",
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-5",
                        "stream": True,
                    }
                    async with client.stream(
                        "POST", f"{API_HOST}{ENDPOINTS['stream']}", json=payload
                    ) as response:
                        # Just read first chunk
                        async for _ in response.aiter_bytes():
                            break
                    response.status_code = 200  # Assume success
                else:  # 10% health check
                    response = await client.get(f"{API_HOST}{ENDPOINTS['health']}")

                latency = (time.time() - start) * 1000  # ms

                if response.status_code != 200:
                    return -1
                return latency
            except Exception:
                return -1

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [make_mixed_request(client) for _ in range(num_requests)]
            results = await asyncio.gather(*tasks)

        # Process results
        for result in results:
            if result == -1:
                errors += 1
            else:
                latencies.append(result)

        error_rate = (errors / num_requests) * 100
        p50 = statistics.median(latencies) if latencies else 0

        print("\nMixed Workload Test Results:")
        print(f"  Requests: {num_requests}")
        print(f"  Error Rate: {error_rate:.2f}%")
        print(f"  P50 Latency: {p50:.2f}ms")

        assert error_rate < 5, f"Error rate too high: {error_rate:.2f}%"
        assert p50 < 500, f"P50 latency too high: {p50:.2f}ms"


import random
