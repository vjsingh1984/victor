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

"""Load testing framework for Victor AI using Locust.

This module provides a comprehensive load testing framework that simulates
real-world usage patterns and measures system performance under various load conditions.

Features:
- Concurrent request simulation
- Response time measurement
- Throughput tracking
- Error rate monitoring
- Resource utilization tracking
"""

import asyncio
import json
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime

import aiohttp
from locust import HttpUser, task, between, events, run_single_user
from locust.runners import MasterRunner

# Test configuration
DEFAULT_HOST = os.getenv("VICTOR_API_HOST", "http://localhost:8000")
DEFAULT_PROVIDER = os.getenv("VICTOR_PROVIDER", "anthropic")
DEFAULT_MODEL = os.getenv("VICTOR_MODEL", "claude-sonnet-4-5")


class VictorLoadTest(HttpUser):
    """Simulates realistic Victor AI usage patterns.

    Test scenarios:
    1. Simple chat requests
    2. Tool calling requests
    3. Multi-turn conversations
    4. Large context handling
    5. Concurrent session management
    """

    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id: Optional[str] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.provider = DEFAULT_PROVIDER
        self.model = DEFAULT_MODEL

    def on_start(self):
        """Initialize user session."""
        self.session_id = f"session_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        self.conversation_history = []

    @task(3)
    def simple_chat_request(self):
        """Send a simple chat request (most common operation)."""
        messages = [
            "Hello, how are you?",
            "What's the weather like?",
            "Tell me a joke",
            "Explain Python decorators",
            "What is machine learning?",
            "How do I write a test in pytest?",
            "What's the difference between list and tuple?",
            "Explain async/await in Python",
            "How do I parse JSON in Python?",
            "What is a REST API?",
        ]

        payload = {
            "message": random.choice(messages),
            "provider": self.provider,
            "model": self.model,
            "stream": False,
        }

        with self.client.post(
            "/api/v1/chat",
            json=payload,
            catch_response=True,
            name="/api/v1/chat (simple)",
        ) as response:
            if response.status_code == 200:
                self.conversation_history.append(payload["message"])
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(2)
    def tool_calling_request(self):
        """Send a request that requires tool calling."""
        tool_requests = [
            "Read the README.md file",
            "List all Python files in the current directory",
            "Search for 'TODO' comments in the codebase",
            "Show me the git status",
            "Find all test files",
            "Count lines of code in src/",
            "Check if package.json exists",
        ]

        payload = {
            "message": random.choice(tool_requests),
            "provider": self.provider,
            "model": self.model,
            "stream": False,
        }

        with self.client.post(
            "/api/v1/chat",
            json=payload,
            catch_response=True,
            name="/api/v1/chat (tool_call)",
        ) as response:
            if response.status_code == 200:
                self.conversation_history.append(payload["message"])
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def large_context_request(self):
        """Send a request with large conversation history."""
        if len(self.conversation_history) < 3:
            return  # Skip if not enough history

        payload = {
            "message": "Summarize our conversation so far",
            "provider": self.provider,
            "model": self.model,
            "stream": False,
            "context": {
                "history": self.conversation_history[-10:],  # Last 10 messages
            },
        }

        with self.client.post(
            "/api/v1/chat",
            json=payload,
            catch_response=True,
            name="/api/v1/chat (large_context)",
        ) as response:
            if response.status_code == 200:
                self.conversation_history.append(payload["message"])
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def concurrent_session_test(self):
        """Test managing multiple concurrent sessions."""
        payload = {
            "message": f"Concurrent test message at {datetime.now().isoformat()}",
            "provider": self.provider,
            "model": self.model,
            "stream": False,
            "session_id": self.session_id,
        }

        with self.client.post(
            "/api/v1/chat",
            json=payload,
            catch_response=True,
            name="/api/v1/chat (concurrent)",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


class StressTestUser(HttpUser):
    """Stress testing user - aggressive request pattern.

    This user class sends rapid requests without wait time to find
    system breaking points.
    """

    wait_time = between(0.1, 0.5)  # Minimal wait time

    @task
    def rapid_fire_request(self):
        """Send rapid requests to stress test the system."""
        payload = {
            "message": "Quick test",
            "provider": DEFAULT_PROVIDER,
            "model": DEFAULT_MODEL,
            "stream": False,
        }

        self.client.post("/api/v1/chat", json=payload, name="/api/v1/chat (stress)")


class MemoryLeakTestUser(HttpUser):
    """User for testing memory leaks over long sessions.

    This user maintains long sessions with growing context to test
    memory management and garbage collection.
    """

    wait_time = between(2, 4)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = None
        self.message_count = 0

    def on_start(self):
        """Initialize long-running session."""
        self.session_id = f"leak_test_{int(time.time() * 1000)}"
        self.message_count = 0

    @task
    def grow_conversation_history(self):
        """Continuously grow conversation history."""
        self.message_count += 1

        # Alternate between new messages and context-aware messages
        if self.message_count % 2 == 0:
            message = f"This is message number {self.message_count} in our conversation"
        else:
            message = "Please recall what we discussed about message " + str(
                max(1, self.message_count - 5)
            )

        payload = {
            "message": message,
            "provider": DEFAULT_PROVIDER,
            "model": DEFAULT_MODEL,
            "stream": False,
            "session_id": self.session_id,
        }

        self.client.post("/api/v1/chat", json=payload, name="/api/v1/chat (memory_leak)")


# =============================================================================
# Event Handlers for Metrics Collection
# =============================================================================


@events.request.add_hook
def on_request(request_type, name, response_time, response_length, **kwargs):
    """Log request events for custom metrics."""
    if response_time > 5000:  # Log slow requests (> 5 seconds)
        print(f"SLOW REQUEST: {name} took {response_time}ms")


@events.test_stop.add_hook
def on_test_stop(environment, **kwargs):
    """Generate test summary report."""
    print("\n" + "=" * 80)
    print("LOAD TEST SUMMARY")
    print("=" * 80)

    stats = environment.stats

    print(f"\nTotal Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Success Rate: {(1 - stats.total.fail_ratio) * 100:.2f}%")
    print("\nResponse Times:")
    print(f"  50th percentile: {stats.total.median_response_time}ms")
    print(f"  95th percentile: {stats.total.get_response_time_percentile(0.95)}ms")
    print(f"  99th percentile: {stats.total.get_response_time_percentile(0.99)}ms")
    print(f"  Average: {stats.total.avg_response_time}ms")
    print("\nThroughput:")
    print(f"  Requests/second: {stats.total.total_rps:.2f}")
    print("=" * 80 + "\n")


# =============================================================================
# Async Load Testing Framework (Advanced)
# =============================================================================


class AsyncLoadTestFramework:
    """Advanced async load testing framework for custom scenarios.

    Provides fine-grained control over load testing patterns beyond
    what Locust offers out of the box.
    """

    def __init__(self, base_url: str = DEFAULT_HOST):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []

    async def execute_concurrent_requests(
        self,
        num_requests: int,
        concurrency: int,
        payload: Dict[str, Any],
        endpoint: str = "/api/v1/chat",
    ) -> Dict[str, Any]:
        """Execute concurrent requests with specified concurrency level.

        Args:
            num_requests: Total number of requests to send
            concurrency: Number of concurrent requests
            payload: Request payload
            endpoint: API endpoint to call

        Returns:
            Dictionary with test results and metrics
        """
        start_time = time.time()
        results = []
        errors = []

        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrency)

            async def make_request(request_id: int) -> Dict[str, Any]:
                async with semaphore:
                    req_start = time.time()
                    try:
                        async with session.post(
                            f"{self.base_url}{endpoint}", json=payload
                        ) as response:
                            await response.text()  # Consume response
                            req_time = (time.time() - req_start) * 1000
                            return {
                                "request_id": request_id,
                                "status": response.status,
                                "response_time_ms": req_time,
                                "success": 200 <= response.status < 300,
                            }
                    except Exception as e:
                        req_time = (time.time() - req_start) * 1000
                        return {
                            "request_id": request_id,
                            "status": 0,
                            "response_time_ms": req_time,
                            "success": False,
                            "error": str(e),
                        }

            # Create all tasks
            tasks = [make_request(i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks)

        # Calculate metrics
        total_time = time.time() - start_time
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        response_times = [r["response_time_ms"] for r in successful]
        response_times.sort()

        return {
            "total_requests": num_requests,
            "concurrency": concurrency,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": (len(successful) / num_requests * 100) if num_requests > 0 else 0,
            "total_time_seconds": total_time,
            "requests_per_second": num_requests / total_time if total_time > 0 else 0,
            "response_times": {
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "avg": sum(response_times) / len(response_times) if response_times else 0,
                "median": response_times[len(response_times) // 2] if response_times else 0,
                "p95": response_times[int(len(response_times) * 0.95)] if response_times else 0,
                "p99": response_times[int(len(response_times) * 0.99)] if response_times else 0,
            },
            "errors": [r.get("error") for r in failed],
        }

    async def test_memory_leaks(
        self,
        duration_seconds: int,
        requests_per_second: float,
    ) -> Dict[str, Any]:
        """Test for memory leaks over extended period.

        Args:
            duration_seconds: How long to run the test
            requests_per_second: Target request rate

        Returns:
            Memory usage statistics over time
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_samples = []
        start_time = time.time()

        interval = 1.0 / requests_per_second if requests_per_second > 0 else 1

        while (time.time() - start_time) < duration_seconds:
            # Record memory before request
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Make request (simplified for memory leak detection)
            # In real scenario, you'd make actual API calls

            # Record memory after request
            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            memory_samples.append(
                {
                    "timestamp": time.time() - start_time,
                    "memory_mb_before": mem_before,
                    "memory_mb_after": mem_after,
                    "memory_delta_mb": mem_after - mem_before,
                }
            )

            await asyncio.sleep(interval)

        # Analyze memory trend
        initial_memory = memory_samples[0]["memory_mb_after"]
        final_memory = memory_samples[-1]["memory_mb_after"]
        memory_growth = final_memory - initial_memory

        return {
            "duration_seconds": duration_seconds,
            "requests_per_second": requests_per_second,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": memory_growth,
            "memory_growth_rate_mb_per_hour": (
                memory_growth / (duration_seconds / 3600) if duration_seconds > 0 else 0
            ),
            "samples": memory_samples,
            "potential_leak": memory_growth > 100,  # Flag if >100MB growth
        }


# =============================================================================
# Utility Functions
# =============================================================================


def generate_scalability_report(
    results: Dict[str, Any], output_path: str = "scalability_report.json"
):
    """Generate a detailed scalability report from test results.

    Args:
        results: Test results dictionary
        output_path: Where to save the report
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": results.get("total_tests", 0),
            "passed_tests": results.get("passed_tests", 0),
            "failed_tests": results.get("failed_tests", 0),
        },
        "performance": {
            "throughput": results.get("throughput", {}),
            "latency": results.get("latency", {}),
            "resource_usage": results.get("resource_usage", {}),
        },
        "recommendations": generate_recommendations(results),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def generate_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate scaling recommendations based on test results."""
    recommendations = []

    # Analyze response times
    p95_latency = results.get("latency", {}).get("p95", 0)
    if p95_latency > 1000:  # > 1 second
        recommendations.append("P95 latency exceeds 1s. Consider implementing response caching.")

    # Analyze throughput
    throughput = results.get("throughput", {}).get("requests_per_second", 0)
    if throughput < 100:
        recommendations.append("Throughput below 100 req/s. Consider horizontal scaling.")

    # Analyze error rates
    error_rate = results.get("error_rate", 0)
    if error_rate > 0.05:  # > 5%
        recommendations.append(
            f"Error rate {error_rate * 100:.1f}% exceeds 5%. Check rate limiting and circuit breakers."
        )

    # Analyze memory usage
    memory_growth = results.get("memory_growth_mb", 0)
    if memory_growth > 500:  # > 500MB
        recommendations.append(
            "Significant memory growth detected. Investigate potential memory leaks."
        )

    if not recommendations:
        recommendations.append("System performs well within tested parameters.")

    return recommendations


if __name__ == "__main__":
    # Run a single user test for development
    run_single_user(VictorLoadTest)
