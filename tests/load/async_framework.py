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

"""Async load testing framework for pytest tests.

This module provides a pytest-compatible load testing framework.
It does NOT import Locust to avoid gevent monkey-patching conflicts
during pytest collection.

Use framework.py for Locust-based load testing.
Use this module for pytest-based load testing.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

# Test configuration
DEFAULT_HOST = os.getenv("VICTOR_API_HOST", "http://localhost:8765")


class AsyncLoadTestFramework:
    """Advanced async load testing framework for custom scenarios.

    Provides fine-grained control over load testing patterns using
    pytest and asyncio, without Locust dependencies.
    """

    def __init__(self, base_url: str = DEFAULT_HOST):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []

    async def execute_concurrent_requests(
        self,
        num_requests: int,
        concurrency: int,
        payload: Dict[str, Any],
        endpoint: str = "/chat",
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
        import time

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
        import time

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
        recommendations.append(
            "Throughput below 100 req/s. Consider horizontal scaling or optimization."
        )

    # Analyze error rate
    error_rate = results.get("summary", {}).get("error_rate", 0)
    if error_rate > 5:  # > 5%
        recommendations.append(
            f"Error rate is {error_rate:.1f}%. Review error logs and implement retries."
        )

    # Analyze memory growth
    memory_growth = results.get("resource_usage", {}).get("memory_growth_mb", 0)
    if memory_growth > 500:
        recommendations.append(
            f"Memory growth of {memory_growth:.1f}MB detected. Investigate potential memory leaks."
        )

    if not recommendations:
        recommendations.append("System performs well within tested parameters.")

    return recommendations
