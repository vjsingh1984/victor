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

"""Locust load test file for Victor AI API endpoints.

This file simulates realistic API load testing for production deployment.
Run with: locust -f tests/load/locustfiles/locustfile_api.py

Or headless: locust -f tests/load/locustfiles/locustfile_api.py --headless --users 100 --spawn-rate 10 --run-time 5m
"""

import random
import time
from datetime import datetime
from typing import Dict, List

from locust import HttpUser, task, between, events, tag


class APIUser(HttpUser):
    """Simulates API user behavior.

    Weight distribution:
    - 70% chat requests (most common)
    - 15% health checks (monitoring)
    - 10% model queries (configuration)
    - 5% streaming requests (advanced users)
    """

    # Wait between tasks: 1-3 seconds (realistic user thinking time)
    wait_time = between(1, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider = "anthropic"
        self.model = "claude-sonnet-4-5"
        self.session_id = f"session_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

    def on_start(self):
        """Initialize user session."""
        # Perform health check on start
        self.client.get("/api/v1/health", name="/api/v1/health")

    @tag("chat", "core")
    @task(7)
    def chat_request(self):
        """Send chat request (core task)."""
        messages = [
            "Hello, how are you?",
            "What's the weather like?",
            "Explain Python decorators",
            "How do I parse JSON?",
            "What is machine learning?",
            "Write a function to sort a list",
            "What's the difference between list and tuple?",
            "Explain async/await",
            "How do I write a test in pytest?",
            "What is a REST API?",
            "Debug this code: def foo(): return",
            "Refactor this function",
            "Add error handling",
            "Optimize this loop",
            "What's the best practice for logging?",
        ]

        payload = {
            "message": random.choice(messages),
            "provider": self.provider,
            "model": self.model,
            "stream": False,
        }

        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="/chat",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @tag("health", "monitoring")
    @task(2)
    def health_check(self):
        """Perform health check (monitoring)."""
        with self.client.get(
            "/api/v1/health",
            name="/api/v1/health",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @tag("models", "config")
    @task(1)
    def list_models(self):
        """List available models (configuration)."""
        with self.client.get(
            "/api/v1/models",
            name="/api/v1/models",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to list models: {response.status_code}")

    @tag("stream", "advanced")
    @task(1)
    def streaming_request(self):
        """Send streaming request (advanced users)."""
        payload = {
            "message": "Tell me a short joke about programming",
            "provider": self.provider,
            "model": self.model,
            "stream": True,
        }

        with self.client.post(
            "/chat/stream",
            json=payload,
            catch_response=True,
            name="/chat/stream",
            timeout=30.0,
        ) as response:
            if response.status_code == 200:
                # Read some of the stream
                try:
                    content = response.content
                    if len(content) > 0:
                        response.success()
                    else:
                        response.failure("Empty response")
                except Exception as e:
                    response.failure(f"Stream read error: {e}")
            else:
                response.failure(f"Stream request failed: {response.status_code}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate test summary on completion."""
    print("\n" + "=" * 80)
    print("VICTOR AI API LOAD TEST SUMMARY")
    print("=" * 80)
    print(f"Test completed at: {datetime.now().isoformat()}")

    if environment.stats:
        stats = environment.stats

        print(f"\nTotal Requests: {stats.total.num_requests}")
        print(f"Successful Requests: {stats.total.num_requests - stats.total.num_failures}")
        print(f"Failed Requests: {stats.total.num_failures}")
        print(f"Failure Rate: {(stats.total.num_failures / stats.total.num_requests * 100):.2f}%")

        print("\nResponse Times:")
        print(f"  Min: {stats.total.min_response_time:.0f}ms")
        print(f"  Avg: {stats.total.avg_response_time:.0f}ms")
        print(f"  Median: {stats.total.median_response_time:.0f}ms")
        print(f"  95th percentile: {stats.total.get_response_time_percentile(0.95):.0f}ms")
        print(f"  99th percentile: {stats.total.get_response_time_percentile(0.99):.0f}ms")

        print("\nThroughput:")
        print(f"  Requests/second: {stats.total.total_avg_rps:.2f}")
        print(f"  Total RPS: {stats.total.total_rps:.2f}")

    print("=" * 80 + "\n")


class WriteHeavyUser(APIUser):
    """User that performs mostly write operations (chat requests)."""

    @task(10)
    def chat_request(self):
        """Override to make chat more frequent."""
        super().chat_request()

    @task(1)
    def health_check(self):
        """Occasional health check."""
        super().health_check()


class ReadHeavyUser(APIUser):
    """User that performs mostly read operations (health, models)."""

    @tag("health", "monitoring")
    @task(6)
    def health_check(self):
        """Frequent health checks."""
        super().health_check()

    @tag("models", "config")
    @task(3)
    def list_models(self):
        """Frequent model queries."""
        super().list_models()

    @tag("chat", "core")
    @task(1)
    def chat_request(self):
        """Occasional chat requests."""
        super().chat_request()
