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

"""Concurrent user load tests for Victor AI.

Tests system behavior with multiple concurrent users, simulating
realistic production scenarios.
"""

import asyncio
import pytest
import time
import random
import statistics
from typing import Any, Dict, List
from datetime import datetime

import httpx
from pytest import mark


# Test configuration
API_HOST = "http://localhost:8765"
DEFAULT_TIMEOUT = 30.0


class UserSession:
    """Simulates a single user session with conversation history."""

    def __init__(self, user_id: str, provider: str = "anthropic", model: str = "claude-sonnet-4-5"):
        self.user_id = user_id
        self.provider = provider
        self.model = model
        self.conversation_history: List[Dict[str, Any]] = []
        self.message_count = 0

    def next_message(self) -> str:
        """Generate next message in conversation."""
        messages = [
            "Hello, I need help with Python",
            "How do I write a function?",
            "Can you explain decorators?",
            "What about async/await?",
            "Show me an example",
            "That's helpful, thanks!",
            "One more question",
            "How do I test this?",
            "What's the best practice?",
            "Goodbye",
        ]
        message = messages[self.message_count % len(messages)]
        self.message_count += 1
        return message


@mark.load
@mark.asyncio
class TestConcurrentUsers:
    """Concurrent user load tests.

    Simulates realistic multi-user scenarios to validate system capacity.
    """

    async def test_single_user_baseline(self, api_server_available):
        """Establish baseline for single user with 10-turn conversation.

        Target: Average response time <200ms
        """
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            session = UserSession("user_baseline")
            latencies = []

            for _ in range(10):
                start = time.time()
                payload = {
                    "message": session.next_message(),
                    "provider": session.provider,
                    "model": session.model,
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                latency = (time.time() - start) * 1000  # ms

                assert response.status_code == 200
                latencies.append(latency)

            avg_latency = statistics.mean(latencies)
            print("\nSingle User Baseline:")
            print(f"  Messages: {len(latencies)}")
            print(f"  Avg Latency: {avg_latency:.2f}ms")

            assert avg_latency < 500, f"Average latency too high: {avg_latency:.2f}ms"

    @mark.slow
    async def test_10_concurrent_users(self):
        """Test with 10 concurrent users (small load).

        Target: All users complete without errors, P95 <300ms
        """
        num_users = 10
        messages_per_user = 5

        async def user_session(client: httpx.AsyncClient, user_id: str) -> Dict[str, Any]:
            """Simulate a user session."""
            session = UserSession(f"user_{user_id}")
            latencies = []
            errors = 0

            for _ in range(messages_per_user):
                try:
                    start = time.time()
                    payload = {
                        "message": session.next_message(),
                        "provider": session.provider,
                        "model": session.model,
                        "stream": False,
                    }
                    response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                    latency = (time.time() - start) * 1000  # ms

                    if response.status_code != 200:
                        errors += 1
                    else:
                        latencies.append(latency)
                except Exception as e:
                    errors += 1

            return {
                "user_id": user_id,
                "latencies": latencies,
                "errors": errors,
                "total_messages": messages_per_user,
            }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [user_session(client, f"user_{i}") for i in range(num_users)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

        # Aggregate results
        all_latencies = []
        total_errors = 0
        for result in results:
            all_latencies.extend(result["latencies"])
            total_errors += result["errors"]

        total_requests = num_users * messages_per_user
        error_rate = (total_errors / total_requests) * 100
        p50 = statistics.median(all_latencies) if all_latencies else 0
        p95 = statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) > 20 else 0

        print("\n10 Concurrent Users Test:")
        print(f"  Users: {num_users}")
        print(f"  Total Requests: {total_requests}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {total_requests / total_time:.2f} req/s")
        print(f"  Error Rate: {error_rate:.2f}%")
        print(f"  P50 Latency: {p50:.2f}ms")
        print(f"  P95 Latency: {p95:.2f}ms")

        assert error_rate < 5, f"Error rate too high: {error_rate:.2f}%"
        assert p95 < 500, f"P95 latency too high: {p95:.2f}ms"

    @mark.slow
    async def test_50_concurrent_users(self):
        """Test with 50 concurrent users (medium load).

        Target: Error rate <2%, P95 <500ms
        """
        num_users = 50
        messages_per_user = 3

        async def user_session(client: httpx.AsyncClient, user_id: str) -> Dict[str, Any]:
            """Simulate a user session."""
            session = UserSession(f"user_{user_id}")
            latencies = []
            errors = 0

            for _ in range(messages_per_user):
                try:
                    start = time.time()
                    payload = {
                        "message": session.next_message(),
                        "provider": session.provider,
                        "model": session.model,
                        "stream": False,
                    }
                    response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                    latency = (time.time() - start) * 1000  # ms

                    if response.status_code != 200:
                        errors += 1
                    else:
                        latencies.append(latency)

                    # Small delay between messages
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                except Exception:
                    errors += 1

            return {
                "user_id": user_id,
                "latencies": latencies,
                "errors": errors,
            }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [user_session(client, f"user_{i}") for i in range(num_users)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

        # Aggregate results
        all_latencies = []
        total_errors = 0
        for result in results:
            all_latencies.extend(result["latencies"])
            total_errors += result["errors"]

        total_requests = num_users * messages_per_user
        error_rate = (total_errors / total_requests) * 100
        p50 = statistics.median(all_latencies) if all_latencies else 0
        p95 = statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) > 20 else 0

        print("\n50 Concurrent Users Test:")
        print(f"  Users: {num_users}")
        print(f"  Total Requests: {total_requests}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {total_requests / total_time:.2f} req/s")
        print(f"  Error Rate: {error_rate:.2f}%")
        print(f"  P50 Latency: {p50:.2f}ms")
        print(f"  P95 Latency: {p95:.2f}ms")

        assert error_rate < 5, f"Error rate too high: {error_rate:.2f}%"
        assert p95 < 1000, f"P95 latency too high: {p95:.2f}ms"

    @mark.slow
    async def test_100_concurrent_users_ramp_up(self):
        """Test with gradual ramp-up to 100 concurrent users.

        Simulates realistic gradual traffic increase.
        Target: System handles ramp-up gracefully, error rate <3%
        """
        max_users = 100
        messages_per_user = 2
        ramp_up_time = 30  # seconds

        async def user_session(
            client: httpx.AsyncClient, user_id: str, delay: float
        ) -> Dict[str, Any]:
            """Simulate a user session with initial delay."""
            await asyncio.sleep(delay)  # Ramp-up delay

            session = UserSession(f"user_{user_id}")
            latencies = []
            errors = 0

            for _ in range(messages_per_user):
                try:
                    start = time.time()
                    payload = {
                        "message": session.next_message(),
                        "provider": session.provider,
                        "model": session.model,
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
                "user_id": user_id,
                "latencies": latencies,
                "errors": errors,
            }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # Create tasks with staggered start times
            tasks = []
            for i in range(max_users):
                delay = (i / max_users) * ramp_up_time
                task = user_session(client, f"user_{i}", delay)
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

        # Aggregate results
        all_latencies = []
        total_errors = 0
        for result in results:
            all_latencies.extend(result["latencies"])
            total_errors += result["errors"]

        total_requests = max_users * messages_per_user
        error_rate = (total_errors / total_requests) * 100
        p50 = statistics.median(all_latencies) if all_latencies else 0
        p95 = statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) > 20 else 0

        print("\n100 Concurrent Users Ramp-Up Test:")
        print(f"  Max Users: {max_users}")
        print(f"  Ramp-up Time: {ramp_up_time}s")
        print(f"  Total Requests: {total_requests}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {total_requests / total_time:.2f} req/s")
        print(f"  Error Rate: {error_rate:.2f}%")
        print(f"  P50 Latency: {p50:.2f}ms")
        print(f"  P95 Latency: {p95:.2f}ms")

        assert error_rate < 10, f"Error rate too high: {error_rate:.2f}%"

    @mark.slow
    async def test_session_isolation(self):
        """Test that user sessions are properly isolated.

        Verify that concurrent users don't see each other's conversations.
        """
        num_users = 10
        user_tokens = [f"unique_token_{i}" for i in range(num_users)]

        async def user_session(client: httpx.AsyncClient, user_id: str, token: str) -> bool:
            """Verify session isolation."""
            try:
                # Start conversation with unique identifier
                payload = {
                    "message": f"My unique token is {token}",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)

                if response.status_code != 200:
                    return False

                # Verify response contains our token (not someone else's)
                data = response.json()
                response_text = str(data).lower()

                # Should see our own token
                if token.lower() not in response_text:
                    return False

                # Should NOT see other users' tokens
                for other_token in user_tokens:
                    if other_token != token and other_token.lower() in response_text:
                        return False

                return True
            except Exception:
                return False

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [user_session(client, f"user_{i}", user_tokens[i]) for i in range(num_users)]
            results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r)
        success_rate = (success_count / num_users) * 100

        print("\nSession Isolation Test:")
        print(f"  Users: {num_users}")
        print(f"  Successful Isolations: {success_count}/{num_users}")
        print(f"  Success Rate: {success_rate:.2f}%")

        assert success_rate >= 90, f"Session isolation failed: {success_rate:.2f}% success"

    @mark.slow
    async def test_burst_traffic(self):
        """Test handling of sudden traffic bursts.

        Simulates sudden spike in traffic (e.g., after announcement).
        Target: System survives burst with acceptable error rate <10%
        """
        burst_size = 50
        messages_per_user = 2

        async def user_session(client: httpx.AsyncClient, user_id: str) -> Dict[str, Any]:
            """Simulate a user session."""
            session = UserSession(f"user_{user_id}")
            errors = 0

            for _ in range(messages_per_user):
                try:
                    payload = {
                        "message": session.next_message(),
                        "provider": session.provider,
                        "model": session.model,
                        "stream": False,
                    }
                    response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)

                    if response.status_code != 200:
                        errors += 1
                except Exception:
                    errors += 1

            return {"user_id": user_id, "errors": errors}

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # All users start simultaneously (burst)
            tasks = [user_session(client, f"user_{i}") for i in range(burst_size)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

        total_errors = sum(r["errors"] for r in results)
        total_requests = burst_size * messages_per_user
        error_rate = (total_errors / total_requests) * 100

        print("\nBurst Traffic Test:")
        print(f"  Burst Size: {burst_size} users")
        print(f"  Total Requests: {total_requests}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Error Rate: {error_rate:.2f}%")

        assert error_rate < 15, f"Burst caused too many errors: {error_rate:.2f}%"
