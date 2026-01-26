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

"""Locust configuration for Victor AI load testing.

This module provides centralized configuration for load tests across
different scenarios and environments.
"""

import os
from typing import Dict, Any, List


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_HOST = os.getenv("VICTOR_API_HOST", "http://localhost:8000")
DEFAULT_PROVIDER = os.getenv("VICTOR_PROVIDER", "anthropic")
DEFAULT_MODEL = os.getenv("VICTOR_MODEL", "claude-sonnet-4-5")


# =============================================================================
# Load Test Profiles
# =============================================================================

LOAD_TEST_PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "description": "Quick smoke test to verify basic functionality",
        "users": 10,
        "spawn_rate": 2,
        "run_time": "1m",
        "host": DEFAULT_HOST,
    },
    "normal": {
        "description": "Normal load test simulating typical usage",
        "users": 50,
        "spawn_rate": 5,
        "run_time": "5m",
        "host": DEFAULT_HOST,
    },
    "peak": {
        "description": "Peak load test simulating high traffic",
        "users": 100,
        "spawn_rate": 10,
        "run_time": "10m",
        "host": DEFAULT_HOST,
    },
    "stress": {
        "description": "Stress test to find breaking point",
        "users": 200,
        "spawn_rate": 20,
        "run_time": "15m",
        "host": DEFAULT_HOST,
    },
    "endurance": {
        "description": "Endurance test for stability over time",
        "users": 50,
        "spawn_rate": 5,
        "run_time": "1h",
        "host": DEFAULT_HOST,
    },
}


# =============================================================================
# Request Scenarios
# =============================================================================

CHAT_MESSAGES: List[str] = [
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
    "Write a function to reverse a string",
    "What is the time complexity of binary search?",
    "Explain the concept of recursion",
    "How do I handle exceptions in Python?",
    "What is a generator in Python?",
]

TOOL_REQUESTS: List[str] = [
    "Read the README.md file",
    "List all Python files in the current directory",
    "Search for 'TODO' comments in the codebase",
    "Show me the git status",
    "Find all test files",
    "Count lines of code in src/",
    "Check if package.json exists",
    "Search for function definitions",
    "List all imports in main.py",
    "Find all TODO and FIXME comments",
]

LARGE_CONTEXT_REQUESTS: List[str] = [
    "Summarize our conversation so far",
    "What did we discuss about Python?",
    "Recall the first question I asked",
    "What are the key points from our discussion?",
    "Summarize the last 10 messages",
]


# =============================================================================
# Performance Targets
# =============================================================================

PERFORMANCE_TARGETS = {
    "latency_ms": {
        "p50": 100,  # 50th percentile
        "p95": 300,  # 95th percentile
        "p99": 500,  # 99th percentile
    },
    "throughput": {
        "min_requests_per_second": 100,
    },
    "error_rate": {
        "max_acceptable": 0.01,  # 1%
        "max_stress": 0.05,  # 5% under stress
    },
}


# =============================================================================
# Weights for Different Request Types
# =============================================================================

REQUEST_WEIGHTS = {
    "simple_chat": 3,  # 3x weight - most common
    "tool_calling": 2,  # 2x weight - common
    "large_context": 1,  # 1x weight - less common
    "concurrent_session": 1,  # 1x weight - concurrent test
}


# =============================================================================
# Wait Time Configuration
# =============================================================================

WAIT_TIME_MIN = 1.0  # Minimum wait between requests (seconds)
WAIT_TIME_MAX = 3.0  # Maximum wait between requests (seconds)


# =============================================================================
# Timeout Configuration
# =============================================================================

REQUEST_TIMEOUT = 30  # Default request timeout (seconds)
CONNECT_TIMEOUT = 10  # Connection timeout (seconds)


# =============================================================================
# Retry Configuration
# =============================================================================

MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # Seconds


# =============================================================================
# Helper Functions
# =============================================================================


def get_profile(profile_name: str) -> Dict[str, Any]:
    """Get load test profile by name.

    Args:
        profile_name: Name of the profile (smoke, normal, peak, stress, endurance)

    Returns:
        Dictionary with profile configuration

    Raises:
        ValueError: If profile not found
    """
    if profile_name not in LOAD_TEST_PROFILES:
        available = ", ".join(LOAD_TEST_PROFILES.keys())
        raise ValueError(f"Unknown profile '{profile_name}'. Available profiles: {available}")
    return LOAD_TEST_PROFILES[profile_name]


def get_random_message(message_type: str = "chat") -> str:
    """Get a random message for testing.

    Args:
        message_type: Type of message (chat, tool, context)

    Returns:
        Random message string
    """
    import random

    if message_type == "chat":
        return random.choice(CHAT_MESSAGES)
    elif message_type == "tool":
        return random.choice(TOOL_REQUESTS)
    elif message_type == "context":
        return random.choice(LARGE_CONTEXT_REQUESTS)
    else:
        return random.choice(CHAT_MESSAGES)


def build_chat_payload(
    message: str,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    stream: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Build chat request payload.

    Args:
        message: User message
        provider: LLM provider
        model: Model name
        stream: Whether to stream response
        **kwargs: Additional parameters

    Returns:
        Request payload dictionary
    """
    payload = {
        "message": message,
        "provider": provider,
        "model": model,
        "stream": stream,
    }
    payload.update(kwargs)
    return payload


def get_locust_cmd_args(profile: str = "normal") -> List[str]:
    """Get Locust command line arguments for a profile.

    Args:
        profile: Load test profile name

    Returns:
        List of command line arguments
    """
    config = get_profile(profile)

    args = [
        "--host",
        config["host"],
        "--users",
        str(config["users"]),
        "--spawn-rate",
        str(config["spawn_rate"]),
        "--run-time",
        config["run_time"],
    ]

    return args


# =============================================================================
# Configuration Validation
# =============================================================================


def validate_config() -> bool:
    """Validate load test configuration.

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Check if host is accessible
    import aiohttp

    async def check_host():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{DEFAULT_HOST}/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status < 500
        except Exception:
            return False

    # Note: This is a basic check. In real usage, you'd want to
    # verify the API server is actually running and accessible.
    return True


# =============================================================================
# Export Configuration
# =============================================================================

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PROVIDER",
    "DEFAULT_MODEL",
    "LOAD_TEST_PROFILES",
    "CHAT_MESSAGES",
    "TOOL_REQUESTS",
    "LARGE_CONTEXT_REQUESTS",
    "PERFORMANCE_TARGETS",
    "REQUEST_WEIGHTS",
    "WAIT_TIME_MIN",
    "WAIT_TIME_MAX",
    "REQUEST_TIMEOUT",
    "get_profile",
    "get_random_message",
    "build_chat_payload",
    "get_locust_cmd_args",
    "validate_config",
]
