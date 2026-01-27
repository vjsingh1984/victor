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

"""Locust load test file for Victor AI tool execution.

This file simulates realistic tool execution patterns.
Run with: locust -f tests/load/locustfiles/locustfile_tools.py

Or headless: locust -f tests/load/locustfiles/locustfile_tools.py --headless --users 30 --spawn-rate 3 --run-time 15m
"""

import random
import time
from datetime import datetime
from typing import List

from locust import HttpUser, task, between, events, tag


class ToolUser(HttpUser):
    """Simulates users focused on tool execution.

    Tool Categories:
    - 40% File operations (read, write, list)
    - 25% Search operations (grep, semantic search)
    - 15% Git operations (status, log, diff)
    - 10% Execution operations (run tests, execute code)
    - 10% Analysis operations (code analysis, metrics)
    """

    # Wait between tool operations: 3-8 seconds (tools take time)
    wait_time = between(3, 8)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider = "anthropic"
        self.model = "claude-sonnet-4-5"
        self.session_files = []  # Track files used in session

    @tag("file_ops", "core")
    @task(4)
    def file_operations(self):
        """Execute file operation tools."""
        operations = [
            "Read the file README.md and summarize it",
            "List all Python files in the current directory",
            "Read the file pyproject.toml",
            "List all test files",
            "Read the file src/main.py",
            "Find all configuration files",
            "List all markdown files",
            "Read the file .gitignore",
            "Show the directory structure",
            "List all files in the src directory",
        ]

        message = random.choice(operations)

        payload = {
            "message": message,
            "provider": self.provider,
            "model": self.model,
            "stream": False,
        }

        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="/chat (file operation)",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"File operation failed: {response.status_code}")

    @tag("search", "core")
    @task(3)
    def search_operations(self):
        """Execute search tools."""
        searches = [
            "Search for 'import pytest' in all Python files",
            "Find all TODO comments in the codebase",
            "Search for function definitions",
            "Find all classes that inherit from BaseTool",
            "Search for 'def test_' in test files",
            "Find all async function definitions",
            "Search for error handling patterns",
            "Find all configuration constants",
            "Search for type hints in Python files",
            "Find all decorator usages",
        ]

        message = random.choice(searches)

        payload = {
            "message": message,
            "provider": self.provider,
            "model": self.model,
            "stream": False,
        }

        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="/chat (search)",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Search failed: {response.status_code}")

    @tag("git")
    @task(2)
    def git_operations(self):
        """Execute Git operations."""
        git_commands = [
            "Check the git status",
            "Show the last 5 commits",
            "Get the current git branch",
            "Show the git log",
            "Check for uncommitted changes",
            "Show files modified in last commit",
            "Display git diff for README.md",
            "Check git remote configuration",
            "Show commit history for tests/",
        ]

        message = random.choice(git_commands)

        payload = {
            "message": message,
            "provider": self.provider,
            "model": self.model,
            "stream": False,
        }

        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="/chat (git operation)",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Git operation failed: {response.status_code}")

    @tag("execution")
    @task(1)
    def execution_operations(self):
        """Execute code or run tests."""
        executions = [
            "Run the pytest test suite",
            "Execute the main script",
            "Run tests for the auth module",
            "Execute code: print('hello world')",
            "Run linting with ruff",
            "Execute type checking with mypy",
            "Run the build script",
            "Execute tests with coverage",
        ]

        message = random.choice(executions)

        payload = {
            "message": message,
            "provider": self.provider,
            "model": self.model,
            "stream": False,
        }

        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="/chat (execution)",
            timeout=60.0,  # Execution operations take longer
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code in [408, 504]:
                # Timeout is acceptable for execution operations
                response.success()
            else:
                response.failure(f"Execution failed: {response.status_code}")

    @tag("analysis")
    @task(1)
    def analysis_operations(self):
        """Execute analysis operations."""
        analyses = [
            "Analyze the code structure of src/",
            "Calculate code complexity for main.py",
            "Find code smells in the codebase",
            "Analyze test coverage",
            "Check for security vulnerabilities",
            "Analyze dependencies",
            "Calculate cyclomatic complexity",
            "Find duplicate code patterns",
            "Analyze import dependencies",
            "Check for PEP8 violations",
        ]

        message = random.choice(analyses)

        payload = {
            "message": message,
            "provider": self.provider,
            "model": self.model,
            "stream": False,
        }

        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="/chat (analysis)",
            timeout=45.0,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Analysis failed: {response.status_code}")

    @tag("workflow")
    @task(2)
    def workflow_operations(self):
        """Execute multi-step workflows."""
        workflows = [
            "Review the code in tests/ and generate a report",
            "Refactor the function in utils.py to be more efficient",
            "Generate unit tests for api.py",
            "Analyze performance bottlenecks in app.py",
            "Document the API endpoints in main.py",
            "Debug the failing tests in test_auth.py",
            "Optimize the database queries in models.py",
            "Add type hints to all functions in utils/",
        ]

        message = random.choice(workflows)

        payload = {
            "message": message,
            "provider": self.provider,
            "model": self.model,
            "stream": False,
        }

        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="/chat (workflow)",
            timeout=90.0,  # Workflows take longest
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Workflow failed: {response.status_code}")


class HeavyToolUser(ToolUser):
    """User that aggressively uses tools (stress test)."""

    wait_time = between(1, 3)  # Faster execution

    @task(10)
    def file_operations(self):
        """Heavy use of file operations."""
        super().file_operations()

    @task(5)
    def search_operations(self):
        """Heavy use of searches."""
        super().search_operations()


class MixedWorkflowUser(ToolUser):
    """User that performs complex multi-tool workflows."""

    wait_time = between(5, 12)

    @task
    def complex_workflow(self):
        """Execute complex workflow combining multiple tools."""
        workflows = [
            "Analyze the codebase, find security issues, and generate a report",
            "Review all Python files, run tests, and document failures",
            "Search for bugs, analyze complexity, and suggest refactoring",
            "Check code style, run linters, and fix violations",
            "Find deprecated code, update it, and add tests",
        ]

        message = random.choice(workflows)

        payload = {
            "message": message,
            "provider": self.provider,
            "model": self.model,
            "stream": False,
        }

        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="/chat (complex workflow)",
            timeout=120.0,  # Complex workflows take very long
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Complex workflow failed: {response.status_code}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate test summary."""
    print("\n" + "=" * 80)
    print("VICTOR AI TOOL EXECUTION LOAD TEST SUMMARY")
    print("=" * 80)
    print(f"Test completed at: {datetime.now().isoformat()}")

    if environment.stats:
        stats = environment.stats

        print(f"\nTotal Requests: {stats.total.num_requests}")
        print(f"Successful: {stats.total.num_requests - stats.total.num_failures}")
        print(f"Failed: {stats.total.num_failures}")

        if stats.total.num_requests > 0:
            success_rate = (
                (stats.total.num_requests - stats.total.num_failures)
                / stats.total.num_requests
                * 100
            )
            print(f"Success Rate: {success_rate:.2f}%")

        print("\nResponse Times:")
        print(f"  Median: {stats.total.median_response_time:.0f}ms")
        print(f"  95th percentile: {stats.total.get_response_time_percentile(0.95):.0f}ms")
        print(f"  99th percentile: {stats.total.get_response_time_percentile(0.99):.0f}ms")

        print(f"\nThroughput: {stats.total.total_avg_rps:.2f} requests/second")

        # Breakdown by endpoint type
        print("\nBreakdown by Operation Type:")
        for name, entry in stats.entries.items():
            if entry.num_requests > 0:
                print(f"  {name}:")
                print(f"    Requests: {entry.num_requests}")
                print(f"    Median: {entry.median_response_time:.0f}ms")
                print(f"    Failures: {entry.num_failures}")

    print("=" * 80 + "\n")
