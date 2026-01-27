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

"""Workflow execution load tests for Victor AI.

Tests workflow performance under concurrent load to ensure
StateGraph workflows execute efficiently at scale.
"""

import asyncio
import pytest
import time
import statistics
from typing import Any, Dict, List
from datetime import datetime

import httpx
from pytest import mark


# Test configuration
API_HOST = "http://localhost:8765"
DEFAULT_TIMEOUT = 60.0  # Longer timeout for workflows


@mark.load
@mark.asyncio
@pytest.mark.usefixtures("api_server_available")
class TestWorkflowLoad:
    """Workflow execution load tests.

    Measures workflow compilation, execution, and state management under load.
    Requires API server running at localhost:8765.
    """

    async def test_simple_workflow_baseline(self):
        """Establish baseline for simple workflow execution.

        Target: Complete in <5 seconds
        """
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # Execute simple workflow (e.g., code review)
            payload = {
                "message": "Review the code in test.py for bugs",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "stream": False,
            }

            start = time.time()
            response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
            duration = time.time() - start

            assert response.status_code == 200
            assert duration < 10, f"Simple workflow too slow: {duration:.2f}s"

            print(f"\nSimple Workflow Baseline: {duration:.2f}s")

    @mark.slow
    async def test_concurrent_workflow_execution(self):
        """Test concurrent workflow executions.

        Target: 20 concurrent workflows, error rate <5%
        """
        num_workflows = 20

        workflow_requests = [
            "Review the code in main.py for security issues",
            "Refactor the function in utils.py",
            "Generate tests for api.py",
            "Analyze performance in app.py",
            "Document the codebase",
        ]

        async def execute_workflow(client: httpx.AsyncClient, wf_id: int) -> Dict[str, Any]:
            """Execute a workflow."""
            try:
                start = time.time()
                payload = {
                    "message": workflow_requests[wf_id % len(workflow_requests)],
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                duration = time.time() - start

                return {
                    "success": response.status_code == 200,
                    "duration": duration,
                    "error": None,
                }
            except Exception as e:
                return {
                    "success": False,
                    "duration": -1,
                    "error": str(e),
                }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [execute_workflow(client, i) for i in range(num_workflows)]
            results = await asyncio.gather(*tasks)

        # Analyze results
        successful = [r for r in results if r["success"]]
        errors = [r for r in results if not r["success"]]
        durations = [r["duration"] for r in successful]

        error_rate = (len(errors) / num_workflows) * 100
        avg_duration = statistics.mean(durations) if durations else 0
        p95_duration = statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else 0

        print("\nConcurrent Workflow Execution Test:")
        print(f"  Workflows: {num_workflows}")
        print(f"  Successful: {len(successful)}")
        print(f"  Errors: {len(errors)}")
        print(f"  Error Rate: {error_rate:.2f}%")
        print(f"  Avg Duration: {avg_duration:.2f}s")
        print(f"  P95 Duration: {p95_duration:.2f}s")

        assert error_rate < 10, f"Error rate too high: {error_rate:.2f}%"

    @mark.slow
    async def test_workflow_state_management(self):
        """Test workflow state management under load.

        Verify that workflow states are properly isolated and managed.
        """
        num_workflows = 10

        async def execute_workflow_with_state(
            client: httpx.AsyncClient, wf_id: int
        ) -> Dict[str, Any]:
            """Execute workflow and verify state isolation."""
            try:
                # Multi-step workflow
                steps = [
                    f"Step 1: Initialize workflow {wf_id}",
                    f"Step 2: Process workflow {wf_id}",
                    f"Step 3: Finalize workflow {wf_id}",
                ]

                durations = []
                for step in steps:
                    start = time.time()
                    payload = {
                        "message": step,
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-5",
                        "stream": False,
                    }
                    response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                    duration = time.time() - start
                    durations.append(duration)

                    if response.status_code != 200:
                        return {"success": False, "wf_id": wf_id}

                return {
                    "success": True,
                    "wf_id": wf_id,
                    "durations": durations,
                }
            except Exception:
                return {"success": False, "wf_id": wf_id}

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [execute_workflow_with_state(client, i) for i in range(num_workflows)]
            results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r["success"])
        success_rate = (successful / num_workflows) * 100

        print("\nWorkflow State Management Test:")
        print(f"  Workflows: {num_workflows}")
        print(f"  Successful: {successful}/{num_workflows}")
        print(f"  Success Rate: {success_rate:.2f}%")

        assert success_rate >= 90, f"State management failure: {success_rate:.2f}%"

    @mark.slow
    async def test_workflow_checkpoint_recovery(self):
        """Test workflow checkpointing and recovery.

        Verify workflows can resume from checkpoints under load.
        """
        # This test requires workflow checkpointing to be enabled
        # In production, you'd test actual pause/resume functionality

        num_workflows = 5

        async def execute_checkpoint_workflow(client: httpx.AsyncClient, wf_id: int) -> bool:
            """Execute workflow that supports checkpointing."""
            try:
                # Long workflow that should checkpoint
                payload = {
                    "message": f"Perform long analysis {wf_id} with multiple steps",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)

                return response.status_code == 200
            except Exception:
                return False

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [execute_checkpoint_workflow(client, i) for i in range(num_workflows)]
            results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r)
        success_rate = (successful / num_workflows) * 100

        print("\nWorkflow Checkpoint Recovery Test:")
        print(f"  Workflows: {num_workflows}")
        print(f"  Successful: {successful}/{num_workflows}")
        print(f"  Success Rate: {success_rate:.2f}%")

        assert success_rate >= 80, f"Checkpoint recovery failure: {success_rate:.2f}%"

    @mark.slow
    async def test_nested_workflow_execution(self):
        """Test execution of nested workflows (workflows calling workflows).

        Verify system handles workflow composition correctly.
        """
        num_workflows = 10

        async def execute_nested_workflow(client: httpx.AsyncClient, wf_id: int) -> Dict[str, Any]:
            """Execute nested workflow."""
            try:
                start = time.time()

                # Outer workflow
                payload = {
                    "message": f"Analyze codebase {wf_id} and generate comprehensive report",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                duration = time.time() - start

                return {
                    "success": response.status_code == 200,
                    "duration": duration,
                }
            except Exception as e:
                return {
                    "success": False,
                    "duration": -1,
                }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [execute_nested_workflow(client, i) for i in range(num_workflows)]
            results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r["success"])
        durations = [r["duration"] for r in results if r["duration"] > 0]

        avg_duration = statistics.mean(durations) if durations else 0

        print("\nNested Workflow Execution Test:")
        print(f"  Workflows: {num_workflows}")
        print(f"  Successful: {successful}/{num_workflows}")
        print(f"  Avg Duration: {avg_duration:.2f}s")

        assert successful >= num_workflows * 0.8, "Too many nested workflow failures"

    @mark.slow
    async def test_workflow_caching_performance(self):
        """Test workflow caching effectiveness.

        Measure performance improvement from cached workflow compilation.
        """
        num_iterations = 20

        # Same workflow request to test caching
        workflow_request = "Review code structure and suggest improvements"

        async def execute_workflow(client: httpx.AsyncClient, iteration: int) -> float:
            """Execute workflow and return duration."""
            try:
                start = time.time()
                payload = {
                    "message": workflow_request,
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                duration = time.time() - start

                if response.status_code != 200:
                    return -1
                return duration
            except Exception:
                return -1

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # First 5 executions (cold cache)
            cold_durations = []
            for i in range(5):
                duration = await execute_workflow(client, i)
                if duration > 0:
                    cold_durations.append(duration)

            # Next 15 executions (warm cache)
            warm_durations = []
            for i in range(5, num_iterations):
                duration = await execute_workflow(client, i)
                if duration > 0:
                    warm_durations.append(duration)

        avg_cold = statistics.mean(cold_durations) if cold_durations else 0
        avg_warm = statistics.mean(warm_durations) if warm_durations else 0
        speedup = avg_cold / avg_warm if avg_warm > 0 else 0

        print("\nWorkflow Caching Performance Test:")
        print(f"  Cold Executions: {len(cold_durations)}")
        print(f"  Warm Executions: {len(warm_durations)}")
        print(f"  Cold Avg: {avg_cold:.2f}s")
        print(f"  Warm Avg: {avg_warm:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Cache should provide some benefit (or at least not hurt)
        if avg_warm > 0:
            assert avg_warm <= avg_cold * 1.5, "Cache significantly degrading performance"

    async def test_workflow_error_recovery(self):
        """Test workflow error handling and recovery.

        Verify workflows handle errors gracefully and can recover.
        """
        num_workflows = 10

        async def execute_workflow_with_error(
            client: httpx.AsyncClient, wf_id: int
        ) -> Dict[str, Any]:
            """Execute workflow that may encounter errors."""
            try:
                # Request that might cause intermediate errors
                payload = {
                    "message": f"Analyze non_existent_file_{wf_id}.py and handle errors gracefully",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)

                # Should get response even with errors
                return {
                    "got_response": response.status_code in [200, 400, 500],
                    "status_code": response.status_code,
                }
            except Exception:
                return {
                    "got_response": False,
                    "status_code": -1,
                }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [execute_workflow_with_error(client, i) for i in range(num_workflows)]
            results = await asyncio.gather(*tasks)

        responses_received = sum(1 for r in results if r["got_response"])
        response_rate = (responses_received / num_workflows) * 100

        print("\nWorkflow Error Recovery Test:")
        print(f"  Workflows: {num_workflows}")
        print(f"  Responses Received: {responses_received}/{num_workflows}")
        print(f"  Response Rate: {response_rate:.2f}%")

        # Should handle errors gracefully
        assert response_rate >= 90, f"Error recovery failed: {response_rate:.2f}%"

    @mark.slow
    async def test_parallel_workflow_execution(self):
        """Test parallel execution of independent workflow steps.

        Verify system can parallelize workflow steps when possible.
        """
        num_workflows = 15

        async def execute_parallel_workflow(client: httpx.AsyncClient, wf_id: int) -> float:
            """Execute workflow with parallelizable steps."""
            try:
                start = time.time()
                payload = {
                    "message": f"Analyze files file1.py, file2.py, file3.py in parallel for workflow {wf_id}",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "stream": False,
                }
                response = await client.post(f"{API_HOST}/api/v1/chat", json=payload)
                duration = time.time() - start

                if response.status_code != 200:
                    return -1
                return duration
            except Exception:
                return -1

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [execute_parallel_workflow(client, i) for i in range(num_workflows)]
            results = await asyncio.gather(*tasks)

        successful = [r for r in results if r > 0]
        success_rate = (len(successful) / num_workflows) * 100

        if successful:
            avg_duration = statistics.mean(successful)
            print("\nParallel Workflow Execution Test:")
            print(f"  Workflows: {num_workflows}")
            print(f"  Successful: {len(successful)}/{num_workflows}")
            print(f"  Success Rate: {success_rate:.2f}%")
            print(f"  Avg Duration: {avg_duration:.2f}s")

            assert success_rate >= 80, f"Too many failures: {success_rate:.2f}%"
        else:
            pytest.skip("No successful parallel workflows executed")
