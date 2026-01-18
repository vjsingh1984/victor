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

"""Tests for victor.workflows.compiled_executor module."""

import pytest
from typing import Any, Dict, AsyncIterator

from victor.workflows.compiled_executor import (
    CompiledWorkflowExecutor,
    ExecutionResult,
    WorkflowExecutor,
)


class MockCompiledGraph:
    """Mock compiled graph for testing."""

    def __init__(self, has_invoke=True, has_stream=True):
        self._has_invoke = has_invoke
        self._has_stream = has_stream
        self.invoked = False
        self.streamed = False
        self.initial_state_received = None
        self.thread_id_received = None
        self.checkpoint_received = None

    async def invoke(self, initial_state: Dict[str, Any], thread_id=None, checkpoint=None):
        """Mock invoke method."""
        self.invoked = True
        self.initial_state_received = initial_state
        self.thread_id_received = thread_id
        self.checkpoint_received = checkpoint
        return ExecutionResult(
            final_state=initial_state, metrics={"nodes_executed": 5, "duration_seconds": 1.5}
        )

    async def stream(self, initial_state: Dict[str, Any], thread_id=None):
        """Mock stream method."""
        self.streamed = True
        self.initial_state_received = initial_state
        self.thread_id_received = thread_id
        yield {"type": "node_start", "node": "test_node"}
        yield {"type": "node_end", "node": "test_node"}


class MockOrchestratorPool:
    """Mock orchestrator pool for testing."""

    def __init__(self):
        self.name = "mock_pool"


class TestCompiledWorkflowExecutor:
    """Test CompiledWorkflowExecutor class."""

    def test_init(self):
        """Test initialization."""
        pool = MockOrchestratorPool()
        executor = CompiledWorkflowExecutor(pool)

        assert executor._orchestrator_pool == pool

    def test_init_with_none_pool(self):
        """Test initialization with None pool."""
        executor = CompiledWorkflowExecutor(None)

        assert executor._orchestrator_pool is None

    @pytest.mark.asyncio
    async def test_execute_with_invoke_method(self):
        """Test execute with graph that has invoke method."""
        pool = MockOrchestratorPool()
        executor = CompiledWorkflowExecutor(pool)

        graph = MockCompiledGraph(has_invoke=True)
        initial_state = {"key": "value"}

        result = await executor.execute(graph, initial_state)

        assert graph.invoked
        assert graph.initial_state_received == initial_state
        assert result.final_state == initial_state
        assert result.metrics["nodes_executed"] == 5

    @pytest.mark.asyncio
    async def test_execute_without_invoke_method(self):
        """Test execute with graph that doesn't have invoke method."""
        pool = MockOrchestratorPool()
        executor = CompiledWorkflowExecutor(pool)

        graph = object()  # No invoke method
        initial_state = {"key": "value"}

        result = await executor.execute(graph, initial_state)

        assert result.final_state == initial_state
        assert result.metrics["nodes_executed"] == 0
        assert result.metrics["duration_seconds"] == 0.0

    @pytest.mark.asyncio
    async def test_execute_with_thread_id(self):
        """Test execute with thread_id parameter."""
        pool = MockOrchestratorPool()
        executor = CompiledWorkflowExecutor(pool)

        graph = MockCompiledGraph(has_invoke=True)
        initial_state = {"key": "value"}
        thread_id = "test-thread-123"

        await executor.execute(graph, initial_state, thread_id=thread_id)

        assert graph.thread_id_received == thread_id

    @pytest.mark.asyncio
    async def test_execute_with_checkpoint(self):
        """Test execute with checkpoint parameter."""
        pool = MockOrchestratorPool()
        executor = CompiledWorkflowExecutor(pool)

        graph = MockCompiledGraph(has_invoke=True)
        initial_state = {"key": "value"}
        checkpoint = "checkpoint-1"

        await executor.execute(graph, initial_state, checkpoint=checkpoint)

        assert graph.checkpoint_received == checkpoint

    @pytest.mark.asyncio
    async def test_execute_with_complex_state(self):
        """Test execute with complex initial state."""
        pool = MockOrchestratorPool()
        executor = CompiledWorkflowExecutor(pool)

        graph = MockCompiledGraph(has_invoke=True)
        initial_state = {
            "key1": "value1",
            "key2": ["value2", "value3"],
            "key3": {"nested": "value4"},
        }

        result = await executor.execute(graph, initial_state)

        assert result.final_state == initial_state

    @pytest.mark.asyncio
    async def test_stream_with_stream_method(self):
        """Test stream with graph that has stream method."""
        pool = MockOrchestratorPool()
        executor = CompiledWorkflowExecutor(pool)

        graph = MockCompiledGraph(has_stream=True)
        initial_state = {"key": "value"}

        events = []
        async for event in executor.stream(graph, initial_state):
            events.append(event)

        assert graph.streamed
        assert len(events) == 2
        assert events[0]["type"] == "node_start"
        assert events[1]["type"] == "node_end"

    @pytest.mark.asyncio
    async def test_stream_without_stream_method(self):
        """Test stream with graph that doesn't have stream method."""
        pool = MockOrchestratorPool()
        executor = CompiledWorkflowExecutor(pool)

        graph = object()  # No stream method
        initial_state = {"key": "value"}

        events = []
        async for event in executor.stream(graph, initial_state):
            events.append(event)

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_stream_with_thread_id(self):
        """Test stream with thread_id parameter."""
        pool = MockOrchestratorPool()
        executor = CompiledWorkflowExecutor(pool)

        graph = MockCompiledGraph(has_stream=True)
        initial_state = {"key": "value"}
        thread_id = "test-thread-456"

        async for _ in executor.stream(graph, initial_state, thread_id=thread_id):
            pass

        assert graph.thread_id_received == thread_id


class TestExecutionResult:
    """Test ExecutionResult class."""

    def test_init(self):
        """Test initialization."""
        final_state = {"key": "value"}
        metrics = {"nodes_executed": 5}

        result = ExecutionResult(final_state, metrics)

        assert result.final_state == final_state
        assert result.metrics == metrics

    def test_final_state_property(self):
        """Test final_state property."""
        final_state = {"key": "value"}
        result = ExecutionResult(final_state, {})

        assert result.final_state == final_state

    def test_metrics_property(self):
        """Test metrics property."""
        metrics = {"nodes_executed": 5, "duration": 1.5}
        result = ExecutionResult({}, metrics)

        assert result.metrics == metrics

    def test_with_complex_final_state(self):
        """Test with complex final state."""
        final_state = {
            "key1": "value1",
            "key2": ["value2", "value3"],
            "key3": {"nested": "value4"},
        }
        result = ExecutionResult(final_state, {})

        assert result.final_state == final_state

    def test_with_complex_metrics(self):
        """Test with complex metrics."""
        metrics = {
            "nodes_executed": 5,
            "duration_seconds": 1.5,
            "memory_used_mb": 256,
            "nested": {"key": "value"},
        }
        result = ExecutionResult({}, metrics)

        assert result.metrics == metrics

    def test_property_immutability(self):
        """Test that properties return consistent values."""
        final_state = {"key": "value"}
        metrics = {"nodes_executed": 5}
        result = ExecutionResult(final_state, metrics)

        assert result.final_state is result.final_state
        assert result.metrics is result.metrics


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""

    def test_workflow_executor_alias(self):
        """Test WorkflowExecutor is an alias for CompiledWorkflowExecutor."""
        assert WorkflowExecutor is CompiledWorkflowExecutor

    def test_workflow_executor_can_be_instantiated(self):
        """Test that WorkflowExecutor alias can be instantiated."""
        pool = MockOrchestratorPool()
        executor = WorkflowExecutor(pool)

        assert isinstance(executor, CompiledWorkflowExecutor)
        assert executor._orchestrator_pool == pool
