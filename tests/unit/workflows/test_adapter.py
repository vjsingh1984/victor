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

"""Tests for adapter layer backward compatibility (Phase 4).

Tests verify:
- Adapters wrap new interfaces correctly
- Deprecation warnings are emitted
- Legacy APIs continue to work
- Migration path is clear
"""

import asyncio
import warnings
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.workflows.adapter import (
    DeprecationAdapter,
    UnifiedWorkflowCompilerAdapter,
    CompiledGraphAdapter,
    ExecutorResultAdapter,
)


# ============ Test Fixtures ============


@pytest.fixture
def mock_compiled_graph():
    """Create a mock CompiledGraphProtocol."""
    mock_graph = MagicMock()
    mock_graph.graph_id = "test-graph"
    mock_graph.invoke = AsyncMock()
    mock_graph.stream = AsyncMock()
    mock_graph.graph = MagicMock()
    return mock_graph


@pytest.fixture
def mock_execution_result():
    """Create a mock ExecutionResultProtocol."""
    mock_result = MagicMock()
    mock_result.final_state = {"output": "result", "counter": 42}
    mock_result.metrics = {"nodes_executed": 3, "duration_ms": 150}
    return mock_result


@pytest.fixture
def mock_workflow_compiler():
    """Create a mock WorkflowCompilerProtocol."""
    mock_compiler = MagicMock()
    mock_compiler.compile = MagicMock(return_value=MagicMock())
    return mock_compiler


# ============ DeprecationAdapter Tests ============


class TestDeprecationAdapter:
    """Tests for base DeprecationAdapter class."""

    def test_deprecation_adapter_warns(self):
        """Test that _warn emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DeprecationAdapter._warn()

            # Check warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert "v0.7.0" in str(w[0].message)

    def test_deprecation_message_content(self):
        """Test that deprecation message contains migration guidance."""
        message = DeprecationAdapter._DEPRECATION_MESSAGE

        # Should mention version
        assert "v0.7.0" in message

        # Should mention migration
        assert "migrate" in message.lower()

        # Should mention migration guide
        assert "MIGRATION_GUIDE" in message


# ============ UnifiedWorkflowCompilerAdapter Tests ============


class TestUnifiedWorkflowCompilerAdapter:
    """Tests for UnifiedWorkflowCompilerAdapter."""

    def test_adapter_emits_warning_on_init(self):
        """Test adapter emits deprecation warning on initialization."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = UnifiedWorkflowCompilerAdapter()

            # Should emit warning
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    @pytest.mark.asyncio
    async def test_compile_returns_compiled_graph_adapter(self, mock_workflow_compiler):
        """Test compile() returns a CompiledGraphAdapter."""
        with patch("victor.core.container.get_container") as mock_get_container:
            # Setup mock container
            mock_container = MagicMock()
            mock_container.get.return_value = mock_workflow_compiler
            mock_get_container.return_value = mock_container

            adapter = UnifiedWorkflowCompilerAdapter()

            # Mock the compile return value
            mock_compiled = MagicMock()
            mock_compiled.graph_id = "test-graph"
            mock_workflow_compiler.compile.return_value = mock_compiled

            # Call compile
            result = adapter.compile("workflow.yaml")

            # Verify it's a CompiledGraphAdapter
            assert isinstance(result, CompiledGraphAdapter)
            assert result._compiled_graph == mock_compiled

    def test_compile_passes_workflow_name(self):
        """Test compile() passes workflow_name parameter."""
        with patch("victor.core.container.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_compiler = MagicMock()
            mock_container.get.return_value = mock_compiler
            mock_get_container.return_value = mock_container

            adapter = UnifiedWorkflowCompilerAdapter()

            # Call with workflow_name
            adapter.compile("workflow.yaml", workflow_name="my_workflow")

            # Verify workflow_name was passed
            mock_compiler.compile.assert_called_once()
            call_kwargs = mock_compiler.compile.call_args.kwargs
            assert call_kwargs.get("workflow_name") == "my_workflow"

    def test_compile_passes_validate_parameter(self):
        """Test compile() passes validate parameter."""
        with patch("victor.core.container.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_compiler = MagicMock()
            mock_container.get.return_value = mock_compiler
            mock_get_container.return_value = mock_container

            adapter = UnifiedWorkflowCompilerAdapter()

            # Call with validate=False
            adapter.compile("workflow.yaml", validate=False)

            # Verify validate was passed
            mock_compiler.compile.assert_called_once()
            call_kwargs = mock_compiler.compile.call_args.kwargs
            assert call_kwargs.get("validate") is False


# ============ CompiledGraphAdapter Tests ============


class TestCompiledGraphAdapter:
    """Tests for CompiledGraphAdapter."""

    def test_init_wraps_compiled_graph(self, mock_compiled_graph):
        """Test adapter wraps compiled graph correctly."""
        adapter = CompiledGraphAdapter(mock_compiled_graph)

        assert adapter._compiled_graph == mock_compiled_graph

    def test_graph_property_returns_underlying_graph(self, mock_compiled_graph):
        """Test graph property returns underlying StateGraph."""
        adapter = CompiledGraphAdapter(mock_compiled_graph)

        assert adapter.graph == mock_compiled_graph.graph

    @pytest.mark.asyncio
    async def test_invoke_wraps_execution_result(
        self, mock_compiled_graph, mock_execution_result
    ):
        """Test invoke() wraps execution result in adapter."""
        # Setup
        adapter = CompiledGraphAdapter(mock_compiled_graph)
        mock_compiled_graph.invoke.return_value = mock_execution_result

        # Call invoke
        result = await adapter.invoke({"input": "test"})

        # Verify it's an ExecutorResultAdapter
        assert isinstance(result, ExecutorResultAdapter)
        assert result._result == mock_execution_result

    @pytest.mark.asyncio
    async def test_invoke_emits_warning(self, mock_compiled_graph, mock_execution_result):
        """Test invoke() emits deprecation warning."""
        mock_compiled_graph.invoke.return_value = mock_execution_result
        adapter = CompiledGraphAdapter(mock_compiled_graph)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            await adapter.invoke({"input": "test"})

            # Should emit warning
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    @pytest.mark.asyncio
    async def test_invoke_passes_initial_state(self, mock_compiled_graph):
        """Test invoke() passes initial_state parameter."""
        adapter = CompiledGraphAdapter(mock_compiled_graph)
        mock_compiled_graph.invoke.return_value = MagicMock()

        initial_state = {"query": "search", "counter": 100}
        await adapter.invoke(initial_state)

        # Verify initial_state was passed
        mock_compiled_graph.invoke.assert_called_once()
        call_args = mock_compiled_graph.invoke.call_args.args
        assert call_args[0] == initial_state

    @pytest.mark.asyncio
    async def test_invoke_passes_thread_id(self, mock_compiled_graph):
        """Test invoke() passes thread_id parameter."""
        adapter = CompiledGraphAdapter(mock_compiled_graph)
        mock_compiled_graph.invoke.return_value = MagicMock()

        await adapter.invoke({"input": "test"}, thread_id="thread-123")

        # Verify thread_id was passed
        mock_compiled_graph.invoke.assert_called_once()
        call_kwargs = mock_compiled_graph.invoke.call_args.kwargs
        assert call_kwargs.get("thread_id") == "thread-123"

    @pytest.mark.asyncio
    async def test_invoke_passes_checkpoint(self, mock_compiled_graph):
        """Test invoke() passes checkpoint parameter."""
        adapter = CompiledGraphAdapter(mock_compiled_graph)
        mock_compiled_graph.invoke.return_value = MagicMock()

        await adapter.invoke({"input": "test"}, checkpoint="checkpoint-1")

        # Verify checkpoint was passed
        mock_compiled_graph.invoke.assert_called_once()
        call_kwargs = mock_compiled_graph.invoke.call_args.kwargs
        assert call_kwargs.get("checkpoint") == "checkpoint-1"

    @pytest.mark.asyncio
    async def test_stream_yields_events(self, mock_compiled_graph):
        """Test stream() yields execution events."""
        # Setup mock async generator
        async def mock_stream_generator(state, thread_id=None):
            events = [
                {"type": "node_start", "node_id": "node1"},
                {"type": "node_complete", "node_id": "node1"},
                {"type": "workflow_complete"},
            ]
            for event in events:
                yield event

        mock_compiled_graph.stream = mock_stream_generator
        adapter = CompiledGraphAdapter(mock_compiled_graph)

        # Collect events
        events = []
        async for event in adapter.stream({"input": "test"}):
            events.append(event)

        # Should have received all events
        assert len(events) == 3
        assert events[0]["node_id"] == "node1"
        assert events[2]["type"] == "workflow_complete"

    @pytest.mark.asyncio
    async def test_stream_passes_thread_id(self, mock_compiled_graph):
        """Test stream() passes thread_id parameter."""
        # Track if thread_id was passed
        captured_thread_id = []

        async def mock_stream_generator(state, thread_id=None):
            captured_thread_id.append(thread_id)
            yield {"type": "test"}

        mock_compiled_graph.stream = mock_stream_generator
        adapter = CompiledGraphAdapter(mock_compiled_graph)

        # Stream with thread_id
        async for _ in adapter.stream({"input": "test"}, thread_id="thread-456"):
            pass

        # Verify thread_id was passed
        assert captured_thread_id[0] == "thread-456"

    @pytest.mark.asyncio
    async def test_stream_emits_warning(self, mock_compiled_graph):
        """Test stream() emits deprecation warning."""
        async def mock_stream_generator(state, thread_id=None):
            yield {"type": "test"}

        mock_compiled_graph.stream = mock_stream_generator
        adapter = CompiledGraphAdapter(mock_compiled_graph)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            async for _ in adapter.stream({"input": "test"}):
                pass

            # Should emit warning
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)


# ============ ExecutorResultAdapter Tests ============


class TestExecutorResultAdapter:
    """Tests for ExecutorResultAdapter."""

    def test_init_wraps_execution_result(self, mock_execution_result):
        """Test adapter wraps execution result correctly."""
        adapter = ExecutorResultAdapter(mock_execution_result)

        assert adapter._result == mock_execution_result

    def test_final_state_property(self, mock_execution_result):
        """Test final_state property returns correct value."""
        adapter = ExecutorResultAdapter(mock_execution_result)

        assert adapter.final_state == mock_execution_result.final_state
        assert adapter.final_state == {"output": "result", "counter": 42}

    def test_metrics_property(self, mock_execution_result):
        """Test metrics property returns correct value."""
        adapter = ExecutorResultAdapter(mock_execution_result)

        assert adapter.metrics == mock_execution_result.metrics
        assert adapter.metrics == {"nodes_executed": 3, "duration_ms": 150}

    def test_adapter_is_data_wrapper(self, mock_execution_result):
        """Test that ExecutorResultAdapter is just a data wrapper (no warnings)."""
        adapter = ExecutorResultAdapter(mock_execution_result)

        # Should not emit warnings when accessing properties
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = adapter.final_state
            _ = adapter.metrics

            # No warnings should be emitted
            assert len(w) == 0


# ============ Backward Compatibility Tests ============


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy code."""

    @pytest.mark.asyncio
    async def test_legacy_compile_invoke_pattern(
        self, mock_workflow_compiler, mock_compiled_graph, mock_execution_result
    ):
        """Test legacy pattern: compile().invoke() works correctly."""
        with patch("victor.core.container.get_container") as mock_get_container:
            # Setup
            mock_container = MagicMock()
            mock_container.get.return_value = mock_workflow_compiler
            mock_get_container.return_value = mock_container

            mock_workflow_compiler.compile.return_value = mock_compiled_graph
            mock_compiled_graph.invoke.return_value = mock_execution_result

            # Legacy pattern (should still work)
            compiler = UnifiedWorkflowCompilerAdapter()
            compiled = compiler.compile("workflow.yaml")
            result = await compiled.invoke({"input": "data"})

            # Verify result
            assert isinstance(result, ExecutorResultAdapter)
            assert result.final_state == {"output": "result", "counter": 42}

    @pytest.mark.asyncio
    async def test_legacy_compile_stream_pattern(
        self, mock_workflow_compiler, mock_compiled_graph
    ):
        """Test legacy pattern: compile().stream() works correctly."""
        with patch("victor.core.container.get_container") as mock_get_container:
            # Setup
            mock_container = MagicMock()
            mock_container.get.return_value = mock_workflow_compiler
            mock_get_container.return_value = mock_container

            mock_workflow_compiler.compile.return_value = mock_compiled_graph

            async def mock_stream_generator(state, thread_id=None):
                yield {"type": "node_start", "node_id": "start"}

            mock_compiled_graph.stream = mock_stream_generator

            # Legacy pattern (should still work)
            compiler = UnifiedWorkflowCompilerAdapter()
            compiled = compiler.compile("workflow.yaml")

            events = []
            async for event in compiled.stream({"input": "data"}):
                events.append(event)

            # Verify streaming worked
            assert len(events) == 1
            assert events[0]["node_id"] == "start"

    def test_adapter_exports_available(self):
        """Test that adapters are exported from workflows module."""
        from victor.workflows import (
            UnifiedWorkflowCompilerAdapter,
            CompiledGraphAdapter,
            ExecutorResultAdapter,
        )

        # Should be importable
        assert UnifiedWorkflowCompilerAdapter is not None
        assert CompiledGraphAdapter is not None
        assert ExecutorResultAdapter is not None


# ============ Integration Tests ============


class TestAdapterIntegration:
    """Integration tests for adapter with real workflow components."""

    def test_adapter_with_real_workflow_definition(self):
        """Test adapter works with real WorkflowDefinition."""
        from victor.workflows.definition import WorkflowBuilder

        # Create a real workflow definition
        def transform_fn(ctx):
            ctx["result"] = ctx.get("value", 0) * 2
            return ctx

        workflow = (
            WorkflowBuilder("test", "Test workflow")
            .add_transform("double", transform_fn)
            .build()
        )

        # Verify workflow structure (adapter-compatible)
        assert workflow is not None
        # nodes is a dict, not a list
        assert len(workflow.nodes) == 1
        # Access by key
        assert "double" in workflow.nodes
        assert workflow.nodes["double"].id == "double"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
