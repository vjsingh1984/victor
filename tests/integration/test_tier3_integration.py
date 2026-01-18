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

"""Integration tests for Tier 3 accelerators.

Tests the integration of:
- GraphAlgorithmsAccelerator (3-6x faster graph metrics)
- BatchProcessorAccelerator (20-40% faster parallel execution)
- SerializationAccelerator (5-10x faster JSON/YAML parsing)
"""

import asyncio
import json
import pytest
from pathlib import Path
from typing import Dict, List

from victor.native.accelerators import (
    get_graph_algorithms_accelerator,
    get_batch_processor_accelerator,
    get_serialization_accelerator,
)


class TestGraphAlgorithmsIntegration:
    """Test GraphAlgorithmsAccelerator integration."""

    def test_graph_accelerator_singleton(self):
        """Test that accelerator singleton works correctly."""
        accelerator = get_graph_algorithms_accelerator()
        assert accelerator is not None
        assert hasattr(accelerator, "rust_available")

    def test_create_graph(self):
        """Test graph creation from edge list."""
        accelerator = get_graph_algorithms_accelerator()

        # Create simple directed graph
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        graph = accelerator.create_graph(edges, node_count=3, directed=True)

        assert graph is not None
        # Graph should have our structure
        if accelerator.rust_available:
            assert hasattr(graph, "directed")
        else:
            # NetworkX fallback
            import networkx as nx

            assert isinstance(graph, (nx.DiGraph, nx.Graph))

    def test_pagerank_computation(self):
        """Test PageRank score computation."""
        accelerator = get_graph_algorithms_accelerator()

        # Create graph with clear importance hierarchy
        # Node 0 is most important (called by both 1 and 2)
        edges = [(1, 0, 1.0), (2, 0, 1.0), (0, 3, 1.0)]
        graph = accelerator.create_graph(edges, node_count=4, directed=True)

        scores = accelerator.pagerank(graph, damping_factor=0.85, iterations=50)

        assert len(scores) == 4
        # Node 0 should have highest PageRank (most incoming edges)
        assert scores[0] > scores[1] + 1e-10
        assert scores[0] > scores[2] + 1e-10
        # All scores should be positive and sum to ~1
        assert all(s > 1e-10 for s in scores)
        assert abs(sum(scores) - 1.0) < 0.01

    def test_betweenness_centrality(self):
        """Test betweenness centrality computation."""
        accelerator = get_graph_algorithms_accelerator()

        # Create line graph: 0 -> 1 -> 2 -> 3
        # Node 1 and 2 should have highest betweenness (on all paths)
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        graph = accelerator.create_graph(edges, node_count=4, directed=True)

        scores = accelerator.betweenness_centrality(graph, normalized=True)

        assert len(scores) == 4
        # Nodes 1 and 2 should have higher betweenness than endpoints
        assert scores[1] > scores[0] + 1e-10
        assert scores[2] > scores[3] + 1e-10

    def test_connected_components(self):
        """Test connected components detection."""
        accelerator = get_graph_algorithms_accelerator()

        # Create graph with two disconnected components
        # Component 1: 0 -> 1 -> 2
        # Component 2: 3 -> 4
        edges = [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)]
        graph = accelerator.create_graph(edges, node_count=5, directed=True)

        components = accelerator.connected_components(graph)

        assert len(components) == 2
        # Check that we have the right components
        all_nodes = set()
        for comp in components:
            all_nodes.update(comp)
        assert all_nodes == {0, 1, 2, 3, 4}

    def test_shortest_path(self):
        """Test shortest path computation."""
        accelerator = get_graph_algorithms_accelerator()

        # Create graph: 0 -> 1 -> 2 -> 3, plus shortcut 0 -> 2
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 2, 0.5)]
        graph = accelerator.create_graph(edges, node_count=4, directed=True)

        # Find path from 0 to 3
        # Should take shortcut: 0 -> 2 -> 3 (not 0 -> 1 -> 2 -> 3)
        result = accelerator.shortest_path(graph, source=0, target=3)

        assert result is not None
        if accelerator.rust_available:
            # Rust returns PathResult object
            assert result.path[0] == 0
            assert result.path[-1] == 3
            assert 2 in result.path  # Shortcut taken
        else:
            # Python returns list
            assert result[0] == 0
            assert result[-1] == 3


class TestBatchProcessorIntegration:
    """Test BatchProcessorAccelerator integration."""

    def test_batch_accelerator_singleton(self):
        """Test that accelerator singleton works correctly."""
        accelerator = get_batch_processor_accelerator()
        assert accelerator is not None
        assert hasattr(accelerator, "rust_available")

    def test_create_processor(self):
        """Test processor creation."""
        accelerator = get_batch_processor_accelerator()

        processor = accelerator.create_processor(
            max_concurrent=5,
            timeout_ms=10000,
            retry_policy="exponential",
        )

        assert processor is not None

    def test_process_simple_batch(self):
        """Test processing a simple batch of tasks."""
        accelerator = get_batch_processor_accelerator()

        # Create tasks
        from victor.native.accelerators.batch_processor import BatchTask

        tasks = [BatchTask(task_id=f"task-{i}", task_data=i) for i in range(5)]

        # Simple executor that returns task_data * 2
        def executor(task):
            return task.task_data * 2

        # Create processor
        processor = accelerator.create_processor(
            max_concurrent=2,
            timeout_ms=5000,
            retry_policy="none",
        )

        # Run batch (async)
        async def run_batch():
            return await accelerator.process_batch(tasks, executor, processor)

        result = asyncio.run(run_batch())

        assert result.total_tasks == 5
        assert result.successful_count == 5
        assert result.failed_count == 0

        # Check results
        expected_results = {i * 2 for i in range(5)}
        actual_results = {r.result for r in result.results if r.success}
        assert actual_results == expected_results

    def test_process_batch_with_dependencies(self):
        """Test processing batch with task dependencies."""
        accelerator = get_batch_processor_accelerator()

        from victor.native.accelerators.batch_processor import BatchTask

        # Create tasks with dependencies
        # task-0 must complete before task-1 and task-2
        tasks = [
            BatchTask(
                task_id="task-0",
                task_data="base",
                dependencies=[],
                priority=10,
            ),
            BatchTask(
                task_id="task-1",
                task_data="dependent-1",
                dependencies=["task-0"],
                priority=5,
            ),
            BatchTask(
                task_id="task-2",
                task_data="dependent-2",
                dependencies=["task-0"],
                priority=5,
            ),
        ]

        # Executor records execution order
        execution_order = []

        def executor(task):
            execution_order.append(task.task_id)
            return f"processed-{task.task_data}"

        processor = accelerator.create_processor(
            max_concurrent=3,
            timeout_ms=5000,
            retry_policy="none",
        )

        async def run_batch():
            return await accelerator.process_batch(tasks, executor, processor)

        result = asyncio.run(run_batch())

        assert result.total_tasks == 3
        assert result.successful_count == 3

        # task-0 should execute first
        assert execution_order[0] == "task-0"

    def test_batch_with_failures(self):
        """Test batch processing with task failures."""
        accelerator = get_batch_processor_accelerator()

        from victor.native.accelerators.batch_processor import BatchTask

        tasks = [
            BatchTask(task_id="task-0", task_data=0),
            BatchTask(task_id="task-1", task_data=1),  # Will fail
            BatchTask(task_id="task-2", task_data=2),
        ]

        def failing_executor(task):
            if task.task_data == 1:
                raise ValueError("Task 1 fails")
            return task.task_data * 10

        processor = accelerator.create_processor(
            max_concurrent=2,
            timeout_ms=5000,
            retry_policy="none",  # No retries for this test
        )

        async def run_batch():
            return await accelerator.process_batch(tasks, failing_executor, processor)

        result = asyncio.run(run_batch())

        assert result.total_tasks == 3
        assert result.successful_count == 2
        assert result.failed_count == 1


class TestSerializationIntegration:
    """Test SerializationAccelerator integration."""

    def test_serialization_accelerator_singleton(self):
        """Test that accelerator singleton works correctly."""
        accelerator = get_serialization_accelerator()
        assert accelerator is not None
        assert hasattr(accelerator, "rust_available")

    def test_parse_json(self):
        """Test JSON parsing."""
        accelerator = get_serialization_accelerator()

        json_str = '{"key": "value", "number": 42, "nested": {"a": 1}}'
        data = accelerator.parse_json(json_str)

        assert data["key"] == "value"
        assert data["number"] == 42
        assert data["nested"]["a"] == 1

    def test_serialize_json(self):
        """Test JSON serialization."""
        accelerator = get_serialization_accelerator()

        data = {"key": "value", "number": 42, "nested": {"a": 1}}

        # Compact serialization
        json_str = accelerator.serialize_json(data, pretty=False)
        parsed = json.loads(json_str)
        assert parsed == data

        # Pretty serialization
        json_pretty = accelerator.serialize_json(data, pretty=True)
        assert "\n" in json_pretty  # Should be multi-line
        parsed = json.loads(json_pretty)
        assert parsed == data

    def test_parse_yaml(self):
        """Test YAML parsing."""
        accelerator = get_serialization_accelerator()

        yaml_str = """
key: value
number: 42
nested:
  a: 1
  b: [1, 2, 3]
"""
        data = accelerator.parse_yaml(yaml_str)

        assert data["key"] == "value"
        assert data["number"] == 42
        assert data["nested"]["a"] == 1
        assert data["nested"]["b"] == [1, 2, 3]

    def test_serialize_yaml(self):
        """Test YAML serialization."""
        accelerator = get_serialization_accelerator()

        data = {"key": "value", "number": 42, "nested": {"a": 1}}

        yaml_str = accelerator.serialize_yaml(data)
        assert "key: value" in yaml_str
        assert "number: 42" in yaml_str

    def test_load_config_file(self, tmp_path):
        """Test loading config file."""
        accelerator = get_serialization_accelerator()

        # Create test JSON config
        json_config = tmp_path / "config.json"
        json_config.write_text('{"setting": "value", "count": 5}')

        data = accelerator.load_config_file(str(json_config))
        assert data["setting"] == "value"
        assert data["count"] == 5

        # Create test YAML config
        yaml_config = tmp_path / "config.yaml"
        yaml_config.write_text("setting: value\ncount: 5\n")

        data = accelerator.load_config_file(str(yaml_config))
        assert data["setting"] == "value"
        assert data["count"] == 5

    def test_config_caching(self, tmp_path):
        """Test config file caching."""
        accelerator = get_serialization_accelerator(cache_size=10)

        # Create test config
        config_file = tmp_path / "cached_config.json"
        config_file.write_text('{"cached": "value"}')

        # Load twice - second should hit cache
        data1 = accelerator.load_config_file(str(config_file), use_cache=True)
        data2 = accelerator.load_config_file(str(config_file), use_cache=True)

        assert data1 == data2

        # Check cache stats
        stats = accelerator.get_cache_stats()
        assert stats["cache_entries"] >= 1

    def test_validate_json(self):
        """Test JSON validation."""
        accelerator = get_serialization_accelerator()

        # Valid JSON
        assert accelerator.validate_json('{"key": "value"}')
        assert accelerator.validate_json("[]")
        assert accelerator.validate_json("null")

        # Invalid JSON
        assert not accelerator.validate_json("{key: value}")  # Missing quotes
        assert not accelerator.validate_json("{")  # Incomplete
        assert not accelerator.validate_json("")  # Empty


class TestAcceleratorIntegration:
    """Test cross-accelerator integration scenarios."""

    def test_graph_and_batch_integration(self):
        """Test using graph and batch accelerators together."""
        graph_accel = get_graph_algorithms_accelerator()
        batch_accel = get_batch_processor_accelerator()

        # Create graph
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        graph = graph_accel.create_graph(edges, node_count=3, directed=True)

        # Use batch processor to compute metrics for multiple graphs
        from victor.native.accelerators.batch_processor import BatchTask

        graphs = [graph] * 5  # Process same graph 5 times

        tasks = [
            BatchTask(
                task_id=f"compute-{i}",
                task_data=graphs[i],
            )
            for i in range(5)
        ]

        def compute_pagerank(task):
            g = task.task_data
            return graph_accel.pagerank(g)

        processor = batch_accel.create_processor(
            max_concurrent=2,
            timeout_ms=5000,
            retry_policy="none",
        )

        async def run_batch():
            return await batch_accel.process_batch(tasks, compute_pagerank, processor)

        result = asyncio.run(run_batch())

        assert result.successful_count == 5
        # All results should be the same
        assert len(set(tuple(r.result) for r in result.results)) == 1

    def test_serialization_and_batch_integration(self, tmp_path):
        """Test using serialization and batch accelerators together."""
        serial_accel = get_serialization_accelerator()
        batch_accel = get_batch_processor_accelerator()

        # Create multiple config files
        configs = []
        for i in range(5):
            config_file = tmp_path / f"config-{i}.yaml"
            config_file.write_text(f"value: {i}\n")
            configs.append(str(config_file))

        # Load configs in batch
        from victor.native.accelerators.batch_processor import BatchTask

        tasks = [BatchTask(task_id=f"load-{i}", task_data=configs[i]) for i in range(5)]

        def load_config(task):
            return serial_accel.load_config_file(task.task_data)

        processor = batch_accel.create_processor(
            max_concurrent=3,
            timeout_ms=5000,
            retry_policy="none",
        )

        async def run_batch():
            return await batch_accel.process_batch(tasks, load_config, processor)

        result = asyncio.run(run_batch())

        assert result.successful_count == 5
        # Verify values
        values = [r.result["value"] for r in result.results]
        assert set(values) == {0, 1, 2, 3, 4}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
